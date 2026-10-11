"""Pick a colored ring from anywhere in the workspace and put it on the AMD sticker.

  python ringbot.py teach        # interactive: teleop with the leader + save rest/sticker poses and grid points
  python ringbot.py find         # print where each color is (camera only, arm doesn't move)
  python ringbot.py run red      # pick the red ring and place it on the sticker

How picking works without camera calibration or IK: during teaching you put one ring at ~12 spots
and steer the follower to grab it. Each grid point stores (ring pixel in the scene camera, joint
angles to hover above / grasp it). For a new ring we interpolate the joint angles of the nearest
grid points by pixel distance.

Frames come from the preview server (cam_preview.py) so the cameras are never opened twice.
"""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import rings
import tape

LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128141-if00"
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125028-if00"
FRAME_URL = "http://127.0.0.1:8090/scene.png"
POSES = Path(__file__).with_name("ringbot_poses.json")
FPS = 30
GRIP = "gripper.pos"
OPEN_WIDE = 50.0  # open the gripper 50% when grabbing
SLOWDOWN = 2.3      # every move takes this many times longer (slower, steadier)
USE_WRIST = False   # one overhead look, then go straight there (the wrist camera kept locking onto the sticker rim)
GRIP_TIGHT = 25.0  # firm but avoids stalling the gripper motor
WRIST_TILT = 0.0  # negative = gripper tips point more toward the table
PAN_OFFSET = -3.5  # sideways correction (normalized pan units) so the whole ring ends up between the jaws


def get_frame():
    data = urllib.request.urlopen(FRAME_URL, timeout=5).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def load_poses():
    return json.loads(POSES.read_text()) if POSES.exists() else {"grid": []}


def save_poses(p):
    POSES.write_text(json.dumps(p, indent=1))
    # (the search area is the hard-coded tape boundary; saving map points never changes it)


def clear_motor_faults(port=FOLLOWER_PORT):
    """Overload trips (status 32) make LeRobot's startup check reject a motor; toggling torque clears them."""
    import scservo_sdk as scs
    ph = scs.PortHandler(port)
    ph.openPort()
    ph.setBaudRate(1_000_000)
    pk = scs.PacketHandler(0)
    for m in range(1, 7):
        if pk.ping(ph, m)[2]:
            pk.write1ByteTxRx(ph, m, 40, 0)
            pk.write2ByteTxRx(ph, m, 42, pk.read2ByteTxRx(ph, m, 56)[0])
            pk.write1ByteTxRx(ph, m, 40, 1)
            time.sleep(0.2)
    ph.closePort()


MAP_REF = Path(__file__).with_name("map_ref.png")
STICKER_REF = Path(__file__).with_name("sticker_empty.png")


GRID_MARGIN = 0.05   # keep the grab area this far (in tape units) inside the tape's inner edge


def realign(log=print):
    """Re-trace the gray tape and rebuild every pixel position from tape coordinates (camera bumps don't matter).
    Falls back to image matching if the tape can't be traced."""
    if rings.load().get("tape_corners"):
        return            # the tape outline was clicked by hand: that is the reference (re-click after a camera move)
    img = get_frame()
    fr = tape.find_frame(img)
    p = load_poses()
    if fr is not None and p.get("tape_uv"):
        for g in p["grid"]:
            if "uv" in g:
                g["px"] = tape.to_px(fr, g["uv"])
        if p.get("base_grasp", {}).get("uv"):
            p["base_grasp"]["px"] = tape.to_px(fr, p["base_grasp"]["uv"])
        if p.get("base_fit", {}).get("centre_uv"):
            p["base_fit"]["centre"] = tape.to_px(fr, p["base_fit"]["centre_uv"])
        POSES.write_text(json.dumps(p, indent=1))
        cal = rings.load()
        a, b = GRID_MARGIN, 1 - GRID_MARGIN
        cal["roi"] = [[round(v) for v in tape.to_px(fr, uv)] for uv in [(a, a), (b, a), (b, b), (a, b)]]
        if cal.get("sticker_uv"):
            sx, sy = tape.to_px(fr, cal["sticker_uv"][:2])
            cal["exclude"] = [[sx, sy, cal["sticker_uv"][2] * np.linalg.norm(fr["ex"])]]
        cal["tape_frame"] = fr
        rings.CALIB.write_text(json.dumps(cal, indent=1))
        cv2.imwrite(str(MAP_REF), img)
        return
    realign_by_matching(log)


def to_tape_coords():
    """One-time: store every position in tape coordinates, measured on the image the current pixels belong to."""
    img = cv2.imread(str(MAP_REF)) if MAP_REF.exists() else get_frame()
    fr = tape.find_frame(img)
    if fr is None:
        raise SystemExit("can't trace the tape in the reference image")
    p = load_poses()
    for g in p["grid"]:
        g["uv"] = tape.to_uv(fr, g["px"])
    if p.get("base_fit"):
        p["base_fit"]["centre_uv"] = tape.to_uv(fr, p["base_fit"]["centre"])
    p["tape_uv"] = True
    POSES.write_text(json.dumps(p, indent=1))
    cal = rings.load()
    x, y, r = cal["exclude"][0]
    cal["sticker_uv"] = tape.to_uv(fr, [x, y]) + [r / np.linalg.norm(fr["ex"])]
    rings.CALIB.write_text(json.dumps(cal, indent=1))
    return fr, len(p["grid"])


def realign_by_matching(log=print):
    """If the scene camera got bumped, find how the table moved in the image (ORB + homography against the
    image the grid was mapped with) and move the grid, grab area and sticker position along with it."""
    if not MAP_REF.exists():
        return
    ref, img = cv2.imread(str(MAP_REF)), get_frame()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[120:, :] = 255
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), mask)
    kb, db = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
    if da is None or db is None:
        return
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 30:
        return
    src = np.float32([ka[x.queryIdx].pt for x in m])
    dst = np.float32([kb[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inl.sum() < 25:
        log("（摄像头对位：匹配点不够，沿用原来的网格）")
        return
    tf = lambda pts: cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    probe = np.float32([[200, 250], [450, 250], [200, 420], [450, 420]])
    shift = float(np.linalg.norm(tf(probe) - probe, axis=1).mean())
    if shift < 3:
        return
    log(f"摄像头移动了约 {shift:.0f}px，自动对齐网格")
    p = load_poses()
    for g, q in zip(p["grid"], tf([g["px"] for g in p["grid"]])):
        g["px"] = q.tolist()
    if p.get("base_grasp"):   # the taught base grab and the base pivot move with the image too
        p["base_grasp"]["px"] = tf([p["base_grasp"]["px"]])[0].tolist()
    if p.get("base_fit"):
        p["base_fit"]["centre"] = tf([p["base_fit"]["centre"]])[0].tolist()
    POSES.write_text(json.dumps(p, indent=1))
    cal = rings.load()
    cal["roi"] = tf(cal["roi"]).round().astype(int).tolist()
    for e in cal["exclude"]:
        e[0], e[1] = (float(v) for v in tf([e[:2]])[0])
    rings.CALIB.write_text(json.dumps(cal, indent=1))
    warped = cv2.warpPerspective(ref, H, (img.shape[1], img.shape[0]))
    x, y, rad = (int(v) for v in cal["exclude"][0])
    cv2.imwrite(str(STICKER_REF), warped[y - rad:y + rad, x - rad:x + rad])
    cv2.imwrite(str(MAP_REF), img)


def make_follower():
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="my_follower", max_relative_target=20.0))


# ---------------------------------------------------------------- teaching
def teach():
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="my_leader"))
    follower = make_follower()
    leader.connect()
    follower.connect()
    state = {"action": None, "run": True}

    def teleop():
        try:
            while state["run"]:
                t = time.perf_counter()
                a = leader.get_action()
                follower.send_action(a)
                state["action"] = dict(a)
                time.sleep(max(0, 1 / FPS - (time.perf_counter() - t)))
        except Exception as e:
            state["error"] = repr(e)
            print(f"\n!!! 遥操作出错，请按 Ctrl+C 退出并告诉 Claude: {e!r}", flush=True)

    th = threading.Thread(target=teleop, daemon=True)
    th.start()
    for _ in range(50):  # wait for the first teleop step so snapshots always have data
        if state["action"] is not None or state.get("error"):
            break
        time.sleep(0.1)
    if state.get("error") or state["action"] is None:
        raise SystemExit(f"遥操作启动失败: {state.get('error')}")
    print("遥操作已开启：从臂现在跟随主臂。", flush=True)
    poses = load_poses()
    ask = lambda msg: (input(f"\n>>> {msg}  [Enter] ") or "").strip().lower()
    snap = lambda: dict(state["action"])
    menu = ("\nr = 保存休息姿势 rest | s = 示教贴纸 sticker | g = 加一个网格点 grid | u = 删除最后一个网格点 | "
            "l = 列表 | q = 保存并退出")
    try:
        while True:
            print(menu)
            print(f"   已有: rest={'✔' if 'rest' in poses else '✘'}  sticker={'✔' if 'place' in poses else '✘'}  "
                  f"grid={len(poses['grid'])} 点")
            cmd = ask("选择")
            if cmd == "r":
                ask("把从臂移到休息姿势")
                poses["rest"] = snap()
                print("rest 已保存")
            elif cmd == "s":
                ask("夹着一个圆环，移到贴纸正上方 5–8 cm")
                poses["place_hover"] = snap()
                ask("慢慢放低，圆环刚好贴到贴纸")
                poses["place"] = snap()
                ask("张开夹爪松开圆环（手臂不动）")
                poses["release_grip"] = snap()[GRIP]
                print("sticker 已保存")
            elif cmd == "g":
                ask(f"把{GRID_COLOR}圆环拿出散放区，从臂抬高、别挡摄像头")
                empty = get_frame()
                ask(f"把{GRID_COLOR}圆环放到散放区的一个新位置（从臂仍别挡摄像头）")
                px = find_grid_ring(empty, get_frame())
                if px is None:
                    print(f"✘ 没找到{GRID_COLOR}圆环，重来（检查实时画面里有没有 blue 标记）")
                    continue
                print(f"✔ 圆环在画面 ({px[0]:.0f}, {px[1]:.0f})  —— 看实时画面确认圈对了，不对就按 u 删除")
                ask("夹爪张开，移到圆环正上方 5–8 cm")
                hover = snap()
                ask("放低，让张开的夹爪刚好围住圆环（还不要夹）")
                grasp = snap()
                ask("合上夹爪夹住圆环")
                closed = snap()[GRIP]
                poses["grid"].append({"px": px, "hover": hover, "grasp": grasp, "closed_grip": closed})
                save_poses(poses)
                print(f"网格点 #{len(poses['grid'])} 已保存。松开夹爪，抬起手臂。")
            elif cmd == "u" and poses["grid"]:
                poses["grid"].pop()
                print(f"已删除，剩 {len(poses['grid'])} 点")
            elif cmd == "l":
                for i, g in enumerate(poses["grid"], 1):
                    print(f"  #{i} px=({g['px'][0]:.0f},{g['px'][1]:.0f})")
            elif cmd == "q":
                break
            save_poses(poses)
    finally:
        save_poses(poses)
        state["run"] = False
        th.join(timeout=1)
        leader.disconnect()
        follower.disconnect()
        print("已保存到", POSES)


GRID_COLOR = "blue"


def find_grid_ring(before, after):
    """The GRID_COLOR ring that appeared between the two frames (whole image, no ROI)."""
    cands = rings.detect(after, roi=[])[GRID_COLOR]
    diff = cv2.absdiff(cv2.GaussianBlur(after, (5, 5), 0), cv2.GaussianBlur(before, (5, 5), 0)).max(axis=2)
    best = None
    for x, y, area, dist in cands:
        x0, y0 = int(x), int(y)
        changed = float(diff[max(0, y0 - 6):y0 + 7, max(0, x0 - 6):x0 + 7].mean())
        if changed > 25 and (best is None or dist < best[2]):
            best = (x, y, dist)
    return [best[0], best[1]] if best else None


def find_new_blob(before, after):
    """Pixel centre of the biggest new thing that appeared between two frames."""
    diff = cv2.absdiff(cv2.GaussianBlur(after, (5, 5), 0), cv2.GaussianBlur(before, (5, 5), 0)).max(axis=2)
    m = (diff > 40).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    cands = [(stats[j][4], cents[j]) for j in range(1, n) if 40 <= stats[j][4] <= 3000]
    if not cands:
        return None
    return [float(v) for v in max(cands, key=lambda c: c[0])[1]]


# ---------------------------------------------------------------- picking
def interpolate(grid, px, key, k=4):
    pts = np.array([g["px"] for g in grid], np.float32)
    d = np.linalg.norm(pts - np.array(px, np.float32), axis=1)
    d = np.where([g.get("good") for g in grid], d * 0.5, d)  # trust verified grasps more
    idx = np.argsort(d)[:k]
    if d[idx[0]] < 1:
        w = np.zeros(len(idx)); w[0] = 1
    else:
        w = 1 / d[idx] ** 2
    w = w / w.sum()
    if key == "closed_grip":
        return float(sum(w[i] * grid[j][key] for i, j in enumerate(idx)))
    joints = grid[0][key].keys()
    return {m: float(sum(w[i] * grid[j][key][m] for i, j in enumerate(idx))) for m in joints}


LIFT_JOINTS = ["shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"]

# Ring colors as seen by the WRIST camera (Lab), measured from grasp photos
WRIST_REFS = {"purple": [120, 166, 121], "blue": [147, 140, 117], "yellow": [208, 133, 191],
              "red": [154, 185, 172], "orange": [165, 176, 178]}
WRIST_TARGET = (323, 160)   # where the ring centre sits in the wrist image when it's right between the jaws
WW = np.array([0.3, 1.0, 1.0], np.float32)
# wrist-image pixels moved per unit of [pan, reach] (measured with `calib_servo` at the pre-grasp height)
SERVO_J = np.array([[18.7, 5.7], [-5.9, -11.2]])
PRE_LIFT = 0.15            # align ~1 cm above the ring: the descent then barely shifts the image
PRE_TARGET = WRIST_TARGET
GRASP_TARGET = (323, 160)  # ring centre in the wrist image at table height when it's deep between the jaws


def nudge(pose, d_pan, d_reach):
    return {**pose, "shoulder_pan.pos": pose["shoulder_pan.pos"] + d_pan,
            "shoulder_lift.pos": pose["shoulder_lift.pos"] + d_reach,
            "elbow_flex.pos": pose["elbow_flex.pos"] - d_reach}


_WT = Path(__file__).with_name("wrist_target.json")
if _WT.exists():   # taught from a correct manual pick-up (teach_target.py)
    _wt = json.loads(_WT.read_text())
    WRIST_TARGET = GRASP_TARGET = PRE_TARGET = tuple(_wt["target"])
    WRIST_REFS.update(_wt.get("refs", {}))


def wrist_refs():
    """Fill in colors not yet measured on the wrist camera from the scene refs + average camera offset."""
    cal = rings.load()["refs"]
    known = [c for c in WRIST_REFS if c in cal]
    off = np.mean([np.array(WRIST_REFS[c]) - np.array(cal[c]) for c in known], axis=0)
    return {c: WRIST_REFS.get(c, (np.array(cal[c]) + off).tolist()) for c in rings.COLORS}


def jaw_zone(img):
    """Pick-up zone in the wrist image = midpoint between the two orange tape pieces on the jaw tips."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # jaw tape: orange (H 4-10), more saturated than the wood table (S>=200) and a bit darker (V<186)
    m = ((hsv[:, :, 0] >= 4) & (hsv[:, :, 0] <= 10) & (hsv[:, :, 1] >= 200) & (hsv[:, :, 2] > 110) & (hsv[:, :, 2] < 186)).astype(np.uint8)
    m[260:, :] = 0                       # tape is on the jaw tips, in the upper part of the image
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(m)
    blobs = sorted([(st[j][4], ce[j]) for j in range(1, n) if st[j][4] > 300], key=lambda b: -b[0])[:2]
    if len(blobs) < 2:
        return None
    (a, pa), (b, pb) = blobs
    return float((pa[0] + pb[0]) / 2), float((pa[1] + pb[1]) / 2)


def get_wrist():
    data = urllib.request.urlopen("http://127.0.0.1:8090/wrist.png", timeout=5).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


PICK_LOWER = 0.48      # go this much lower than the trained pose (in down units; ~4 cm)
PICK_BACK = 4.0        # going lower also reaches forward: pull back this much (reach units) so it goes straight down
DEMO_WEIGHT = 0.5      # grab pose = this much of the training demos + the rest from the earlier grid result
# on a missed grab, the next attempt varies the grab: (extra lower, pan, reach)
RETRY_VARIATIONS = [(0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.04, -2.5, 0.0), (0.04, 2.5, 0.0), (0.08, 0.0, 2.0), (0.04, 0.0, -2.0)]
EXACT_TRAINED = True   # go straight to the trained pose, no touch-down search
MAX_RETRIES = 1        # first try + this many retries, then finish
GRAB_FEEL = 3.0        # jaws stopping this far short of the target = holding a ring
EXTRA_SQUEEZE = 10.0   # close this much further than in the training grabs
CLOSE_SECS = 2.6       # how long the close takes
JAW_CHECK = False   # wrist "ring behind the tips" check: off, it misjudged too often in this light
JAW_BOX = (140, 20, 430, 230)   # wrist image: between the two jaws, above (behind) the tips: x0, y0, x1, y1


def ring_in_jaws(img, color=None, box=JAW_BOX):
    """Is there a ring between the jaws, behind the tips, in the wrist image? The wrist camera has a strong warm
    color cast, so this doesn't check the exact color (the overhead camera already picked the ring): it looks for a
    ring-sized blob that is not the orange table (far from its color), not the gray tape and not the white jaws
    (both low saturation)."""
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    table = np.median(lab[200:300, 150:450].reshape(-1, 3), axis=0)          # table in front of the tips
    far = np.linalg.norm((lab - table) * np.array([0.5, 1, 1], np.float32), axis=2) > 30
    m = (far & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 230)).astype(np.uint8)   # not white jaws / glare
    keep = np.zeros_like(m)
    keep[y0:y1, x0:x1] = 1
    m = cv2.morphologyEx(m & keep, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(m)
    return any(st[j][4] >= 600 and 40 <= st[j][2] <= 130 and 40 <= st[j][3] <= 130     # ring-sized
               and 0.6 < st[j][2] / st[j][3] < 1.6 and ce[j][1] > 50                    # ring-shaped, below the jaw base
               for j in range(1, n))


def find_in_wrist(img, color, near=WRIST_TARGET):
    """Centre of the ring blob (of `color`, or any ring color if None) closest to `near` in the wrist image.
    Only compact, roughly round blobs count, so the sticker's red rim or long edges are ignored."""
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (5, 5), 0)
    table = np.median(lab[220:420, 120:520].reshape(-1, 3), axis=0)
    d_tab = np.linalg.norm((lab - table) * WW, axis=2)
    refs = wrist_refs()
    cands = []
    for c in ([color] if color else rings.COLORS):
        ref = np.array(refs[c], np.float32)
        d_ref = np.linalg.norm((lab - ref) * WW, axis=2)
        m = ((d_ref < 22) & (d_ref < 0.7 * d_tab)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, ce = cv2.connectedComponentsWithStats(m)
        for j in range(1, n):
            x, y, w, h, area = st[j]
            if not (400 < area < 12000 and 30 < max(w, h) < 150 and 0.55 < w / max(h, 1) < 1.8 and area > 0.3 * w * h):
                continue
            cands.append(ce[j])
    if not cands:
        return None
    best = min(cands, key=lambda c: np.hypot(c[0] - near[0], c[1] - near[1]))
    return float(best[0]), float(best[1])


def lift_above(pose, p, amount=0.8):
    """A pose above `pose`: shift the lift joints by the sticker's (hover - place) difference,
    i.e. the same 'go up' motion you taught over the sticker."""
    up = {k: p["place_hover"][k] - p["place"][k] for k in LIFT_JOINTS}
    return {k: v + amount * up[k] if k in up else v for k, v in pose.items()}


LOCK_PATH = "/tmp/so101_follower.lock"


def save_sticker_ref():
    img = get_frame()
    x, y, rad = (int(v) for v in rings.load()["exclude"][0])
    cv2.imwrite(str(STICKER_REF), img[y - rad:y + rad, x - rad:x + rad])


def fit_base(grid):
    """Where the robot's base pivot is in the image, and how base rotation maps to angle around it:
    pan ~= k * angle(px - centre) + b, fitted over all mapped points (grid search for the centre)."""
    P = np.array([g["px"] for g in grid], float)
    pan = np.array([g["grasp"]["shoulder_pan.pos"] for g in grid], float)
    best = None
    for cx in range(150, 520, 6):
        for cy in range(380, 760, 6):
            ang = np.unwrap(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
            A = np.c_[ang, np.ones(len(ang))]
            coef, res, *_ = np.linalg.lstsq(A, pan, rcond=None)
            err = float(np.median(np.abs(A @ coef - pan)))
            if best is None or err < best[0]:
                best = (err, (cx, cy), coef)
    return {"centre": best[1], "k": float(best[2][0]), "err": best[0]}


def demo_model(p):
    """True when there are enough training demos to use them (no side effects: never writes the poses file)."""
    return len(p.get("demos", [])) >= 5


TRAIN_CALIB = Path(__file__).with_name("rings_calib.LOCKED.json")   # the camera view all the training was done in


def _quad(q):
    """4 corners in a fixed order: top-left, top-right, bottom-right, bottom-left."""
    q = np.array(q, float)
    s, d = q.sum(1), q[:, 0] - q[:, 1]
    return np.float32([q[s.argmin()], q[d.argmax()], q[s.argmax()], q[d.argmin()]])


def to_train(px):
    """Camera got moved? Map a pixel of the current view (clicked tape corners) into the training view
    (locked tape corners), so the trained grabs/grid are used unchanged."""
    cur = rings.load().get("tape_corners")
    ref = json.loads(TRAIN_CALIB.read_text()).get("tape_corners") if TRAIN_CALIB.exists() else None
    if not cur or not ref:
        return tuple(px)
    H = cv2.getPerspectiveTransform(_quad(cur), _quad(ref))
    return tuple(float(v) for v in cv2.perspectiveTransform(np.float32([[px[:2]]]), H)[0, 0])


def demo_pose(p, px):
    """Pose for a ring at pixel px from the training demos (best per joint by leave-one-out test):
    base turn + elbow = local linear fit on the 6 nearest demos; shoulder + wrist = blend of the 3 nearest."""
    demos = p["demos"]
    px = np.asarray(px, float)
    ver = [x for x in demos if x.get("verified")]
    if len(ver) >= 6 and min(np.linalg.norm(np.array(x["px"]) - px) for x in ver) < 60:
        demos = ver            # confirmed-successful grabs nearby: train only on those
    P = np.array([d["px"] for d in demos], float)
    d = np.linalg.norm(P - px, axis=1)
    if d.min() < 15:     # a training grab at (almost) this exact spot: copy it exactly
        return dict(demos[int(d.argmin())]["grasp"])
    near3 = np.argsort(d)[:3]
    w3 = 1 / (d[near3] + 5) ** 2
    w3 /= w3.sum()
    pose = {k: float(sum(w3[i] * demos[j]["grasp"][k] for i, j in enumerate(near3))) for k in demos[0]["grasp"]}
    near6 = np.argsort(d)[:6]
    A = np.c_[P[near6], np.ones(len(near6))]
    w6 = 1 / (d[near6] + 10)
    for k in ("shoulder_pan.pos", "elbow_flex.pos"):
        y = np.array([demos[j]["grasp"][k] for j in near6])
        coef = np.linalg.lstsq(A * w6[:, None], y * w6, rcond=None)[0]
        val = float(np.r_[px, 1] @ coef)
        lo, hi = y.min() - 8, y.max() + 8       # don't extrapolate wildly
        pose[k] = float(np.clip(val, lo, hi))
    return pose


def base_formula_pose(p, px):
    """Same distance as the taught grab: copy its pose, only turn the base toward the new ring."""
    bg, fb = p["base_grasp"], p["base_fit"]
    cx, cy = fb["centre"]
    a0 = np.arctan2(bg["px"][1] - cy, bg["px"][0] - cx)
    a1 = np.arctan2(px[1] - cy, px[0] - cx)
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    return {**bg["grasp"], "shoulder_pan.pos": bg["grasp"]["shoulder_pan.pos"] + fb["k"] * d}


class Arm:
    def __init__(self):
        import fcntl
        self._lock = open(LOCK_PATH, "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("机械臂正忙（另一个程序正在控制它）")
        self.poses = load_poses()
        clear_motor_faults()
        self.f = make_follower()
        self.f.connect()
        obs = self.f.get_observation()
        self.cur = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}

    def move(self, target, secs):
        start = dict(self.cur)
        n = max(1, int(secs * SLOWDOWN * FPS))
        for i in range(1, n + 1):
            t = time.perf_counter()
            a = i / n
            a = a * a * (3 - 2 * a)  # ease in/out
            self.cur = {k: start[k] + (target[k] - start[k]) * a for k in start}
            self.f.send_action(self.cur)
            time.sleep(max(0, 1 / FPS - (time.perf_counter() - t)))

    def lift_error(self):
        """How far the lift joints lag behind their command (grows when the gripper is pushing on the table)."""
        obs = self.f.get_observation()
        return sum(abs(self.cur[k] - float(obs[k])) for k in LIFT_JOINTS)

    def touch_down(self, top, p, max_down=0.9, step=0.02, threshold=5.0, back=0.01, log=print):
        """Lower from `top` along the taught 'down' direction until the arm meets resistance (the table).
        Returns the pose `back` above the contact point."""
        up = {k: p["place_hover"][k] - p["place"][k] for k in LIFT_JOINTS}
        at = lambda a: {k: v - a * up[k] if k in up else v for k, v in top.items()}
        self.move(top, 1.0)          # get to the start height first, otherwise the first step looks like a collision
        time.sleep(0.4)
        base = self.lift_error()
        a = 0.0
        while a < max_down:
            a += step
            self.cur = at(a)
            self.f.send_action(self.cur)
            time.sleep(0.07 * SLOWDOWN)
            e = self.lift_error() - base
            if e > threshold:
                log(f"  触底: 下降 {a:.2f}（阻力 {e:.1f}）")
                break
        else:
            log(f"  没碰到桌面，降到最低 {a:.2f}")
        pose = at(a - back)   # stop where it touched (tiny lift), no extra push forward
        self.move(pose, 0.4)
        return pose

    def grip(self, value, secs=0.6):
        self.move({**self.cur, GRIP: value}, secs)

    def locate(self, color, tries=5):
        for _ in range(tries):     # a single frame can miss a ring that's near the color threshold
            det = rings.detect(get_frame())[color]
            if det:
                return det[0][:2]
            time.sleep(0.3)
        return None

    def grab(self, grasp, color, tag, log=print, open_w=None, close_to=None):
        """Align over the ring with the wrist camera, touch down, squeeze slowly, lift. Returns (grasp, hover, closed)."""
        p = self.poses
        self.track, self.track_strict = None, False
        grasp = {**grasp, "wrist_flex.pos": grasp["wrist_flex.pos"] + WRIST_TILT}  # tips pointing more down
        hover = lift_above(grasp, p)
        closed = (close_to - EXTRA_SQUEEZE) if close_to is not None else GRIP_TIGHT  # squeeze the whole ring firmly
        open_grip = open_w or OPEN_WIDE  # wide enough for the whole ring, narrower when other rings are close
        grasp = {**grasp, GRIP: open_grip}
        self.move({**hover, GRIP: open_grip}, 2.0)
        Jinv = np.linalg.inv(SERVO_J)
        # --- align with the wrist camera while descending in stages (hover -> ... -> ~1 cm above the ring)
        last = None
        # --- if the wrist camera can't see the ring from above, scan around the estimate until it does
        if USE_WRIST and self.wrist_look(color) is None:
            log("  手腕摄像头没看到圆环，在附近搜索")
            for dp, dr in [(-6, 0), (6, 0), (0, -4), (0, 4), (-12, 0), (12, 0), (-6, -4), (6, -4), (-6, 4), (6, 4),
                           (-18, 0), (18, 0), (0, -8), (0, 8)]:
                cand = nudge(grasp, dp, dr)
                self.move(lift_above(cand, p, 1.0), 0.6)
                if self.wrist_look(color) is not None:
                    grasp = cand
                    log(f"  搜索到了（底座 {dp:+d}，伸缩 {dr:+d}）")
                    break
            else:
                log("  附近也没找到")
        if getattr(self, "track", None) is not None:
            self.track_strict = True       # locked onto the ring: from here on ignore big jumps
        for level in ([1.0, 0.6, 0.35, PRE_LIFT] if USE_WRIST else []):
            pose = lift_above(grasp, p, level)
            self.move(pose, 0.8 if level < 1.0 else 0.1)
            scale = 0.6 + 0.4 * (1.0 - level)          # rings look bigger / move more as we get closer
            for it in range(5):
                ring = self.wrist_look(color)
                if ring is None:
                    log(f"  高度 {level}: 没看到圆环")
                    break
                last = ring
                err = np.array(PRE_TARGET) - np.array(ring)
                log(f"  高度 {level} 对准 {it}: 圆环 ({ring[0]:.0f},{ring[1]:.0f}) 偏差 ({err[0]:+.0f},{err[1]:+.0f})")
                if abs(err[0]) < (12 if level == PRE_LIFT else 30) :
                    break
                if False:   # (in/out correction at height disabled: it overshot)
                    d_pan, d_reach = np.clip(0.7 * np.linalg.inv(SERVO_J * scale) @ err, -5, 5)
                else:
                    d_pan = float(np.clip(0.8 * err[0] / (SERVO_J[0, 0] * scale), -6, 6)); d_reach = 0.0
                grasp = nudge(grasp, d_pan, d_reach)
                self.move(lift_above(grasp, p, level), 0.6)
        if USE_WRIST and last is None:
            log("手腕摄像头没看到圆环，按估算位置抓")
        # --- find the real table height instead of trusting the estimate
        # taught base pose is already at table height: only search a little below it
        if EXACT_TRAINED:
            self.move(lift_above(grasp, p, PRE_LIFT), 0.8)
            grasp = lift_above(grasp, p, -PICK_LOWER)   # a little lower than trained, so the tips reach the ring
            grasp = nudge(grasp, 0.0, -PICK_BACK)        # ...but not further forward
            self.move(grasp, 1.2)
        else:
            grasp = self.touch_down(lift_above(grasp, p, PRE_LIFT), p, max_down=0.9, log=log)
        # --- at the bottom: make sure the ring sits deep between the jaws; if not, lift a little, shift, go down again
        for it in range(6 if USE_WRIST else 0):
            ring = self.wrist_look(color)
            if ring is None:
                break
            zone = jaw_zone(get_wrist())
            target = zone if zone is not None else GRASP_TARGET
            err = np.array(target) - np.array(ring)
            log(f"  底部检查 {it}: 夹爪区域 ({target[0]:.0f},{target[1]:.0f}) 圆环 ({ring[0]:.0f},{ring[1]:.0f}) 偏差 ({err[0]:+.0f},{err[1]:+.0f})")
            if abs(err[0]) < 18 and abs(err[1]) < 25:
                break
            # decoupled: left/right with the base only, in/out with reach only (small steps)
            d_pan = float(np.clip(0.6 * err[0] / (SERVO_J[0, 0] * 1.3), -3, 3))
            # at table height the in/out direction is reversed vs. the pre-grasp height (measured: corrections
            # with the old sign pushed the ring deeper every time), so flip it here
            d_reach = float(np.clip(-0.6 * err[1] / (SERVO_J[1, 1] * 1.3), -3, 3))
            up_pose = nudge(lift_above(grasp, p, 0.12), d_pan, d_reach)
            self.move(up_pose, 0.6)
            grasp = self.touch_down(up_pose, p, max_down=0.4, log=log)
        time.sleep(0.6)
        self.snapshot(f"{tag}_grasp")
        if JAW_CHECK and color and not ring_in_jaws(get_wrist(), color):
            log("  手腕摄像头：圆环不在夹爪尖后面，不夹，重新定位")
            self.move(lift_above(grasp, p), 1.0)
            return None
        self.grip(closed, CLOSE_SECS / SLOWDOWN)   # faster, firmer close
        # feel the grab: on a ring the jaws stop early (blocked); on nothing they reach the target
        time.sleep(0.5)
        actual = float(self.f.get_observation()[GRIP])
        if actual - closed < GRAB_FEEL:
            log(f"  夹空：夹爪合到 {actual:.0f}（目标 {closed:.0f}），没夹住圆环，松开重来")
            self.grip(open_grip, 0.6)
            self.move(lift_above(grasp, p, 0.3), 0.8)
            return None
        log(f"  夹住了：夹爪停在 {actual:.0f}（目标 {closed:.0f}）")
        time.sleep(0.3)
        self.snapshot(f"{tag}_closed")
        self.move({**hover, GRIP: closed}, 1.5)
        self.snapshot(f"{tag}_lifted")  # lift straight up with the ring
        return grasp, hover, closed

    def pick_place(self, color, log=print, attempt=1):
        self.poses = p = load_poses()
        if "rest" in p:   # get the arm out of the overhead camera's view before looking
            self.move(p["rest"], 2.0)
            time.sleep(0.5)
        realign(log)
        self.poses = p = load_poses()
        if "rest" not in p or "place" not in p or len(p["grid"]) < 3:
            return log("还没示教完（需要 rest、sticker 和至少 3 个网格点）")
        px = self.locate(color)
        if px is None:
            return log(f"没找到 {color} 圆环")
        log(f"{color} 圆环在画面 ({px[0]:.0f}, {px[1]:.0f})")
        tpx = to_train(px)            # same spot in the camera view the training was done in
        demo_open = demo_closed = None
        if demo_model(p):
            grasp = demo_pose(p, tpx)
            gridp = interpolate(p["grid"], tpx, "grasp")   # the earlier (pre-demo) grid result
            grasp = {k: DEMO_WEIGHT * v + (1 - DEMO_WEIGHT) * gridp[k] if k in gridp and k != GRIP else v
                     for k, v in grasp.items()}
            demo_open = grasp[GRIP]                     # open exactly as wide as in the training grabs
            P = np.array([d["px"] for d in p["demos"]], float)
            near = np.argsort(np.linalg.norm(P - np.asarray(tpx, float), axis=1))[:3]
            demo_closed = float(np.mean([p["demos"][j]["closed_grip"] for j in near]))   # ...and close as far
            log(f"  用训练演示：底座转到 {grasp['shoulder_pan.pos']:.1f}，夹爪张开 {demo_open:.0f} → 合到 {demo_closed:.0f}")
        elif p.get("base_grasp") and p.get("base_fit"):
            grasp = base_formula_pose(p, tpx)
            log(f"  用基准抓取公式：底座转到 {grasp['shoulder_pan.pos']:.1f}")
        else:
            grasp = interpolate(p["grid"], tpx, "grasp")
            if not any(g.get("good") and np.hypot(g["px"][0] - tpx[0], g["px"][1] - tpx[1]) < 40 for g in p["grid"]):
                grasp = {**grasp, "shoulder_pan.pos": grasp["shoulder_pan.pos"] + PAN_OFFSET}
        others = [b[:2] for c, bl in rings.detect(get_frame()).items() if c != color for b in bl[:1]]
        near = min([np.hypot(o[0] - px[0], o[1] - px[1]) for o in others] or [999])
        open_w = OPEN_WIDE
        if near <= 55:
            log(f"  附近 {near:.0f}px 有别的圆环，夹爪少张开一点")
        # self-correction: a correction learned near here + this attempt's variation
        corr = [c for c in p.get("corrections", []) if np.hypot(c["px"][0] - tpx[0], c["px"][1] - tpx[1]) < 45]
        base = min(corr, key=lambda c: np.hypot(c["px"][0] - tpx[0], c["px"][1] - tpx[1]))["off"] if corr else (0.0, 0.0, 0.0)
        var = RETRY_VARIATIONS[(attempt - 1) % len(RETRY_VARIATIONS)]
        off = (base[0] + var[0], base[1] + var[1], base[2] + var[2])
        if any(off):
            grasp = nudge(lift_above(grasp, p, -off[0]), off[1], off[2])
            log(f"  修正：更低 {off[0]:.2f}，底座 {off[1]:+.1f}，伸缩 {off[2]:+.1f}")
        if demo_open is not None:
            open_w = demo_open
        got = self.grab(grasp, color, color, log, open_w=open_w, close_to=demo_closed)
        if got is None:
            self.move(p["rest"], 1.5)
            return False
        grasp, hover, closed = got
        self.move({**p["place_hover"], GRIP: closed}, 2.0)
        # did we really grab it? the arm is now over the sticker, away from the pick spot: look at that spot
        time.sleep(0.4)
        still_there = rings.detect(get_frame())[color]
        if any(np.hypot(b[0] - px[0], b[1] - px[1]) < 35 for b in still_there):
            log("  抓空了：圆环还在原处，回去重新抓")
            self.move(p["rest"], 1.5)
            return False
        # lower until the ring touches the sticker, then let go (instead of releasing at a fixed height)
        self.move({**lift_above(p["place"], p, 0.3), GRIP: closed}, 1.0)
        self.move({**p["place"], GRIP: closed}, 1.0)        # straight to the taught sticker pose, no further descent
        self.grip(p["release_grip"], 0.6)
        self.move({**p["place_hover"], GRIP: p["release_grip"]}, 0.8)
        self.move(p["rest"], 2.0)
        time.sleep(0.5)
        if any(off):   # remember the correction that worked here
            others = [c for c in p.get("corrections", []) if np.hypot(c["px"][0] - tpx[0], c["px"][1] - tpx[1]) >= 25]
            p["corrections"] = others + [{"px": list(tpx), "off": list(off)}]
            save_poses(p)
            log("  已记住这个位置的修正")
        if True:   # we only get here when the ring was confirmed in the gripper and dropped on the sticker
            log("完成 ✔ 成功")   # (the grid is locked: successful grabs are not added to it)
            return True
        else:
            log("完成 ✘ 圆环还在桌上，抓取失败")
            return False

    def sticker_diff(self):
        """How different the sticker looks from the saved empty-sticker photo (mean abs diff inside the disc)."""
        img = get_frame()
        x, y, rad = rings.load()["exclude"][0]
        x, y, rad = int(x), int(y), int(rad)
        crop = img[y - rad:y + rad, x - rad:x + rad].astype(np.float32)
        ref = cv2.imread(str(STICKER_REF)).astype(np.float32)
        m = np.zeros(crop.shape[:2], np.uint8)
        cv2.circle(m, (rad, rad), rad - 4, 1, -1)
        return float(np.abs(crop - ref).max(axis=2)[m > 0].mean())

    def sticker_busy(self, threshold=18.0):
        if not STICKER_REF.exists():
            return False
        return self.sticker_diff() > threshold

    def free_spot(self, rng, margin=45):
        """A random pixel inside the taught area, away from every ring on the table."""
        pts = np.array([g["px"] for g in self.poses["grid"]], np.float32)
        hull = cv2.convexHull(pts)
        others = [b[:2] for bl in rings.detect(get_frame()).values() for b in bl[:1]]
        x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
        for m in (margin, 35, 28):
            for _ in range(400):
                q = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                if cv2.pointPolygonTest(hull, q, True) > 0 and all(np.hypot(q[0] - o[0], q[1] - o[1]) > m for o in others):
                    return q
        return None

    def sticker_color(self):
        """Which ring color is lying on the sticker (looks inside the red rim only), or None."""
        img = get_frame()
        cal = rings.load()
        x, y, rad = cal["exclude"][0]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(m, (int(x), int(y)), int(rad) - 12, 1, -1)
        sel = (m > 0) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)
        if sel.sum() < 80:
            return None
        col = np.median(rings.to_lab(img)[sel], axis=0)
        return min(rings.COLORS, key=lambda c: np.linalg.norm((np.array(cal["refs"][c]) - col) * rings.W))

    def return_ring(self, spot, log=print):
        """Pick the ring up from the sticker (same careful grab as on the table) and set it down at `spot`."""
        p = self.poses
        est = {**p["place"], GRIP: OPEN_WIDE}
        grasp, hover, closed = self.grab(est, None, "sticker", log)   # any ring color on the sticker
        g = {**(base_formula_pose(p, spot) if p.get("base_grasp") else interpolate(p["grid"], spot, "grasp")), GRIP: closed}
        self.move(lift_above(g, p), 2.0)
        self.move(lift_above(g, p, 0.15), 1.2)
        self.grip(OPEN_WIDE, 1.0)
        self.move({**lift_above(g, p), GRIP: OPEN_WIDE}, 1.0)
        self.move(p["rest"], 2.0)
        if self.sticker_color() is not None:
            log("✘ 贴纸上的圆环没拿走")
            return False
        log(f"已把圆环放回桌上 ({spot[0]:.0f},{spot[1]:.0f})")
        return True

    def remap(self, color="blue", pan_offsets=(0.0, -8.0, 8.0), log=print):
        """Robot re-maps its grid for a new camera view: it sets a held ring down at each taught pose (and pan
        variations), steps back, records where the camera sees it, and re-grabs it from the same pose."""
        p = self.poses
        old = [g for g in p["grid"]]
        self.move({**p["place_hover"], GRIP: OPEN_WIDE}, 2.0)
        log(f"把{color}圆环放到夹爪中间（20 秒后夹住）")
        time.sleep(20)
        self.grip(GRIP_TIGHT, 2.0)
        new = []
        for i, g in enumerate(old):
            for off in pan_offsets:
                pose = {**g["grasp"], "shoulder_pan.pos": g["grasp"]["shoulder_pan.pos"] + off, GRIP: GRIP_TIGHT}
                self.move(lift_above(pose, p), 1.5)
                self.move(lift_above(pose, p, 0.05), 1.0)
                self.grip(OPEN_WIDE, 0.8)
                self.move({**lift_above(pose, p), GRIP: OPEN_WIDE}, 0.8)
                self.move({**p["rest"], GRIP: OPEN_WIDE}, 1.5)
                time.sleep(0.5)
                det = rings.detect(get_frame(), roi=[])[color]   # whole table, not the (growing) grid area
                seen = det[0][:2] if det else None
                # re-grab from exactly where it was put down
                self.move({**lift_above(pose, p), GRIP: OPEN_WIDE}, 1.5)
                self.move({**lift_above(pose, p, 0.05), GRIP: OPEN_WIDE}, 1.0)
                self.grip(GRIP_TIGHT, 1.5)
                self.move({**lift_above(pose, p), GRIP: GRIP_TIGHT}, 0.8)
                if seen is None:
                    log(f"  点 {i + 1} 偏移 {off:+.0f}: 没看到圆环，跳过")
                    continue
                q = {**lift_above(pose, p, 0.05), GRIP: OPEN_WIDE}
                new.append({"px": list(seen), "grasp": q, "hover": lift_above(q, p), "closed_grip": GRIP_TIGHT, "good": True})
                log(f"  点 {i + 1} 偏移 {off:+.0f}: 圆环在 ({seen[0]:.0f},{seen[1]:.0f})")
                POSES.with_name("ringbot_poses.remap_partial.json").write_text(json.dumps({**p, "grid": new}, indent=1))
        p["grid"] = new
        save_poses(p)
        self.move({**p["place_hover"], GRIP: GRIP_TIGHT}, 2.0)
        self.move({**p["place"], GRIP: GRIP_TIGHT}, 1.0)
        self.grip(OPEN_WIDE, 0.8)
        self.move(p["rest"], 2.0)
        log(f"重新映射完成: {len(new)} 个点（圆环留在贴纸上）")

    def wrist_look(self, color):
        """Wrist-camera ring position, tracked: prefer the blob closest to where the ring was last seen,
        and ignore sudden jumps (the sticker's red rim, other rings)."""
        time.sleep(0.5)
        last = getattr(self, "track", None)
        ring = find_in_wrist(get_wrist(), color, near=last or WRIST_TARGET)
        if ring is not None and last is not None and np.hypot(ring[0] - last[0], ring[1] - last[1]) > 120:
            return None if getattr(self, "track_strict", False) else ring
        if ring is not None:
            self.track = ring
        return ring

    def calib_servo(self, color, step=2.0):
        """Measure how the ring moves in the wrist image for small pan / reach moves."""
        p = self.poses
        px = self.locate(color)
        g = interpolate(p["grid"], px, "grasp")
        g = {**g, "shoulder_pan.pos": g["shoulder_pan.pos"] + PAN_OFFSET, GRIP: OPEN_WIDE}
        pre = lift_above(g, p, 0.35)
        self.move(lift_above(g, p), 1.5)
        self.move(pre, 1.0)
        base = self.wrist_look(color)
        print("ring at", base)
        res = {}
        for name, delta in [("pan", {"shoulder_pan.pos": step}),
                            ("reach", {"shoulder_lift.pos": step, "elbow_flex.pos": -step})]:
            self.move({k: v + delta.get(k, 0) for k, v in pre.items()}, 0.8)
            q = self.wrist_look(color)
            self.move(pre, 0.8)
            res[name] = None if (q is None or base is None) else ((q[0] - base[0]) / step, (q[1] - base[1]) / step)
            print(name, "-> px per unit", res[name])
        self.move(lift_above(g, p), 0.8)
        self.move(p["rest"], 2.0)
        return res

    def probe(self, color, offsets):
        """Go to the grasp pose with different pan offsets (gripper wide open), photo each, no grasp."""
        p = self.poses
        px = self.locate(color)
        grasp0 = interpolate(p["grid"], px, "grasp")
        for off in offsets:
            g = {**grasp0, "shoulder_pan.pos": grasp0["shoulder_pan.pos"] + off, GRIP: OPEN_WIDE}
            self.move(lift_above(g, p), 1.5)
            self.move(g, 1.0)
            time.sleep(0.6)
            self.snapshot(f"probe_{off:+.1f}")
            self.move(lift_above(g, p), 0.8)
        self.move(p["rest"], 2.0)

    def snapshot(self, tag):
        for cam in ("scene", "wrist"):
            try:
                data = urllib.request.urlopen(f"http://127.0.0.1:8090/{cam}.png", timeout=5).read()
                Path(f"/home/amd-gpu-4/debug_{tag}_{cam}.png").write_bytes(data)
            except Exception:
                pass

    def close(self):
        self.f.disconnect()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "teach":
        teach()
    elif cmd == "recalib":
        # all 7 rings visible in the workspace: refresh each reference color from what the camera sees now
        img = get_frame()
        cal = rings.load()
        found = {c: b[0] for c, b in rings.detect(img, cal).items() if b}
        missing = [c for c in rings.COLORS if c not in found]
        if missing:
            sys.exit(f"没找到: {missing}，把 7 个圆环都放进散放区再试")
        cal = rings.calibrate(img, {c: (int(b[0]), int(b[1])) for c, b in found.items()},
                              table_xy=tuple(cal.get("table_xy", (200, 390))), roi=cal.get("roi"))
        print("颜色已重新校准")
    elif cmd == "remap":
        arm = Arm()
        try:
            arm.remap(sys.argv[2] if len(sys.argv) > 2 else "blue")
        finally:
            arm.close()
    elif cmd == "roundtrip":
        # python ringbot.py roundtrip red : pick it, put it on the sticker, then put it back where it was
        color = sys.argv[2]
        arm = Arm()
        try:
            realign()
            start = rings.detect(get_frame())[color]
            if not start:
                raise SystemExit(f"没找到 {color}")
            home = start[0][:2]
            print(f"{color} 原来在 ({home[0]:.0f},{home[1]:.0f})", flush=True)
            if arm.pick_place(color):
                arm.return_ring(home)
                after = rings.detect(get_frame())[color]
                if after:
                    d = np.hypot(after[0][0] - home[0], after[0][1] - home[1])
                    print(f"放回后在 ({after[0][0]:.0f},{after[0][1]:.0f})，离原位 {d:.0f}px", flush=True)
        finally:
            arm.close()
    elif cmd == "set_base":
        # take the most recent manual grab as the base formula and fit the base pivot
        p = load_poses()
        demo = [g for g in p["grid"] if "t" in g]
        bg = max(demo, key=lambda g: g["t"])
        p["base_grasp"] = {"px": bg["px"], "grasp": bg["grasp"]}
        p["base_fit"] = fit_base([g for g in p["grid"] if "t" not in g] or p["grid"])
        save_poses(p)
        print("base grab at", [round(v) for v in bg["px"]], "| pivot", p["base_fit"]["centre"],
              "| pan per radian", round(p["base_fit"]["k"], 1), "| fit error", round(p["base_fit"]["err"], 2))
    elif cmd == "unload":
        # python ringbot.py unload 400 247 : take the ring off the sticker and set it down at that pixel
        arm = Arm()
        try:
            arm.return_ring((float(sys.argv[2]), float(sys.argv[3])))
        finally:
            arm.close()
    elif cmd == "random":
        # grab one of the rings currently on the table, chosen at random
        import random
        arm = Arm()
        try:
            realign()
            arm.poses = load_poses()
            seen = [c for c in rings.COLORS if arm.locate(c, tries=2) is not None]
            if not seen:
                raise SystemExit("桌上没有圆环")
            color = random.choice(seen)
            print(f"随机选中: {color}（桌上有 {seen}）", flush=True)
            for attempt in range(1, 1 + 1 + MAX_RETRIES):
                print(f"第 {attempt} 次尝试", flush=True)
                if arm.pick_place(color, attempt=attempt):
                    break
                time.sleep(1.0)
                if arm.locate(color, tries=10) is None:
                    print(f"桌上找不到 {color} 了", flush=True)
                    break
        finally:
            arm.close()
    elif cmd == "loop":
        # python ringbot.py loop red orange yellow ...  -> waits for the sticker to be cleared by hand, then picks the next
        arm = Arm()
        res = []
        try:
            for color in sys.argv[2:]:
                while arm.sticker_busy():
                    print("请把贴纸上的圆环拿走…", flush=True)
                    time.sleep(2)
                time.sleep(2)
                ok = arm.pick_place(color, log=lambda m: print(f"[{color}] {m}", flush=True))
                res.append(f"{color}:{'✔' if ok else '✘'}")
                print("RESULTS", " ".join(res), flush=True)
        finally:
            arm.close()
    elif cmd == "to_tape":
        fr, n = to_tape_coords()
        print("tape frame", {k: [round(v) for v in fr[k]] for k in fr}, "|", n, "grid points converted")
    elif cmd == "sticker_ref":
        save_sticker_ref()
        print("saved empty-sticker reference")
    elif cmd == "relearn":
        # python ringbot.py relearn red : the only ring currently in the area is <color>; update its reference color
        color = sys.argv[2]
        img = get_frame()
        cal = rings.load()
        x, y = int(sys.argv[3]), int(sys.argv[4])   # python ringbot.py relearn red 452 230
        tmp = rings.calibrate(img, {color: (x, y)}, table_xy=tuple(cal.get("table_xy", (300, 230))))
        cal["refs"][color] = tmp["refs"][color]
        rings.CALIB.write_text(json.dumps(cal, indent=1))
        print(f"{color} re-learned from the ring at ({x},{y})")
    elif cmd == "find":
        for c, b in rings.detect(get_frame()).items():
            print(f"{c:<7}", [(round(x), round(y)) for x, y, *_ in b[:2]] or "-")
    elif cmd == "autotest":
        # python ringbot.py autotest 3 blue green red   -> 3 attempts per color, ring put back at random spots
        reps, colors = int(sys.argv[2]), sys.argv[3:]
        rng = np.random.default_rng()
        arm = Arm()
        results = []
        try:
            for color in colors:
                for k in range(reps):
                    if arm.locate(color) is None:
                        print(f"[{color} #{k+1}] 桌上找不到 {color}，跳过"); break
                    ok = arm.pick_place(color, log=lambda m: print(f"[{color} #{k+1}] {m}", flush=True))
                    results.append((color, k + 1, bool(ok)))
                    if ok:
                        spot = arm.free_spot(rng)
                        if spot is None:
                            print("没有空位置了，停止"); raise SystemExit
                        arm.return_ring(spot, log=lambda m: print(f"[{color} #{k+1}] {m}", flush=True))
                    time.sleep(1.0)
        finally:
            arm.close()
            n_ok = sum(r[2] for r in results)
            print(f"RESULT {n_ok}/{len(results)} 成功", " ".join(f"{c}{k}:{'✔' if o else '✘'}" for c, k, o in results), flush=True)
    elif cmd == "calib_servo":
        arm = Arm()
        try:
            arm.calib_servo(sys.argv[2])
        finally:
            arm.close()
    elif cmd == "probe":
        arm = Arm()
        try:
            arm.probe(sys.argv[2], [float(v) for v in sys.argv[3:]])
        finally:
            arm.close()
    elif cmd == "run":
        arm = Arm()
        try:
            for color in sys.argv[2:]:
                for attempt in range(1, 1 + 1 + MAX_RETRIES):          # keep re-locating and retrying until it's picked up
                    print(f"第 {attempt} 次尝试", flush=True)
                    ok = arm.pick_place(color, attempt=attempt)
                    if ok:
                        break
                    time.sleep(1.0)                  # arm back at rest, out of the camera's view
                    if arm.locate(color, tries=10) is None:
                        print(f"桌上找不到 {color} 了", flush=True)
                        break
        finally:
            arm.close()
    else:
        print(__doc__)
