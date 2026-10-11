"""Hands-free grid teaching: teleop runs, everything is logged, grid points are extracted afterwards.

  python demo_session.py record     # teleop + log until ~/demo_stop exists (touch it to stop)
  python demo_session.py extract    # find grasps in the log -> grid points in ringbot_poses.json

The user just keeps picking up the GRID_COLOR ring and putting it down somewhere new.
A grasp = the gripper closing. Its grid point:
  grasp pose = arm pose right before the gripper started closing
  hover pose = arm pose HOVER_LEAD seconds earlier (on the way down)
  ring pixel = where the camera saw the ring while it was lying still before that grasp
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import rings
import ringbot

LOG = Path.home() / "demo_log.json"
STOP = Path.home() / "demo_stop"
GRID_COLOR = ringbot.GRID_COLOR
GRIP = ringbot.GRIP
HOVER_LEAD = 1.2


def sticker_color(img):
    """Ring color lying on the sticker (by hue, inside the red rim), or None."""
    cal = rings.load()
    x, y, r = cal["exclude"][0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(m, (int(x), int(y)), int(r) - 12, 1, -1)
    sel = (m > 0) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 50)
    if sel.sum() < 80:
        return None
    hs = hsv[:, :, 0][sel].astype(float)
    if np.ptp(hs) > 90:
        hs = (hs + 90) % 180 - 90
    return rings.hue_class(float(np.median(hs)) % 180, 255)


def record():
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    STOP.unlink(missing_ok=True)
    ringbot.clear_motor_faults()
    leader = SO101Leader(SO101LeaderConfig(port=ringbot.LEADER_PORT, id="my_leader"))
    follower = ringbot.make_follower()
    leader.connect()
    follower.connect()
    log = {"actions": [], "rings": []}
    run = {"on": True}

    def camera():
        while run["on"]:
            try:
                img = ringbot.get_frame()
                dd = rings.detect(img)
                log["rings"].append([time.time(), {c: list(b[0][:2]) for c, b in dd.items() if b}])
                log.setdefault("sticker", []).append([time.time(), sticker_color(img)])
            except Exception:
                pass
            time.sleep(0.3)

    th = threading.Thread(target=camera, daemon=True)
    th.start()
    print("recording; touch ~/demo_stop to finish", flush=True)
    last_save = time.time()
    try:
        while not STOP.exists():
            t = time.perf_counter()
            a = leader.get_action()
            follower.send_action(a)
            log["actions"].append([time.time(), {k: float(v) for k, v in a.items()}])
            if time.time() - last_save > 10:
                LOG.write_text(json.dumps(log))
                last_save = time.time()
            time.sleep(max(0, 1 / ringbot.FPS - (time.perf_counter() - t)))
    finally:
        run["on"] = False
        LOG.write_text(json.dumps(log))
        leader.disconnect()
        # keep the follower holding its pose (no torque-off): just close the port
        follower.bus.disconnect(disable_torque=False)
        print(f"saved {len(log['actions'])} actions, {len(log['rings'])} camera samples", flush=True)


def extract(apply=True):
    """Grid points from 'precise' demos: arm held still with the ring between the open jaws, then a slow close.
    grasp pose = mean pose over the still period right before the gripper closes."""
    log = json.loads(LOG.read_text())
    ts = np.array([a[0] for a in log["actions"]])
    keys = [k for k in log["actions"][0][1] if k != GRIP]
    J = np.array([[a[1][k] for k in keys] for a in log["actions"]])
    g = np.array([a[1][GRIP] for a in log["actions"]])
    gs = np.convolve(g, np.ones(5) / 5, mode="same")
    speed = np.r_[0, np.abs(np.diff(J, axis=0)).sum(axis=1)]
    speed = np.convolve(speed, np.ones(9) / 9, mode="same")
    points = []
    i = 1
    while i < len(gs):
        # a real close: gripper drops by >= 8 within 4 s
        j = int(np.searchsorted(ts, ts[i] + 4.0))
        if gs[i - 1] > gs[i] and j < len(gs) and gs[i] - gs[min(j, len(gs) - 1)] >= 5 and gs[i - 1] - gs[i] > 0.1:
            t_c = ts[i]
            # still period: within the 3 s before the close, the arm barely moving for >= 0.6 s
            win = np.where((ts > t_c - 3.0) & (ts < t_c - 0.05))[0]
            still = win[speed[win] < 0.5]
            if len(still) < int(0.35 * ringbot.FPS):
                print(f"  close @ {time.strftime('%H:%M:%S', time.localtime(t_c))}: arm not held still, skipped")
                i = j
                continue
            # only the still stretch that ends right before the close (not an earlier pause, e.g. at rest)
            run = [still[-1]]
            for s in still[::-1][1:]:
                if run[-1] - s <= 2:
                    run.append(s)
                else:
                    break
            if ts[i] - ts[run[0]] > 1.0:
                print(f"  close @ {time.strftime('%H:%M:%S', time.localtime(t_c))}: not held still right before closing, skipped")
                i = j
                continue
            still = np.array(sorted(run))
            grasp = {k: float(J[still, n].mean()) for n, k in enumerate(keys)}
            if grasp["shoulder_lift.pos"] < -50 and grasp["elbow_flex.pos"] > 85:
                print(f"  close @ {time.strftime('%H:%M:%S', time.localtime(t_c))}: arm was folded (rest pose), skipped")
                i = j
                continue
            grasp[GRIP] = float(gs[still].mean())
            # which ring was grasped: the one that was lying still before and is gone/moved a few seconds after
            px, who = None, None
            for color in rings.COLORS:
                before = np.array([d[color] for t, d in log["rings"] if t_c - 10 < t < ts[still[0]] - 0.3 and color in d])
                after = np.array([d[color] for t, d in log["rings"] if t_c + 2.5 < t < t_c + 6 and color in d])
                if len(before) < 3:
                    continue
                pos = np.median(before, axis=0)
                if len(after) == 0 or np.linalg.norm(np.median(after, axis=0) - pos) > 25:
                    px, who = pos.tolist(), color
            if px is None:
                print(f"  close @ {time.strftime('%H:%M:%S', time.localtime(t_c))}: couldn't tell which ring, skipped")
                i = j
                continue
            # verify: was the ring actually lifted? (it must not still be sitting at its spot 1.5-3.5 s later)
            after = [dd[who] for t, dd in log["rings"] if t_c + 1.5 < t < t_c + 3.5 and isinstance(dd, dict) and who in dd]
            left = sum(np.hypot(a[0] - px[0], a[1] - px[1]) < 20 for a in after)
            if after and left >= len(after) * 0.3:
                print(f"  close @ {time.strftime('%H:%M:%S', time.localtime(t_c))}: {who} ring left on the table, not used")
                i = j
                continue
            points.append({"px": px, "grasp": grasp, "hover": grasp, "closed_grip": float(gs[min(j, len(gs) - 1)]),
                           "good": True, "verified": True, "t": t_c})
            print(f"  close @ {time.strftime('%H:%M:%S', time.localtime(t_c))}: {who} ring ({px[0]:.0f},{px[1]:.0f}) "
                  f"held {len(still) / ringbot.FPS:.1f}s, gripper {grasp[GRIP]:.0f} -> {gs[min(j, len(gs) - 1)]:.0f}")
            i = j
        else:
            i += 1
    if apply:
        poses = ringbot.load_poses()
        if "--demos" in sys.argv:            # training demos: kept apart from the locked grid
            poses["demos"] = poses.get("demos", []) + points
            ringbot.save_poses(poses)
            print(f"{len(points)} demos saved ({len(poses['demos'])} total)")
            return points
        poses["grid"] = (poses["grid"] if "--append" in sys.argv else []) + points
        ringbot.save_poses(poses)
        print(f"{len(points)} grid points saved")
    return points


def extract_drops():
    log = json.loads(LOG.read_text())
    ts = np.array([a[0] for a in log["actions"]])
    g = np.convolve(np.array([a[1][GRIP] for a in log["actions"]]), np.ones(5) / 5, mode="same")
    st = log.get("sticker", [])
    drops = []
    i = 1
    while i < len(g):
        j = int(np.searchsorted(ts, ts[i] + 2.0))
        if j < len(g) and g[i - 1] < g[i] and g[min(j, len(g) - 1)] - g[i] >= 5:      # gripper opening = release
            t_r = ts[i]
            before = [c for t, c in st if t_r - 1.5 < t < t_r - 0.2]
            after = [c for t, c in st if t_r + 1.0 < t < t_r + 3.0 and c]
            new = max(set(after), key=after.count) if after else None
            if new and before.count(new) < len(before) * 0.5:     # a ring appeared on the sticker -> a drop
                drops.append({"pose": log["actions"][i][1], "color": new, "t": t_r})
                print(f"  drop @ {time.strftime('%H:%M:%S', time.localtime(t_r))}: {new} placed on the sticker")
            i = j
        else:
            i += 1
    poses = ringbot.load_poses()
    poses["drops"] = poses.get("drops", []) + drops
    ringbot.save_poses(poses)
    print(f"{len(drops)} drops saved ({len(poses['drops'])} total)")


if __name__ == "__main__":
    {"record": record, "extract": extract, "drops": extract_drops}[sys.argv[1]]()
