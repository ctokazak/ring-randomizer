"""Teach the 'correct pick-up' wrist view: teleop to the right grasp, hold still, then close slowly.
The wrist frame from just before the close is saved and the ring in it becomes the alignment target.

  python teach_target.py red
"""
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import ringbot

color = sys.argv[1] if len(sys.argv) > 1 else "red"
OUT = Path.home() / "wrist_target.json"


def ring_in(img):
    """Biggest compact, saturated blob that isn't the white gripper: the ring."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    table = np.median(lab[300:420, 100:540].reshape(-1, 3), axis=0)
    diff = np.linalg.norm((lab - table) * np.array([0.3, 1, 1], np.float32), axis=2)
    m = ((hsv[:, :, 1] > 90) & (diff > 25)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, st, ce = cv2.connectedComponentsWithStats(m)
    best = None
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if 800 < a < 20000 and 0.5 < w / max(h, 1) < 2.0 and (best is None or a > st[best][4]):
            best = j
    if best is None:
        return None
    x, y, w, h, a = st[best]
    return {"center": [float(ce[best][0]), float(ce[best][1])], "diameter": float(max(w, h)),
            "lab": np.median(lab[lbl == best], axis=0).tolist()}


def main():
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    ringbot.clear_motor_faults()
    leader = SO101Leader(SO101LeaderConfig(port=ringbot.LEADER_PORT, id="my_leader"))
    follower = ringbot.make_follower()
    leader.connect()
    follower.connect()
    frames = deque(maxlen=20)
    run = {"on": True}

    def cam():
        while run["on"]:
            try:
                frames.append((time.time(), ringbot.get_wrist()))
            except Exception:
                pass
            time.sleep(0.15)

    threading.Thread(target=cam, daemon=True).start()
    grips = deque(maxlen=60)
    print("teleop on: move to the correct pick-up pose, hold still 2 s, then close slowly", flush=True)
    try:
        while True:
            t = time.perf_counter()
            a = leader.get_action()
            follower.send_action(a)
            grips.append((time.time(), a[ringbot.GRIP]))
            recent = [g for ts, g in grips if ts > time.time() - 1.0]
            if len(recent) > 10 and max(recent) - recent[-1] > 6:      # the gripper is closing
                t_close = time.time() - 0.6
                img = min(frames, key=lambda f: abs(f[0] - t_close))[1]
                cv2.imwrite(str(Path.home() / "wrist_target.png"), img)
                r = ring_in(img)
                if r is None:
                    print("couldn't find the ring in the saved wrist frame; try again", flush=True)
                    grips.clear()
                    continue
                data = json.loads(OUT.read_text()) if OUT.exists() else {}
                data.update({"target": r["center"], "diameter": r["diameter"], "refs": {**data.get("refs", {}), color: r["lab"]}})
                OUT.write_text(json.dumps(data, indent=1))
                print(f"SAVED target {r['center'][0]:.0f},{r['center'][1]:.0f} diameter {r['diameter']:.0f}", flush=True)
                break
            time.sleep(max(0, 1 / 30 - (time.perf_counter() - t)))
    finally:
        run["on"] = False
        leader.disconnect()
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
