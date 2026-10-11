"""Build the pixel <-> arm-pose grid by watching a colored tape marker on the gripper (no rings needed).

  python map_marker.py      # rings out of the area, tape on the fixed jaw tip
"""
import json
import time

import cv2
import numpy as np

import rings
import ringbot

import sys

PAN_OFFSETS = (-10.0, -5.0, 0.0, 5.0, 10.0)
WIDE = "--wide" in sys.argv   # extended sweep to cover the whole taped area (adds to the existing map)
WIDE_PAN = (-35.0, -28.0, -21.0, -14.0, 14.0, 21.0, 28.0, 35.0)
WIDE_REACH = (-6.0, 0.0, 6.0)


def marker(empty, img, roi):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    changed = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(empty, (5, 5), 0)).max(axis=2) > 35
    m = ((hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 70) & changed).astype(np.uint8)
    r = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(r, [np.array(roi, np.int32)], 1)
    m &= r
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(m)
    c = [(st[j][4], ce[j]) for j in range(1, n) if 30 < st[j][4] < 3000]
    return [float(v) for v in max(c, key=lambda t: t[0])[1]] if c else None


def main():
    arm = ringbot.Arm()
    p = arm.poses
    roi = rings.load()["roi"]
    taught = json.load(open(ringbot.POSES.with_name("ringbot_poses.bak-oldview.json")))["grid"]
    if WIDE:   # 10 well-spread seed poses are enough for the wide sweep
        taught = taught[::2]
    try:
        arm.move({**p["rest"], ringbot.GRIP: 40.0}, 2.0)
        time.sleep(0.8)
        empty = ringbot.get_frame()
        new = []
        combos = [(g, off, 0.0) for g in taught for off in PAN_OFFSETS] if not WIDE else \
                 [(g, off, dr) for g in taught for off in WIDE_PAN for dr in WIDE_REACH] + \
                 [(g, 0.0, dr) for g in taught for dr in (-6.0, 6.0)]
        for i, (g, off, dr) in enumerate(combos):
            pose = ringbot.nudge({**g["grasp"], "shoulder_pan.pos": g["grasp"]["shoulder_pan.pos"] + off, ringbot.GRIP: 40.0}, 0.0, dr)
            top = ringbot.lift_above(pose, p, 0.4)
            arm.move(top, 0.8)
            pose = arm.touch_down(top, p, max_down=0.8, back=0.02, log=lambda m: None) if WIDE else (arm.move(pose, 0.6) or pose)
            time.sleep(0.4)
            px = marker(empty, ringbot.get_frame(), roi)
            arm.move(top, 0.4)
            if px is None:
                print(f"{i + 1}/{len(combos)} pan {off:+.0f} reach {dr:+.0f}: marker not seen", flush=True)
                continue
            new.append({"px": px, "grasp": {**pose, ringbot.GRIP: ringbot.OPEN_WIDE}, "hover": ringbot.lift_above(pose, p),
                        "closed_grip": ringbot.GRIP_TIGHT, "good": True})
            print(f"{i + 1}/{len(combos)} pan {off:+.0f} reach {dr:+.0f}: marker at ({px[0]:.0f},{px[1]:.0f})", flush=True)
        arm.move(p["rest"], 2.0)
        p["grid"] = (p["grid"] + new) if WIDE else new
        ringbot.save_poses(p)
        print(f"MAPPED {len(new)} points", flush=True)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
