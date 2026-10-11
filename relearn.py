"""Re-learn ring colors from rings currently on the table, measured exactly the way detect() measures blobs.
  python relearn.py red 250 282 blue 342 250 green 405 247 purple 412 323
"""
import json
import sys

import cv2
import numpy as np

import rings
import ringbot

args = sys.argv[1:]
pts = {args[i]: (int(args[i + 1]), int(args[i + 2])) for i in range(0, len(args), 3)}
cal = rings.load()
img = ringbot.get_frame()
lab = cv2.GaussianBlur(rings.to_lab(img), (3, 3), 0)
m = np.zeros(lab.shape[:2], np.uint8)
cv2.fillPoly(m, [np.array(cal["roi"], np.int32)], 1)
table = np.median(lab[m > 0], axis=0)
fg = rings.foreground(lab, table) & m
for ex, ey, er in cal.get("exclude", []):
    cv2.circle(fg, (int(ex), int(ey)), int(er), 0, -1)
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
n, lbl, st, ce = cv2.connectedComponentsWithStats(fg)
for c, (x, y) in pts.items():
    cand = [j for j in range(1, n) if 25 <= st[j][4] <= 2500]
    j = min(cand, key=lambda k: np.hypot(ce[k][0] - x, ce[k][1] - y))
    if np.hypot(ce[j][0] - x, ce[j][1] - y) > 25:
        print(f"{c}: no ring blob near ({x},{y})")
        continue
    cal["refs"][c] = np.median(lab[lbl == j], axis=0).tolist()
    print(f"{c}: learned {np.round(cal['refs'][c])} from blob at ({ce[j][0]:.0f},{ce[j][1]:.0f})")
rings.CALIB.write_text(json.dumps(cal, indent=1))
res = rings.detect(ringbot.get_frame(), cal)
print({c: [(round(a), round(b)) for a, b, *_ in v[:2]] for c, v in res.items() if v})
