"""Clean the marker map and draw the grab grid.
Outliers: a smooth (quadratic) model joints -> pixel is fit; points far off it are dropped (twice)."""
import json
import cv2
import numpy as np
import ringbot

p = ringbot.load_poses()
g = p["grid"]
J = np.array([[x["grasp"][k] for k in ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"]] for x in g])
P = np.array([x["px"] for x in g])
feat = lambda J: np.c_[np.ones(len(J)), J, J ** 2, J[:, :1] * J[:, 1:3]]
keep = np.ones(len(g), bool)
for _ in range(2):
    A = feat(J[keep]); coef, *_ = np.linalg.lstsq(A, P[keep], rcond=None)
    res = np.linalg.norm(feat(J) @ coef - P, axis=1)
    keep = res < max(25, 2.5 * np.median(res[keep]))
print(f"kept {keep.sum()}/{len(g)}  median error {np.median(res[keep]):.1f}px")
p["grid"] = [x for x, k in zip(g, keep) if k]
ringbot.save_poses(p)

img = ringbot.get_frame()
pts = np.array([x["px"] for x in p["grid"]], np.float32)
hull = cv2.convexHull(pts)
over = img.copy()
for u in range(0, 640, 25):
    for v in range(0, 480, 25):
        if cv2.pointPolygonTest(hull, (u + 12.5, v + 12.5), False) >= 0:
            cv2.rectangle(over, (u, v), (u + 25, v + 25), (0, 200, 0), -1)
img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
for u in range(0, 640, 25):
    cv2.line(img, (u, 0), (u, 480), (80, 80, 80), 1)
for v in range(0, 480, 25):
    cv2.line(img, (0, v), (640, v), (80, 80, 80), 1)
for (x, y), k in zip(P, keep):
    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0) if k else (0, 0, 255), -1)
cv2.imwrite("/home/amd-gpu-4/grid_view.png", img)
