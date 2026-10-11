"""The gray tape square is the coordinate system.

Each frame we find the INNER edges of the top, left and right tape strips:
  origin = inner top-left corner, x axis -> inner top-right corner (that length = 1 unit),
  y axis = down along the left strip (same unit length).
Positions are stored as (u, v) in these units, so a bumped camera doesn't break anything.
"""
import cv2
import numpy as np


def tape_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # the "gray" tape reads bluish-gray to this camera (H 105-125), or neutral when washed out
    bluish = (h >= 95) & (h <= 135) & (sat < 140)
    neutral = sat < 30
    m = ((bluish | neutral) & (v > 60)).astype(np.uint8)   # sunlit tape can be very bright: no upper limit
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _fit_line(pts):
    """Robust line through points: returns (point, unit direction)."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 8:
        return None
    best = None
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        d = b - a
        if np.linalg.norm(d) < 1:
            continue
        d = d / np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]])
        inl = np.abs((pts - a) @ nrm) < 3
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 6:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts[best], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def find_frame(img, hint=None):
    """Returns {'origin', 'ex', 'ey'} in pixels, or None. hint = previous frame (to look in the right place)."""
    m = tape_mask(img)
    h, w = m.shape
    # the top strip: the longest mostly-horizontal run of tape rows in the upper part of the image
    rows = m[:, w // 5: 4 * w // 5].mean(axis=1)
    cand = np.where(rows > 0.5)[0]
    cand = cand[cand < h * 0.75]
    if len(cand) == 0:
        return None
    top_rows = cand
    # inner (lower) edge of the top strip, per column
    top_pts = []
    for x in range(w // 6, 5 * w // 6, 6):
        col = m[:, x]
        ys = np.where(col[: int(top_rows.max()) + 40] > 0)[0]
        ys = ys[ys >= top_rows.min() - 5]
        if len(ys):
            top_pts.append((x, ys.max()))
    top = _fit_line(top_pts)
    if top is None:
        return None
    y_top = int(np.median([p[1] for p in top_pts]))
    # inner edges of the left / right strips, per row below the top strip
    left_pts, right_pts = [], []
    for y in range(y_top + 15, h - 5, 6):
        row = m[y]
        xs = np.where(row > 0)[0]
        if not len(xs):
            continue
        mid = w // 2
        lx = xs[xs < mid]
        rx = xs[xs > mid]
        # keep only strips (runs of >= 12 px), take their inner edge
        if len(lx):
            runs = np.split(lx, np.where(np.diff(lx) > 2)[0] + 1)
            runs = [r for r in runs if len(r) >= 12]
            if runs:
                left_pts.append((runs[-1].max(), y))
        if len(rx):
            runs = np.split(rx, np.where(np.diff(rx) > 2)[0] + 1)
            runs = [r for r in runs if len(r) >= 12]
            if runs:
                right_pts.append((runs[0].min(), y))
    left, right = _fit_line(left_pts), _fit_line(right_pts)
    if left is None or right is None:
        return None
    tl, tr = _intersect(top, left), _intersect(top, right)
    ex = tr - tl
    unit = np.linalg.norm(ex)
    # the tape is a square: the sides are at right angles to the top edge (the gripper or the image edge
    # can hide/distort the side strips, the top strip is always clear)
    ey = np.array([-ex[1], ex[0]])
    if ey[1] < 0:
        ey = -ey
    if not (100 < unit < 640):
        return None
    return {"origin": tl.tolist(), "ex": ex.tolist(), "ey": ey.tolist()}


def to_uv(frame, px):
    o, ex, ey = (np.array(frame[k], float) for k in ("origin", "ex", "ey"))
    M = np.array([ex, ey]).T
    return np.linalg.solve(M, np.asarray(px, float) - o).tolist()


def to_px(frame, uv):
    o, ex, ey = (np.array(frame[k], float) for k in ("origin", "ex", "ey"))
    return (o + uv[0] * ex + uv[1] * ey).tolist()
