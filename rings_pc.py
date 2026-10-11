"""Find colored rings in the scene camera image.

Each pixel is assigned to the nearest reference color in Lab space (or to "background" if it's
closer to the table than to any ring color). Connected blobs of a color, inside the workspace
region, become ring detections. References come from rings_calib.json (see calibrate()).
"""
import json
from pathlib import Path

import cv2
import numpy as np

COLORS = ["red", "yellow", "green", "cyan", "blue", "purple"]   # no orange: orange is the jaw-tip tape
CALIB = Path(__file__).with_name("rings_calib.json")
MIN_AREA, MAX_AREA = 25, 2500


def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def ring_pixels(lab_patch, table_lab, keep=0.3):
    """The pixels in a patch that differ most from the table: the ring itself, not the hole/table."""
    px = lab_patch.reshape(-1, 3)
    d = np.linalg.norm(px - table_lab, axis=1)
    return px[d >= np.quantile(d, 1 - keep)]


def calibrate(img, points, table_xy, roi=None, exclude=None):
    """points: {color: (x, y)} ring centres in img. Each reference = median color of that ring's blob,
    computed exactly the way detect() measures blobs."""
    lab = cv2.GaussianBlur(to_lab(img), (3, 3), 0)
    tx, ty = table_xy
    table = np.median(lab[ty - 10:ty + 10, tx - 10:tx + 10].reshape(-1, 3), axis=0)
    fg = (np.linalg.norm((lab - table) * W, axis=2) > FG_DIST).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(fg)
    refs = {}
    for c, (x, y) in points.items():
        win = labels[y - 8:y + 9, x - 8:x + 9]
        ids, counts = np.unique(win[win > 0], return_counts=True)
        if len(ids) == 0:
            raise ValueError(f"no ring found at {c} {(x, y)}")
        refs[c] = np.median(lab[labels == ids[counts.argmax()]], axis=0).tolist()
    data = {"table": table.tolist(), "table_xy": list(table_xy), "refs": refs, "roi": roi, "exclude": exclude or []}
    CALIB.write_text(json.dumps(data, indent=1))
    return data


def load():
    return json.loads(CALIB.read_text())


W = np.array([0.4, 1.0, 1.0], np.float32)  # weight L less: lighting changes brightness more than hue
FG_DIST = 22    # how different from the table a pixel must be to count as "something on the table"
MAX_CLASS_DIST = 30
MIN_SAT = 60          # rings are strongly colored; the gray boundary tape is not


def detect(img, calib=None, roi=None):
    """Return {color: [(x, y, area, dist), ...]}, best match first.

    1. foreground = pixels clearly different from the table, inside the workspace ROI
    2. each foreground blob is classified as a whole by its median color -> nearest reference
    """
    calib = calib or load()
    roi = roi if roi is not None else calib.get("roi")
    lab = cv2.GaussianBlur(to_lab(img), (3, 3), 0)
    m = None
    if roi:
        m = np.zeros(lab.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.array(roi, np.int32)], 1)
    # table color measured live from this frame (the grab area is mostly bare table) -> robust to lighting changes
    table = np.median(lab[m > 0], axis=0) if m is not None else np.array(calib["table"], np.float32)
    fg = (np.linalg.norm((lab - table) * W, axis=2) > FG_DIST).astype(np.uint8)
    if m is not None:
        fg &= m
    for x, y, r in calib.get("exclude", []):  # e.g. the AMD sticker (its red rim looks like a ring)
        cv2.circle(fg, (int(x), int(y)), int(r), 0, -1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fg)
    refs = np.array([calib["refs"][c] for c in COLORS], np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    out = {c: [] for c in COLORS}
    for j in range(1, n):
        area = int(stats[j][4])
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        if np.median(sat[labels == j]) < MIN_SAT:   # gray tape / shadows / table marks are not rings
            continue
        col = np.median(lab[labels == j], axis=0)
        d = np.linalg.norm((refs - col) * W, axis=1)
        k = int(d.argmin())
        if d[k] < MAX_CLASS_DIST:
            out[COLORS[k]].append((float(cents[j][0]), float(cents[j][1]), area, float(d[k])))
    for c in COLORS:
        out[c].sort(key=lambda b: b[3])
    return out


def draw(img, dets):
    vis = img.copy()
    for c, blobs in dets.items():
        for x, y, *_ in blobs[:1]:
            cv2.circle(vis, (int(x), int(y)), 14, (255, 255, 255), 2)
            cv2.putText(vis, c, (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, c, (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis
