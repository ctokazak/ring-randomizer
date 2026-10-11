"""Find colored rings in the scene camera image.

Each pixel is assigned to the nearest reference color in Lab space (or to "background" if it's
closer to the table than to any ring color). Connected blobs of a color, inside the workspace
region, become ring detections. References come from rings_calib.json (see calibrate()).
"""
import json
from pathlib import Path

import cv2
import numpy as np

COLORS = ["red", "green", "blue", "purple"]   # the only rings in use
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
MIN_SAT = 60
CLEAR_MARGIN = 4.0    # the best color must beat the runner-up by this much          # rings are strongly colored; the gray boundary tape is not


# hue bands (OpenCV H, 0-179) measured in shade and in direct sun; the wood table / tape sit at 7-18
HUE_BANDS = [("red", 0, 8), ("green", 40, 95), ("blue", 96, 128), ("purple", 129, 164), ("red", 165, 179)]


def hue_class(h, s):
    for name, lo, hi in HUE_BANDS:
        if lo <= h <= hi:
            return name if s >= 45 else None
    return None


def foreground(lab, table):
    """Pixels that differ from the table in COLOR (a,b), or are clearly DARKER. Sunlight only makes the table
    brighter without changing its color, so bright sun patches are not foreground."""
    chroma = np.linalg.norm(lab[:, :, 1:] - table[1:], axis=2)
    darker = table[0] - lab[:, :, 0]
    return ((chroma > 11) | (darker > 28)).astype(np.uint8)


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
    fg = foreground(lab, table)
    if m is not None:
        fg &= m
    for x, y, r in calib.get("exclude", []):  # e.g. the AMD sticker (its red rim looks like a ring)
        cv2.circle(fg, (int(x), int(y)), int(r), 0, -1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fg)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = {c: [] for c in COLORS}
    for j in range(1, n):
        area = int(stats[j][4])
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        sel = labels == j
        hs = hsv_img[sel]
        top = hs[hs[:, 1] >= np.percentile(hs[:, 1], 70)]           # the most colored pixels of the blob
        hues = top[:, 0].astype(float)
        if np.ptp(hues) > 90:                                        # red wraps around 0/180
            hues = (hues + 90) % 180 - 90
        h = float(np.median(hues)) % 180
        sat = float(np.median(top[:, 1]))
        c = hue_class(h, sat)
        if c is None:
            continue
        cx, cy = float(cents[j][0]), float(cents[j][1])
        if c == "red" and any(np.hypot(cx - ex, cy - ey) < er + 12 for ex, ey, er in calib.get("exclude", [])):
            continue                                                 # leftover of the sticker's red rim
        out[c].append((cx, cy, area, -sat))   # most saturated first
    for c in COLORS:   # one ring can show up as two halves (the hole splits it): merge blobs closer than 30 px
        merged = []
        for b in sorted(out[c], key=lambda b: -b[2]):
            for i, mb in enumerate(merged):
                if np.hypot(mb[0] - b[0], mb[1] - b[1]) < 30:
                    a = mb[2] + b[2]
                    merged[i] = ((mb[0] * mb[2] + b[0] * b[2]) / a, (mb[1] * mb[2] + b[1] * b[2]) / a, a, min(mb[3], b[3]))
                    break
            else:
                merged.append(b)
        out[c] = sorted(merged, key=lambda b: b[3])
    return out


def draw(img, dets):
    vis = img.copy()
    for c, blobs in dets.items():
        for x, y, *_ in blobs[:1]:
            cv2.circle(vis, (int(x), int(y)), 14, (255, 255, 255), 2)
            cv2.putText(vis, c, (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, c, (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis
