"""Live MJPEG preview of both cameras, with a board-alignment overlay, on localhost:8090 (view via SSH tunnel).

Buttons on the page:
  Save reference  - save the current scene frame as the reference (do this once the peg board is taped in place)
  对位叠加    - toggle drawing the reference's edges (magenta) over the live scene view
"""
import http.server
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import json

import rings

POSES = Path.home() / "ringbot_poses.json"


def grid_overlay(f):
    """Shade the grabbable area (cells inside the mapped points) green and draw the mapped points."""
    try:
        if state.get("grid_t", 0) < time.time() - 2:
            pts = np.array([g["px"] for g in json.loads(POSES.read_text())["grid"]], np.float32)
            state["grid_pts"], state["grid_t"] = pts, time.time()
        pts = state["grid_pts"]
        hull = cv2.convexHull(pts)
        over = f.copy()
        for u in range(0, f.shape[1], 25):
            for v in range(0, f.shape[0], 25):
                if cv2.pointPolygonTest(hull, (u + 12.5, v + 12.5), False) >= 0:
                    cv2.rectangle(over, (u, v), (u + 24, v + 24), (0, 200, 0), -1)
        f = cv2.addWeighted(over, 0.3, f, 0.7, 0)
        for x, y in pts:
            cv2.circle(f, (int(x), int(y)), 3, (0, 255, 0), -1)
    except Exception:
        pass
    return f

CAMS = {
    "scene": "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
    "wrist": "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
}
REF_PATH = Path.home() / "scene_ref.png"
state = {"grid": False, "jpg": b"", "jpg_small": b"", "scene_raw": None, "overlay": True, "ref_edges": None, "detect": True, "dets": None, "n": 0}


def load_ref():
    if REF_PATH.exists():
        ref = cv2.imread(str(REF_PATH))
        state["ref_edges"] = cv2.dilate(cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 60, 150), np.ones((2, 2), np.uint8))


def grab_loop():
    caps = {name: cv2.VideoCapture(path, cv2.CAP_V4L2) for name, path in CAMS.items()}
    # Let auto exposure / white balance settle, then lock them on the scene camera so ring colors stay stable
    sc = caps["scene"]
    sc.set(cv2.CAP_PROP_AUTO_WB, 1)
    sc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    for _ in range(60):
        sc.read()
    # (white balance stays automatic: this camera reports a bogus WB value, and locking it tints the image)
    # Manual exposure, stepped down until the table (lower half of the image) isn't washed out
    sc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    exp, level = None, None
    fixed = rings.load().get("exposure") if rings.CALIB.exists() else None  # colors were calibrated at this exposure
    for exp in ([fixed] if fixed else [200, 150, 120, 100, 80, 65, 50, 40, 30, 20]):
        sc.set(cv2.CAP_PROP_EXPOSURE, exp)
        for _ in range(6):
            ok, f = sc.read()
        level = float(np.median(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[240:, :, 2]))
        if level < 175:
            break
    print(f"scene camera exposure locked at {exp} (table brightness {level:.0f})", flush=True)
    # wrist camera: it was blowing out (V=255 everywhere), which hid the ring and the jaw tape -> lock it darker
    wc = caps["wrist"]
    wc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    wexp, wlevel = None, None
    for wexp in [150, 120, 100, 80, 65, 50, 40, 30, 20, 15, 10]:
        wc.set(cv2.CAP_PROP_EXPOSURE, wexp)
        for _ in range(6):
            ok, f = wc.read()
        wlevel = float(np.median(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[150:330, :, 2]))
        if wlevel < 190:
            break
    print(f"wrist camera exposure locked at {wexp} (brightness {wlevel:.0f})", flush=True)
    while True:
        frames = []
        for name, cap in caps.items():
            ok, f = cap.read()
            f = f if ok else np.zeros((480, 640, 3), np.uint8)
            state[name + "_raw"] = f.copy()
            if name == "scene":
                if state["overlay"] and state["ref_edges"] is not None:
                    f[state["ref_edges"] > 0] = (255, 0, 255)
                if state["grid"]:
                    f = grid_overlay(f)
                try:   # the tape outline (4 clicked inner corners) = the grid / grab area, in yellow
                    cal = rings.load()
                    quad = cal.get("tape_corners") or cal["roi"]
                    cv2.polylines(f, [np.array(quad, np.int32)], True, (0, 255, 255), 2)
                    for i, (x, y) in enumerate(state.get("clicks", [])):
                        cv2.circle(f, (int(x), int(y)), 6, (0, 0, 255), 2)
                        cv2.putText(f, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except Exception:
                    pass
                if state["detect"]:
                    state["n"] += 1
                    if state["n"] % 5 == 0 or state["dets"] is None:
                        try:
                            state["dets"] = rings.detect(state["scene_raw"])
                        except Exception:
                            state["dets"] = {}
                    f = rings.draw(f, state["dets"] or {})

            label = name + ("  [ref overlay]" if name == "scene" and state["overlay"] and state["ref_edges"] is not None else "")
            cv2.putText(f, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frames.append(f)
        both = np.hstack(frames)
        state["jpg"] = cv2.imencode(".jpg", both, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        state["jpg_small"] = cv2.imencode(".jpg", cv2.resize(both, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 55])[1].tobytes()
        time.sleep(0.05)


PAGE = """<html><head><meta charset="utf-8"><style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif}
.bar{padding:8px;display:flex;gap:8px;align-items:center}
button{font-size:16px;padding:6px 14px}
</style></head><body>
<div class="bar"><button onclick="fetch('/save_ref').then(r=>r.text()).then(t=>msg.textContent=t)">Save reference</button>
<button onclick="fetch('/toggle').then(r=>r.text()).then(t=>msg.textContent=t)">Alignment overlay on/off</button>
<button onclick="fetch('/toggle_grid').then(r=>r.text()).then(t=>msg.textContent=t)">Grab grid on/off</button>
<button onclick="fetch('/toggle_detect').then(r=>r.text()).then(t=>msg.textContent=t)">Color labels on/off</button>
<button onclick="clicks=[];fetch('/corners_reset').then(r=>r.text()).then(t=>msg.textContent=t)">Click tape corners</button>
<span id="msg"></span></div>
<img id="v" src="/stream" style="width:100%">
<script>
// click the 4 INNER corners of the gray tape on the scene (left) image: top-left, top-right, bottom-right, bottom-left
document.getElementById("v").addEventListener("click", e => {
  const r = e.target.getBoundingClientRect();
  const fullW = e.target.naturalWidth, fullH = e.target.naturalHeight;          // both cameras side by side
  const x = (e.clientX - r.left) * fullW / r.width, y = (e.clientY - r.top) * fullH / r.height;
  const sx = x * 1280 / fullW, sy = y * 480 / fullH;                               // in 1280x480 stream coords
  if (sx >= 640) return;                                                           // only the scene image
  fetch(`/corner?x=${sx.toFixed(1)}&y=${sy.toFixed(1)}`).then(r => r.text()).then(t => msg.textContent = t);
});
</script>
<script>if (location.hostname !== "localhost" || location.search.includes("small")) {}
if (location.search.includes("small")) document.getElementById("v").src = "/small";</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def reply(self, body, ctype="text/plain; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def do_GET(self):
        if self.path == "/save_ref":
            if state["scene_raw"] is None:
                return self.reply("no frame yet")
            cv2.imwrite(str(REF_PATH), state["scene_raw"])
            load_ref()
            state["overlay"] = True
            return self.reply("reference saved " + time.strftime("%H:%M:%S"))
        if self.path == "/toggle":
            state["overlay"] = not state["overlay"]
            return self.reply("overlay: " + ("on" if state["overlay"] else "off"))
        if self.path == "/corners_reset":
            state["clicks"] = []
            return self.reply("click the 4 INNER tape corners: top-left, top-right, bottom-right, bottom-left")
        if self.path.startswith("/corner?"):
            q = dict(kv.split("=") for kv in self.path.split("?", 1)[1].split("&"))
            state.setdefault("clicks", []).append((float(q["x"]), float(q["y"])))
            n = len(state["clicks"])
            if n < 4:
                return self.reply(f"corner {n}/4 saved")
            quad = state["clicks"][:4]
            state["clicks"] = []
            cal = rings.load()
            cal["tape_corners"] = [[round(x), round(y)] for x, y in quad]
            # grab area = the tape square pulled in 5% toward its centre
            c = np.mean(np.array(quad), axis=0)
            cal["roi"] = [[round(c[0] + (x - c[0]) * 0.92), round(c[1] + (y - c[1]) * 0.92)] for x, y in quad]
            rings.CALIB.write_text(json.dumps(cal, indent=1))
            return self.reply("tape outline saved: " + str(cal["tape_corners"]))
        if self.path == "/toggle_grid":
            state["grid"] = not state["grid"]
            return self.reply("grid: " + ("on" if state["grid"] else "off"))
        if self.path == "/toggle_detect":
            state["detect"] = not state["detect"]
            return self.reply("color labels: " + ("on" if state["detect"] else "off"))
        if self.path in ("/scene.png", "/wrist.png"):
            raw = state.get(self.path[1:-4] + "_raw")
            if raw is None:
                self.send_response(503); self.end_headers(); return
            return self.reply(cv2.imencode(".png", raw)[1].tobytes(), "image/png")
        if self.path not in ("/stream", "/small"):
            return self.reply(PAGE, "text/html; charset=utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpg = state["jpg_small"] if self.path == "/small" else state["jpg"]
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


load_ref()
threading.Thread(target=grab_loop, daemon=True).start()
http.server.ThreadingHTTPServer(("127.0.0.1", 8090), Handler).serve_forever()
