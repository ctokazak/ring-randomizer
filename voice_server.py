"""Voice control web page: hold the button, say a color (中文 / English / Español), the arm does the rest.

Runs in the `voice` env on the PC at localhost:8091 (reach it from the Mac through an SSH tunnel).
Speech -> faster-whisper (auto language) -> color keyword -> `ringbot.py run <color>` in the lerobot env.
"""
import http.server
import io
import json
import re
import subprocess
import threading
import time
from pathlib import Path

from faster_whisper import WhisperModel

ROBOT_PY = str(Path.home() / "miniforge3/envs/lerobot/bin/python")
RINGBOT = str(Path.home() / "ringbot.py")

# checked in order: multi-word / more specific names first (light blue before blue)
KEYWORDS = [
    ("purple", ["紫", "purple", "violet", "morado", "morada", "violeta", "púrpura", "purpura"]),
    ("green", ["绿", "綠", "green", "verde"]),
    ("blue", ["蓝", "藍", "blue", "azul"]),
    ("red", ["红", "紅", "red", "rojo", "roja"]),
]
NAMES = {"red": "红色", "green": "绿色", "blue": "蓝色", "purple": "紫色"}

model = WhisperModel("small", device="cpu", compute_type="int8")
LANG = "en"   # None = auto-detect language
state = {"busy": False, "log": [], "arm": False}
lock = threading.Lock()


def log(msg):
    state["log"] = (state["log"] + [time.strftime("%H:%M:%S ") + msg])[-12:]
    print(msg, flush=True)


def find_color(text):
    t = text.lower()
    for color, words in KEYWORDS:
        if any(re.search(r"\b" + re.escape(w) + r"\b", t) if w.isascii() else w in t for w in words):
            return color
    return None


def run_arm(color):
    with lock:
        state["busy"] = True
        try:
            log("Arm: picking a random ring" if color == "random" else f"Arm: picking the {color} ring")
            args = ["random"] if color == "random" else ["run", color]
            out = subprocess.run([ROBOT_PY, RINGBOT, *args], capture_output=True, text=True, timeout=400,
                                 cwd=str(Path.home()))
            msgs = []
            for l in out.stdout.splitlines():          # the robot script logs in Chinese: show English here
                if "随机选中" in l:
                    picked = l.split(":")[1].split("（")[0].strip()
                    state["last_pick"] = picked
                    msgs.append("random pick: " + picked)
                elif "圆环在画面" in l:
                    msgs.append("found " + l.split()[0] + " at " + l.split("画面")[1].strip())
                elif "完成 ✔" in l:
                    msgs.append("done ✔ placed on the sticker")
                    state["last_result"] = "placed"
                elif "夹空" in l:
                    msgs.append("empty grab, redoing")
                elif "夹住了" in l:
                    msgs.append("ring gripped ✔")
                elif "抓空了" in l:
                    msgs.append("missed (ring still there), retrying")
                elif "完成 ✘" in l:
                    msgs.append("missed ✘")
                elif "没找到" in l:
                    msgs.append(f"no {color} ring found on the table")
                elif "忙" in l:
                    msgs.append("arm is busy")
            log(" / ".join(msgs) or f"exit {out.returncode}")
        except Exception as e:
            log(f"Error: {e!r}")
        finally:
            state["busy"] = False


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Ring Bot</title><style>
body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
.wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
img{max-width:100%;border-radius:8px}
.panel{flex:1;min-width:280px}
#talk{width:100%;padding:28px;font-size:26px;border:0;border-radius:14px;background:#e4002b;color:#fff;cursor:pointer}
#talk.rec{background:#2e7d32}
.colors{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:14px 0}
.colors button{padding:12px;font-size:16px;border:0;border-radius:8px;color:#000;cursor:pointer}
#heard{font-size:22px;margin:12px 0;min-height:30px}
#log{font-family:monospace;font-size:13px;white-space:pre-wrap;color:#aaa}
</style></head><body><div class="wrap">
<div class="panel" style="flex:2"><img src="http://localhost:8090/stream"></div>
<div class="panel">
<button id="talk">🎤 Hold to talk</button>
<button id="armbtn" onclick="fetch('/toggle_arm').then(r=>r.json()).then(j=>{armbtn.textContent='Arm: '+j.msg.split(' ')[1]})" style="margin:10px 0;padding:10px;font-size:16px">Arm: OFF (listen only)</button>
<div id="heard"></div>
<div class="colors">
<button style="background:#e53935" onclick="go('red')">red</button>
<button style="background:#43a047" onclick="go('green')">green</button>
<button style="background:#5c6bc0" onclick="go('blue')">blue</button>
<button style="background:#8e24aa;color:#fff" onclick="go('purple')">purple</button>
<button style="background:#eee" onclick="go('random')">random</button>
</div><div id="log"></div></div></div>
<script>
const talk = document.getElementById('talk'), heard = document.getElementById('heard');
let rec, chunks = [];
async function start(e){ e.preventDefault();
  const s = await navigator.mediaDevices.getUserMedia({audio:true});
  rec = new MediaRecorder(s); chunks = [];
  rec.ondataavailable = ev => chunks.push(ev.data);
  rec.onstop = async () => { s.getTracks().forEach(t=>t.stop());
    heard.textContent = 'Listening…';
    const r = await fetch('/voice', {method:'POST', body:new Blob(chunks, {type: rec.mimeType})});
    const j = await r.json();
    heard.textContent = '"' + j.text + '" → ' + (j.color ? j.color : 'no color heard') + (j.msg ? '  ' + j.msg : ''); };
  rec.start(); talk.classList.add('rec'); talk.textContent = '🔴 Recording… release to send';
}
function stop(e){ e.preventDefault(); if (rec && rec.state === 'recording'){ rec.stop(); }
  talk.classList.remove('rec'); talk.textContent = '🎤 Hold to talk'; }
talk.addEventListener('mousedown', start); talk.addEventListener('mouseup', stop); talk.addEventListener('mouseleave', stop);
talk.addEventListener('touchstart', start); talk.addEventListener('touchend', stop);
function go(c){ fetch('/go?c=' + c).then(r=>r.json()).then(j=>heard.textContent = c + '  ' + (j.msg||'')); }
setInterval(()=>fetch('/log').then(r=>r.json()).then(j=>document.getElementById('log').textContent =
  (j.busy ? '⏳ Arm running…\\n' : '') + j.log.join('\\n')), 1000);
</script></body></html>"""


NBA_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA Champion Picker</title><style>
:root{--bg:#0d1117;--card:#161b22;--text:#e6edf3;--muted:#8b949e}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px;display:grid;grid-template-columns:2fr 1fr;gap:16px}
@media(max-width:800px){.wrap{grid-template-columns:1fr}}
h1{grid-column:1/-1;margin:4px 0;font-size:28px}
.cam{width:100%;border-radius:10px;background:#000}
.panel{background:var(--card);border-radius:12px;padding:16px}
.team{display:flex;align-items:center;gap:10px;padding:10px;border-radius:8px;margin:6px 0;font-size:18px;border:2px solid transparent}
.dot{width:22px;height:22px;border-radius:50%}
.team.win{border-color:#ffd700;background:#2a2410}
#go{width:100%;padding:22px;font-size:24px;border:0;border-radius:12px;background:#ffd700;color:#000;cursor:pointer;margin-top:10px}
#go:disabled{opacity:.5;cursor:wait}
#result{font-size:26px;margin:14px 0;min-height:34px;text-align:center}
#status{color:var(--muted);font-size:14px;text-align:center}
</style></head><body><div class="wrap">
<h1>🏆 Who wins the NBA championship?</h1>
<img class="cam" src="http://localhost:8090/stream">
<div class="panel">
<div class="team" id="t-red"><span class="dot" style="background:#e53935"></span>Miami Heat</div>
<div class="team" id="t-green"><span class="dot" style="background:#43a047"></span>Boston Celtics</div>
<div class="team" id="t-blue"><span class="dot" style="background:#3f51b5"></span>Philadelphia 76ers</div>
<div class="team" id="t-purple"><span class="dot" style="background:#8e24aa"></span>Los Angeles Lakers</div>
<button id="go">Pick the champion!</button>
<div id="result"></div><div id="status"></div>
</div></div>
<script>
const TEAMS = {red: "Miami Heat", green: "Boston Celtics", blue: "Philadelphia 76ers", purple: "Los Angeles Lakers"};
const go = document.getElementById("go"), result = document.getElementById("result"), status = document.getElementById("status");
let waiting = false;
go.onclick = () => {
  document.querySelectorAll(".team").forEach(t => t.classList.remove("win"));
  result.textContent = "🤖 The robot is choosing…"; status.textContent = ""; go.disabled = true; waiting = true;
  fetch("/go?c=random").then(r => r.json()).then(j => { if (j.msg && j.msg !== "Running") { status.textContent = j.msg; go.disabled = false; waiting = false; } });
};
setInterval(() => fetch("/log").then(r => r.json()).then(j => {
  if (!waiting) return;
  if (j.last_pick) {
    document.getElementById("t-" + j.last_pick)?.classList.add("win");
    result.textContent = "🏆 " + TEAMS[j.last_pick] + "!";
  }
  status.textContent = j.busy ? "⏳ The robot is placing the ring on the sticker…" : (j.last_result === "placed" ? "✔ Ring placed on the sticker" : "");
  if (!j.busy && j.last_pick) { go.disabled = false; waiting = false; }
}), 1000);
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def start_arm(self, color):
        if color == "random":
            state["last_pick"], state["last_result"] = None, None
        if not state["arm"]:
            return "Listen-only: arm control is off"
        if state["busy"]:
            return "Arm is busy, please wait"
        threading.Thread(target=run_arm, args=(color,), daemon=True).start()
        return "Running"

    def do_GET(self):
        if self.path.startswith("/go?c="):
            color = self.path.split("=", 1)[1]
            msg = self.start_arm(color) if (color in NAMES or color == "random") else "Unknown color"
            return self.send(json.dumps({"msg": msg}))
        if self.path == "/toggle_arm":
            state["arm"] = not state["arm"]
            log("arm control " + ("ON" if state["arm"] else "OFF"))
            return self.send(json.dumps({"msg": "arm " + ("ON" if state["arm"] else "OFF")}))
        if self.path == "/nba":
            return self.send(NBA_PAGE, "text/html")
        if self.path == "/log":
            return self.send(json.dumps(state))
        return self.send(PAGE, "text/html")

    def do_POST(self):
        audio = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        t = time.time()
        segs, info = model.transcribe(io.BytesIO(audio), beam_size=1, vad_filter=False, language=LANG,
                                      initial_prompt="red green blue purple")
        text = "".join(s.text for s in segs).strip()
        color = find_color(text)
        log(f'Heard [{info.language}] "{text}" -> {color} ({time.time() - t:.1f}s)')
        msg = self.start_arm(color) if color else ""
        self.send(json.dumps({"text": text, "lang": info.language, "color": color, "msg": msg}, ensure_ascii=False))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    log("voice server ready on :8091")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8091), Handler).serve_forever()
