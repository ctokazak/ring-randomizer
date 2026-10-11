"""Record a motion by teleoperating with the leader, then let the follower replay it on its own.

  python motion.py record grab_bottle 20   # teleop for 20 s, saved to ~/motions/grab_bottle.json
  python motion.py play grab_bottle        # follower replays it by itself
  python motion.py list

Recordings should start and end in the rest pose: the follower goes limp when playback ends.
"""
import json
import sys
import time
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128141-if00"
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125028-if00"
FPS = 30
MOTIONS = Path.home() / "motions"


def make_follower():
    # max_relative_target caps how far any joint may move in one step
    return SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="my_follower", max_relative_target=10))


def record(name, seconds):
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="my_leader"))
    follower = make_follower()
    leader.connect()
    follower.connect()
    frames = []
    try:
        for i in (3, 2, 1):
            print(f"recording in {i}...", flush=True)
            time.sleep(1)
        print(f"RECORDING {seconds}s", flush=True)
        end = time.time() + seconds
        while time.time() < end:
            t = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            frames.append(action)
            time.sleep(max(0, 1 / FPS - (time.perf_counter() - t)))
    finally:
        MOTIONS.mkdir(exist_ok=True)
        (MOTIONS / f"{name}.json").write_text(json.dumps({"fps": FPS, "frames": frames}))
        print(f"saved {len(frames)} frames to {MOTIONS / name}.json", flush=True)
        leader.disconnect()
        follower.disconnect()


def play(name):
    data = json.loads((MOTIONS / f"{name}.json").read_text())
    frames, fps = data["frames"], data["fps"]
    follower = make_follower()
    follower.connect()
    try:
        # Ease from wherever the arm is to the first frame over 2 s
        obs = follower.get_observation()
        start = {k: obs[k] for k in frames[0]}
        steps = 2 * fps
        for s in range(1, steps + 1):
            a = s / steps
            follower.send_action({k: start[k] + (frames[0][k] - start[k]) * a for k in frames[0]})
            time.sleep(1 / fps)
        print(f"playing {name} ({len(frames) / fps:.1f}s)", flush=True)
        for action in frames:
            t = time.perf_counter()
            follower.send_action(action)
            time.sleep(max(0, 1 / fps - (time.perf_counter() - t)))
        time.sleep(0.5)
    finally:
        follower.disconnect()
        print("done", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "record":
        record(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif cmd == "play":
        play(sys.argv[2])
    else:
        print(sorted(p.stem for p in MOTIONS.glob("*.json")) if MOTIONS.exists() else [])
