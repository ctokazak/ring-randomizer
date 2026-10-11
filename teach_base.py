"""Teach the base grab with the leader: rings are located first, the pose just before the gripper closes is
saved, and the ring that has left its spot afterwards tells which ring (and where) was grabbed."""
import time
from collections import deque

import numpy as np

import rings
import ringbot


def ring_positions():
    return {c: b[0][:2] for c, b in rings.detect(ringbot.get_frame()).items() if b}


def main():
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    ringbot.clear_motor_faults()
    leader = SO101Leader(SO101LeaderConfig(port=ringbot.LEADER_PORT, id="my_leader"))
    follower = ringbot.make_follower()
    leader.connect()
    follower.connect()
    before = ring_positions()
    print("rings before:", {c: (round(x), round(y)) for c, (x, y) in before.items()}, flush=True)
    print("teleop on: grab ONE ring (hold still, close slowly), lift it away from its spot", flush=True)
    hist = deque(maxlen=150)
    pose, t_close = None, None
    try:
        while True:
            t = time.perf_counter()
            a = leader.get_action()
            follower.send_action(a)
            hist.append((time.time(), dict(a)))
            g = [h[1][ringbot.GRIP] for h in hist if h[0] > time.time() - 3.0]
            if pose is None and len(g) > 30 and max(g) - g[-1] >= 5:
                # closing: take the pose from when the gripper was still near its max (just before closing)
                top = max(g)
                cand = [h for h in hist if h[1][ringbot.GRIP] >= top - 0.7]
                pose, t_close = dict(cand[-1][1]), time.time()
                print("close detected, pose saved; now lift the ring away", flush=True)
            if pose is not None and time.time() - t_close > 4.0:
                after = ring_positions()
                gone = [c for c, (x, y) in before.items()
                        if c not in after or np.hypot(after[c][0] - x, after[c][1] - y) > 25]
                if len(gone) != 1:
                    print(f"couldn't tell which ring was grabbed (changed: {gone}); grab again", flush=True)
                    before, pose = ring_positions(), None
                    continue
                c = gone[0]
                p = ringbot.load_poses()
                p["base_grasp"] = {"px": list(before[c]), "grasp": pose, "color": c}
                fr = __import__("tape").find_frame(ringbot.get_frame())
                if fr is not None:   # store it in tape coordinates so camera bumps don't matter
                    p["base_grasp"]["uv"] = __import__("tape").to_uv(fr, list(before[c]))
                if "base_fit" not in p:
                    p["base_fit"] = ringbot.fit_base(p["grid"])
                ringbot.save_poses(p)
                print(f"SAVED base grab: {c} ring at ({before[c][0]:.0f},{before[c][1]:.0f})", flush=True)
                break
            time.sleep(max(0, 1 / 30 - (time.perf_counter() - t)))
    finally:
        leader.disconnect()
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
