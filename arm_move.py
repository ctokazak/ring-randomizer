"""Minimal SO-101 motor tool using the raw Feetech SDK (no LeRobot calibration needed).

  python arm_move.py scan                      # read-only: list motors + positions on each port
  python arm_move.py watch /dev/ttyACM0        # read-only: print positions live (move the arm by hand)
  python arm_move.py wiggle /dev/ttyACM0 6 200 # move one motor +200 ticks, then back (4096 ticks = 1 turn)

Motor IDs on the SO-101: 1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex, 5 wrist_roll, 6 gripper
"""
import sys
import time

import scservo_sdk as scs

BAUD = 1_000_000
ADDR_TORQUE_ENABLE = 40
ADDR_ACCELERATION = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
PORTS = ["/dev/ttyACM0", "/dev/ttyACM1"]
MAX_DELTA = 400  # safety cap: ~35 degrees


def open_bus(port):
    ph = scs.PortHandler(port)
    if not ph.openPort() or not ph.setBaudRate(BAUD):
        sys.exit(f"cannot open {port}")
    return ph, scs.PacketHandler(0)


def check(pk, res, err, what):
    if res != scs.COMM_SUCCESS:
        raise RuntimeError(f"{what}: {pk.getTxRxResult(res)}")
    if err:
        print(f"  warning on {what}: {pk.getRxPacketError(err)}")


def read_pos(ph, pk, mid):
    pos, res, err = pk.read2ByteTxRx(ph, mid, ADDR_PRESENT_POSITION)
    check(pk, res, err, f"read motor {mid}")
    return pos


def found_ids(ph, pk):
    return [mid for mid in range(1, 7) if pk.ping(ph, mid)[1] == scs.COMM_SUCCESS]


def scan():
    for port in PORTS:
        ph, pk = open_bus(port)
        ids = found_ids(ph, pk)
        print(f"{port}: motors {ids}")
        for mid in ids:
            print(f"  {mid} {NAMES[mid]:<14} pos={read_pos(ph, pk, mid)}")
        ph.closePort()


def watch(port):
    ph, pk = open_bus(port)
    ids = found_ids(ph, pk)
    print("Ctrl+C to stop")
    try:
        while True:
            print("  ".join(f"{mid}:{read_pos(ph, pk, mid):4d}" for mid in ids), flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        ph.closePort()


def move_to(ph, pk, mid, goal, timeout=3.0):
    check(pk, *pk.write2ByteTxRx(ph, mid, ADDR_GOAL_POSITION, goal), f"goal motor {mid}")
    t = time.time()
    while time.time() - t < timeout:
        pos = read_pos(ph, pk, mid)
        if abs(pos - goal) < 15:
            break
        time.sleep(0.05)
    return pos


def wiggle(port, mid, delta):
    delta = max(-MAX_DELTA, min(MAX_DELTA, delta))
    ph, pk = open_bus(port)
    start = read_pos(ph, pk, mid)
    target = max(0, min(4095, start + delta))
    print(f"motor {mid} ({NAMES[mid]}): {start} -> {target} -> {start}")
    try:
        # Goal = current position BEFORE torque on, otherwise the motor jumps to a stale goal
        check(pk, *pk.write2ByteTxRx(ph, mid, ADDR_GOAL_POSITION, start), "set goal")
        check(pk, *pk.write1ByteTxRx(ph, mid, ADDR_ACCELERATION, 20), "set accel")
        check(pk, *pk.write2ByteTxRx(ph, mid, ADDR_GOAL_SPEED, 300), "set speed")  # slow
        check(pk, *pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 1), "torque on")
        print("  reached", move_to(ph, pk, mid, target))
        time.sleep(0.5)
        print("  back at", move_to(ph, pk, mid, start))
    finally:
        pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 0)
        ph.closePort()
        print("  torque off")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "scan":
        scan()
    elif cmd == "watch":
        watch(sys.argv[2])
    elif cmd == "wiggle":
        wiggle(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    else:
        sys.exit(__doc__)
