"""Read-only: show leader vs follower joint positions as LeRobot sees them (normalized).

Put both arms in the same pose. Joints whose numbers differ a lot are mis-calibrated
and will make the follower jump to a "random" pose when teleop starts.

  python compare_arms.py [seconds]
"""
import json
import sys
import time
from pathlib import Path

import scservo_sdk as scs

CAL = Path.home() / ".cache/huggingface/lerobot/calibration"
ARMS = {
    "leader": ("/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128141-if00", CAL / "teleoperators/so101_leader/my_leader.json"),
    "follower": ("/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125028-if00", CAL / "robots/so101_follower/my_follower.json"),
}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def normalize(raw, c, joint):
    # Same mapping LeRobot uses: -100..100 for joints, 0..100 for the gripper
    frac = (raw - c["range_min"]) / (c["range_max"] - c["range_min"])
    return frac * 100 if joint == "gripper" else frac * 200 - 100


buses = {}
for name, (port, cal_file) in ARMS.items():
    ph = scs.PortHandler(port)
    ph.openPort()
    ph.setBaudRate(1_000_000)
    buses[name] = (ph, scs.PacketHandler(0), json.loads(cal_file.read_text()))

end = time.time() + (float(sys.argv[1]) if len(sys.argv) > 1 else 15)
while time.time() < end:
    rows = []
    for j in JOINTS:
        vals = []
        for name in ("leader", "follower"):
            ph, pk, cal = buses[name]
            raw = pk.read2ByteTxRx(ph, cal[j]["id"], 56)[0]
            vals.append(normalize(raw, cal[j], j))
        flag = "  <-- MISMATCH" if abs(vals[0] - vals[1]) > 15 else ""
        rows.append(f"{j:<14} leader {vals[0]:7.1f}   follower {vals[1]:7.1f}{flag}")
    print("\n".join(rows) + "\n", flush=True)
    time.sleep(1)

for ph, _, _ in buses.values():
    ph.closePort()
