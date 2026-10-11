"""Align leader + follower calibrations from one matched pose, so teleop copies joint-for-joint.

Put both arms side by side in the SAME pose (folded rest pose is easiest), then:
  python align_follower.py           # measure only, show the plan
  python align_follower.py --apply   # back up and rewrite both calibration files + motor limits

For each joint we measure d = follower_raw - leader_raw in the matched pose and give both arms
ranges of identical span, with the follower's shifted by d. LeRobot's normalization is linear in
the range, so the follower's target becomes exactly leader_raw + d: a 1:1 copy anchored on that pose.
"""
import shutil
import sys
import time

from lerobot.motors import MotorCalibration
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128141-if00"
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125028-if00"
MARGIN = 20  # keep clear of the 0 / 4095 encoder wrap

leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="my_leader"))
follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="my_follower"))
apply = "--apply" in sys.argv

for dev in (leader, follower):
    dev.bus.connect()

lead_raw = leader.bus.sync_read("Present_Position", normalize=False)
foll_raw = follower.bus.sync_read("Present_Position", normalize=False)

new_lead, new_foll = {}, {}
print(f"{'joint':<14}{'leader':>7}{'follower':>9}{'offset':>8}   new leader range   new follower range")
for m in leader.bus.motors:
    d = foll_raw[m] - lead_raw[m]
    lo = max(MARGIN, MARGIN - d)
    hi = min(4095 - MARGIN, 4095 - MARGIN - d)
    lc, fc = leader.calibration[m], follower.calibration[m]
    new_lead[m] = MotorCalibration(lc.id, lc.drive_mode, lc.homing_offset, lo, hi)
    new_foll[m] = MotorCalibration(fc.id, fc.drive_mode, fc.homing_offset, lo + d, hi + d)
    print(f"{m:<14}{lead_raw[m]:>7}{foll_raw[m]:>9}{d:>+8}   {lo:>5}..{hi:<5}        {lo + d:>5}..{hi + d:<5}")

if apply:
    stamp = time.strftime("%H%M%S")
    for dev, cal in ((leader, new_lead), (follower, new_foll)):
        shutil.copy(dev.calibration_fpath, f"{dev.calibration_fpath}.bak-{stamp}")
        dev.bus.disable_torque()
        dev.calibration = cal
        dev.bus.write_calibration(cal)
        dev._save_calibration()
        print("saved", dev.calibration_fpath, "(backup .bak-" + stamp + ")")
else:
    print("\nDry run: nothing written. Re-run with --apply once both arms are posed identically.")

for dev in (leader, follower):
    dev.bus.disconnect(disable_torque=False)
