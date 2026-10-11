# Kimchi Tacos: voice-controlled ring-picking robot arm

**AMD × LeRobot hackathon project.** Say a color out loud (or tap a button) and an SO-101 arm finds that colored ring anywhere inside a gray-tape square, picks it up, and drops it on the AMD sticker. Everything runs locally on an **AMD Ryzen AI MAX+ 395** PC.

## What it does
- **Voice + web control:** say *red, green, blue* or *purple*, or press a button. Speech-to-text runs locally with faster-whisper.
- **Vision:** an overhead camera finds the requested ring with hue-based color detection that still works in sunlight. The clicked gray-tape square marks the work area.
- **Pick and place:** the arm grabs the ring, drops it on the AMD sticker, goes back to rest and waits for the next command.
- **Grab checks:** after closing, the gripper must stop on the ring (not on air), and the camera must see the ring gone from its spot. If either check fails, it finds the ring again and retries once.
- **NBA Champion Picker:** Heat = red, Celtics = green, 76ers = blue, Lakers = purple. The robot "predicts" the champion by picking up a random ring.

## How it works
| Piece | File |
|---|---|
| Main robot logic (find → pose → grab → check → drop) | `ringbot.py` |
| Ring color detection (HSV hue bands, works in sunlight) | `rings.py` |
| Voice server + color page + NBA page (port 8091) | `voice_server.py` |
| Live camera preview, tape-corner clicking (port 8090) | `cam_preview.py` |
| Record leader-arm demos and extract the grab poses | `demo_session.py` |
| Calibration, teaching and diagnostic helpers | `tape.py`, `teach_*.py`, `relearn.py`, `map_marker.py`, `grid_fit.py`, `align_follower.py`, `arm_move.py`, `compare_arms.py`, `motion.py` |
| Setup scripts for Ryzen AI / Strix Halo (ROCm) | `setup/` |

**Learning from demonstration:** we drove the leader arm by hand to record many pick-ups. `ringbot_poses.json` holds those demos plus a grid of taught positions. For a new ring position, the grab pose is a blend of the nearest successful demos and the grid result.

**Camera-shift correction:** `rings_calib.LOCKED.json` is the camera view the training was done in. If the camera gets bumped, click the 4 inner tape corners again on the preview page. Ring positions are then converted back into the training view, so nothing has to be retrained.

## Hardware
- SO-101 follower + leader arms (Feetech STS3215 servos)
- Overhead USB camera and wrist camera
- AMD Ryzen AI MAX+ 395 PC running Ubuntu, with PyTorch on ROCm

## Running it
```bash
conda activate lerobot
python cam_preview.py          # camera page on http://localhost:8090
python voice_server.py         # voice / color / NBA page on http://localhost:8091 (/nba)
python ringbot.py run green    # or: python ringbot.py random
```
Serial ports and camera device paths are set at the top of `ringbot.py`; change them for your machine. The calibration and pose files in this repo match our table setup, so on a different setup you'll need to re-teach.

## Challenges
- Sunlight washing out colors (we switched to hue-based detection and limited it to 4 colors that are easy to tell apart)
- The camera kept getting bumped (fixed with tape-corner re-calibration and conversion back to the training view)
- A grip check fooled by an empty gripper that physically can't close past a certain point
- Getting the pick-up low enough without the arm pushing forward
- Motor overload faults and flaky Wi-Fi during a live hackathon

## What's next
- Train an end-to-end policy (e.g. ACT) on the AMD GPU from our recorded demos
- Use the wrist camera to center the gripper on the ring before closing
- More objects and commands ("stack", "sort by color")
