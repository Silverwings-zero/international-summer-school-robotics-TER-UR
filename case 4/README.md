# Case 4, Eye-in-Hand Visual Servoing

A RealSense D435 on the UR wrist, YOLO26 finding the objects on the table,
and a servo loop that steers the robot so the object you click stays in the
middle of the image -- then holds a fixed standoff distance from it using the
camera's depth. Click a cup, press SPACE, and the robot centres on the cup
and follows it when you slide it across the table.

*Can a modern detector plus a 30-second empirical hand-eye calibration turn
"that object there" into a robot motion target, with no CAD, no markers, and
no manual camera-mount math?*

## What's in the folder

| File | What |
|------|------|
| `servo.py` | The app: live window, target selection, servo loop, auto-calibration |
| `camera.py` | RealSense D435 seam (color + aligned depth + intrinsics), webcam fallback |
| `detector.py` | YOLO26 seam: frames in, tracked detections out |
| `vision_tools.py` | MCP server: "track the cup" as LLM-callable tools, with a sim mode |
| `test_vision_tools.py` | In-process tool test against the live simulator |
| `requirements.txt` | ultralytics (YOLO26), OpenCV, RealSense bindings |
| `hand_eye.json` | Appears after you press `c`: the measured hand-eye Jacobian |

The robot side reuses `../case 1/ur_client.py` unchanged -- stdlib sockets,
URScript `movel` for motion, RTDE for state. Nothing new to install for the
robot; everything new is vision.

## Setup

```bash
# from the repo root, in the repo venv
.venv/bin/pip install -r "case 4/requirements.txt"
```

First run of `servo.py` downloads the YOLO26 nano weights (~6 MB).

**Camera notes (macOS, verified on this machine):** the
`pyrealsense2-macosx` wheel covers Apple Silicon + Python 3.9-3.14 and the
D435 enumerates fine, but *streaming* needs root -- without it librealsense
raises `failed to set power state` (a documented macOS 12+ limitation, plus
a harmless segfault on exit). So on a Mac, run the RealSense path as:

```bash
sudo ../.venv/bin/python servo.py
```

Plug the camera into a direct USB-C port (not a low-power hub). The webcam
path (`--camera webcam`, also the D435's RGB sensor over plain UVC) instead
needs macOS *camera permission*: the first run pops a prompt for your
terminal app -- grant it, then rerun (centering only -- no depth, so no
approach axis). On Linux neither issue exists; `pip install pyrealsense2`
and go.

**Robot:** point `UR_HOST` (or `--host`) at the robot. The PolyScope X
simulator works for the motion side, but it has no camera -- pair it with a
webcam looking at your desk, or use `--dry-run` (no robot at all) to try the
vision stack.

## Run

```bash
cd "case 4"
../.venv/bin/python servo.py                    # RealSense + robot at $UR_HOST
../.venv/bin/python servo.py --dry-run          # vision only, no robot
../.venv/bin/python servo.py --camera webcam --host 192.168.1.10
../.venv/bin/python servo.py --no-approach      # centre only, never approach
```

In the window:

| Key | Action |
|-----|--------|
| click | select the object under the cursor as the target |
| `n` | cycle target through current detections |
| `SPACE` | servo on/off |
| `c` | hand-eye auto-calibration (servo off, target selected, scene still) |
| `v` | movej to the `VIEW_Q` observation pose -- **edit it for your cell first** |
| `q` / ESC | quit |

Suggested first session on the real robot: hand-guide (freedrive) the wrist
to a pose where the camera sees the table, keep a hand on the e-stop, select
a target, press `c` once, then SPACE. Slide the object around slowly and
watch the robot re-centre.

## Talking to it: the MCP tools

`vision_tools.py` exposes the same engine as tools an LLM can call, so "track
the cup" and "descend on the phone" become tool calls. These no longer get an
MCP entry of their own: `run_vision_root.sh` starts case 1's `server.py` with
`UR_VISION=1`, which mounts this module's `mcp` into it, so the repo
`.mcp.json` lists one `ur-tools` server carrying motion and vision together
(names stay unprefixed -- `look`, not `vision_look`). Restart your MCP client
after editing. To run this server on its own, launch `vision_tools.py`
directly:

| Tool | What it does |
|------|--------------|
| `look()` | every visible object: name, confidence, off-centre offset, distance |
| `what_can_you_see()` | "what can you see?" -- the YOLO object list, grouped by name with counts and a plain-English summary |
| `track_object("cup")` | centre it and hold the standoff; keeps following afterwards |
| `descend_on("phone", standoff_m=0.15)` | same law, tight standoff: close in on it |
| `stop_tracking()` | freeze; nothing moves until the next command; closes the live view |
| `tracking_status()` | LOCKED?, current error, depth -- poll during long moves |
| `go_view_pose()` | joint-move to the safe viewing pose (the recovery from "refused: singular") |
| `calibrate_hand_eye()` | the 30 s probing calibration, saved across restarts |
| `show_camera_view()` / `hide_camera_view()` | open/close the live view without tracking |
| `place_sim_object("cup", 0.5, 0.1, 0.05)` | sim mode only: put a virtual object in view |

**Live view.** When tracking starts, a window pops up on the machine running
the server: the camera image (or the simulated camera: grid + virtual
objects) with YOLO boxes, the target in green, the centring crosshair, and
the servo status. `stop_tracking` closes it; `q` in the window closes the
VIEW only (the robot keeps its orders). It runs as a small subprocess
(`viewer.py`) because macOS insists GUI windows own a main thread, and the
MCP server's main thread belongs to stdio -- a viewer crash can never touch
the robot. Set `VISION_VIEW=off` to disable auto-opening, `always` to open
it at startup.

Every command re-arms the safety box at the current pose, so each
`track_object`/`descend_on` gets a fresh (but bounded) +-30 cm envelope.
Servoing refuses to start -- and stops mid-flight -- near the wrist AND
elbow singularities, where PolyScope X linear moves protective-stop with
C204A1; `go_view_pose()` is the way out.

`VISION_MODE=sim` (the default in `.mcp.json`) needs no camera at all: virtual
objects are watched by a virtual wrist camera glued to the LIVE simulated TCP
(read over one persistent RTDE stream), so the whole loop -- servo law,
URScript moves, URSim's controller -- is real except the pixels. Set
`VISION_MODE=real` for the D435 + YOLO26 (note: on macOS the RealSense needs
the server to run under sudo, which MCP clients won't do -- use the webcam, or
drive the real camera through `servo.py` instead).

Simulator health note: the PolyScope X sim is fragile under rapid program
uploads and RTDE connection churn -- overdo it and it degrades its real-time
scale and eventually protective-stops (clear it in the web UI at
http://localhost). The vision server therefore keeps exactly one RTDE
subscription and paces servo steps (`step_period_s`).

## How it works

Every frame: YOLO26 detects and tracks the objects; the target's box centre
becomes an **observation** `(x, y, Z)` -- the pixel error from the image
centre in normalized camera coordinates, plus the median depth on the object.
Every servo step: the worker asks for the observation to move a fraction
toward `(0, 0, standoff)` and solves a small linear system for the tool-frame
translation that does it, executed as a clamped `movel`. Look, step, look.

The linear system is the hand-eye Jacobian `J = d(observation)/d(tool step)`.
You can trust the analytic default (camera along tool +Z; edit `R_TOOL_CAM`
in `servo.py` if your bracket differs), or press `c`: the robot probes each
tool axis by 15 mm and *measures* J from how the image and depth actually
responded. That makes the loop indifferent to how the camera is bolted on --
any rigid mount, any rotation, calibrated in half a minute, saved to
`hand_eye.json`, reloaded on the next run.

Because the image rows of J scale like 1/Z, the calibration distance is
stored and J is rescaled to the live depth each step -- calibrate at 40 cm,
servo at 70 cm, the gains stay right.

## Safety layers

- every step is capped at `max_step_m` (3 cm) and executed at 0.1 m/s;
- the TCP is confined to a box around the pose where servoing was enabled
  (+-30 cm horizontally, -30/+20 cm vertically);
- the approach axis regulates *around* the standoff -- too close means it
  backs off, and losing depth freezes the range axis entirely;
- observations older than 0.8 s (target lost, detector hiccup) freeze the
  robot until the target is seen again;
- a protective stop or controller refusal surfaces as an on-screen error and
  disables servoing;
- servoing refuses to start with wrist2 near 0 deg, where UR linear moves
  hit the wrist singularity (press `v` / jog away first).

Orientation is never commanded -- the tool keeps the attitude you started
with; only translation servos. That is deliberate: it keeps the failure modes
translational and the demo predictable.

## Tuning

All knobs sit in `ServoConfig` at the top of `servo.py`:

| Knob | Default | Effect |
|------|---------|--------|
| `gain_image` / `gain_depth` | 0.5 | fraction of the error removed per step; higher = snappier, oscillates sooner |
| `standoff_m` | 0.35 | distance held to the object (`--standoff`) |
| `deadband_norm` | 0.02 | "centred" tolerance (~12 px at 640x480) |
| `max_step_m` | 0.03 | per-step travel cap |
| `box_*` | 0.2-0.3 | the safety box around the enable pose |

## Ideas to take it further

- **Grasp what you centred:** once LOCKED, descend the last `standoff` along
  the approach axis and close the gripper (`set_tool_digital_out`).
- **Expose it as an MCP tool:** `servo_to("cup")` in case 1's `server.py`,
  and the case 3 agent can fetch things by name.
- **Track a person's hand** (`--model yolo26n-pose.pt`) and hand objects over.
- **Smooth pursuit:** replace step-wise `movel` with URScript `speedl`
  streaming for continuous motion (watch the PolyScope X quirks first).
