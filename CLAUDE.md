# Kitchen robot — operating manual

This repo runs a UR arm as a **collaborative kitchen robot**. The user gives a
general command ("pour milk into the coffee"); Claude plans it with the MCP
tools below, narrates the plan, and executes it **step by step together with
the user** — the robot does what it can, and asks the human for everything it
cannot do. **One MCP server, `ur-tools`, provides every tool** (35 of them):

- motion, gripper, waypoints, patterns, IO — `case 1/server.py`.
- wrist-camera perception and visual servoing (look, track, descend) —
  `case 1/camera/vision_tools.py`, mounted into the same process when
  `UR_VISION=1` (`mcp.mount(vision_tools.mcp)`, no namespace), so tool names
  stay unprefixed (`look`, not `vision_look`).

`case 1/run_server.sh` is the single entry point and launches both halves as
one server under `sudo` — librealsense cannot claim the D435 on macOS without
root, and the NOPASSWD rule in `/etc/sudoers.d/vision-tools` names that exact
path, so **do not move the wrapper** without editing sudoers too (the space in
`case 1` must be backslash-escaped there). sudo strips the environment, so
every `UR_*`/`VISION_*` setting lives *in the wrapper*, not in `.mcp.json`;
the robot host arrives as its first **argument**.

`.mcp.json` selects the target: copy `mcp.simulator.json` (local PolyScope X
sim, UR10e) or `mcp.real-robot.json` (real UR5e at 192.168.1.100) over it and
restart the MCP client. The wrapper pins `UR_MODEL` to the host it was given
(no argument → `ur10e` on 127.0.0.1; an IP → `ur5e` plus the real-robot speed
caps), so the two can no longer split-brain. Never drive the real UR5e with
`UR_MODEL=ur10e`.

Without `UR_VISION`, `case 1/server.py` still starts alone as a robot-only
server with 27 tools and needs no root — that is what the voice front-end
(`case 1/voice/run_voice.py --no-vision`) launches. A missing camera stack
costs the perception tools, not the whole robot: the mount is wrapped in a
`try`, and an import failure only prints `camera unavailable` to stderr.
Camera diagnostics run the vision server standalone:
`sudo -n "case 1/run_server.sh" --rs-probe` (or `--cam-test`).

## The command loop

For a general command:

1. **Look first.** `move_robot_to_position` with no arguments (= the home
   pose, camera straight down over the table), then `what_can_you_see` / `look`. State what was found and
   what is missing; ask the user to add missing items to the table.
2. **Plan aloud.** List the steps and which are robot steps vs. human steps.
   Get the user's OK before the first motion of a task.
3. **Execute stepwise.** After each motion, verify (robot state, tracking
   status, gripper object-detection) before the next step. Narrate briefly.
4. **When something is beyond the robot's ability, say so and hand it to the
   human** (see "Hand-over protocol"). Never improvise around a limitation
   with an unverified motion.
5. **Finish** by returning to home (`move_robot_to_position`, no arguments)
   and opening/parking the gripper sensibly.

Speak in short sentences when the voice front-end (`case 1/voice/`) is in the
loop: listen → act → speak one short status sentence → listen again.

## What the robot CAN do

- See the table from the home pose (YOLO object detection; class names are
  COCO-ish — a pen may detect as "baseball bat"; use `look` to learn names).
- Visually track and approach an object: `track_object` (servoing stops
  at a standoff ≥ 0.10 m; it can never reach contact), then close the last
  measured centimetres with `grasp_tracked_object`.
- Grasp **regular, rigid objects that fit between the open fingers** at the
  grasp point (pen, spoon handle, small cup rim): see "Auto-grasp
  procedure".
- Open and close the gripper, fast or slow, via its two tool digital
  outputs (`set_tool_digital_out` — the only gripper tool there is).
- Move precisely: joint/linear/relative moves, blended trajectories (sync or
  async job), stored named waypoints (`handover_pose`, `plate`, …).
- Continuous patterns: `stir` in a container, circles/figure-eights, with
  live speed adjustment and pause/resume.
- Freedrive so the user can physically teach a pose, then store it.

## What the robot CANNOT do (ask the human instead)

- Grasp irregular, floppy, or oversized things, or anything wider than the
  open fingers at the grasp point (cloth, a plate from a stack, a mug across
  its body) — use the hand-over protocol.
- Feel ANYTHING at all: no force/torque sensing, no gripper feedback (no
  width, no force, no object detection), no liquid level, no temperature,
  no weight estimate. A grasp can never be confirmed by the robot — look at
  it, or ask the user.
- See outside the camera view or through occlusion; no depth = no approach.
- Judge "poured enough", "stirred enough", "is it clean" — ask the user to
  confirm quantity/quality checkpoints.
- Recover a protective stop — the user must clear it on the pendant.

## Gripper (ability 0) — two tool digital outputs

The gripper is wired to the **wrist tool connector** and driven entirely by
its two digital outputs. **`set_tool_digital_out` is the only tool to call
for the gripper** — there is no gripper protocol, no width or force control,
and no feedback of any kind:

| pin `n` | line | `b=false` | `b=true` |
| --- | --- | --- | --- |
| 0 | jaws | **close** | **open** |
| 1 | speed | fast | slow |

Set the speed pin *before* the jaw pin when speed matters — the jaws move at
whatever speed is selected at that moment. Use slow (`n=1, b=true`) near
hands and for anything fragile; that is the default choice in this cell.

The tool connector must supply **24 V** (pendant: Installation > General >
Tool I/O -- the pendant is the only way, there is no tool-voltage MCP tool);
at 0 V the gripper is unpowered and
the outputs do nothing. `preflight_real_robot.py` reports both the voltage
and the current state of the two lines.

**Grasp verification: there is none.** Nothing is fed back, so the robot
cannot tell a holding grasp from a missed one. `get_robot_state` reports
`gripper_open` and `gripper_speed`, but those are only the COMMANDED lines.
After every close, verify by *looking* (`look` / `what_can_you_see`) or by
asking the user to confirm the item is held — and do that before lifting or
moving away. Anything that does not fit between the open fingers must be
grasped by a narrower feature — rim, neck, handle — or handed over.

## Auto-grasp procedure (ability 1)

For a regular, rigid object that fits between the open fingers. The pick is a
**fixed routine**, not a search: servoing centres the *camera* on the object,
and `grasp_tracked_object` closes the known camera-to-fingertip gap measured
for this cell (3 cm down, 1.5 cm sideways).

1. `move_robot_to_position` (no arguments); `look` to confirm the object,
   its name, and `distance_m`.
2. Select slow (`set_tool_digital_out(n=1, b=true)`) and open the gripper
   (`set_tool_digital_out(n=0, b=true)`). **Open it before tracking** —
   `grasp_tracked_object` never opens the jaws, because opening them over the
   table would drop whatever is already held.
3. `track_object(object_name, standoff_m=0.15)` and wait for **LOCKED**. If
   servoing diverges or refuses near a singularity, `stop_tracking`, go home
   with `move_robot_to_position`, and try again.
4. `grasp_tracked_object()` **immediately** — it needs the lock to still be
   live, and it only corrects a fixed offset from where the lock left the arm.
   It freezes tracking, corrects 3 cm down the approach axis and 1.5 cm
   sideways at 30 mm/s, closes the jaws, waits out their travel, then lifts
   10 cm (`lift_m` changes that; `lift_m=0` grips without lifting).
   `dry_run=true` reports the motion without moving — worth one call whenever
   the camera or tool has been re-mounted.
5. **Verify by looking.** The gripper reports nothing at all, so `status:
   "gripped"` means the commands were sent, NOT that the object is held.
   `look` again, or ask the user. If it missed: reopen, `move_robot_to_position`,
   re-look, retry once, then fall back to hand-over.
6. Set `set_payload_mass` to gripper + object mass when the object is heavier
   than ~0.5 kg.

The offset is one cell constant at the top of `case 1/camera/vision_tools.py`:
`GRASP_CORRECTION_TOOL_M = (0.015, 0.0, 0.030)`, in the TOOL frame — sideways
in +X, down the approach axis in +Z. At the home orientation that comes out as
1.5 cm along base +Y and 3 cm straight down, i.e. 1.5 cm to the LEFT of someone
standing in front of the robot. Re-measure it if the camera or the gripper is
re-mounted; nothing measures it automatically.

## Hand-over protocol (ability 2)

When the object is beyond the gripper (fork, packet, lid) — or a grasp
failed twice:

1. Announce it: "I can't grasp the fork reliably — please hand it to me."
2. `load_stored_waypoint("handover_pose")` — it only reads the pose back, so
   feed its `value_deg` to `move_robot_to_position`. Then open the gripper:
   `set_tool_digital_out(n=1, b=true)` (slow) and
   `set_tool_digital_out(n=0, b=true)`.
3. Ask the user to hold the object **between the fingers** and say/confirm
   when ready. Never close on a signal the user did not give.
4. Close slowly: `set_tool_digital_out(n=0, b=false)`. There is no contact
   feedback — ask the user to confirm you have it. If not, reopen and ask
   again.
5. Confirm "I have it", wait a beat for the user's hand to clear, then move
   slowly away from the handover pose.

Reverse hand-over (giving something back): move to `handover_pose`, ask the
user to take hold, wait for their confirmation, then open the gripper with
`set_tool_digital_out(n=0, b=true)`.

## Home pose (ability 3)

Home = view pose = `[0, -90, +90, -90, -90, 0]` deg (base..wrist3): first
link vertical, forearm horizontal, tool pointing straight down, wrist2 at
−90 (non-singular). It is `HOME_Q_RAD` in `case 1/ur_client.py` (used by
`move_robot_to_position` with no args), `home_q` in the waypoint bank, and
the default `VIEW_Q` in `case 1/camera/servo.py`. `move_robot_to_position`
with no arguments is the ONLY tool that goes there — the camera half no
longer carries its own `go_view_pose`. The base angle is cell-specific: teach
the bearing that faces YOUR table, then edit `HOME_Q_RAD`, re-store `home_q`,
and set `VISION_VIEW_Q_DEG` in `case 1/run_server.sh` so the standalone
camera viewer's `v` key agrees with it.

## Safety, always

- People share this workspace. Default speeds are capped by the real-robot
  config; keep motions near hands slow (`speed` ≤ 0.1 m/s linear).
- Confirm with the user before the first motion of a task and before any
  motion while their hands are near the robot.
- Never bypass a safety-layer rejection by nudging numbers until it passes —
  report it and rethink.
- If tracking/servoing behaves oddly, `stop_tracking` first, questions later.
- Robot unreachable / protective stop → stop the task and tell the user
  exactly what to check (power, e-stop, pendant, `preflight_real_robot.py`).

## Real-robot bring-up checklist

1. Power the robot, release brakes; `python "case 1/preflight_real_robot.py"
   192.168.1.100` must print READY (it also reports the tool IO).
2. **Remote Control must be ON** — an e-Series arm silently ignores external
   URScript in Local mode, so every motion tool times out. Pendant: Settings
   > System > Remote Control > Enable, then the top-right Local/**Remote**
   toggle. The preflight fails on this; `is in remote control` on the
   dashboard (port 29999) must answer `true`.
3. **Gripper needs tool power.** *Tool Output Voltage must be 24 V* — at
   0 V the gripper is unpowered and the tool outputs do nothing. Pendant:
   Installation > General > Tool I/O > Tool Output Voltage = 24 V. The
   pendant is the only path -- no MCP tool sets tool voltage.
   `preflight_real_robot.py` prints the voltage and the state of both
   gripper lines.
4. `.mcp.json` = mcp.real-robot.json; restart the MCP client. One server
   named `ur-tools` should appear, carrying the camera tools too; if `look`
   is missing, read its stderr for `camera unavailable`.
5. `get_robot_state` → RUNNING/NORMAL, and `gripper_open` / `gripper_speed`
   show the two tool lines. Exercise the gripper once with
   `set_tool_digital_out` (slow first, then open/close) with the fingers
   clear.
6. First session after re-mounting the camera or the gripper: re-measure the
   grasp offsets (see "Auto-grasp procedure") and check them with
   `grasp_tracked_object(dry_run=true)`. The hand-eye mount calibration in
   `case 1/camera/hand_eye.json` is loaded at startup; re-measuring it now
   needs the standalone viewer (`python "case 1/camera/servo.py"`, `c` key).
7. Verify home: `move_robot_to_position` (no arguments), then
   `what_can_you_see` shows the table.
