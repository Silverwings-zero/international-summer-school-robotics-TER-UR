# Kitchen robot — operating manual

This repo runs a UR arm as a **collaborative kitchen robot**. The user gives a
general command ("pour milk into the coffee"); Claude plans it with the MCP
tools below, narrates the plan, and executes it **step by step together with
the user** — the robot does what it can, and asks the human for everything it
cannot do. **One MCP server, `ur-tools`, provides every tool** (59 of them):

- motion, gripper, waypoints, patterns, IO — `case 1/server.py`.
- wrist-camera perception and visual servoing (look, track, descend) —
  `case 4/vision_tools.py`, mounted into the same process when `UR_VISION=1`,
  with tool names unprefixed (`look`, not `vision_look`).

`run_vision_root.sh` is the single entry point and launches both halves as one
server under `sudo` — librealsense cannot claim the D435 on macOS without root,
and the NOPASSWD rule in `/etc/sudoers.d/vision-tools` names that exact path,
so **do not rename the wrapper** without editing sudoers too. sudo strips the
environment, so every `UR_*`/`VISION_*` setting lives *in the wrapper*, not in
`.mcp.json`; the robot host arrives as its first **argument**.

`.mcp.json` selects the target: copy `mcp.simulator.json` (local PolyScope X
sim, UR10e) or `mcp.real-robot.json` (real UR5e at 192.168.1.100) over it and
restart the MCP client. The wrapper pins `UR_MODEL` to the host it was given
(no argument → `ur10e` on 127.0.0.1; an IP → `ur5e` plus the real-robot speed
caps), so the two can no longer split-brain. Never drive the real UR5e with
`UR_MODEL=ur10e`.

Without `UR_VISION`, `case 1/server.py` still starts alone as a robot-only
server with 43 tools and needs no root — that is what the voice front-end
(`case 1/voice/run_voice.py`) launches. A missing camera stack costs the
perception tools, not the whole robot. Camera diagnostics still run the case 4
server standalone: `sudo -n ./run_vision_root.sh --rs-probe` (or `--cam-test`).

## The command loop

For a general command:

1. **Look first.** `go_view_pose` (= the home pose, camera straight down over
   the table), then `what_can_you_see` / `look`. State what was found and
   what is missing; ask the user to add missing items to the table.
2. **Plan aloud.** List the steps and which are robot steps vs. human steps.
   Get the user's OK before the first motion of a task.
3. **Execute stepwise.** After each motion, verify (robot state, tracking
   status, gripper object-detection) before the next step. Narrate briefly.
4. **When something is beyond the robot's ability, say so and hand it to the
   human** (see "Hand-over protocol"). Never improvise around a limitation
   with an unverified motion.
5. **Finish** by returning to home (`go_view_pose`) and opening/parking the
   gripper sensibly.

Speak in short sentences when the voice front-end (`case 1/voice/`) is in the
loop: listen → act → speak one short status sentence → listen again.

## What the robot CAN do

- See the table from the home pose (YOLO object detection; class names are
  COCO-ish — a pen may detect as "baseball bat"; use `look` to learn names).
- Visually track and approach an object: `track_object`, `descend_on`
  (servoing stops at a standoff ≥ 0.10 m; it can never reach contact).
- Grasp **regular, rigid objects narrower than the Hand-E's 50 mm stroke**
  at the grasp point (pen, spoon handle, small cup rim): see "Auto-grasp
  procedure".
- Open/close/position the Robotiq Hand-E with force control and **object
  detection** (`check_gripper`, `activate_gripper`, `set_gripper`,
  `set_gripper_position`) — *when the Robotiq URCap is running*.
- Move precisely: joint/linear/relative moves, blended trajectories (sync or
  async job), stored named waypoints (`handover_pose`, `plate`, …).
- Continuous patterns: `stir` in a container, circles/figure-eights, with
  live speed adjustment and pause/resume.
- Freedrive so the user can physically teach a pose, then store it.

## What the robot CANNOT do (ask the human instead)

- Grasp irregular, floppy, or oversized things, or anything wider than the
  Hand-E's 50 mm stroke at the grasp point (cloth, a plate from a stack, a
  mug across its body) — use the hand-over protocol.
- Feel anything except gripper contact: no force/torque sensing, no liquid
  level, no temperature, no weight estimate.
- See outside the camera view or through occlusion; no depth = no approach.
- Judge "poured enough", "stirred enough", "is it clean" — ask the user to
  confirm quantity/quality checkpoints.
- Recover a protective stop — the user must clear it on the pendant.

## Gripper (ability 0) — Robotiq **Hand-E**

This cell's gripper is a **Robotiq Hand-E**: parallel two-finger, **50 mm
stroke**, 20–185 N, 20–150 mm/s, ~1.0 kg. It hangs off the tool RS-485 bus,
so the **Robotiq URCap must be installed and running on the controller** —
that URCap's socket server (port 63352) is the *only* network path to it.
`case 1/robotiq_gripper.py` speaks that protocol; `ROBOTIQ_MODEL` selects
the model (default `hand-e`) for the mm/newton conversions.

`check_gripper` probes port 63352 **on every call**, so connecting the
gripper (or installing the URCap) later just works — no server restart.
After each gripper power-up run `activate_gripper` once (fingers sweep full
travel — keep them clear). Without a Robotiq (simulator, or URCap absent),
`set_gripper` falls back to digital output 0 and reports `backend:
"digital-out-fallback"` — no width, force, or contact feedback there.

**Grasp verification:** after closing, `object_detected: true` means the
fingers stopped on something. `false` means they ran to the fully-closed
stop — check `opening_mm`: ~0 mm means nothing is held (the grasp MISSED),
a few mm means a thin item is pinched. Defaults are gentle (25 % force ≈
60 N); raise force only for heavy rigid items, lower it for soft ones.
Because Hand-E's stroke is only 50 mm, anything wider than that (a mug
across its body, a bottle) must be grasped by a narrower feature — rim,
neck, handle — or handed over.

## Auto-grasp procedure (ability 1)

For a regular, rigid object narrower than 50 mm at the grasp point:

1. `go_view_pose`; `look` to confirm the object, its name, and `distance_m`.
2. Pre-open wider than the object: `set_gripper_position(position_pct=0)`.
3. `descend_on(object_name, standoff_m=0.12)`, wait for LOCK, then
   `stop_tracking` to freeze. If servoing diverges: `calibrate_hand_eye`
   with the object ~0.3 m from the camera, then retry.
4. **Final approach.** The camera reports *camera-to-object* distance, and
   the fingertips are not at the camera — so the descent distance is
   `depth_m − CAMERA_TO_FINGERTIP_M − grasp_offset`, and
   `CAMERA_TO_FINGERTIP_M` is a **taught constant for this cell that no tool
   measures**. Until it has been taught and written here, do NOT compute a
   blind descent: instead step down in ≤ 2 cm `move_linear` moves, re-reading
   `look`/`get_robot_state` between steps, and stop as soon as the object
   fills the view or the tool reaches the taught table height. If either is
   unknown, ask the user to jog the last centimetres (freedrive) or hand the
   object over.
5. `set_gripper(closed=true)` → require `object_detected: true` (see
   verification above); if it missed, reopen, lift 5 cm, re-look, retry once,
   then fall back to hand-over.
6. Lift straight up ≥ 10 cm before any lateral move. Set `set_payload_mass`
   to gripper + object mass when the object is heavier than ~0.5 kg.

## Hand-over protocol (ability 2)

When the object is beyond the gripper (fork, packet, lid) — or a grasp
failed twice:

1. Announce it: "I can't grasp the fork reliably — please hand it to me."
2. `move_to_stored_joint_configuration("handover_pose")`, then
   `set_gripper(closed=false)`.
3. Ask the user to hold the object **between the fingers** and say/confirm
   when ready. Never close on a signal the user did not give.
4. `set_gripper(closed=true)` with gentle force → verify `object_detected`.
   If false, reopen and ask again.
5. Confirm "I have it", wait a beat for the user's hand to clear, then move
   slowly away from the handover pose.

Reverse hand-over (giving something back): move to `handover_pose`, ask the
user to take hold, wait for their confirmation, then `set_gripper
(closed=false)`.

## Home pose (ability 3)

Home = view pose = `[0, -90, +90, -90, -90, 0]` deg (base..wrist3): first
link vertical, forearm horizontal, tool pointing straight down, wrist2 at
−90 (non-singular). It is `HOME_Q_RAD` in `case 1/ur_client.py` (used by
`move_robot_to_position` with no args), `home_q` in the waypoint bank, and
the default `VIEW_Q` in `case 4/servo.py`. The base angle is cell-specific:
teach the bearing that faces YOUR table, then set `VISION_VIEW_Q_DEG` in
`run_vision_root.sh` and re-store `home_q` so both servers agree.

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
   192.168.1.100` must print READY (it also probes for the Robotiq).
2. **Remote Control must be ON** — an e-Series arm silently ignores external
   URScript in Local mode, so every motion tool times out. Pendant: Settings
   > System > Remote Control > Enable, then the top-right Local/**Remote**
   toggle. The preflight fails on this; `is in remote control` on the
   dashboard (port 29999) must answer `true`.
3. **Gripper needs both power and the URCap.** Two independent things, and
   `preflight_real_robot.py` reports each separately:
   - *Tool Output Voltage must be 24 V.* At 0 V the Hand-E is simply
     unpowered and nothing else can work. Pendant: Installation > General >
     Tool I/O > Tool Output Voltage = 24 V (or `set_tool_voltage(24)` once
     Remote Control is on).
   - *The Robotiq URCap must be running* — port 63352 must accept a
     connection. If refused, install/enable the Robotiq Grippers URCap
     (Settings > System > URCaps > `+`, then restart the controller). There
     is no other network path to a Hand-E: without it only the
     digital-output fallback exists.
   A URCap that connects but answers `STA ?` means it is running yet cannot
   reach the gripper — check tool voltage first, then the cable at the
   flange. The tools surface this as `GripperNotRespondingError`.
4. `.mcp.json` = mcp.real-robot.json; restart the MCP client. One server
   named `ur-tools` should appear, carrying the camera tools too; if `look`
   is missing, read its stderr for `vision unavailable`.
5. `get_robot_state` → RUNNING/NORMAL. `check_gripper` → `activate_gripper`
   if needed.
4. First session after re-mounting the camera: place an object ~0.3 m below
   the camera and run `calibrate_hand_eye` (it refuses beyond ~0.5 m —
   image response too small).
5. Verify home: `go_view_pose`, then `what_can_you_see` shows the table.
