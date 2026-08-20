# Case 1: MCP Server for Robot Tools

An MCP server that exposes UR robot capabilities as tools an LLM can call, so an
agent can operate the robot in natural language. It is the robot's "hands": any
MCP client (Claude Code, Cursor, or the Case 3 agent) launches this server and
calls its tools.

## The task

Turn robot capabilities into well-described, LLM-callable tools. One tool ships
fully worked, `move_robot_to_position`, and it moves the robot end to end. Your
job is to add more tools of your own (a state reader, a linear move, a gripper),
each following the same four-step shape, so a client can do more than move to a
joint pose.

The robot seam is **not** your job here: `ur_client.py` already speaks to the
robot over sockets. Your work is the tool layer on top of it, especially the
docstrings, because the model reads them to decide when and how to call a tool.

## What's provided vs what you build

| File | Role | Provided? |
|------|------|-----------|
| `server.py` | the MCP server; `move_robot_to_position` (worked) + `example` (template) | worked tool + template, add your own |
| `ur_client.py` | pure standard-library seam over the robot (motion + state) | done, do not touch |
| `test_server.py` | in-process smoke test for both tools | provided |
| `requirements.txt` | one dependency, the MCP framework | provided |
| `motion_patterns.py` | engine for continuous, steerable motions (stirring) | ours |
| `test_motion_patterns.py` | in-process test of a full stirring session | ours |

## Setup

1. **The PSX simulator is running.** Start it from `../simulation environment`:
   ```bash
   cd "../simulation environment" && docker compose up -d
   ```
   First boot takes about 40 seconds.
2. **The robot is powered on.** Open http://localhost, then power the robot on
   and release the brakes in the control panel at the bottom. It must read
   RUNNING before it will move.
3. **Python deps installed** in this folder:
   ```bash
   pip install -r requirements.txt
   ```

The server connects to `127.0.0.1` (the simulator). Set `UR_HOST` to target a
different host or a real robot.

## Run

Cheapest checks first, then connect a client.

**1. Socket seam (no MCP).** Confirms the robot is reachable and state reads:
```bash
python -c "from ur_client import URClient; r=URClient(); r.connect(); print(r.get_state())"
```

**2. MCP tools in-process.** Exercises both tools, input validation, and real
motion, without an LLM or a subprocess:
```bash
python test_server.py
```
If the robot is off, both fail with a clear "not powered on" message. That error
is the guard working. The steerable motions have their own test:
```bash
python test_motion_patterns.py
```

**3. Connect a client.** The server speaks MCP over stdio: the client launches
`server.py` and talks to it over stdin/stdout, so you do not start it and connect
to a port. A sanity run (`python3 server.py`) only checks it imports and reaches
the robot; there is nothing to connect to there.

Set up a free LLM client with [`../llm-client`](../llm-client), then add this
server to it as an MCP named `ur-tools`. Two free paths:

- **Option A, self-hosted (Bionic, local).** Follow
  [`../llm-client/self-hosted.md`](../llm-client/self-hosted.md). In its "Add an
  MCP server" step, use Name `ur-tools`, Command the absolute path to your
  `python3`, and one Argument: the absolute path to `case 1/server.py`.
- **Option B, cloud-hosted (OpenClaw).** Follow
  [`../llm-client/cloud-hosted.md`](../llm-client/cloud-hosted.md), then register
  the server:
  ```bash
  openclaw mcp add ur-tools \
    --command /PATH/TO/python3 \
    --arg "/PATH/TO/case 1/server.py"
  openclaw mcp probe ur-tools     # expect 2 tools
  ```

With either, open the chat and ask in plain language: `move the robot home.` The
model reads the tool docstrings, calls `move_robot_to_position`, and the robot
moves.

### Using Claude Code (paid, optional)

If you already have Claude Code, register the server directly (absolute path,
quoted because of the space):
```bash
claude mcp add ur-tools -- python3 "/PATH/TO/essre2026-cases/case 1/server.py"
claude mcp list        # check it is connected
```
Then ask `Move the robot home.`; remove it with `claude mcp remove ur-tools`.
Other clients (Cursor, Claude Desktop) use the same idea: an MCP entry whose
command is `python3` and whose argument is the absolute path to `server.py`.

## Tiers

- **Bronze, run it:** bring up the sim, connect a client, and move the robot home
  with the provided `move_robot_to_position` tool.
- **Silver, read state:** add a `get_robot_state` tool (copy `example`, follow the
  four-step pattern) that returns joints, TCP pose, and mode, so an agent can
  observe before it acts.
- **Gold, richer motion:** add a more capable motion tool, a relative move, a
  linear/TCP move, or a multi-waypoint path, with full input validation and limit
  checks.
- **Diamond, real skills:** add a gripper / IO tool or a compound skill (a
  pick-and-place primitive), and surface `safety_status` so a client can detect a
  protective stop.

## Continuous motion patterns

Every other tool here is one-shot: it blocks until the robot arrives, then
returns. That cannot express an open-ended motion the operator steers while it
runs, because while a tool call is pending the LLM is not running, so the next
thing the operator says cannot reach it. Stirring a pan is exactly that kind of
motion, so it works differently.

Position the tool in the centre of the container first (by hand in freedrive,
or with a move tool), then:

| Say | Tool | What happens |
|-----|------|--------------|
| "stir the pan, it's 10 cm across" | `stir` | anchors on the current pose, starts circling, **returns at once** |
| "faster" / "slower" | `adjust_pattern_speed` | `factor=1.3` / `0.7`, keeping the tool's place on the path |
| "stop", "wait", "hold on" | `pause_motion` | smooth stop; anchor, speed and phase all kept |
| "keep going" | `resume_motion` | continues from wherever the tool now sits |
| "that's enough", "we're done" | `finish_motion` | stops for good and releases the anchor |
| "is it still going?" | `get_motion_status` | laps, speed, live pose, safety status |

`start_motion_pattern` is the general form for motions that are not about a
container, with `figure_eight`, `spiral` and `linear_sweep` alongside `circle`.
A new pattern is one function plus a registry entry in `motion_patterns.py`.

Three things worth knowing before changing any of it:

- **The motion runs on the controller, not here.** Each tool uploads a URScript
  loop; a new upload preempts the running one, which is how a speed change
  becomes seamless. It also means the loop outlives this process, so the server
  stops it on exit, and `pause_motion` / `finish_motion` stop a leftover loop
  even when no job is registered.
- **Circles use `movec`, everything else a polygon of blended `movel`.** On the
  polygon the blend corners cap the speed near `sqrt(a · r_blend)`: commanding
  0.128 m/s instead of 0.080 m/s measured 0.0715 vs 0.0718 m/s, i.e. no change
  at all. `movec` follows a true arc and honours the commanded speed to ~95%.
- **The controller enforces no centripetal limit.** 0.40 m/s on a 30 mm circle
  ran fine while demanding 5.3 m/s². Keeping the contents in the pan is the
  server's job: `max_speed_for_radius` caps it, which is why a 10 cm pan tops
  out near 0.23 m/s.

```bash
python test_motion_patterns.py   # start, faster, pause, resume, finish
```

## The tool pattern

Every tool has the same four steps. See `move_robot_to_position`:
1. Validate inputs. Raise `ValueError` with a plain reason (the LLM reads it).
2. Convert request units to robot units (degrees to radians).
3. Check feasibility against the joint limits.
4. Execute only after the checks pass, then report the resulting state.

Return JSON-serializable dicts with units in the key names. `example` is a
minimal template; copy it to start each new tool.

## Robot interface

`ur_client.py` is the only file that touches the robot. It speaks two UR network
interfaces over plain TCP sockets:
- Primary interface (port 30001): motion. Uploads a small URScript `movej`.
- RTDE (port 30004): state. Reads joint angles and TCP pose.

Keep tools calling `URClient` methods so the server stays portable between the
simulator and a real robot. Only `UR_HOST` changes.

## Real robot (UR5e)

The same MCP servers drive a real arm -- only configuration changes. The
safety layer is model-aware: `UR_MODEL` selects the DH table, reach, and
workspace box the checks run against (`ur10e` = the simulator default,
`ur5e`, `ur5`). Never drive a UR5e with the ur10e default: every geometric
safety check would assume an arm twice the size.

1. **Preflight.** With the robot powered on and this machine on its network:

       python3 preflight_real_robot.py <robot-ip>

   It checks the controller ports, PolyScope version, safety state, Remote
   Control mode (e-Series ignores external URScript without it: Settings >
   System > Remote Control, then the Local/Remote toggle), and does a full
   RTDE state read. It never moves the robot.

2. **Activate.** Fill the robot IP into `../mcp.real-robot.json` (and
   `run_server.sh` for the camera servo), copy it over
   `../.mcp.json`, restart Claude Code. The template also caps speeds well
   below the simulator defaults (`UR_MAX_JOINT_SPEED` etc. -- env vars, no
   code changes) for the first sessions around people.

3. **First session.** Pendant speed slider at 25% or less, a hand on the
   e-stop, nobody inside the reach envelope. Confirm payload + TCP on the
   pendant match the mounted tool. Start with `get_robot_state`, then a
   small `move_joints_relative`, before anything Cartesian.

Flip back to the simulator by restoring `.mcp.json` (git checkout works).

### Robotiq gripper (Hand-E)

This cell's gripper is a **Robotiq Hand-E** (50 mm stroke, 20-185 N). It is
driven through the Robotiq URCap's command server (port 63352 on the
controller; `robotiq_gripper.py`, pure stdlib -- `ROBOTIQ_MODEL` selects
`hand-e`/`2f-85`/`2f-140` for the mm and newton conversions). The tools:

| tool | does |
| --- | --- |
| `check_gripper` | probe + full status; re-probes every call, so plugging the gripper in later just works |
| `activate_gripper` | one-time self-calibration after power-up (fingers sweep full travel -- keep them clear) |
| `set_gripper` | open/close; with a Robotiq it reports `object_detected` (false after a close = the grasp missed) |
| `set_gripper_position` | width + speed + force control, gentle defaults, blocks until arrival/contact |

Without a Robotiq (the simulator) `set_gripper` falls back to digital
output 0 and says so (`backend: "digital-out-fallback"`).

Two things must be true on the real cell, and `preflight_real_robot.py`
checks both: **Tool Output Voltage = 24 V** (Installation > General > Tool
I/O -- at 0 V the gripper is unpowered and nothing works) and the **Robotiq
URCap running** (port 63352 accepts a connection). A URCap that connects but
answers `STA ?` is running yet cannot reach the gripper -- that is a
`GripperNotRespondingError`, and tool power is the usual cause.

`test_robotiq_gripper.py` exercises the driver against a fake URCap
(activation, contact detection, reply framing, concurrent commands, the
`?` reply) and needs no robot and no gripper:

    python3 test_robotiq_gripper.py

### Home pose

`HOME_Q_RAD` is the kitchen home `[0, -90, +90, -90, -90, 0]` deg: upper arm
vertical, forearm horizontal, tool pointing straight down -- the wrist
camera overlooks the table, and wrist2 at -90 keeps linear moves possible
straight from home. `move_robot_to_position` with no arguments goes there;
the camera's `go_view_pose` uses the same pose (override per cell via
`VISION_VIEW_Q_DEG` in `run_server.sh`).
