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
is the guard working.

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
