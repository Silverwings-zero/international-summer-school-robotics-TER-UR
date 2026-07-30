# Simulation Environment: PolyScope X (URSim)

The robot for all three ESSRE cases is the **PolyScope X simulator** (URSim). It
runs the same PolyScope X software UR is rolling its whole fleet onto, so you
develop against the real thing rather than a legacy sim. One container, native on
Apple silicon and Intel.

## Bring it up
```bash
docker compose up -d
```
First boot takes ~40 s while the controller starts. Then open **http://localhost**
(hard-refresh if you catch the splash screen). You get the full PolyScope X UI
with a 3D view of a **UR10**.

![PolyScope X full UI: 3D view, Move/Joints panel, live joint angles](images/polyscopex-full.png)

Tear down:
```bash
docker compose down       # stop; keeps robot state
docker compose down -v    # stop and wipe all robot state (fresh next boot)
```

## Power the robot on (once per boot)
A simulated robot boots **powered off**, exactly like a real one. It will not
move until you enable it:

1. Open http://localhost.
2. In the robot-control panel (bottom of the screen), **power on** and
   **release the brakes** until the robot indicator is green (RUNNING).

That's the only manual step. The Case 1 tools return a clear error if you forget,
so you'll know. (If a settings screen ever asks for a safety password, it is
`easybot`; you do not need it for normal use.)

Powered on, the UI looks like this (note **Robot State: Active** at the bottom
left, and the live joint angles on the right):

![PolyScope X, robot powered on and Active](images/polyscopex-active.png)

## What's exposed
The controller's network interfaces are published to your machine so the case
code can read state and command motion over plain TCP sockets:

| Port  | Interface            | Used for                                   |
|-------|----------------------|--------------------------------------------|
| 80    | PolyScope X web UI   | drive / enable the robot, watch it move    |
| 30001 | Primary interface    | motion, upload a URScript `movej`          |
| 30002 | Secondary interface  | (available)                                |
| 30004 | RTDE                 | state, read joint angles / TCP pose        |
| 29999 | Dashboard            | (PolyScope X does not use the classic protocol) |

Case 1's `ur_client.py` talks to 30001 (motion) and 30004 (state) with the
Python standard library only, nothing to compile.

## Robot type
`ROBOT_TYPE` defaults to `UR10` (see `docker-compose.yml`). Override at bring-up
for a different arm, using the **internal** controller name, e.g.:
```bash
ROBOT_TYPE=UR20 docker compose up -d
```
Valid values: `UR3 UR5 UR7e UR10 UR12e UR15 UR16 UR18 UR20 UR30 UR8LONG` (and the
`g` variants). Note it's `UR10`, not `UR10e`. The consumer suffix is not a valid
controller type here. If you change the type, `docker compose down -v` first so
the old robot's state is wiped.

## Notes
- `:latest` is a moving image tag. For a frozen classroom image, pin a digest.
- The container is `privileged` and runs an inner Docker daemon (the controller
  launches its own service containers), that's expected for URSim.
