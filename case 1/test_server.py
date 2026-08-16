"""Smoke test for the Case 1 MCP server, run in-process (no subprocess, no LLM).

Calls every tool through a FastMCP client against the live robot, so it checks
the tool logic, input validation, the Diamond safety layer, and real motion in
one go. Bronze through Diamond:

  * Bronze  -- move_robot_to_position (home + explicit pose)
  * Silver  -- get_robot_state
  * Gold    -- move_joints_relative, move_linear, run_trajectory (blending)
  * Diamond -- safety layer rejections (speed caps, workspace, floor),
               set_gripper via digital IO

    python test_server.py

Prereqs: the simulator is up (../simulation environment) and the robot is
powered on (RUNNING). The server connects to UR_HOST, or 127.0.0.1 by default.
"""
from __future__ import annotations

import asyncio
import logging

from fastmcp import Client

from server import mcp, robot

# The validation checks below trigger expected errors; keep the framework from
# logging their tracebacks so the test output stays clean.
logging.disable(logging.CRITICAL)

HOME_DEG = [0, -90, 0, -90, 0, 0]

REQUIRED_TOOLS = {
    "move_robot_to_position", "get_robot_state", "move_joints_relative",
    "move_linear", "run_trajectory", "set_gripper", "example",
}


async def expect_rejection(client, tool: str, args: dict, label: str,
                           must_contain: str) -> None:
    """Call a tool with bad input; require it raises for the RIGHT reason."""
    try:
        await client.call_tool(tool, args)
    except Exception as exc:
        message = str(exc)
        assert must_contain.lower() in message.lower(), (
            f"{label}: rejected, but not for the expected reason "
            f"({must_contain!r} not in {message!r})")
        print(f"{label} rejected:", message.splitlines()[-1].strip()[:100])
    else:
        raise AssertionError(f"{label} was accepted -- safety layer failed")


async def main() -> None:
    async with Client(mcp) as client:
        names = [t.name for t in await client.list_tools()]
        assert REQUIRED_TOOLS.issubset(set(names)), names
        print("tools:", sorted(names))

        result = await client.call_tool("example", {})
        assert result.data == "this was an example"
        print("example: ok")

        # --- Silver: the state reader returns the full snapshot ----------- #
        state = await client.call_tool("get_robot_state", {})
        assert set(state.data["joints_deg"]) == {"base", "shoulder", "elbow",
                                                 "wrist1", "wrist2", "wrist3"}
        assert len(state.data["tcp_pose_m_rad"]) == 6
        assert state.data["robot_mode_name"] == "RUNNING"
        assert state.data["ready_to_move"] is True
        assert "moving" in state.data and "gripper_closed" in state.data
        print("get_robot_state:", state.data["robot_mode_name"],
              state.data["safety_status_name"], state.data["joints_deg"])

        # --- Bronze: move home (no arguments) and to an explicit pose ----- #
        home = await client.call_tool("move_robot_to_position", {})
        assert home.data["status"] == "reached"
        print("move home:", home.data["joints_deg"])

        pose = await client.call_tool(
            "move_robot_to_position", {"joint_angles_deg": [10, -90, 0, -90, 0, 0]}
        )
        assert pose.data["status"] == "reached"
        print("move pose:", pose.data["joints_deg"])

        # --- Gold: relative move ------------------------------------------ #
        rel = await client.call_tool(
            "move_joints_relative", {"delta_deg": [10, 0, 0, 0, 0, 0]}
        )
        assert rel.data["status"] == "reached"
        assert abs(rel.data["joints_deg"]["base"] - 20) < 1, rel.data
        print("relative move: base ->", rel.data["joints_deg"]["base"])

        # --- Gold: linear move (straight-line TCP, down 10 cm and back) --- #
        # Linear motion from a singular start (stretched elbow or aligned
        # wrist, both true at home) is rejected by the safety layer, so move
        # to a bent, non-singular pose first.
        bent = await client.call_tool(
            "move_robot_to_position", {"joint_angles_deg": [20, -70, 45, -65, -30, 0]}
        )
        tcp = bent.data["tcp_pose"]
        down = await client.call_tool(
            "move_linear", {"position_m": [tcp[0], tcp[1], tcp[2] - 0.10]}
        )
        assert down.data["status"] == "reached"
        assert abs(down.data["distance_moved_m"] - 0.10) < 0.02, down.data
        print("linear move: down", down.data["distance_moved_m"], "m")
        await client.call_tool(
            "move_linear", {"position_m": [tcp[0], tcp[1], tcp[2]]}
        )

        # --- Gold: blended multi-waypoint trajectory ---------------------- #
        traj = await client.call_tool("run_trajectory", {
            "waypoints_deg": [HOME_DEG,
                              [15, -90, 10, -90, 0, 0],
                              [30, -80, 20, -85, 0, 0]],
        })
        assert traj.data["status"] == "reached"
        assert traj.data["waypoints"] == 3
        assert traj.data["motion_log"], "no motion samples captured"
        assert abs(traj.data["joints_deg"]["base"] - 30) < 1
        print(f"trajectory: {traj.data['waypoints']} waypoints in "
              f"{traj.data['duration_s']}s, {len(traj.data['motion_log'])} "
              f"samples, peak {traj.data['peak_joint_speed_rad_s']} rad/s")

        # --- Diamond: the safety layer must block unsafe commands --------- #
        await expect_rejection(
            client, "move_robot_to_position", {"joint_angles_deg": [0, 0, 0]},
            "bad input", must_contain="Expected 6 joint angles")
        await expect_rejection(
            client, "move_robot_to_position",
            {"joint_angles_deg": HOME_DEG, "speed": 99},
            "over-speed", must_contain="speed")
        await expect_rejection(
            client, "move_robot_to_position",
            {"joint_angles_deg": [0, 45, 0, -90, 0, 0]},
            "arm-into-table", must_contain="Unsafe command")
        await expect_rejection(
            client, "move_linear", {"position_m": [0.4, -0.4, -0.2]},
            "tool-below-floor", must_contain="outside the safe workspace")
        await expect_rejection(
            client, "move_linear", {"position_m": [1.2, -1.2, 1.2]},
            "beyond-reach", must_contain="reach")
        await expect_rejection(
            client, "run_trajectory", {"waypoints_deg": [HOME_DEG]},
            "single-waypoint", must_contain="2 to 20 waypoints")
        await expect_rejection(
            client, "move_joints_relative", {"delta_deg": [10, 0]},
            "bad delta count", must_contain="Expected 6 joint deltas")
        # From home the arm is singular (stretched elbow, aligned wrist):
        # a linear move must be refused, not protective-stopped.
        await client.call_tool("move_robot_to_position", {})
        await expect_rejection(
            client, "move_linear", {"position_m": [0.0, -0.29, 1.30]},
            "linear-from-singularity", must_contain="singularity")
        # Orientation matters: tool z-axis up puts the wrist flange 0.117 m
        # below the TCP; a low target must be refused for the FLANGE.
        await expect_rejection(
            client, "move_linear",
            {"position_m": [0.5, -0.4, 0.05], "rotation_rad": [0, 0, 0]},
            "wrist-under-table", must_contain="flange")

        # --- Diamond: gripper via digital IO, confirmed over RTDE --------- #
        grip = await client.call_tool("set_gripper", {"closed": True})
        assert grip.data["gripper"] == "closed" and grip.data["pin_state"] is True
        release = await client.call_tool("set_gripper", {"closed": False})
        assert release.data["gripper"] == "open" and release.data["pin_state"] is False
        print("gripper: close + open confirmed on DO pin", grip.data["pin"])

        # --- The review-found race: an out-and-back path that ends where it
        # starts must actually RUN, not return 'reached' instantly. -------- #
        loop_traj = await client.call_tool("run_trajectory", {
            "waypoints_deg": [[15, -90, 10, -90, 0, 0], HOME_DEG],
        })
        assert loop_traj.data["duration_s"] > 1.0, (
            f"out-and-back path finished suspiciously fast "
            f"({loop_traj.data['duration_s']}s) -- script probably dropped")
        moved = any(abs(e["joints_deg"][0] - 15) < 5
                    for e in loop_traj.data["motion_log"])
        assert moved, "motion log never shows the intermediate waypoint"
        print("out-and-back trajectory:", loop_traj.data["duration_s"], "s,",
              len(loop_traj.data["motion_log"]), "samples")

        await client.call_tool("move_robot_to_position", {})  # park home
    print("ALL PASSED")


if __name__ == "__main__":
    robot.connect()  # fails fast if the simulator is down or the robot is off
    asyncio.run(main())
