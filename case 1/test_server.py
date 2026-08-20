"""Smoke test for the Case 1 MCP server, run in-process (no subprocess, no LLM).

Calls every tool through a FastMCP client against the live robot, so it checks
the tool logic, input validation, the Diamond safety layer, and real motion in
one go. Bronze through Diamond:

  * Bronze  -- move_robot_to_position (home + explicit pose)
  * Silver  -- get_robot_state
  * Gold    -- move_joints_relative, move_linear, run_trajectory (blending)
  * Diamond -- safety layer rejections (speed caps, workspace, floor),
               the gripper via its two tool digital outputs

    python test_server.py

Prereqs: the simulator is up (../simulation environment) and the robot is
powered on (RUNNING). The server connects to UR_HOST, or 127.0.0.1 by default.
"""
from __future__ import annotations

import asyncio
import logging

from fastmcp import Client

import os

# The geometric test targets (reach, singularity probes) are UR10e-scale;
# pin the model so the suite is independent of the ambient UR_MODEL.
os.environ.setdefault("UR_MODEL", "ur10e")

from server import mcp, robot

# The validation checks below trigger expected errors; keep the framework from
# logging their tracebacks so the test output stays clean.
logging.disable(logging.CRITICAL)

# The kitchen home: upper arm vertical, forearm horizontal, tool straight
# down, wrist2 clear of the singularity (must match ur_client.HOME_Q_RAD).
HOME_DEG = [0, -90, 90, -90, -90, 0]
# The old stretched home -- kept around because it is the canonical SINGULAR
# configuration (straight elbow AND aligned wrist) the safety tests need.
STRETCHED_DEG = [0, -90, 0, -90, 0, 0]

REQUIRED_TOOLS = {
    "move_robot_to_position", "get_robot_state", "move_joints_relative",
    "move_linear", "run_trajectory", "start_trajectory_job",
    "get_trajectory_job_status", "move_joint", "set_payload",
    "set_payload_mass", "set_gravity", "get_digital_in", "set_digital_out",
    "store_waypoint_pose_on_ur", "store_joint_configuration_on_ur",
    "move_to_stored_tcp_waypoint", "move_to_stored_joint_configuration",
    "get_tool_digital_in", "set_tool_digital_out",
    "example",
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
        assert "moving" in state.data and "gripper_open" in state.data
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

        # Alias compatibility: move_joint should behave exactly like
        # move_robot_to_position.
        alias_pose = await client.call_tool(
            "move_joint", {"joint_angles_deg": [15, -90, 0, -90, 0, 0]}
        )
        assert alias_pose.data["status"] == "reached"
        print("move_joint alias:", alias_pose.data["joints_deg"])

        # --- Gold: relative move ------------------------------------------ #
        base_before_rel = alias_pose.data["joints_deg"]["base"]
        rel = await client.call_tool(
            "move_joints_relative", {"delta_deg": [10, 0, 0, 0, 0, 0]}
        )
        assert rel.data["status"] == "reached"
        expected_base = base_before_rel + 10
        assert abs(rel.data["joints_deg"]["base"] - expected_base) < 1, rel.data
        print("relative move: base ->", rel.data["joints_deg"]["base"])

        # --- Gold: linear move (straight-line TCP, down 10 cm and back) --- #
        # Linear motion from a singular start (stretched elbow or aligned
        # wrist, both true at home) is rejected by the safety layer, so move
        # to a bent, non-singular pose first.
        bent = await client.call_tool(
            "move_robot_to_position", {"joint_angles_deg": [20, -70, 45, -65, -30, 0]}
        )
        tcp = bent.data["tcp_pose_m_rad"]
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

        # --- Gold: async trajectory job with incremental progress --------- #
        started = await client.call_tool("start_trajectory_job", {
            "waypoints_deg": [[30, -80, 20, -85, 0, 0], HOME_DEG],
            "blend_m": 0.0,
        })
        job_id = started.data["job_id"]
        next_index = 0
        saw_progress = False
        final_status = None
        final_result = None
        for _ in range(240):
            status = await client.call_tool("get_trajectory_job_status", {
                "job_id": job_id,
                "from_index": next_index,
                "limit": 20,
            })
            chunk = status.data["progress_samples"]
            if chunk:
                saw_progress = True
            next_index = status.data["next_index"]
            final_status = status.data["status"]
            if final_status in ("completed", "failed"):
                final_result = status.data.get("result")
                break
            await asyncio.sleep(0.25)
        assert final_status == "completed", (
            f"trajectory job did not complete successfully: {final_status}")
        assert saw_progress, "trajectory job produced no incremental progress"
        assert final_result and final_result["status"] == "reached"
        print("trajectory job: completed with", next_index, "progress samples")

        # --- Silver/Gold: store + immediate reuse across calls ----------- #
        stored_tcp = await client.call_tool("store_waypoint_pose_on_ur", {
            "variable_name": "tmp_waypoint_now",
        })
        assert stored_tcp.data["status"] == "stored_on_ur"
        moved_tcp = await client.call_tool("move_to_stored_tcp_waypoint", {
            "variable_name": "tmp_waypoint_now",
            "speed": 0.2,
            "acceleration": 1.0,
            "timeout_s": 20.0,
        })
        assert moved_tcp.data["status"] == "executed"

        stored_q = await client.call_tool("store_joint_configuration_on_ur", {
            "variable_name": "tmp_joint_now",
        })
        assert stored_q.data["status"] == "stored_on_ur"
        moved_q = await client.call_tool("move_to_stored_joint_configuration", {
            "variable_name": "tmp_joint_now",
            "speed": 0.8,
            "acceleration": 1.0,
            "timeout_s": 20.0,
        })
        assert moved_q.data["status"] == "executed"
        print("stored variables: immediate waypoint + joint reuse confirmed")

        # --- Silver: commissioning/utility tools + input validation ------- #
        payload = await client.call_tool("set_payload", {
            "mass_kg": 0.5,
            "cog_m": [0.0, 0.0, 0.02],
        })
        assert payload.data["status"] == "applied"
        assert abs(payload.data["mass_kg"] - 0.5) < 1e-9
        print("set_payload:", payload.data)

        payload_mass = await client.call_tool("set_payload_mass", {
            "mass_kg": 0.25,
        })
        assert payload_mass.data["status"] == "applied"
        assert abs(payload_mass.data["mass_kg"] - 0.25) < 1e-9
        print("set_payload_mass:", payload_mass.data)

        gravity = await client.call_tool("set_gravity", {
            "gravity_m_s2": [0.0, 0.0, -9.82],
        })
        assert gravity.data["status"] == "applied"
        assert abs(gravity.data["magnitude_m_s2"] - 9.82) < 0.05
        print("set_gravity:", gravity.data)

        await expect_rejection(
            client, "set_payload", {"mass_kg": -1.0},
            "payload-negative-mass", must_contain="non-negative")
        await expect_rejection(
            client, "set_payload", {"mass_kg": 0.2, "cog_m": [0.0, 0.0]},
            "payload-bad-cog", must_contain="Expected cog_m")
        await expect_rejection(
            client, "set_payload", {"mass_kg": 12.5, "cog_m": [0.6, 0.9, 0.5]},
            "payload-bad-cog", must_contain="Invalid CoG")
        await expect_rejection(
            client, "set_gravity", {"gravity_m_s2": [0.0, 0.0, -1.0]},
            "gravity-magnitude", must_contain="looks invalid")

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
        # From the stretched pose the arm is singular (straight elbow,
        # aligned wrist): a linear move must be refused, not
        # protective-stopped. (This used to be home; the kitchen home is
        # deliberately non-singular, so stretch explicitly here.)
        await client.call_tool("move_robot_to_position",
                               {"joint_angles_deg": STRETCHED_DEG})
        await expect_rejection(
            client, "move_linear", {"position_m": [0.0, -0.29, 1.30]},
            "linear-from-singularity", must_contain="singularity")
        # Orientation matters: tool z-axis up puts the wrist flange 0.117 m
        # below the TCP; a low target must be refused for the FLANGE.
        await expect_rejection(
            client, "move_linear",
            {"position_m": [0.5, -0.4, 0.05], "rotation_rad": [0, 0, 0]},
            "wrist-under-table", must_contain="flange")

        # --- Freedrive mode: start + stop, with a long sleep in between -------- #
        freedrive = await client.call_tool("start_freedrive_mode", {})
        print("in free drivemode")
        
        assert freedrive.data["status"] == "started"
        await asyncio.sleep(5.0)
        stop_freedrive = await client.call_tool("stop_freedrive_mode", {})
        assert stop_freedrive.data["status"] == "stopped"

        # --- Diamond: the gripper is set_tool_digital_out and nothing else.
        # Pin 0 is the jaws (False closes, True opens), pin 1 the speed
        # (True slow, False fast); both are confirmed over RTDE. ---------- #
        slow = await client.call_tool("set_tool_digital_out", {"n": 1, "b": True})
        assert slow.data["value"] is True and slow.data["meaning"] == "slow"
        grip = await client.call_tool("set_tool_digital_out", {"n": 0, "b": False})
        assert grip.data["value"] is False and grip.data["meaning"] == "closed"
        state = await client.call_tool("get_robot_state", {})
        assert state.data["gripper_open"] is False
        assert state.data["gripper_speed"] == "slow"
        release = await client.call_tool("set_tool_digital_out", {"n": 0, "b": True})
        assert release.data["value"] is True and release.data["meaning"] == "open"
        fast = await client.call_tool("set_tool_digital_out", {"n": 1, "b": False})
        assert fast.data["meaning"] == "fast"
        print("gripper: close + open confirmed on tool DO 0, speed on tool DO 1")
        # Only pins 0 and 1 exist on the tool connector.
        await expect_rejection(
            client, "set_tool_digital_out", {"n": 2, "b": True},
            "tool-do-out-of-range", must_contain="0 or 1")

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
