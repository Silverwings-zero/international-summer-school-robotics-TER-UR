"""Smoke test for the continuous motion patterns, run in-process (no LLM).

Drives a full stirring session through the MCP tools against the live robot:
start, speed up, pause, resume, refuse an unsafe speed, finish. The speed
assertions are the important ones -- an earlier design commanded a higher speed
and the robot silently ignored it, so "faster" is measured here, not trusted.

    python test_motion_patterns.py

Prereqs: the simulator is up (../simulation environment) and the robot is
powered on (RUNNING).
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

from fastmcp import Client

from server import mcp, robot

logging.disable(logging.CRITICAL)

# Bent, non-singular, well inside the workspace: a legal place to start a
# Cartesian motion from.
START_Q_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
PAN_DIAMETER_CM = 10.0
EXPECTED_RADIUS_M = 0.85 * (PAN_DIAMETER_CM / 100.0) / 2.0


def measure_tcp_speed(seconds: float = 2.5) -> float:
    """Median tool speed over a window, in m/s."""
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        samples.append((time.monotonic(), list(robot.get_tcp_pose()[:3])))
    speeds = sorted(math.dist(p1, p2) / (t2 - t1)
                    for (t1, p1), (t2, p2) in zip(samples, samples[1:])
                    if t2 > t1)
    return speeds[len(speeds) // 2] if speeds else 0.0


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
        print(f"{label} rejected:", message.splitlines()[-1].strip()[:110])
    else:
        raise AssertionError(f"{label} was accepted -- validation failed")


async def main() -> None:
    async with Client(mcp) as client:
        names = {t.name for t in await client.list_tools()}
        required = {"stir", "start_motion_pattern", "adjust_pattern_speed",
                    "pause_motion", "resume_motion", "finish_motion",
                    "get_motion_status"}
        assert required.issubset(names), sorted(required - names)
        print("motion tools registered:", sorted(required))

        # --- Validation, before anything moves ---------------------------- #
        idle = await client.call_tool("get_motion_status", {})
        assert idle.data["active"] is False, idle.data
        print("status while idle:", idle.data["status"])

        await expect_rejection(client, "adjust_pattern_speed", {"factor": 1.5},
                               "speed change with nothing running",
                               "no motion pattern is running")
        # Stop verbs never refuse: they must also stop a loop left running by a
        # previous server session, which has no job to point at.
        quiet = (await client.call_tool("pause_motion", {})).data
        assert quiet["status"] == "idle", quiet
        print("pause with nothing running:", quiet["status"])

        await expect_rejection(client, "stir", {"container_diameter_cm": 200.0},
                               "absurd container", "outside the workable range")
        await expect_rejection(client, "start_motion_pattern",
                               {"pattern": "zigzag", "radius_m": 0.04,
                                "speed_m_s": 0.08},
                               "unknown pattern", "unknown pattern")
        await expect_rejection(client, "start_motion_pattern",
                               {"pattern": "circle", "radius_m": 0.02,
                                "speed_m_s": 0.30},
                               "speed past the centripetal limit",
                               "contents in the container")

        # --- Position the tool where the pan would be --------------------- #
        placed = await client.call_tool("move_robot_to_position",
                                        {"joint_angles_deg": START_Q_DEG})
        anchor = placed.data["tcp_pose_m_rad"]
        print("tool placed at:", [round(v, 3) for v in anchor])

        # --- Start stirring ----------------------------------------------- #
        started = (await client.call_tool(
            "stir", {"container_diameter_cm": PAN_DIAMETER_CM})).data
        assert started["status"] == "started", started
        assert abs(started["radius_m"] - EXPECTED_RADIUS_M) < 1e-4, started
        assert started["exact_path"] is True, "a circle must run as true arcs"
        base_speed = started["speed_m_s"]
        print(f"stirring: r={started['radius_m'] * 1000:.1f}mm "
              f"v={base_speed} m/s  {started['laps_per_min']} laps/min  "
              f"(max safe {started['max_safe_speed_m_s']})")

        running = (await client.call_tool("get_motion_status", {})).data
        assert running["state"] == "running" and running["robot_moving"], running
        assert running["safety_status"] == 1, running

        slow = measure_tcp_speed()
        print(f"measured at 1.0x: {slow:.4f} m/s "
              f"({slow / base_speed * 100:.0f}% of commanded)")
        assert slow > 0.5 * base_speed, "the arm is not tracking the command"

        # The circle must stay a circle: every sample one radius from the anchor.
        radii = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            tcp = robot.get_tcp_pose()
            radii.append(math.dist(tcp[:2], anchor[:2]))
        error_mm = max(abs(r - EXPECTED_RADIUS_M) for r in radii) * 1000
        print(f"radius held to {error_mm:.2f} mm of the commanded "
              f"{EXPECTED_RADIUS_M * 1000:.1f} mm")
        assert error_mm < 3.0, f"path is not a clean circle ({error_mm:.2f} mm)"

        # --- Faster: the regression test that matters --------------------- #
        faster = (await client.call_tool("adjust_pattern_speed",
                                         {"factor": 1.5})).data
        assert faster["status"] == "speed_changed", faster
        assert abs(faster["speed_m_s"] - base_speed * 1.5) < 1e-6, faster
        quick = measure_tcp_speed()
        print(f"measured at 1.5x: {quick:.4f} m/s "
              f"(ratio {quick / slow:.2f}, commanded ratio 1.50)")
        assert quick > slow * 1.25, (
            f"'faster' did not actually speed the robot up: "
            f"{slow:.4f} -> {quick:.4f} m/s")

        # 2.2x lands at 0.264 m/s: inside the absolute cap, past the
        # centripetal one, so it is the CONTENTS check that must fire.
        await expect_rejection(client, "adjust_pattern_speed", {"factor": 2.2},
                               "faster past the safe limit",
                               "contents in the container")
        await expect_rejection(client, "adjust_pattern_speed",
                               {"factor": 1.2, "speed_m_s": 0.1},
                               "both speed arguments", "exactly one of")

        # --- Pause, check, resume ----------------------------------------- #
        paused = (await client.call_tool("pause_motion", {})).data
        assert paused["state"] == "paused", paused
        assert not robot.get_state().is_moving, "pause left the arm moving"
        held = robot.get_tcp_pose()
        time.sleep(1.0)
        drift = math.dist(held[:3], robot.get_tcp_pose()[:3])
        print(f"paused after {paused['laps_completed']} laps, "
              f"drift while held {drift * 1000:.2f} mm")
        assert drift < 0.001, "the arm moved while paused"

        status = (await client.call_tool("get_motion_status", {})).data
        assert status["state"] == "paused" and not status["robot_moving"], status

        resumed = (await client.call_tool("resume_motion", {})).data
        assert resumed["state"] == "running", resumed
        assert abs(resumed["speed_m_s"] - base_speed * 1.5) < 1e-6, (
            "resume lost the adjusted speed")
        time.sleep(0.5)
        assert robot.get_state().is_moving, "resume did not restart the motion"
        print(f"resumed at {resumed['speed_m_s']} m/s")

        # --- Finish -------------------------------------------------------- #
        finished = (await client.call_tool("finish_motion", {})).data
        assert finished["status"] == "finished", finished
        assert not robot.get_state().is_moving, "finish left the arm moving"
        print(f"finished after {finished['laps_completed']} laps "
              f"in {finished['total_duration_s']}s")

        after = (await client.call_tool("get_motion_status", {})).data
        assert after["active"] is False, after
        await expect_rejection(client, "resume_motion", {},
                               "resume after finish", "no motion pattern")

        # --- A sampled (non-circular) pattern still runs -------------------- #
        eight = (await client.call_tool(
            "start_motion_pattern",
            {"pattern": "figure_eight", "radius_m": 0.05,
             "speed_m_s": 0.06, "points_per_lap": 16})).data
        assert eight["exact_path"] is False, eight
        time.sleep(2.0)
        assert robot.get_state().is_moving, "figure_eight did not move"
        print(f"figure_eight running at {eight['speed_m_s']} m/s, "
              f"{eight['laps_per_min']} laps/min")
        await client.call_tool("finish_motion", {})

        state = robot.get_state()
        assert state.safety_status == 1, (
            f"safety status ended at {state.safety_status}, not NORMAL")
        print("\nall motion pattern checks passed; safety NORMAL throughout")


if __name__ == "__main__":
    asyncio.run(main())
