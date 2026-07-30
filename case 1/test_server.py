"""Smoke test for the Case 1 MCP server, run in-process (no subprocess, no LLM).

Calls both tools through a FastMCP client against the live robot, so it checks
the tool logic, input validation, and real motion in one go.

    python test_server.py

Prereqs: the simulator is up (../simulation environment) and the robot is
powered on (RUNNING). The server connects to UR_HOST, or 127.0.0.1 by default.
"""
from __future__ import annotations

import asyncio
import logging

from fastmcp import Client

from server import mcp, robot

# The validation check below triggers an expected error; keep the framework from
# logging its traceback so the test output stays clean.
logging.disable(logging.CRITICAL)

HOME_DEG = [0, -90, 0, -90, 0, 0]


async def main() -> None:
    async with Client(mcp) as client:
        names = [t.name for t in await client.list_tools()]
        assert set(names) == {"move_robot_to_position", "example"}, names
        print("tools:", names)

        result = await client.call_tool("example", {})
        assert result.data == "this was an example"
        print("example: ok")

        # Move home (no arguments) and to an explicit pose.
        home = await client.call_tool("move_robot_to_position", {})
        assert home.data["status"] == "reached"
        print("move home:", home.data["joints_deg"])

        pose = await client.call_tool(
            "move_robot_to_position", {"joint_angles_deg": [10, -90, 0, -90, 0, 0]}
        )
        assert pose.data["status"] == "reached"
        print("move pose:", pose.data["joints_deg"])

        # Validation: the wrong number of angles must be rejected.
        try:
            await client.call_tool("move_robot_to_position", {"joint_angles_deg": [0, 0, 0]})
        except Exception as exc:
            print("bad input rejected:", str(exc).splitlines()[-1].strip())
        else:
            raise AssertionError("bad input was accepted")

        await client.call_tool("move_robot_to_position", {})  # park home
    print("ALL PASSED")


if __name__ == "__main__":
    robot.connect()  # fails fast if the simulator is down or the robot is off
    asyncio.run(main())
