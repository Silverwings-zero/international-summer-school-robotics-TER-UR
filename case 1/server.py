"""MCP server exposing UR robot capabilities as LLM-callable tools.

Case 1 baseline. Run this and connect any MCP client (Claude Code, Cursor, or
the Case 3 agent). Each ``@mcp.tool`` becomes a function the LLM can call by
name; the docstring is what the model reads to decide when and how to use it, so
write it for the model, not just for humans.

This file ships TWO tools:
  * ``move_robot_to_position`` -- the one fully worked, robot-moving tool. It is
    your template. Its shape, for every tool you add:
      1. Validate inputs first and raise ValueError with a plain-language reason.
         The client surfaces that text to the LLM, which can then self-correct.
      2. Convert the request units (degrees) into the robot's units (radians).
      3. Check feasibility against limits before anything moves.
      4. Execute only after the checks pass, and report back the new state.
  * ``example`` -- a do-nothing template showing the minimal tool shape. Copy it
    to start a new tool of your own.

Return JSON-serializable dicts with explicit units in the key names.
"""
from __future__ import annotations

import math

from fastmcp import FastMCP

from ur_client import HOME_Q_RAD, JOINT_LIMIT, JOINT_NAMES, URClient

mcp = FastMCP("ur-tools")
robot = URClient()

# Conservative joint-move defaults (rad/s, rad/s^2).
DEFAULT_SPEED = 1.0
DEFAULT_ACCEL = 1.4

# Home pose as degrees, for readable tool output and defaults.
HOME_DEG = [round(math.degrees(a)) for a in HOME_Q_RAD]


# =========================================================================== #
# WORKED TOOL  --  the one tool that actually moves the robot. Copy its
# four-step shape for every tool you add: 1) validate  2) convert units
# 3) check limits  4) execute + report.
# =========================================================================== #
@mcp.tool
def move_robot_to_position(
    joint_angles_deg: list[float] | None = None,
    speed: float = DEFAULT_SPEED,
    acceleration: float = DEFAULT_ACCEL,
) -> dict:
    """Move the robot to an absolute joint configuration and report the result.

    Give six target joint angles in degrees, ordered base, shoulder, elbow,
    wrist1, wrist2, wrist3. Omit them to send the robot to its HOME position
    ([0, -90, 0, -90, 0, 0] degrees) -- "move the robot home" is a call with no
    arguments.

    The move blocks until the robot arrives, then returns the new robot state
    (so you can observe where it ended up before deciding the next step).

    Args:
        joint_angles_deg: Six target angles in degrees, base..wrist3. Defaults to
            the home pose when omitted.
        speed: Joint speed (rad/s).
        acceleration: Joint acceleration (rad/s^2).

    Returns:
        A dict with the target that was commanded and the resulting state:
        ``joints_deg`` (per-joint angles), ``tcp_pose`` [x, y, z, rx, ry, rz]
        in metres and radians, and ``robot_mode``.

    Raises:
        ValueError: Wrong number of angles, or a target past the joint limit.
        RuntimeError: The robot is not powered on (enable it in the UI first).
    """
    # 1. Validate inputs.
    if joint_angles_deg is None:
        joint_angles_deg = list(HOME_DEG)
    if len(joint_angles_deg) != len(JOINT_NAMES):
        raise ValueError(
            f"Expected {len(JOINT_NAMES)} joint angles "
            f"({', '.join(JOINT_NAMES)}), got {len(joint_angles_deg)}."
        )

    # 2. Convert the request (degrees) to the robot's units (radians).
    target_rad = [math.radians(a) for a in joint_angles_deg]

    # 3. Check feasibility against the software joint limits.
    for name, angle_rad, angle_deg in zip(JOINT_NAMES, target_rad, joint_angles_deg):
        if abs(angle_rad) > JOINT_LIMIT:
            raise ValueError(
                f"Target {angle_deg} deg for {name} exceeds the +/-"
                f"{math.degrees(JOINT_LIMIT):.0f} deg limit. Choose a smaller angle."
            )

    # 4. Execute (blocks until reached), then report the new state.
    state = robot.move_joint(target_rad, speed, acceleration)
    return {
        "status": "reached",
        "target_deg": {n: round(a, 1) for n, a in zip(JOINT_NAMES, joint_angles_deg)},
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "tcp_pose": [round(v, 4) for v in state.tcp_pose],
        "robot_mode": state.robot_mode,
    }


# =========================================================================== #
# TEMPLATE TOOL  --  the minimal shape of a tool, doing nothing real. Copy this
# to start your own (e.g. get_robot_state, move_linear, open/close a gripper),
# then follow the four-step pattern above. Add as many as the tiers need.
# =========================================================================== #
@mcp.tool
def example() -> str:
    """A placeholder tool that performs no action.

    Returns a fixed string. Use it as the skeleton for a real tool of your own.
    """
    return "this was an example"


if __name__ == "__main__":
    robot.connect()
    mcp.run()
