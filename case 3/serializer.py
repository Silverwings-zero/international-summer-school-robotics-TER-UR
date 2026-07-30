"""State serializer: turn raw robot state into a sentence the model can read.

A UR exposes 100+ RTDE signals. Dumping them at the LLM drowns it. The serializer
compresses the state into a short, honest status line, and its quality sets the
ceiling on the whole agent: the model can only reason about what you show it.

A working baseline is provided. It is deliberately generic and reads whatever the
robot-state tool returns. IMPROVING IT IS PART OF THE EXERCISE (Silver+): add the
distance to the current goal, the gripper state, whether the robot is moving, a
protective-stop flag, named objects (Diamond), and drop anything irrelevant.
"""
from __future__ import annotations

import json
import math

# Field-name aliases, because the exact keys depend on how the server's
# get_robot_state tool is written. Extend these to match your server.
JOINT_KEYS = ("joints", "joint_angles_deg", "joints_deg", "q_deg")
TCP_KEYS = ("tcp", "tcp_pose", "pose", "actual_tcp_pose")


def parse_tool_text(text: str) -> dict:
    """Parse a tool result string into a dict (tools return JSON as text).

    Returns an empty dict if the text is not JSON, so callers can degrade
    gracefully instead of crashing.
    """
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {"value": value}


def _first(state: dict, keys) -> object | None:
    for k in keys:
        if k in state:
            return state[k]
    return None


def serialize_state(state: dict) -> str:
    """Return a one-line, human-readable summary of the robot state.

    Args:
        state: Parsed result of the server's robot-state tool.

    Returns:
        A short status sentence. Falls back to a compact key dump for any
        fields it does not recognise (so nothing is silently lost).
    """
    if not state:
        return "Robot state unavailable."

    parts: list[str] = []

    tcp = _first(state, TCP_KEYS)
    if isinstance(tcp, (list, tuple)) and len(tcp) >= 3:
        x, y, z = tcp[0], tcp[1], tcp[2]
        parts.append(f"TCP at ({x:.3f}, {y:.3f}, {z:.3f}) m")

    joints = _first(state, JOINT_KEYS)
    if isinstance(joints, dict):
        js = ", ".join(f"{k} {float(v):.0f}°" for k, v in joints.items())
        parts.append(f"joints [{js}]")
    elif isinstance(joints, (list, tuple)):
        js = ", ".join(f"{float(v):.0f}°" for v in joints)
        parts.append(f"joints [{js}]")

    if "gripper" in state:
        parts.append(f"gripper {state['gripper']}")
    if state.get("moving") is not None:
        parts.append("moving" if state["moving"] else "stationary")

    # Anything we did not explicitly format: append as key=value so no signal is
    # lost. Remove this once your serializer covers the fields you care about.
    known = set(JOINT_KEYS) | set(TCP_KEYS) | {"gripper", "moving"}
    extra = {k: v for k, v in state.items() if k not in known}
    if extra:
        parts.append("other: " + ", ".join(f"{k}={v}" for k, v in extra.items()))

    return "; ".join(parts) + "." if parts else "Robot state unavailable."


def distance_m(pose_a, pose_b) -> float:
    """Euclidean distance (metres) between the xyz of two TCP poses."""
    return math.dist(pose_a[:3], pose_b[:3])
