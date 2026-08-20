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

Prompting guidance for MCP clients (important for reliable tool use):
    * If a user asks to "save/store a position" but does not specify whether they
        mean Cartesian TCP pose or joint configuration, ASK a clarification question
        before calling a storage or motion tool.
    * Use TCP/cartesian language with pose variables (p[x,y,z,rx,ry,rz]); use
        joint/configuration language with six joint angles.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastmcp import FastMCP

from motion_patterns import PatternRunner, max_speed_for_radius
from robotiq_gripper import (
    HARDWARE_FAULTS,
    GripperError,
    RobotiqGripper,
)
from ur_client import (
    HOME_Q_RAD,
    JOINT_LIMIT,
    JOINT_NAMES,
    ROBOT_MODE_RUNNING,
    URClient,
)

mcp = FastMCP(
    "ur-tools",
    instructions="""
You are a robot assistant helping a non-expert user with simple kitchen tasks.
The user does not know anything about robotics, joints, or coordinates — never
use that language with them.

CONVERSATION STYLE
- Start by asking what the user wants to do, in plain language.
- Once you know the goal (e.g. "scramble some eggs"), lay out the full sequence
  of steps up front in plain language before doing anything.
- Work through the steps one at a time, narrating briefly before each action.
- You do not have a way to sense temperature. Treat "fry" or "cook" as: pour
  the ingredients into the pan, then pick up the spatula and stir. Do not wait
  for the pan to heat up or judge doneness.

ACTING INDEPENDENTLY
- You have safety and limit checks built into your motion tools. You do not
  need to ask for confirmation before ordinary, known-safe actions (moving to
  a known position, pouring, stirring). Just do them and briefly say what
  you're doing.
- Only pause and ask the user when you genuinely need their input — not as a
  general courtesy check-in.

GETTING ACCESS TO THINGS — pick the right one of these three:
1. Known checkpoint: if the item's location is a known, saved position (e.g.
   an ingredient at a fixed spot), move there directly. No need to ask first.
2. Tool handoff: if you need a tool (spatula, whisk) that the user is holding,
   ask them to place it in your open gripper. Ask one simple yes/no question
   — e.g. "Is it between my gripper fingers now?" — and wait for their answer
   before closing the gripper. Do not write more than that one short question.
3. Freedrive (unknown/complex position): if you do not know a position and it
   is not a known checkpoint or a handoff, ask the user to guide you there
   using freedrive, then confirm once they say they're done.
Never guess a position you don't have.

STAYING ON TRACK
- If the user's request is ambiguous (e.g. "put it there" without saying
  where), ask a short clarifying question rather than assuming.
- If a tool call fails (a safety/limit check, an unreachable pose), explain
  the problem in plain terms and say what the user can do to help — hand you
  the item differently, freedrive you to the spot — rather than retrying
  blindly.
""".strip()
)
robot = URClient()

# Conservative joint-move defaults (rad/s, rad/s^2).
DEFAULT_SPEED = 1.0
DEFAULT_ACCEL = 1.4

# Home pose as degrees, for readable tool output and defaults.
HOME_DEG = [round(math.degrees(a)) for a in HOME_Q_RAD]

# Human-readable names for the UR mode/safety codes, so state reads back as
# words the LLM can reason over instead of bare integers.
ROBOT_MODE_NAMES = {
    0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
    4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}
SAFETY_STATUS_NAMES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
    5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION", 9: "FAULT", 10: "VALIDATE_JOINT_ID",
    11: "UNDEFINED_SAFETY_MODE", 12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
}

# The digital output a gripper listens on (tool IO convention; the simulator
# has no physical gripper, so this drives the command line one would read).
# On the real cell a Robotiq is driven through its URCap instead -- see
# _robotiq() below; the pin stays as the simulator fallback.
GRIPPER_PIN = 0

# The Robotiq driver, connected lazily. None means "not found last time we
# looked" -- and we look again on every gripper call, so plugging the gripper
# in (or installing the URCap) later just works without a server restart.
_GRIPPER_LOCK = threading.Lock()
_ROBOTIQ: RobotiqGripper | None = None


def _robotiq() -> RobotiqGripper | None:
    """Return a live Robotiq driver, or None when no URCap answers."""
    global _ROBOTIQ
    with _GRIPPER_LOCK:
        if _ROBOTIQ is not None:
            return _ROBOTIQ
    # Probe OUTSIDE the lock: when the controller is unreachable this blocks
    # for the connect timeout, and holding the lock would serialize every
    # gripper call (and get_robot_state) behind it.
    candidate = RobotiqGripper(host=robot.host)
    try:
        candidate.connect()
    except OSError:
        return None
    with _GRIPPER_LOCK:
        if _ROBOTIQ is not None:
            # Another thread won the race; keep its connection, drop ours.
            winner = _ROBOTIQ
            candidate.close()
            return winner
        _ROBOTIQ = candidate
        return _ROBOTIQ


def _drop_robotiq(failed: RobotiqGripper) -> None:
    """Forget a dead connection so the next call re-probes from scratch.

    Identity-aware: a late error from an old connection must not tear down
    the healthy one that replaced it.
    """
    global _ROBOTIQ
    with _GRIPPER_LOCK:
        if _ROBOTIQ is failed:
            _ROBOTIQ = None
    failed.close()


# =========================================================================== #
# DIAMOND SAFETY LAYER  --  every motion tool funnels through these checks
# (geometry comes from the UR_MODEL table below; the ur10e row is verified
# to match the simulator's TCP to the millimetre)
# BEFORE anything moves. A forward-kinematics model (UR10e DH parameters,
# verified to match the simulator's TCP to the millimetre) predicts where a
# joint target puts the whole arm, so unsafe commands are rejected with a
# plain-language reason the LLM can read and correct, instead of executed.
# =========================================================================== #

# Command caps. Joint units rad/s, rad/s^2 (UR10e's slowest joints allow
# 2.09 rad/s); TCP units m/s, m/s^2.
# Dynamics caps. Overridable per cell (e.g. gentler for a real arm around
# people) without touching code: UR_MAX_JOINT_SPEED etc.
MAX_JOINT_SPEED = float(os.environ.get("UR_MAX_JOINT_SPEED", "2.0"))
MAX_JOINT_ACCEL = float(os.environ.get("UR_MAX_JOINT_ACCEL", "4.0"))
MAX_TCP_SPEED = float(os.environ.get("UR_MAX_TCP_SPEED", "1.0"))
MAX_TCP_ACCEL = float(os.environ.get("UR_MAX_TCP_ACCEL", "2.5"))

# The safety layer computes forward kinematics, reach, and workspace bounds
# from the robot's geometry, so it must know WHICH arm it is guarding.
# UR_MODEL selects a row; the default matches the PolyScope X simulator
# (a UR10e). Set UR_MODEL=ur5e (or ur5) when driving the real arm.
# DH values (a, d) are UR's published standard-DH tables; reach_m is the
# kinematic maximum from the shoulder point, wrist offsets included;
# workspace_m boxes the tool tip in the base frame.
_UR_MODELS = {
    "ur10e": {
        "dh_a": (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0),
        "dh_d": (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655),
        "reach_m": 1.37,
        "payload_kg": 12.5,
        "workspace_m": {"x": (-1.30, 1.30), "y": (-1.30, 1.30),
                        "z": (0.02, 1.60)},
    },
    "ur5e": {
        "dh_a": (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
        "dh_d": (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
        "reach_m": 0.95,
        "payload_kg": 5.0,
        "workspace_m": {"x": (-0.95, 0.95), "y": (-0.95, 0.95),
                        "z": (0.02, 1.20)},
    },
    "ur5": {  # CB-series UR5
        "dh_a": (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0),
        "dh_d": (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823),
        "reach_m": 0.92,
        "payload_kg": 5.0,
        "workspace_m": {"x": (-0.92, 0.92), "y": (-0.92, 0.92),
                        "z": (0.02, 1.05)},
    },
}
UR_MODEL = os.environ.get("UR_MODEL", "ur10e").strip().lower()
if UR_MODEL not in _UR_MODELS:
    raise SystemExit(
        f"Unknown UR_MODEL '{UR_MODEL}'; pick one of {sorted(_UR_MODELS)}")

# The default above is the SIMULATOR's arm, which is only safe because the
# host default is the simulator too. Pointing at real hardware without saying
# which arm it is would hand the safety layer a UR10e's reach (1.37 m vs the
# UR5e's 0.95), payload (12.5 kg vs 5.0) and workspace box -- so every check
# below would APPROVE targets the real robot cannot survive, and the ur10e
# waypoint bank would be loaded on top. That is the one combination CLAUDE.md
# rules out, so refuse it here rather than guarding the wrong arm.
_UR_HOST = os.environ.get("UR_HOST", "127.0.0.1").strip()
_HOST_IS_LOOPBACK = (_UR_HOST in ("localhost", "::1", "")
                     or _UR_HOST.startswith("127."))
if not _HOST_IS_LOOPBACK and "UR_MODEL" not in os.environ:
    raise SystemExit(
        f"Refusing to start: UR_HOST={_UR_HOST} is not the simulator, but "
        f"UR_MODEL is unset so the safety layer would guard a "
        f"'{UR_MODEL}'.\n"
        f"Name the arm explicitly -- UR_MODEL=ur5e for the summer-school "
        f"UR5e -- or launch through run_server.sh, which pins the host "
        f"and the model together so they cannot disagree.")
_MODEL = _UR_MODELS[UR_MODEL]

# Safe workspace for the tool, base frame, metres. The tool tip may work down
# to FLOOR_Z_M; the arm's joint origins (thick links, ~5 cm radius) must keep
# the larger ARM_CLEARANCE_M so the physical link never touches the table even
# though the check runs on the centreline.
WORKSPACE_M = _MODEL["workspace_m"]
FLOOR_Z_M = 0.02
ARM_CLEARANCE_M = 0.08
# A Cartesian target beyond this can never be reached and would only make
# movel stall until timeout.
REACH_M = _MODEL["reach_m"]

_DH_A = _MODEL["dh_a"]
_DH_D = _MODEL["dh_d"]
_DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)

# Which frame origin sits where on the arm, for readable error messages.
_FRAME_NAMES = ("shoulder", "elbow", "wrist1", "wrist2", "wrist3", "tool")

# Trajectory job registry for long-running progress polling.
_TRAJECTORY_JOBS_LOCK = threading.Lock()
_TRAJECTORY_JOBS: dict[str, "TrajectoryJob"] = {}
_MAX_STORED_PROGRESS_SAMPLES = 500
# Stored poses are only meaningful for the arm they were taught on, so every
# non-simulator model gets its own bank file; the legacy unsuffixed file
# belongs to the ur10e simulator it was recorded on.
_WAYPOINT_LOOKUP_TABLE_PATH = Path(__file__).with_name(
    "waypoints_lookup_table.json" if UR_MODEL == "ur10e"
    else f"waypoints_lookup_table.{UR_MODEL}.json")
_WAYPOINT_BANK_LOCK = threading.Lock()


@dataclass
class TrajectoryJob:
    """In-memory status for a trajectory started asynchronously."""

    job_id: str
    status: str = "running"
    created_at_s: float = field(default_factory=time.monotonic)
    updated_at_s: float = field(default_factory=time.monotonic)
    requested_waypoints: int = 0
    progress_samples: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None


def _build_motion_log(
    samples: list[tuple[float, list[float], float]],
    *,
    max_entries: int = 25,
) -> list[dict]:
    """Return a compact, evenly sampled motion log for response payloads."""
    step = max(1, len(samples) // max_entries)
    return [
        {
            "t_s": round(t, 2),
            "joints_deg": [round(math.degrees(q), 1) for q in qs],
        }
        for t, qs, _ in samples[::step][:max_entries]
    ]


def _run_trajectory_impl(
    waypoints_deg: list[list[float]],
    speed: float,
    acceleration: float,
    blend_m: float,
    *,
    progress_callback=None,
) -> dict:
    """Shared trajectory implementation used by sync and async tools."""
    if not 2 <= len(waypoints_deg) <= 20:
        raise ValueError(
            f"Expected 2 to 20 waypoints, got {len(waypoints_deg)}. "
            "For a single target use move_robot_to_position."
        )
    if not 0 <= blend_m <= 0.1:
        raise ValueError(
            f"Blend radius {blend_m} m is outside the allowed range [0, 0.1]."
        )

    targets_rad = []
    for i, wp in enumerate(waypoints_deg, start=1):
        try:
            targets_rad.append(_checked_joint_target(wp))
        except ValueError as exc:
            raise ValueError(f"Waypoint {i}: {exc}") from exc

    _check_joint_dynamics(speed, acceleration)
    for i, target in enumerate(targets_rad, start=1):
        _check_pose_safe(target, what=f"waypoint {i}")

    prev = robot.get_state().q_rad
    travel_s = 0.0
    for target in targets_rad:
        travel_s += max(abs(t - p) for t, p in zip(target, prev)) / speed
        prev = target
    timeout = max(60.0, 2.0 * travel_s + 10.0)

    t0 = time.monotonic()
    samples: list[tuple[float, list[float], float]] = []

    def on_sample(st) -> None:
        elapsed_s = time.monotonic() - t0
        peak_qd = max(abs(v) for v in st.qd_rad)
        q_rad = list(st.q_rad)
        samples.append((elapsed_s, q_rad, peak_qd))
        if progress_callback is not None:
            progress_callback({
                "t_s": round(elapsed_s, 2),
                "joints_deg": [round(math.degrees(q), 1) for q in q_rad],
                "joint_speed_rad_s": round(peak_qd, 4),
            })

    state = robot.move_path(
        targets_rad,
        speed,
        acceleration,
        blend_m=blend_m,
        timeout_s=timeout,
        on_sample=on_sample,
    )
    duration = time.monotonic() - t0

    return {
        "status": "reached",
        "waypoints": len(waypoints_deg),
        "duration_s": round(duration, 2),
        "peak_joint_speed_rad_s": round(
            max((s[2] for s in samples), default=0.0),
            3,
        ),
        "motion_log": _build_motion_log(samples),
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
        "tcp_pose": [round(v, 4) for v in state.tcp_pose],
        "robot_mode": state.robot_mode,
    }


def _set_trajectory_job_failed(job: TrajectoryJob, exc: Exception) -> None:
    """Mark a trajectory job as failed with a plain-language error."""
    with _TRAJECTORY_JOBS_LOCK:
        job.status = "failed"
        job.error = str(exc)
        job.updated_at_s = time.monotonic()


def _start_trajectory_job_runner(
    job: TrajectoryJob,
    waypoints_deg: list[list[float]],
    speed: float,
    acceleration: float,
    blend_m: float,
) -> None:
    """Start the background worker that executes one trajectory job."""

    def _runner() -> None:
        try:
            def _on_progress(sample: dict) -> None:
                with _TRAJECTORY_JOBS_LOCK:
                    job.progress_samples.append(sample)
                    if len(job.progress_samples) > _MAX_STORED_PROGRESS_SAMPLES:
                        del job.progress_samples[:
                                                 len(job.progress_samples)
                                                 - _MAX_STORED_PROGRESS_SAMPLES]
                    job.updated_at_s = time.monotonic()

            result = _run_trajectory_impl(
                waypoints_deg=waypoints_deg,
                speed=speed,
                acceleration=acceleration,
                blend_m=blend_m,
                progress_callback=_on_progress,
            )
            with _TRAJECTORY_JOBS_LOCK:
                job.status = "completed"
                job.result = result
                job.updated_at_s = time.monotonic()
        except Exception as exc:
            _set_trajectory_job_failed(job, exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


def _get_trajectory_job_or_raise(job_id: str) -> TrajectoryJob:
    """Return a known trajectory job by id or raise a readable ValueError."""
    with _TRAJECTORY_JOBS_LOCK:
        job = _TRAJECTORY_JOBS.get(job_id)
    if job is None:
        raise ValueError(
            f"Unknown trajectory job_id '{job_id}'. Start one with "
            "start_trajectory_job first."
        )
    return job


def _validate_ur_variable_name(name: str) -> str:
    """Return a safe URScript variable name or raise ValueError.

    URScript identifiers: letters/underscore first, then letters/digits/underscore.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("variable_name must be a non-empty string.")
    candidate = name.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
        raise ValueError(
            "Invalid variable_name. Use only letters, digits, and underscore, "
            "and do not start with a digit."
        )
    return candidate


def _load_waypoint_bank() -> dict:
    """Load local waypoint bank or create an empty in-memory structure."""
    if not _WAYPOINT_LOOKUP_TABLE_PATH.exists():
        return {
            "pose_variables": {},
            "joint_variables": {},
        }
    try:
        data = json.loads(_WAYPOINT_LOOKUP_TABLE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"waypoints_lookup_table.json is invalid JSON: {_WAYPOINT_LOOKUP_TABLE_PATH} ({exc})."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("waypoints_lookup_table.json root must be an object.")
    if "pose_variables" not in data or not isinstance(data["pose_variables"], dict):
        data["pose_variables"] = {}
    if "joint_variables" not in data or not isinstance(data["joint_variables"], dict):
        data["joint_variables"] = {}
    return data


def _write_waypoint_bank(data: dict) -> None:
    """Write local waypoint bank to disk."""
    _WAYPOINT_LOOKUP_TABLE_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _save_pose_to_waypoint_bank(name: str, values_m_rad: list[float]) -> None:
    """Create or update one pose waypoint entry in waypoints_lookup_table.json."""
    with _WAYPOINT_BANK_LOCK:
        bank = _load_waypoint_bank()
        bank["pose_variables"][name] = [float(v) for v in values_m_rad]
        _write_waypoint_bank(bank)


def _save_joint_to_waypoint_bank(name: str, values_rad: list[float]) -> None:
    """Create or update one joint waypoint entry in waypoints_lookup_table.json."""
    with _WAYPOINT_BANK_LOCK:
        bank = _load_waypoint_bank()
        bank["joint_variables"][name] = [float(v) for v in values_rad]
        _write_waypoint_bank(bank)


def _fk_points(q_rad: list[float]) -> list[tuple[float, float, float]]:
    """Base-frame position of each joint frame origin, ending at the tool."""
    T = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    points = []
    for theta, a, d, alpha in zip(q_rad, _DH_A, _DH_D, _DH_ALPHA):
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        step = [[ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0]]
        T = [[sum(T[i][k] * step[k][j] for k in range(4)) for j in range(4)]
             for i in range(4)]
        points.append((T[0][3], T[1][3], T[2][3]))
    return points


def _checked_joint_target(joint_angles_deg: list[float]) -> list[float]:
    """Validate a joint target (count + limits) and return it in radians."""
    if len(joint_angles_deg) != len(JOINT_NAMES):
        raise ValueError(
            f"Expected {len(JOINT_NAMES)} joint angles "
            f"({', '.join(JOINT_NAMES)}), got {len(joint_angles_deg)}."
        )
    target_rad = [math.radians(a) for a in joint_angles_deg]
    for name, angle_rad, angle_deg in zip(JOINT_NAMES, target_rad, joint_angles_deg):
        if abs(angle_rad) > JOINT_LIMIT:
            raise ValueError(
                f"Target {angle_deg} deg for {name} exceeds the +/-"
                f"{math.degrees(JOINT_LIMIT):.0f} deg limit. Choose a smaller angle."
            )
    return target_rad


def _check_joint_dynamics(speed: float, acceleration: float) -> None:
    """Reject joint speed / acceleration above the safety caps."""
    if not 0 < speed <= MAX_JOINT_SPEED:
        raise ValueError(
            f"Joint speed {speed} rad/s is outside the allowed range "
            f"(0, {MAX_JOINT_SPEED}]. Use a slower speed."
        )
    if not 0 < acceleration <= MAX_JOINT_ACCEL:
        raise ValueError(
            f"Joint acceleration {acceleration} rad/s^2 is outside the allowed "
            f"range (0, {MAX_JOINT_ACCEL}]. Use a gentler acceleration."
        )


def _check_tcp_dynamics(speed: float, acceleration: float) -> None:
    """Reject TCP speed / acceleration above the safety caps."""
    if not 0 < speed <= MAX_TCP_SPEED:
        raise ValueError(
            f"TCP speed {speed} m/s is outside the allowed range "
            f"(0, {MAX_TCP_SPEED}]. Use a slower speed."
        )
    if not 0 < acceleration <= MAX_TCP_ACCEL:
        raise ValueError(
            f"TCP acceleration {acceleration} m/s^2 is outside the allowed "
            f"range (0, {MAX_TCP_ACCEL}]. Use a gentler acceleration."
        )


def _check_box(x: float, y: float, z: float, what: str = "the target") -> None:
    """Reject a tool position outside the safe workspace box."""
    for axis, value in (("x", x), ("y", y), ("z", z)):
        lo, hi = WORKSPACE_M[axis]
        if not lo <= value <= hi:
            raise ValueError(
                f"Unsafe command: {what} puts the tool at {axis}="
                f"{value:.3f} m, outside the safe workspace "
                f"({axis} between {lo} and {hi} m)."
            )


def _check_reach(x: float, y: float, z: float,
                 what: str = "the target") -> None:
    """Reject a Cartesian target the arm can never physically reach.

    Only meaningful for Cartesian requests (move_linear): a joint target is
    reachable by construction, its FK pose never needs this test.
    """
    shoulder_z = _DH_D[0]
    if math.sqrt(x * x + y * y + (z - shoulder_z) ** 2) > REACH_M:
        raise ValueError(
            f"Unsafe command: {what} at ({x:.3f}, {y:.3f}, {z:.3f}) m is "
            f"beyond the robot's {REACH_M} m reach; it could never get there."
        )


def _joint_move_timeout(deltas_rad: list[float], speed: float) -> float:
    """A completion budget that adapts to how long the move should take.

    A fixed 20 s cap silently fails slow-but-valid commands (180 deg at
    0.1 rad/s needs ~31 s), so scale with travel time and keep 20 s as the
    floor for short hops.
    """
    travel_s = max((abs(d) for d in deltas_rad), default=0.0) / speed
    return max(20.0, 2.0 * travel_s + 5.0)


def _check_linear_start(q_rad: list[float]) -> None:
    """Reject a Cartesian move that starts from a singular configuration.

    Verified against the simulator: commanding movel from a wrist singularity
    (wrist2 near 0 or 180 deg, joints 4 and 6 aligned) makes the controller's
    inverse kinematics pick a different wrist solution than the robot's actual
    one -- the first setpoint jumps discontinuously and the safety system
    answers with a protective stop (C204A1). Fully stretched (elbow near 0)
    is equally degenerate: linear motion crawls or faults. Rejecting up front
    turns a protective stop into an error the LLM can plan around.
    """
    wrist2_deg = math.degrees(q_rad[4])
    if abs(math.sin(q_rad[4])) < 0.05:
        raise ValueError(
            f"Cannot move linearly: wrist2 is at {wrist2_deg:.1f} deg, a "
            "wrist singularity (triggers a protective stop). First make a "
            "joint move that bends wrist2 away from 0/180 (for example "
            "move_joints_relative with wrist2 -30), then retry."
        )
    elbow_deg = math.degrees(q_rad[2])
    if abs(math.sin(q_rad[2])) < 0.05:
        raise ValueError(
            f"Cannot move linearly: the elbow is at {elbow_deg:.1f} deg, so "
            "the arm is fully stretched (a singularity where linear motion "
            "stalls). First bend the elbow with a joint move (for example "
            "move_joints_relative with elbow 30 or -30), then retry."
        )


def _check_pose_safe(target_rad: list[float], what: str = "the target") -> None:
    """Forward-kinematics check: the whole arm must stay in safe space."""
    points = _fk_points(target_rad)
    x, y, z = points[-1]
    _check_box(x, y, z, what)
    for name, (_, _, pz) in zip(_FRAME_NAMES[1:-1], points[1:-1]):
        if pz < ARM_CLEARANCE_M:
            raise ValueError(
                f"Unsafe command: {what} would put the {name} at z="
                f"{pz:.3f} m, under the {ARM_CLEARANCE_M} m arm clearance -- "
                "the link would touch the table. Keep the arm higher."
            )


def _tool_z_axis(rotation_rad: list[float]) -> tuple[float, float, float]:
    """The tool z-axis in the base frame, from an axis-angle rotation vector."""
    rx, ry, rz = rotation_rad
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-9:
        return (0.0, 0.0, 1.0)
    kx, ky, kz = rx / angle, ry / angle, rz / angle
    c, s = math.cos(angle), math.sin(angle)
    return (kx * kz * (1 - c) + ky * s,
            ky * kz * (1 - c) - kx * s,
            c + kz * kz * (1 - c))


def _check_flange_above_table(position: list[float],
                              rotation_rad: list[float]) -> None:
    """Reject a TCP pose whose ORIENTATION puts the wrist under the table.

    The box check sees only the tool point; with the tool z-axis pointing up,
    the wrist flange sits d6 (the active model's wrist offset, _DH_D[5]) BELOW the TCP and can pass through the
    table while the TCP itself stays legal. The flange point needs no inverse
    kinematics: it is TCP - d6 * (tool z-axis in base frame).
    """
    zx, zy, zz = _tool_z_axis(rotation_rad)
    d6 = _DH_D[5]
    fz = position[2] - d6 * zz
    if fz < ARM_CLEARANCE_M:
        raise ValueError(
            f"Unsafe command: with that orientation the wrist flange would "
            f"sit at z={fz:.3f} m, under the {ARM_CLEARANCE_M} m arm "
            "clearance. Aim the tool differently or raise the target."
        )


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
    ([0, -90, 90, -90, -90, 0] degrees: upper arm vertical, forearm
    horizontal, tool pointing straight down, so the wrist camera overlooks
    the table) -- "move the robot home" is a call with no arguments.

    The move blocks until the robot arrives (the wait budget adapts to the
    distance and speed), then returns the new robot state (so you can observe
    where it ended up before deciding the next step).

    Args:
        joint_angles_deg: Six target angles in degrees, base..wrist3. Defaults to
            the home pose when omitted.
        speed: Joint speed (rad/s).
        acceleration: Joint acceleration (rad/s^2).

    Returns:
        A dict with the target that was commanded and the resulting state:
        ``joints_deg`` (per-joint angles), ``tcp_pose_m_rad``
        [x, y, z, rx, ry, rz] in metres and radians, and ``robot_mode``.
        For backward compatibility, ``tcp_pose`` is kept as an alias.

    Raises:
        ValueError: Wrong number of angles, a target past the joint limit,
            speed/acceleration above the safety caps, or a pose the safety
            layer rejects (arm would leave the workspace or hit the table).
        RuntimeError: The robot is not powered on (enable it in the UI first).
        TimeoutError: The move did not complete (blocked, or a protective
            stop -- check get_robot_state's safety_status).
    """
    # 1. Validate inputs.
    if joint_angles_deg is None:
        joint_angles_deg = list(HOME_DEG)

    # 2. Convert the request (degrees) to the robot's units (radians).
    target_rad = _checked_joint_target(joint_angles_deg)

    # 3. Check feasibility: joint limits (above), speed caps, and the
    #    forward-kinematics workspace check (Diamond safety layer).
    _check_joint_dynamics(speed, acceleration)
    _check_pose_safe(target_rad)

    # 4. Execute (blocks until reached), then report the new state.
    current = robot.get_state().q_rad
    timeout = _joint_move_timeout(
        [t - c for t, c in zip(target_rad, current)], speed)
    state = robot.move_joint(target_rad, speed, acceleration,
                             timeout_s=timeout)
    return {
        "status": "reached",
        "target_deg": {n: round(a, 1) for n, a in zip(JOINT_NAMES, joint_angles_deg)},
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
        "tcp_pose": [round(v, 4) for v in state.tcp_pose],
        "robot_mode": state.robot_mode,
    }


# =========================================================================== #
# SILVER TOOL  --  read-only state observation. No inputs to validate and no
# limits to check, so of the four steps only "execute + report" applies: read
# one RTDE snapshot and report it in human units with decoded mode names.
# =========================================================================== #
@mcp.tool
def get_robot_state() -> dict:
    """Read the robot's current state without moving it.

    Call this to observe before acting or to check the result of a move: where
    the joints and tool are, whether the robot is powered on and ready, or why
    a move failed (for example a protective stop). It is always safe to call --
    it only reads state and never commands motion.

    Returns:
        A dict with the full snapshot:
        ``joints_deg``: per-joint angles in degrees, base..wrist3.
        ``tcp_pose_m_rad``: tool pose [x, y, z, rx, ry, rz] in the base frame,
            metres and radians (axis-angle rotation).
        ``robot_mode`` / ``robot_mode_name``: controller mode; 7 (RUNNING)
            means powered on with brakes released.
        ``safety_status`` / ``safety_status_name``: 1 (NORMAL) is nominal; any
            other value (for example PROTECTIVE_STOP) means the robot will not
            move until it is recovered in the PolyScope UI.
        ``ready_to_move``: True only when the mode is RUNNING and the safety
            status is NORMAL, so motion commands will be accepted.
        ``moving``: True while any joint is still in motion.
        ``gripper_closed``: the COMMANDED gripper state (a Robotiq holding a
            slim object sits mid-travel but still reads closed).
        ``gripper_holding``: True when the Robotiq's fingers stopped on an
            object, False when they reached the commanded position freely,
            None when no Robotiq is connected (no contact sensing).
    """
    # 4. Execute (a read), then report in human units.
    state = robot.get_state()
    # Both backends report the COMMANDED state, so the meaning of
    # gripper_closed does not change with the hardware: on a Robotiq that is
    # the requested position (PRE), not the actual one -- fingers holding a
    # 30 mm cup sit mid-travel and must still read as "closed". Whether
    # something is actually held is a separate flag. Only an
    # already-connected Robotiq is consulted, so the state-read path never
    # pays for a probe.
    gripper_closed = bool(state.digital_out >> GRIPPER_PIN & 1)
    gripper_holding: bool | None = None
    robotiq = _ROBOTIQ
    if robotiq is not None:
        try:
            gripper_closed = robotiq.get("PRE") > 128
            gripper_holding = robotiq.get("OBJ") in (1, 2)
        except (OSError, GripperError):
            _drop_robotiq(robotiq)
            gripper_holding = None
    return {
        "joints_deg": {n: round(math.degrees(q), 2)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
        "robot_mode": state.robot_mode,
        "robot_mode_name": ROBOT_MODE_NAMES.get(
            state.robot_mode, f"UNKNOWN({state.robot_mode})"),
        "safety_status": state.safety_status,
        "safety_status_name": SAFETY_STATUS_NAMES.get(
            state.safety_status, f"UNKNOWN({state.safety_status})"),
        "ready_to_move": (state.robot_mode == ROBOT_MODE_RUNNING
                          and state.safety_status == 1),
        "moving": state.is_moving,
        "gripper_closed": gripper_closed,
        "gripper_holding": gripper_holding,
    }


# =========================================================================== #
# GOLD TOOLS  --  richer motion. All three follow the same four-step shape and
# funnel through the Diamond safety layer before anything moves.
# =========================================================================== #
@mcp.tool
def move_joints_relative(
    delta_deg: list[float],
    speed: float = DEFAULT_SPEED,
    acceleration: float = DEFAULT_ACCEL,
) -> dict:
    """Rotate joints by an amount, relative to where they are now.

    Use this for "rotate the base 45 degrees", "nudge the elbow -10 degrees":
    adjustments relative to the current pose, when the caller does not know or
    care about absolute angles. Give six deltas in degrees, ordered base,
    shoulder, elbow, wrist1, wrist2, wrist3; use 0 for joints that stay put.

    The move blocks until the robot arrives, then returns the new state.

    Args:
        delta_deg: Six angle changes in degrees, base..wrist3 (0 = no change).
        speed: Joint speed (rad/s).
        acceleration: Joint acceleration (rad/s^2).

    Returns:
        A dict with ``target_deg`` (the absolute pose commanded), the
        resulting ``joints_deg``, ``tcp_pose_m_rad``, and ``robot_mode``.
        For backward compatibility, ``tcp_pose`` is kept as an alias.

    Raises:
        ValueError: Wrong number of deltas, a resulting angle past the joint
            limit, capped speed/acceleration exceeded, or an unsafe pose.
        RuntimeError: The robot is not powered on.
        TimeoutError: The move did not complete (blocked, or a protective
            stop -- check get_robot_state's safety_status).
    """
    # 1. Validate the shape of the request.
    if len(delta_deg) != len(JOINT_NAMES):
        raise ValueError(
            f"Expected {len(JOINT_NAMES)} joint deltas "
            f"({', '.join(JOINT_NAMES)}), got {len(delta_deg)}."
        )

    # 2. Convert: read the current pose and form the absolute target.
    current_deg = [math.degrees(q) for q in robot.get_state().q_rad]
    target_deg = [c + d for c, d in zip(current_deg, delta_deg)]
    try:
        target_rad = _checked_joint_target(target_deg)
    except ValueError as exc:
        raise ValueError(
            f"After applying the deltas to the current pose: {exc}"
        ) from exc

    # 3. Check feasibility: speed caps and the FK workspace check.
    _check_joint_dynamics(speed, acceleration)
    _check_pose_safe(target_rad, what="the relative move's end pose")

    # 4. Execute, then report.
    timeout = _joint_move_timeout(
        [math.radians(d) for d in delta_deg], speed)
    state = robot.move_joint(target_rad, speed, acceleration,
                             timeout_s=timeout)
    return {
        "status": "reached",
        "target_deg": {n: round(a, 1) for n, a in zip(JOINT_NAMES, target_deg)},
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
        "tcp_pose": [round(v, 4) for v in state.tcp_pose],
        "robot_mode": state.robot_mode,
    }


@mcp.tool
def move_linear(
    position_m: list[float],
    rotation_rad: list[float] | None = None,
    speed: float = 0.25,
    acceleration: float = 1.2,
) -> dict:
    """Move the TOOL TIP in a straight line to a Cartesian position.

    Use this when the path matters, not just the destination: approaching,
    inserting, drawing -- the tool tip travels a straight line in space
    (a joint move would sweep an arc). Position is [x, y, z] in metres in the
    robot's base frame; for scale, the upright HOME pose puts the tool at
    z = 1.48 m on the UR10e simulator and z = 1.08 m on a UR5e.

    Caution: linear motion is impossible from a singular configuration and
    such calls are rejected up front -- if the arm is fully stretched (elbow
    near 0, like HOME) or the wrist is aligned (wrist2 near 0 or 180, also
    like HOME), first make a joint move to a bent pose (bend the elbow and
    wrist2, for example with move_joints_relative), then move linearly.

    The move blocks until the tool arrives, then returns the new state.

    Args:
        position_m: Target tool position [x, y, z] in metres, base frame.
        rotation_rad: Optional tool orientation [rx, ry, rz] (axis-angle,
            radians). Omit to keep the current orientation.
        speed: TOOL speed in m/s (Cartesian, not joint units).
        acceleration: Tool acceleration in m/s^2.

    Returns:
        A dict with the commanded ``target_pose_m_rad``, the resulting
        ``tcp_pose_m_rad`` and ``joints_deg``, and ``distance_moved_m``.

    Raises:
        ValueError: Malformed position/rotation, capped speed/acceleration
            exceeded, a target outside the safe workspace or reach, an
            orientation that would put the wrist under the table, or a
            singular start configuration.
        RuntimeError: The robot is not powered on.
        TimeoutError: The move did not complete (blocked, or a protective
            stop -- check get_robot_state's safety_status).
    """
    # 1. Validate the shape of the request.
    if len(position_m) != 3:
        raise ValueError(
            f"Expected position_m as [x, y, z] in metres, got "
            f"{len(position_m)} values."
        )
    if rotation_rad is not None and len(rotation_rad) != 3:
        raise ValueError(
            f"Expected rotation_rad as [rx, ry, rz] in radians, got "
            f"{len(rotation_rad)} values."
        )

    # 2. Convert: fill the orientation from the current pose if omitted.
    before = robot.get_state()
    rot = list(rotation_rad) if rotation_rad is not None else before.tcp_pose[3:]
    target_pose = [float(v) for v in position_m] + [float(v) for v in rot]

    # 3. Check feasibility: TCP speed caps, workspace/reach, the wrist-flange
    #    orientation check, and that the start pose is not singular.
    _check_tcp_dynamics(speed, acceleration)
    _check_box(*target_pose[:3], what="the linear target")
    _check_reach(*target_pose[:3], what="the linear target")
    _check_flange_above_table(target_pose[:3], target_pose[3:])
    _check_linear_start(before.q_rad)

    # 4. Execute, then report. A timeout here usually means the start pose
    #    was singular (fully stretched), so tell the model how to recover.
    timeout = max(20.0,
                  2.0 * math.dist(before.tcp_pose[:3], target_pose[:3]) / speed
                  + 5.0)
    try:
        state = robot.move_linear(target_pose, speed, acceleration,
                                  timeout_s=timeout)
    except TimeoutError as exc:
        raise RuntimeError(
            f"{exc} If the arm was fully stretched (elbow near 0, like home), "
            "linear motion is near-singular and crawls: make a joint move to "
            "a bent pose first (e.g. bend the elbow 30-60 deg), then retry."
        ) from exc
    moved = math.dist(before.tcp_pose[:3], state.tcp_pose[:3])
    return {
        "status": "reached",
        "target_pose_m_rad": [round(v, 4) for v in target_pose],
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "distance_moved_m": round(moved, 4),
        "robot_mode": state.robot_mode,
    }


@mcp.tool
def run_trajectory(
    waypoints_deg: list[list[float]],
    speed: float = DEFAULT_SPEED,
    acceleration: float = DEFAULT_ACCEL,
    blend_m: float = 0.02,
) -> dict:
    """Run a multi-waypoint joint trajectory as one continuous motion.

    Use this for multi-step moves: the whole sequence is sent as a single
    program and every waypoint except the last gets a blend radius, so the
    robot rounds the corners smoothly instead of stopping at each one. State
    is sampled throughout and returned as a motion log, so the caller can see
    how the move actually unfolded, not just where it ended.

    Args:
        waypoints_deg: 2 to 20 waypoints, each six joint angles in degrees
            (base..wrist3), visited in order.
        speed: Joint speed (rad/s).
        acceleration: Joint acceleration (rad/s^2).
        blend_m: Corner-rounding radius in metres (0 = stop at each waypoint,
            max 0.1). Applied to every waypoint except the last.

    Returns:
        A dict with ``waypoints`` (count), ``duration_s``, the sampled
        ``motion_log`` (time-stamped joint angles at ~7 Hz, trimmed to 25
        entries), ``peak_joint_speed_rad_s`` observed, and the final
        ``joints_deg`` / ``tcp_pose_m_rad``. For backward compatibility,
        ``tcp_pose`` is kept as an alias.

    Raises:
        ValueError: Fewer than 2 or more than 20 waypoints, a malformed or
            limit-violating waypoint, an unsafe pose anywhere along the
            sequence, capped speed/acceleration exceeded, or a bad blend.
        RuntimeError: The robot is not powered on.
        TimeoutError: The path did not complete (blocked, or a protective
            stop -- check get_robot_state's safety_status).
    """
    return _run_trajectory_impl(
        waypoints_deg=waypoints_deg,
        speed=speed,
        acceleration=acceleration,
        blend_m=blend_m,
    )


@mcp.tool
def start_trajectory_job(
    waypoints_deg: list[list[float]],
    speed: float = DEFAULT_SPEED,
    acceleration: float = DEFAULT_ACCEL,
    blend_m: float = 0.02,
) -> dict:
    """Start a trajectory in the background and return a job id.

    Use this for longer moves where the caller wants progress updates while the
    robot is moving. Poll ``get_trajectory_job_status`` with the returned id to
    receive incremental samples and final completion details.

    Args mirror ``run_trajectory``.
    """
    job_id = uuid.uuid4().hex
    job = TrajectoryJob(
        job_id=job_id,
        requested_waypoints=len(waypoints_deg),
    )
    with _TRAJECTORY_JOBS_LOCK:
        _TRAJECTORY_JOBS[job_id] = job

    _start_trajectory_job_runner(
        job=job,
        waypoints_deg=waypoints_deg,
        speed=speed,
        acceleration=acceleration,
        blend_m=blend_m,
    )
    return {
        "status": "running",
        "job_id": job_id,
        "requested_waypoints": len(waypoints_deg),
        "next_call": "get_trajectory_job_status",
    }


@mcp.tool
def get_trajectory_job_status(
    job_id: str,
    from_index: int = 0,
    limit: int = 25,
) -> dict:
    """Read status and incremental progress for a started trajectory job.

    Args:
        job_id: Identifier returned by ``start_trajectory_job``.
        from_index: First unread progress sample index (0-based).
        limit: Max number of progress samples to return (1..200).

    Returns:
        Job status and a chunk of progress samples. Use ``next_index`` in the
        response as ``from_index`` in the next poll.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string.")
    if from_index < 0:
        raise ValueError(f"from_index must be >= 0, got {from_index}.")
    if not 1 <= limit <= 200:
        raise ValueError(f"limit must be in [1, 200], got {limit}.")

    job = _get_trajectory_job_or_raise(job_id.strip())
    with _TRAJECTORY_JOBS_LOCK:
        total_samples = len(job.progress_samples)
        start = min(from_index, total_samples)
        end = min(total_samples, start + limit)
        chunk = list(job.progress_samples[start:end])
        response = {
            "job_id": job.job_id,
            "status": job.status,
            "created_at_monotonic_s": round(job.created_at_s, 3),
            "updated_at_monotonic_s": round(job.updated_at_s, 3),
            "requested_waypoints": job.requested_waypoints,
            "sample_count": total_samples,
            "next_index": end,
            "progress_samples": chunk,
        }
        if job.error is not None:
            response["error"] = job.error
        if job.result is not None:
            response["result"] = job.result
    return response

_FREEDRIVE_ACTIVE = False


def _freedrive_script(
    free_axes: list[int] | None = None,
    feature: list[float] | None = None,
) -> str:
    script = "  freedrive_mode"
    if free_axes is not None or feature is not None:
        args = []
        if free_axes is not None:
            if len(free_axes) != 6:
                raise ValueError(
                    f"free_axes must have 6 entries, got {len(free_axes)}.")
            args.append(
                "freeAxes=[" + ",".join(str(int(v)) for v in free_axes) + "]"
            )
        if feature is not None:
            if len(feature) not in (5, 6):
                raise ValueError(
                    "feature must have 5 or 6 values, got "
                    f"{len(feature)}.")
            args.append(
                "feature=p[" + ",".join(f"{float(v):.6f}" for v in feature) + "]"
            )
        script += "(" + ", ".join(args) + ")"
    return script + "\n"



def _freedrive_worker(
    free_axes: list[int] | None = None,
    feature: list[float] | None = None,
):
    global _FREEDRIVE_ACTIVE

    try:
        script = _freedrive_script(free_axes=free_axes, feature=feature)
        # This must stay alive long enough for freedrive to remain active.
        # If your API supports async execution, prefer that.
        robot.run_script(script + f"  sleep({float(300):.3f})\n", timeout_s=300.0)
    finally:
        _FREEDRIVE_ACTIVE = False

# =========================================================================== #
# DIAMOND TOOLS  --  gripper. On the real cell a Robotiq 2F is driven through
# its URCap command server (port 63352 on the controller) with real position,
# speed, force, and object-detection feedback. The simulator has no gripper,
# so there the tools fall back to the digital-output convention and say so.
# =========================================================================== #
@mcp.tool
def check_gripper() -> dict:
    """Detect the gripper and report its state. Never moves anything.

    Call this after (re)connecting hardware, and before the first grasp of a
    session. It probes the Robotiq URCap port on the controller: when a
    Robotiq answers, the report includes activation, position, and fault
    state, plus the next step to take (``activate_gripper`` if it is not
    yet activated). When nothing answers, gripper commands fall back to
    driving digital output pin 0 -- the simulator convention, with no
    force or object feedback.

    Returns:
        A dict with ``connected`` (a Robotiq answered), ``backend``
        ("robotiq" or "digital-out-fallback"), and, for a live Robotiq,
        its full status and a ``next_step`` hint.
    """
    gripper = _robotiq()
    if gripper is None:
        report = {
            "connected": False,
            "backend": "digital-out-fallback",
            "detail": (
                f"No Robotiq URCap answering on {robot.host}:63352. On the "
                "simulator this is normal. On the real robot the URCap is "
                "the only network path to the gripper: install/enable the "
                "Robotiq Grippers URCap on the pendant (Settings > System > "
                "URCaps), then call this again."),
        }
        # An unpowered tool connector is the other common reason a mounted
        # gripper is silent, and it is worth reporting before the operator
        # goes hunting through URCap menus.
        try:
            power = robot.get_tool_power()
            report["tool_power"] = {
                "voltage_v": power.voltage_v,
                "current_a": round(power.current_a, 3),
                "interface": power.tool_mode_name,
                "powered": power.powered,
            }
            if not power.powered:
                report["next_step"] = (
                    "The tool connector is supplying 0 V, so a mounted "
                    "gripper has no power at all. Set Tool Output Voltage to "
                    "24 V (pendant: Installation > General > Tool I/O, or "
                    "the set_tool_voltage tool) before debugging further.")
        except Exception:  # noqa: BLE001 - diagnostics must never fail the tool
            pass
        return report
    try:
        status = gripper.status()
    except (OSError, GripperError) as exc:
        _drop_robotiq(gripper)
        raise RuntimeError(
            f"A gripper port answered but the status read failed: {exc}. "
            "Retry check_gripper; if it keeps failing, power-cycle the "
            "gripper.") from exc
    report = {"connected": True, "backend": "robotiq", **status.as_dict()}
    # A hardware fault outranks "not activated": an unplugged gripper is
    # BOTH, and telling the operator to activate it just wastes their time.
    if status.fault in HARDWARE_FAULTS:
        report["next_step"] = (
            f"Gripper reports {status.fault_name}: the URCap is running but "
            "the gripper itself is not answering. Check the tool cable and "
            "that the tool connector supplies 24 V, then call this again.")
    elif status.fault:
        report["next_step"] = (
            f"Gripper reports fault {status.fault_name}; power-cycle the "
            "gripper or re-run activate_gripper.")
    elif not status.activated:
        report["next_step"] = (
            "Gripper is not activated. Clear the space in front of the "
            "fingers (they sweep full travel to self-calibrate) and call "
            "activate_gripper.")
    return report


@mcp.tool
def activate_gripper() -> dict:
    """Run the Robotiq activation cycle and wait until it is ready.

    A Robotiq must be activated once after each power-up before it accepts
    motion commands. Activation sweeps the fingers through their FULL travel
    to self-calibrate -- make sure nothing (and nobody) is between them.
    Already-activated grippers return immediately, so this is always safe
    to call before a grasping task.

    Returns:
        The gripper status dict after activation (``activated`` true).

    Raises:
        ValueError: No Robotiq is connected (nothing to activate).
        RuntimeError: Activation failed or timed out.
    """
    gripper = _robotiq()
    if gripper is None:
        raise ValueError(
            f"No Robotiq URCap answering on {robot.host}:63352 -- nothing "
            "to activate. Run check_gripper for diagnostics.")
    try:
        status = gripper.activate()
    except GripperError as exc:
        raise RuntimeError(f"Gripper activation failed: {exc}") from exc
    except OSError as exc:
        _drop_robotiq(gripper)
        raise RuntimeError(f"Gripper activation failed: {exc}") from exc
    return {"backend": "robotiq", **status.as_dict()}


@mcp.tool
def set_gripper_position(position_pct: float, speed_pct: float = 50.0,
                         force_pct: float = 25.0) -> dict:
    """Move the gripper fingers to a width, with speed and force control.

    Use this instead of ``set_gripper`` when the opening matters: pre-opening
    just wider than an object, or closing gently around something soft.
    Blocks until the fingers arrive or stop on contact, so the returned
    ``object_detected`` is trustworthy. Defaults are gentle on purpose --
    this cell works alongside people.

    Args:
        position_pct: Target closure, 0 = fully open .. 100 = fully closed.
        speed_pct: Finger speed, 0 (slowest) .. 100 (fastest).
        force_pct: Grip force limit, 0 (lightest) .. 100 (strongest).

    Returns:
        The gripper status dict: ``position_pct`` and ``opening_mm``
        actually reached, ``object_detected`` (fingers stopped early on
        something), and the ``model`` the mm figure assumes.

    Raises:
        ValueError: Argument out of range, no Robotiq connected, or the
            gripper is not activated yet.
        RuntimeError: The gripper stopped answering mid-move.
    """
    # 1. Validate ranges before touching hardware.
    for name, value in (("position_pct", position_pct),
                        ("speed_pct", speed_pct), ("force_pct", force_pct)):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be within 0..100, got {value}.")
    gripper = _robotiq()
    if gripper is None:
        raise ValueError(
            "Position control needs a Robotiq (none is connected). For the "
            "simulator's on/off gripper use set_gripper instead.")
    # 2. Convert percentages to the gripper's 0..255 registers.
    try:
        status = gripper.move(
            position=round(position_pct / 100.0 * 255),
            speed=round(speed_pct / 100.0 * 255),
            force=round(force_pct / 100.0 * 255),
        )
    except GripperError as exc:
        raise ValueError(str(exc)) from exc
    except OSError as exc:
        _drop_robotiq(gripper)
        raise RuntimeError(
            f"Lost the gripper connection mid-move: {exc}") from exc
    return {"backend": "robotiq", **status.as_dict()}


@mcp.tool
def set_gripper(closed: bool) -> dict:
    """Open or close the gripper.

    Use ``closed=true`` to grip (before lifting an object) and
    ``closed=false`` to release. With a Robotiq connected this closes at a
    gentle default force and reports ``object_detected``: true means the
    fingers stopped on something, so the grasp holds. False after a close
    means they ran to the fully-closed stop -- usually a MISS, though a very
    thin or soft item can also let them reach it, so confirm with
    ``opening_mm`` (0 mm means nothing is between the fingers).
    Without a Robotiq (the simulator) it falls back to driving the gripper
    digital output and confirming it over RTDE. For finer control of width,
    speed, or force use ``set_gripper_position``.

    Args:
        closed: True to close the gripper, False to open it.

    Returns:
        A dict with ``gripper`` ("closed" or "open") and ``backend``
        ("robotiq": full status incl. ``object_detected``;
        "digital-out-fallback": the ``pin`` and confirmed ``pin_state``).

    Raises:
        ValueError: The Robotiq is present but not activated yet.
        RuntimeError: The robot is not powered on (fallback path), or the
            gripper stopped answering.
        TimeoutError: The controller did not confirm the output change.
    """
    gripper = _robotiq()
    if gripper is not None:
        try:
            status = gripper.move(position=255 if closed else 0)
        except GripperError as exc:
            raise ValueError(str(exc)) from exc
        except OSError as exc:
            _drop_robotiq(gripper)
            raise RuntimeError(
                f"Lost the gripper connection mid-move: {exc}") from exc
        return {
            "gripper": "closed" if closed else "open",
            "backend": "robotiq",
            **status.as_dict(),
        }
    # Fallback: the simulator's digital-output convention, confirmed by
    # reading the IO state back from the controller.
    state = robot.set_digital_out(GRIPPER_PIN, closed)
    confirmed = bool(state.digital_out >> GRIPPER_PIN & 1)
    return {
        "gripper": "closed" if confirmed else "open",
        "backend": "digital-out-fallback",
        "pin": GRIPPER_PIN,
        "pin_state": confirmed,
    }


# =========================================================================== #
# COMMISSIONING TOOLS  --  setup, IO checks, sensor prep, and manual guidance.
# These wrap common URScript commissioning calls as MCP tools.
# =========================================================================== #
@mcp.tool
def set_tcp(tcp_pose_m_rad: list[float]) -> dict:
    """Set the active Tool Center Point using a URScript TCP pose.

    Args:
        tcp_pose_m_rad: TCP pose [x, y, z, rx, ry, rz] in metres/radians.

    Returns:
        A dict confirming the TCP that was applied.
    """
    if len(tcp_pose_m_rad) != 6:
        raise ValueError(
            f"Expected tcp_pose_m_rad with 6 values [x,y,z,rx,ry,rz], "
            f"got {len(tcp_pose_m_rad)}."
        )
    tcp = [float(v) for v in tcp_pose_m_rad]
    body = "  set_tcp(p[" + ", ".join(f"{v:.6f}" for v in tcp) + "])\n"
    robot.run_script(body)
    return {
        "status": "applied",
        "tcp_pose_m_rad": [round(v, 6) for v in tcp],
    }


# Mass and CoG are limited by the selected UR_MODEL's rated payload.
@mcp.tool
def set_payload(mass_kg: float, cog_m: list[float] | None = None) -> dict:
    """Set payload mass and center-of-gravity for commissioning checks.

    Args:
        mass_kg: Payload mass in kilograms.
        cog_m: Center of gravity [x, y, z] in metres relative to the tool.
            Defaults to [0, 0, 0].
    Returns:
        A dict confirming the payload mass and CoG that was applied.
    Raises:
        ValueError: Negative mass, mass above the selected model's rated
            payload, malformed CoG, or a mass/CoG-offset combination beyond
            the model's rating.
    """
    limit = _MODEL["payload_kg"]
    if mass_kg < 0:
        raise ValueError(f"Payload mass must be non-negative, got {mass_kg} kg.")
    if mass_kg > limit:
        raise ValueError(
            f"Payload mass on a {UR_MODEL} must be at most {limit} kg, "
            f"got {mass_kg} kg.")
    cog = [0.0, 0.0, 0.0] if cog_m is None else [float(v) for v in cog_m]
    if len(cog) != 3:
        raise ValueError(f"Expected cog_m as [x, y, z], got {len(cog)} values.")
    offset_distance = math.hypot(*cog)
    print(f"Payload mass: {mass_kg} kg, CoG offset distance: {offset_distance:.3f} m")
    if (mass_kg * offset_distance) > limit:
        raise ValueError(
            f"Invalid CoG for a {UR_MODEL}: {mass_kg} kg cannot be offset "
            f"by {offset_distance:.3f} m.")
    body = (
        f"  set_payload({float(mass_kg):.6f}, "
        f"[{cog[0]:.6f}, {cog[1]:.6f}, {cog[2]:.6f}])\n"
    )
    robot.run_script(body)
    return {
        "status": "applied",
        "mass_kg": round(float(mass_kg), 6),
        "cog_m": [round(v, 6) for v in cog],
    }


@mcp.tool
def set_payload_mass(mass_kg: float) -> dict:
    """Set payload mass only (CoG unchanged by controller defaults)."""
    if mass_kg < 0:
        raise ValueError(f"Payload mass must be non-negative, got {mass_kg} kg.")
    if mass_kg > _MODEL["payload_kg"]:
        raise ValueError(
            f"Payload mass on a {UR_MODEL} must be at most "
            f"{_MODEL['payload_kg']} kg, got {mass_kg} kg.")
    body = f"  set_payload_mass({float(mass_kg):.6f})\n"
    robot.run_script(body)
    return {
        "status": "applied",
        "mass_kg": round(float(mass_kg), 6),
    }


@mcp.tool
def set_gravity(gravity_m_s2: list[float]) -> dict:
    """Set gravity vector [gx, gy, gz] for non-floor robot mounting.

    Typical floor-mounted value is [0, 0, -9.82].
    """
    if len(gravity_m_s2) != 3:
        raise ValueError(
            f"Expected gravity_m_s2 as [gx, gy, gz], got {len(gravity_m_s2)} values."
        )
    g = [float(v) for v in gravity_m_s2]
    magnitude = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
    if not 7.0 <= magnitude <= 12.0:
        raise ValueError(
            f"Gravity magnitude {magnitude:.3f} m/s^2 looks invalid; expected "
            "roughly Earth's gravity (about 9.82 m/s^2)."
        )
    body = f"  set_gravity([{g[0]:.6f}, {g[1]:.6f}, {g[2]:.6f}])\n"
    robot.run_script(body)
    return {
        "status": "applied",
        "gravity_m_s2": [round(v, 6) for v in g],
        "magnitude_m_s2": round(magnitude, 6),
    }


@mcp.tool
def is_within_safety_limits(
    pose_m_rad: list[float] | None = None,
    qnear_deg: list[float] | None = None,
) -> dict:
    """Check whether a candidate pose/joint target passes server safety checks.

    This mirrors commissioning intent of URScript's is_within_safety_limits,
    but evaluates against this server's configured safety layer.

    Args:
        pose_m_rad: Optional candidate [x, y, z, rx, ry, rz].
        qnear_deg: Optional candidate joints [base..wrist3] in degrees.

    Returns:
        A dict with `within_limits` and a plain-language `reason`.
    """
    if pose_m_rad is None and qnear_deg is None:
        raise ValueError("Provide at least one of pose_m_rad or qnear_deg.")
    if pose_m_rad is not None and len(pose_m_rad) != 6:
        raise ValueError(f"Expected pose_m_rad with 6 values, got {len(pose_m_rad)}.")

    try:
        if qnear_deg is not None:
            target = _checked_joint_target([float(v) for v in qnear_deg])
            _check_pose_safe(target, what="the joint target")

        if pose_m_rad is not None:
            pose = [float(v) for v in pose_m_rad]
            _check_box(*pose[:3], what="the Cartesian target")
            _check_reach(*pose[:3], what="the Cartesian target")
            _check_flange_above_table(pose[:3], pose[3:])
    except ValueError as exc:
        return {
            "within_limits": False,
            "reason": str(exc),
            "checked_pose": pose_m_rad,
            "checked_qnear_deg": qnear_deg,
        }

    return {
        "within_limits": True,
        "reason": "Target is within configured server safety checks.",
        "checked_pose": pose_m_rad,
        "checked_qnear_deg": qnear_deg,
    }


@mcp.tool
def get_digital_in(n: int) -> dict:
    """Read digital input pin n (0..17).
    The digital in and outputs relate to the robots IO pins. 
    IT can read external devices like fixtures that are ready, etc. 
    The digital in pins are read only and can be used to read the state of external devices. 
    The digital out pins can be set to control external devices.
    """
    value = robot.get_digital_in(n)
    return {"pin": n, "value": value}


@mcp.tool
def set_digital_out(n: int, b: bool) -> dict:
    """Set digital output pin n (0..17) and confirm readback.
        The digital in and outputs relate to the robots IO pins. 
    IT can control external devices like lights, etc. 
    """
    state = robot.set_digital_out(n, b)
    value = bool(state.digital_out >> n & 1)
    return {"pin": n, "value": value}


@mcp.tool
def get_tool_digital_in(n: int) -> dict:
    """Read tool digital input pin n (0..1)"""
    value = robot.get_tool_digital_in(n)
    return {"pin": n, "value": value}


@mcp.tool
def set_tool_digital_out(n: int, b: bool) -> dict:
    """Set tool digital output pin n (0..1) and confirm readback."""
    state = robot.set_tool_digital_out(n, b)
    value = bool(state.digital_out >> (16 + n) & 1)
    return {"pin": n, "value": value}


@mcp.tool
def set_tool_voltage(voltage: int) -> dict:
    """Set tool connector voltage to 0 V, 12 V, or 24 V."""
    robot.set_tool_voltage(voltage)
    return {"status": "applied", "voltage_v": voltage}


@mcp.tool
def get_analog_in(n: int) -> dict:
    """Read analog input channel n (0 or 1)."""
    value = robot.get_analog_in(n)
    return {"channel": n, "value": round(value, 6)}


@mcp.tool
def set_analog_out(n: int, f: float) -> dict:
    """Set analog output channel n (0 or 1) to normalized value f in [0,1]."""
    robot.set_analog_out(n, f)
    return {"status": "applied", "channel": n, "value": round(float(f), 6)}



@mcp.tool
def timed_freedrive_mode(
    duration_s: float = 10.0,
    free_axes: list[int] | None = None,
    feature: list[float] | None = None,
) -> dict:
    """Temporarily enable manual hand-guiding in freedrive mode.

    Use this when the operator wants to move the robot by hand for a short
    teaching or repositioning step. The robot remains compliant for the given
    duration, then exits freedrive automatically.

    Args:
        duration_s: Length of the freedrive window in seconds. Typical values are
            5-30 s; must be in [0.5, 120].
        free_axes: Optional six-element list of 0/1 values that limits which
            Cartesian/rotational axes are free. Example: [1, 0, 0, 0, 0, 0]
            means only the X axis is free in the selected feature frame.
        feature: Optional pose-like reference frame used together with
            free_axes. Example: [0.1, 0.0, 0.0, 0.0, 0.785] creates a feature
            frame offset from the current tool frame for the guided motion.

    Returns:
        A dict with ``status`` set to ``completed`` and the robot state at the
        end of the timed freedrive period. The robot should be in normal mode
        again after the timeout, because the script ends freedrive_mode() before
        returning.

    Example URScript equivalent:
        freedrive_mode(freeAxes=[1,0,0,0,0,0], feature=p[0.1,0,0,0,0.785])
    """
    if not 0.5 <= duration_s <= 120.0:
        raise ValueError(
            f"duration_s must be in [0.5, 120.0], got {duration_s}."
        )
    body = _freedrive_script(free_axes=free_axes, feature=feature)
    body += f"  sleep({float(duration_s):.3f})\n"
    body += "  end_freedrive_mode()\n"
    state = robot.run_script(body, timeout_s=max(10.0, float(duration_s) + 5.0))
    return {
        "status": "completed",
        "duration_s": round(float(duration_s), 3),
        "joints_deg": {n: round(math.degrees(q), 2)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
    }

@mcp.tool
def start_freedrive_mode(
    free_axes: list[int] | None = None,
    feature: list[float] | None = None,
) -> dict:
    global _FREEDRIVE_ACTIVE
    """Start manual hand-guiding in freedrive mode until stop_freedrive_mode() is called.

    Use this when the operator needs to reposition the robot by hand without
    resisting motion, such as teaching a target pose, adjusting alignment, or
    moving the tool clear of an obstacle. Freedrive is persistent until explicitly
    ended; other motion commands should not be started while it is active.

    Args:
        free_axes: Optional six-element list of 0/1 values. A 1 means that axis is
            free to move in the selected feature frame; a 0 locks it. Example:
            [1, 0, 0, 0, 0, 0] allows motion only along the X axis.
        feature: Optional pose-like frame definition used together with free_axes.
            It defines the coordinate system in which the free axes are applied.
            Example: [0.1, 0.0, 0.0, 0.0, 0.785] sets a local feature frame.

    Returns:
        A dict with ``status`` set to ``started`` when freedrive begins, or
        ``already_started`` if it is already active. The expected robot behavior
        is that the arm becomes compliant and can be moved by hand until
        stop_freedrive_mode() is called.

    Example URScript equivalent:
        freedrive_mode(freeAxes=[1,0,0,0,0,0], feature=p[0.1,0,0,0,0.785])
    """
    if _FREEDRIVE_ACTIVE:
        return {"status": "already_started", "mode": "freedrive"}
    else:
        thread = threading.Thread(
            target=_freedrive_worker,
            args=(free_axes, feature),
            daemon=True,
        )
        thread.start()
        _FREEDRIVE_ACTIVE = True
    return {"status": "started", "mode": "freedrive"}


@mcp.tool
def stop_freedrive_mode() -> dict:
    global _FREEDRIVE_ACTIVE
    """Stop freedrive for manual guiding. It will stop the freedrive.
    Freedrive means the robot can be moved by hand without motors resisting, for example to teach a pose or to move the arm out of the way. 
    """
    if not _FREEDRIVE_ACTIVE:
        return {"status": "not_active", "mode": "freedrive"}
    try:
        robot.run_script("  end_freedrive_mode()\n", timeout_s=5.0)
        _FREEDRIVE_ACTIVE = False
        return {"status": "stopped", "mode": "normal"}
    except:
        return {"status": "error", "mode": "freedrive", "message": "Failed to stop freedrive mode."}




@mcp.tool
def get_actual_tcp_pose() -> dict:
    """Read current TCP pose [x, y, z, rx, ry, rz] in metres/radians."""
    pose = robot.get_tcp_pose()
    return {"tcp_pose_m_rad": [round(v, 6) for v in pose]}


@mcp.tool
def get_actual_joint_positions() -> dict:
    """Read current joint positions in radians and degrees."""
    q_rad = robot.get_joint_positions()
    return {
        "joints_rad": {n: round(v, 6) for n, v in zip(JOINT_NAMES, q_rad)},
        "joints_deg": {n: round(math.degrees(v), 3) for n, v in zip(JOINT_NAMES, q_rad)},
    }


@mcp.tool
def store_waypoint_pose_on_ur(
    variable_name: str,
    tcp_pose_m_rad: list[float] | None = None,
) -> dict:
    """Store a waypoint-compatible TCP pose variable on the UR controller.

    Use this when the stored value should later be consumed by a PolyScope
    Variable Waypoint. The variable will be written as a UR pose value:
    p[x, y, z, rx, ry, rz].

    Args:
        variable_name: Target UR variable name (e.g. "pick_tcp").
        tcp_pose_m_rad: Optional explicit pose [x,y,z,rx,ry,rz] in m/rad.
            If omitted, stores the current actual TCP pose.

    Returns:
        A dict with storage details and waypoint compatibility flags.

    Notes:
        To keep the value available across restarts and visible in PolyScope,
        create the same name as an Installation variable and use that variable
        in a Variable Waypoint node. The server also caches the expression so
        ``move_to_stored_tcp_waypoint`` can use it immediately in later calls,
        even when the controller program context has restarted.
    """
    name = _validate_ur_variable_name(variable_name)
    if tcp_pose_m_rad is None:
        values = [float(v) for v in robot.get_tcp_pose()]
    else:
        if len(tcp_pose_m_rad) != 6:
            raise ValueError(
                f"Expected tcp_pose_m_rad with 6 values [x,y,z,rx,ry,rz], "
                f"got {len(tcp_pose_m_rad)}."
            )
        values = [float(v) for v in tcp_pose_m_rad]
        _check_reach(*values[:3], what="the pose being stored")
        _check_flange_above_table(values[:3], values[3:])
    robot.define_pose_variable(name, values)
    _save_pose_to_waypoint_bank(name, values)
    return {
        "status": "stored_on_ur",
        "tool": "store_waypoint_pose_on_ur",
        "variable_name": name,
        "pose_type": "tcp",
        "value_m_rad": [round(v, 6) for v in values],
        "storage": "ur_controller_variable + waypoints_lookup_table",
        "waypoints_lookup_table_path": str(_WAYPOINT_LOOKUP_TABLE_PATH),
        "waypoint_compatible": True,
        "recommended_program_use": "PolyScope Variable Waypoint",
    }


@mcp.tool
def store_joint_configuration_on_ur(
    variable_name: str,
    joint_angles_deg: list[float] | None = None,
) -> dict:
    """Store a joint configuration variable on the UR controller.

    Use this when you want to save a robot configuration as six joint values,
    then later move to it with ``move_to_stored_joint_configuration``.

    Args:
        variable_name: Target UR variable name (for example ``home_q``).
        joint_angles_deg: Optional explicit joint target in degrees. If omitted,
            stores the current actual joints.

    Returns:
        A dict with the stored joint configuration.

    Notes:
        The server caches the written value so
        ``move_to_stored_joint_configuration`` can use it immediately in later
        calls, even when the controller program context has restarted.
    """
    name = _validate_ur_variable_name(variable_name)
    if joint_angles_deg is None:
        q_rad = [float(v) for v in robot.get_joint_positions()]
    else:
        q_rad = _checked_joint_target([float(v) for v in joint_angles_deg])
    robot.define_joint_variable(name, q_rad)
    _save_joint_to_waypoint_bank(name, q_rad)
    return {
        "status": "stored_on_ur",
        "tool": "store_joint_configuration_on_ur",
        "variable_name": name,
        "pose_type": "joints",
        "value_rad": [round(v, 6) for v in q_rad],
        "value_deg": [round(math.degrees(v), 3) for v in q_rad],
        "storage": "ur_controller_variable + waypoints_lookup_table",
        "waypoints_lookup_table_path": str(_WAYPOINT_LOOKUP_TABLE_PATH),
        "waypoint_compatible": False,
        "recommended_program_use": "Joint configuration/reference data",
    }


@mcp.tool
def move_to_stored_tcp_waypoint(
    variable_name: str,
    speed: float = 0.25,
    acceleration: float = 1.2,
    timeout_s: float = 30.0,
) -> dict:
    """Move linearly to a UR-side pose variable (no local memory lookup).

    This executes `movel(<variable_name>, ...)` on the controller, so the target
    is resolved from the UR variable directly. Use this with variables created by
    `store_waypoint_pose_on_ur` (pose values like p[x,y,z,rx,ry,rz]).

    Args:
        variable_name: Name of a UR pose variable present on the controller.
        speed: Cartesian speed in m/s.
        acceleration: Cartesian acceleration in m/s^2.
        timeout_s: Completion timeout in seconds.

    Returns:
        Final robot state snapshot after the move.
    """
    name = _validate_ur_variable_name(variable_name)
    _check_tcp_dynamics(speed, acceleration)
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s}.")

    # The controller resolves the variable, but the move must still pass the
    # same geometric gate as move_linear -- a stored pose taught on a bigger
    # arm (or another cell) would otherwise replay unchecked. The pose is
    # looked up locally: first the waypoint bank, then the cached expression.
    pose = None
    bank = _load_waypoint_bank()
    entry = bank.get("pose_variables", {}).get(name)
    if isinstance(entry, list) and len(entry) == 6:
        pose = [float(v) for v in entry]
    elif isinstance(entry, dict) and isinstance(entry.get("value"), list):
        pose = [float(v) for v in entry["value"]]
    if pose is None:
        expr = robot._cached_ur_variable_expr(name)
        if expr and expr.startswith("p["):
            try:
                pose = [float(v) for v in expr[2:-1].split(",")]
            except ValueError:
                pose = None
    if pose is None or len(pose) != 6:
        raise ValueError(
            f"'{name}' is not in the local waypoint bank, so its target "
            "cannot be safety-checked before moving. Re-store it with "
            "store_waypoint_pose_on_ur first.")
    _check_reach(*pose[:3], what=f"stored waypoint '{name}'")
    _check_flange_above_table(pose[:3], pose[3:])

    state = robot.move_linear_to_variable(
        name,
        speed=float(speed),
        acceleration=float(acceleration),
        timeout_s=float(timeout_s),
    )
    return {
        "status": "executed",
        "source": "ur_controller_variable",
        "variable_name": name,
        "motion": "movel",
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "robot_mode": state.robot_mode,
    }


@mcp.tool
def move_to_stored_joint_configuration(
    variable_name: str,
    speed: float = DEFAULT_SPEED,
    acceleration: float = DEFAULT_ACCEL,
    timeout_s: float = 30.0,
) -> dict:
    """Move in joint space to a UR-side joint variable (no local memory lookup).

    This executes `movej(<variable_name>, ...)` on the controller, so the target
    is resolved from the UR variable directly. Use this with variables created by
    `store_joint_configuration_on_ur`.

    Args:
        variable_name: Name of a UR joint variable present on the controller.
        speed: Joint speed in rad/s.
        acceleration: Joint acceleration in rad/s^2.
        timeout_s: Completion timeout in seconds.

    Returns:
        Final robot state snapshot after the move.
    """
    name = _validate_ur_variable_name(variable_name)
    _check_joint_dynamics(speed, acceleration)
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s}.")

    state = robot.move_joint_to_variable(
        name,
        speed=float(speed),
        acceleration=float(acceleration),
        timeout_s=float(timeout_s),
    )
    return {
        "status": "executed",
        "source": "ur_controller_variable",
        "variable_name": name,
        "motion": "movej",
        "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "robot_mode": state.robot_mode,
    }


@mcp.tool
def movej(
    joint_angles_deg: list[float],
    speed: float = DEFAULT_SPEED,
    acceleration: float = DEFAULT_ACCEL,
) -> dict:
    """URScript-style alias for move_robot_to_position (absolute joint move)."""
    return move_robot_to_position(
        joint_angles_deg=joint_angles_deg,
        speed=speed,
        acceleration=acceleration,
    )


@mcp.tool
def move_joint(
    joint_angles_deg: list[float] | None = None,
    speed: float = DEFAULT_SPEED,
    acceleration: float = DEFAULT_ACCEL,
) -> dict:
    """Compatibility alias for absolute joint motion.

    Equivalent to ``move_robot_to_position``. Exposed so MCP clients and
    rubrics that look for a ``move_joint`` tool name can call it directly.
    """
    return move_robot_to_position(
        joint_angles_deg=joint_angles_deg,
        speed=speed,
        acceleration=acceleration,
    )


@mcp.tool
def movel(
    position_m: list[float],
    rotation_rad: list[float] | None = None,
    speed: float = 0.25,
    acceleration: float = 1.2,
) -> dict:
    """URScript-style alias for move_linear (Cartesian straight-line move)."""
    return move_linear(
        position_m=position_m,
        rotation_rad=rotation_rad,
        speed=speed,
        acceleration=acceleration,
    )


# =========================================================================== #
# CONTINUOUS MOTION PATTERNS  --  open-ended motions the operator steers while
# they run ("faster", "stop", "keep going"). Every tool here returns at once;
# the motion itself lives in a URScript loop on the controller, and each
# command preempts it. The engine is in motion_patterns.py; these are the
# LLM-facing wrappers. See that module for why blocking tools cannot do this.
# =========================================================================== #

# Fraction of the container's diameter the stirring circle spans. Below 1.0 so
# the tool sweeps the contents without scraping the wall.
STIR_DIAMETER_FRACTION = 0.85

# A gentle stir. Clamped down automatically when a small container's
# centripetal limit is lower than this.
DEFAULT_STIR_SPEED_M_S = 0.08


def _validate_pattern_pose(pose_m_rad: list[float], what: str) -> None:
    """Run one pattern waypoint through the Diamond safety layer."""
    _check_box(*pose_m_rad[:3], what=what)
    _check_reach(*pose_m_rad[:3], what=what)
    _check_flange_above_table(pose_m_rad[:3], pose_m_rad[3:])


pattern_runner = PatternRunner(
    robot=robot,
    validate_pose=_validate_pattern_pose,
    validate_start=_check_linear_start,
)


@mcp.tool
def stir(
    container_diameter_cm: float,
    motion: str = "circle",
    speed_m_s: float | None = None,
    direction: str = "counterclockwise",
) -> dict:
    """Start stirring inside a container, and keep stirring until told to stop.

    Call this when the operator asks to stir, mix, swirl or fold something --
    "stir the pan", "mix that sauce". The robot must already be holding the
    utensil with its tip IN THE CENTRE of the container, at the depth it should
    work at: this tool takes the current tool pose as the centre of the circle
    and never moves up or down from it. If the tool is not in place yet, say so
    and ask for it to be positioned (by hand in freedrive, or with a move tool)
    before calling this.

    Returns immediately, while the robot keeps moving. Afterwards use
    adjust_pattern_speed, pause_motion, resume_motion and finish_motion to
    steer it, and get_motion_status to check on it.

    Args:
        container_diameter_cm: Inside diameter of the pan, pot or bowl in
            centimetres. The stirring circle spans 85% of it, so the utensil
            sweeps the contents without scraping the wall.
        motion: "circle" for an ordinary stir (the default), "figure_eight" to
            fold, "spiral" to work from the centre outwards and back.
        speed_m_s: Tool speed in metres per second. Omit for a gentle stir;
            small containers are limited further by how hard the contents can
            be thrown sideways.
        direction: "counterclockwise" (default) or "clockwise".

    Returns:
        A dict with the ``job_id``, the ``radius_m`` being stirred,
        ``speed_m_s`` and ``laps_per_min``, and ``max_safe_speed_m_s`` -- the
        ceiling for this container, useful when the operator asks for more.

    Raises:
        ValueError: A container too small or too large to work in, a speed
            beyond the safe limit for that radius, a path that leaves the safe
            workspace, or a pattern already running.
        RuntimeError: The robot is not powered on, is in a singular pose that
            cannot start a Cartesian move, or did not begin moving.
    """
    if not 2.0 <= container_diameter_cm <= 60.0:
        raise ValueError(
            f"Container diameter {container_diameter_cm} cm is outside the "
            "workable range [2, 60] cm."
        )
    if direction not in ("counterclockwise", "clockwise"):
        raise ValueError(
            f"direction must be 'counterclockwise' or 'clockwise', got "
            f"'{direction}'."
        )

    radius_m = STIR_DIAMETER_FRACTION * (container_diameter_cm / 100.0) / 2.0
    if speed_m_s is None:
        speed_m_s = min(DEFAULT_STIR_SPEED_M_S,
                        0.8 * max_speed_for_radius(radius_m))

    return pattern_runner.start(
        pattern_name=motion,
        radius_m=radius_m,
        speed_m_s=float(speed_m_s),
        direction=1 if direction == "counterclockwise" else -1,
    )


@mcp.tool
def start_motion_pattern(
    pattern: str,
    radius_m: float,
    speed_m_s: float,
    direction: str = "counterclockwise",
    turns: float = 3.0,
    points_per_lap: int = 24,
) -> dict:
    """Start any repeating motion around the current tool pose.

    The general form of ``stir``, for motions that are not about a container:
    sweeping a tray, polishing, working a surface. The current tool pose
    becomes the centre and the orientation is held throughout, so position the
    tool first.

    Returns immediately, while the robot keeps moving. Steer it afterwards with
    adjust_pattern_speed, pause_motion, resume_motion and finish_motion.

    Args:
        pattern: "circle", "figure_eight", "spiral", or "linear_sweep"
            (back and forth along one axis).
        radius_m: Half-width of the motion in metres, 0.005 to 0.30.
        speed_m_s: Tool speed. The safe ceiling falls as the radius shrinks;
            the error message states the exact limit if you exceed it.
        direction: "counterclockwise" (default) or "clockwise".
        turns: For "spiral" only, how many revolutions from centre to rim.
        points_per_lap: How finely non-circular patterns are sampled. Higher is
            geometrically truer but slower, because the tighter blend corners
            cap the speed. Circles ignore this: they run as exact arcs.

    Returns:
        A dict with ``job_id``, ``laps_per_min``, ``max_safe_speed_m_s``, and
        ``exact_path`` (True when the path is a true arc rather than sampled).

    Raises:
        ValueError: Unknown pattern, radius or speed out of range, a path
            leaving the safe workspace, or a pattern already running.
        RuntimeError: The robot is not powered on, is in a singular pose, or
            did not begin moving.
    """
    if direction not in ("counterclockwise", "clockwise"):
        raise ValueError(
            f"direction must be 'counterclockwise' or 'clockwise', got "
            f"'{direction}'."
        )
    return pattern_runner.start(
        pattern_name=pattern,
        radius_m=float(radius_m),
        speed_m_s=float(speed_m_s),
        direction=1 if direction == "counterclockwise" else -1,
        turns=float(turns),
        points_per_lap=int(points_per_lap),
    )


@mcp.tool
def adjust_pattern_speed(
    factor: float | None = None,
    speed_m_s: float | None = None,
) -> dict:
    """Speed the running motion up or slow it down, without interrupting it.

    This is what "faster", "slower", "a bit quicker", "gently" mean while a
    pattern is running. Prefer ``factor``, since those requests are relative:
    1.3 for faster, 0.7 for slower, and repeat it if asked again. The tool
    keeps its place on the path, so the motion continues rather than restarting.

    If the request would throw the contents out of the container, the call is
    refused and the message gives the fastest safe speed -- report that number
    rather than retrying blindly. Speed can also be set while paused; it then
    takes effect on resume.

    Args:
        factor: Multiplier on the current speed, 0.1 to 10.0.
        speed_m_s: Absolute tool speed instead, when a number was given.

    Returns:
        A dict with the new ``speed_m_s`` and ``laps_per_min``, plus
        ``max_safe_speed_m_s`` for this radius.

    Raises:
        ValueError: No pattern running, both or neither argument given, or a
            speed beyond the safe limit.
    """
    return pattern_runner.set_speed(factor=factor, speed_m_s=speed_m_s)


@mcp.tool
def pause_motion() -> dict:
    """Stop the arm now, keeping the motion ready to continue.

    This is what a bare "stop", "wait", "hold on" or "pause" means: the
    operator wants the arm still, usually to look at something, and expects to
    say "keep going" afterwards. The arm decelerates smoothly and the anchor,
    the speed and the place on the path are all kept, so resume_motion carries
    on from exactly here.

    Prefer this over finish_motion whenever it is not clear that the job is
    over. Both stop the arm just as quickly; this one is simply recoverable.

    Safe to call at any time: if no pattern is registered but the arm is still
    moving (a loop left over from an earlier server session), this stops it
    too, rather than refusing.

    Returns:
        A dict with ``state`` "paused" and the laps completed so far, or
        ``status`` "idle" when there was nothing to stop.
    """
    return pattern_runner.pause()


@mcp.tool
def resume_motion() -> dict:
    """Continue a paused motion from where the tool is now.

    This is "keep going", "resume", "continue", "carry on". The phase is read
    back from the robot's actual position, so the motion picks up smoothly even
    if the arm was nudged while it was paused.

    Returns:
        A dict with ``state`` "running" and the current speed.

    Raises:
        ValueError: No pattern to resume.
        RuntimeError: The robot is not powered on, or did not begin moving.
    """
    return pattern_runner.resume()


@mcp.tool
def finish_motion() -> dict:
    """End the motion for good and forget the anchor.

    This is "that's enough", "we're done", "stop stirring" -- the task is over,
    not merely interrupted. Once finished the motion cannot be resumed; a new
    stir has to be positioned and started again. When in doubt between this and
    pause_motion, pause.

    Safe to call at any time: if no pattern is registered but the arm is still
    moving, this stops it too, rather than refusing.

    Returns:
        A dict with the total laps completed and how long the job lasted, or
        ``status`` "idle" when there was nothing to stop.
    """
    return pattern_runner.finish()


@mcp.tool
def get_motion_status() -> dict:
    """Report what the running motion is doing, without disturbing it.

    Always safe to call. Use it to answer "is it still stirring?", to check
    progress, or to find out why a motion stopped on its own -- a protective
    stop shows up here as a ``safety_status`` other than 1.

    Returns:
        When idle, ``status`` "idle" and ``active`` False. When a pattern
        exists: its ``state`` ("running" or "paused"), ``motion`` name,
        ``speed_m_s``, ``laps_per_min``, ``laps_completed``, the ``anchor``
        pose, and the live ``tcp_pose_m_rad``, ``robot_moving`` and
        ``safety_status``.
    """
    return pattern_runner.status()


# =========================================================================== #
# TEMPLATE TOOL  --  the minimal shape of a tool, doing nothing real. Copy this
# to start your own, then follow the four-step pattern above. Add as many as
# the tiers need.
# =========================================================================== #
@mcp.tool
def example() -> str:
    """A placeholder tool that performs no action.

    Returns a fixed string. Use it as the skeleton for a real tool of your own.
    """
    return "this was an example"


# --- Wrist-camera tools (optional) --------------------------------------- #
# UR_VISION=1 folds the wrist-camera server into this one, so the client sees a
# single "ur-tools" MCP carrying both the motion and the perception tools.
# The import is deferred to here on purpose: it pulls in OpenCV and YOLO, and
# on macOS the RealSense it drives needs root (see camera/README.md) -- the
# robot-only path, which the voice front-end launches directly, should have to
# satisfy neither. Mounting with no namespace keeps the vision tool names
# unprefixed, so `look` stays `look` and CLAUDE.md keeps describing reality.
VISION_ENABLED = os.environ.get("UR_VISION", "").strip().lower() in {
    "1", "on", "true", "yes"}


def _mount_vision() -> None:
    """Fold camera/'s vision tools into this server. Never fatal: a missing
    camera stack costs the perception tools, not the whole robot."""
    if not VISION_ENABLED:
        return
    # camera/ is a sibling of voice/ inside case 1, so this resolves from this
    # file alone -- no assumption about where the repo sits or what the
    # sibling case folders are called.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "camera"))
    try:
        import vision_tools
    except Exception as exc:      # no OpenCV/YOLO, unreadable VISION_MODEL, ...
        print(f"[ur-tools] camera unavailable, motion tools only: {exc}",
              file=sys.stderr)
        return
    mcp.mount(vision_tools.mcp)
    print("[ur-tools] vision tools mounted (camera opens on first use)",
          file=sys.stderr)


if __name__ == "__main__":
    _mount_vision()
    robot.connect()
    # A motion pattern runs as a loop ON THE CONTROLLER, so it outlives this
    # process. MCP clients stop their servers with SIGTERM, which would skip
    # the atexit cleanup and leave the robot moving; turn it into a normal exit.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    mcp.run()
