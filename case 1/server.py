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
import time

from fastmcp import FastMCP

from ur_client import (
    HOME_Q_RAD,
    JOINT_LIMIT,
    JOINT_NAMES,
    ROBOT_MODE_RUNNING,
    URClient,
)

mcp = FastMCP("ur-tools")
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
GRIPPER_PIN = 0


# =========================================================================== #
# DIAMOND SAFETY LAYER  --  every motion tool funnels through these checks
# BEFORE anything moves. A forward-kinematics model (UR10e DH parameters,
# verified to match the simulator's TCP to the millimetre) predicts where a
# joint target puts the whole arm, so unsafe commands are rejected with a
# plain-language reason the LLM can read and correct, instead of executed.
# =========================================================================== #

# Command caps. Joint units rad/s, rad/s^2 (UR10e's slowest joints allow
# 2.09 rad/s); TCP units m/s, m/s^2.
MAX_JOINT_SPEED = 2.0
MAX_JOINT_ACCEL = 4.0
MAX_TCP_SPEED = 1.0
MAX_TCP_ACCEL = 2.5

# Safe workspace for the tool, base frame, metres. The tool tip may work down
# to FLOOR_Z_M; the arm's joint origins (thick links, ~5 cm radius) must keep
# the larger ARM_CLEARANCE_M so the physical link never touches the table even
# though the check runs on the centreline.
WORKSPACE_M = {"x": (-1.30, 1.30), "y": (-1.30, 1.30), "z": (0.02, 1.60)}
FLOOR_Z_M = 0.02
ARM_CLEARANCE_M = 0.08
# Kinematic maximum distance from the shoulder point (~1.364 m, wrist offsets
# included). A Cartesian target beyond this can never be reached and would
# only make movel stall until timeout.
REACH_M = 1.37

# UR10e standard DH parameters (a, d, alpha), from UR's published table.
_DH_A = (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0)
_DH_D = (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655)
_DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)

# Which frame origin sits where on the arm, for readable error messages.
_FRAME_NAMES = ("shoulder", "elbow", "wrist1", "wrist2", "wrist3", "tool")


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
    the wrist flange sits d6 = 0.117 m BELOW the TCP and can pass through the
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
    ([0, -90, 0, -90, 0, 0] degrees) -- "move the robot home" is a call with no
    arguments.

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
        ``joints_deg`` (per-joint angles), ``tcp_pose`` [x, y, z, rx, ry, rz]
        in metres and radians, and ``robot_mode``.

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
        ``gripper_closed``: the commanded gripper state (digital output).
    """
    # 4. Execute (a read), then report in human units.
    state = robot.get_state()
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
        "gripper_closed": bool(state.digital_out >> GRIPPER_PIN & 1),
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
    """Rotate joints BY an amount, relative to where they are now.

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
        resulting ``joints_deg``, ``tcp_pose``, and ``robot_mode``.

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
    robot's base frame; for scale, home is [0, -0.29, 1.48].

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
        ``joints_deg`` / ``tcp_pose``.

    Raises:
        ValueError: Fewer than 2 or more than 20 waypoints, a malformed or
            limit-violating waypoint, an unsafe pose anywhere along the
            sequence, capped speed/acceleration exceeded, or a bad blend.
        RuntimeError: The robot is not powered on.
        TimeoutError: The path did not complete (blocked, or a protective
            stop -- check get_robot_state's safety_status).
    """
    # 1. Validate the sequence shape and each waypoint.
    if not 2 <= len(waypoints_deg) <= 20:
        raise ValueError(
            f"Expected 2 to 20 waypoints, got {len(waypoints_deg)}. "
            "For a single target use move_robot_to_position."
        )
    if not 0 <= blend_m <= 0.1:
        raise ValueError(
            f"Blend radius {blend_m} m is outside the allowed range [0, 0.1]."
        )

    # 2. Convert each waypoint to radians (validates count + joint limits).
    targets_rad = []
    for i, wp in enumerate(waypoints_deg, start=1):
        try:
            targets_rad.append(_checked_joint_target(wp))
        except ValueError as exc:
            raise ValueError(f"Waypoint {i}: {exc}") from exc

    # 3. Check feasibility: speed caps, then the FK check on EVERY waypoint.
    _check_joint_dynamics(speed, acceleration)
    for i, target in enumerate(targets_rad, start=1):
        _check_pose_safe(target, what=f"waypoint {i}")

    # 4. Execute the whole path, sampling state as it runs, then report.
    #    The wait budget adapts to the summed travel time of all segments.
    prev = robot.get_state().q_rad
    travel_s = 0.0
    for target in targets_rad:
        travel_s += max(abs(t - p) for t, p in zip(target, prev)) / speed
        prev = target
    timeout = max(60.0, 2.0 * travel_s + 10.0)

    t0 = time.monotonic()
    samples: list[tuple[float, list[float], float]] = []

    def on_sample(st) -> None:
        samples.append((time.monotonic() - t0, st.q_rad, max(abs(v) for v in st.qd_rad)))

    state = robot.move_path(targets_rad, speed, acceleration,
                            blend_m=blend_m, timeout_s=timeout,
                            on_sample=on_sample)
    duration = time.monotonic() - t0

    step = max(1, len(samples) // 25)
    motion_log = [
        {"t_s": round(t, 2),
         "joints_deg": [round(math.degrees(q), 1) for q in qs]}
        for t, qs, _ in samples[::step][:25]
    ]
    return {
        "status": "reached",
        "waypoints": len(waypoints_deg),
        "duration_s": round(duration, 2),
        "peak_joint_speed_rad_s": round(max((s[2] for s in samples), default=0.0), 3),
        "motion_log": motion_log,
        "joints_deg": {n: round(math.degrees(q), 1)
                       for n, q in zip(JOINT_NAMES, state.q_rad)},
        "tcp_pose": [round(v, 4) for v in state.tcp_pose],
        "robot_mode": state.robot_mode,
    }


# =========================================================================== #
# DIAMOND TOOL  --  gripper via digital IO. The simulator has no physical
# gripper; this drives the digital output a real one (tool IO) would follow,
# and reads the commanded state back over RTDE so the report is ground truth.
# =========================================================================== #
@mcp.tool
def set_gripper(closed: bool) -> dict:
    """Open or close the gripper.

    Use ``closed=true`` to grip (before lifting an object) and
    ``closed=false`` to release. The command drives the robot's gripper
    digital output and confirms it by reading the IO state back from the
    controller. Check ``get_robot_state``'s ``gripper_closed`` afterwards if
    you need to re-verify during a longer task.

    Args:
        closed: True to close the gripper, False to open it.

    Returns:
        A dict with ``gripper`` ("closed" or "open"), the gripper ``pin``
        number, and ``pin_state`` (the confirmed digital-output level).

    Raises:
        RuntimeError: The robot is not powered on.
        TimeoutError: The controller did not confirm the output change.
    """
    # 1.-3. A boolean needs no validation, conversion, or limit checks.
    # 4. Execute and report the state read back from the controller.
    state = robot.set_digital_out(GRIPPER_PIN, closed)
    confirmed = bool(state.digital_out >> GRIPPER_PIN & 1)
    return {
        "gripper": "closed" if confirmed else "open",
        "pin": GRIPPER_PIN,
        "pin_state": confirmed,
    }


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


if __name__ == "__main__":
    robot.connect()
    mcp.run()
