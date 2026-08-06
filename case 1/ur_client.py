"""Thin seam over the UR robot interface, using only the Python standard library.

Two UR network interfaces are spoken directly over TCP sockets, so there is no
C++ toolchain to install and nothing to compile:

  * Primary interface (port 30001) -- MOTION. We upload a tiny URScript program
    (a ``movej``) and the controller runs it. That is all a joint move needs.
  * RTDE, Real-Time Data Exchange (port 30004) -- STATE. We speak just enough of
    the RTDE protocol to read the actual joint angles and TCP pose.

This is the seam between the robot and the MCP tools: ``server.py`` calls these
methods and never touches sockets or the RTDE byte format directly. Keeping the
boundary here means the tool layer stays portable between the PolyScope X
simulator (see ``../simulation environment``) and a real UR robot -- only the
host address changes.
"""
from __future__ import annotations

import math
import os
import socket
import struct
import time
from dataclasses import dataclass, field

# UR joint order: base, shoulder, elbow, wrist1, wrist2, wrist3.
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")

# Software joint limits (radians). UR joints rotate +/- 2*pi; we keep a small
# margin so a move never parks exactly on the mechanical limit.
JOINT_LIMIT = 2 * math.pi - math.radians(2.0)

# Canonical home pose, joint angles in radians (base..wrist3): upright column,
# elbow square, tool pointing down. Same convention used across the three cases.
HOME_Q_RAD = [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0]

# UR robot mode 7 == RUNNING (powered on, brakes released, ready to move).
ROBOT_MODE_RUNNING = 7

PRIMARY_PORT = 30001  # motion: upload URScript here
RTDE_PORT = 30004     # state: read joint angles / TCP pose here

# RTDE protocol command bytes (see the UR RTDE guide).
_RTDE_REQUEST_PROTOCOL_VERSION = 86  # 'V'
_RTDE_SETUP_OUTPUTS = 79             # 'O'
_RTDE_START = 83                     # 'S'
_RTDE_DATA_PACKAGE = 85              # 'U'
_RTDE_OUTPUTS = (
    "actual_q,actual_qd,actual_TCP_pose,robot_mode,safety_status,"
    "actual_digital_output_bits"
)


@dataclass
class RobotState:
    """A single snapshot of the robot, read over RTDE."""

    q_rad: list[float]        # actual joint angles (radians), base..wrist3
    qd_rad: list[float]       # actual joint velocities (rad/s), base..wrist3
    tcp_pose: list[float]     # actual TCP pose [x, y, z, rx, ry, rz] (m, rad)
    robot_mode: int           # 7 == RUNNING
    safety_status: int        # 1 == NORMAL
    digital_out: int          # standard digital output bits (bit n == DO n)

    @property
    def is_moving(self) -> bool:
        """True while any joint is still moving (above noise threshold)."""
        return max(abs(v) for v in self.qd_rad) > 0.01


@dataclass
class URClient:
    """Connection to a UR robot or the PolyScope X simulator.

    Args:
        host: Robot / simulator IP. Defaults to the ``UR_HOST`` env var, then to
            ``127.0.0.1`` (the simulator's published address on your machine).
    """

    host: str = field(default_factory=lambda: os.environ.get("UR_HOST", "127.0.0.1"))

    def connect(self) -> None:
        """Verify the robot is reachable, failing fast with a readable reason."""
        try:
            self.get_state()
        except OSError as exc:
            raise ConnectionError(
                f"Cannot reach the robot at {self.host}:{RTDE_PORT} ({exc}). "
                "Is the simulator up? See ../simulation environment "
                "(docker compose up -d), then open http://localhost."
            ) from exc

    # --- State (RTDE receive) ------------------------------------------- #
    def get_state(self) -> RobotState:
        """Read one RTDE snapshot: joint angles, TCP pose, mode, safety."""
        with socket.create_connection((self.host, RTDE_PORT), timeout=5) as s:
            self._rtde_send(s, _RTDE_REQUEST_PROTOCOL_VERSION, struct.pack(">H", 2))
            self._rtde_recv(s)  # version-accepted reply
            self._rtde_send(
                s, _RTDE_SETUP_OUTPUTS,
                struct.pack(">d", 125.0) + _RTDE_OUTPUTS.encode(),
            )
            self._rtde_recv(s)  # recipe + variable types
            self._rtde_send(s, _RTDE_START)
            self._rtde_recv(s)  # start-accepted reply

            cmd, body = self._rtde_recv(s)
            while cmd != _RTDE_DATA_PACKAGE:  # skip any non-data control replies
                cmd, body = self._rtde_recv(s)

        off = 1  # first byte is the recipe id
        q = list(struct.unpack(">6d", body[off:off + 48])); off += 48
        qd = list(struct.unpack(">6d", body[off:off + 48])); off += 48
        tcp = list(struct.unpack(">6d", body[off:off + 48])); off += 48
        mode = struct.unpack(">i", body[off:off + 4])[0]; off += 4
        safety = struct.unpack(">i", body[off:off + 4])[0]; off += 4
        dout = struct.unpack(">Q", body[off:off + 8])[0]
        return RobotState(q_rad=q, qd_rad=qd, tcp_pose=tcp, robot_mode=mode,
                          safety_status=safety, digital_out=dout)

    def get_joint_positions(self) -> list[float]:
        """Actual joint angles in radians, base..wrist3."""
        return self.get_state().q_rad

    def get_tcp_pose(self) -> list[float]:
        """Actual TCP pose [x, y, z, rx, ry, rz] (metres, radians)."""
        return self.get_state().tcp_pose

    # --- Motion (primary interface) ------------------------------------- #
    def _require_running(self) -> RobotState:
        """Return the current state, or raise if the robot cannot move."""
        state = self.get_state()
        if state.robot_mode != ROBOT_MODE_RUNNING:
            raise RuntimeError(
                f"Robot is not powered on (mode {state.robot_mode}, need "
                f"{ROBOT_MODE_RUNNING}=RUNNING). Open http://localhost and power "
                "the robot on + release brakes, then try again."
            )
        return state

    def _run_program(
        self,
        body: str,
        *,
        done,
        timeout_s: float,
        on_sample=None,
    ) -> RobotState:
        """Upload a one-shot URScript program and block until ``done(state)``.

        Keeps the upload connection open until the wait ends: PolyScope X can
        discard a freshly uploaded script when the uploader disconnects around
        program start, so a fire-and-close upload intermittently does nothing.
        While open, the controller streams state messages at us; drain them so
        the buffer never fills, and watch for the program-started marker to
        tell a dropped script from a slow move.

        Args:
            body: URScript statements (each line indented two spaces, newline
                terminated) forming the program body.
            done: predicate on RobotState; returning True ends the wait.
            timeout_s: give up after this many seconds.
            on_sample: optional callback receiving every polled RobotState,
                for callers that want the motion history, not just the end.
        """
        script = "def move_to():\n" + body + "end\n"
        started = stopped = False
        stopped_at = 0.0
        tail = b""
        with socket.create_connection((self.host, PRIMARY_PORT), timeout=5) as s:
            s.sendall(script.encode())
            s.setblocking(False)

            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                time.sleep(0.15)
                try:
                    while True:
                        chunk = s.recv(65536)
                        if not chunk:
                            break
                        blob = tail + chunk
                        started = started or b"PROGRAM_XXX_STARTED" in blob
                        stopped = stopped or b"PROGRAM_XXX_STOPPED" in blob
                        tail = blob[-32:]
                except (BlockingIOError, TimeoutError):
                    pass
                if stopped and not stopped_at:
                    stopped_at = time.monotonic()
                state = self.get_state()
                if on_sample is not None:
                    on_sample(state)
                # Success only counts after the program has both started and
                # ended. Judging done() alone would return before the program
                # even begins whenever the condition already holds at upload
                # time (an out-and-back path ending where it started), and
                # closing the socket then makes PolyScope X discard the script.
                if started and stopped and done(state):
                    return state
                # Program over but the target never satisfied (after a short
                # settle grace): fail fast with the likely reason instead of
                # burning the rest of the timeout.
                if stopped_at and time.monotonic() - stopped_at > 1.5 \
                        and not done(state):
                    raise TimeoutError(
                        "The move program ended without reaching the target "
                        "(a protective stop, or the controller rejected the "
                        "motion). Check get_robot_state's safety_status."
                    )
        if not started:
            raise TimeoutError(
                f"The move script never started within {timeout_s:.0f}s "
                "(is the UI in control, or another program running?)."
            )
        raise TimeoutError(
            f"Robot did not reach the target within {timeout_s:.0f}s "
            "(still moving, blocked, or a protective stop?)."
        )

    def move_joint(
        self,
        q: list[float],
        speed: float,
        acceleration: float,
        *,
        tol_rad: float = 0.01,
        timeout_s: float = 20.0,
    ) -> RobotState:
        """Joint-space move to absolute angles ``q`` (radians), then block until
        the robot arrives (or ``timeout_s`` elapses).

        Raises:
            RuntimeError: The robot is not in RUNNING mode, so it cannot move.
                Power it on / release brakes in the PolyScope X UI first.
            TimeoutError: The robot did not reach ``q`` within ``timeout_s``.
        """
        self._require_running()
        joints = ", ".join(f"{v:.6f}" for v in q)
        body = f"  movej([{joints}], a={acceleration:.4f}, v={speed:.4f})\n"
        return self._run_program(
            body,
            done=lambda st: max(abs(a - b)
                                for a, b in zip(st.q_rad, q)) <= tol_rad,
            timeout_s=timeout_s,
        )

    def move_linear(
        self,
        pose: list[float],
        speed: float,
        acceleration: float,
        *,
        tol_pos_m: float = 0.005,
        timeout_s: float = 20.0,
    ) -> RobotState:
        """Straight-line TCP move to ``pose`` [x, y, z, rx, ry, rz] (m, rad).

        ``speed`` and ``acceleration`` are Cartesian (m/s, m/s^2), unlike the
        joint-space units of :meth:`move_joint`. Completion is judged on TCP
        position within ``tol_pos_m`` plus all joints stationary; orientation
        is not compared numerically because equivalent axis-angle rotations
        can differ in sign.
        """
        self._require_running()
        target = ", ".join(f"{v:.6f}" for v in pose)
        body = f"  movel(p[{target}], a={acceleration:.4f}, v={speed:.4f})\n"

        def done(st: RobotState) -> bool:
            pos_err = max(abs(a - b) for a, b in zip(st.tcp_pose[:3], pose[:3]))
            return pos_err <= tol_pos_m and not st.is_moving

        return self._run_program(body, done=done, timeout_s=timeout_s)

    def move_path(
        self,
        waypoints: list[list[float]],
        speed: float,
        acceleration: float,
        *,
        blend_m: float = 0.02,
        tol_rad: float = 0.01,
        timeout_s: float = 60.0,
        on_sample=None,
    ) -> RobotState:
        """Run a multi-waypoint joint-space path as ONE URScript program.

        Every waypoint except the last gets blend radius ``blend_m`` (metres),
        so the robot rounds the corners instead of stopping at each waypoint.
        Completion is judged on the last waypoint: joints within ``tol_rad``
        and stationary (a blended path may pass near earlier waypoints without
        stopping). ``on_sample`` sees every polled state for motion streaming.
        """
        self._require_running()
        lines = []
        for i, q in enumerate(waypoints):
            joints = ", ".join(f"{v:.6f}" for v in q)
            r = 0.0 if i == len(waypoints) - 1 else blend_m
            lines.append(f"  movej([{joints}], a={acceleration:.4f}, "
                         f"v={speed:.4f}, r={r:.4f})\n")
        final = waypoints[-1]

        def done(st: RobotState) -> bool:
            at_final = max(abs(a - b)
                           for a, b in zip(st.q_rad, final)) <= tol_rad
            return at_final and not st.is_moving

        return self._run_program("".join(lines), done=done,
                                 timeout_s=timeout_s, on_sample=on_sample)

    def set_digital_out(self, pin: int, value: bool,
                        *, timeout_s: float = 5.0) -> RobotState:
        """Set standard digital output ``pin`` and wait for RTDE to confirm it."""
        self._require_running()
        body = f"  set_standard_digital_out({pin}, {'True' if value else 'False'})\n"
        return self._run_program(
            body,
            done=lambda st: bool(st.digital_out >> pin & 1) == bool(value),
            timeout_s=timeout_s,
        )

    # --- RTDE framing helpers ------------------------------------------- #
    @staticmethod
    def _rtde_send(s: socket.socket, cmd: int, payload: bytes = b"") -> None:
        s.sendall(struct.pack(">HB", 3 + len(payload), cmd) + payload)

    @staticmethod
    def _rtde_recv(s: socket.socket) -> tuple[int, bytes]:
        # An empty recv means the controller closed the connection; raising
        # beats looping on b"" forever (header) or returning a short body the
        # caller would trip over while unpacking.
        hdr = b""
        while len(hdr) < 3:
            chunk = s.recv(3 - len(hdr))
            if not chunk:
                raise ConnectionError(
                    "RTDE connection closed by the controller mid-message.")
            hdr += chunk
        size, cmd = struct.unpack(">HB", hdr)
        body = b""
        while len(body) < size - 3:
            chunk = s.recv(size - 3 - len(body))
            if not chunk:
                raise ConnectionError(
                    "RTDE connection closed by the controller mid-message.")
            body += chunk
        return cmd, body
