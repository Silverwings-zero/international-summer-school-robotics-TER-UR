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
_RTDE_OUTPUTS = "actual_q,actual_TCP_pose,robot_mode,safety_status"


@dataclass
class RobotState:
    """A single snapshot of the robot, read over RTDE."""

    q_rad: list[float]        # actual joint angles (radians), base..wrist3
    tcp_pose: list[float]     # actual TCP pose [x, y, z, rx, ry, rz] (m, rad)
    robot_mode: int           # 7 == RUNNING
    safety_status: int        # 1 == NORMAL


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
        tcp = list(struct.unpack(">6d", body[off:off + 48])); off += 48
        mode = struct.unpack(">i", body[off:off + 4])[0]; off += 4
        safety = struct.unpack(">i", body[off:off + 4])[0]
        return RobotState(q_rad=q, tcp_pose=tcp, robot_mode=mode, safety_status=safety)

    def get_joint_positions(self) -> list[float]:
        """Actual joint angles in radians, base..wrist3."""
        return self.get_state().q_rad

    def get_tcp_pose(self) -> list[float]:
        """Actual TCP pose [x, y, z, rx, ry, rz] (metres, radians)."""
        return self.get_state().tcp_pose

    # --- Motion (primary interface) ------------------------------------- #
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
        state = self.get_state()
        if state.robot_mode != ROBOT_MODE_RUNNING:
            raise RuntimeError(
                f"Robot is not powered on (mode {state.robot_mode}, need "
                f"{ROBOT_MODE_RUNNING}=RUNNING). Open http://localhost and power "
                "the robot on + release brakes, then try again."
            )

        joints = ", ".join(f"{v:.6f}" for v in q)
        script = (
            "def move_to():\n"
            f"  movej([{joints}], a={acceleration:.4f}, v={speed:.4f})\n"
            "end\n"
        )
        with socket.create_connection((self.host, PRIMARY_PORT), timeout=5) as s:
            s.sendall(script.encode())

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.3)
            state = self.get_state()
            if max(abs(a - b) for a, b in zip(state.q_rad, q)) <= tol_rad:
                return state
        raise TimeoutError(
            f"Robot did not reach the target within {timeout_s:.0f}s "
            "(still moving, blocked, or a protective stop?)."
        )

    # --- RTDE framing helpers ------------------------------------------- #
    @staticmethod
    def _rtde_send(s: socket.socket, cmd: int, payload: bytes = b"") -> None:
        s.sendall(struct.pack(">HB", 3 + len(payload), cmd) + payload)

    @staticmethod
    def _rtde_recv(s: socket.socket) -> tuple[int, bytes]:
        hdr = b""
        while len(hdr) < 3:
            hdr += s.recv(3 - len(hdr))
        size, cmd = struct.unpack(">HB", hdr)
        body = b""
        while len(body) < size - 3:
            chunk = s.recv(size - 3 - len(body))
            if not chunk:
                break
            body += chunk
        return cmd, body
