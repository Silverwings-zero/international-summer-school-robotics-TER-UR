"""Record UR runs (target vs actual) to the dataset CSV schema.

Run this ON a real UR (or URSim, though there the gap is near zero) to build the
dataset that ``dataset.py`` loads. It sweeps a set of point-to-point moves over a
grid of speed / acceleration / payload values, and for each move logs the
commanded (``target_q``) and measured (``actual_q``) joint angles at the RTDE
rate into one CSV per run.

    python record.py --robot-ip 192.168.1.10 --out data

One CSV = one run = one (move, speed, accel, payload) combination, columns
exactly as documented in dataset.py. Keep this in sync with that schema.

Robot interface: the Python standard library only, exactly like Case 1's
``case 1/ur_client.py`` -- nothing to compile, no ``ur_rtde``. Two UR network
interfaces are spoken directly over TCP sockets:

  * Primary interface (30001) -- MOTION. Upload a tiny URScript ``movej``.
  * RTDE (30004) -- STATE. Stream ``target_q`` + ``actual_q`` while the robot
    moves and rings down, so the reality gap is captured.

It is NOT run during the exercise: the dataset is recorded once and shared.
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import struct
import time

import numpy as np

from dataset import (ACTUAL_Q_COLS, META_COLS, N_JOINTS, TARGET_Q_COLS,
                     TIME_COL)

# --- UR network interfaces (same as case 1/ur_client.py) ----------------- #
PRIMARY_PORT = 30001   # motion: upload URScript here
RTDE_PORT = 30004      # state: stream joint angles here

ROBOT_MODE_RUNNING = 7  # powered on, brakes released, ready to move

# RTDE protocol command bytes (see the UR RTDE guide).
_RTDE_REQUEST_PROTOCOL_VERSION = 86  # 'V'
_RTDE_SETUP_OUTPUTS = 79             # 'O'
_RTDE_START = 83                     # 'S'
_RTDE_DATA_PACKAGE = 85              # 'U'
# We record the commanded setpoint AND the measured angle at each instant; their
# difference over a move is the reality gap the whole case is about.
_RTDE_OUTPUTS = "target_q,actual_q,robot_mode"

# Home pose for the sweep (radians). More diverse moves make a better gap model.
HOME = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]

# Parameter sweep. Every combination is recorded for every move.
SPEED_GRID = [0.5, 1.0, 1.75, 2.5]      # rad/s
ACCEL_GRID = [1.0, 3.0, 6.0]            # rad/s^2
PAYLOAD_GRID = [0.0]                    # kg; set the real TCP payload(s) here

LOG_HZ = 125.0                          # RTDE sample rate
SETTLE_S = 0.6                          # extra logging after the move completes
ARRIVE_TOL = 1e-3                       # rad; target_q within this of goal == done


def _build_moves() -> dict:
    """Named goal poses: each is HOME with one joint rotated 90 degrees."""
    moves = {}
    for j in range(N_JOINTS):
        goal = list(HOME)
        goal[j] += np.pi / 2
        moves[f"j{j}_90"] = goal
    return moves


# --- RTDE framing (identical to case 1/ur_client.py) --------------------- #
def _rtde_send(s: socket.socket, cmd: int, payload: bytes = b"") -> None:
    s.sendall(struct.pack(">HB", 3 + len(payload), cmd) + payload)


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


class URRecorder:
    """Minimal pure-socket UR client for recording (mirrors ur_client.py).

    Streaming is needed here (Case 1's one-shot ``get_state`` is not enough): to
    see the gap we must sample ``target_q`` and ``actual_q`` continuously while
    the robot is in motion, so we hold one RTDE connection open per run.
    """

    def __init__(self, host: str):
        self.host = host

    def _open_stream(self) -> socket.socket:
        """Open an RTDE connection and start it streaming our output recipe."""
        s = socket.create_connection((self.host, RTDE_PORT), timeout=5)
        _rtde_send(s, _RTDE_REQUEST_PROTOCOL_VERSION, struct.pack(">H", 2))
        _rtde_recv(s)  # version-accepted reply
        _rtde_send(s, _RTDE_SETUP_OUTPUTS,
                   struct.pack(">d", LOG_HZ) + _RTDE_OUTPUTS.encode())
        _rtde_recv(s)  # recipe + variable types
        _rtde_send(s, _RTDE_START)
        _rtde_recv(s)  # start-accepted reply
        return s

    @staticmethod
    def _read_sample(s: socket.socket) -> tuple[list, list, int]:
        """Read the next RTDE data package -> (target_q, actual_q, robot_mode)."""
        cmd, body = _rtde_recv(s)
        while cmd != _RTDE_DATA_PACKAGE:  # skip any non-data control replies
            cmd, body = _rtde_recv(s)
        off = 1  # first byte is the recipe id
        target = list(struct.unpack(">6d", body[off:off + 48])); off += 48
        actual = list(struct.unpack(">6d", body[off:off + 48])); off += 48
        mode = struct.unpack(">i", body[off:off + 4])[0]
        return target, actual, mode

    def snapshot(self) -> tuple[list, list, int]:
        """One (target_q, actual_q, robot_mode) reading; opens/closes a stream."""
        with self._open_stream() as s:
            return self._read_sample(s)

    def send_move(self, goal, speed: float, accel: float) -> None:
        """Upload a non-blocking URScript ``movej`` to the primary interface."""
        joints = ", ".join(f"{v:.6f}" for v in goal)
        script = (
            "def move_to():\n"
            f"  movej([{joints}], a={accel:.4f}, v={speed:.4f})\n"
            "end\n"
        )
        with socket.create_connection((self.host, PRIMARY_PORT), timeout=5) as s:
            s.sendall(script.encode())

    def move_blocking(self, goal, speed: float, accel: float,
                      *, tol_rad: float = 0.01, timeout_s: float = 20.0) -> None:
        """Move to ``goal`` and wait until the robot arrives (used to reset home)."""
        self.send_move(goal, speed, accel)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.3)
            _, actual, _ = self.snapshot()
            if max(abs(a - b) for a, b in zip(actual, goal)) <= tol_rad:
                return
        raise TimeoutError(f"Robot did not reach home within {timeout_s:.0f}s.")

    def require_running(self) -> None:
        """Fail fast with a readable reason if the robot is not powered on."""
        try:
            _, _, mode = self.snapshot()
        except OSError as exc:
            raise ConnectionError(
                f"Cannot reach the robot at {self.host}:{RTDE_PORT} ({exc}). Is "
                "the simulator up (../simulation environment: docker compose up "
                "-d) or the robot on the network?"
            ) from exc
        if mode != ROBOT_MODE_RUNNING:
            raise RuntimeError(
                f"Robot is not powered on (mode {mode}, need {ROBOT_MODE_RUNNING}"
                "=RUNNING). Power it on + release brakes first (URSim: open "
                "http://localhost)."
            )


def record_run(robot: URRecorder, *, goal, speed, accel):
    """Execute one move and log it. Returns (times, target_q, actual_q).

    Streams RTDE while the non-blocking ``movej`` runs, then keeps sampling for
    ``SETTLE_S`` after the commanded setpoint reaches the goal, to capture the
    end-of-move vibration (the ring-down) that defines the gap.
    """
    goal_arr = np.asarray(goal, dtype=float)
    stream = robot._open_stream()
    try:
        robot.send_move(goal, speed, accel)
        t0 = time.monotonic()
        times, tgt, act = [], [], []
        settle_until = None
        while True:
            target, actual, _ = robot._read_sample(stream)
            now = time.monotonic()
            times.append(now - t0)
            tgt.append(target)
            act.append(actual)
            arrived = np.max(np.abs(np.asarray(target) - goal_arr)) < ARRIVE_TOL
            if arrived and settle_until is None:
                settle_until = now + SETTLE_S
            if settle_until is not None and now >= settle_until:
                break
    finally:
        stream.close()
    return np.array(times), np.array(tgt), np.array(act)


def write_csv(path, times, target_q, actual_q, *, speed, accel, blend, payload):
    """Write one run to CSV in the dataset.py schema."""
    header = [TIME_COL, *TARGET_Q_COLS, *ACTUAL_Q_COLS, *META_COLS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(times)):
            w.writerow([
                f"{times[i]:.6f}",
                *[f"{v:.6f}" for v in target_q[i]],
                *[f"{v:.6f}" for v in actual_q[i]],
                speed, accel, blend, payload,
            ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-ip", default=os.environ.get("UR_HOST", "127.0.0.1"),
                    help="UR controller IP (default: $UR_HOST or 127.0.0.1)")
    ap.add_argument("--out", default="data", help="output folder for run CSVs")
    ap.add_argument("--speed-home", type=float, default=0.8,
                    help="speed for returning to home between runs (rad/s)")
    args = ap.parse_args()

    robot = URRecorder(args.robot_ip)
    robot.require_running()  # clear error if unreachable or not powered on

    os.makedirs(args.out, exist_ok=True)
    moves = _build_moves()
    try:
        for payload in PAYLOAD_GRID:
            # NOTE: payload is logged as a run parameter; set the robot's TCP
            # payload in PolyScope / your installation to match before recording.
            for name, goal in moves.items():
                for speed in SPEED_GRID:
                    for accel in ACCEL_GRID:
                        robot.move_blocking(HOME, args.speed_home, 2.0)  # reset
                        times, tgt, act = record_run(
                            robot, goal=goal, speed=speed, accel=accel)
                        run_id = f"{name}_v{speed}_a{accel}_p{payload}"
                        path = os.path.join(args.out, f"{run_id}.csv")
                        write_csv(path, times, tgt, act, speed=speed,
                                  accel=accel, blend=0.0, payload=payload)
                        print(f"wrote {path}  ({len(times)} samples)")
    finally:
        robot.move_blocking(HOME, args.speed_home, 2.0)


if __name__ == "__main__":
    main()
