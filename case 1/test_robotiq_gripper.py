"""Tests for the Robotiq driver against a FAKE URCap server -- no hardware.

The fake emulates the register machine the real URCap exposes on port 63352:
activation (STA 0 -> 1 -> 3), motion (OBJ 0 while moving, then 3 on arrival
or 2 when a virtual object blocks the fingers), the PRE echo, and faults. It
can also reply in the awkward ways a real daemon does -- without a trailing
newline, or split across TCP segments -- so the framing logic is covered.

    python test_robotiq_gripper.py

Prereqs: none. This runs anywhere, including CI.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robotiq_gripper import (  # noqa: E402
    MODELS,
    OBJ_AT_POSITION,
    OBJ_STOPPED_CLOSING,
    GripperError,
    RobotiqGripper,
    probe,
)


class FakeURCap(threading.Thread):
    """A minimal stand-in for the Robotiq URCap command server.

    Args:
        object_at_pos: If set, the fingers stop at this position when asked
            to close past it -- the "an object is gripped" case.
        framing: "newline" (POS 0\\n), "bare" (no terminator), or "split"
            (bare, delivered one byte at a time).
        fault: Value the FLT register reports.
    """

    def __init__(self, object_at_pos: int | None = None,
                 framing: str = "newline", fault: int = 0):
        super().__init__(daemon=True)
        self.regs = {"ACT": 0, "GTO": 0, "POS": 0, "PRE": 0, "SPE": 0,
                     "FOR": 0, "STA": 0, "OBJ": OBJ_AT_POSITION, "FLT": fault}
        self.object_at_pos = object_at_pos
        self.framing = framing
        self.commands: list[str] = []
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.port = self.srv.getsockname()[1]
        self._target = 0
        self._activated_at = 0.0
        self._move_started = 0.0
        self._lock = threading.Lock()

    def run(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,),
                             daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        buf = b""
        while True:
            try:
                data = conn.recv(1024)
            except OSError:
                return
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._send(conn, self.handle(line.decode().strip()))

    def _send(self, conn: socket.socket, reply: str) -> None:
        if self.framing == "newline":
            conn.sendall(reply.encode() + b"\n")
        elif self.framing == "bare":
            conn.sendall(reply.encode())
        else:  # split: one byte at a time, no terminator
            for byte in reply.encode():
                conn.sendall(bytes([byte]))
                time.sleep(0.002)

    def handle(self, line: str) -> str:
        with self._lock:
            self.commands.append(line)
            parts = line.split()
            if parts[0] == "GET":
                self._tick()
                return f"{parts[1]} {self.regs[parts[1]]}"
            if parts[0] == "SET":
                for key, value in zip(parts[1::2], parts[2::2]):
                    self.regs[key] = int(value)
                    if key == "ACT":
                        self.regs["STA"] = 1 if int(value) else 0
                        self._activated_at = time.monotonic()
                    elif key == "POS":
                        self.regs["PRE"] = int(value)
                        self._target = int(value)
                        self.regs["OBJ"] = 0
                        self._move_started = time.monotonic()
                return "ack"
            return "ERR"

    def _tick(self) -> None:
        """Advance the simulated gripper; called on every register read."""
        if (self.regs["STA"] == 1
                and time.monotonic() - self._activated_at > 0.2):
            self.regs["STA"] = 3
        if (self.regs["OBJ"] == 0
                and time.monotonic() - self._move_started > 0.2):
            if (self.object_at_pos is not None
                    and self._target > self.object_at_pos):
                self.regs["POS"] = self.object_at_pos
                self.regs["OBJ"] = OBJ_STOPPED_CLOSING
            else:
                self.regs["POS"] = self._target
                self.regs["OBJ"] = OBJ_AT_POSITION


def connect_to(fake: FakeURCap, **kwargs) -> RobotiqGripper:
    return RobotiqGripper(host="127.0.0.1", port=fake.port, **kwargs)


def test_activation_and_motion() -> None:
    fake = FakeURCap()
    fake.start()
    gripper = connect_to(fake)

    try:
        gripper.move(255)
        raise AssertionError("move before activation must raise")
    except GripperError as exc:
        assert "not activated" in str(exc), exc

    status = gripper.activate()
    assert status.activated, status
    # Already active: a second call is a cheap no-op, not another sweep.
    before = len(fake.commands)
    gripper.activate()
    assert len(fake.commands) - before < 8, "re-activation swept again"

    status = gripper.move(255, speed=200, force=100)
    assert status.position == 255 and not status.object_detected
    assert status.object_status == OBJ_AT_POSITION
    assert status.commanded_closed is True
    status = gripper.move(0)
    assert status.position == 0 and status.commanded_closed is False
    print("activation + motion: ok")


def test_object_detection_and_units() -> None:
    """Closing onto an object stops early; mm come from the model table."""
    fake = FakeURCap(object_at_pos=140)
    fake.start()
    gripper = connect_to(fake)
    gripper.activate()
    status = gripper.move(255)
    assert status.object_detected and status.object_status == OBJ_STOPPED_CLOSING
    assert status.position == 140
    # Hand-E: 50 mm stroke, so position 140/255 leaves ~22.5 mm of opening.
    hand_e = MODELS["hand-e"]
    assert status.model == "Hand-E", status.model
    assert abs(status.opening_mm - hand_e.opening_mm(140)) < 0.01
    assert 22.0 < status.opening_mm < 23.0, status.opening_mm
    # A 2F-140 would read the same register as a much wider opening.
    wide = connect_to(fake, model=MODELS["2f-140"]).status()
    assert wide.opening_mm > 60.0, wide.opening_mm
    print(f"object detection + units: ok ({status.opening_mm} mm on Hand-E, "
          f"{wide.opening_mm} mm on 2F-140)")


def test_reply_framing() -> None:
    """Replies without a newline, and split across segments, still parse."""
    for framing in ("bare", "split"):
        fake = FakeURCap(framing=framing)
        fake.start()
        gripper = connect_to(fake)
        gripper.activate()
        status = gripper.move(200)
        assert status.position == 200, (framing, status)
        print(f"framing {framing!r}: ok")


def test_concurrent_moves_do_not_interleave() -> None:
    """Two threads commanding the gripper must not read each other's motion.

    Without an operation-level lock, thread B's SET POS lands between A's
    SET POS and A's PRE poll, and A either times out or reports B's result.
    """
    fake = FakeURCap()
    fake.start()
    gripper = connect_to(fake)
    gripper.activate()

    results: list = []

    def close_it():
        try:
            results.append(("close", gripper.move(255).position))
        except Exception as exc:  # noqa: BLE001 - recorded, asserted below
            results.append(("close", exc))

    def open_it():
        try:
            results.append(("open", gripper.move(0).position))
        except Exception as exc:  # noqa: BLE001
            results.append(("open", exc))

    threads = [threading.Thread(target=close_it),
               threading.Thread(target=open_it)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert len(results) == 2, results
    for label, value in results:
        assert not isinstance(value, Exception), f"{label} raised {value}"
    # Each call must observe the position IT asked for: serialization means
    # the second operation starts only after the first finished.
    by_label = dict(results)
    assert by_label["close"] == 255, results
    assert by_label["open"] == 0, results
    print("concurrent moves serialized: ok")


def test_unreadable_register() -> None:
    """A live URCap that cannot reach the gripper answers "<VAR> ?".

    Seen on the real UR5e with the tool connector at 0 V: the daemon accepts
    the connection and replies, but has no register values to give.
    """
    class SilentGripperURCap(FakeURCap):
        def handle(self, line: str) -> str:
            reply = super().handle(line)
            return f"{line.split()[1]} ?" if line.startswith("GET") else reply

    fake = SilentGripperURCap()
    fake.start()
    gripper = connect_to(fake)
    try:
        gripper.status()
        raise AssertionError("an unreadable register must raise")
    except GripperError as exc:
        assert "not communicating with the gripper" in str(exc), exc
        assert "24 V" in str(exc), "the message must name the likely cause"
    print("unreadable register ('?'): ok")


def test_probe_and_missing_urcap() -> None:
    fake = FakeURCap()
    fake.start()
    assert probe("127.0.0.1", fake.port) is True
    # A port nobody listens on: probe says no, connect raises OSError, which
    # is exactly what server.py treats as "no Robotiq present".
    with socket.create_server(("127.0.0.1", 0)) as free:
        dead_port = free.getsockname()[1]
    assert probe("127.0.0.1", dead_port, timeout_s=0.5) is False
    try:
        RobotiqGripper(host="127.0.0.1", port=dead_port).connect()
        raise AssertionError("connect to a closed port must raise OSError")
    except OSError:
        pass
    print("probe + missing URCap: ok")


if __name__ == "__main__":
    test_activation_and_motion()
    test_object_detection_and_units()
    test_reply_framing()
    test_concurrent_moves_do_not_interleave()
    test_unreadable_register()
    test_probe_and_missing_urcap()
    print("ALL PASSED")
