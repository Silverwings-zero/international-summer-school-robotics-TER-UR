"""Robotiq gripper driver, spoken over the URCap socket server (port 63352).

Covers the Robotiq **Hand-E** (this cell's gripper), 2F-85 and 2F-140 -- they
share one register interface, differing only in stroke, force, and speed
ranges (see ``MODELS``).

When the Robotiq URCap is installed on a UR controller, it runs a small ASCII
command server on port 63352: ``GET <VAR>`` reads a register ("STA 3"), and
``SET <VAR> <value> ...`` writes one or more and answers ``ack``. This module
speaks that protocol directly over TCP -- pure standard library, same
philosophy as ``ur_client.py``: no vendor SDK to install, nothing to compile.

The registers that matter here (all integers):

  ACT  activation bit (1 = activate; the gripper self-calibrates on rising edge)
  GTO  go-to bit (1 = move to the requested position)
  POS  actual position, 0 = fully open .. 255 = fully closed
  SPE  speed 0..255, FOR force 0..255
  STA  activation status: 0 reset, 1 activating, 3 active
  PRE  echo of the last requested position (confirms a command registered)
  OBJ  motion status: 0 moving, 1 stopped by contact while OPENING,
       2 stopped by contact while CLOSING (an object is gripped),
       3 arrived at the requested position (nothing gripped)
  FLT  fault code, 0 = no fault

If port 63352 is closed, the Robotiq URCap is not installed or not running on
the controller -- there is no other network path to the gripper (it hangs off
the tool RS-485 bus, which only a URCap bridges). Callers (``server.py``)
treat a refused connection as "no Robotiq present" and fall back to the
digital-output convention, which is also what the simulator gets.
"""
from __future__ import annotations

import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field

GRIPPER_PORT = 63352

# Robotiq fault register decoded to words the LLM (and a student) can act on.
FAULT_NAMES = {
    0: "NO_FAULT",
    5: "ACTION_DELAYED_ACTIVATION_NEEDED",
    7: "ACTIVATION_BIT_NOT_SET",
    8: "MAX_TEMPERATURE_EXCEEDED",
    9: "NO_COMMUNICATION_WITH_GRIPPER",
    10: "UNDER_MINIMUM_VOLTAGE",
    11: "AUTO_RELEASE_IN_PROGRESS",
    12: "INTERNAL_FAULT",
    13: "ACTIVATION_FAULT",
    14: "OVERCURRENT",
    15: "AUTO_RELEASE_COMPLETED",
}

# Faults that mean "the controller's URCap is talking, but the gripper itself
# is not" -- the message to the operator is about cabling, not activation.
HARDWARE_FAULTS = {9, 10, 12, 14}

# OBJ register values (see module docstring).
OBJ_MOVING = 0
OBJ_STOPPED_OPENING = 1
OBJ_STOPPED_CLOSING = 2
OBJ_AT_POSITION = 3

_STA_ACTIVE = 3

# A reply is "ack", "<VAR> <int>", or "<VAR> ?" -- the URCap answers with a
# question mark when it cannot read the register at all, which is what an
# unpowered or unplugged gripper looks like from the controller's side. The
# daemon does not always terminate replies with a newline, so both framings
# are accepted.
_REPLY_RE = re.compile(r"^(?:ack|[A-Z]{3} (?:-?\d+|\?))$", re.IGNORECASE)

# How long an unterminated reply must stay quiet before it counts as
# complete. Without a terminator "PRE 2" is indistinguishable from the first
# five bytes of "PRE 200", so a settle window is the only safe way to tell.
_SETTLE_S = 0.03


@dataclass(frozen=True)
class GripperModel:
    """Physical ranges of one Robotiq model, for unit conversion and advice."""

    name: str
    stroke_mm: float       # finger opening at POS 0 (fully open)
    min_force_n: float     # grip force at FOR 0
    max_force_n: float     # grip force at FOR 255
    min_speed_mm_s: float  # closing speed at SPE 0
    max_speed_mm_s: float  # closing speed at SPE 255
    mass_kg: float         # gripper mass, for set_payload

    def opening_mm(self, position: int) -> float:
        """Finger opening in millimetres for a raw position register."""
        return round(self.stroke_mm * (1.0 - position / 255.0), 1)

    def force_n(self, force: int) -> float:
        span = self.max_force_n - self.min_force_n
        return round(self.min_force_n + span * force / 255.0, 1)


MODELS = {
    # Hand-E: parallel two-finger, 50 mm stroke, flat interchangeable fingers.
    "hand-e": GripperModel("Hand-E", 50.0, 20.0, 185.0, 20.0, 150.0, 1.0),
    "2f-85": GripperModel("2F-85", 85.0, 20.0, 235.0, 20.0, 150.0, 0.9),
    "2f-140": GripperModel("2F-140", 140.0, 10.0, 125.0, 30.0, 250.0, 1.0),
}

# This cell runs a Hand-E; override per cell with ROBOTIQ_MODEL.
DEFAULT_MODEL = MODELS[os.environ.get("ROBOTIQ_MODEL", "hand-e").strip().lower()]


class GripperError(RuntimeError):
    """The gripper answered, but not the way a healthy Robotiq would."""


class GripperNotRespondingError(GripperError):
    """The URCap is alive but is not talking to the gripper itself.

    Distinct from a missing URCap (an OSError on connect) and from a
    protocol error: the daemon is fine, the gripper is not answering it --
    almost always no 24 V on the tool connector, or an unseated cable.
    """


@dataclass
class GripperStatus:
    """One decoded snapshot of the gripper registers."""

    activated: bool          # STA == 3: calibrated and ready for motion
    moving: bool             # OBJ == 0: fingers still travelling
    object_detected: bool    # fingers stopped early on contact (OBJ 1 or 2)
    object_status: int       # raw OBJ value, see module docstring
    position: int            # actual position, 0 open .. 255 closed
    position_pct: float      # the same as a percentage, 0.0 open .. 100.0 closed
    opening_mm: float        # actual finger opening in mm (model-specific)
    requested_position: int  # PRE: last position the gripper accepted
    commanded_closed: bool   # PRE past half travel: what was last asked for
    fault: int               # raw FLT value
    fault_name: str          # FLT decoded, "NO_FAULT" when healthy
    model: str               # which Robotiq the mm/N conversions assume

    def as_dict(self) -> dict:
        return {
            "activated": self.activated,
            "moving": self.moving,
            "object_detected": self.object_detected,
            "object_status": self.object_status,
            "position": self.position,
            "position_pct": self.position_pct,
            "opening_mm": self.opening_mm,
            "requested_position": self.requested_position,
            "commanded_closed": self.commanded_closed,
            "fault": self.fault,
            "fault_name": self.fault_name,
            "model": self.model,
        }


@dataclass
class RobotiqGripper:
    """Connection to the Robotiq URCap command server on the UR controller.

    Args:
        host: Robot IP (the URCap listens on the controller, not the gripper).
            Defaults to the ``UR_HOST`` env var, then to the local simulator.
        port: URCap command port, 63352 unless reconfigured.
        timeout_s: Per-exchange socket budget.
        connect_timeout_s: Budget for opening the socket -- shorter, because a
            missing URCap should be reported quickly, not waited on.
        model: Physical model, for mm/newton conversions.
    """

    host: str = field(
        default_factory=lambda: os.environ.get("UR_HOST", "127.0.0.1"))
    port: int = GRIPPER_PORT
    timeout_s: float = 2.0
    # Long enough to let a UR controller answer a closed port with a proper
    # reset (measured ~1.0 s on the UR5e): a shorter budget turns "no URCap
    # installed" into a misleading "timed out".
    connect_timeout_s: float = 1.5
    model: GripperModel = DEFAULT_MODEL
    _sock: socket.socket | None = field(default=None, repr=False)
    _buf: bytes = field(default=b"", repr=False)
    # Guards one request/reply exchange and the socket itself. Re-entrant so
    # _call's error path can close() while already holding it.
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False)
    # Guards a whole logical operation (activate/move/status), which is many
    # exchanges: without it two concurrent tool calls interleave their
    # SET POS / PRE-poll sequences and each reads the other's motion.
    _op_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------------ #
    # Connection plumbing
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Open the command socket. Raises OSError if nothing listens there
        (no URCap installed -- i.e. no Robotiq reachable on this robot)."""
        with self._lock:
            if self._sock is not None:
                return
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_s)
            sock.settimeout(self.timeout_s)
            self._sock, self._buf = sock, b""

    def close(self) -> None:
        """Close the socket. Safe to call from another thread: it waits for
        any in-flight exchange instead of pulling the fd out from under it."""
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                finally:
                    self._sock, self._buf = None, b""

    def _read_more(self, timeout_s: float) -> bool:
        """Append one chunk to the buffer. False if nothing arrived in time."""
        assert self._sock is not None
        self._sock.settimeout(max(0.001, timeout_s))
        try:
            chunk = self._sock.recv(1024)
        except TimeoutError:
            return False
        finally:
            self._sock.settimeout(self.timeout_s)
        if not chunk:
            raise GripperError(
                "Gripper server closed the connection mid-command.")
        self._buf += chunk
        return True

    def _recv_reply(self) -> str:
        """Read exactly one reply, tolerating both framings and TCP splits."""
        assert self._sock is not None
        deadline = time.monotonic() + self.timeout_s
        while True:
            # A newline always terminates a reply when the URCap sends one.
            if b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                return line.decode("ascii", errors="replace").strip()
            text = self._buf.decode("ascii", errors="replace").strip()
            if _REPLY_RE.match(text):
                # The buffer forms a complete token, but with no terminator
                # it may still be a PREFIX of the real reply ("PRE 2" of
                # "PRE 200"), so accept it only once the line goes quiet.
                if not self._read_more(_SETTLE_S):
                    self._buf = b""
                    return text
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._read_more(remaining):
                raise GripperError(
                    f"Timed out waiting for a gripper reply (got {text!r}).")

    def _call(self, command: str) -> str:
        """One request/reply exchange. Drops the socket on ANY error -- a
        half-read reply would desync every later exchange on it."""
        with self._lock:
            try:
                self.connect()
                assert self._sock is not None
                self._sock.sendall(command.encode("ascii") + b"\n")
                return self._recv_reply()
            except (OSError, GripperError):
                self.close()
                raise

    # ------------------------------------------------------------------ #
    # Register access
    # ------------------------------------------------------------------ #
    def get(self, variable: str) -> int:
        """Read one register, e.g. ``get("STA") -> 3``."""
        reply = self._call(f"GET {variable}")
        parts = reply.split()
        if len(parts) != 2 or parts[0].upper() != variable.upper():
            self.close()  # unparseable reply: assume the stream is desynced
            raise GripperError(
                f"Unexpected reply to GET {variable}: {reply!r}")
        if parts[1] == "?":
            # The URCap is alive but has no value for the register: it is not
            # talking to the gripper itself.
            raise GripperNotRespondingError(
                f"The Robotiq URCap is running but cannot read {variable} "
                "-- it is not communicating with the gripper. Check that the "
                "tool connector supplies 24 V (Installation > General > Tool "
                "I/O) and that the gripper cable is seated at the flange.")
        return int(parts[1])

    def set(self, **variables: int) -> None:
        """Write registers in one command, e.g. ``set(POS=255, GTO=1)``."""
        pairs = " ".join(f"{k} {int(v)}" for k, v in variables.items())
        reply = self._call(f"SET {pairs}")
        if reply.lower() != "ack":
            raise GripperError(f"Gripper refused SET {pairs}: {reply!r}")

    def _fault_suffix(self) -> str:
        """Read FLT and describe it, for appending to a failure message."""
        try:
            flt = self.get("FLT")
        except (OSError, GripperError):
            return ""
        if not flt:
            return ""
        name = FAULT_NAMES.get(flt, f"UNKNOWN_FAULT_{flt}")
        if flt in HARDWARE_FAULTS:
            return (f" Gripper reports {name}: check the tool cable and that "
                    "the gripper is powered (24 V on the tool connector).")
        return f" Gripper reports fault {name}."

    # ------------------------------------------------------------------ #
    # High-level operations
    # ------------------------------------------------------------------ #
    def status(self) -> GripperStatus:
        """Read and decode everything a caller needs to reason about."""
        with self._op_lock:
            sta = self.get("STA")
            obj = self.get("OBJ")
            pos = self.get("POS")
            pre = self.get("PRE")
            flt = self.get("FLT")
        return GripperStatus(
            activated=sta == _STA_ACTIVE,
            moving=obj == OBJ_MOVING,
            object_detected=obj in (OBJ_STOPPED_OPENING, OBJ_STOPPED_CLOSING),
            object_status=obj,
            position=pos,
            position_pct=round(pos / 255.0 * 100.0, 1),
            opening_mm=self.model.opening_mm(pos),
            requested_position=pre,
            commanded_closed=pre > 128,
            fault=flt,
            fault_name=FAULT_NAMES.get(flt, f"UNKNOWN_FAULT_{flt}"),
            model=self.model.name,
        )

    def activate(self, timeout_s: float = 12.0) -> GripperStatus:
        """Run the activation cycle if needed and wait until the gripper is
        ready. On first activation the fingers sweep their full travel to
        self-calibrate -- keep the workspace in front of them clear.

        Raises:
            GripperError: activation did not complete inside ``timeout_s``,
                or the gripper reports a fault.
        """
        with self._op_lock:
            if self.get("STA") == _STA_ACTIVE:
                return self.status()
            # Rising edge on ACT starts the calibration sweep; clear it first
            # so a half-activated gripper restarts cleanly.
            self.set(ACT=0)
            deadline = time.monotonic() + timeout_s
            while self.get("STA") != 0:
                if time.monotonic() > deadline:
                    raise GripperError(
                        "Gripper did not reset for activation."
                        + self._fault_suffix())
                time.sleep(0.1)
            self.set(ACT=1)
            while self.get("STA") != _STA_ACTIVE:
                if time.monotonic() > deadline:
                    raise GripperError(
                        "Gripper activation timed out."
                        + self._fault_suffix()
                        + " Is a Robotiq physically connected to the tool "
                        "flange?")
                time.sleep(0.1)
            return self.status()

    def move(self, position: int, speed: int = 128, force: int = 64,
             wait: bool = True, timeout_s: float = 10.0) -> GripperStatus:
        """Command the fingers to ``position`` (0 open .. 255 closed).

        With ``wait`` (the default) this blocks until the fingers either
        arrive or stop early on contact, so the returned status's
        ``object_detected`` is meaningful. Raises GripperError if the
        gripper is not activated.
        """
        position = max(0, min(255, int(position)))
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))
        with self._op_lock:
            if self.get("STA") != _STA_ACTIVE:
                raise GripperError(
                    "Gripper is not activated -- run the activation cycle "
                    "first." + self._fault_suffix())
            self.set(POS=position, SPE=speed, FOR=force, GTO=1)
            if not wait:
                return self.status()
            deadline = time.monotonic() + timeout_s
            # First wait for the command to register (PRE echoes the request),
            # then for the motion to end (OBJ leaves the "moving" state).
            while self.get("PRE") != position:
                if time.monotonic() > deadline:
                    raise GripperError(
                        "Gripper never accepted the position request (PRE did "
                        "not update)." + self._fault_suffix())
                time.sleep(0.05)
            while self.get("OBJ") == OBJ_MOVING:
                if time.monotonic() > deadline:
                    raise GripperError(
                        "Gripper motion did not finish in time."
                        + self._fault_suffix())
                time.sleep(0.05)
            return self.status()


def probe(host: str, port: int = GRIPPER_PORT,
          timeout_s: float = 1.0) -> bool:
    """True if something is listening on the URCap gripper port -- the
    cheap "is a Robotiq present?" test that never raises."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False
