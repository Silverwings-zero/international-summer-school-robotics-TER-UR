"""Touch sensing for human-robot interaction (the HRI extension).

Turns the robot's force/current telemetry into discrete CONTACT EVENTS a
language model can read and interpret: someone tapped the tool, pushed the arm
toward +y, twisted the wrist, leaned on it. Two RTDE signals carry the
physics:

  * ``actual_TCP_force`` -- the controller's gravity-compensated estimate of
    the EXTERNAL wrench at the tool. A hand pushing or twisting shows up here.
  * current gap -- per-joint |actual - commanded| motor current. Contact
    anywhere on the arm loads a joint the controller did not plan for, and
    which joints deviate says where the touch landed.

The URSim simulator has no contact physics (both signals read zero there), so
the monitor also accepts INJECTED wrenches that flow through the very same
pipeline: classifier, buffer, and tools behave identically in simulation and
on a real UR10e -- only the source of the wrench differs.

Interpretation stays out of this file on purpose. The classifier names WHAT
happened (kind, direction, strength, where, when); deciding what the human
MEANT by it is the language model's job, through the MCP tools in server.py.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from ur_client import URClient

# --- Classification thresholds (hysteresis: onset above, release below) ---- #
FORCE_ONSET_N = 4.0      # a contact starts above this force ...
FORCE_RELEASE_N = 2.0    # ... and ends below this one
TORQUE_ONSET_NM = 1.5
TORQUE_RELEASE_NM = 0.7
TAP_MAX_S = 0.35         # shorter contact == tap
LEAN_MIN_S = 2.0         # sustained force this long == lean
DOUBLE_TAP_WINDOW_S = 0.8  # two taps this close merge into a double tap
GAP_ONSET_A = 0.8        # current gap that localises a touch to a joint

# Which arm segment a deviating joint implicates, base..wrist3.
_JOINT_SEGMENT = ("shoulder", "shoulder", "forearm", "wrist", "wrist", "tool")

_AXIS_VECTORS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0),
}


def _direction_word(vec: list[float]) -> str:
    """Dominant base-frame axis of a force vector, as a word."""
    axis = max(range(3), key=lambda i: abs(vec[i]))
    if axis == 2:
        return "up" if vec[2] > 0 else "down"
    return ("+" if vec[axis] > 0 else "-") + "xy"[axis]


@dataclass
class ContactEvent:
    """One classified touch, ready to be told to a language model."""

    kind: str          # tap | double_tap | push | twist | lean
    direction: str     # "+y", "down", "about +z (counter-clockwise)", ...
    magnitude: float   # peak N (force kinds) or Nm (twist)
    duration_s: float
    where: str         # tool | wrist | forearm | shoulder
    t_wall: float      # time.time() when the contact ended

    def words(self) -> str:
        unit = "Nm" if self.kind == "twist" else "N"
        prep = "" if self.kind == "twist" else "toward "
        return (f"{self.kind.replace('_', ' ')} on the {self.where}, "
                f"{prep}{self.direction}, {self.magnitude:.1f} {unit}, "
                f"{self.duration_s:.2f} s")


class ContactMonitor:
    """Watches the robot's wrench stream and classifies touch events.

    One instance per server. ``start()`` spawns a daemon thread that holds a
    persistent RTDE stream (reconnecting if it drops); every sample -- plus
    any active injected wrench -- runs through a small onset/release state
    machine whose committed events land in a bounded buffer.
    """

    def __init__(self, client: URClient, hz: float = 25.0,
                 keep_events: int = 50):
        self.client = client
        self.hz = hz
        self.keep_events = keep_events
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: list[ContactEvent] = []
        self._injections: list[tuple[float, float, list[float], str]] = []
        # Live view, refreshed every sample (guarded by _lock).
        self._live_wrench = [0.0] * 6
        self._active_since: float | None = None
        self._samples_seen = 0
        # Accumulators for the contact currently in progress.
        self._acc: dict | None = None
        # Baseline the real sensor drifts around (stays ~0 in the sim).
        self._baseline = [0.0] * 6

    # --- lifecycle ------------------------------------------------------- #
    def start(self) -> None:
        """Start the sampling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="contact-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for state in self.client.stream_states(self.hz):
                    self._process(state)
                    if self._stop.is_set():
                        break
            except (OSError, ConnectionError):
                time.sleep(1.0)  # controller away; retry the stream

    # --- injection (the simulator has no touch physics) ------------------ #
    def inject(self, force_n: list[float], torque_nm: list[float],
               duration_s: float, where: str = "tool",
               delay_s: float = 0.0) -> None:
        """Overlay a synthetic wrench for ``duration_s``, through the same
        classification pipeline a real touch takes."""
        t0 = time.monotonic() + delay_s
        wrench = list(force_n) + list(torque_nm)
        with self._lock:
            self._injections.append((t0, t0 + duration_s, wrench, where))

    # --- per-sample state machine ---------------------------------------- #
    def _process(self, state) -> None:
        now = time.monotonic()
        with self._lock:
            self._injections = [i for i in self._injections if i[1] > now]
            active_inj = [i for i in self._injections if i[0] <= now]
        wrench = list(state.tcp_force)
        inj_where = None
        for _, _, w, where in active_inj:
            wrench = [a + b for a, b in zip(wrench, w)]
            inj_where = where
        corrected = [a - b for a, b in zip(wrench, self._baseline)]
        fmag = math.sqrt(sum(v * v for v in corrected[:3]))
        tmag = math.sqrt(sum(v * v for v in corrected[3:]))

        in_contact = self._acc is not None
        if not in_contact:
            # Track the sensor's slow drift only while untouched.
            self._baseline = [b + 0.02 * (w - b)
                              for b, w in zip(self._baseline, wrench)]
            if fmag > FORCE_ONSET_N or tmag > TORQUE_ONSET_NM:
                self._acc = {
                    "t0_mono": now, "t0_wall": time.time(),
                    "f_sum": [0.0] * 3, "t_sum": [0.0] * 3,
                    "f_peak": 0.0, "t_peak": 0.0,
                    "where_votes": {}, "n": 0,
                }
        if self._acc is not None:
            acc = self._acc
            acc["n"] += 1
            acc["f_sum"] = [s + v for s, v in zip(acc["f_sum"], corrected[:3])]
            acc["t_sum"] = [s + v for s, v in zip(acc["t_sum"], corrected[3:])]
            acc["f_peak"] = max(acc["f_peak"], fmag)
            acc["t_peak"] = max(acc["t_peak"], tmag)
            where = inj_where or self._locate(state)
            acc["where_votes"][where] = acc["where_votes"].get(where, 0) + 1
            if fmag < FORCE_RELEASE_N and tmag < TORQUE_RELEASE_NM:
                self._commit(acc, now)
                self._acc = None

        with self._lock:
            self._live_wrench = [round(v, 3) for v in corrected]
            self._active_since = None if self._acc is None \
                else self._acc["t0_mono"]
            self._samples_seen += 1

    @staticmethod
    def _locate(state) -> str:
        """Which arm segment feels the touch, from the joint current gap."""
        gaps = state.current_gap_A
        worst = max(range(6), key=lambda i: gaps[i])
        if gaps[worst] < GAP_ONSET_A:
            return "tool"  # only the TCP sensor sees it (or simulator zeros)
        return _JOINT_SEGMENT[worst]

    def _commit(self, acc: dict, now: float) -> None:
        """Classify a finished contact and append it to the event buffer."""
        duration = now - acc["t0_mono"]
        n = max(1, acc["n"])
        f_mean = [v / n for v in acc["f_sum"]]
        t_mean = [v / n for v in acc["t_sum"]]
        torque_dominant = (acc["t_peak"] / TORQUE_ONSET_NM
                           > acc["f_peak"] / FORCE_ONSET_N)
        where = max(acc["where_votes"], key=acc["where_votes"].get)

        if torque_dominant:
            kind = "twist"
            axis = max(range(3), key=lambda i: abs(t_mean[i]))
            sign = "+" if t_mean[axis] > 0 else "-"
            spin = ""
            if axis == 2:
                spin = (" (counter-clockwise)" if t_mean[2] > 0
                        else " (clockwise)")
            direction = f"about {sign}{'xyz'[axis]}{spin}"
            magnitude = acc["t_peak"]
        else:
            direction = _direction_word(f_mean)
            magnitude = acc["f_peak"]
            if duration < TAP_MAX_S:
                kind = "tap"
            elif duration >= LEAN_MIN_S:
                kind = "lean"
            else:
                kind = "push"

        event = ContactEvent(kind=kind, direction=direction,
                             magnitude=round(magnitude, 1),
                             duration_s=round(duration, 2), where=where,
                             t_wall=time.time())
        with self._lock:
            # Two quick taps merge into one double tap.
            if (kind == "tap" and self._events
                    and self._events[-1].kind in ("tap", "double_tap")
                    and event.t_wall - self._events[-1].t_wall
                    <= DOUBLE_TAP_WINDOW_S):
                prev = self._events.pop()
                event = ContactEvent(
                    kind="double_tap", direction=event.direction,
                    magnitude=max(prev.magnitude, event.magnitude),
                    duration_s=round(event.t_wall - prev.t_wall
                                     + prev.duration_s, 2),
                    where=event.where, t_wall=event.t_wall)
            self._events.append(event)
            del self._events[:-self.keep_events]

    # --- queries (thread-safe) ------------------------------------------- #
    def snapshot(self) -> dict:
        """The live picture: monitoring health, current wrench, touch state."""
        with self._lock:
            active = self._active_since
            return {
                "monitoring": self._thread is not None
                and self._thread.is_alive(),
                "samples_seen": self._samples_seen,
                "touched_now": active is not None,
                "contact_duration_s": round(time.monotonic() - active, 2)
                if active is not None else 0.0,
                "force_n": self._live_wrench[:3],
                "torque_nm": self._live_wrench[3:],
                "last_event": self._events[-1].words()
                if self._events else None,
            }

    def events(self, since_s: float) -> list[ContactEvent]:
        """Committed events newer than ``since_s`` seconds, oldest first."""
        cutoff = time.time() - since_s
        with self._lock:
            return [e for e in self._events if e.t_wall >= cutoff]


# --- Gesture injection (simulator demos and tests) ------------------------- #
GESTURES = ("tap", "double_tap", "push", "twist", "lean")

# Pulse length per gesture, seconds -- chosen so the classifier re-derives
# the kind from the physics alone (tap < TAP_MAX_S < push < LEAN_MIN_S < lean).
_PULSE_S = {"tap": 0.15, "double_tap": 0.15, "push": 0.8, "lean": 2.6,
            "twist": 0.8}
_DOUBLE_TAP_SPACING_S = 0.5


def inject_gesture(monitor: ContactMonitor, kind: str, direction: str = "+y",
                   strength: float | None = None,
                   where: str = "tool") -> float:
    """Translate a named human gesture into raw wrench pulses.

    The injection carries no label: the monitor's classifier sees only the
    synthetic physics and must re-derive the kind, exactly as it would for a
    real touch. Returns the seconds until the last pulse ends.

    Raises:
        ValueError: Unknown gesture kind or direction.
    """
    if kind not in GESTURES:
        raise ValueError(
            f"Unknown gesture {kind!r}. Pick one of: {', '.join(GESTURES)}.")
    pulse = _PULSE_S[kind]

    if kind == "twist":
        spin = {"ccw": 1.0, "+z": 1.0, "cw": -1.0, "-z": -1.0}.get(direction)
        if spin is None:
            raise ValueError(
                f"Twist direction {direction!r} not understood. Use 'ccw' "
                "(counter-clockwise, about +z) or 'cw' (clockwise).")
        torque = [0.0, 0.0, spin * (strength or 3.0)]
        monitor.inject([0.0] * 3, torque, pulse, where)
        return pulse

    vec = _AXIS_VECTORS.get(direction)
    if vec is None:
        raise ValueError(
            f"Direction {direction!r} not understood. Use one of: "
            f"{', '.join(_AXIS_VECTORS)}.")
    force = [v * (strength or 12.0) for v in vec]
    monitor.inject(force, [0.0] * 3, pulse, where)
    if kind == "double_tap":
        monitor.inject(force, [0.0] * 3, pulse, where,
                       delay_s=_DOUBLE_TAP_SPACING_S)
        return _DOUBLE_TAP_SPACING_S + pulse
    return pulse
