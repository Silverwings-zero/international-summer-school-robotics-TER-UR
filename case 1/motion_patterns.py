"""Continuous, speed-adjustable motion patterns (stirring, sweeping, folding).

A pattern is an OPEN-ENDED motion: it has no natural end, and the operator
steers it while it runs -- "faster", "stop", "keep going". A blocking tool
cannot express that. While an MCP tool call is pending the LLM is not running,
so the operator's next words never reach it: the conversation is frozen for as
long as the robot moves. Every entry point here therefore returns immediately,
and the motion itself lives in a URScript loop on the controller.

Control works by PREEMPTION: uploading a new script to the primary interface
replaces the running program. Measured on PolyScope X, the swap stops the arm
in <=0.3 s and never raised a protective stop in ~15 attempts, so "faster" is
just the same loop re-uploaded with a new speed, resuming from the phase the
tool is currently at.

Two script shapes, chosen per pattern:

  * Circles use ``movec``, a true arc. The commanded speed is honoured
    (measured 94-96% of it from 0.08 to 0.40 m/s) and the traced radius is
    exact (30.00 mm commanded, 30.00 mm measured, 0.00 mm spread).

  * Any other pattern is sampled into a polygon of blended ``movel`` segments.
    There the BLEND CORNERS, not the speed argument, set the ceiling: their
    local radius is a few millimetres, so the controller caps tangential speed
    near sqrt(accel * blend_radius). Commanding 0.128 m/s instead of 0.080 m/s
    on such a path measured 0.0715 vs 0.0718 m/s -- no change at all. Speed on
    polygon patterns is governed by acceleration, which is why
    :func:`_derive_acceleration` scales it with the requested speed.

The controller enforces no centripetal limit of its own: 0.40 m/s on a 30 mm
circle ran happily while demanding 5.3 m/s^2 with a=2.5 commanded. Keeping the
contents inside the pan is this module's job, via :func:`max_speed_for_radius`.
"""
from __future__ import annotations

import atexit
import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from ur_client import PRIMARY_PORT, ROBOT_MODE_RUNNING, URClient

# Tangential speed bounds for pattern motion. Deliberately below a free-space
# transit: the tool is inside a container and usually carrying something.
MAX_PATTERN_SPEED_M_S = 0.35
MIN_PATTERN_SPEED_M_S = 0.01

# Centripetal acceleration the contents are assumed to tolerate. On small radii
# this, not the speed cap, is the binding constraint: v_max = sqrt(CAP * r), so
# a 42.5 mm stirring circle tops out near 0.23 m/s.
CENTRIPETAL_ACCEL_CAP_M_S2 = 1.25

# Path acceleration bounds for the generated scripts.
MIN_PATH_ACCEL_M_S2 = 0.6
MAX_PATH_ACCEL_M_S2 = 2.5

# Deceleration used by stopl when pausing or finishing.
STOP_DECEL_M_S2 = 3.0

# One lap as this many movec arcs. Each arc is defined by three points, and
# four arcs reproduced the commanded radius to 0.01 mm in testing.
ARCS_PER_LAP = 4

# Sampling density for polygon patterns, and for the pre-flight safety sweep.
DEFAULT_POINTS_PER_LAP = 24
SAFETY_SAMPLES_PER_CYCLE = 72

# A blend radius must stay below half the shortest adjoining segment or the
# controller rejects it; 0.4 leaves margin for rounding.
BLEND_FRACTION_OF_CHORD = 0.4

# Skip the lead-in move when the tool is already this close to the entry point.
LEAD_IN_TOLERANCE_M = 0.002

# One upload in ~15 started and then sat still, apparently accepted while the
# previous stop was still settling. Motion is verified after every upload and
# retried once.
MOTION_START_TIMEOUT_S = 2.5
STATIONARY_TIMEOUT_S = 4.0


# --------------------------------------------------------------------------- #
# Patterns: a parametric curve, phase -> offset from the anchor pose.
# Adding one is a single function plus a registry entry; nothing else changes.
# --------------------------------------------------------------------------- #
def _circle(phase: float, p: dict) -> tuple[float, float, float]:
    r = p["radius_m"]
    return (r * math.cos(phase), r * math.sin(phase), 0.0)


def _figure_eight(phase: float, p: dict) -> tuple[float, float, float]:
    r = p["radius_m"]
    return (r * math.sin(phase), r * math.sin(phase) * math.cos(phase), 0.0)


def _spiral(phase: float, p: dict) -> tuple[float, float, float]:
    """Spiral out to the full radius, then back in, so the path never jumps."""
    half = 2 * math.pi * p["turns"]
    frac = (phase % (2 * half)) / half
    grow = frac if frac <= 1.0 else 2.0 - frac
    r = p["radius_m"] * grow
    return (r * math.cos(phase), r * math.sin(phase), 0.0)


def _linear_sweep(phase: float, p: dict) -> tuple[float, float, float]:
    return (p["radius_m"] * math.sin(phase), 0.0, 0.0)


@dataclass(frozen=True)
class Pattern:
    """One motion shape and how it should be turned into URScript."""

    name: str
    summary: str
    offset: Callable[[float, dict], tuple[float, float, float]]
    # True only for the exact circle, which movec can trace natively.
    exact_arcs: bool
    # Total phase of one full cycle, given the parameters.
    cycle_phase: Callable[[dict], float]
    extra_params: tuple[str, ...] = ()


PATTERNS: dict[str, Pattern] = {
    "circle": Pattern(
        name="circle",
        summary="A closed circle around the anchor. The stirring motion.",
        offset=_circle,
        exact_arcs=True,
        cycle_phase=lambda p: 2 * math.pi,
    ),
    "figure_eight": Pattern(
        name="figure_eight",
        summary="A lying figure of eight, for folding rather than swirling.",
        offset=_figure_eight,
        exact_arcs=False,
        cycle_phase=lambda p: 2 * math.pi,
    ),
    "spiral": Pattern(
        name="spiral",
        summary="Spirals from the centre out to the full radius and back in.",
        offset=_spiral,
        exact_arcs=False,
        cycle_phase=lambda p: 4 * math.pi * p["turns"],
        extra_params=("turns",),
    ),
    "linear_sweep": Pattern(
        name="linear_sweep",
        summary="Back and forth along one axis, for scraping or levelling.",
        offset=_linear_sweep,
        exact_arcs=False,
        cycle_phase=lambda p: 2 * math.pi,
    ),
}


# --------------------------------------------------------------------------- #
# Geometry and limits
# --------------------------------------------------------------------------- #
def _pose_at(anchor: list[float], offset: tuple[float, float, float]) -> list[float]:
    """Anchor pose displaced by an offset; orientation is never touched."""
    return [anchor[0] + offset[0], anchor[1] + offset[1], anchor[2] + offset[2],
            anchor[3], anchor[4], anchor[5]]


def _fmt_pose(pose: list[float]) -> str:
    return "p[" + ", ".join(f"{v:.6f}" for v in pose) + "]"


def _cycle_points(pattern: Pattern, params: dict, count: int,
                  start_phase: float = 0.0,
                  direction: int = 1) -> list[tuple[float, float, float]]:
    """Sample one full cycle of the pattern as offsets."""
    span = pattern.cycle_phase(params)
    return [pattern.offset(start_phase + direction * span * i / count, params)
            for i in range(count)]


def _cycle_length_m(pattern: Pattern, params: dict) -> float:
    """Path length of one cycle, used for lap counting and time estimates."""
    pts = _cycle_points(pattern, params, SAFETY_SAMPLES_PER_CYCLE)
    closed = pts + [pts[0]]
    return sum(math.dist(a, b) for a, b in zip(closed, closed[1:]))


def max_speed_for_radius(radius_m: float) -> float:
    """Highest tangential speed whose centripetal load stays within the cap."""
    return min(MAX_PATTERN_SPEED_M_S,
               math.sqrt(CENTRIPETAL_ACCEL_CAP_M_S2 * radius_m))


def _derive_acceleration(speed_m_s: float, radius_m: float) -> float:
    """Path acceleration to give the script.

    Generous relative to the centripetal demand, because an acceleration set
    too close to v^2/r is what silently throttles the motion: the controller
    slows down to respect it instead of refusing the command.
    """
    demand = 3.0 * speed_m_s * speed_m_s / max(radius_m, 1e-4)
    return min(MAX_PATH_ACCEL_M_S2, max(MIN_PATH_ACCEL_M_S2, demand))


def _validate_speed(speed_m_s: float, radius_m: float) -> None:
    if not MIN_PATTERN_SPEED_M_S <= speed_m_s <= MAX_PATTERN_SPEED_M_S:
        raise ValueError(
            f"Speed {speed_m_s:.3f} m/s is outside the allowed range "
            f"[{MIN_PATTERN_SPEED_M_S}, {MAX_PATTERN_SPEED_M_S}] m/s."
        )
    limit = max_speed_for_radius(radius_m)
    if speed_m_s > limit:
        centripetal = speed_m_s * speed_m_s / radius_m
        raise ValueError(
            f"Speed {speed_m_s:.3f} m/s on a {radius_m * 1000:.0f} mm radius "
            f"would pull {centripetal:.2f} m/s^2 sideways, above the "
            f"{CENTRIPETAL_ACCEL_CAP_M_S2} m/s^2 limit that keeps the contents "
            f"in the container. The fastest safe speed here is "
            f"{limit:.3f} m/s ({limit / (2 * math.pi * radius_m) * 60:.0f} "
            "laps per minute)."
        )


# --------------------------------------------------------------------------- #
# Script generation
# --------------------------------------------------------------------------- #
def _lead_in(anchor: list[float], entry: list[float], speed: float,
             accel: float, current_tcp: list[float]) -> str:
    """Straight move onto the path, skipped when the tool is already there.

    A pattern starts with the tool at the anchor (the centre of the pan), which
    is not on the path; on resume it already is. Emitting a zero-length movel
    would be a degenerate command, hence the tolerance.
    """
    if math.dist(current_tcp[:3], entry[:3]) <= LEAD_IN_TOLERANCE_M:
        return ""
    return f"  movel({_fmt_pose(entry)}, a={accel:.4f}, v={speed:.4f})\n"


def _circle_loop(anchor: list[float], radius_m: float, speed: float,
                 accel: float, start_phase: float, direction: int) -> str:
    """A never-ending true circle, as ARCS_PER_LAP movec segments."""
    lines = ["  while True:\n"]
    step = 2 * math.pi / ARCS_PER_LAP
    for i in range(ARCS_PER_LAP):
        base = start_phase + direction * step * i
        via = _pose_at(anchor, _circle(base + direction * step / 2, {"radius_m": radius_m}))
        end = _pose_at(anchor, _circle(base + direction * step, {"radius_m": radius_m}))
        lines.append(f"    movec({_fmt_pose(via)}, {_fmt_pose(end)}, "
                     f"a={accel:.4f}, v={speed:.4f}, r=0.0)\n")
    lines.append("  end\n")
    return "".join(lines)


def _polygon_loop(anchor: list[float], pattern: Pattern, params: dict,
                  speed: float, accel: float, start_phase: float,
                  direction: int, points_per_lap: int) -> str:
    """A never-ending sampled path, as blended movel segments.

    Blend radii are per-segment because patterns are not evenly sampled in
    space: linear_sweep bunches its points at the turning points, and a blend
    wider than half the shortest adjoining segment is rejected outright.
    """
    offsets = _cycle_points(pattern, params, points_per_lap, start_phase, direction)
    poses = [_pose_at(anchor, off) for off in offsets]
    ring = poses + [poses[0]]
    lengths = [math.dist(a[:3], b[:3]) for a, b in zip(ring, ring[1:])]

    lines = ["  while True:\n"]
    for i, pose in enumerate(poses):
        incoming = lengths[i - 1]
        outgoing = lengths[i]
        blend = BLEND_FRACTION_OF_CHORD * min(incoming, outgoing)
        lines.append(f"    movel({_fmt_pose(pose)}, a={accel:.4f}, "
                     f"v={speed:.4f}, r={blend:.4f})\n")
    lines.append("  end\n")
    return "".join(lines)


def _build_script(anchor: list[float], pattern: Pattern, params: dict,
                  speed: float, accel: float, start_phase: float,
                  direction: int, points_per_lap: int,
                  current_tcp: list[float]) -> str:
    entry = _pose_at(anchor, pattern.offset(start_phase, params))
    body = _lead_in(anchor, entry, speed, accel, current_tcp)
    if pattern.exact_arcs:
        body += _circle_loop(anchor, params["radius_m"], speed, accel,
                             start_phase, direction)
    else:
        body += _polygon_loop(anchor, pattern, params, speed, accel,
                              start_phase, direction, points_per_lap)
    return "def motion_pattern():\n" + body + "end\n"


# --------------------------------------------------------------------------- #
# Script session: owns one uploaded program and keeps its socket alive
# --------------------------------------------------------------------------- #
class _ScriptSession:
    """One uploaded URScript program.

    PolyScope X discards a freshly uploaded script if the uploader disconnects
    around program start, so the socket must stay open for the life of the
    program. The controller streams status messages back over it; a drain
    thread reads them so the receive buffer never fills.
    """

    def __init__(self, host: str, script: str) -> None:
        self._sock = socket.create_connection((host, PRIMARY_PORT), timeout=5)
        self._sock.sendall(script.encode())
        self._sock.setblocking(False)
        self.started = threading.Event()
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        tail = b""
        while not self._closing.is_set():
            try:
                chunk = self._sock.recv(65536)
                if not chunk:
                    return
                blob = tail + chunk
                if b"PROGRAM_XXX_STARTED" in blob:
                    self.started.set()
                tail = blob[-32:]
            except (BlockingIOError, TimeoutError):
                time.sleep(0.05)
            except OSError:
                return

    def close(self) -> None:
        self._closing.set()
        try:
            self._sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# The job and the runner
# --------------------------------------------------------------------------- #
@dataclass
class PatternJob:
    """One pattern the robot is running or holding, paused."""

    job_id: str
    pattern: Pattern
    params: dict
    anchor_pose: list[float]
    speed_m_s: float
    direction: int
    points_per_lap: int
    cycle_length_m: float
    state: str = "running"                  # "running" | "paused"
    session: _ScriptSession | None = None
    cycles_done: float = 0.0
    running_since_s: float | None = None
    created_at_s: float = field(default_factory=time.monotonic)
    last_change_s: float = field(default_factory=time.monotonic)

    def cycles(self) -> float:
        """Completed cycles, extrapolated from elapsed time and speed."""
        total = self.cycles_done
        if self.state == "running" and self.running_since_s is not None:
            elapsed = time.monotonic() - self.running_since_s
            total += elapsed * self.speed_m_s / self.cycle_length_m
        return total

    def settle(self) -> None:
        """Fold the time spent running into the cycle count."""
        self.cycles_done = self.cycles()
        self.running_since_s = None


class PatternRunner:
    """Owns the single active motion pattern.

    One arm, one pattern: a second start is refused rather than queued, so the
    LLM gets a readable reason instead of two programs fighting for the robot.
    Safety checks are injected by the server, keeping this module free of any
    import back into it.
    """

    def __init__(self, robot: URClient,
                 validate_pose: Callable[[list[float], str], None],
                 validate_start: Callable[[list[float]], None]) -> None:
        self._robot = robot
        self._validate_pose = validate_pose
        self._validate_start = validate_start
        self._lock = threading.RLock()
        self._job: PatternJob | None = None
        atexit.register(self.shutdown)

    # --- robot helpers ---------------------------------------------------- #
    def _require_ready(self) -> None:
        state = self._robot.get_state()
        if state.robot_mode != ROBOT_MODE_RUNNING:
            raise RuntimeError(
                f"Robot is not powered on (mode {state.robot_mode}, need "
                f"{ROBOT_MODE_RUNNING}=RUNNING). Power it on and release the "
                "brakes at http://localhost, then try again."
            )
        if state.safety_status != 1:
            raise RuntimeError(
                f"Robot safety status is {state.safety_status}, not NORMAL. "
                "Recover it in the PolyScope UI before commanding motion."
            )

    def _wait_stationary(self, timeout_s: float = STATIONARY_TIMEOUT_S) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._robot.get_state().is_moving:
                return True
        return False

    def _upload_verified(self, script: str, what: str) -> _ScriptSession:
        """Upload, confirm the arm really started moving, retry once if not."""
        last_error = ""
        for attempt in (1, 2):
            self._wait_stationary()
            session = _ScriptSession(self._robot.host, script)
            deadline = time.monotonic() + MOTION_START_TIMEOUT_S
            while time.monotonic() < deadline:
                if self._robot.get_state().is_moving:
                    return session
            session.close()
            last_error = (f"the arm did not move within "
                          f"{MOTION_START_TIMEOUT_S}s of upload")
            if attempt == 1:
                time.sleep(0.5)
        raise RuntimeError(
            f"Could not start {what}: {last_error}. The controller may be "
            "busy with another program, or the robot may have stopped. Check "
            "get_robot_state, then try again."
        )

    def _stop_arm(self) -> None:
        """Preempt whatever is running with a smooth decelerating stop."""
        stopper = _ScriptSession(
            self._robot.host,
            f"def motion_stop():\n  stopl({STOP_DECEL_M_S2:.4f})\nend\n",
        )
        self._wait_stationary()
        stopper.close()

    def _current_phase(self, job: PatternJob) -> float:
        """Recover the phase from where the tool actually is.

        Reading it back beats tracking a counter: after a pause, a speed change
        or a nudge, the robot's own position is the only trustworthy source.
        Only meaningful for the closed patterns; the others restart their cycle.
        """
        if not job.pattern.exact_arcs:
            return 0.0
        tcp = self._robot.get_tcp_pose()
        return math.atan2(tcp[1] - job.anchor_pose[1],
                          tcp[0] - job.anchor_pose[0])

    def _launch(self, job: PatternJob, what: str) -> None:
        """(Re)build the loop from the tool's current position and run it."""
        accel = _derive_acceleration(job.speed_m_s, job.params["radius_m"])
        script = _build_script(
            anchor=job.anchor_pose,
            pattern=job.pattern,
            params=job.params,
            speed=job.speed_m_s,
            accel=accel,
            start_phase=self._current_phase(job),
            direction=job.direction,
            points_per_lap=job.points_per_lap,
            current_tcp=self._robot.get_tcp_pose(),
        )
        previous = job.session
        job.session = self._upload_verified(script, what)
        if previous is not None:
            previous.close()
        job.state = "running"
        job.running_since_s = time.monotonic()
        job.last_change_s = time.monotonic()

    # --- reporting -------------------------------------------------------- #
    def _report(self, job: PatternJob, status: str, **extra) -> dict:
        radius = job.params["radius_m"]
        laps_per_min = job.speed_m_s / job.cycle_length_m * 60
        report = {
            "status": status,
            "job_id": job.job_id,
            "motion": job.pattern.name,
            "state": job.state,
            "speed_m_s": round(job.speed_m_s, 4),
            "laps_per_min": round(laps_per_min, 1),
            "max_safe_speed_m_s": round(max_speed_for_radius(radius), 4),
            "radius_m": round(radius, 4),
            "direction": "counterclockwise" if job.direction > 0 else "clockwise",
            "laps_completed": round(job.cycles(), 2),
            "anchor_tcp_m_rad": [round(v, 4) for v in job.anchor_pose],
        }
        report.update(extra)
        return report

    # --- public operations ------------------------------------------------ #
    def start(self, pattern_name: str, radius_m: float, speed_m_s: float,
              direction: int = 1, turns: float = 3.0,
              points_per_lap: int = DEFAULT_POINTS_PER_LAP) -> dict:
        """Anchor the pattern at the current tool pose and start moving."""
        with self._lock:
            if self._job is not None:
                raise ValueError(
                    f"A {self._job.pattern.name} pattern is already "
                    f"{self._job.state} (job {self._job.job_id}). Finish it "
                    "with finish_motion before starting another one."
                )
            if pattern_name not in PATTERNS:
                raise ValueError(
                    f"Unknown pattern '{pattern_name}'. Available: "
                    f"{', '.join(sorted(PATTERNS))}."
                )
            if not 0.005 <= radius_m <= 0.30:
                raise ValueError(
                    f"Radius {radius_m:.3f} m is outside the workable range "
                    "[0.005, 0.30] m."
                )
            if not 4 <= points_per_lap <= 90:
                raise ValueError(
                    f"points_per_lap must be in [4, 90], got {points_per_lap}."
                )
            if turns <= 0:
                raise ValueError(f"turns must be positive, got {turns}.")

            pattern = PATTERNS[pattern_name]
            params = {"radius_m": float(radius_m), "turns": float(turns)}
            _validate_speed(speed_m_s, radius_m)

            self._require_ready()
            state = self._robot.get_state()
            # movel and movec both fail from a singular configuration, and
            # after a freedrive teach-in that is a real possibility.
            self._validate_start(state.q_rad)

            anchor = list(state.tcp_pose)
            for offset in _cycle_points(pattern, params, SAFETY_SAMPLES_PER_CYCLE):
                self._validate_pose(_pose_at(anchor, offset),
                                    f"the {pattern_name} path")

            job = PatternJob(
                job_id=uuid.uuid4().hex[:8],
                pattern=pattern,
                params=params,
                anchor_pose=anchor,
                speed_m_s=float(speed_m_s),
                direction=1 if direction >= 0 else -1,
                points_per_lap=int(points_per_lap),
                cycle_length_m=_cycle_length_m(pattern, params),
            )
            self._launch(job, f"the {pattern_name} motion")
            self._job = job
            return self._report(job, "started",
                                exact_path=pattern.exact_arcs)

    def set_speed(self, factor: float | None = None,
                  speed_m_s: float | None = None) -> dict:
        """Change speed, keeping the tool's place on the path."""
        with self._lock:
            job = self._require_job()
            if (factor is None) == (speed_m_s is None):
                raise ValueError(
                    "Give exactly one of factor (relative, e.g. 1.3 for "
                    "'faster') or speed_m_s (absolute)."
                )
            if factor is not None:
                if not 0.1 <= factor <= 10.0:
                    raise ValueError(
                        f"factor must be in [0.1, 10.0], got {factor}."
                    )
                target = job.speed_m_s * factor
            else:
                target = float(speed_m_s)

            radius = job.params["radius_m"]
            _validate_speed(target, radius)

            job.settle()
            job.speed_m_s = target
            if job.state == "running":
                self._launch(job, "the motion at the new speed")
                return self._report(job, "speed_changed")
            # Paused: remember the new speed, apply it on resume.
            job.last_change_s = time.monotonic()
            return self._report(
                job, "speed_changed",
                note="The motion is paused; the new speed applies on resume.",
            )

    def _stop_stray(self) -> dict:
        """Stop the arm even when no pattern is registered here.

        A stop must always stop. A pattern loop outlives the process that
        started it -- the program runs on the controller, not in this process
        -- so after a server restart the robot can still be circling with no
        job to point at. Refusing on bookkeeping grounds would leave it going.
        """
        if not self._robot.get_state().is_moving:
            return {"status": "idle", "active": False,
                    "note": "Nothing is moving and no pattern is registered."}
        self._stop_arm()
        return {
            "status": "stopped",
            "active": False,
            "note": "The arm was moving with no pattern registered here "
                    "(most likely a loop left over from a previous server "
                    "session) and has been stopped.",
        }

    def shutdown(self) -> None:
        """Stop a running pattern when the process exits.

        Without this the robot keeps moving after the server is gone, because
        the loop lives on the controller. Registered with atexit, so it covers
        a normal exit; a SIGKILL cannot be caught, and then the next
        pause_motion or finish_motion stops the leftover loop.
        """
        with self._lock:
            job = self._job
            self._job = None
            if job is None:
                return
            try:
                if job.state == "running":
                    self._stop_arm()
                if job.session is not None:
                    job.session.close()
            except OSError:
                pass

    def pause(self) -> dict:
        """Stop the arm but keep the anchor, the phase and the parameters."""
        with self._lock:
            if self._job is None:
                return self._stop_stray()
            job = self._job
            if job.state == "paused":
                return self._report(job, "already_paused")
            self._stop_arm()
            job.settle()
            if job.session is not None:
                job.session.close()
                job.session = None
            job.state = "paused"
            job.last_change_s = time.monotonic()
            return self._report(job, "paused",
                                note="Say resume to continue from here.")

    def resume(self) -> dict:
        """Pick the motion up from wherever the tool now sits on the path."""
        with self._lock:
            job = self._require_job()
            if job.state == "running":
                return self._report(job, "already_running")
            self._require_ready()
            self._validate_start(self._robot.get_state().q_rad)
            self._launch(job, "the resumed motion")
            return self._report(job, "resumed")

    def finish(self) -> dict:
        """Stop for good and release the anchor."""
        with self._lock:
            if self._job is None:
                return self._stop_stray()
            job = self._job
            if job.state == "running":
                self._stop_arm()
            job.settle()
            if job.session is not None:
                job.session.close()
                job.session = None
            job.state = "finished"
            self._job = None
            return self._report(
                job, "finished",
                total_duration_s=round(time.monotonic() - job.created_at_s, 1),
            )

    def status(self) -> dict:
        """Describe the pattern and the robot, safe to call at any time."""
        with self._lock:
            state = self._robot.get_state()
            live = {
                "robot_moving": state.is_moving,
                "safety_status": state.safety_status,
                "tcp_pose_m_rad": [round(v, 4) for v in state.tcp_pose],
            }
            if self._job is None:
                return {"status": "idle", "active": False,
                        "note": "No motion pattern is running.", **live}
            report = self._report(self._job, "active", active=True)
            report.update(live)
            if self._job.state == "running":
                distance = math.dist(state.tcp_pose[:3],
                                     self._job.anchor_pose[:3])
                report["distance_from_anchor_m"] = round(distance, 4)
            return report

    def _require_job(self) -> PatternJob:
        if self._job is None:
            raise ValueError(
                "No motion pattern is running. Start one with "
                "start_motion_pattern (or stir) first."
            )
        return self._job
