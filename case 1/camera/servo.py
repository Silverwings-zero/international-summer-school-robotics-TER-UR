"""Eye-in-hand visual servoing: click an object, the robot keeps it centred.

A RealSense D435 on the UR wrist streams frames through YOLO26; you click one
of the detected objects in the live window and toggle servoing. The loop then
repeats LOOK -> STEP: measure how far the object sits from the image centre
(in normalized image coordinates), turn that error into a small Cartesian
step of the tool, execute it with a blocking ``movel`` through case 1's
``URClient``, and look again. With depth available it also closes the range
axis, holding the camera a fixed standoff from the object, so "steer to the
object of interest" falls out of the same law: centre it, then approach it.

The step law. For an eye-in-hand camera, a small tool translation ``dt``
(tool frame, metres) changes the observation ``obs = (x_norm, y_norm, depth)``
approximately linearly: ``d(obs) = J @ dt``. The servo solves the inverse
problem each iteration: pick the desired observation change (a fraction of
the current error, toward zero) and least-squares ``dt`` out of it, clamped.

Where does J come from? Two sources, and this is the part that makes the
demo robust on any bracket:

  * ANALYTIC (default): assumes the camera looks along tool +Z with camera
    X/Y aligned to tool X/Y (edit ``R_TOOL_CAM`` if your mount differs).
  * CALIBRATED (press ``c``): the robot probes each tool axis a few
    millimetres and MEASURES how the image and depth respond, building J
    empirically. No hand-eye math, no mounting angles -- thirty seconds with
    any rigid mount, saved to ``hand_eye.json`` and reloaded next run.

Safety, in order of authority: every step is clamped to ``max_step_m``; the
TCP is confined to a box around where servoing was enabled; the approach axis
regulates AROUND the standoff distance (too close means it backs off);
observations older than ``stale_after_s`` freeze the robot ("target lost");
any protective stop or controller refusal surfaces as a caught error that
disables servoing. Servoing also refuses to start from a wrist singularity
(wrist2 near 0), where UR linear moves protective-stop -- press ``v`` first.

Keys in the video window:
  click  select the object under/near the cursor as the target
  n      cycle target through current detections
  SPACE  toggle servoing on/off
  c      run hand-eye auto-calibration (servo off, target selected)
  v      movej to the VIEW_Q observation pose (EDIT for your cell!)
  q/ESC  quit (servo off first)

Run ``python servo.py --dry-run`` to try the full vision stack with no robot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The robot client lives one level up, in case 1 itself: camera/ is a
# subpackage of it now, not a separate case reaching sideways.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ur_client import URClient  # noqa: E402
from camera import open_camera  # noqa: E402
from detector import Detection, ObjectDetector  # noqa: E402

# --- Cell-specific constants (edit these) ---------------------------------- #
# Rotation TOOL <- CAMERA for the ANALYTIC Jacobian: columns are the camera's
# X (image right), Y (image down), Z (optical axis) expressed in tool axes.
# Identity means "camera looks along tool +Z, camera X along tool X". If your
# bracket differs, either edit this or just press 'c' and let calibration
# measure the true mapping -- calibration overrides this entirely.
R_TOOL_CAM = np.eye(3)

# Observation pose for the 'v' key and go_view_pose: joints (rad). Default is
# the kitchen-cell HOME (same pose as case 1's HOME_Q_RAD): upper arm
# vertical, forearm horizontal, tool pointing straight down, so the wrist
# camera overlooks the table from ~0.5 m with wrist2 at -90 deg -- well clear
# of the wrist singularity. Adapt it to YOUR cell without touching code via
# VISION_VIEW_Q_DEG: six comma-separated joint angles in degrees, e.g.
# "-115,-90,90,-90,-90,0" spins the same L-shape to face a table at a
# different bearing (set it in run_server.sh -- sudo strips env vars).
_VIEW_Q_DEFAULT_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)


def _view_q() -> list[float]:
    raw = os.environ.get("VISION_VIEW_Q_DEG", "").strip()
    if not raw:
        return [math.radians(a) for a in _VIEW_Q_DEFAULT_DEG]
    parts = [float(p) for p in raw.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"VISION_VIEW_Q_DEG needs 6 comma-separated degrees, got {raw!r}")
    return [math.radians(a) for a in parts]


VIEW_Q = _view_q()

HAND_EYE_PATH = Path(__file__).with_name("hand_eye.json")

# Singular arm configurations where UR linear moves protective-stop
# (C204A1): IK near these flips to a different solution branch. Wrist2 near
# 0/pi aligns wrist axes; an elbow near 0 is a straight (fully extended) arm.
WRIST_SINGULARITY_RAD = math.radians(5.0)
ELBOW_SINGULARITY_RAD = math.radians(5.0)


@dataclass
class ServoConfig:
    """Gains and limits for the step law. Conservative by design."""

    gain_image: float = 0.5      # fraction of the image error removed per step
    gain_depth: float = 0.5      # fraction of the standoff error per step
    standoff_m: float = 0.35     # hold the camera this far from the object
    assumed_depth_m: float = 0.5  # used when the camera has no depth channel
    deadband_norm: float = 0.02  # image error below this counts as centred
    deadband_depth_m: float = 0.02
    max_step_m: float = 0.03     # hard cap on any single Cartesian step
    min_step_m: float = 0.002    # steps smaller than this are not worth a move
    stale_after_s: float = 0.8   # observations older than this freeze the robot
    box_xy_m: float = 0.30       # allowed |x-x0| and |y-y0| from the enable pose
    box_z_down_m: float = 0.30   # allowed descent below the enable pose
    box_z_up_m: float = 0.20     # allowed rise above the enable pose
    speed_ms: float = 0.10       # movel Cartesian speed
    accel_ms2: float = 0.4       # movel Cartesian acceleration
    probe_m: float = 0.015       # calibration probe distance per tool axis
    step_timeout_s: float = 20.0  # per-movel budget (sims can run slow)
    step_period_s: float = 0.25  # breather between steps: rapid-fire
                                 # program uploads can starve the PolyScope
                                 # X sim into a protective stop


@dataclass
class Observation:
    """Where the target sits, as the servo law sees it."""

    x_norm: float          # (u - cx) / fx : >0 means object right of centre
    y_norm: float          # (v - cy) / fy : >0 means object below centre
    depth_m: float | None  # median depth on the target, None without depth
    t: float               # time.monotonic() of the frame


def rotvec_to_matrix(rv) -> np.ndarray:
    """Rodrigues: UR axis-angle [rx, ry, rz] -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-9:
        return np.eye(3)
    k = rv / theta
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


@dataclass
class HandEye:
    """The measured (or assumed) sensitivity J = d(observation)/d(tool step).

    Rows: x_norm, y_norm, depth. Columns: tool X, Y, Z translation (metres).
    The two image rows scale like 1/depth, so they are stored as measured at
    depth ``z0`` and rescaled to the current depth before solving.
    """

    J: np.ndarray
    z0: float
    source: str  # "analytic" or "calibrated"

    def solve(self, want: np.ndarray, z: float, has_depth: bool) -> np.ndarray:
        """Least-squares tool step that produces the wanted observation change.

        Without depth the third row is zeroed: the minimum-norm solution then
        puts (near) zero motion along the unobserved approach direction
        instead of inventing one.
        """
        J = self.J.copy()
        J[:2] *= self.z0 / max(z, 0.05)
        if not has_depth:
            J[2] = 0.0
        dt, *_ = np.linalg.lstsq(J, want, rcond=None)
        return dt

    @classmethod
    def analytic(cls) -> "HandEye":
        # Camera-frame sensitivities at depth 1 m: moving the camera +X shifts
        # the image point by -1/Z in x_norm; moving +Z (toward the object)
        # shortens the depth reading one-for-one. Expressed in tool axes via
        # R_TOOL_CAM (dt_cam = R_TOOL_CAM^T @ dt_tool).
        J_cam = np.array([[-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0]])
        return cls(J=J_cam @ R_TOOL_CAM.T, z0=1.0, source="analytic")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"J": self.J.tolist(), "z0": self.z0, "source": self.source,
             "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "HandEye | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(J=np.array(data["J"], dtype=float),
                       z0=float(data["z0"]), source=str(data["source"]))
        except (ValueError, KeyError):
            return None


class DryRunRobot:
    """Stand-in for URClient: full vision loop, no robot, instant 'moves'."""

    def __init__(self):
        # Somewhere plausible, tool pointing down (rotvec pi about X).
        self._pose = [0.4, 0.0, 0.4, math.pi, 0.0, 0.0]

    def connect(self) -> None:
        pass

    def get_tcp_pose(self) -> list[float]:
        return list(self._pose)

    def get_joint_positions(self) -> list[float]:
        return [0.0, -1.22, 0.79, -1.13, -0.52, 0.0]  # non-singular

    def get_state(self):
        from types import SimpleNamespace
        return SimpleNamespace(tcp_pose=self.get_tcp_pose(),
                               q_rad=self.get_joint_positions(),
                               safety_status=1)

    def move_linear(self, pose, speed, acceleration, **_kw):
        self._pose = list(pose)
        time.sleep(0.15)  # pretend the move takes a moment

    def move_joint(self, q, speed, acceleration, **_kw):
        time.sleep(0.3)


class Shared:
    """Everything the GUI thread and the robot thread exchange."""

    def __init__(self):
        self._lock = threading.Lock()
        self._obs: Observation | None = None
        self._status: tuple[str, str] = ("idle -- click an object", "info")
        self.servo_on = threading.Event()
        # Set by the worker whenever it force-disables servoing (robot
        # error, divergence). The GUI's next SPACE only ACKNOWLEDGES the
        # error instead of re-enabling -- otherwise a stop-keypress racing
        # the async disable would read as a fresh start command.
        self.error_latch = threading.Event()
        # Ask the worker to re-arm the safety box at the CURRENT pose (a
        # fresh tracking command grants a fresh, bounded motion envelope).
        self.rearm = threading.Event()
        self.cal_requested = threading.Event()
        self.view_requested = threading.Event()
        # Completion signals for the above: the request flags are cleared
        # BEFORE the motion runs (so an exception cannot re-trigger it), so
        # a caller who wants to BLOCK until the motion is over must clear
        # the done-event, set the request, then wait on the done-event.
        self.cal_done = threading.Event()
        self.view_done = threading.Event()
        self.cal_done.set()
        self.view_done.set()
        self.quit = threading.Event()

    def set_obs(self, obs: Observation) -> None:
        with self._lock:
            self._obs = obs

    def clear_obs(self) -> None:
        """Forget the current observation (e.g. the target just changed:
        the last measurement belongs to the OLD target)."""
        with self._lock:
            self._obs = None

    def get_obs(self) -> Observation | None:
        with self._lock:
            return self._obs

    def set_status(self, text: str, level: str = "info") -> None:
        """level: info (grey), good (green), warn (yellow), error (red)."""
        with self._lock:
            self._status = (text, level)

    def get_status(self) -> tuple[str, str]:
        with self._lock:
            return self._status


class ServoWorker(threading.Thread):
    """Owns the robot. Consumes observations, produces movel steps.

    All robot I/O lives on this thread so a blocking move never freezes the
    video window, and the vision loop never races the motion socket.
    """

    def __init__(self, robot, shared: Shared, cfg: ServoConfig,
                 hand_eye: HandEye, approach: bool,
                 hand_eye_path: Path = HAND_EYE_PATH):
        super().__init__(daemon=True, name="servo-worker")
        self.robot = robot
        self.shared = shared
        self.cfg = cfg
        self.hand_eye = hand_eye
        self.approach = approach
        self.hand_eye_path = hand_eye_path
        self._p0: np.ndarray | None = None  # box centre, set on servo enable
        self._min_obs_t = 0.0     # only act on frames captured after this
        self._err_prev: float | None = None  # divergence watchdog state
        self._diverging = 0

    # --- helpers ---------------------------------------------------------- #
    def _fresh_obs(self, newer_than: float, timeout_s: float = 1.0
                   ) -> Observation | None:
        """Wait for an observation captured after ``newer_than``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self.shared.quit.is_set():
            obs = self.shared.get_obs()
            if obs is not None and obs.t > newer_than \
                    and time.monotonic() - obs.t < self.cfg.stale_after_s:
                return obs
            time.sleep(0.02)
        return None

    def _mean_obs(self, n: int = 4) -> Observation | None:
        """Average n distinct fresh observations (None if the target drops)."""
        xs, ys, ds = [], [], []
        last_t = time.monotonic()
        for _ in range(n):
            obs = self._fresh_obs(newer_than=last_t, timeout_s=1.5)
            if obs is None:
                return None
            last_t = obs.t
            xs.append(obs.x_norm)
            ys.append(obs.y_norm)
            ds.append(obs.depth_m)
        depth = None
        if all(d is not None for d in ds):
            depth = float(np.mean([d for d in ds if d is not None]))
        return Observation(x_norm=float(np.mean(xs)), y_norm=float(np.mean(ys)),
                           depth_m=depth, t=last_t)

    def _clamp_to_box(self, p: np.ndarray) -> np.ndarray:
        """Confine a target position to the safety box around the enable pose."""
        if self._p0 is None:
            return p
        lo = self._p0 + [-self.cfg.box_xy_m, -self.cfg.box_xy_m,
                         -self.cfg.box_z_down_m]
        hi = self._p0 + [self.cfg.box_xy_m, self.cfg.box_xy_m,
                         self.cfg.box_z_up_m]
        return np.clip(p, lo, hi)

    @staticmethod
    def _config_why_bad(q: list[float]) -> str | None:
        """None if the arm configuration can movel safely, else the reason.

        Learned the hard way on PolyScope X: a movel FROM a singular
        configuration makes IK pick a different solution branch and the
        controller protective-stops with C204A1 (observed live: elbow at
        0.06 deg, setpoint jumped to a completely different arm posture).
        """
        if abs(math.remainder(q[4], math.pi)) < WRIST_SINGULARITY_RAD:
            return "wrist2 is at the wrist singularity"
        if abs(q[2]) < ELBOW_SINGULARITY_RAD:
            return "the elbow is nearly straight (arm singular)"
        return None

    # --- the loop ---------------------------------------------------------- #
    # A single failed movel (PolyScope X occasionally drops an uploaded
    # script) is retried; this many CONSECUTIVE failures disables servoing.
    MAX_MOVE_FAILURES = 3
    # Error growing over this many consecutive steps means the hand-eye
    # mapping is wrong (mirrored mount, stale calibration): stop, do not
    # walk to the box boundary.
    MAX_DIVERGING_STEPS = 4

    def run(self) -> None:
        was_on = False
        fails = 0
        while not self.shared.quit.is_set():
            try:
                if self.shared.view_requested.is_set():
                    # Clear BEFORE reacting, never after: if the handler
                    # raises with the flag still set, the loop would retry
                    # the motion forever and fire it unrequested the moment
                    # a fault clears.
                    self.shared.view_requested.clear()
                    try:
                        self._go_view()
                    finally:
                        self.shared.view_done.set()
                    continue
                if self.shared.cal_requested.is_set():
                    self.shared.cal_requested.clear()
                    try:
                        self._calibrate()
                    finally:
                        self.shared.cal_done.set()
                    continue
                if not self.shared.servo_on.is_set():
                    was_on = False
                    time.sleep(0.05)
                    continue
                if self.shared.rearm.is_set():
                    self.shared.rearm.clear()
                    was_on = False  # force the rising-edge (re-)arming below
                if not was_on:  # rising edge: arm the safety box
                    st = self.robot.get_state()
                    why = self._config_why_bad(st.q_rad)
                    if why:
                        self.shared.servo_on.clear()
                        self.shared.set_status(
                            f"refused: {why} -- go to the view pose first "
                            "('v' key / go_view_pose tool)", "error")
                        continue
                    self._p0 = np.array(st.tcp_pose[:3])
                    self._err_prev = None
                    self._diverging = 0
                    was_on = True
                    self.shared.set_status("servoing", "good")
                self._servo_step()
                fails = 0
            except (TimeoutError, ConnectionError, RuntimeError, OSError) as exc:
                if self.shared.servo_on.is_set() \
                        and fails + 1 < self.MAX_MOVE_FAILURES:
                    # One dropped script / transient refusal: retry. The
                    # step is recomputed from a fresh observation, so a
                    # retry can never replay a stale command.
                    fails += 1
                    self.shared.set_status(
                        f"robot hiccup (retry {fails}): {exc}", "warn")
                    time.sleep(0.5)
                    continue
                if self.shared.servo_on.is_set():
                    self.shared.error_latch.set()
                self.shared.servo_on.clear()
                was_on = False
                fails = 0
                self.shared.set_status(f"robot error: {exc}", "error")
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001 -- last-resort backstop
                # Anything unexpected must not silently kill this thread
                # while the overlay still says SERVO ON.
                if self.shared.servo_on.is_set():
                    self.shared.error_latch.set()
                self.shared.servo_on.clear()
                was_on = False
                self.shared.set_status(
                    f"internal error, servo off: {exc!r}", "error")
                time.sleep(1.0)

    def _servo_step(self) -> None:
        cfg = self.cfg
        obs = self.shared.get_obs()
        if obs is None or time.monotonic() - obs.t > cfg.stale_after_s:
            self.shared.set_status("servo on -- target lost, holding", "warn")
            time.sleep(0.05)
            return
        if obs.t <= self._min_obs_t:
            # Captured before/while the robot last moved: worthless for a
            # new step. Wait for the vision loop to catch up instead.
            time.sleep(0.02)
            return

        has_depth = obs.depth_m is not None
        z = obs.depth_m if has_depth else cfg.assumed_depth_m
        centred = (abs(obs.x_norm) < cfg.deadband_norm
                   and abs(obs.y_norm) < cfg.deadband_norm)
        range_ok = (not self.approach or not has_depth
                    or abs(z - cfg.standoff_m) < cfg.deadband_depth_m)
        if centred and range_ok:
            self.shared.set_status("LOCKED on target", "good")
            time.sleep(0.1)
            return

        # Divergence watchdog: if the error keeps GROWING while we step, the
        # hand-eye mapping is wrong (mirrored mount, stale calibration) and
        # every further step makes it worse -- stop instead of marching to
        # the box boundary.
        err = math.hypot(obs.x_norm, obs.y_norm) \
            + (abs(z - cfg.standoff_m) if self.approach and has_depth else 0.0)
        if self._err_prev is not None and err > self._err_prev * 1.05:
            self._diverging += 1
            if self._diverging >= self.MAX_DIVERGING_STEPS:
                self.shared.servo_on.clear()
                self.shared.error_latch.set()
                self.shared.set_status(
                    "stopped: error GROWS with every step -- the hand-eye "
                    "mapping looks wrong, run calibration ('c')", "error")
                return
        else:
            self._diverging = 0

        want = np.array([
            -cfg.gain_image * obs.x_norm,
            -cfg.gain_image * obs.y_norm,
            (cfg.gain_depth * (cfg.standoff_m - z)
             if self.approach and has_depth else 0.0),
        ])
        dt_tool = self.hand_eye.solve(want, z, has_depth)
        norm = float(np.linalg.norm(dt_tool))
        if norm < cfg.min_step_m:
            time.sleep(0.05)
            return
        if norm > cfg.max_step_m:
            dt_tool *= cfg.max_step_m / norm

        st = self.robot.get_state()
        if st.safety_status != 1:
            raise RuntimeError(
                f"robot safety status is {st.safety_status} (1=NORMAL) -- "
                "clear it in the PolyScope UI at http://localhost")
        why = self._config_why_bad(st.q_rad)
        if why:
            self.shared.servo_on.clear()
            self.shared.error_latch.set()
            self.shared.set_status(
                f"stopped: {why} -- a movel from here would protective-stop; "
                "reposition via the view pose", "error")
            return
        pose = st.tcp_pose
        R = rotvec_to_matrix(pose[3:])
        target_p = self._clamp_to_box(np.array(pose[:3]) + R @ dt_tool)
        step_mm = float(np.linalg.norm(target_p - np.array(pose[:3]))) * 1000
        if step_mm < cfg.min_step_m * 1000:
            # The box clamp ate the whole step: the target is outside the
            # allowed workspace. Hold instead of uploading no-op moves.
            self.shared.set_status(
                "at the safety-box boundary -- target out of reach, holding "
                "(toggle servo to re-arm the box here)", "warn")
            time.sleep(0.25)
            return
        self.shared.set_status(
            f"servoing  err=({obs.x_norm:+.3f},{obs.y_norm:+.3f})  "
            f"Z={z:.3f}m  step={step_mm:.0f}mm", "good")
        self.robot.move_linear(list(target_p) + list(pose[3:]),
                               cfg.speed_ms, cfg.accel_ms2,
                               timeout_s=cfg.step_timeout_s)
        self._err_prev = err
        # Never act on a picture taken before/while the robot was moving:
        # remember when the move settled and wait for a newer frame.
        self._min_obs_t = time.monotonic()
        self._fresh_obs(newer_than=self._min_obs_t, timeout_s=0.6)
        time.sleep(cfg.step_period_s)

    def _go_view(self) -> None:
        if self.shared.servo_on.is_set():
            self.shared.set_status("turn servo off before 'v'", "warn")
            return
        self.shared.set_status("moving to view pose...", "warn")
        self.robot.move_joint(VIEW_Q, 0.8, 1.0)
        self._min_obs_t = time.monotonic()
        self.shared.set_status("at view pose -- click an object", "info")

    def _calibrate(self) -> None:
        """Measure J: probe each tool axis, watch how the observation moves."""
        cfg = self.cfg
        if self.shared.servo_on.is_set():
            self.shared.set_status("turn servo off before calibrating", "warn")
            return
        self.shared.set_status("calibrating: hold the scene still...", "warn")
        base = self._mean_obs()
        if base is None:
            self.shared.set_status("calibration aborted: no stable target",
                                   "error")
            return
        pose0 = self.robot.get_tcp_pose()
        R = rotvec_to_matrix(pose0[3:])
        J = np.zeros((3, 3))
        depth_measured = base.depth_m is not None
        displaced = False
        try:
            for axis in range(3):
                if self.shared.quit.is_set():
                    self.shared.set_status("calibration aborted: quitting",
                                           "warn")
                    return
                step = R @ (np.eye(3)[axis] * cfg.probe_m)
                probe_pose = list(np.array(pose0[:3]) + step) + list(pose0[3:])
                self.shared.set_status(
                    f"calibrating: probing tool {'XYZ'[axis]}...", "warn")
                self.robot.move_linear(probe_pose, cfg.speed_ms, cfg.accel_ms2,
                                       timeout_s=cfg.step_timeout_s)
                displaced = True
                moved = self._mean_obs()
                self.robot.move_linear(pose0, cfg.speed_ms, cfg.accel_ms2,
                                       timeout_s=cfg.step_timeout_s)
                displaced = False
                if moved is None:
                    self.shared.set_status(
                        "calibration aborted: target lost mid-probe", "error")
                    return
                J[0, axis] = (moved.x_norm - base.x_norm) / cfg.probe_m
                J[1, axis] = (moved.y_norm - base.y_norm) / cfg.probe_m
                if moved.depth_m is not None and base.depth_m is not None:
                    J[2, axis] = (moved.depth_m - base.depth_m) / cfg.probe_m
                else:
                    depth_measured = False
        except (TimeoutError, ConnectionError, RuntimeError, OSError) as exc:
            self.shared.set_status(f"calibration failed: {exc}", "error")
            if displaced:  # best effort: do not abandon the probe offset
                try:
                    self.robot.move_linear(pose0, cfg.speed_ms, cfg.accel_ms2,
                                           timeout_s=cfg.step_timeout_s)
                except (TimeoutError, ConnectionError, RuntimeError, OSError):
                    self.shared.set_status(
                        f"calibration failed AND the robot is still at the "
                        f"probe pose: {exc}", "error")
            return
        finally:
            self._min_obs_t = time.monotonic()

        # The image rows must actually respond, or the probes did not move
        # the camera / the detector is jittering worse than the probe.
        if np.linalg.norm(J[:2]) < 0.2:
            self.shared.set_status(
                "calibration rejected: image barely responded to probes "
                "(is the mount rigid? target steady?)", "error")
            return
        # The depth row is trustworthy only if every probe had depth AND the
        # numbers are physical: a unit tool translation changes a distance
        # reading by at most ~1 (direction cosine). Anything else -- webcam
        # mode, depth holes, noise -- gets the ANALYTIC depth row instead of
        # silently persisting a row that can never regulate the approach.
        depth_row = "measured"
        if not depth_measured or not 0.5 <= np.linalg.norm(J[2]) <= 1.8 \
                or np.max(np.abs(J[2])) > 1.5:
            J[2] = HandEye.analytic().J[2]
            depth_row = "analytic"
        z0 = base.depth_m if base.depth_m is not None else cfg.assumed_depth_m
        self.hand_eye = HandEye(J=J, z0=z0, source="calibrated")
        self.hand_eye.save(self.hand_eye_path)
        self.shared.set_status(
            f"calibrated, depth row {depth_row} "
            f"(saved to {self.hand_eye_path.name})", "good")


class TargetSelector:
    """Which detection is THE target, across frames and brief dropouts.

    Identity is the tracker id when the tracker provides one; when the id
    disappears (occlusion, re-detection) the selector re-acquires the nearest
    same-class detection close to where the target was last seen.
    """

    REACQUIRE_PX = 150.0
    CLICK_PX = 120.0

    def __init__(self):
        self.track_id: int | None = None
        self.cls: str | None = None
        self.last_center: tuple[float, float] | None = None

    def select_at(self, x: float, y: float, dets: list[Detection]) -> None:
        best, best_d = None, self.CLICK_PX
        for d in dets:
            u, v = d.center
            dist = math.hypot(u - x, v - y)
            if dist < best_d:
                best, best_d = d, dist
        if best is not None:
            self._adopt(best)

    def cycle(self, dets: list[Detection]) -> None:
        if not dets:
            return
        ordered = sorted(dets, key=lambda d: d.center)
        idx = 0
        if self.track_id is not None:
            ids = [d.track_id for d in ordered]
            if self.track_id in ids:
                idx = (ids.index(self.track_id) + 1) % len(ordered)
        self._adopt(ordered[idx])

    def _adopt(self, d: Detection) -> None:
        self.track_id, self.cls, self.last_center = d.track_id, d.name, d.center

    def update(self, dets: list[Detection]) -> Detection | None:
        if self.cls is None:
            return None
        if self.track_id is not None:
            for d in dets:
                if d.track_id == self.track_id:
                    self.last_center = d.center
                    return d
        # Tracker lost the id: re-acquire by class near the last position.
        best, best_d = None, self.REACQUIRE_PX
        for d in dets:
            if d.name != self.cls or self.last_center is None:
                continue
            dist = math.hypot(d.center[0] - self.last_center[0],
                              d.center[1] - self.last_center[1])
            if dist < best_d:
                best, best_d = d, dist
        if best is not None:
            self._adopt(best)
        return best


_STATUS_COLORS = {"info": (200, 200, 200), "good": (80, 220, 80),
                  "warn": (40, 200, 255), "error": (60, 60, 255)}


def draw_overlay(img, frame, dets, target, shared: Shared, cfg: ServoConfig,
                 cam_name: str, he_source: str) -> None:
    h, w = img.shape[:2]
    cx, cy = int(frame.cx), int(frame.cy)
    # All detections in grey, the target in green.
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        is_target = target is not None and d is target
        color = (80, 220, 80) if is_target else (140, 140, 140)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if is_target else 1)
        tid = f"#{d.track_id}" if d.track_id is not None else ""
        cv2.putText(img, f"{d.name}{tid} {d.conf:.2f}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    # Crosshair + deadband circle: "centred" means inside this circle.
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 22, 1)
    cv2.circle(img, (cx, cy), int(cfg.deadband_norm * frame.fx),
               (255, 255, 255), 1, cv2.LINE_AA)
    if target is not None:
        u, v = (int(x) for x in target.center)
        cv2.line(img, (u, v), (cx, cy), (80, 220, 80), 1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 5, (80, 220, 80), -1, cv2.LINE_AA)
    text, level = shared.get_status()
    mode = "SERVO ON" if shared.servo_on.is_set() else "servo off"
    lines = [f"[{mode}]  {text}",
             f"cam: {cam_name}   hand-eye: {he_source}",
             "click=select  n=next  SPACE=servo  c=calibrate  v=view  q=quit"]
    for i, line in enumerate(lines):
        y = h - 10 - 18 * (len(lines) - 1 - i)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    _STATUS_COLORS[level] if i == 0 else (200, 200, 200), 1,
                    cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eye-in-hand YOLO26 visual servoing on a UR robot.")
    ap.add_argument("--host", default=os.environ.get("UR_HOST", "127.0.0.1"),
                    help="robot/simulator IP (default: $UR_HOST or 127.0.0.1)")
    ap.add_argument("--camera", default="auto",
                    choices=("auto", "realsense", "webcam"))
    ap.add_argument("--webcam-index", type=int, default=0)
    ap.add_argument("--model", default="yolo26n.pt",
                    help="ultralytics detection weights (default yolo26n.pt)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None,
                    help="inference device: mps / cpu / cuda (default: auto)")
    ap.add_argument("--standoff", type=float, default=0.35,
                    help="approach distance to hold, metres (default 0.35)")
    ap.add_argument("--no-approach", action="store_true",
                    help="centre in the image only; never move along range")
    ap.add_argument("--dry-run", action="store_true",
                    help="no robot: run camera+YOLO and print what would move")
    args = ap.parse_args()

    cfg = ServoConfig(standoff_m=args.standoff)
    print("[servo] loading YOLO26 (first run downloads the weights)...")
    detector = ObjectDetector(args.model, conf=args.conf, device=args.device)
    camera = open_camera(args.camera, webcam_index=args.webcam_index)

    if args.dry_run:
        robot = DryRunRobot()
        print("[servo] DRY RUN -- no robot connection")
    else:
        robot = URClient(host=args.host)
        robot.connect()
        print(f"[servo] robot reachable at {args.host}")

    hand_eye = HandEye.load(HAND_EYE_PATH) or HandEye.analytic()
    print(f"[servo] hand-eye: {hand_eye.source}"
          + (f" (from {HAND_EYE_PATH.name})" if hand_eye.source == "calibrated"
             else " -- press 'c' with a target selected to calibrate"))

    shared = Shared()
    worker = ServoWorker(robot, shared, cfg, hand_eye,
                         approach=not args.no_approach)
    worker.start()

    selector = TargetSelector()
    window = "case 1 -- YOLO26 visual servoing"
    cv2.namedWindow(window)
    latest_dets: list[Detection] = []
    cv2.setMouseCallback(
        window,
        lambda ev, x, y, flags, param: (
            selector.select_at(x, y, latest_dets)
            if ev == cv2.EVENT_LBUTTONDOWN else None))

    try:
        while not shared.quit.is_set():
            frame = camera.read()
            dets = detector.detect(frame.color)
            latest_dets = dets
            target = selector.update(dets)
            if target is not None:
                u, v = target.center
                xn, yn = frame.pixel_to_normalized(u, v)
                shared.set_obs(Observation(
                    x_norm=xn, y_norm=yn,
                    depth_m=frame.box_depth(target.xyxy), t=frame.t))

            img = frame.color  # camera hands us a fresh buffer each read
            draw_overlay(img, frame, dets, target, shared, cfg,
                         camera.name, worker.hand_eye.source)
            cv2.imshow(window, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if shared.error_latch.is_set():
                    # An async disable (robot error, divergence) may have
                    # raced this keypress: never turn that into a restart.
                    shared.error_latch.clear()
                    shared.servo_on.clear()
                    shared.set_status(
                        "error acknowledged -- SPACE again to restart",
                        "info")
                elif shared.servo_on.is_set():
                    shared.servo_on.clear()
                    shared.set_status("servo off", "info")
                elif target is None:
                    shared.set_status("select a target first (click it)",
                                      "warn")
                else:
                    shared.servo_on.set()
            elif key == ord("n"):
                selector.cycle(dets)
            elif key == ord("c"):
                if target is None:
                    shared.set_status("select a target before calibrating",
                                      "warn")
                else:
                    shared.cal_requested.set()
            elif key == ord("v"):
                shared.view_requested.set()
    finally:
        shared.servo_on.clear()
        shared.quit.set()
        worker.join(timeout=15.0)
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
