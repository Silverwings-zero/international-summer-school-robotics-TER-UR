"""MCP server: the wrist camera as LLM-callable tools.

Turns the wrist camera's visual servoing into language: "track the cup" becomes
``track_object("cup")``, "descend on the phone" becomes
``descend_on("phone")``. The tools wrap the same engine ``servo.py`` drives
from its GUI -- one perception thread feeding observations, one worker thread
owning the robot -- so everything the GUI mode has (empirical hand-eye
calibration, step clamps, the safety box, target-lost freezing) guards the
LLM's calls too.

Two perception modes, chosen by the ``VISION_MODE`` environment variable:

  * ``real`` (default) -- RealSense D435 (or webcam) + YOLO26. The camera
    opens lazily on the first tool call that needs eyes, so the server comes
    up even with the camera unplugged and reports a readable error instead.
  * ``sim`` -- the PolyScope X simulator has no camera, so perception is
    synthesized: virtual objects at fixed BASE-frame positions, viewed by a
    virtual pinhole camera rigidly attached to the LIVE simulated TCP (read
    over RTDE, camera axes = tool axes). Everything downstream -- the servo
    law, the URScript moves, URSim's own controller -- is exactly the real
    pipeline; only the pixels are synthetic. ``place_sim_object`` puts
    something in front of the camera to chase.

Register next to ur-tools in ``.mcp.json``::

    "vision-tools": {
      "command": "python3",
      "args": [".../case 1/camera/vision_tools.py"],
      "env": {"VISION_MODE": "sim"}
    }
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# This directory (camera.py, detector.py, servo.py), then case 1 above it for
# ur_client. Both are relative to this file, so the tree can be cloned or moved
# anywhere without either import breaking.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP  # noqa: E402

from ur_client import URClient  # noqa: E402
from servo import (  # noqa: E402
    HAND_EYE_PATH,
    HandEye,
    Observation,
    ServoConfig,
    ServoWorker,
    Shared,
    rotvec_to_matrix,
)

VISION_MODE = os.environ.get("VISION_MODE", "real").strip().lower()
SIM_HAND_EYE_PATH = Path(__file__).with_name("hand_eye_sim.json")

# Virtual camera model for sim mode (D435-like: 640x480, ~69 deg HFOV).
SIM_FX = SIM_FY = 600.0
SIM_CX, SIM_CY = 320.0, 240.0
SIM_HALF_FOV_X = SIM_CX / SIM_FX   # normalized-coordinate visibility bound
SIM_HALF_FOV_Y = SIM_CY / SIM_FY

MIN_STANDOFF_M = 0.10   # never regulate closer than this (D435 min range)

# --- Grasp approach ------------------------------------------------------ #
# The camera sits ABOVE the tool flange, so it is further from the object
# than the fingertips are: the depth the camera reports is NOT the distance
# left to travel. CAMERA_TO_FINGERTIP is that offset along the approach
# axis, and no sensor can measure it -- it is taught per cell by
# start_grasp_calibration/finish_grasp_calibration and cached here.
GRASP_CAL_PATH = Path(__file__).with_name("grasp_calibration.json")
# Depth below MIN_STANDOFF_M is unmeasurable, so the last stretch of a grasp
# approach is necessarily open-loop. Cap how far we are willing to go blind.
MAX_BLIND_DESCENT_M = 0.06
GRASP_SPEED_MS = 0.03     # the final approach happens near the human's hands
GRASP_ACCEL_MS2 = 0.2
MAX_WAIT_S = 180.0

# Camera-view window policy: "auto" pops it up when tracking starts and
# closes it on stop_tracking; "off" never opens it automatically (the
# show_camera_view tool still works); "always" opens it with the engine.
VISION_VIEW = os.environ.get("VISION_VIEW", "auto").strip().lower()


@dataclass
class Percept:
    """One visible object, camera-independent: what the servo law needs."""

    name: str
    conf: float
    track_id: int | None
    u: float               # pixel column (for readable output)
    v: float               # pixel row
    x_norm: float          # (u-cx)/fx, >0 means right of centre
    y_norm: float          # (v-cy)/fy, >0 means below centre
    depth_m: float | None
    xyxy: tuple[float, float, float, float] | None = None  # box, for the view


class Viewer:
    """The pop-up camera window, run as a subprocess (see viewer.py).

    All methods are safe to call from any thread and never raise: a machine
    with no display, a denied window, or a user-closed window simply turns
    every ``push`` into a no-op. Visualization must never take the robot
    server down with it.
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def open(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name("viewer.py"))],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            except OSError as exc:
                print(f"[vision-tools] viewer unavailable: {exc}",
                      file=sys.stderr)
                self._proc = None

    def push(self, img: np.ndarray) -> None:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return
            ok, buf = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return
            data = buf.tobytes()
            try:
                proc.stdin.write(struct.pack(">I", len(data)) + data)
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._close_locked()  # user closed the window: fine

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()  # EOF ends the viewer's read loop
        except OSError:
            pass
        try:
            self._proc.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
        self._proc = None


_STATUS_COLORS = {"info": (200, 200, 200), "good": (80, 220, 80),
                  "warn": (40, 200, 255), "error": (60, 60, 255)}


def draw_view(img: np.ndarray, percepts: list[Percept],
              target: Percept | None, fx: float, cx: float, cy: float,
              deadband_norm: float, status: tuple[str, str],
              tracking: bool, mode_line: str) -> np.ndarray:
    """The HUD both modes share: boxes, target, crosshair, status."""
    h, w = img.shape[:2]
    for p in percepts:
        is_target = target is not None and p is target
        color = (80, 220, 80) if is_target else (150, 150, 150)
        if p.xyxy is not None:
            x1, y1, x2, y2 = (int(v) for v in p.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), color,
                          2 if is_target else 1)
            ty = max(14, y1 - 6)
        else:
            ty = int(p.v) - 10
        depth = f" {p.depth_m:.2f}m" if p.depth_m is not None else ""
        tid = f"#{p.track_id}" if p.track_id is not None else ""
        cv2.putText(img, f"{p.name}{tid} {p.conf:.2f}{depth}",
                    (max(2, int(p.u) - 40), ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    if not percepts:
        cv2.putText(img, "nothing in view -- place/move an object or "
                         "reposition the robot",
                    (max(6, int(cx) - 260), int(cy) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 255), 1,
                    cv2.LINE_AA)
    cv2.drawMarker(img, (int(cx), int(cy)), (255, 255, 255),
                   cv2.MARKER_CROSS, 22, 1)
    cv2.circle(img, (int(cx), int(cy)), max(2, int(deadband_norm * fx)),
               (255, 255, 255), 1, cv2.LINE_AA)
    if target is not None:
        u, v = int(target.u), int(target.v)
        cv2.line(img, (u, v), (int(cx), int(cy)), (80, 220, 80), 1,
                 cv2.LINE_AA)
        cv2.circle(img, (u, v), 5, (80, 220, 80), -1, cv2.LINE_AA)
    text, level = status
    mode = "TRACKING" if tracking else "idle"
    lines = [f"[{mode}]  {text}", mode_line,
             "q in this window closes the VIEW only -- stop_tracking stops "
             "the robot"]
    for i, line in enumerate(lines):
        y = h - 10 - 18 * (len(lines) - 1 - i)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    _STATUS_COLORS[level] if i == 0 else (200, 200, 200), 1,
                    cv2.LINE_AA)
    return img


class RealPerception:
    """RealSense/webcam + YOLO26, opened lazily on first use."""

    def __init__(self):
        self._camera = None
        self._detector = None
        self._lock = threading.Lock()
        self._last_frame = None  # only the vision thread touches this
        self._read_failures = 0

    def start(self) -> str:
        with self._lock:
            if self._camera is None:
                from camera import open_camera
                from detector import ObjectDetector
                self._camera = open_camera(
                    os.environ.get("VISION_CAMERA", "auto"),
                    int(os.environ.get("VISION_CAMERA_INDEX", "0")))
                self._detector = ObjectDetector(
                    os.environ.get("VISION_MODEL", "yolo26n.pt"))
            return self._camera.name

    def snapshot(self) -> list[Percept]:
        self.start()
        try:
            frame = self._camera.read()
        except Exception:
            # A stale librealsense pipeline (camera replugged, link wedged)
            # fails every read forever; after a few in a row, drop the
            # camera so the next start() reopens it from scratch.
            self._read_failures += 1
            if self._read_failures >= 3:
                self._read_failures = 0
                with self._lock:
                    if self._camera is not None:
                        try:
                            self._camera.close()
                        except Exception:
                            pass
                        self._camera = None
            raise
        self._read_failures = 0
        self._last_frame = frame
        percepts = []
        for d in self._detector.detect(frame.color):
            u, v = d.center
            xn, yn = frame.pixel_to_normalized(u, v)
            percepts.append(Percept(
                name=d.name, conf=d.conf, track_id=d.track_id, u=u, v=v,
                x_norm=xn, y_norm=yn, depth_m=frame.box_depth(d.xyxy),
                xyxy=d.xyxy))
        return percepts

    def render_base(self) -> tuple[np.ndarray, float, float, float, str]:
        """(image to draw on, fx, cx, cy, caption) for the viewer window."""
        f = self._last_frame
        if f is None:
            img = np.full((480, 640, 3), 40, np.uint8)
            return img, 600.0, 320.0, 240.0, "real camera -- no frame yet"
        return f.color.copy(), f.fx, f.cx, f.cy, "REAL wrist camera + YOLO26"


class _RTDEPoseStream:
    """One persistent low-rate RTDE subscription to the TCP pose.

    Polling ``URClient.get_tcp_pose`` opens a fresh RTDE session (with a
    125 Hz recipe) per call; at perception rates that connection churn is
    enough to starve the PolyScope X simulator into a real-time violation
    and a protective stop. One long-lived socket at 10 Hz costs the
    controller nearly nothing. Reconnects by itself if the stream drops.
    """

    RATE_HZ = 10.0

    def __init__(self, host: str):
        self._host = host
        self._lock = threading.Lock()
        self._pose: list[float] | None = None
        self._t = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="rtde-pose-stream")
        self._thread.start()

    def _run(self) -> None:
        import socket
        import struct
        while not self._stop.is_set():
            try:
                with socket.create_connection((self._host, 30004),
                                              timeout=5) as s:
                    URClient._rtde_send(s, 86, struct.pack(">H", 2))
                    URClient._rtde_recv(s)
                    URClient._rtde_send(
                        s, 79, struct.pack(">d", self.RATE_HZ)
                        + b"actual_TCP_pose")
                    URClient._rtde_recv(s)
                    URClient._rtde_send(s, 83)
                    URClient._rtde_recv(s)
                    while not self._stop.is_set():
                        cmd, body = URClient._rtde_recv(s)
                        if cmd != 85:  # not a data package
                            continue
                        pose = list(struct.unpack(">6d", body[1:49]))
                        with self._lock:
                            self._pose, self._t = pose, time.monotonic()
            except (OSError, ConnectionError):
                time.sleep(1.0)  # controller away; retry

    def pose(self, max_age_s: float = 2.0) -> list[float]:
        with self._lock:
            pose, t = self._pose, self._t
        if pose is None or time.monotonic() - t > max_age_s:
            raise ConnectionError(
                f"No live TCP pose from the robot at {self._host}:30004.")
        return pose


class SimPerception:
    """Virtual objects seen by a virtual camera riding the live sim TCP."""

    def __init__(self, host: str):
        self._host = host
        self._stream: _RTDEPoseStream | None = None
        self._objects: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def start(self) -> str:
        if self._stream is None:
            self._stream = _RTDEPoseStream(self._host)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    self._stream.pose()
                    break
                except ConnectionError:
                    time.sleep(0.1)
            else:
                raise ConnectionError(
                    f"Simulator not reachable at {self._host}:30004 -- is "
                    "the container up and the robot powered on?")
        return "simulated wrist camera (VISION_MODE=sim)"

    def place(self, name: str, position_base_m: list[float]) -> None:
        with self._lock:
            self._objects[name] = np.asarray(position_base_m, dtype=float)

    def place_ahead(self, name: str, forward_m: float, right_m: float,
                    down_m: float) -> list[float]:
        """Drop a virtual object relative to the CURRENT camera pose."""
        pose = self._stream.pose()
        R = rotvec_to_matrix(pose[3:])
        pos = np.array(pose[:3]) + R @ [right_m, down_m, forward_m]
        self.place(name, pos.tolist())
        return [round(float(x), 4) for x in pos]

    def objects(self) -> dict[str, list[float]]:
        with self._lock:
            return {k: [round(float(x), 4) for x in v]
                    for k, v in self._objects.items()}

    def snapshot(self) -> list[Percept]:
        with self._lock:
            objs = dict(self._objects)
        if not objs:
            return []
        pose = self._stream.pose()
        p = np.array(pose[:3])
        R = rotvec_to_matrix(pose[3:])
        percepts = []
        for tid, (name, obj) in enumerate(sorted(objs.items()), start=1):
            rel = obj - p
            z = float(rel @ R[:, 2])   # camera looks along tool +Z
            if z < 0.05:
                continue               # behind or on top of the camera
            xn = float(rel @ R[:, 0]) / z
            yn = float(rel @ R[:, 1]) / z
            if abs(xn) > SIM_HALF_FOV_X or abs(yn) > SIM_HALF_FOV_Y:
                continue               # outside the virtual field of view
            u, v = SIM_CX + xn * SIM_FX, SIM_CY + yn * SIM_FY
            # A nominal 6 cm object drawn with pinhole size falloff, so the
            # viewer window shows it growing as the robot approaches.
            r = float(np.clip(0.03 * SIM_FX / z, 4, 80))
            percepts.append(Percept(
                name=name, conf=0.99, track_id=tid, u=u, v=v,
                x_norm=xn, y_norm=yn, depth_m=z,
                xyxy=(u - r, v - r, u + r, v + r)))
        return percepts

    def render_base(self) -> tuple[np.ndarray, float, float, float, str]:
        """(image to draw on, fx, cx, cy, caption) for the viewer window."""
        w, h = int(2 * SIM_CX), int(2 * SIM_CY)
        img = np.full((h, w, 3), 28, np.uint8)
        # A grid every 0.1 normalized units plus the FOV border, so the
        # window reads as a camera view even when no object is in sight.
        step = int(0.1 * SIM_FX)
        for x in range(int(SIM_CX) % step, w, step):
            cv2.line(img, (x, 0), (x, h), (45, 45, 45), 1)
        for y in range(int(SIM_CY) % step, h, step):
            cv2.line(img, (0, y), (w, y), (45, 45, 45), 1)
        cv2.rectangle(img, (1, 1), (w - 2, h - 2), (70, 70, 70), 1)
        return (img, SIM_FX, SIM_CX, SIM_CY,
                "SIMULATED wrist camera (virtual objects, live sim TCP)")


def _matches(target: str, name: str) -> bool:
    """Loose class matching: 'phone' hits COCO's 'cell phone', etc."""
    t, n = target.strip().lower(), name.strip().lower()
    return t == n or t in n or n in t


@dataclass
class GraspCalibration:
    """Distance from the camera's depth origin to the fingertip plane.

    Measured along the approach (tool +Z) axis, in metres. With it, the
    fingertip gap for any camera reading ``d`` is simply ``d - camera_to_
    fingertip_m``: zero means the fingers are at the taught grasp point.
    """

    camera_to_fingertip_m: float
    source: str            # "taught" or "manual"
    saved_at: str

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "camera_to_fingertip_m": self.camera_to_fingertip_m,
            "source": self.source, "saved_at": self.saved_at}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GraspCalibration | None":
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return cls(camera_to_fingertip_m=float(d["camera_to_fingertip_m"]),
                       source=str(d.get("source", "manual")),
                       saved_at=str(d.get("saved_at", "unknown")))
        except (ValueError, KeyError):
            return None


def _approach_axis(pose: list[float]) -> np.ndarray:
    """Unit vector, in base coordinates, pointing from the tool at the table.

    This is tool +Z: the direction the camera looks and the fingers close
    around, and the axis a depth reading shrinks along.
    """
    return rotvec_to_matrix(pose[3:])[:, 2]


# Reference pose + depth captured by start_grasp_calibration, consumed by
# finish_grasp_calibration. One teach at a time; not persisted on purpose --
# an interrupted teach should not survive a restart as a half-state.
_grasp_teach: dict = {"pose": None, "depth_m": None, "object": None}


def _require_grasp_calibration() -> GraspCalibration:
    cal = GraspCalibration.load(GRASP_CAL_PATH)
    if cal is None:
        raise ValueError(
            "This cell has no grasp calibration yet, so the distance from "
            "the camera to the fingertips is unknown and any descent would "
            "be a guess. Teach it once: put the object on the table, "
            "start_grasp_calibration(object_name), then move the robot "
            "(freedrive is easiest) until the fingers are exactly where "
            "they should be to grip it, and call "
            "finish_grasp_calibration(). Or set it directly with "
            "set_grasp_calibration(camera_to_fingertip_m=...) if you have "
            "measured it.")
    return cal


class Engine:
    """Perception thread + servo worker + target-by-name selection."""

    def __init__(self):
        host = os.environ.get("UR_HOST", "127.0.0.1")
        self.sim = VISION_MODE == "sim"
        self.perception = SimPerception(host) if self.sim else RealPerception()
        self.robot = URClient(host=host)
        self.shared = Shared()
        self.cfg = ServoConfig()
        hand_eye = (HandEye.analytic() if self.sim
                    else HandEye.load(HAND_EYE_PATH) or HandEye.analytic())
        self.worker = ServoWorker(
            self.robot, self.shared, self.cfg, hand_eye, approach=True,
            hand_eye_path=SIM_HAND_EYE_PATH if self.sim else HAND_EYE_PATH)
        self._lock = threading.Lock()
        self._target_name: str | None = None
        self._target_tid: int | None = None
        self._latest: list[Percept] = []
        self._latest_t = 0.0
        self._vision_error: str | None = None
        self._started = False
        self.viewer = Viewer()

    # --- lifecycle -------------------------------------------------------- #
    def ensure_started(self) -> None:
        """Open perception and spin the threads up, once, on first use."""
        with self._lock:
            if self._started:
                if self._vision_error:
                    raise RuntimeError(
                        f"Perception is down: {self._vision_error}")
                return
            source_name = self.perception.start()  # raises readable errors
            self.robot.connect()
            self.worker.start()
            threading.Thread(target=self._vision_loop, daemon=True,
                             name="vision-loop").start()
            self._started = True
            print(f"[vision-tools] perception: {source_name}",
                  file=sys.stderr)
        if VISION_VIEW == "always":
            self.viewer.open()

    def _vision_loop(self) -> None:
        while not self.shared.quit.is_set():
            try:
                percepts = self.perception.snapshot()
                self._vision_error = None
            except Exception as exc:  # camera unplugged, sim down, ...
                self._vision_error = str(exc)
                time.sleep(1.0)
                continue
            now = time.monotonic()
            with self._lock:
                self._latest, self._latest_t = percepts, now
                target = self._pick_target(percepts)
            if target is not None:
                self.shared.set_obs(Observation(
                    x_norm=target.x_norm, y_norm=target.y_norm,
                    depth_m=target.depth_m, t=now))
            if self.viewer.is_open:
                img, fx, cx, cy, caption = self.perception.render_base()
                draw_view(img, percepts, target, fx, cx, cy,
                          self.cfg.deadband_norm, self.shared.get_status(),
                          self.shared.servo_on.is_set(), caption)
                self.viewer.push(img)
            if self.sim:
                time.sleep(0.1)  # the pose stream updates at 10 Hz anyway

    def _pick_target(self, percepts: list[Percept]) -> Percept | None:
        """Sticky selection: keep the tracker id, fall back to the name."""
        if self._target_name is None:
            return None
        if self._target_tid is not None:
            for p in percepts:
                if p.track_id == self._target_tid:
                    return p
        best = None
        for p in percepts:
            if _matches(self._target_name, p.name):
                if best is None or p.conf > best.conf:
                    best = p
        if best is not None:
            self._target_tid = best.track_id
        return best

    # --- queries ---------------------------------------------------------- #
    def visible(self) -> list[Percept]:
        """Objects in a frame captured AFTER this call (read-your-writes:
        a place_sim_object or a move that just finished is always seen)."""
        self.ensure_started()
        t_call = time.monotonic()
        deadline = t_call + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_t > t_call:
                    return list(self._latest)
            if self._vision_error:
                raise RuntimeError(f"Perception is down: {self._vision_error}")
            time.sleep(0.05)
        raise RuntimeError("No fresh camera frame within 5 s.")

    def set_target(self, name: str) -> None:
        with self._lock:
            self._target_name = name
            self._target_tid = None

    def status(self) -> dict:
        text, level = self.shared.get_status()
        with self._lock:
            target = self._target_name
        obs = self.shared.get_obs()
        fresh = (obs is not None
                 and time.monotonic() - obs.t < self.cfg.stale_after_s)
        return {
            "tracking": self.shared.servo_on.is_set(),
            "target": target,
            "state": text,
            "level": level,
            "locked": text.startswith("LOCKED"),
            "target_visible_now": fresh,
            "offcenter_norm": None if not fresh else
                [round(obs.x_norm, 4), round(obs.y_norm, 4)],
            "depth_m": None if not fresh or obs.depth_m is None
                else round(obs.depth_m, 4),
            "standoff_m": self.cfg.standoff_m,
            "mode": "sim" if self.sim else "real",
        }

    def wait_for_lock(self, wait_s: float) -> dict:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            st = self.status()
            if st["locked"]:
                return st
            if not st["tracking"]:   # worker turned itself off: robot error
                return st
            time.sleep(0.2)
        return self.status()


engine = Engine()
mcp = FastMCP("vision-tools")


@mcp.tool
def look() -> dict:
    """See what the wrist camera sees right now: every detected object with
    its class name, confidence, how far it sits from the image centre
    (normalized: 0 is centred, ~0.5 is the image edge; x>0 right, y>0 below)
    and its distance from the camera in metres (null if depth is unknown).

    Call this first to learn what is on the table and the exact object names
    to pass to track_object / descend_on.
    """
    percepts = engine.visible()
    return {
        "camera_mode": "sim" if engine.sim else "real",
        "objects": [{
            "name": p.name, "confidence": round(p.conf, 3),
            "offcenter_norm": [round(p.x_norm, 4), round(p.y_norm, 4)],
            "distance_m": None if p.depth_m is None else round(p.depth_m, 4),
        } for p in percepts],
    }


@mcp.tool
def what_can_you_see() -> dict:
    """Answer "what can you see?" with the plain list of objects YOLO is
    detecting right now.

    The conversational counterpart to look(): the same detections, but
    grouped by class name with a ready-to-read sentence, and without the
    servo geometry. Reach for look() instead when you need the off-centre
    offsets and distances that track_object / descend_on consume.
    """
    percepts = engine.visible()
    camera = engine.perception.start()

    groups: dict[str, list] = {}
    for p in percepts:
        groups.setdefault(p.name, []).append(p)

    objects = []
    for name, seen in sorted(groups.items(),
                             key=lambda kv: -max(p.conf for p in kv[1])):
        depths = [p.depth_m for p in seen if p.depth_m is not None]
        objects.append({
            "name": name,
            "count": len(seen),
            "confidence": round(max(p.conf for p in seen), 3),
            "nearest_m": round(min(depths), 3) if depths else None,
        })

    phrases = [f"{o['count']} {o['name']}s" if o["count"] > 1
               else f"{'an' if o['name'][:1].lower() in 'aeiou' else 'a'} "
                    f"{o['name']}"
               for o in objects]
    if not phrases:
        summary = "I cannot see any objects right now."
    elif len(phrases) == 1:
        summary = f"I can see {phrases[0]}."
    else:
        summary = ("I can see " + ", ".join(phrases[:-1])
                   + f" and {phrases[-1]}.")

    return {
        "summary": summary,
        "object_names": [o["name"] for o in objects],
        "objects": objects,
        "total_detections": len(percepts),
        "camera": camera,
        "camera_mode": "sim" if engine.sim else "real",
    }


@mcp.tool
def track_object(object_name: str, standoff_m: float = 0.35,
                 wait_s: float = 45.0) -> dict:
    """Visually track an object: steer the robot so the named object stays
    centred in the wrist camera, holding ``standoff_m`` metres from it, and
    KEEP holding after this call returns (the object can be slid around and
    the robot follows). Matching is loose ("phone" finds "cell phone");
    call look() first to see the available names.

    Blocks up to ``wait_s`` seconds for the first LOCK, then returns the
    tracking status either way. Tracking continues in the background until
    stop_tracking() or a new track_object call.

    Raises:
        ValueError: bad standoff/wait, or no such object is visible.
    """
    if not MIN_STANDOFF_M <= standoff_m <= 1.0:
        raise ValueError(
            f"standoff_m must be between {MIN_STANDOFF_M} and 1.0 m "
            f"(got {standoff_m}); the camera cannot measure depth closer.")
    if not 0 < wait_s <= MAX_WAIT_S:
        raise ValueError(f"wait_s must be in (0, {MAX_WAIT_S}] s.")
    percepts = engine.visible()
    if not any(_matches(object_name, p.name) for p in percepts):
        names = sorted({p.name for p in percepts}) or ["nothing at all"]
        raise ValueError(
            f"No {object_name!r} in view right now; the camera sees: "
            f"{', '.join(names)}. Reposition the robot or pick one of those.")
    engine.cfg.standoff_m = standoff_m
    engine.set_target(object_name)
    # The last observation and any LOCKED status belong to the previous
    # target/standoff: wipe them so wait_for_lock can only succeed on a
    # lock freshly earned against THIS command.
    engine.shared.clear_obs()
    engine.shared.set_status(f"retargeting to {object_name!r}...", "info")
    if VISION_VIEW != "off":
        engine.viewer.open()  # let the human watch the servoing live
    # A fresh explicit command acknowledges any earlier robot error and
    # re-arms the safety box around wherever the robot is right now.
    engine.shared.error_latch.clear()
    engine.shared.rearm.set()
    engine.shared.servo_on.set()
    return engine.wait_for_lock(wait_s)


@mcp.tool
def descend_on(object_name: str, standoff_m: float = 0.15,
               wait_s: float = 60.0) -> dict:
    """Move IN CLOSE on an object: centre it and close the distance to
    ``standoff_m`` metres (default 0.15 -- near-touching for a wrist camera),
    approaching along the camera's viewing axis. The same as track_object
    with a tight standoff: the approach is depth-regulated, so it slows into
    position and backs off if it overshoots. Use stop_tracking() afterwards
    to freeze the robot where it is.

    Raises:
        ValueError: bad standoff/wait, or no such object is visible.
    """
    return track_object(object_name, standoff_m=standoff_m, wait_s=wait_s)


@mcp.tool
def approach_to_grasp(object_name: str, grasp_depth_m: float = 0.0,
                      wait_s: float = 90.0) -> dict:
    """Bring the OPEN gripper around an object so it can be closed on it.

    ``descend_on`` regulates the CAMERA to a standoff; this regulates the
    FINGERTIPS to the object, using the taught camera-to-fingertip offset
    (the camera sits above the flange, so it is always further away than
    the fingers). Open the gripper BEFORE calling -- this positions the
    fingers around the object and deliberately does not close them, so the
    caller can check the view and then grip.

    Tracking is stopped on return, leaving the robot frozen in the grasp
    pose. If the object sits closer than the camera can measure, the last
    stretch is a single open-loop step along the approach axis, capped at
    ``MAX_BLIND_DESCENT_M``; ``blind_descent_m`` in the result says how much
    of the approach was unservoed.

    Args:
        object_name: Loose match, as in track_object ("phone" finds
            "cell phone"). Call look() first to learn the names.
        grasp_depth_m: How far PAST the taught grasp point to go, in metres.
            0 (default) stops exactly where the calibration was taught.
            Positive descends deeper, negative stops higher; the useful
            range is a couple of centimetres either way.
        wait_s: Seconds to allow for the visual approach.

    Raises:
        ValueError: No calibration yet, no such object visible, or the
            requested depth is out of range.
        RuntimeError: The approach did not converge, so no blind descent
            was attempted.
    """
    cal = _require_grasp_calibration()
    if not -0.05 <= grasp_depth_m <= 0.05:
        raise ValueError(
            f"grasp_depth_m must be within +/-0.05 m of the taught grasp "
            f"point (got {grasp_depth_m}).")
    if not 0 < wait_s <= MAX_WAIT_S:
        raise ValueError(f"wait_s must be in (0, {MAX_WAIT_S}] s.")

    # Where the camera must end up for the fingertips to be at the object.
    target_depth = cal.camera_to_fingertip_m - grasp_depth_m
    if target_depth <= 0:
        raise ValueError(
            f"A grasp {grasp_depth_m} m past the taught point would put the "
            f"camera through the object (target depth {target_depth:.3f} m). "
            f"Reduce grasp_depth_m.")
    if target_depth > 1.0:
        raise ValueError(
            f"Target camera depth {target_depth:.3f} m exceeds the 1.0 m "
            f"tracking limit -- the calibration looks wrong; re-teach it.")

    # Servo as close as depth can still be measured, then finish open-loop.
    servo_depth = max(target_depth, MIN_STANDOFF_M)
    blind = servo_depth - target_depth
    if blind > MAX_BLIND_DESCENT_M:
        raise ValueError(
            f"The grasp point is {blind * 1000:.0f} mm below the closest "
            f"distance the camera can measure ({MIN_STANDOFF_M} m), more "
            f"than the {MAX_BLIND_DESCENT_M * 1000:.0f} mm this tool will "
            f"travel blind. Mount the camera lower, or drive the last "
            f"stretch by hand.")

    st = track_object(object_name, standoff_m=servo_depth, wait_s=wait_s)
    if not st.get("locked"):
        engine.shared.servo_on.clear()
        engine.viewer.close()
        raise RuntimeError(
            f"Visual approach did not lock on {object_name!r} within "
            f"{wait_s:.0f} s (state: {st.get('state')}). The robot is "
            f"stopped where it is and NO blind descent was attempted -- "
            f"descending now would be aiming at an object the camera is "
            f"not tracking.")
    engine.shared.servo_on.clear()   # freeze: the pose is the grasp pose
    engine.viewer.close()

    result = dict(st)
    result["blind_descent_m"] = round(blind, 4)
    result["camera_to_fingertip_m"] = cal.camera_to_fingertip_m
    result["target_camera_depth_m"] = round(target_depth, 4)
    if blind > 1e-4:
        pose = engine.robot.get_tcp_pose()
        goal = np.array(pose[:3]) + _approach_axis(pose) * blind
        engine.robot.move_linear(list(goal) + list(pose[3:]),
                                 GRASP_SPEED_MS, GRASP_ACCEL_MS2)
        result["state"] = (f"at grasp pose ({blind * 1000:.0f} mm of it "
                           f"travelled blind)")
    else:
        result["state"] = "at grasp pose (fully servoed)"
    result["tracking"] = False
    result["next_step"] = ("close the gripper, then verify the grasp before "
                           "lifting")
    return result


@mcp.tool
def start_grasp_calibration(object_name: str) -> dict:
    """Begin teaching where the fingertips are relative to the camera.

    Step 1 of 2. Call this with the object visible at a comfortable
    distance -- depth is read HERE, at a range the camera measures well,
    not down at the grasp where it cannot. Then move the robot until the
    fingers sit exactly where they would grip the object (freedrive is
    easiest) and call finish_grasp_calibration().

    The robot must not be tracking: this records a reference pose, and it
    is only meaningful if the robot holds still afterwards until you move
    it deliberately.

    Raises:
        ValueError: Tracking is still on, the object is not visible, or the
            camera reports no depth for it.
    """
    if engine.shared.servo_on.is_set():
        raise ValueError(
            "Stop tracking first (stop_tracking) -- calibration needs the "
            "robot to stay where you put it.")
    percepts = [p for p in engine.visible() if _matches(object_name, p.name)]
    if not percepts:
        names = sorted({p.name for p in engine.visible()}) or ["nothing"]
        raise ValueError(
            f"No {object_name!r} in view; the camera sees: "
            f"{', '.join(names)}.")
    target = max(percepts, key=lambda p: p.conf)
    if target.depth_m is None:
        raise ValueError(
            f"The camera reports no depth for {object_name!r}, so there is "
            f"no reference to measure from. Move so the object is 0.3-0.5 m "
            f"away and well inside the frame, then retry.")
    pose = engine.robot.get_tcp_pose()
    _grasp_teach["depth_m"] = float(target.depth_m)
    _grasp_teach["pose"] = list(pose)
    _grasp_teach["object"] = target.name
    return {
        "step": "1 of 2",
        "object": target.name,
        "reference_depth_m": round(float(target.depth_m), 4),
        "next_step": ("move the robot until the fingers are exactly at the "
                      "grip position on the object, then call "
                      "finish_grasp_calibration()"),
    }


@mcp.tool
def finish_grasp_calibration() -> dict:
    """Finish teaching the camera-to-fingertip offset, and save it.

    Step 2 of 2, called with the fingers sitting where they would grip the
    object. The offset is the reference depth minus how far the tool has
    travelled along the approach axis since start_grasp_calibration() --
    so it never needs a depth reading at close range, where the camera has
    none. Saved to grasp_calibration.json and used by approach_to_grasp.

    Raises:
        ValueError: No calibration was started, or the movement since then
            does not look like an approach.
    """
    if _grasp_teach.get("pose") is None:
        raise ValueError(
            "Nothing to finish -- call start_grasp_calibration(object_name) "
            "first, with the object visible at a comfortable distance.")
    ref_pose = _grasp_teach["pose"]
    ref_depth = _grasp_teach["depth_m"]
    now = engine.robot.get_tcp_pose()
    travel = float(np.dot(np.array(now[:3]) - np.array(ref_pose[:3]),
                          _approach_axis(ref_pose)))
    offset = ref_depth - travel
    if travel <= 0:
        raise ValueError(
            f"The tool has not moved toward the object since the reference "
            f"pose (travel {travel * 1000:+.0f} mm along the approach "
            f"axis). Move it down to the grip position, then retry.")
    if offset <= 0:
        raise ValueError(
            f"The tool travelled {travel * 1000:.0f} mm but the object was "
            f"only {ref_depth * 1000:.0f} mm away, which would put the "
            f"camera past it. Re-run start_grasp_calibration().")
    cal = GraspCalibration(camera_to_fingertip_m=round(offset, 4),
                           source="taught",
                           saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    cal.save(GRASP_CAL_PATH)
    _grasp_teach["pose"] = None
    return {
        "camera_to_fingertip_m": cal.camera_to_fingertip_m,
        "travelled_m": round(travel, 4),
        "reference_depth_m": round(ref_depth, 4),
        "saved_to": str(GRASP_CAL_PATH),
        "next_step": ("open the gripper and try "
                      "approach_to_grasp(object_name); adjust with "
                      "grasp_depth_m if it stops high or low"),
    }


@mcp.tool
def get_grasp_calibration() -> dict:
    """Report the taught camera-to-fingertip offset, or that none exists."""
    cal = GraspCalibration.load(GRASP_CAL_PATH)
    if cal is None:
        return {"calibrated": False,
                "next_step": ("teach it with start_grasp_calibration(...) "
                              "then finish_grasp_calibration()")}
    return {"calibrated": True,
            "camera_to_fingertip_m": cal.camera_to_fingertip_m,
            "source": cal.source, "saved_at": cal.saved_at,
            "max_blind_descent_m": MAX_BLIND_DESCENT_M,
            "min_measurable_depth_m": MIN_STANDOFF_M}


@mcp.tool
def set_grasp_calibration(camera_to_fingertip_m: float) -> dict:
    """Set the camera-to-fingertip offset directly, in metres.

    Use when the offset has been measured by hand (camera depth origin to
    the fingertip plane, along the tool axis) instead of taught. The
    two-step teach is preferred: it measures the same number the way the
    servo will use it.

    Raises:
        ValueError: The value is outside a plausible range for a wrist
            camera and gripper.
    """
    if not 0.02 <= camera_to_fingertip_m <= 0.60:
        raise ValueError(
            f"camera_to_fingertip_m must be 0.02-0.60 m (got "
            f"{camera_to_fingertip_m}); outside that the mounting does not "
            f"make sense and a grasp would drive into the table.")
    cal = GraspCalibration(camera_to_fingertip_m=float(camera_to_fingertip_m),
                           source="manual",
                           saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    cal.save(GRASP_CAL_PATH)
    return {"camera_to_fingertip_m": cal.camera_to_fingertip_m,
            "source": cal.source, "saved_to": str(GRASP_CAL_PATH)}


@mcp.tool
def stop_tracking() -> dict:
    """Stop visual tracking. The robot freezes where it is (nothing moves
    until the next track_object / descend_on call), and the live camera
    window closes."""
    engine.shared.servo_on.clear()
    engine.viewer.close()
    st = engine.status()
    st["state"] = "tracking stopped"
    return st


@mcp.tool
def show_camera_view() -> dict:
    """Open a live window on the user's screen showing what the wrist
    camera (or, in sim mode, the simulated camera) sees: YOLO detections,
    the tracked target, the centring crosshair and the servo status. It
    opens automatically when tracking starts; call this to let the user
    watch WITHOUT tracking, e.g. while they arrange objects on the table.
    """
    engine.ensure_started()
    engine.viewer.open()
    return {"viewer_open": engine.viewer.is_open}


@mcp.tool
def hide_camera_view() -> dict:
    """Close the live camera window. Tracking, if any, continues -- this
    only removes the visualization."""
    engine.viewer.close()
    return {"viewer_open": False}


@mcp.tool
def tracking_status() -> dict:
    """Current tracking state: what is targeted, whether it is LOCKED
    (centred at the standoff distance), how far off-centre it sits, and the
    measured distance. Poll this after track_object to monitor progress."""
    return engine.status()


@mcp.tool
def calibrate_hand_eye(wait_s: float = 90.0) -> dict:
    """Measure how the camera is mounted (the hand-eye Jacobian) by probing
    each tool axis 15 mm and watching how the image responds. Needs an object
    already being tracked or at least visible and steady. Takes ~30 s of
    small robot motion; tracking must be OFF. The result is saved and reused
    across restarts. Run this once after mounting or re-mounting the camera.
    """
    engine.ensure_started()
    if engine.shared.servo_on.is_set():
        raise ValueError("Tracking is on -- call stop_tracking() first.")
    with engine._lock:
        has_target = engine._target_name is not None
    if not has_target:
        # Calibration needs a steady observation stream: adopt the most
        # confident visible object as the reference target.
        percepts = engine.visible()
        if not percepts:
            raise ValueError("Nothing visible to calibrate against; put an "
                             "object in view first.")
        engine.set_target(max(percepts, key=lambda p: p.conf).name)
    engine.shared.cal_done.clear()
    engine.shared.cal_requested.set()
    if engine.shared.cal_done.wait(timeout=wait_s):
        text, level = engine.shared.get_status()
        return {"ok": level != "error", "result": text,
                "hand_eye_source": engine.worker.hand_eye.source}
    return {"ok": False, "result": f"calibration still running after "
                                   f"{wait_s:.0f} s -- check tracking_status"}


@mcp.tool
def place_sim_object(object_name: str, forward_m: float = 0.45,
                     right_m: float = 0.0, down_m: float = 0.0) -> dict:
    """SIMULATION ONLY: drop a virtual object into the scene, placed relative
    to where the wrist camera currently looks (forward = along the viewing
    axis, right/down = across the image). It stays fixed in the world at that
    spot, so the robot can then track it or descend on it. Example: place a
    "cup" 45 cm ahead and 8 cm right, then track_object("cup").

    Raises:
        ValueError: not in sim mode, or the placement is behind the camera.
    """
    if not engine.sim:
        raise ValueError(
            "place_sim_object only exists in sim mode (VISION_MODE=sim); "
            "the real camera sees real objects.")
    if not 0.15 <= forward_m <= 1.5:
        raise ValueError("forward_m must be 0.15..1.5 m so the object lands "
                         "in front of the camera at a trackable range.")
    engine.ensure_started()
    pos = engine.perception.place_ahead(object_name, forward_m, right_m,
                                        down_m)
    return {"placed": object_name, "position_base_m": pos,
            "all_sim_objects": engine.perception.objects()}


if __name__ == "__main__":
    if "--rs-probe" in sys.argv:
        # Narrate every librealsense step so a native segfault names its site.
        print("1: importing pyrealsense2", flush=True)
        import pyrealsense2 as rs
        print("2: creating context", flush=True)
        ctx = rs.context()
        print("3: query_devices", flush=True)
        devs = list(ctx.query_devices())
        print(f"4: {len(devs)} device(s)", flush=True)
        for d in devs:
            print("   -", d.get_info(rs.camera_info.name), flush=True)
        if devs and "--reset" in sys.argv:
            print("5: hardware_reset", flush=True)
            devs[0].hardware_reset()
            time.sleep(4)
            print("6: after reset:",
                  len(list(ctx.query_devices())), "device(s)",
                  flush=True)
        print("7: starting color-only pipeline", flush=True)
        pipe, cfg = rs.pipeline(ctx), rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipe.start(cfg)
        print("8: waiting for a frame", flush=True)
        pipe.wait_for_frames(7000)
        print("9: got a frame; stopping", flush=True)
        pipe.stop()
        print("10: clean", flush=True)
        sys.exit(0)
    if "--cam-test" in sys.argv:
        # Probe the real camera without the MCP stack or the robot, e.g.
        # `"case 1/run_server.sh" --cam-test`. Prints arrive before
        # exit so they survive librealsense's harmless macOS exit segfault.
        from camera import open_camera
        cam = open_camera(os.environ.get("VISION_CAMERA", "auto"))
        print(f"opened: {cam.name}", flush=True)
        t0 = time.monotonic()
        n, depth_ok = 0, False
        while time.monotonic() - t0 < 3.0:
            f = cam.read()
            n += 1
            depth_ok = depth_ok or f.depth_m is not None
        print(f"frames in 3s: {n} (~{n / 3:.0f} fps), "
              f"depth: {'yes' if depth_ok else 'no'}", flush=True)
        out = os.environ.get("CAM_TEST_SAVE") or next(
            (a for a in sys.argv[1:] if a.endswith((".jpg", ".png"))), None)
        if out:
            cv2.imwrite(out, f.color)
            print(f"saved last frame: {out}", flush=True)
        cam.close()  # exiting with a live pipeline wedges the D435 on macOS
        sys.exit(0)
    mcp.run()
