"""MCP server: the wrist camera as LLM-callable tools.

Turns the wrist camera's visual servoing into language: "track the cup" becomes
``track_object("cup")``, and "now pick it up" becomes
``grasp_tracked_object()``. The tools wrap the same engine ``servo.py`` drives
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
    pipeline; only the pixels are synthetic. ``SimPerception.place_ahead`` puts
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
# Where a LOCKED track_object leaves the gripper, relative to the grasp point.
# Measured by hand on this cell: 3 cm ABOVE the object, and 1.5 cm to the
# operator's RIGHT when they stand in front of the robot looking at it. That
# gap is a camera-versus-fingertip MOUNT offset -- servoing centres the camera,
# not the jaws -- so it is fixed in the TOOL frame and stays correct whatever
# bearing the base is at.
#
# Tool axes at the home/observation orientation (tool pointing straight down,
# TCP rotvec ~[2.221, 2.221, 0] -- a 180 deg turn about (1,1,0)):
#     tool +X = base +Y     tool +Y = base +X     tool +Z = base -Z (down)
# At home the arm reaches toward base -X, so "in front of the robot" is that
# side, facing the base: the operator's RIGHT is base -Y = tool -X. The gripper
# sits on that side of the object, so the correction goes the other way, +X.
# Down is +Z, the approach axis.
#
# Re-measure if the camera or the gripper is re-mounted; nothing measures this
# automatically. grasp_tracked_object(dry_run=True) prints what it resolves to
# in base coordinates, which is the quick way to check a change.
GRASP_CORRECTION_TOOL_M = (0.015, 0.0, 0.030)   # (x, y, z) in the tool frame
GRASP_CLOSE_DWELL_S = 1.5     # blind wait for the jaws: nothing reports back
SERVO_SETTLE_S = 3.0          # how long to let a servo step land before moving
# Nothing is fed back during that move (the camera is blind this close and the
# gripper has no sensing), so cap how far the routine will ever go open-loop.
MAX_BLIND_MOVE_M = 0.06
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
        a placement or a move that just finished is always seen)."""
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
    to pass to track_object.
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
    offsets and distances that track_object consumes.
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
def stop_tracking() -> dict:
    """Stop visual tracking. The robot freezes where it is (nothing moves
    until the next track_object call), and the live camera
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
def grasp_tracked_object(lift_m: float = 0.10, dry_run: bool = False) -> dict:
    """Grasp the object track_object is LOCKED onto: correct, close, lift.

    The fixed pick routine for this cell. Servoing centres the CAMERA on the
    object, and the fingertips are not where the camera is -- measured by hand
    on this cell, a lock leaves the tool 3 cm above the grasp point and 1.5 cm
    to the operator's right of it (standing in front of the robot). This tool closes that known gap in one shot, then grips
    and lifts. Nothing is measured while it runs: the camera cannot focus this
    close and the gripper reports nothing, so the whole move is open-loop and
    deliberately small, slow, and capped.

    Order of operations: freeze tracking, correct the offset along the tool's
    own axes at 30 mm/s, select slow gripper speed, close the jaws, wait for
    them to travel, then lift straight up.

    Call it right after ``track_object`` reports LOCKED. Open the gripper
    first (``set_tool_digital_out(n=0, b=true)``) -- this tool never opens it,
    because opening a full gripper over the table would drop whatever it holds.

    Args:
        lift_m: How far to lift straight up after gripping, 0 to 0.30 m.
            Pass 0 to grip without lifting.
        dry_run: Report the motion this would command and change nothing --
            tracking keeps running and the arm does not move. Worth one call
            the first time, to check the sideways correction goes toward the
            object rather than away from it.

    Returns:
        A dict with ``status`` ("gripped" or "planned" for a dry run), the
        ``target`` object name, ``correction_tool_m`` (the offset applied,
        in the tool frame) and ``correction_base_m`` (the same motion in base
        coordinates), ``tcp_before`` / ``tcp_after``, and ``verify``: this
        gripper has NO feedback, so the caller must confirm the grasp by
        eye or by asking the user before trusting it.

    Raises:
        ValueError: Not locked onto anything, or lift_m out of range.
        RuntimeError: A move was refused or the controller stopped answering.
    """
    if not 0.0 <= lift_m <= 0.30:
        raise ValueError(f"lift_m must be within 0..0.30 m, got {lift_m}.")

    st = engine.status()
    # "locked" is the servo's status LINE, and that line outlives the servo:
    # it still reads LOCKED long after stop_tracking, when the arm may have
    # been driven somewhere else entirely. Only a lock that is still being
    # held counts, so require live servoing too.
    if not (st["locked"] and st["tracking"]):
        raise ValueError(
            f"Not locked onto anything right now (tracking={st['tracking']}, "
            f"state: {st['state']!r}). Run track_object(object_name) and wait "
            "for LOCKED, then call this immediately -- it only corrects a "
            "known offset from where the lock left the arm, so it cannot "
            "find the object and cannot recover a stale lock.")

    if not dry_run:
        # Freeze first: the servo worker owns the same arm, and a correction
        # it does not know about would fight the next servo step. A dry run
        # skips this -- it promises to change nothing, and stopping the
        # tracking loop is a change.
        engine.shared.servo_on.clear()
        engine.viewer.close()
        # The worker may be part way through a movel. Let that land before
        # taking the arm, or two URScript programs drive the same joints.
        settle_deadline = time.monotonic() + SERVO_SETTLE_S
        while time.monotonic() < settle_deadline:
            if not engine.robot.get_state().is_moving:
                break
            time.sleep(0.15)
        else:
            raise RuntimeError(
                "The arm is still moving after stopping the servo loop; "
                "refusing to command a grasp on top of it. Call "
                "tracking_status and get_robot_state to see what it is doing.")

    pose = engine.robot.get_tcp_pose()
    R = rotvec_to_matrix(pose[3:])
    # Tool frame: +Z is the approach axis (toward the object), +X is the
    # sideways axis. See GRASP_CORRECTION_TOOL_M for how those map to the
    # operator's view.
    correction_tool = np.array(GRASP_CORRECTION_TOOL_M, dtype=float)
    reach = float(np.linalg.norm(correction_tool))
    if reach > MAX_BLIND_MOVE_M:
        raise ValueError(
            f"The configured correction is {reach * 1000:.0f} mm, more than "
            f"the {MAX_BLIND_MOVE_M * 1000:.0f} mm this routine will travel "
            "without seeing. Re-measure the offsets at the top of "
            "vision_tools.py.")
    grasp_p = np.array(pose[:3]) + R @ correction_tool
    plan = {
        "target": st["target"],
        "correction_tool_m": [round(float(v), 4) for v in correction_tool],
        "correction_base_m": [round(float(v), 4) for v in (R @ correction_tool)],
        "tcp_before": [round(v, 4) for v in pose],
        "grasp_pose": [round(float(v), 4) for v in grasp_p],
        "lift_m": lift_m,
    }
    if dry_run:
        return {"status": "planned", **plan,
                "verify": ("Nothing moved, and tracking is still running. "
                           "Check correction_base_m points TOWARD the object; "
                           "if it points away, fix the signs in "
                           "GRASP_CORRECTION_TOOL_M in vision_tools.py.")}

    try:
        # 1. Close the measured gap, slowly -- hands may be near.
        engine.robot.move_linear(list(grasp_p) + list(pose[3:]),
                                 GRASP_SPEED_MS, GRASP_ACCEL_MS2)
        # 2. Grip: slow speed line first, then the jaws (pin 0 low = close).
        engine.robot.set_tool_digital_out(1, True)
        engine.robot.set_tool_digital_out(0, False)
        # The jaws report nothing, so the only way to let them finish is to
        # wait out their travel.
        time.sleep(GRASP_CLOSE_DWELL_S)
        # 3. Lift straight up in the BASE frame, so the object clears the
        #    table however the tool happens to be oriented.
        lifted = [grasp_p[0], grasp_p[1], grasp_p[2] + lift_m]
        if lift_m > 0:
            engine.robot.move_linear(list(lifted) + list(pose[3:]),
                                     GRASP_SPEED_MS, GRASP_ACCEL_MS2)
    except TimeoutError as exc:
        raise RuntimeError(
            f"The grasp move was not confirmed: {exc}. Check the robot state "
            "before commanding anything else -- the arm may be part way "
            "through the routine.") from exc

    after = engine.robot.get_tcp_pose()
    return {
        "status": "gripped",
        **plan,
        "tcp_after": [round(v, 4) for v in after],
        "gripper": "closed (slow)",
        "verify": ("This gripper has no feedback: 'gripped' means the "
                   "commands were sent and confirmed, NOT that the object is "
                   "held. Look at it (look / what_can_you_see) or ask the "
                   "user before moving on."),
    }


@mcp.tool
def tracking_status() -> dict:
    """Current tracking state: what is targeted, whether it is LOCKED
    (centred at the standoff distance), how far off-centre it sits, and the
    measured distance. Poll this after track_object to monitor progress."""
    return engine.status()


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
