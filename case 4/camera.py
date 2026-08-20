"""Camera seam for the eye-in-hand servo loop.

Two implementations behind one tiny interface (``read() -> Frame``):

  * ``RealSenseCamera`` -- an Intel RealSense D435 on USB. Streams color plus
    depth ALIGNED to the color image, and reads the factory intrinsics off the
    device, so a pixel + its depth convert straight to a 3D offset in the
    camera frame. This is the one you want on the real setup. By default it
    is reached through ``SubprocessCamera``, which runs it in a child process
    so a librealsense segfault cannot take the caller down with it.
  * ``WebcamCamera`` -- any UVC webcam (including the D435's RGB sensor when
    the librealsense Python bindings are not installed on this machine).
    No depth; the servo loop then centers in the image plane only and skips
    the approach axis, assuming ``assumed_depth_m`` for gain scaling.

``open_camera()`` picks whichever works and says which one you got. Nothing
else in case 4 imports pyrealsense2 directly, so the rest of the code cannot
break on a machine where it is missing.
"""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    """One synchronized camera sample.

    Attributes:
        color: HxWx3 BGR image (what YOLO and the overlay consume).
        depth_m: HxW float32 depth in METRES aligned to ``color``, or None
            when the camera cannot measure depth. Zeros mean "no reading at
            this pixel" (shadows, dark or shiny surfaces), not zero distance.
        fx, fy, cx, cy: pinhole intrinsics of the color image, pixels.
        t: time.monotonic() when the frame arrived.
    """

    color: np.ndarray
    depth_m: np.ndarray | None
    fx: float
    fy: float
    cx: float
    cy: float
    t: float

    def pixel_to_normalized(self, u: float, v: float) -> tuple[float, float]:
        """Pixel -> normalized image coordinates ((u-cx)/fx, (v-cy)/fy).

        Normalized coordinates are what the servo law uses: they are
        camera-resolution independent and, multiplied by depth, become metres
        in the camera's X (right) / Y (down) axes.
        """
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy

    def box_depth(self, xyxy: tuple[float, float, float, float]
                  ) -> float | None:
        """Robust depth (metres) of the object inside a detection box.

        The box centre pixel can land on a hole in the object (a mug handle,
        a phone's dark screen) and read the TABLE behind it -- which would
        make an approach overshoot into the object. Sampling the central
        half of the box and taking a LOW percentile of the valid depths
        prefers the object surface over the background peeking through.
        """
        if self.depth_m is None:
            return None
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        qw, qh = max(2.0, (x2 - x1) / 4), max(2.0, (y2 - y1) / 4)
        h, w = self.depth_m.shape
        u0, u1 = max(0, int(cx - qw)), min(w, int(cx + qw) + 1)
        v0, v1 = max(0, int(cy - qh)), min(h, int(cy + qh) + 1)
        window = self.depth_m[v0:v1, u0:u1]
        valid = window[(window > 0.05) & np.isfinite(window)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 20))

    def depth_at(self, u: float, v: float, patch: int = 5) -> float | None:
        """Median valid depth (metres) in a ``patch``x``patch`` window.

        A single depth pixel on an object edge is noisy or missing; the median
        over a small window centred on the detection is stable. Returns None
        when depth is unavailable or every pixel in the window is a hole.
        """
        if self.depth_m is None:
            return None
        h, w = self.depth_m.shape
        r = patch // 2
        u0, v0 = int(round(u)), int(round(v))
        window = self.depth_m[max(0, v0 - r):min(h, v0 + r + 1),
                              max(0, u0 - r):min(w, u0 + r + 1)]
        valid = window[(window > 0.05) & np.isfinite(window)]
        if valid.size == 0:
            return None
        return float(np.median(valid))


# librealsense race, seen as SIGSEGV in libusb_get_device_list: every
# rs.context() spawns an internal device-watcher thread, and destroying one
# context while another's watcher is mid-poll locks a freed libusb mutex.
# One process therefore gets exactly ONE context, created once and never
# destroyed; every enumerate/reset/pipeline goes through it.
_RS_CTX = None


def _rs_context(rs):
    global _RS_CTX
    if _RS_CTX is None:
        _RS_CTX = rs.context()
    return _RS_CTX


class RealSenseCamera:
    """Intel RealSense D435: color + depth aligned to color, USB3 preferred.

    Raises:
        RuntimeError: pyrealsense2 is not installed, or no device is plugged
            in / claimed by another process.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed in this environment "
                "(see case 4/README.md for the macOS install path)."
            ) from exc
        self._rs = rs
        # On macOS every librealsense process exit -- even after a clean
        # pipeline.stop() -- segfaults in teardown and leaves the D435
        # wedged for the NEXT opener, sometimes as a native segfault inside
        # pipeline.start() that no except-clause can catch. Resetting the
        # device up front sidesteps the segfault path entirely; the first
        # start cycle after a reset delivers no frames, which is what the
        # second pass below is for. RS_HW_RESET=off skips the reset.
        def _step(msg: str) -> None:
            print(f"[camera] {msg}", file=sys.stderr, flush=True)

        ctx = _rs_context(rs)
        if os.environ.get("RS_HW_RESET", "on").lower() not in ("off", "0"):
            try:
                _step("enumerating for reset")
                devs = list(ctx.query_devices())
                if devs:
                    _step("hardware reset")
                    devs[0].hardware_reset()
                    time.sleep(4.0)  # let the device re-enumerate
                    _step(f"after reset: {len(list(ctx.query_devices()))} "
                          f"device(s)")
            except RuntimeError as exc:
                _step(f"hardware reset failed: {exc}")
        # A USB2 link cannot carry color+depth at 640x480@30 -- and worse, a
        # pipeline can accept that config, start(), and then never deliver a
        # frame. So each candidate profile is probed for one actual frame
        # before being accepted, stepping down to USB2-safe rates and then to
        # color-only (a single stream always fits USB2; depth_m is None).
        candidates = [(width, height, fps, True), (640, 480, 15, True),
                      (640, 480, 30, False), (640, 480, 15, False)]
        candidates = [c for i, c in enumerate(candidates)
                      if c not in candidates[:i]]
        profile, errors, with_depth = None, [], True
        self._pipeline = None
        for attempt in (1, 2):
            for w, h, f, d in candidates:
                config = rs.config()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                if d:
                    config.enable_stream(rs.stream.depth, w, h,
                                         rs.format.z16, f)
                # A pipeline object that ever starved reports "No device
                # connected" for every later start in the same process --
                # each attempt needs a FRESH pipeline.
                pipeline = rs.pipeline(ctx)
                started = False
                try:
                    _step(f"trying {w}x{h}@{f}"
                          f"{'+depth' if d else ' color-only'}")
                    profile = pipeline.start(config)
                    started = True
                    _step("started; probing for a frame")
                    pipeline.wait_for_frames(5000)  # link delivers?
                    _step("frames flowing")
                    self._pipeline = pipeline
                    width, height, fps, with_depth = w, h, f, d
                    break
                except RuntimeError as exc:
                    errors.append(
                        f"{w}x{h}@{f}{'+depth' if d else ' color-only'}: "
                        f"{exc}")
                    profile = None
                    if started:
                        # stop() a never-started pipeline can hang natively
                        try:
                            pipeline.stop()
                        except RuntimeError:
                            pass
                    time.sleep(1.0)  # let the device settle after a failure
                    if "No device connected" in str(exc):
                        # The streaming interface is claimed elsewhere (a
                        # crashed or live process): every remaining profile
                        # fails the same way -- skip to the reset, or give
                        # up so the caller sees the message below.
                        break
            if profile is not None or attempt == 2:
                break
            print("[camera] no frames on pass 1 (normal right after a "
                  "reset); retrying all profiles", file=sys.stderr,
                  flush=True)
        if profile is None:
            detail = "; ".join(errors)
            if "power state" in detail:
                hint = ("macOS blocks raw USB device control for non-root "
                        "processes: run the script with sudo (documented "
                        "librealsense requirement on macOS 12+).")
            elif "No device connected" in detail:
                hint = ("The camera enumerates but its streaming interface "
                        "cannot be claimed -- another (possibly crashed) "
                        "process holds it. Unplug and replug the D435, then "
                        "try again.")
            else:
                hint = ("Is the D435 plugged in (directly into a USB3 port, "
                        "not a low-power hub) and not in use by another "
                        "program?")
            raise RuntimeError(
                f"Could not start the RealSense pipeline ({detail}). {hint}")
        self._depth_scale, self._align = 0.0, None
        if with_depth:
            # Depth pixels come as uint16 device units; convert to metres.
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
            # Reproject depth into the color camera so (u,v) indexes both.
            self._align = rs.align(rs.stream.color)
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self._fx, self._fy = float(intr.fx), float(intr.fy)
        self._cx, self._cy = float(intr.ppx), float(intr.ppy)
        self.name = (f"RealSense D435 ("
                     f"{'color+depth' if with_depth else 'color only'}, "
                     f"{width}x{height}@{fps})")

    def read(self, timeout_ms: int = 2000) -> Frame:
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame() if self._align is not None else None
        if not color:
            raise RuntimeError("RealSense returned no color frame.")
        color_np = np.asanyarray(color.get_data()).copy()
        depth_np = None
        if depth:
            depth_np = (np.asanyarray(depth.get_data())
                        .astype(np.float32) * self._depth_scale)
        return Frame(color=color_np, depth_m=depth_np,
                     fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
                     t=time.monotonic())

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass  # already stopped


class SubprocessCamera:
    """A ``RealSenseCamera`` in a child process, behind the same ``read()``.

    librealsense 2.56.5 on macOS segfaults in
    ``usb_context::usb_context() -> libusb_get_device_list`` (SIGSEGV at
    0x28: a NULL libusb context used after ``libusb_init()`` failed). It
    fires from ``rs2_create_pipeline`` and from the ~1 Hz device-watcher
    thread every ``rs.context()`` starts -- and ``rs.pipeline(ctx)`` builds
    its own ``device_hub``/``usb_context`` regardless of the cached
    ``_RS_CTX``, so no amount of Python discipline prevents it and no
    ``except`` clause can catch a native crash.

    In the merged ``ur-tools`` server that crash killed the process, so a
    camera fault also took every MOTION tool down (the MCP client silently
    respawned it, surfacing only as "Connection closed"). Isolating the
    camera turns that into an ordinary tool error: the pipe closes, the exit
    status is negative, and the robot half never notices.

    ``camera_worker.py`` documents the wire protocol. This side is a plain
    request/response client with its own buffer -- ``select`` on the raw fd
    would miss bytes already sitting in a Python-level buffer, so the reads
    below go through ``os.read`` only.
    """

    #: Opening the D435 walks a retry ladder (up to eight profile attempts,
    #: each with a 5 s frame probe) and may hardware-reset first, so the
    #: handshake needs a generous budget. A crash still surfaces at once:
    #: the pipe hits EOF rather than idling out this deadline.
    OPEN_TIMEOUT_S = 150.0
    #: Added to the caller's ``timeout_ms`` to cover the frame's trip up the
    #: pipe, so the worker's own timeout is what actually fires first.
    READ_GRACE_S = 5.0

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        here = os.path.dirname(os.path.abspath(__file__))
        worker = os.path.join(here, "camera_worker.py")
        env = dict(os.environ, VISION_CAMERA_WIDTH=str(width),
                   VISION_CAMERA_HEIGHT=str(height),
                   VISION_CAMERA_FPS=str(fps), PYTHONUNBUFFERED="1")
        # stderr is inherited on purpose: the worker's "[camera] trying
        # 640x480@30+depth" steps then land in the server's stderr, which is
        # where anyone debugging a camera fault already looks.
        self._proc = subprocess.Popen(
            [sys.executable, worker], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0, cwd=here, env=env)
        self._fd = self._proc.stdout.fileno()
        self._buf = bytearray()
        try:
            hello = json.loads(
                self._read_line(time.monotonic() + self.OPEN_TIMEOUT_S))
            if not hello.get("ok"):
                raise RuntimeError(hello.get("error", "camera worker failed"))
        except BaseException:
            self._terminate()
            raise
        self.name = f"{hello['name']} [out-of-process]"

    # -- wire helpers --------------------------------------------------
    def _fill(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "The camera worker stopped answering (timed out). "
                "stop_tracking, then retry; if it persists, unplug and "
                "replug the D435.")
        if select.select([self._fd], [], [], remaining)[0]:
            chunk = os.read(self._fd, 1 << 16)
            if not chunk:
                raise self._worker_died()
            self._buf += chunk

    def _read_line(self, deadline: float) -> bytes:
        while b"\n" not in self._buf:
            self._fill(deadline)
        line, _, rest = bytes(self._buf).partition(b"\n")
        self._buf = bytearray(rest)
        return line

    def _read_exact(self, n: int, deadline: float) -> bytes:
        while len(self._buf) < n:
            self._fill(deadline)
        out, self._buf = bytes(self._buf[:n]), self._buf[n:]
        return out

    def _send(self, request: str) -> None:
        try:
            self._proc.stdin.write(request.encode() + b"\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise self._worker_died() from exc

    def _worker_died(self) -> RuntimeError:
        """The error to raise once the worker's pipe is gone."""
        code = self._proc.poll()
        if code is None:
            try:
                code = self._proc.wait(timeout=2.0)  # may still be exiting
            except subprocess.TimeoutExpired:
                code = None
        if code is not None and code < 0:
            try:
                named = signal.Signals(-code).name
            except ValueError:      # a signal number Python does not name
                named = f"signal {-code}"
            crash = ""
            if -code == signal.SIGSEGV:
                crash = (" -- the known librealsense/libusb segfault in "
                         "usb_context()")
            return RuntimeError(
                f"The camera worker was killed by {named}{crash}. The robot "
                "is unaffected and motion tools still work. The D435 usually "
                "needs an unplug/replug into a USB3 port (no hub) before it "
                "will stream again.")
        return RuntimeError(
            f"The camera worker exited (status {code}) without answering. "
            "Its [camera] log lines on stderr say how far it got.")

    def _terminate(self) -> None:
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass

    # -- Camera interface ----------------------------------------------
    def read(self, timeout_ms: int = 2000) -> Frame:
        deadline = time.monotonic() + timeout_ms / 1000.0 + self.READ_GRACE_S
        self._send(f"READ {int(timeout_ms)}")
        head = json.loads(self._read_line(deadline))
        if not head.get("ok"):
            raise RuntimeError(head.get("error", "camera read failed"))
        h, w = int(head["h"]), int(head["w"])
        # .copy() because frombuffer is read-only and the detector and the
        # viewer both draw on the image they are handed.
        color = np.frombuffer(
            self._read_exact(int(head["color_bytes"]), deadline),
            np.uint8).reshape(h, w, 3).copy()
        depth = None
        if head["depth_bytes"]:
            depth = np.frombuffer(
                self._read_exact(int(head["depth_bytes"]), deadline),
                np.float32).reshape(h, w).copy()
        return Frame(color=color, depth_m=depth,
                     fx=float(head["fx"]), fy=float(head["fy"]),
                     cx=float(head["cx"]), cy=float(head["cy"]),
                     t=time.monotonic())

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._send("CLOSE")  # lets it stop() the pipeline cleanly
                self._proc.wait(timeout=5.0)
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
        self._terminate()


class WebcamCamera:
    """Any UVC camera through OpenCV. Color only -- no depth.

    Intrinsics are ESTIMATED from an assumed horizontal field of view (the
    D435 RGB sensor is ~69 degrees). Good enough for centering, since the
    servo law only needs directions and rough scale; do not use these numbers
    for metric reconstruction.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 hfov_deg: float = 69.0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {index}. On macOS, grant the "
                "terminal camera permission (System Settings > Privacy)."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, probe = self._cap.read()
        if not ok:
            raise RuntimeError(f"Webcam index {index} opened but returns no frames.")
        h, w = probe.shape[:2]
        self._fx = self._fy = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
        self._cx, self._cy = w / 2, h / 2
        self.name = f"webcam #{index} (color only, {w}x{h})"

    def read(self) -> Frame:
        ok, color = self._cap.read()
        if not ok:
            raise RuntimeError("Webcam stopped returning frames.")
        return Frame(color=color, depth_m=None,
                     fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy,
                     t=time.monotonic())

    def close(self) -> None:
        self._cap.release()


def _open_realsense(width: int = 640, height: int = 480, fps: int = 30):
    """Open the D435, out-of-process unless VISION_CAMERA_SUBPROCESS=off.

    The escape hatch exists for debugging (a traceback is easier to read
    when the camera runs in your own process) and for camera_worker.py
    itself, which is already the child -- everything else wants the
    isolation, because librealsense crashes natively. See SubprocessCamera.
    """
    if os.environ.get("VISION_CAMERA_SUBPROCESS",
                      "on").lower() in ("off", "0", "false"):
        return RealSenseCamera(width, height, fps)
    return SubprocessCamera(width, height, fps)


def open_camera(prefer: str = "auto", webcam_index: int = 0):
    """Open the best available camera.

    Args:
        prefer: "realsense" (fail if unavailable), "webcam", or "auto"
            (RealSense first, fall back to webcam with a printed warning).
        webcam_index: OpenCV device index for the webcam path.
    """
    if prefer not in ("auto", "realsense", "webcam"):
        raise ValueError(f"Unknown camera preference {prefer!r}.")
    if prefer in ("auto", "realsense"):
        try:
            cam = _open_realsense()
            print(f"[camera] {cam.name}")
            return cam
        except RuntimeError as exc:
            if prefer == "realsense":
                raise
            print(f"[camera] RealSense unavailable ({exc})")
            print("[camera] falling back to webcam -- centering only, no depth")
    cam = WebcamCamera(index=webcam_index)
    print(f"[camera] {cam.name}")
    return cam
