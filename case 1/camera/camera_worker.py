"""Out-of-process host for the RealSense: the crash blast radius, contained.

librealsense 2.56.5 on macOS segfaults inside
``usb_context::usb_context() -> libusb_get_device_list`` (SIGSEGV,
KERN_INVALID_ADDRESS at 0x28 -- a NULL libusb context used after
``libusb_init()`` failed without being checked). It fires from two places:
the ~1 Hz ``polling_device_watcher`` thread that every ``rs.context()``
spawns, and ``rs2_create_pipeline``, which builds its OWN ``device_hub``
-- and therefore its own ``usb_context`` -- on top of whatever context you
hand it, so caching one context is not enough to avoid it.

A native segfault cannot be caught in Python. In-process that killed the
whole merged ``ur-tools`` MCP server, so a camera fault took every MOTION
tool down with it (the client silently respawned the server, which is why
``look`` reported only "Connection closed"). Here the D435 lives in a child
process instead: when it dies the parent sees a closed pipe plus a negative
exit status, raises a real error naming the crash, and the robot keeps
working.

Protocol, on a private dup of stdout -- line-delimited JSON headers, each
optionally followed by raw payload bytes:

    -> parent   {"ok": true, "name": ...}       once, after the camera opens
                {"ok": false, "error": ...}     the camera would not open
    <- parent   ``READ <timeout_ms>`` | ``CLOSE``
    -> parent   {"ok": true, "h":.., "w":.., "color_bytes":.., ...}
                followed by <color bytes><depth bytes>
                {"ok": false, "error": ...}     that one read failed

Color is HxWx3 uint8 BGR and depth is HxW float32 metres -- the same arrays
``RealSenseCamera.read`` already produces, so the parent rebuilds an
identical ``Frame`` and nothing downstream can tell the difference.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def _emit(out, header: dict, *payloads: bytes) -> None:
    out.write(json.dumps(header).encode() + b"\n")
    for payload in payloads:
        if payload:
            out.write(payload)
    out.flush()


def main() -> int:
    # librealsense, OpenCV and the odd import banner all write to stdout
    # unprompted, and one stray byte would desynchronize the frame stream.
    # Keep the real stdout for the protocol alone and point fd 1 at stderr,
    # where the server's log already collects the "[camera] ..." steps.
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    width = int(os.environ.get("VISION_CAMERA_WIDTH", "640"))
    height = int(os.environ.get("VISION_CAMERA_HEIGHT", "480"))
    fps = int(os.environ.get("VISION_CAMERA_FPS", "30"))

    try:
        from camera import RealSenseCamera
        cam = RealSenseCamera(width=width, height=height, fps=fps)
    except Exception as exc:
        # RealSenseCamera's message already carries the actionable hint
        # (unplug/replug, run as root, USB3 not a hub) -- pass it through
        # verbatim so the tool caller sees it unchanged.
        _emit(proto, {"ok": False, "error": str(exc)})
        return 1
    _emit(proto, {"ok": True, "name": cam.name})

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break  # parent went away
        request = line.decode("utf-8", "replace").strip()
        if not request or request == "CLOSE":
            break
        if not request.startswith("READ"):
            _emit(proto, {"ok": False, "error": f"unknown request {request!r}"})
            continue
        parts = request.split()
        timeout_ms = int(parts[1]) if len(parts) > 1 else 2000
        try:
            frame = cam.read(timeout_ms)
        except Exception as exc:
            _emit(proto, {"ok": False, "error": str(exc)})
            continue
        color = np.ascontiguousarray(frame.color, dtype=np.uint8)
        depth = (None if frame.depth_m is None
                 else np.ascontiguousarray(frame.depth_m, dtype=np.float32))
        _emit(proto,
              {"ok": True,
               "h": int(color.shape[0]), "w": int(color.shape[1]),
               "color_bytes": int(color.nbytes),
               "depth_bytes": 0 if depth is None else int(depth.nbytes),
               "fx": frame.fx, "fy": frame.fy,
               "cx": frame.cx, "cy": frame.cy},
              color.tobytes(),
              b"" if depth is None else depth.tobytes())

    try:
        cam.close()  # exiting with a live pipeline wedges the D435 on macOS
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
