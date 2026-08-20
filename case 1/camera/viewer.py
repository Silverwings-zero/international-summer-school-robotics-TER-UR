"""Tiny frame-viewer subprocess for the vision-tools MCP server.

Reads length-prefixed JPEG frames on stdin and shows them in an OpenCV
window. It exists as a SEPARATE PROCESS because macOS demands that GUI
windows are driven from a process's main thread, and the MCP server's main
thread belongs to the stdio protocol loop. As a subprocess, the window gets
a main thread of its own, and a viewer crash can never touch the server.

Lifecycle: the server opens the window by spawning this script and closes it
by closing the pipe (EOF ends the loop below). Pressing q/ESC in the window
closes the VIEW only -- the robot keeps doing whatever it was told; use the
stop_tracking tool to stop motion.
"""
from __future__ import annotations

import struct
import sys

import cv2
import numpy as np

WINDOW = "vision-tools -- wrist camera"


def main() -> None:
    stdin = sys.stdin.buffer
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            header = stdin.read(4)
            if len(header) < 4:  # pipe closed: the server said goodbye
                break
            (size,) = struct.unpack(">I", header)
            payload = stdin.read(size)
            if len(payload) < size:
                break
            img = cv2.imdecode(np.frombuffer(payload, np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imshow(WINDOW, img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
