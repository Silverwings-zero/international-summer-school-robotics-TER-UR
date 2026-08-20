"""YOLO26 detection seam: frames in, a small list of Detections out.

Wraps the ultralytics API (YOLO26 is end-to-end / NMS-free, but the calling
convention is unchanged from YOLOv8/11) so the servo loop never touches
tensors. Tracking mode (``model.track(persist=True)``) gives every object a
stable integer id across frames, which is what lets "the object of interest"
survive the robot moving, the object shifting in the image, and brief
occlusions -- the id, not the pixel position, is the identity of the target.

First run downloads the weights (a few MB for yolo26n) next to this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """One detected object in one frame."""

    name: str            # COCO class name, e.g. "cup"
    conf: float          # detector confidence, 0..1
    xyxy: tuple[float, float, float, float]  # box corners, pixels
    track_id: int | None  # stable across frames in tracking mode, else None

    @property
    def center(self) -> tuple[float, float]:
        """Box centre (u, v) in pixels -- the point the servo law centres."""
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


class ObjectDetector:
    """YOLO26 with per-frame tracking.

    Args:
        model_name: any ultralytics detection weight ("yolo26n.pt" nano is
            real-time on a laptop; "yolo26s.pt" if you have headroom).
        conf: minimum confidence kept.
        device: "mps" on Apple Silicon, "cuda", "cpu", or None to let
            ultralytics choose.
    """

    def __init__(self, model_name: str = "yolo26n.pt", conf: float = 0.35,
                 device: str | None = None):
        from ultralytics import YOLO  # deferred: import is slow (~seconds)
        self._model = YOLO(model_name)
        self._conf = conf
        self._device = device
        self.names: dict[int, str] = dict(self._model.names)

    def detect(self, frame_bgr: np.ndarray, track: bool = True) -> list[Detection]:
        """Run YOLO26 on one BGR frame.

        Args:
            track: True runs the tracker (stable ``track_id`` per object,
                needs consecutive frames of the same stream); False is a
                plain single-image detection with ``track_id=None``.
        """
        kwargs = dict(conf=self._conf, verbose=False)
        if self._device is not None:
            kwargs["device"] = self._device
        if track:
            results = self._model.track(frame_bgr, persist=True, **kwargs)
        else:
            results = self._model(frame_bgr, **kwargs)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        ids = (boxes.id.cpu().numpy().astype(int).tolist()
               if boxes.id is not None else [None] * len(confs))
        return [
            Detection(name=self.names.get(c, str(c)), conf=float(p),
                      xyxy=tuple(float(v) for v in b), track_id=i)
            for b, p, c, i in zip(xyxy, confs, clses, ids)
        ]
