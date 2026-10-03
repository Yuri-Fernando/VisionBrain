from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from visionbrain.camera.base import CameraRead
from visionbrain.config import CameraConfig


class SyntheticCamera:
    """Deterministic frame source for CI, notebooks and offline demos."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.frame_index = 0
        self._open = False

    def open(self) -> "SyntheticCamera":
        self._open = True
        self.frame_index = 0
        return self

    def read(self) -> CameraRead:
        if not self._open:
            raise RuntimeError("Synthetic camera is not open.")
        height, width = self.config.height, self.config.width
        x_gradient = np.linspace(25, 95, width, dtype=np.uint8)
        frame = np.repeat(x_gradient[None, :, None], height, axis=0)
        frame = np.repeat(frame, 3, axis=2)
        box_w = max(24, width // 10)
        box_h = max(24, height // 7)
        travel = max(width - box_w, 1)
        x1 = int((self.frame_index * max(width // 40, 1)) % travel)
        y1 = max(0, height // 2 - box_h // 2)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (40, 210, 245), -1)
        cv2.putText(frame, "VisionBrain synthetic", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
        self.frame_index += 1
        return CameraRead(True, frame, time.monotonic())

    def warmup(self, frames: int | None = None) -> None:
        count = self.config.warmup_frames if frames is None else frames
        for _ in range(max(0, count)):
            self.read()

    def probe(self) -> dict[str, Any]:
        return {
            "source": "synthetic",
            "backend_name": "VISIONBRAIN_SYNTHETIC",
            "properties": {
                "width": float(self.config.width),
                "height": float(self.config.height),
                "fps": float(self.config.fps),
            },
            "fourcc_text": "BGR",
        }

    def release(self) -> None:
        self._open = False

    def __enter__(self) -> "SyntheticCamera":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.release()
