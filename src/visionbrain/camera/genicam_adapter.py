from __future__ import annotations

import time
from typing import Any

from visionbrain.camera.base import CameraRead
from visionbrain.camera.genicam_source import GenICamCamera
from visionbrain.config import CameraConfig


class GenICamFrameSource:
    """Adapts the optional Harvester source to the common frame contract.

    Configure ``camera.source`` as ``genicam:C:/path/vendor.cti``. This keeps
    the existing public configuration backward compatible while making the
    industrial source available to the runtime.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        raw_source = str(config.source)
        self.cti_path = raw_source.partition(":")[2]
        self.camera = GenICamCamera(self.cti_path, device_index=0)

    def open(self) -> "GenICamFrameSource":
        if not self.cti_path:
            raise RuntimeError("Use camera.source='genicam:C:/path/to/vendor.cti'.")
        self.camera.open()
        return self

    def read(self) -> CameraRead:
        frame = self.camera.read()
        return CameraRead(True, frame, time.monotonic())

    def warmup(self, frames: int | None = None) -> None:
        count = self.config.warmup_frames if frames is None else frames
        for _ in range(max(0, count)):
            self.read()

    def probe(self) -> dict[str, Any]:
        return {
            "source": str(self.config.source),
            "backend_name": "GenICam/GenTL",
            "cti_path": self.cti_path,
            "device_index": 0,
        }

    def release(self) -> None:
        self.camera.release()

    def __enter__(self) -> "GenICamFrameSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.release()
