from __future__ import annotations

import numpy as np

from visionbrain.config import DetectorConfig
from visionbrain.inference.base import DetectorBackend
from visionbrain.inference.yolo import InferenceOutput, YoloEngine


class DisabledDetector:
    """No-op backend for acquisition tests and deterministic demos."""

    def load(self) -> None:
        return None

    def infer(self, frame: np.ndarray) -> InferenceOutput:
        return InferenceOutput(detections=[], latency_ms=0.0)


def create_detector(config: DetectorConfig) -> DetectorBackend:
    if not config.enabled:
        return DisabledDetector()
    return YoloEngine(config)
