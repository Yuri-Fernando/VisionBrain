from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionbrain.config import MotionAnomalyConfig


@dataclass(slots=True)
class MotionAnomalyResult:
    motion_ratio: float
    anomalous: bool
    mask: np.ndarray | None = None


class MotionAnomalyDetector:
    """Training-free scene-change/motion anomaly baseline using MOG2.

    This is deliberately separate from learned industrial anomaly detection (PatchCore, PaDiM,
    EfficientAD, etc.). It is useful as an online baseline and as a trigger for evidence capture.
    """

    def __init__(self, config: MotionAnomalyConfig):
        self.config = config
        self.frame_count = 0
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows,
        )

    def update(self, frame: np.ndarray) -> MotionAnomalyResult:
        if not self.config.enabled:
            return MotionAnomalyResult(0.0, False, None)
        self.frame_count += 1
        mask = self.bg.apply(frame)
        # Ignore shadows (MOG2 encodes shadows around 127) and remove speckle noise.
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        motion_ratio = float(np.mean(binary > 0))
        ready = self.frame_count > self.config.warmup_frames
        anomalous = ready and motion_ratio >= self.config.min_motion_ratio
        return MotionAnomalyResult(motion_ratio, anomalous, binary)
