from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from visionbrain.config import DetectorConfig
from visionbrain.models import BoundingBox, Detection


@dataclass(slots=True)
class InferenceOutput:
    detections: list[Detection]
    latency_ms: float
    raw_result: object | None = None


class YoloEngine:
    """Lazy-loaded Ultralytics detector/tracker adapter."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run `pip install -e '.[yolo]'` or install ultralytics."
            ) from exc
        self._model = YOLO(self.config.model)

    def infer(self, frame: np.ndarray) -> InferenceOutput:
        if not self.config.enabled:
            return InferenceOutput(detections=[], latency_ms=0.0)
        self.load()
        kwargs = {
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.imgsz,
            "verbose": False,
        }
        if self.config.device is not None:
            kwargs["device"] = self.config.device
        if self.config.classes is not None:
            kwargs["classes"] = self.config.classes
        if self.config.half:
            kwargs["half"] = True

        start = time.perf_counter()
        if self.config.tracking:
            results = self._model.track(
                frame,
                persist=True,
                tracker=self.config.tracker,
                **kwargs,
            )
        else:
            results = self._model.predict(frame, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        result = results[0] if results else None
        detections: list[Detection] = []
        if result is None or result.boxes is None:
            return InferenceOutput(detections, latency_ms, result)

        names = result.names
        boxes = result.boxes
        ids = boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None else [None] * len(boxes)
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()

        for box, conf, class_id, track_id in zip(xyxy, confs, classes, ids):
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            detections.append(
                Detection(
                    class_id=int(class_id),
                    label=str(label),
                    confidence=float(conf),
                    bbox=BoundingBox(*map(float, box)),
                    track_id=int(track_id) if track_id is not None else None,
                )
            )
        return InferenceOutput(detections, latency_ms, result)
