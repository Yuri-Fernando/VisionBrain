from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_int(self) -> tuple[int, int, int, int]:
        return tuple(map(int, (self.x1, self.y1, self.x2, self.y2)))


@dataclass(slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameQuality:
    brightness_mean: float
    contrast_std: float
    sharpness_laplacian: float
    entropy: float
    black_clip_ratio: float
    white_clip_ratio: float
    noise_score: float
    color_cast_score: float
    quality_score: float
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VisionEvent:
    event_type: str
    timestamp: str
    frame_index: int
    severity: str = "info"
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        event_type: str,
        frame_index: int,
        *,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> "VisionEvent":
        return cls(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_index=frame_index,
            severity=severity,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeStats:
    frame_index: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    preprocessing_ms: float = 0.0
    quality_ms: float = 0.0
    detector_count: int = 0
    active_tracks: int = 0
    events_emitted: int = 0
