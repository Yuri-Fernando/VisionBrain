from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from visionbrain.config import DisplayConfig, ZoneConfig
from visionbrain.models import Detection, FrameQuality, RuntimeStats, VisionEvent


def _put(frame: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.5) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    stats: RuntimeStats,
    config: DisplayConfig,
    *,
    quality: FrameQuality | None = None,
    zones: list[ZoneConfig] | None = None,
    motion_ratio: float = 0.0,
    events: list[VisionEvent] | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if config.draw_zones:
        for zone in zones or []:
            pts = np.array([(int(x * w), int(y * h)) for x, y in zone.points], dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
            x, y = pts[0]
            _put(out, f"ZONE {zone.name}", (int(x), max(20, int(y) - 6)), 0.48)

    for det in detections:
        x1, y1, x2, y2 = det.bbox.as_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        tid = f" id={det.track_id}" if config.show_track_ids and det.track_id is not None else ""
        _put(out, f"{det.label} {det.confidence:.2f}{tid}", (x1, max(18, y1 - 6)), 0.48)

    y = 22
    if config.show_fps:
        _put(out, f"FPS {stats.fps:.1f} | infer {stats.inference_ms:.1f} ms | objects {len(detections)}", (10, y), 0.52)
        y += 22
    if config.show_quality and quality is not None:
        issues = ",".join(quality.issues) if quality.issues else "ok"
        _put(out, f"Quality {quality.quality_score:.2f} | B {quality.brightness_mean:.0f} C {quality.contrast_std:.1f} S {quality.sharpness_laplacian:.0f} | {issues}", (10, y), 0.48)
        y += 22
    _put(out, f"Motion {motion_ratio:.3f}", (10, y), 0.48)
    y += 22

    counts = Counter(d.label for d in detections)
    if counts:
        _put(out, "Counts: " + " | ".join(f"{k}:{v}" for k, v in counts.most_common(5)), (10, y), 0.46)
        y += 22
    if events:
        _put(out, "EVENT: " + ", ".join(e.event_type for e in events[-3:]), (10, y), 0.48)

    return out
