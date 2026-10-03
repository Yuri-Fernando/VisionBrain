from __future__ import annotations

import operator
import time
from collections import Counter

import cv2
import numpy as np

from visionbrain.config import EventConfig, ZoneConfig
from visionbrain.models import Detection, FrameQuality, VisionEvent


_COMPARATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "<=": operator.le,
    "<": operator.lt,
}


class EventEngine:
    def __init__(self, config: EventConfig):
        self.config = config
        self._last_seen_track: dict[int, int] = {}
        self._known_tracks: set[int] = set()
        self._zone_membership: dict[tuple[str, int], bool] = {}
        self._last_emitted: dict[str, float] = {}

    def _cooldown_ok(self, key: str, cooldown: float) -> bool:
        now = time.monotonic()
        last = self._last_emitted.get(key, -1e18)
        if now - last >= cooldown:
            self._last_emitted[key] = now
            return True
        return False

    @staticmethod
    def _inside_zone(det: Detection, zone: ZoneConfig, frame_shape: tuple[int, ...]) -> bool:
        h, w = frame_shape[:2]
        points = np.array([(int(x * w), int(y * h)) for x, y in zone.points], dtype=np.int32)
        return cv2.pointPolygonTest(points, (float(det.bbox.cx), float(det.bbox.cy)), False) >= 0

    def evaluate(
        self,
        *,
        frame_index: int,
        frame_shape: tuple[int, ...],
        detections: list[Detection],
        quality: FrameQuality | None = None,
        motion_anomalous: bool = False,
        motion_ratio: float = 0.0,
    ) -> list[VisionEvent]:
        events: list[VisionEvent] = []
        current_track_ids = {d.track_id for d in detections if d.track_id is not None}

        for det in detections:
            if det.track_id is not None:
                tid = det.track_id
                self._last_seen_track[tid] = frame_index
                if self.config.emit_object_entered and tid not in self._known_tracks:
                    key = f"object_entered:{tid}"
                    if self._cooldown_ok(key, self.config.global_cooldown_seconds):
                        events.append(VisionEvent.now(
                            "object_entered",
                            frame_index,
                            payload={
                                "track_id": tid,
                                "label": det.label,
                                "confidence": det.confidence,
                                "bbox": det.bbox.as_int(),
                            },
                        ))
                    self._known_tracks.add(tid)

        expired = [
            tid for tid, last in self._last_seen_track.items()
            if tid not in current_track_ids and frame_index - last > self.config.track_ttl_frames
        ]
        for tid in expired:
            if self.config.emit_object_left:
                events.append(VisionEvent.now("object_left", frame_index, payload={"track_id": tid}))
            self._last_seen_track.pop(tid, None)
            self._known_tracks.discard(tid)
            stale_zone_keys = [key for key in self._zone_membership if key[1] == tid]
            for key in stale_zone_keys:
                self._zone_membership.pop(key, None)

        counts = Counter(d.label for d in detections)
        for rule in self.config.count_rules:
            value = counts.get(rule.label, 0)
            if _COMPARATORS[rule.comparison](value, rule.threshold):
                key = f"count_rule:{rule.name}"
                if self._cooldown_ok(key, rule.cooldown_seconds):
                    events.append(VisionEvent.now(
                        "count_threshold",
                        frame_index,
                        severity=rule.severity,
                        payload={"rule": rule.name, "label": rule.label, "count": value, "threshold": rule.threshold},
                    ))

        for zone in self.config.zones:
            for det in detections:
                if det.track_id is None:
                    continue
                if zone.classes and det.label not in zone.classes:
                    continue
                key_tuple = (zone.name, det.track_id)
                inside = self._inside_zone(det, zone, frame_shape)
                was_inside = self._zone_membership.get(key_tuple, False)
                if inside and not was_inside:
                    events.append(VisionEvent.now(
                        "zone_enter",
                        frame_index,
                        payload={"zone": zone.name, "track_id": det.track_id, "label": det.label},
                    ))
                elif was_inside and not inside:
                    events.append(VisionEvent.now(
                        "zone_exit",
                        frame_index,
                        payload={"zone": zone.name, "track_id": det.track_id, "label": det.label},
                    ))
                self._zone_membership[key_tuple] = inside

        if motion_anomalous and self._cooldown_ok("motion_anomaly", 2.0):
            events.append(VisionEvent.now(
                "motion_anomaly",
                frame_index,
                severity="warning",
                payload={"motion_ratio": motion_ratio},
            ))

        if quality is not None and quality.issues and quality.quality_score < 0.45:
            if self._cooldown_ok("quality_degraded", 5.0):
                events.append(VisionEvent.now(
                    "capture_quality_degraded",
                    frame_index,
                    severity="warning",
                    payload={"quality_score": quality.quality_score, "issues": quality.issues},
                ))

        return events
