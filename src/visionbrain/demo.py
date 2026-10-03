from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from visionbrain.camera.synthetic_source import SyntheticCamera
from visionbrain.config import CameraConfig, DisplayConfig, EventConfig, PreprocessConfig, ZoneConfig
from visionbrain.events.engine import EventEngine
from visionbrain.events.sinks import DiskEventSink
from visionbrain.models import BoundingBox, Detection, RuntimeStats
from visionbrain.overlay import draw_overlay
from visionbrain.preprocess.pipeline import PreprocessPipeline
from visionbrain.quality.metrics import analyze_frame_quality


def run_synthetic_e2e(output_dir: str | Path = "outputs/demo", frames: int = 24) -> dict:
    """Run a deterministic, model-free end-to-end demonstration.

    The flow exercises acquisition, quality analysis, adaptive preprocessing,
    tracked detections, zone/count rules, overlays, JSONL persistence and
    evidence snapshots without a webcam, model download or external service.
    """

    output_dir = Path(output_dir)
    events_path = output_dir / "events.jsonl"
    snapshots_dir = output_dir / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path.unlink(missing_ok=True)

    camera_config = CameraConfig(source="synthetic", width=320, height=180, fps=30, warmup_frames=0)
    event_config = EventConfig(
        global_cooldown_seconds=0.0,
        zones=[ZoneConfig(name="inspection_zone", points=[(0.35, 0.15), (0.75, 0.15), (0.75, 0.85), (0.35, 0.85)])],
        jsonl_path=str(events_path),
        snapshots_dir=str(snapshots_dir),
        save_snapshots=True,
    )
    engine = EventEngine(event_config)
    sink = DiskEventSink(event_config)
    preprocess = PreprocessPipeline(PreprocessConfig(adaptive=True))
    stats = RuntimeStats()
    display = DisplayConfig(enabled=False)
    emitted_types: Counter[str] = Counter()
    qualities: list[float] = []

    with SyntheticCamera(camera_config) as camera:
        for index in range(1, max(1, frames) + 1):
            read = camera.read()
            if not read.ok or read.frame is None:
                continue
            quality = analyze_frame_quality(read.frame)
            processed = preprocess.apply(read.frame, quality)
            x1 = -20 + index * (340 / max(frames, 1))
            detection = Detection(
                class_id=0,
                label="demo_part",
                confidence=0.94,
                bbox=BoundingBox(x1, 65, x1 + 42, 115),
                track_id=1,
            )
            events = engine.evaluate(
                frame_index=index,
                frame_shape=processed.shape,
                detections=[detection],
                quality=quality,
            )
            stats.frame_index = index
            stats.detector_count = 1
            stats.active_tracks = 1
            stats.events_emitted += len(events)
            visual = draw_overlay(
                processed,
                [detection],
                stats,
                display,
                quality=quality,
                zones=event_config.zones,
                events=events,
            )
            for event in events:
                sink.emit(event, visual)
                emitted_types[event.event_type] += 1
            qualities.append(quality.quality_score)

    summary = {
        "frames_processed": stats.frame_index,
        "events_emitted": stats.events_emitted,
        "event_types": dict(emitted_types),
        "mean_quality_score": sum(qualities) / len(qualities) if qualities else 0.0,
        "events_path": str(events_path),
        "snapshots_dir": str(snapshots_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
