from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionbrain.config import EventConfig
from visionbrain.image_io import write_image
from visionbrain.models import VisionEvent


class DiskEventSink:
    def __init__(self, config: EventConfig):
        self.config = config
        self.jsonl_path = Path(config.jsonl_path)
        self.snapshots_dir = Path(config.snapshots_dir)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, event: VisionEvent, frame: np.ndarray | None = None) -> str | None:
        snapshot_path: str | None = None
        if self.config.save_snapshots and frame is not None:
            safe_ts = event.timestamp.replace(":", "-").replace("+", "_")
            path = self.snapshots_dir / f"{event.event_type}_{event.frame_index}_{safe_ts}.jpg"
            write_image(path, frame)
            snapshot_path = str(path)
            event.payload.setdefault("snapshot", snapshot_path)

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return snapshot_path


class WebhookEventSink:
    """Backward-compatible synchronous sink; prefer EventDispatcher in runtimes."""

    def __init__(self, url: str | None):
        self.url = url

    def emit(self, event: VisionEvent) -> None:
        if not self.url:
            return
        try:
            import httpx

            httpx.post(self.url, json=event.to_dict(), timeout=1.5)
        except Exception:
            return
