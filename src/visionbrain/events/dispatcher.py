from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from visionbrain.config import EventConfig
from visionbrain.events.sinks import DiskEventSink
from visionbrain.models import VisionEvent


class EventDispatcher:
    """Persist events immediately and deliver webhooks outside the frame loop."""

    def __init__(self, config: EventConfig, *, queue_size: int = 256, retries: int = 3):
        self.config = config
        self.disk = DiskEventSink(config)
        self.retries = max(1, retries)
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=max(1, queue_size))
        self._thread: threading.Thread | None = None
        self.spool_path = Path(config.jsonl_path).with_name("webhook_spool.jsonl")
        if config.webhook_url:
            self._thread = threading.Thread(target=self._worker, name="visionbrain-webhook", daemon=True)
            self._thread.start()

    def emit(self, event: VisionEvent, frame: np.ndarray | None = None) -> str | None:
        snapshot = self.disk.emit(event, frame)
        if not self.config.webhook_url:
            return snapshot
        payload = event.to_dict()
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._spool(payload, reason="queue_full")
        return snapshot

    def _worker(self) -> None:
        import httpx

        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                delivered = False
                error = "unknown"
                for attempt in range(self.retries):
                    try:
                        response = httpx.post(self.config.webhook_url, json=payload, timeout=1.5)
                        response.raise_for_status()
                        delivered = True
                        break
                    except Exception as exc:  # network failure must not stop inference
                        error = type(exc).__name__
                        if attempt + 1 < self.retries:
                            time.sleep(0.25 * (2**attempt))
                if not delivered:
                    self._spool(payload, reason=error)
            finally:
                self._queue.task_done()

    def _spool(self, payload: dict, *, reason: str) -> None:
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"delivery_status": "pending", "reason": reason, "event": payload}
        with self.spool_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "EventDispatcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
