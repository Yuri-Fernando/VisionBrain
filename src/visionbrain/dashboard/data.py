from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_events(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Load a JSONL audit trail while counting malformed records."""

    source = Path(path)
    if not source.exists():
        return [], 0
    events: list[dict[str, Any]] = []
    malformed = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(item, dict):
                events.append(item)
            else:
                malformed += 1
    return events, malformed


def flatten_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    row = {
        "timestamp": event.get("timestamp"),
        "event_type": event.get("event_type", "unknown"),
        "severity": event.get("severity", "info"),
        "frame_index": event.get("frame_index"),
    }
    for key in ("label", "track_id", "zone", "rule", "count", "quality_score", "snapshot"):
        row[key] = payload.get(key)
    return row


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    types = Counter(str(event.get("event_type", "unknown")) for event in events)
    severities = Counter(str(event.get("severity", "info")) for event in events)
    timestamps = [event.get("timestamp") for event in events if event.get("timestamp")]
    parsed: list[datetime] = []
    for value in timestamps:
        try:
            parsed.append(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
        except ValueError:
            continue
    return {
        "total": len(events),
        "types": dict(types),
        "severities": dict(severities),
        "first_timestamp": min(parsed).isoformat() if parsed else None,
        "last_timestamp": max(parsed).isoformat() if parsed else None,
        "snapshots": sum(bool((event.get("payload") or {}).get("snapshot")) for event in events),
    }
