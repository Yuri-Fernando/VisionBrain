import json

from visionbrain.dashboard.data import flatten_event, load_events, summarize_events


def test_dashboard_loader_skips_malformed_records(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"event_type": "zone_enter", "timestamp": "2026-10-03T12:00:00+00:00", "payload": {"zone": "z"}})
        + "\nnot-json\n",
        encoding="utf-8",
    )
    events, malformed = load_events(path)
    assert len(events) == 1
    assert malformed == 1
    assert summarize_events(events)["types"] == {"zone_enter": 1}
    assert flatten_event(events[0])["zone"] == "z"
