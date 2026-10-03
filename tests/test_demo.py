from visionbrain.dashboard.data import load_events
from visionbrain.demo import run_synthetic_e2e


def test_synthetic_demo_generates_events_and_evidence(tmp_path):
    result = run_synthetic_e2e(tmp_path / "demo", frames=24)
    events, malformed = load_events(result["events_path"])
    event_types = {event["event_type"] for event in events}
    assert malformed == 0
    assert result["frames_processed"] == 24
    assert {"object_entered", "zone_enter", "zone_exit"} <= event_types
    assert all((event.get("payload") or {}).get("snapshot") for event in events)
