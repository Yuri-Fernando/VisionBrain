from visionbrain.config import CountRule, EventConfig, ZoneConfig
from visionbrain.events.engine import EventEngine
from visionbrain.models import BoundingBox, Detection


def det(track_id=1, label="person", x=50, y=50):
    return Detection(0, label, 0.9, BoundingBox(x, y, x + 20, y + 20), track_id=track_id)


def test_object_entered_emits_once_for_track():
    engine = EventEngine(EventConfig())
    e1 = engine.evaluate(frame_index=1, frame_shape=(100, 100, 3), detections=[det()])
    e2 = engine.evaluate(frame_index=2, frame_shape=(100, 100, 3), detections=[det()])
    assert any(e.event_type == "object_entered" for e in e1)
    assert not any(e.event_type == "object_entered" for e in e2)


def test_zone_enter():
    cfg = EventConfig(zones=[ZoneConfig(name="z", points=[(0.4, 0.4), (0.9, 0.4), (0.9, 0.9), (0.4, 0.9)])])
    engine = EventEngine(cfg)
    events = engine.evaluate(frame_index=1, frame_shape=(100, 100, 3), detections=[det(x=50, y=50)])
    assert any(e.event_type == "zone_enter" for e in events)


def test_count_rule():
    cfg = EventConfig(count_rules=[CountRule(name="crowd", label="person", threshold=2, comparison=">=")])
    engine = EventEngine(cfg)
    events = engine.evaluate(frame_index=1, frame_shape=(100, 100, 3), detections=[det(1), det(2)])
    assert any(e.event_type == "count_threshold" for e in events)
