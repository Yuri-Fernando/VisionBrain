from visionbrain.camera.factory import create_frame_source
from visionbrain.config import CameraConfig, DetectorConfig
from visionbrain.inference.factory import DisabledDetector, create_detector


def test_source_factory_supports_synthetic_frames():
    source = create_frame_source(CameraConfig(source="synthetic", width=96, height=64, warmup_frames=0))
    with source:
        read = source.read()
        probe = source.probe()
    assert read.ok
    assert read.frame is not None
    assert read.frame.shape == (64, 96, 3)
    assert probe["backend_name"] == "VISIONBRAIN_SYNTHETIC"


def test_detector_factory_has_noop_backend_when_disabled():
    detector = create_detector(DetectorConfig(enabled=False))
    assert isinstance(detector, DisabledDetector)
