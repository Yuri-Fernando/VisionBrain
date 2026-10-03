from visionbrain.config import RuntimeConfig


def test_default_config_constructs():
    cfg = RuntimeConfig()
    assert cfg.camera.width > 0
    assert cfg.detector.confidence > 0
