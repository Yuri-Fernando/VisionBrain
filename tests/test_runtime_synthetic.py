from visionbrain.config import CameraConfig, DetectorConfig, DisplayConfig, EventConfig, MotionAnomalyConfig, RuntimeConfig
from visionbrain.runtime import VisionRuntime


def test_runtime_runs_without_camera_or_model(tmp_path):
    config = RuntimeConfig(
        camera=CameraConfig(source="synthetic", width=96, height=64, warmup_frames=0),
        detector=DetectorConfig(enabled=False),
        motion_anomaly=MotionAnomalyConfig(enabled=False),
        events=EventConfig(
            save_snapshots=False,
            jsonl_path=str(tmp_path / "events.jsonl"),
            snapshots_dir=str(tmp_path / "snapshots"),
        ),
        display=DisplayConfig(enabled=False),
    )
    result = VisionRuntime(config).run(max_frames=3)
    assert result["frames"] == 3
    assert result["camera"]["backend_name"] == "VISIONBRAIN_SYNTHETIC"
