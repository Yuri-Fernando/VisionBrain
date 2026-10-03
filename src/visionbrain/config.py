from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    source: int | str = 0
    backend: Literal["auto", "dshow", "msmf", "v4l2", "gstreamer", "ffmpeg"] = "auto"
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1
    autofocus: bool | None = True
    auto_exposure: bool | None = True
    auto_white_balance: bool | None = True
    exposure: float | None = None
    gain: float | None = None
    focus: float | None = None
    warmup_frames: int = 30


class QualityConfig(BaseModel):
    enabled: bool = True
    every_n_frames: int = 10
    target_brightness: float = 125.0
    min_contrast: float = 28.0
    min_sharpness: float = 75.0
    max_black_clip_ratio: float = 0.08
    max_white_clip_ratio: float = 0.08
    max_noise_score: float = 18.0


class PreprocessConfig(BaseModel):
    enabled: bool = True
    adaptive: bool = True
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    denoise: Literal["off", "gaussian", "median", "bilateral"] = "bilateral"
    denoise_strength: int = 5
    auto_gamma: bool = True
    sharpen: bool = True
    sharpen_amount: float = 0.75


class DetectorConfig(BaseModel):
    enabled: bool = True
    model: str = "yolo26n.pt"
    confidence: float = 0.35
    iou: float = 0.55
    imgsz: int = 640
    device: str | int | None = None
    half: bool = False
    classes: list[int] | None = None
    tracking: bool = True
    tracker: str = "bytetrack.yaml"


class MotionAnomalyConfig(BaseModel):
    enabled: bool = True
    history: int = 500
    var_threshold: float = 32.0
    detect_shadows: bool = True
    min_motion_ratio: float = 0.08
    warmup_frames: int = 60


class ZoneConfig(BaseModel):
    name: str
    points: list[tuple[float, float]] = Field(min_length=3)
    classes: list[str] | None = None


class CountRule(BaseModel):
    name: str
    label: str
    threshold: int = 1
    comparison: Literal[">=", ">", "==", "<=", "<"] = ">="
    severity: Literal["info", "warning", "critical"] = "warning"
    cooldown_seconds: float = 5.0


class EventConfig(BaseModel):
    emit_object_entered: bool = True
    emit_object_left: bool = False
    track_ttl_frames: int = 45
    global_cooldown_seconds: float = 0.5
    zones: list[ZoneConfig] = Field(default_factory=list)
    count_rules: list[CountRule] = Field(default_factory=list)
    save_snapshots: bool = True
    jsonl_path: str = "outputs/events/events.jsonl"
    snapshots_dir: str = "outputs/snapshots"
    webhook_url: str | None = None


class DisplayConfig(BaseModel):
    enabled: bool = True
    window_name: str = "VisionBrain"
    show_quality: bool = True
    show_fps: bool = True
    show_track_ids: bool = True
    draw_zones: bool = True
    resize_max_width: int | None = 1600


class RuntimeConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    quality: QualityConfig = QualityConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    detector: DetectorConfig = DetectorConfig()
    motion_anomaly: MotionAnomalyConfig = MotionAnomalyConfig()
    events: EventConfig = EventConfig()
    display: DisplayConfig = DisplayConfig()


def load_config(path: str | Path) -> RuntimeConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RuntimeConfig.model_validate(data)


def save_config(config: RuntimeConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
