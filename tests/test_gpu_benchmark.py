from __future__ import annotations

import pytest

from visionbrain.benchmark import compare_cpu_gpu
from visionbrain.camera.autotune import recommend_detector_profile
from visionbrain.config import load_config
from visionbrain.inference.gpu_info import detect_gpu

_GPU = detect_gpu()


def test_detect_gpu_never_raises_and_reports_consistent_shape():
    info = detect_gpu()
    assert isinstance(info.torch_available, bool)
    assert isinstance(info.cuda_available, bool)
    assert info.device_count >= 0
    if not info.cuda_available:
        assert info.device_name is None
        assert info.device_count == 0


def test_recommend_detector_profile_without_gpu_uses_cpu_and_smaller_resolution():
    profile = recommend_detector_profile(cuda_available=False)
    assert profile.device == "cpu"
    assert profile.half is False
    assert profile.imgsz == 320


def test_recommend_detector_profile_with_gpu_uses_cuda_and_larger_resolution():
    profile = recommend_detector_profile(cuda_available=True, device_name="NVIDIA GeForce RTX 3060 Ti")
    assert profile.device == "cuda:0"
    assert profile.half is True
    assert profile.imgsz == 640


def test_recommend_detector_profile_decision_actually_changes_with_gpu_presence():
    """The autotune decision must differ, not just the rationale text."""
    cpu_profile = recommend_detector_profile(cuda_available=False)
    gpu_profile = recommend_detector_profile(cuda_available=True, device_name="RTX 3060 Ti")
    assert cpu_profile.device != gpu_profile.device
    assert cpu_profile.imgsz < gpu_profile.imgsz
    assert cpu_profile.half != gpu_profile.half


@pytest.mark.skipif(not _GPU.cuda_available, reason="Requires a real CUDA-capable NVIDIA GPU")
def test_compare_cpu_gpu_measures_real_latency_on_available_hardware():
    """Runs the real Ultralytics engine on CPU and CUDA against identical synthetic
    frames and checks the measurement pipeline behaves sanely. Skipped (not mocked)
    on machines without a CUDA GPU, so CI without hardware stays green while this
    exercises the real path whenever a GPU is present (e.g. this RTX 3060 Ti box).
    """
    cfg = load_config("config/default.yaml")
    result = compare_cpu_gpu(cfg, frames=6, warmup_iters=2)

    assert result["gpu"]["cuda_available"] is True
    assert result["cpu"]["frames_measured"] == 6
    assert result["cuda"]["frames_measured"] == 6
    assert result["cpu"]["fps_mean"] > 0
    assert result["cuda"]["fps_mean"] > 0
    assert result["cpu"]["device"] == "cpu"
    assert result["cuda"]["device"] == "cuda:0"


@pytest.mark.skipif(_GPU.cuda_available, reason="Only exercises the no-GPU guard path")
def test_compare_cpu_gpu_raises_clearly_without_cuda():
    cfg = load_config("config/default.yaml")
    with pytest.raises(RuntimeError):
        compare_cpu_gpu(cfg, frames=2, warmup_iters=1)
