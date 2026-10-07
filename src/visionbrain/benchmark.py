from __future__ import annotations

import statistics
import time
from dataclasses import asdict

from visionbrain.camera.factory import create_frame_source
from visionbrain.camera.synthetic_source import SyntheticCamera
from visionbrain.config import RuntimeConfig
from visionbrain.inference.factory import create_detector
from visionbrain.inference.gpu_info import detect_gpu
from visionbrain.inference.yolo import YoloEngine
from visionbrain.preprocess.pipeline import PreprocessPipeline
from visionbrain.quality.metrics import analyze_frame_quality


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * percentile)
    return float(ordered[index])


def benchmark(config: RuntimeConfig, frames: int = 200) -> dict:
    camera = create_frame_source(config.camera).open()
    camera.warmup()
    detector = create_detector(config.detector)
    prep = PreprocessPipeline(config.preprocess)
    latencies = []
    prep_ms = []
    capture_intervals = []
    quality = None
    last_capture = None
    good = 0
    start = time.perf_counter()
    probe = camera.probe()
    try:
        for i in range(frames):
            read = camera.read()
            if not read.ok or read.frame is None:
                continue
            good += 1
            if last_capture is not None:
                capture_intervals.append(read.timestamp_monotonic - last_capture)
            last_capture = read.timestamp_monotonic
            if quality is None or i % max(config.quality.every_n_frames, 1) == 0:
                quality = analyze_frame_quality(read.frame)
            started = time.perf_counter()
            frame = prep.apply(read.frame, quality)
            prep_ms.append((time.perf_counter() - started) * 1000)
            out = detector.infer(frame)
            latencies.append(out.latency_ms)
    finally:
        camera.release()
    elapsed = max(time.perf_counter() - start, 1e-6)

    return {
        "frames_requested": frames,
        "frames_processed": good,
        "end_to_end_fps": good / elapsed,
        "capture_fps_observed": (1.0 / statistics.mean(capture_intervals)) if capture_intervals else 0.0,
        "inference_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "inference_ms_p50": _percentile(latencies, 0.50),
        "inference_ms_p95": _percentile(latencies, 0.95),
        "preprocess_ms_mean": statistics.mean(prep_ms) if prep_ms else 0.0,
        "camera": probe,
    }


def _measure_device(
    detector_config,
    camera_config,
    frames: int,
    warmup_iters: int,
) -> dict:
    """Run the real YOLO engine against deterministic synthetic frames on one device.

    Uses :class:`SyntheticCamera` (not a stub detector) so the only variable between
    the CPU and CUDA runs is the inference device itself: identical frames, identical
    model weights, identical pre/post-processing. ``warmup_iters`` inferences are run
    and discarded first so CUDA context/kernel warm-up does not skew the measured
    latency of the real comparison.
    """
    engine = YoloEngine(detector_config)
    engine.load()

    camera = SyntheticCamera(camera_config).open()
    camera.warmup()
    try:
        for _ in range(max(warmup_iters, 0)):
            read = camera.read()
            engine.infer(read.frame)

        latencies: list[float] = []
        for _ in range(frames):
            read = camera.read()
            out = engine.infer(read.frame)
            latencies.append(out.latency_ms)
    finally:
        camera.release()

    mean_ms = statistics.mean(latencies) if latencies else 0.0
    return {
        "device": str(detector_config.device),
        "half": detector_config.half,
        "imgsz": detector_config.imgsz,
        "frames_measured": len(latencies),
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "fps_mean": (1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def compare_cpu_gpu(
    config: RuntimeConfig,
    frames: int = 60,
    warmup_iters: int = 5,
    model: str | None = None,
) -> dict:
    """Measure real CPU vs real CUDA inference latency/FPS with the same model.

    This is a real hardware measurement, not an estimate: it loads the configured
    Ultralytics model twice (once with ``device="cpu"``, once with ``device="cuda:0"``
    and half precision) and runs both against the same deterministic synthetic
    frames, timing actual ``model.predict``/``model.track`` calls.

    Raises ``RuntimeError`` if no CUDA GPU is available, so a caller never reports a
    "GPU benchmark" that silently fell back to CPU-only numbers. Use
    :func:`visionbrain.inference.gpu_info.detect_gpu` first if you want to check
    availability without raising.
    """
    gpu = detect_gpu()
    if not gpu.cuda_available:
        raise RuntimeError(
            "CUDA indisponivel nesta maquina (torch.cuda.is_available() = False). "
            "compare_cpu_gpu() exige uma GPU NVIDIA real; use benchmark() para o "
            "caminho sintetico/CPU."
        )

    detector_config = config.detector.model_copy(deep=True)
    if model is not None:
        detector_config.model = model
    detector_config.enabled = True
    detector_config.tracking = False  # isola o custo de inferencia, sem estado de tracker

    camera_config = config.camera.model_copy(deep=True)

    cpu_config = detector_config.model_copy(deep=True)
    cpu_config.device = "cpu"
    cpu_config.half = False

    cuda_config = detector_config.model_copy(deep=True)
    cuda_config.device = "cuda:0"
    cuda_config.half = True

    cpu_result = _measure_device(cpu_config, camera_config, frames, warmup_iters)
    cuda_result = _measure_device(cuda_config, camera_config, frames, warmup_iters)

    speedup = (
        cpu_result["latency_ms_mean"] / cuda_result["latency_ms_mean"]
        if cuda_result["latency_ms_mean"] > 0
        else 0.0
    )

    return {
        "model": detector_config.model,
        "frames_per_device": frames,
        "warmup_iters": warmup_iters,
        "gpu": asdict(gpu),
        "cpu": cpu_result,
        "cuda": cuda_result,
        "speedup_cuda_over_cpu": speedup,
    }
