from __future__ import annotations

import statistics
import time

from visionbrain.camera.factory import create_frame_source
from visionbrain.config import RuntimeConfig
from visionbrain.inference.factory import create_detector
from visionbrain.preprocess.pipeline import PreprocessPipeline
from visionbrain.quality.metrics import analyze_frame_quality


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

    def pct(values, percentile):
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int((len(ordered) - 1) * percentile)
        return float(ordered[index])

    return {
        "frames_requested": frames,
        "frames_processed": good,
        "end_to_end_fps": good / elapsed,
        "capture_fps_observed": (1.0 / statistics.mean(capture_intervals)) if capture_intervals else 0.0,
        "inference_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "inference_ms_p50": pct(latencies, 0.50),
        "inference_ms_p95": pct(latencies, 0.95),
        "preprocess_ms_mean": statistics.mean(prep_ms) if prep_ms else 0.0,
        "camera": probe,
    }
