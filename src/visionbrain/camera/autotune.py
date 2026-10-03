from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from visionbrain.camera.opencv_source import OpenCVCamera
from visionbrain.quality.metrics import analyze_frame_quality


@dataclass(slots=True)
class CandidateResult:
    requested_width: int
    requested_height: int
    actual_width: int
    actual_height: int
    actual_fps: float
    measured_capture_fps: float
    quality_score: float
    brightness: float
    sharpness: float
    contrast: float

    @property
    def score(self) -> float:
        resolution_factor = min((self.actual_width * self.actual_height) / (1280 * 720), 1.5)
        fps_factor = min(self.measured_capture_fps / 30.0, 1.0)
        return self.quality_score * (0.65 + 0.20 * resolution_factor + 0.15 * fps_factor)


DEFAULT_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]


def stabilize_auto_controls(camera: OpenCVCamera, seconds: float = 1.0) -> None:
    camera.set_property("autofocus", 1.0)
    camera.set_property("auto_wb", 1.0)
    camera.set_property("auto_exposure", 0.75)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        camera.read()


def benchmark_camera_candidate(
    camera: OpenCVCamera,
    width: int,
    height: int,
    samples: int = 20,
) -> CandidateResult | None:
    camera.set_property("width", width)
    camera.set_property("height", height)
    for _ in range(8):
        camera.read()

    qualities = []
    start = time.perf_counter()
    good = 0
    for _ in range(samples):
        read = camera.read()
        if read.ok and read.frame is not None:
            good += 1
            qualities.append(analyze_frame_quality(read.frame))
    elapsed = max(time.perf_counter() - start, 1e-6)
    if not qualities:
        return None

    return CandidateResult(
        requested_width=width,
        requested_height=height,
        actual_width=int(camera.get_property("width")),
        actual_height=int(camera.get_property("height")),
        actual_fps=camera.get_property("fps"),
        measured_capture_fps=good / elapsed,
        quality_score=float(np.mean([q.quality_score for q in qualities])),
        brightness=float(np.mean([q.brightness_mean for q in qualities])),
        sharpness=float(np.mean([q.sharpness_laplacian for q in qualities])),
        contrast=float(np.mean([q.contrast_std for q in qualities])),
    )


def auto_tune_resolution(
    camera: OpenCVCamera,
    candidates: list[tuple[int, int]] | None = None,
    samples: int = 20,
) -> tuple[CandidateResult, list[CandidateResult]]:
    stabilize_auto_controls(camera)
    results: list[CandidateResult] = []
    for width, height in candidates or DEFAULT_RESOLUTIONS:
        result = benchmark_camera_candidate(camera, width, height, samples=samples)
        if result is not None:
            results.append(result)
    if not results:
        raise RuntimeError("Could not benchmark any camera resolution.")
    best = max(results, key=lambda x: x.score)
    camera.set_property("width", best.actual_width)
    camera.set_property("height", best.actual_height)
    return best, results
