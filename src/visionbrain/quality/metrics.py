from __future__ import annotations

import math

import cv2
import numpy as np

from visionbrain.models import FrameQuality


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _noise_score(gray: np.ndarray) -> float:
    # High-frequency residual proxy. Not a sensor-noise estimator, but useful online.
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, smooth)
    return float(np.mean(residual))


def _color_cast_score(frame: np.ndarray) -> float:
    means = np.mean(frame.reshape(-1, 3), axis=0)
    denom = max(float(np.mean(means)), 1.0)
    return float((np.max(means) - np.min(means)) / denom)


def analyze_frame_quality(
    frame: np.ndarray,
    *,
    target_brightness: float = 125.0,
    min_contrast: float = 28.0,
    min_sharpness: float = 75.0,
    max_black_clip_ratio: float = 0.08,
    max_white_clip_ratio: float = 0.08,
    max_noise_score: float = 18.0,
) -> FrameQuality:
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy = _entropy(gray)
    black_clip = float(np.mean(gray <= 5))
    white_clip = float(np.mean(gray >= 250))
    noise = _noise_score(gray)
    color_cast = _color_cast_score(frame) if frame.ndim == 3 else 0.0

    brightness_score = max(0.0, 1.0 - abs(brightness - target_brightness) / max(target_brightness, 1.0))
    contrast_score = min(1.0, contrast / max(min_contrast * 1.8, 1.0))
    sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(max(min_sharpness * 5, 1.0)))
    entropy_score = min(1.0, entropy / 7.0)
    clipping_penalty = min(1.0, (black_clip / max(max_black_clip_ratio, 1e-6) + white_clip / max(max_white_clip_ratio, 1e-6)) / 2)
    noise_penalty = min(1.0, noise / max(max_noise_score * 2, 1.0))
    color_penalty = min(1.0, color_cast / 0.8)

    score = (
        0.25 * brightness_score
        + 0.20 * contrast_score
        + 0.30 * sharpness_score
        + 0.15 * entropy_score
        + 0.10 * (1.0 - color_penalty)
    )
    score *= (1.0 - 0.25 * clipping_penalty) * (1.0 - 0.15 * noise_penalty)
    score = float(np.clip(score, 0.0, 1.0))

    issues: list[str] = []
    if brightness < target_brightness * 0.55:
        issues.append("underexposed")
    elif brightness > target_brightness * 1.55:
        issues.append("overexposed")
    if contrast < min_contrast:
        issues.append("low_contrast")
    if sharpness < min_sharpness:
        issues.append("low_sharpness")
    if black_clip > max_black_clip_ratio:
        issues.append("black_clipping")
    if white_clip > max_white_clip_ratio:
        issues.append("white_clipping")
    if noise > max_noise_score:
        issues.append("high_noise")
    if color_cast > 0.35:
        issues.append("color_cast")

    return FrameQuality(
        brightness_mean=brightness,
        contrast_std=contrast,
        sharpness_laplacian=sharpness,
        entropy=entropy,
        black_clip_ratio=black_clip,
        white_clip_ratio=white_clip,
        noise_score=noise,
        color_cast_score=color_cast,
        quality_score=score,
        issues=issues,
    )
