from __future__ import annotations

import cv2
import numpy as np

from visionbrain.config import PreprocessConfig
from visionbrain.models import FrameQuality


def _gamma_from_brightness(mean: float, target: float = 125.0) -> float:
    mean_norm = np.clip(mean / 255.0, 1e-3, 0.999)
    target_norm = np.clip(target / 255.0, 1e-3, 0.999)
    gamma = np.log(target_norm) / np.log(mean_norm)
    return float(np.clip(gamma, 0.55, 1.8))


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_clahe_bgr(frame: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    luminance = clahe.apply(luminance)
    return cv2.cvtColor(cv2.merge([luminance, channel_a, channel_b]), cv2.COLOR_LAB2BGR)


def denoise(frame: np.ndarray, mode: str, strength: int) -> np.ndarray:
    kernel_size = max(3, strength | 1)
    if mode == "gaussian":
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    if mode == "median":
        return cv2.medianBlur(frame, kernel_size)
    if mode == "bilateral":
        return cv2.bilateralFilter(frame, d=kernel_size, sigmaColor=50, sigmaSpace=50)
    return frame


def unsharp_mask(frame: np.ndarray, amount: float = 0.75) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


class PreprocessPipeline:
    def __init__(self, config: PreprocessConfig):
        self.config = config

    def apply(self, frame: np.ndarray, quality: FrameQuality | None = None) -> np.ndarray:
        if not self.config.enabled:
            return frame
        output = frame
        issues = set(quality.issues if quality else [])

        if self.config.auto_gamma and quality is not None and ({"underexposed", "overexposed"} & issues):
            output = apply_gamma(output, _gamma_from_brightness(quality.brightness_mean))
        if self.config.clahe and (not self.config.adaptive or "low_contrast" in issues):
            output = apply_clahe_bgr(output, self.config.clahe_clip_limit, self.config.clahe_grid_size)
        should_denoise = self.config.denoise != "off" and (
            not self.config.adaptive or quality is None or "high_noise" in issues
        )
        if should_denoise:
            output = denoise(output, self.config.denoise, self.config.denoise_strength)
        if self.config.sharpen and (not self.config.adaptive or "low_sharpness" in issues):
            output = unsharp_mask(output, self.config.sharpen_amount)
        return output
