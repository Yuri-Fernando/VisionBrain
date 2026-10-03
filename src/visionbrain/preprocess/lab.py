from __future__ import annotations

import cv2
import numpy as np

from visionbrain.camera.opencv_source import OpenCVCamera
from visionbrain.config import CameraConfig


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray


def build_filter_grid(frame: np.ndarray, tile_width: int = 480) -> np.ndarray:
    scale = tile_width / frame.shape[1]
    base = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(base, (5, 5), 0)
    median = cv2.medianBlur(base, 5)
    bilateral = cv2.bilateralFilter(base, 7, 60, 60)

    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    clahe = cv2.cvtColor(cv2.merge([luminance, channel_a, channel_b]), cv2.COLOR_LAB2BGR)

    canny = _to_bgr(cv2.Canny(gray, 80, 160))
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = _to_bgr(cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y)))

    scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    scharr = _to_bgr(cv2.convertScaleAbs(cv2.magnitude(scharr_x, scharr_y)))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = _to_bgr(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))
    laplacian = _to_bgr(cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F)))
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)

    tiles = [
        _label(base, "RAW"),
        _label(gaussian, "GAUSSIAN"),
        _label(median, "MEDIAN"),
        _label(bilateral, "BILATERAL"),
        _label(clahe, "CLAHE (LAB-L)"),
        _label(canny, "CANNY"),
        _label(sobel, "SOBEL magnitude"),
        _label(scharr, "SCHARR magnitude"),
        _label(morph, "OTSU + MORPH CLOSE"),
        _label(laplacian, "LAPLACIAN"),
        _label(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "GRAYSCALE"),
        _label(hsv, "HSV channels as BGR view"),
    ]
    rows = [np.hstack(tiles[index:index + 3]) for index in range(0, len(tiles), 3)]
    return np.vstack(rows)


def run_filter_lab(camera_config: CameraConfig) -> None:
    camera = OpenCVCamera(camera_config).open()
    camera.warmup()
    try:
        while True:
            read = camera.read()
            if not read.ok or read.frame is None:
                continue
            grid = build_filter_grid(read.frame)
            cv2.imshow("VisionBrain Filter Lab | Q/ESC exit", grid)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
