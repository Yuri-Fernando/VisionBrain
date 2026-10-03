from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_image(path: str | Path, frame: np.ndarray) -> Path:
    """Write an OpenCV image through pathlib for Unicode-safe Windows paths."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extension = destination.suffix or ".jpg"
    encoded, buffer = cv2.imencode(extension, frame)
    if not encoded:
        raise RuntimeError(f"OpenCV could not encode image as {extension}: {destination}")
    destination.write_bytes(buffer.tobytes())
    return destination
