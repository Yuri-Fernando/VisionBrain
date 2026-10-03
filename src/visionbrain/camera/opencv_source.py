from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from visionbrain.config import CameraConfig


_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
    "ffmpeg": cv2.CAP_FFMPEG,
}


@dataclass(slots=True)
class CameraRead:
    ok: bool
    frame: np.ndarray | None
    timestamp_monotonic: float


class OpenCVCamera:
    """OpenCV camera/video source with capability probing and verified property writes."""

    PROPERTY_MAP: dict[str, int] = {
        "width": cv2.CAP_PROP_FRAME_WIDTH,
        "height": cv2.CAP_PROP_FRAME_HEIGHT,
        "fps": cv2.CAP_PROP_FPS,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "focus": cv2.CAP_PROP_FOCUS,
        "autofocus": cv2.CAP_PROP_AUTOFOCUS,
        "auto_wb": cv2.CAP_PROP_AUTO_WB,
        "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
        "sharpness": cv2.CAP_PROP_SHARPNESS,
        "gamma": cv2.CAP_PROP_GAMMA,
        "buffer_size": cv2.CAP_PROP_BUFFERSIZE,
        "fourcc": cv2.CAP_PROP_FOURCC,
        "backend": cv2.CAP_PROP_BACKEND,
    }

    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture: cv2.VideoCapture | None = None

    @staticmethod
    def resolve_backend(name: str) -> int:
        if name != "auto":
            return _BACKENDS[name]
        system = platform.system().lower()
        if system == "windows":
            return cv2.CAP_DSHOW
        if system == "linux":
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    def open(self) -> "OpenCVCamera":
        backend = self.resolve_backend(self.config.backend)
        source = self.config.source
        if backend == cv2.CAP_ANY:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened() and self.config.backend == "auto":
            cap.release()
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}.")
        self.capture = cap
        self._apply_requested_properties()
        return self

    def _apply_requested_properties(self) -> None:
        self.set_property("width", self.config.width)
        self.set_property("height", self.config.height)
        self.set_property("fps", self.config.fps)
        self.set_property("buffer_size", self.config.buffer_size)

        if self.config.autofocus is not None:
            self.set_property("autofocus", float(self.config.autofocus))
        if self.config.auto_white_balance is not None:
            self.set_property("auto_wb", float(self.config.auto_white_balance))
        if self.config.auto_exposure is not None:
            # Backend semantics differ. The verified actual value is recorded by probe().
            self.set_property("auto_exposure", 0.75 if self.config.auto_exposure else 0.25)
        if self.config.exposure is not None:
            self.set_property("exposure", self.config.exposure)
        if self.config.gain is not None:
            self.set_property("gain", self.config.gain)
        if self.config.focus is not None:
            self.set_property("focus", self.config.focus)

    def read(self) -> CameraRead:
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        ok, frame = self.capture.read()
        return CameraRead(ok=bool(ok), frame=frame if ok else None, timestamp_monotonic=time.monotonic())

    def warmup(self, frames: int | None = None) -> None:
        frames = self.config.warmup_frames if frames is None else frames
        for _ in range(max(0, frames)):
            result = self.read()
            if not result.ok:
                break

    def get_property(self, name: str) -> float:
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        return float(self.capture.get(self.PROPERTY_MAP[name]))

    def set_property(self, name: str, value: float) -> bool:
        if self.capture is None:
            return False
        return bool(self.capture.set(self.PROPERTY_MAP[name], float(value)))

    def set_verified(self, name: str, value: float, tolerance: float = 0.05) -> dict[str, Any]:
        before = self.get_property(name)
        accepted = self.set_property(name, value)
        time.sleep(0.03)
        after = self.get_property(name)
        scale = max(abs(float(value)), 1.0)
        verified = abs(after - float(value)) <= tolerance * scale
        return {
            "property": name,
            "requested": float(value),
            "before": before,
            "after": after,
            "driver_accepted": accepted,
            "verified": verified,
        }

    def probe(self) -> dict[str, Any]:
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        props = {name: self.get_property(name) for name in self.PROPERTY_MAP}
        fourcc = int(props["fourcc"])
        chars = [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)] if fourcc > 0 else []
        return {
            "source": self.config.source,
            "backend_name": self.capture.getBackendName() if hasattr(self.capture, "getBackendName") else "unknown",
            "properties": props,
            "fourcc_text": "".join(chars).strip("\x00"),
        }

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "OpenCVCamera":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.release()
