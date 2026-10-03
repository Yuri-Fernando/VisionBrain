from __future__ import annotations

import cv2

from visionbrain.camera.opencv_source import OpenCVCamera
from visionbrain.config import CameraConfig


def discover_webcams(max_index: int = 6, backend: str = "auto") -> list[dict]:
    devices: list[dict] = []
    for index in range(max_index):
        cfg = CameraConfig(source=index, backend=backend, width=640, height=480, fps=30, warmup_frames=2)
        camera = OpenCVCamera(cfg)
        try:
            camera.open()
            result = camera.read()
            if result.ok and result.frame is not None:
                probe = camera.probe()
                probe["frame_shape"] = list(result.frame.shape)
                devices.append(probe)
        except Exception:
            pass
        finally:
            camera.release()
    cv2.destroyAllWindows()
    return devices
