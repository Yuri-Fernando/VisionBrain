from __future__ import annotations

from visionbrain.camera.base import FrameSource
from visionbrain.camera.opencv_source import OpenCVCamera
from visionbrain.camera.synthetic_source import SyntheticCamera
from visionbrain.config import CameraConfig


def create_frame_source(config: CameraConfig) -> FrameSource:
    """Build a source without coupling the runtime to a camera vendor.

    Existing integer, file, and RTSP values continue through OpenCV. The
    reserved values ``synthetic`` and ``genicam:<cti-path>`` select sources
    that do not use OpenCV's VideoCapture boundary.
    """

    source = str(config.source)
    if source == "synthetic":
        return SyntheticCamera(config)
    if source.startswith("genicam:"):
        from visionbrain.camera.genicam_adapter import GenICamFrameSource

        return GenICamFrameSource(config)
    return OpenCVCamera(config)
