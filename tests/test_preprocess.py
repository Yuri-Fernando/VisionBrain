import numpy as np

from visionbrain.config import PreprocessConfig
from visionbrain.models import FrameQuality
from visionbrain.preprocess.pipeline import PreprocessPipeline


def quality(*issues):
    return FrameQuality(40, 10, 20, 5, 0, 0, 2, 0.1, 0.3, list(issues))


def test_preprocess_preserves_shape():
    frame = np.full((100, 120, 3), 40, dtype=np.uint8)
    cfg = PreprocessConfig(adaptive=True, clahe=True, denoise="bilateral", auto_gamma=True, sharpen=True)
    out = PreprocessPipeline(cfg).apply(frame, quality("underexposed", "low_contrast", "low_sharpness"))
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_auto_gamma_brightens_dark_frame():
    frame = np.full((40, 40, 3), 40, dtype=np.uint8)
    cfg = PreprocessConfig(adaptive=True, clahe=False, denoise="off", auto_gamma=True, sharpen=False)
    out = PreprocessPipeline(cfg).apply(frame, quality("underexposed"))
    assert float(out.mean()) > float(frame.mean())
