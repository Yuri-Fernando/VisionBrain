import numpy as np

from visionbrain.quality.metrics import analyze_frame_quality


def test_dark_frame_is_underexposed():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    q = analyze_frame_quality(frame)
    assert "underexposed" in q.issues
    assert q.brightness_mean == 0.0


def test_textured_frame_has_higher_sharpness_than_flat_frame():
    flat = np.full((120, 160, 3), 120, dtype=np.uint8)
    checker = np.indices((120, 160)).sum(axis=0) % 2 * 255
    checker = np.repeat(checker[:, :, None].astype(np.uint8), 3, axis=2)
    assert analyze_frame_quality(checker).sharpness_laplacian > analyze_frame_quality(flat).sharpness_laplacian
