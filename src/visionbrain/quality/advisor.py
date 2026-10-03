from __future__ import annotations

from visionbrain.models import FrameQuality


def recommendations(q: FrameQuality) -> list[str]:
    recs: list[str] = []
    issues = set(q.issues)
    if "underexposed" in issues:
        recs.append("Increase scene illumination or exposure time; prefer more light before adding sensor gain when noise matters.")
    if "overexposed" in issues or "white_clipping" in issues:
        recs.append("Reduce exposure/illumination or diffuse specular highlights; clipped pixels cannot be recovered in preprocessing.")
    if "black_clipping" in issues:
        recs.append("Raise useful illumination in dark regions or change light angle/fill; verify that auto exposure is not protecting bright highlights.")
    if "low_sharpness" in issues:
        recs.append("Verify focus/autofocus, lens cleanliness, working distance and motion blur; for motion, prefer shorter exposure plus more light.")
    if "low_contrast" in issues:
        recs.append("Improve object/background separation or illumination geometry; CLAHE can help digitally but should not replace better optics/lighting.")
    if "high_noise" in issues:
        recs.append("Reduce gain and increase photon budget/illumination when possible; validate denoising because it can erase small defects.")
    if "color_cast" in issues:
        recs.append("Stabilize illuminant spectrum and white balance; consider locking white balance after warm-up in controlled stations.")
    if not recs:
        recs.append("Capture quality is within the current online heuristic thresholds; validate task-specific quality against labeled performance.")
    return recs
