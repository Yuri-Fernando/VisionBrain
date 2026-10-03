from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from visionbrain.camera.opencv_source import OpenCVCamera
from visionbrain.config import CameraConfig


def collect_and_calibrate(
    camera_config: CameraConfig,
    *,
    cols: int = 9,
    rows: int = 6,
    square_size_mm: float = 25.0,
    required_samples: int = 15,
    output_path: str = "outputs/calibration.yaml",
) -> dict:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    camera = OpenCVCamera(camera_config).open()
    camera.warmup()

    print("Calibration: show the checkerboard. Press C to capture a valid pose, Q to finish.")
    print("Move/tilt the board and vary its distance; avoid nearly identical views.")
    try:
        while len(objpoints) < required_samples:
            read = camera.read()
            if not read.ok or read.frame is None:
                continue
            frame = read.frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            vis = frame.copy()
            if found:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(vis, (cols, rows), corners2, found)
            cv2.putText(vis, f"samples {len(objpoints)}/{required_samples} | C capture | Q finish", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("VisionBrain Calibration", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and found:
                objpoints.append(objp.copy())
                imgpoints.append(corners2)
                print(f"Captured sample {len(objpoints)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(objpoints) < 5 or image_size is None:
        raise RuntimeError("Need at least 5 valid checkerboard samples for calibration.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    total_error = 0.0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        total_error += cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
    mean_reprojection_error = total_error / len(objpoints)

    result = {
        "rms": float(rms),
        "mean_reprojection_error": float(mean_reprojection_error),
        "image_size": list(image_size),
        "pattern": {"cols": cols, "rows": rows, "square_size_mm": square_size_mm},
        "samples": len(objpoints),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return result
