# Quick Start — Windows Webcam

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-yolo.txt
pip install -e . --no-build-isolation
```

Then run:

```powershell
visionbrain devices
visionbrain probe --source 0
visionbrain diagnose --source 0
visionbrain autotune --source 0
visionbrain filter-lab --source 0
visionbrain run --config config/fast_webcam.yaml --source 0
```

For the full configuration:

```powershell
visionbrain run --config config/default.yaml --source 0
```

Benchmark 300 frames:

```powershell
visionbrain benchmark --config config/fast_webcam.yaml --source 0 --frames 300
```

Calibrate the camera:

```powershell
visionbrain calibrate --source 0 --cols 9 --rows 6 --square-mm 25 --samples 15
```

If `camera 0` is not the desired webcam, use the index shown by `visionbrain devices`.
