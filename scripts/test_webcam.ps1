$ErrorActionPreference = "Stop"
python -m pip install -e ".[yolo,dev]"
visionbrain devices
visionbrain probe --source 0
visionbrain autotune --source 0 --samples 15
visionbrain run --config config/fast_webcam.yaml --source 0
