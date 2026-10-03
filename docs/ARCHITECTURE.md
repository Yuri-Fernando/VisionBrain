# Architecture

## Design principles

1. **Sensor before model.** Bad exposure, focus, optics or illumination cannot be repaired reliably by throwing a larger network at the problem.
2. **Deterministic real-time path.** Webcam/video inference, tracking and rules remain independent from slower VLM or research modules.
3. **Adapters around vendors.** Acquisition and inference implementations sit behind narrow boundaries so a deployment can replace OpenCV, Ultralytics or a camera SDK.
4. **Measurements before automation.** Auto-tuning records requested and observed camera behavior rather than assuming a driver honored a setting.
5. **Event-oriented outputs.** Downstream systems receive semantic events and evidence, not an uncontrolled flood of frames.
6. **Research/prod separation.** Experimental modules can be evaluated without destabilizing the serving loop.

## Runtime stages

### Acquisition

Inputs may be an integer webcam index, video path, RTSP URL, GStreamer source or an industrial camera adapter. The OpenCV source records backend, effective frame size, FPS, FOURCC and available property values.

### Quality gate

The quality gate generates frame-level observability. Metrics should be tracked over time because a drift in focus, clipping or noise can be a camera/lighting failure even when model confidence has not collapsed yet.

### Preprocess

Adaptive preprocessing is conservative. It only applies corrections when diagnostics justify them. A production recipe should be validated per station/camera because enhancement can also destroy defect features.

### Inference

The initial backend is YOLO because it supports a broad set of tasks and tracking. The boundary is intentionally narrow so future adapters can use ONNX Runtime, OpenVINO, TensorRT, TorchVision, RT-DETR or proprietary detectors.

### Temporal analytics

Tracking is the bridge between per-frame recognition and real video analytics. Stable object identity enables entry/exit, zone occupancy, dwell time, counting and process-state reasoning.

### Events

Rules turn detections into business/experimental semantics. Events are persisted locally and can be forwarded to n8n/SaaS systems. Production should use a queue/outbox for guaranteed delivery.

## SaaS boundary

The CV engine should not own billing, tenants, RBAC or customer-facing dashboards. It should expose camera recipes, model versions, health, metrics and events to the SaaS. See `SAAS_INTEGRATION.md`.
