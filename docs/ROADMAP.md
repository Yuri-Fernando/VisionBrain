# VisionBrain Roadmap

## Phase 1 — Webcam R&D core — implemented

- source discovery/probe;
- capture auto-tune;
- quality metrics;
- adaptive preprocessing;
- YOLO detection/tracking;
- zones/count rules;
- event evidence;
- motion anomaly baseline;
- filter lab;
- benchmark;
- intrinsic calibration;
- snapshot VQA adapter.

## Phase 2 — Industrial inspection

- recipe-specific fixed exposure/gain/WB;
- GenICam feature-node configuration;
- hardware trigger and strobe coordination;
- ROI/line-scan aware acquisition where required;
- segmentation/OBB adapters;
- OCR/barcode/data-matrix branch;
- learned anomaly detection using Anomalib;
- golden-sample/reference-image comparison;
- measurement with calibrated pixels-to-mm / planar homography;
- defect taxonomy and severity model;
- false-positive review queue.

## Phase 3 — Video analytics

- line crossing;
- dwell time;
- occupancy;
- trajectory histories;
- speed estimation after geometric calibration;
- action/behavior recognition;
- optical-flow features;
- re-identification where justified;
- multi-camera identity/fusion experiments.

## Phase 4 — Camera/lighting intelligence

- controlled sweep of exposure/gain/focus/WB;
- lighting recipe entity;
- experiment orchestration across light intensity/angle/diffusion;
- scene drift detection;
- lens occlusion/dirty-lens detection;
- focus drift alarm;
- auto-lock exposure and white balance after stabilization;
- camera capability database by model/backend.

## Phase 5 — Model lifecycle

- dataset registry;
- label schema/versioning;
- MLflow or equivalent experiment tracking;
- model registry;
- offline evaluation gates;
- robustness/corruption suite;
- quantization;
- ONNX Runtime/OpenVINO/TensorRT backends;
- shadow/canary deployment;
- model and data drift observability.

## Phase 6 — SaaS/edge productization

- edge worker identity;
- signed/versioned recipes;
- camera fleet provisioning;
- secure secret distribution;
- health heartbeat;
- durable message bus;
- object storage for evidence;
- tenant-aware event schema;
- dashboard/control plane;
- human review and annotation loop;
- active learning;
- audit trail;
- RBAC and retention policies.

## Phase 7 — Multimodal reasoning

VLMs should consume selected event evidence and metadata, not blindly replace deterministic CV. Candidate uses:

- event explanation;
- operator Q&A;
- inspection report generation;
- semantic search over evidence;
- cross-checking deterministic model outputs;
- few-shot triage of novel defects.

Every multimodal decision that affects an industrial control action should have task-specific validation, latency limits, fallbacks and deterministic safety boundaries.
