# Experimental Validation Plan

## Objective

Evaluate the effect of acquisition configuration, illumination and preprocessing on computer-vision quality and latency under controlled conditions.

## Independent variables

- camera/device;
- resolution;
- frame rate;
- exposure mode/value;
- gain;
- autofocus/focus;
- white balance;
- illumination source/intensity/angle/diffusion;
- working distance;
- target orientation;
- preprocessing recipe;
- model and input resolution;
- tracker.

## Dependent variables

### Acquisition

- achieved FPS;
- dropped/failed frame rate;
- luminance;
- contrast;
- sharpness;
- clipping ratios;
- noise proxy;
- camera warm-up/stabilization time.

### Model

For labeled data:

- precision;
- recall;
- F1;
- mAP50;
- mAP50-95;
- confusion matrix;
- per-class metrics.

For tracking:

- ID switches;
- track fragmentation;
- HOTA/IDF1 when ground truth is available;
- object-count error.

For real-time performance:

- preprocessing p50/p95;
- inference p50/p95;
- end-to-end FPS;
- event latency.

For industrial anomaly datasets:

- image-level AUROC/AP;
- pixel-level AUROC/AP when masks exist;
- PRO where appropriate;
- false alarms per operating interval.

## Recommended experiment matrix

Start with a factorial screening experiment instead of changing everything at once. For example:

```text
Resolution:       720p / 1080p
Exposure:         auto / locked
Lighting:         ambient / controlled diffuse
Preprocessing:    raw / adaptive
Model size:       nano / small
```

Run repeated clips for each condition and log both CV metrics and latency. Lock random seeds for offline model comparisons where applicable.

## Camera calibration experiment

Capture checkerboard views across the field of view, with meaningful tilt and multiple working distances. Record RMS and reprojection error. After calibration, compare geometry-sensitive measurements before/after undistortion.

## Acceptance gates

Define project-specific gates before testing. Example structure:

```text
capture_quality_score >= threshold
p95 inference latency <= budget
count error <= tolerance
detection recall >= target
false alarms <= target
```

Do not use generic thresholds as final industrial acceptance criteria. Derive them from the actual task, cost of misses/false alarms and validated dataset.
