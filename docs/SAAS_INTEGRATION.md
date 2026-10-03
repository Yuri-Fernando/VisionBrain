# SaaS Integration Contract

VisionBrain should run as a worker/edge service. The SaaS should treat it as a versioned execution engine.

## Recommended entities

```text
Tenant
Site
Camera
CameraCapabilitySnapshot
VisionRecipe
ModelVersion
RuleSet
Deployment
VisionEvent
EvidenceAsset
HealthMetric
ExperimentRun
```

## Recipe

A `VisionRecipe` should version:

```json
{
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "exposure_mode": "auto"
  },
  "preprocess": {
    "adaptive": true,
    "clahe": true,
    "denoise": "bilateral"
  },
  "model": {
    "task": "detect",
    "artifact": "model.onnx",
    "input_size": 640,
    "confidence": 0.35
  },
  "tracker": {
    "name": "bytetrack"
  },
  "rules": []
}
```

## Event payload

Current local event model maps cleanly to an external schema:

```json
{
  "event_type": "zone_enter",
  "timestamp": "2026-10-03T12:34:56Z",
  "frame_index": 12345,
  "severity": "info",
  "payload": {
    "camera_id": "cam-01",
    "recipe_version": "17",
    "model_version": "detector-12",
    "zone": "conveyor_exit",
    "track_id": 42,
    "label": "box",
    "snapshot": "..."
  }
}
```

## Production transport

For a prototype, HTTP webhook is fine. For production, use a durable event boundary such as Redis Streams, NATS, RabbitMQ, Kafka or a cloud queue. Use an outbox/retry policy so network failure never blocks the frame loop.

## Multi-tenant rule

Never place tenant secrets or customer authorization logic inside a model adapter. The worker receives a validated deployment/recipe identity and emits results under that identity.

## Edge / cloud split

Prefer edge inference when latency, bandwidth, privacy or plant connectivity matter. Send events, metrics and selected evidence to the cloud instead of all raw video. Retain raw clips only when the customer's policy and experiment requirements justify it.
