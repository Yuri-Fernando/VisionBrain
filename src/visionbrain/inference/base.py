from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from visionbrain.inference.yolo import InferenceOutput


@runtime_checkable
class DetectorBackend(Protocol):
    """Model-agnostic contract consumed by the runtime."""

    def load(self) -> None: ...

    def infer(self, frame: np.ndarray) -> InferenceOutput: ...
