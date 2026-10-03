from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True)
class CameraRead:
    """One acquired frame and the monotonic timestamp captured with it."""

    ok: bool
    frame: np.ndarray | None
    timestamp_monotonic: float


@runtime_checkable
class FrameSource(Protocol):
    """Narrow acquisition contract used by the real-time runtime."""

    def open(self) -> "FrameSource": ...

    def read(self) -> CameraRead: ...

    def warmup(self, frames: int | None = None) -> None: ...

    def probe(self) -> dict[str, Any]: ...

    def release(self) -> None: ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, *_: object) -> None: ...
