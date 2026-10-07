from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GpuInfo:
    """Snapshot of the torch/CUDA capability actually available on this machine.

    Detection never raises: when torch is missing or no CUDA device is present the
    fields simply report that, so callers can degrade to CPU/synthetic paths instead
    of crashing. This mirrors the project's existing graceful-fallback philosophy
    (see ``isCodeIntelAvailable``-style checks across the codebase).
    """

    torch_available: bool
    cuda_available: bool
    device_count: int
    device_name: str | None
    torch_version: str | None
    cuda_version: str | None


def detect_gpu() -> GpuInfo:
    """Probe torch for a real CUDA-capable GPU without ever raising.

    Returns a :class:`GpuInfo` describing what was actually found. Benchmark and
    autotune code should branch on ``cuda_available`` instead of assuming GPU
    presence from configuration alone.
    """
    try:
        import torch
    except ImportError:
        return GpuInfo(
            torch_available=False,
            cuda_available=False,
            device_count=0,
            device_name=None,
            torch_version=None,
            cuda_version=None,
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    device_count = 0
    device_name = None
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_count = 0
            device_name = None

    return GpuInfo(
        torch_available=True,
        cuda_available=cuda_available,
        device_count=device_count,
        device_name=device_name,
        torch_version=getattr(torch, "__version__", None),
        cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
    )
