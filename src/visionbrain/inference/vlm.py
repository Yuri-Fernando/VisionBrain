from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


class SnapshotVLM:
    """Optional snapshot-level VLM/VQA adapter.

    Keep this off the real-time frame loop. Trigger it on an event, a saved image, or on demand.
    """

    def __init__(self, model: str = "llava-hf/llava-interleave-qwen-0.5b-hf", device: int | str | None = None):
        self.model_name = model
        self.device = device
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install the VLM extra: pip install -e '.[vlm]'") from exc
        kwargs: dict[str, Any] = {"task": "image-text-to-text", "model": self.model_name}
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipe = pipeline(**kwargs)

    def ask(self, frame_bgr: np.ndarray, question: str, max_new_tokens: int = 96) -> str:
        self.load()
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        output = self._pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
        if isinstance(output, list) and output:
            item = output[0]
            if isinstance(item, dict):
                return str(item.get("generated_text", item))
        return str(output)
