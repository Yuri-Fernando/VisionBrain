"""Optional GenICam/GenTL acquisition path for industrial cameras.

Requires a camera-vendor GenTL Producer (.cti) and ``harvesters``. This module
is isolated so webcam/OpenCV users have zero dependency on it.
"""

from __future__ import annotations

import numpy as np


class GenICamCamera:
    def __init__(self, cti_path: str, device_index: int = 0):
        self.cti_path = cti_path
        self.device_index = device_index
        self._harvester = None
        self._acquirer = None

    def open(self) -> "GenICamCamera":
        try:
            from harvesters.core import Harvester
        except ImportError as exc:
            raise RuntimeError("Install industrial camera support: pip install -e '.[genicam]'") from exc
        harvester = Harvester()
        harvester.add_file(self.cti_path)
        harvester.update()
        if not harvester.device_info_list:
            raise RuntimeError("No GenICam device discovered by the supplied GenTL Producer.")
        acquirer = harvester.create(self.device_index)
        acquirer.start()
        self._harvester = harvester
        self._acquirer = acquirer
        return self

    def read(self) -> np.ndarray:
        if self._acquirer is None:
            raise RuntimeError("GenICam camera is not open.")
        with self._acquirer.fetch() as buffer:
            component = buffer.payload.components[0]
            data = component.data.reshape(component.height, component.width, -1)
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            return np.ascontiguousarray(data[:, :, :3])

    def release(self) -> None:
        if self._acquirer is not None:
            self._acquirer.stop()
            self._acquirer.destroy()
            self._acquirer = None
        if self._harvester is not None:
            self._harvester.reset()
            self._harvester = None
