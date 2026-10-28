"""Per-channel DC-offset descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._common import EPS, as_channels_first


def dc_offset(audio: ArrayLike) -> NDArray[np.float64]:
    """Return each channel mean normalized by its absolute peak."""

    values = as_channels_first(audio)
    return np.asarray(
        [
            np.mean(channel) / max(np.max(np.abs(channel)), EPS)
            for channel in values
        ]
    )
