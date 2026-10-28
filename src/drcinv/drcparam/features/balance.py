"""Inter-channel energy-balance descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._common import EPS, as_channels_first


def balance(audio: ArrayLike) -> NDArray[np.float64]:
    """Return channel-one energy divided by each other channel energy."""

    values = as_channels_first(audio)
    if values.shape[0] == 1:
        return np.ones(1)
    reference_energy = np.sum(np.abs(values[0]) ** 2)
    return np.asarray(
        [
            reference_energy / (EPS + np.sum(np.abs(channel) ** 2))
            for channel in values[1:]
        ]
    )
