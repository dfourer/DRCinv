"""Zero-lag cross-channel-correlation descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._common import EPS, as_channels_first


def cross_channel_correlation(audio: ArrayLike) -> float:
    """Return normalized zero-lag correlation of the first two channels."""

    values = as_channels_first(audio)
    if values.shape[0] == 1:
        return 1.0
    denominator = np.linalg.norm(values[0]) * np.linalg.norm(values[1])
    return float(np.dot(values[0], values[1]) / max(denominator, EPS))
