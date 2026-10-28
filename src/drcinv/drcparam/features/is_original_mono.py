"""Duplicated-mono detection for multichannel audio."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from ._common import EPS, as_channels_first


def is_original_mono(audio: ArrayLike) -> bool:
    """Detect channels copied from the same mono source near zero lag."""

    values = as_channels_first(audio).copy()
    if values.shape[0] < 2:
        return False
    values /= np.maximum(
        np.max(np.abs(values), axis=1, keepdims=True),
        EPS,
    )
    center = values.shape[1] - 1
    normalization = np.mean(
        [np.dot(values[0], values[0]), np.dot(values[-1], values[-1])]
    )
    for channel in values[1:]:
        correlation = signal.correlate(
            values[0], channel, mode="full", method="fft"
        ) / max(normalization, EPS)
        if np.any(correlation[center - 5 : center + 6] > 0.99):
            return True
    return False
