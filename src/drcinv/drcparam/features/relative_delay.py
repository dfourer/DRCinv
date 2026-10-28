"""Relative-delay descriptor for stereo audio."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from ._common import as_channels_first


def relative_delay(audio: ArrayLike, sample_rate: int) -> float:
    """Estimate stereo delay within the MATLAB ±0.5 ms search range."""

    values = as_channels_first(audio)
    if values.shape[1] < sample_rate / 2:
        raise ValueError("audio must contain at least 0.5 seconds")
    if values.shape[0] != 2:
        return 0.0
    max_lag = round(0.0005 * sample_rate)
    correlation = signal.correlate(
        values[0], values[1], mode="full", method="fft"
    )
    center = values.shape[1] - 1
    local = correlation[center - max_lag : center + max_lag + 1]
    lag = int(np.argmax(local)) - max_lag
    return float(-lag / sample_rate)
