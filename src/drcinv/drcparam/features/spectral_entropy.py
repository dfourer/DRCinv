"""Spectral-entropy descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from ._common import EPS, as_channels_first, matlab_spectrogram


def spectral_entropy(audio: ArrayLike) -> NDArray[np.float64]:
    """Compute frame-wise Shannon entropy of each channel spectrum."""

    values = as_channels_first(audio)
    window = signal.windows.hann(2048, sym=False)
    entropy_values = []
    for samples in values:
        spectrum = matlab_spectrogram(samples, window, 2.0)
        power = np.abs(spectrum[: spectrum.shape[0] // 2]) ** 2
        probabilities = power / (np.sum(power, axis=0, keepdims=True) + EPS)
        entropy_values.append(
            -np.sum(probabilities * np.log2(probabilities + EPS), axis=0)
        )
    return np.asarray(entropy_values)
