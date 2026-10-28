"""Inter-channel cochlea/STFT difference descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from ._common import as_channels_first, matlab_spectrogram


def cochlea_difference(
    audio: ArrayLike,
    sample_rate: int,
) -> NDArray[np.float64]:
    """Compute the STFT-based branch of MATLAB ``cochleaDiff``."""

    values = as_channels_first(audio)
    window = signal.windows.hann(1024, sym=False)
    overlap = 1024 / max(1, round(0.010 * sample_rate))
    left = (
        np.abs(matlab_spectrogram(values[0], window, overlap)[:512])
        / np.sum(window)
    )
    if values.shape[0] == 1:
        return np.zeros_like(left)
    right = (
        np.abs(matlab_spectrogram(values[1], window, overlap)[:512])
        / np.sum(window)
    )
    frame_count = min(left.shape[1], right.shape[1])
    return left[:, :frame_count] - right[:, :frame_count]
