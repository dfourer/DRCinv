"""Frequency-bandwidth descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from ._common import EPS, as_channels_first


def bandwidth(audio: ArrayLike, sample_rate: int) -> NDArray[np.float64]:
    """Estimate the thresholded bandwidth of successive 10 ms frames."""

    values = as_channels_first(audio)
    hop = max(1, round(0.010 * sample_rate))
    frame_count = int(np.ceil(values.shape[1] / hop))
    result = np.zeros((values.shape[0], frame_count), dtype=np.float64)

    for channel, samples in enumerate(values):
        for frame_index in range(frame_count):
            frame = samples[frame_index * hop : frame_index * hop + 2048]
            if frame.size == 0:
                continue
            power = np.abs(
                np.fft.fft(signal.windows.hann(frame.size) * frame, 2048)
            ) ** 2
            half_spectrum = power[:1024]
            peak = np.max(half_spectrum)
            if peak <= 0:
                continue
            spectrum_db = 10.0 * np.log10(
                np.maximum(half_spectrum / peak, EPS)
            )
            threshold = (np.min(spectrum_db) - np.max(spectrum_db)) * 0.7
            below_threshold = np.flatnonzero(spectrum_db < threshold)
            if below_threshold.size:
                result[channel, frame_index] = (
                    below_threshold[0] / 2048.0 * sample_rate
                )
    return result
