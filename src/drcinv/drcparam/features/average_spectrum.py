"""Average-spectrum descriptors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._common import EPS, matlab_spectrogram


def average_spectrum(
    audio: ArrayLike,
    sample_rate: int,
    delta_frequency: float = 44100 / 4096,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    float,
    NDArray[np.float64],
]:
    """Return normalized average spectrum and its scalar descriptors."""

    values = np.asarray(audio, dtype=np.float64).ravel()
    fft_length = max(2, round(sample_rate / delta_frequency))
    spectrum = matlab_spectrogram(values, np.ones(fft_length), 1.0)
    half_length = round(fft_length / 2)
    if spectrum.shape[1] == 0:
        raise ValueError("audio is shorter than the average-spectrum FFT length")
    power = np.mean(np.abs(spectrum[:half_length]) ** 2, axis=1)
    power /= max(np.sum(power), EPS)
    cumulative = np.cumsum(power)
    frequencies = np.arange(half_length) / fft_length * sample_rate
    centroid = float(np.sum(frequencies * power))
    peak_index = int(np.argmax(power))
    return (
        power,
        cumulative,
        centroid,
        float(power[peak_index]),
        float(frequencies[peak_index]),
        frequencies,
    )
