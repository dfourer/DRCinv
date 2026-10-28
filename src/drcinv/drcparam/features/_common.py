"""Shared array and spectrogram helpers for feature extractors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal


EPS = np.finfo(np.float64).eps


def as_channels_first(audio: ArrayLike) -> NDArray[np.float64]:
    """Return audio as a floating-point ``(channels, samples)`` array."""

    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        return values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError("audio must be a one- or two-dimensional array")
    # Audio libraries normally return samples x channels. Preserve an already
    # channel-first short matrix for MATLAB interoperability.
    return values.T if values.shape[0] > values.shape[1] else values


def matlab_spectrogram(
    audio: NDArray[np.float64],
    window: NDArray[np.float64],
    overlap: float,
) -> NDArray[np.complex128]:
    """Reproduce the custom MATLAB ``spectrogram.m`` framing behavior."""

    values = np.asarray(audio, dtype=np.float64).ravel()
    window = np.asarray(window, dtype=np.float64).ravel()
    frame_length = window.size
    hop = max(1, round(frame_length / overlap))
    frame_count = values.size // hop
    result = np.zeros((frame_length, frame_count), dtype=np.complex128)

    for frame_index in range(frame_count):
        start = frame_index * hop
        frame = values[start : min(start + frame_length, values.size)]
        current_window = window
        if frame.size != frame_length:
            current_window = signal.resample(window, frame.size)
        result[: frame.size, frame_index] = np.fft.fft(frame * current_window)
    return result
