"""Background-noise-level descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._common import EPS, as_channels_first


def background_noise_level(audio: ArrayLike, sample_rate: int) -> float:
    """Measure the quietest 5 ms frame relative to the signal peak."""

    values = as_channels_first(audio)
    frame_length = max(1, round(0.005 * sample_rate))
    peak_db = 10.0 * np.log10(EPS + np.max(np.abs(values)))
    minimum_power_db = np.inf

    # MATLAB's linear indexing visits only the first channel because its loop
    # count is based on length(x), not numel(x).
    first_channel = values[0]
    for start in range(0, first_channel.size, frame_length):
        frame = first_channel[start : start + frame_length]
        energy_db = 10.0 * np.log10(EPS + np.mean(frame**2))
        minimum_power_db = min(minimum_power_db, energy_db)
    return float(minimum_power_db - peak_db)
