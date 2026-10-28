"""Spectral stereo phase-spread descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._common import EPS, as_channels_first


def stereo_phase_spread(audio: ArrayLike) -> float:
    """Measure the standard-deviation ratio of side and mid signals."""

    values = as_channels_first(audio)
    if values.shape[0] != 2:
        return 0.0
    side_deviation = np.std(values[0] - values[1], ddof=1)
    mid_deviation = np.std(values[0] + values[1], ddof=1)
    return float(side_deviation / max(mid_deviation, EPS))
