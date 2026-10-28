"""Dynamic histogram and its eight summary descriptors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._common import EPS


def dynamic_histogram(
    audio: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the normalized -100 to 0 dB sample-level histogram."""

    values = np.asarray(audio, dtype=np.float64).ravel()
    with np.errstate(divide="ignore"):
        levels_db = 20.0 * np.log10(np.abs(values))
    levels_db = levels_db[(levels_db >= -100.0) & (levels_db <= 0.0)]
    counts, _ = np.histogram(levels_db, bins=np.arange(-100.0, 1.0))
    histogram = counts.astype(np.float64)
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    return histogram, np.arange(-99.0, 1.0), levels_db


def dynamic_summary(
    audio: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the histogram, axis, and MATLAB-compatible DH1-DH8 values."""

    histogram, axis_db, levels_db = dynamic_histogram(audio)
    if not np.any(histogram):
        return histogram, axis_db, np.full(8, np.nan)
    peak_index = int(np.argmax(histogram))
    nonzero = histogram > EPS
    entropy = -np.sum(histogram[nonzero] * np.log2(histogram[nonzero]))
    mean_energy_db = np.sum(axis_db * histogram)
    level_std = (
        0.0
        if levels_db.size < 2
        else float(np.std(levels_db, ddof=1))
    )
    skewness = (
        0.0
        if levels_db.size < 3 or level_std <= 1e-12
        else float(stats.skew(levels_db, bias=False))
    )
    summary = np.asarray(
        [
            mean_energy_db,
            histogram[peak_index],
            axis_db[peak_index],
            mean_energy_db / np.sum(histogram),
            np.median(levels_db),
            entropy,
            level_std,
            skewness,
        ]
    )
    return histogram, axis_db, summary
