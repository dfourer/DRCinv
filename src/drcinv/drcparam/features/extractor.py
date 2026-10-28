"""Combined 26-dimensional audio-quality feature extractor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._common import EPS, as_channels_first
from .average_spectrum import average_spectrum
from .background_noise_level import background_noise_level
from .balance import balance
from .bandwidth import bandwidth
from .cochlea_difference import cochlea_difference
from .cross_channel_correlation import cross_channel_correlation
from .dc_offset import dc_offset
from .dynamic import dynamic_summary
from .is_original_mono import is_original_mono
from .relative_delay import relative_delay
from .spectral_entropy import spectral_entropy
from .stereo_phase_spread import stereo_phase_spread


FEATURE_NAMES = (
    "BW",
    "BNL",
    "SE",
    "Bal",
    "mean CochDiff",
    "std CochDiff",
    "CCCor",
    "DCOff",
    "DCOff ratio",
    "SSPS",
    "RDelay",
    "isMono",
    "DH1",
    "DH2",
    "DH3",
    "DH4",
    "DH5",
    "DH6",
    "DH7",
    "DH8",
    "AVSP1",
    "AVSP2",
    "AVSP3",
    "AVSP4",
    "AVSP5",
    "AVSP6",
)


@dataclass(frozen=True)
class AudioFeatures:
    """Feature vector plus intermediate data used by the MATLAB workflow."""

    values: NDArray[np.float64]
    names: tuple[str, ...]
    dynamic_histogram: NDArray[np.float64]
    dynamic_axis_db: NDArray[np.float64]
    dynamic_summary: NDArray[np.float64]
    average_spectrum: NDArray[np.float64]
    cumulative_spectrum: NDArray[np.float64]
    frequency_axis_hz: NDArray[np.float64]


def extract_features(audio: ArrayLike, sample_rate: int) -> AudioFeatures:
    """Extract the 26 descriptors returned by MATLAB ``F_audioQ_desc``."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = as_channels_first(audio)
    if values.shape[1] < sample_rate / 2:
        raise ValueError("audio must contain at least 0.5 seconds")

    bandwidth_values = bandwidth(values, sample_rate)
    noise_level = background_noise_level(values, sample_rate)
    entropy = spectral_entropy(values)
    channel_balance = balance(values)
    cochlea = cochlea_difference(values, sample_rate)
    offsets = dc_offset(values)
    offsets = np.pad(offsets, (0, max(0, 2 - offsets.size)), mode="edge")
    histogram, histogram_axis, histogram_summary = dynamic_summary(values[0])
    (
        spectrum,
        cumulative,
        centroid,
        peak,
        peak_frequency,
        frequency_axis,
    ) = average_spectrum(values[0], sample_rate)

    feature_values = np.concatenate(
        [
            np.asarray(
                [
                    np.mean(bandwidth_values),
                    noise_level,
                    np.mean(entropy),
                    channel_balance[0],
                    100.0 * np.max(np.mean(cochlea, axis=1)),
                    100.0 * np.std(np.mean(cochlea, axis=1), ddof=1),
                    cross_channel_correlation(values),
                    np.mean(offsets),
                    offsets[0] / (EPS + offsets[1]),
                    stereo_phase_spread(values),
                    relative_delay(values, sample_rate),
                    float(is_original_mono(values)),
                ]
            ),
            histogram_summary,
            np.asarray(
                [
                    np.mean(spectrum),
                    np.median(spectrum),
                    np.std(spectrum, ddof=1),
                    centroid,
                    peak,
                    peak_frequency,
                ]
            ),
        ]
    ).astype(np.float64)
    return AudioFeatures(
        feature_values,
        FEATURE_NAMES,
        histogram,
        histogram_axis,
        histogram_summary,
        spectrum,
        cumulative,
        frequency_axis,
    )


def extract_selected_features(
    audio: ArrayLike,
    sample_rate: int,
    feature_indices: ArrayLike,
) -> NDArray[np.float64]:
    """Extract only the requested entries of the 26-dimensional vector.

    The returned values follow ``feature_indices`` exactly, including its
    order. Descriptors from the same family (for example DH1--DH8) share one
    intermediate calculation, while unrequested families are not evaluated.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = as_channels_first(audio)
    if values.shape[1] < sample_rate / 2:
        raise ValueError("audio must contain at least 0.5 seconds")

    raw_indices = np.asarray(feature_indices)
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        raise ValueError("feature_indices must be a non-empty one-dimensional array")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("feature_indices must contain integers")
    indices = raw_indices.astype(np.int64, copy=False)
    if np.any((indices < 0) | (indices >= len(FEATURE_NAMES))):
        raise ValueError(
            f"feature indices must be between 0 and {len(FEATURE_NAMES) - 1}"
        )

    cache: dict[str, NDArray[np.float64]] = {}

    def feature_value(index: int) -> float:
        if index == 0:
            return float(np.mean(bandwidth(values, sample_rate)))
        if index == 1:
            return float(background_noise_level(values, sample_rate))
        if index == 2:
            return float(np.mean(spectral_entropy(values)))
        if index == 3:
            return float(balance(values)[0])
        if index in (4, 5):
            if "cochlea" not in cache:
                cochlea = cochlea_difference(values, sample_rate)
                channel_means = np.mean(cochlea, axis=1)
                cache["cochlea"] = np.asarray(
                    [
                        100.0 * np.max(channel_means),
                        100.0 * np.std(channel_means, ddof=1),
                    ],
                    dtype=np.float64,
                )
            return float(cache["cochlea"][index - 4])
        if index == 6:
            return float(cross_channel_correlation(values))
        if index in (7, 8):
            if "dc_offset" not in cache:
                offsets = dc_offset(values)
                offsets = np.pad(
                    offsets,
                    (0, max(0, 2 - offsets.size)),
                    mode="edge",
                )
                cache["dc_offset"] = np.asarray(
                    [np.mean(offsets), offsets[0] / (EPS + offsets[1])],
                    dtype=np.float64,
                )
            return float(cache["dc_offset"][index - 7])
        if index == 9:
            return float(stereo_phase_spread(values))
        if index == 10:
            return float(relative_delay(values, sample_rate))
        if index == 11:
            return float(is_original_mono(values))
        if 12 <= index <= 19:
            if "dynamic" not in cache:
                cache["dynamic"] = dynamic_summary(values[0])[2]
            return float(cache["dynamic"][index - 12])
        if "spectrum" not in cache:
            spectrum, _, centroid, peak, peak_frequency, _ = average_spectrum(
                values[0], sample_rate
            )
            cache["spectrum"] = np.asarray(
                [
                    np.mean(spectrum),
                    np.median(spectrum),
                    np.std(spectrum, ddof=1),
                    centroid,
                    peak,
                    peak_frequency,
                ],
                dtype=np.float64,
            )
        return float(cache["spectrum"][index - 20])

    return np.asarray([feature_value(int(index)) for index in indices])
