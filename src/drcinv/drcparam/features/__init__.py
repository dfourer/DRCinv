"""Objective audio-quality feature extractors.

Each descriptor or closely related descriptor family lives in its own module,
mirroring the organization of the original MATLAB ``feature`` directory.
"""

from .average_spectrum import average_spectrum
from .background_noise_level import background_noise_level
from .balance import balance
from .bandwidth import bandwidth
from .cochlea_difference import cochlea_difference
from .cross_channel_correlation import cross_channel_correlation
from .dc_offset import dc_offset
from .dynamic import dynamic_histogram, dynamic_summary
from .extractor import (
    FEATURE_NAMES,
    AudioFeatures,
    extract_features,
    extract_selected_features,
)
from .is_original_mono import is_original_mono
from .relative_delay import relative_delay
from .spectral_entropy import spectral_entropy
from .stereo_phase_spread import stereo_phase_spread

__all__ = [
    "AudioFeatures",
    "FEATURE_NAMES",
    "average_spectrum",
    "background_noise_level",
    "balance",
    "bandwidth",
    "cochlea_difference",
    "cross_channel_correlation",
    "dc_offset",
    "dynamic_histogram",
    "dynamic_summary",
    "extract_features",
    "extract_selected_features",
    "is_original_mono",
    "relative_delay",
    "spectral_entropy",
    "stereo_phase_spread",
]
