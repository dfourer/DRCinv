"""DRC parameter estimation with MEE or feature-space black-box search."""

from __future__ import annotations

from .black_box import (
    BlackBoxEstimationResult,
    BlackBoxParameterEstimator,
    SUPPORTED_REFERENCE_METHODS,
    load_reference_setup,
)
from .mee import (
    DRCParameters,
    MEEParameterEstimator,
    download_checkpoint,
    estimate_mee_parameters,
)


__version__ = "0.3.0"

__all__ = [
    "BlackBoxEstimationResult",
    "BlackBoxParameterEstimator",
    "DRCParameters",
    "MEEParameterEstimator",
    "SUPPORTED_REFERENCE_METHODS",
    "download_checkpoint",
    "estimate_drc_parameters",
    "estimate_mee_parameters",
    "load_reference_setup",
]


def estimate_drc_parameters(audio, sample_rate: int, method: str = "mee", **kwargs):
    """Estimate a seven-element DRC vector with the selected method.

    ``method="mee"`` preserves the original one-shot behavior. For
    ``method="black_box"``, ``reference_method`` must be provided explicitly.
    """

    method_name = method.lower()
    if method_name == "mee":
        return MEEParameterEstimator(**kwargs).estimate(audio, sample_rate).as_vector()
    if method_name in {"black_box", "blackbox"}:
        return BlackBoxParameterEstimator(**kwargs).estimate(
            audio, sample_rate
        ).as_vector()
    raise ValueError("method must be 'mee' or 'black_box'.")
