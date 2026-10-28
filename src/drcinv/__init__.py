"""Dynamic-range control, inversion, and parameter estimation."""

from .model_based import drc_forward, drc_inverse
from . import drcparam

__version__ = "0.3.0"

__all__ = [
    "drc_forward",
    "drc_inverse",
    "drcparam",
]
