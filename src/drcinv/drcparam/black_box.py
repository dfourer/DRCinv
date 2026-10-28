"""Feature-space black-box estimation of DRC parameters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .mee import DRCParameters
from .optimization import (
    EstimationResult as OptimizerResult,
    ReferenceSetup,
    estimate_parameters,
    parameter_bounds,
)


SUPPORTED_REFERENCE_METHODS = (
    "mrmr",
    "mutual_information",
    "binary_fisher",
    "inverse_fisher",
)
SUPPORTED_PROCESS_TYPES = ("compression", "expansion")


@dataclass(frozen=True)
class BlackBoxEstimationResult:
    """Detailed result from feature-space parameter optimization."""

    parameters: DRCParameters
    objective: float
    success: bool
    message: str
    optimizer: str
    reference_method: str
    process_type: str


def _resource(filename: str):
    return resources.files("drcinv.drcparam.data").joinpath(filename)


def _reference_from_arrays(arrays) -> ReferenceSetup:
    required = {"reference_point", "feature_indices", "whitening_matrix"}
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(f"Reference point is missing {sorted(missing)}.")
    return ReferenceSetup(
        values=np.asarray(arrays["reference_point"], dtype=np.float64).copy(),
        feature_indices=np.asarray(
            arrays["feature_indices"], dtype=np.int64
        ).copy(),
        whitening_matrix=np.asarray(
            arrays["whitening_matrix"], dtype=np.float64
        ).copy(),
    )


def load_reference_setup(
    reference_method: str,
    process_type: str,
    reference_path: Optional[Union[str, Path]] = None,
) -> tuple[ReferenceSetup, dict]:
    """Load a packaged or user-supplied AQFeature reference point."""

    reference_method = reference_method.lower()
    process_type = process_type.lower()
    if reference_method not in SUPPORTED_REFERENCE_METHODS:
        raise ValueError(
            "reference_method must be one of "
            f"{list(SUPPORTED_REFERENCE_METHODS)}."
        )
    if process_type not in SUPPORTED_PROCESS_TYPES:
        raise ValueError("process_type must be 'compression' or 'expansion'.")

    if reference_path is None:
        stem = f"ref_point_10s_{reference_method}_{process_type}_develop"
        npz_resource = _resource(f"{stem}.npz")
        json_resource = _resource(f"{stem}.json")
        with npz_resource.open("rb") as stream:
            with np.load(stream) as arrays:
                reference = _reference_from_arrays(arrays)
        metadata = json.loads(json_resource.read_text(encoding="utf-8"))
    else:
        npz_path = Path(reference_path).expanduser()
        if npz_path.suffix.lower() != ".npz":
            npz_path = npz_path.with_suffix(".npz")
        metadata_path = npz_path.with_suffix(".json")
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Reference requires both {npz_path} and {metadata_path}."
            )
        with np.load(npz_path) as arrays:
            reference = _reference_from_arrays(arrays)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected = {
        "process_type": process_type,
        "selection_method": reference_method,
        "dataset_split": "develop",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Reference metadata mismatch: {mismatches}")
    return reference, metadata


def _load_profiles(process_type: str, profiles_path=None):
    if profiles_path is None:
        resource = _resource(f"profiles_{process_type}.json")
        profiles = json.loads(resource.read_text(encoding="utf-8"))
    else:
        profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Profile data must be a non-empty JSON mapping.")
    return profiles


def _prepare_wet_audio(audio, sample_rate: int) -> np.ndarray:
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    waveform = np.asarray(audio, dtype=np.float64)
    if waveform.ndim == 2:
        if waveform.shape[1] <= 8:
            waveform = waveform.mean(axis=1)
        elif waveform.shape[0] <= 8:
            waveform = waveform.mean(axis=0)
        else:
            raise ValueError(
                "For 2-D audio, one dimension must contain at most 8 channels."
            )
    if waveform.ndim != 1 or waveform.size < sample_rate / 2:
        raise ValueError("audio must contain at least 0.5 seconds of samples.")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("audio contains NaN or Inf.")
    return np.ascontiguousarray(waveform)


class BlackBoxParameterEstimator:
    """Estimate DRC controls by derivative-free AQFeature optimization.

    ``reference_method`` is deliberately required: the library does not choose
    mRMR, mutual information, Binary Fisher, or Inverse Fisher on the user's
    behalf.
    """

    def __init__(
        self,
        *,
        reference_method: str,
        process_type: str = "compression",
        optimizer: str = "powell",
        max_evaluations: int = 100,
        seed_points: int = 12,
        random_seed: int = 42,
        detector_order: int = 2,
        normalize_recovered: bool = True,
        reference_path: Optional[Union[str, Path]] = None,
        profiles_path: Optional[Union[str, Path]] = None,
    ):
        self.reference_method = reference_method.lower()
        self.process_type = process_type.lower()
        self.optimizer = optimizer.lower()
        self.max_evaluations = int(max_evaluations)
        self.seed_points = int(seed_points)
        self.random_seed = int(random_seed)
        self.detector_order = int(detector_order)
        self.normalize_recovered = bool(normalize_recovered)
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive.")
        if self.detector_order not in (1, 2):
            raise ValueError("detector_order must be 1 or 2.")
        self.reference, self.reference_metadata = load_reference_setup(
            self.reference_method,
            self.process_type,
            reference_path,
        )
        profiles = _load_profiles(self.process_type, profiles_path)
        self.lower_bounds, self.upper_bounds = parameter_bounds(profiles)
        self.last_result: Optional[BlackBoxEstimationResult] = None

    def estimate_result(
        self,
        audio: Union[np.ndarray, Sequence[float]],
        sample_rate: int,
        *,
        initial: Optional[Sequence[float]] = None,
        prior: Optional[Sequence[float]] = None,
        prior_weight: float = 0.0,
    ) -> BlackBoxEstimationResult:
        """Return parameters plus optimizer diagnostics."""

        waveform = _prepare_wet_audio(audio, sample_rate)
        optimized: OptimizerResult = estimate_parameters(
            waveform,
            sample_rate,
            self.reference,
            self.lower_bounds,
            self.upper_bounds,
            self.process_type,
            method=self.optimizer,
            initial=initial,
            max_evaluations=self.max_evaluations,
            prior=prior,
            prior_weight=prior_weight,
            seed_points=self.seed_points,
            random_seed=self.random_seed,
            detector_power=self.detector_order,
            if_normalize=self.normalize_recovered,
        )
        values = optimized.parameters
        parameters = DRCParameters(
            threshold_dbfs=float(values[0]),
            ratio=float(values[1]),
            envelope_attack_s=float(values[2] / 1000.0),
            envelope_release_s=float(values[3] / 1000.0),
            gain_attack_s=float(values[4] / 1000.0),
            gain_release_s=float(values[5] / 1000.0),
            detector_order=self.detector_order,
            normalized=tuple(
                float(value) for value in optimized.normalized_parameters
            ),
        )
        result = BlackBoxEstimationResult(
            parameters=parameters,
            objective=optimized.objective,
            success=optimized.success,
            message=optimized.message,
            optimizer=self.optimizer,
            reference_method=self.reference_method,
            process_type=self.process_type,
        )
        self.last_result = result
        return result

    def estimate(self, audio, sample_rate: int, **kwargs) -> DRCParameters:
        """Return parameters in the same form as the MEE estimator."""

        return self.estimate_result(audio, sample_rate, **kwargs).parameters
