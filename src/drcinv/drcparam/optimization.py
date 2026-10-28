"""Blind DRC parameter estimation in the selected feature space."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
import pickle
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

PARAMETER_KEYS = tuple(f"param{index}" for index in range(1, 7))


@lru_cache(maxsize=1)
def _runtime_dependencies():
    """Load audio processing dependencies only when optimization is executed."""

    from ..model_based import drc_inverse
    from .features import extract_selected_features
    return drc_inverse, extract_selected_features


@dataclass(frozen=True)
class ReferenceSetup:
    values: NDArray[np.float64]
    feature_indices: NDArray[np.int64]
    whitening_matrix: NDArray[np.float64]


@dataclass(frozen=True)
class EstimationResult:
    parameters: NDArray[np.float64]
    normalized_parameters: NDArray[np.float64]
    objective: float
    success: bool
    message: str


def load_profiles(path: str | Path) -> dict[str, dict[str, float]]:
    """Load profile dictionaries from JSON or trusted local pickle files."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            data = pickle.load(stream)
    else:
        raise ValueError("profile file must be JSON or pickle")
    if not isinstance(data, dict) or not data:
        raise ValueError("profile file must contain a non-empty mapping")
    return data


def parameter_bounds(
    profiles: dict[str, dict[str, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.asarray(
        [
            [profile[key] for key in PARAMETER_KEYS]
            for profile in profiles.values()
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("profile parameters must be finite")
    lower, upper = np.min(matrix, axis=0), np.max(matrix, axis=0)
    if np.any(upper <= lower):
        raise ValueError("every parameter must vary across the supplied profiles")
    return lower, upper


def invert_drc(
    wet_audio: ArrayLike,
    sample_rate: int,
    parameters: ArrayLike,
    process_type: str,
    detector_power: float = 2.0,
    if_normalize: bool = False,
) -> NDArray[np.float64]:
    """Invert DRC using profile parameters in their stored units.

    Profile JSON files store the four time constants (parameters 3--6) in
    milliseconds. :func:`drcinv.drc_inverse` expects them in seconds, so
    conversion is deliberately kept at this single interface boundary.
    """

    values = np.asarray(parameters, dtype=np.float64).ravel()
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain six finite values")
    if process_type == "compression":
        mode = "comp"
    elif process_type == "expansion":
        mode = "exp"
    else:
        raise ValueError("process_type must be 'compression' or 'expansion'")
    if detector_power not in (1, 2):
        raise ValueError("detector_power must be 1 or 2")
    drc_parameters = np.concatenate(
        (values[:2], values[2:6] / 1000.0, [float(detector_power)])
    )
    drc_inverse, _ = _runtime_dependencies()
    return np.asarray(
        drc_inverse(
            wet_audio,
            sample_rate,
            drc_parameters,
            mode=mode,
            optim="brentq",
            if_normalize=if_normalize,
        ),
        dtype=np.float64,
    )


def make_reference_setup(
    reference_features: ArrayLike,
    feature_indices: ArrayLike,
    covariance_shrinkage: float = 0.1,
) -> ReferenceSetup:
    """Build a regularized Mahalanobis target from dry features only."""

    if not 0.0 <= covariance_shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must be between 0 and 1")
    indices = np.asarray(feature_indices, dtype=np.int64)
    references = np.asarray(reference_features, dtype=np.float64)[:, indices]
    if references.ndim != 2 or len(references) < 2:
        raise ValueError("at least two dry feature rows are required")
    if not np.all(np.isfinite(references)):
        raise ValueError("dry reference features must be finite")

    center = np.mean(references, axis=0)
    covariance = np.atleast_2d(np.cov(references, rowvar=False, ddof=1))
    dimension = covariance.shape[0]
    variances = np.diag(covariance).copy()
    positive_variances = variances[variances > 0]
    typical_variance = (
        float(np.median(positive_variances)) if positive_variances.size else 1.0
    )
    variance_floor = max(typical_variance, 1.0) * 1e-10
    diagonal_covariance = np.diag(np.maximum(variances, variance_floor))
    regularized_covariance = (
        (1.0 - covariance_shrinkage) * covariance
        + covariance_shrinkage * diagonal_covariance
        + variance_floor * np.eye(dimension)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(regularized_covariance)
    eigenvalues = np.maximum(eigenvalues, variance_floor)
    whitening_matrix = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return ReferenceSetup(
        center,
        indices,
        whitening_matrix,
    )


def feature_distance(
    normalized_parameters: ArrayLike,
    wet_audio: ArrayLike,
    sample_rate: int,
    reference: ReferenceSetup,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    process_type: str,
    detector_power: float = 2.0,
    if_normalize: bool = False,
    prior: ArrayLike | None = None,
    prior_weight: float = 0.0,
    penalty: float = 1e6,
) -> float:
    z = np.asarray(normalized_parameters, dtype=np.float64).ravel()
    lower, upper = np.asarray(lower_bounds), np.asarray(upper_bounds)
    if z.size != 6 or np.any(~np.isfinite(z)) or np.any((z < 0) | (z > 1)):
        return penalty
    try:
        _, extract_selected_features = _runtime_dependencies()
        parameters = lower + z * (upper - lower)
        recovered = invert_drc(
            wet_audio,
            sample_rate,
            parameters,
            process_type,
            detector_power,
            if_normalize,
        )
        selected = extract_selected_features(
            recovered,
            sample_rate,
            reference.feature_indices,
        )
        distance = np.linalg.norm(
            reference.whitening_matrix @ (selected - reference.values)
        )
        if prior is not None and prior_weight > 0:
            distance += prior_weight * np.linalg.norm(z - np.asarray(prior).ravel())
        return float(distance) if np.isfinite(distance) else penalty
    except (ValueError, RuntimeError, FloatingPointError, OverflowError):
        return penalty


def estimate_parameters(
    wet_audio: ArrayLike,
    sample_rate: int,
    reference: ReferenceSetup,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    process_type: str,
    method: str = "powell",
    initial: ArrayLike | None = None,
    max_evaluations: int = 300,
    prior: ArrayLike | None = None,
    prior_weight: float = 0.0,
    seed_points: int = 12,
    random_seed: int = 42,
    detector_power: float = 2.0,
    if_normalize: bool = False,
) -> EstimationResult:
    """Estimate normalized controls with a derivative-free optimizer."""

    initial = (
        np.full(6, 0.5)
        if initial is None
        else np.clip(np.asarray(initial, dtype=np.float64), 0, 1)
    )

    def objective(z):
        return feature_distance(
            z,
            wet_audio,
            sample_rate,
            reference,
            lower_bounds,
            upper_bounds,
            process_type,
            detector_power=detector_power,
            if_normalize=if_normalize,
            prior=prior,
            prior_weight=prior_weight,
        )

    method_name = method.lower()
    if method_name == "powell":
        result = minimize(
            objective,
            initial,
            method="Powell",
            bounds=[(0.0, 1.0)] * 6,
            options={
                "maxfev": max_evaluations,
                "xtol": 1e-4,
                "ftol": 1e-4,
            },
        )
    elif method_name in {"patternsearch", "pattern_search"}:
        result = _pattern_search(
            objective, initial, max_evaluations, random_seed=random_seed
        )
    elif method_name in {"bayesopt", "bayesian", "bayesian_optimization"}:
        result = _bayesian_optimization(
            objective, initial, max_evaluations, seed_points, random_seed
        )
    elif method_name == "differential_evolution":
        result = differential_evolution(
            objective,
            [(0.0, 1.0)] * 6,
            maxiter=max(1, max_evaluations // 15),
            popsize=5,
            polish=True,
            seed=random_seed,
        )
    else:
        raise ValueError(
            "method must be 'patternsearch', 'bayesopt', 'powell', or "
            "'differential_evolution'"
        )
    lower, upper = np.asarray(lower_bounds), np.asarray(upper_bounds)
    parameters = lower + result.x * (upper - lower)
    return EstimationResult(
        np.asarray(parameters, dtype=np.float64),
        np.asarray(result.x, dtype=np.float64),
        float(result.fun),
        bool(result.success),
        str(result.message),
    )


@dataclass(frozen=True)
class _OptimizerResult:
    x: NDArray[np.float64]
    fun: float
    success: bool
    message: str


def _pattern_search(
    objective,
    initial: NDArray[np.float64],
    max_evaluations: int,
    random_seed: int,
) -> _OptimizerResult:
    """Run pymoo's bounded Hooke--Jeeves pattern-search implementation."""

    try:
        from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.termination.default import DefaultSingleObjectiveTermination
    except ImportError as error:
        raise ImportError(
            "Pattern search requires pymoo; install the project dependencies"
        ) from error

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    point = np.clip(np.asarray(initial, dtype=np.float64).ravel(), 0.0, 1.0)
    evaluations = 0
    best_point: NDArray[np.float64] | None = None
    best_value = np.inf

    class EvaluationBudgetReached(Exception):
        pass

    class ObjectiveProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=point.size,
                n_obj=1,
                xl=np.zeros(point.size),
                xu=np.ones(point.size),
            )

        def _evaluate(self, candidate, out, *args, **kwargs):
            nonlocal evaluations, best_point, best_value
            if evaluations >= max_evaluations:
                raise EvaluationBudgetReached
            value = float(objective(candidate))
            evaluations += 1
            if value < best_value:
                best_point = np.asarray(candidate, dtype=np.float64).copy()
                best_value = value
            out["F"] = value

    budget_exhausted = False
    try:
        pymoo_minimize(
            ObjectiveProblem(),
            PatternSearch(
                x0=point,
                init_delta=0.25,
                init_rho=0.5,
                step_size=1.0,
            ),
            termination=DefaultSingleObjectiveTermination(
                xtol=1e-4,
                ftol=1e-4,
                period=5,
                n_max_evals=max_evaluations,
            ),
            seed=random_seed,
            verbose=False,
        )
    except EvaluationBudgetReached:
        budget_exhausted = True
    if best_point is None or not np.isfinite(best_value):
        raise RuntimeError("pymoo pattern search did not return a solution")
    return _OptimizerResult(
        best_point,
        best_value,
        not budget_exhausted and evaluations < max_evaluations,
        (
            f"pymoo Hooke-Jeeves reached the maximum of {evaluations} "
            "objective evaluations"
            if budget_exhausted or evaluations >= max_evaluations
            else f"pymoo Hooke-Jeeves converged after {evaluations} evaluations"
        ),
    )


def _bayesian_optimization(
    objective,
    initial: NDArray[np.float64],
    max_evaluations: int,
    seed_points: int,
    random_seed: int,
) -> _OptimizerResult:
    """Run sequential Bayesian optimization through Ax and BoTorch."""

    try:
        from ax.adapter.registry import Generators
        from ax.api.client import Client
        from ax.api.configs import RangeParameterConfig
        from ax.generation_strategy.generation_node import GenerationStep
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        from linear_operator.utils.warnings import NumericalWarning
    except ImportError as error:
        raise ImportError(
            "Bayesian optimization requires ax-platform; install the project "
            "dependencies"
        ) from error

    # Ax reports every attached and completed trial at INFO level. A complete
    # audio file can contain hundreds of trials, so keep the console focused on
    # the per-file parameter summary printed by black_box_opt.py.
    for logger_name in ("ax", "ax.api", "ax.api.client", "botorch", "gpytorch"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    if seed_points < 1:
        raise ValueError("seed_points must be positive")

    initial = np.clip(np.asarray(initial, dtype=np.float64).ravel(), 0.0, 1.0)
    parameter_names = [f"z{index + 1}" for index in range(initial.size)]
    client = Client(random_seed=random_seed)
    client.configure_experiment(
        name="drc_parameter_estimation",
        parameters=[
            RangeParameterConfig(
                name=name,
                bounds=(0.0, 1.0),
                parameter_type="float",
            )
            for name in parameter_names
        ],
    )
    client.configure_optimization(objective="-1 * distance")

    count = min(max(1, seed_points), max_evaluations)
    points = qmc.LatinHypercube(initial.size, seed=random_seed).random(count)
    points[0] = initial
    values: list[float] = []

    def evaluate_and_record(point: NDArray[np.float64]) -> float:
        value = float(objective(point))
        values.append(value)
        return value

    for point in points:
        parameters = dict(zip(parameter_names, point.tolist()))
        trial_index = client.attach_trial(parameters=parameters)
        value = evaluate_and_record(point)
        client.complete_trial(
            trial_index=trial_index,
            raw_data={"distance": (value, 0.0)},
        )

    client.set_generation_strategy(
        GenerationStrategy(
            name="BoTorch",
            nodes=[
                GenerationStep(
                    generator=Generators.BOTORCH_MODULAR,
                    num_trials=-1,
                )
            ],
        )
    )

    evaluated_points = [point.copy() for point in points]
    while len(values) < max_evaluations:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumericalWarning)
            trials = client.get_next_trials(max_trials=1)
        if len(trials) != 1:
            raise RuntimeError("Ax did not return exactly one Bayesian trial")
        trial_index, parameters = next(iter(trials.items()))
        point = np.asarray(
            [parameters[name] for name in parameter_names], dtype=np.float64
        )
        value = evaluate_and_record(point)
        evaluated_points.append(point)
        client.complete_trial(
            trial_index=trial_index,
            raw_data={"distance": (value, 0.0)},
        )

    best = int(np.argmin(values))
    return _OptimizerResult(
        np.asarray(evaluated_points[best], dtype=np.float64),
        float(values[best]),
        True,
        f"Ax/BoTorch completed {max_evaluations} objective evaluations",
    )
