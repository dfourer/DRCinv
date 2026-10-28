import numpy as np
import pytest

from drcinv.drcparam import (
    BlackBoxParameterEstimator,
    SUPPORTED_REFERENCE_METHODS,
    load_reference_setup,
)
from drcinv.drcparam.optimization import EstimationResult


@pytest.mark.parametrize("method", SUPPORTED_REFERENCE_METHODS)
@pytest.mark.parametrize("process_type", ["compression", "expansion"])
def test_packaged_reference_points(method, process_type):
    reference, metadata = load_reference_setup(method, process_type)
    assert reference.values.shape == (3,)
    assert reference.feature_indices.shape == (3,)
    assert reference.whitening_matrix.shape == (3, 3)
    assert metadata["selection_method"] == method
    assert metadata["process_type"] == process_type


def test_reference_method_is_required():
    with pytest.raises(TypeError, match="reference_method"):
        BlackBoxParameterEstimator()


def test_black_box_converts_milliseconds_to_seconds(monkeypatch):
    physical_ms = np.asarray([-20, 4, 5, 50, 10, 100], dtype=float)

    def fake_estimate(*args, **kwargs):
        return EstimationResult(
            parameters=physical_ms,
            normalized_parameters=np.full(6, 0.5),
            objective=1.25,
            success=True,
            message="test",
        )

    monkeypatch.setattr(
        "drcinv.drcparam.black_box.estimate_parameters", fake_estimate
    )
    estimator = BlackBoxParameterEstimator(
        reference_method="mutual_information",
        max_evaluations=1,
    )
    result = estimator.estimate_result(np.zeros(4000), 8000)
    np.testing.assert_allclose(
        result.parameters.as_vector(),
        [-20, 4, 0.005, 0.05, 0.01, 0.1, 2],
    )
    assert result.reference_method == "mutual_information"
    assert estimator.last_result is result
