
import numpy as np
import pytest

from drcinv import drc_forward, drc_inverse


FS = 44100


@pytest.mark.parametrize("mode", ["comp", "exp"])
def test_identity(mode):
    """A ratio of 1 should leave the signal unchanged."""
    x = np.array([0.1, -0.2, 0.3, -0.4])
    q = [-20, 1, 0.01, 0.1, 0.01, 0.1, 2]

    y = drc_forward(x, FS, q, mode=mode)
    x_hat = drc_inverse(y, FS, q, mode=mode)

    np.testing.assert_allclose(y, x, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(x_hat, x, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("mode", ["comp", "exp"])
def test_zero_signal(mode):
    """A zero signal should remain zero."""
    x = np.zeros(32)
    q = [-20, 4, 0.01, 0.1, 0.01, 0.1, 2]

    y = drc_forward(x, FS, q, mode=mode)
    x_hat = drc_inverse(y, FS, q, mode=mode)

    np.testing.assert_allclose(y, x, atol=1e-10)
    np.testing.assert_allclose(x_hat, x, atol=1e-10)


@pytest.mark.parametrize(
    "mode,threshold",
    [
        ("comp", -30.0),
        ("exp", -10.0),
    ],
)
@pytest.mark.parametrize("optim", ["brentq", "hybr"])
def test_reconstruction(mode, threshold, optim):
    """Forward processing followed by inversion should recover x."""
    rng = np.random.default_rng(42)

    x = 0.5 * rng.standard_normal(64)

    q = [
        threshold,
        4.0,
        0.001,
        0.050,
        0.001,
        0.080,
        2,
    ]

    y = drc_forward(x, FS, q, mode=mode)
    x_hat = drc_inverse(y, FS, q, mode=mode, optim=optim)

    assert np.all(np.isfinite(x_hat))
    np.testing.assert_allclose(
        x_hat, x,
        rtol=1e-4,
        atol=1e-6,
    )


def test_invalid_ratio():
    """An invalid ratio should raise ValueError."""
    x = np.array([0.1, 0.2])
    q = [-20, 0.5, 0.01, 0.1, 0.01, 0.1, 2]

    with pytest.raises(ValueError):
        drc_forward(x, FS, q)
