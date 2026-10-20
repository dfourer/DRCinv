
import numpy as np
import pytest

from drclibrary import drc_forward, drc_inverse


@pytest.mark.parametrize("mode", ["comp", "exp"])
def test_identity(mode):
    Fs = 44100
    x = np.array([0.1, -0.2, 0.3, -0.4])

    q = [-20, 1, 0.01, 0.1, 0.01, 0.1, 2]

    y = drc_forward(x, Fs, q, mode=mode)
    x_hat = drc_inverse(y, Fs, q, mode=mode)

    np.testing.assert_allclose(y, x)
    np.testing.assert_allclose(x_hat, x)


def test_invalid_ratio():
    x = np.array([0.1, 0.2])
    q = [-20, 0.5, 0.01, 0.1, 0.01, 0.1, 2]

    with pytest.raises(ValueError):
        drc_forward(x, 44100, q)
 
