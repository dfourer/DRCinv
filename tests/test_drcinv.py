
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

def test_reconstruction(mode, threshold):
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

    # Forward processing
    y = drc_forward(x, FS, q, mode=mode)
    
    # Check that processing is effective
    relative_processing = np.linalg.norm(x - y) / np.linalg.norm(x)
    assert relative_processing > 1e-3, (
        f"{mode}: processing did not significantly modify the signal"
    )
    
    # Inverse processing (only if forward processing is effective)
    x_hat = drc_inverse(y, FS, q, mode=mode, optim="brentq")
    
    # Check reconstruction quality
    relative_error = np.linalg.norm(x - x_hat) / np.linalg.norm(x)
    
    snr_db = (
        np.inf if relative_error == 0
        else -20 * np.log10(relative_error)
    )
    
    print(
        f"\n{mode}: "
        f"processing={relative_processing:.3e}, "
        f"reconstruction={relative_error:.3e}, "
        f"SNR={snr_db:.2f} dB"
    )
    
    assert np.all(np.isfinite(x_hat))
    assert relative_error < 1e-3


def test_invalid_ratio():
    """An invalid ratio should raise ValueError."""
    x = np.array([0.1, 0.2])
    q = [-20, 0.5, 0.01, 0.1, 0.01, 0.1, 2]

    with pytest.raises(ValueError):
        drc_forward(x, FS, q)

@pytest.mark.parametrize(
    "mode,threshold",
    [
        ("comp", -20.0),
        ("exp", -20.0),
    ],
)


def test_processing_activation(mode, threshold):
    """Check that compression/expansion is active and accurately invertible."""

    # Generate a 0.5 s sine wave with alternating amplitude levels
    duration = 0.5
    t = np.arange(int(FS * duration)) / FS

    # Amplitude changes every 125 ms
    amplitude = np.select(
        [
            t < 0.125,
            t < 0.250,
            t < 0.375,
        ],
        [
            0.02,
            0.8,
            0.02,
        ],
        default=0.8,
    )

    x = amplitude * np.sin(2 * np.pi * 440 * t)

    # DRC parameters
    q = [
        threshold,
        4.0,
        0.001,
        0.050,
        0.001,
        0.050,
        2,
    ]

    # Forward processing
    y = drc_forward(x, FS, q, mode=mode)

    # Ensure compression/expansion actually modifies the signal
    relative_processing = np.linalg.norm(x - y) / np.linalg.norm(x)

    assert relative_processing > 1e-3, (
        f"{mode}: processing did not significantly modify the signal"
    )

    # Inverse processing
    x_hat = drc_inverse(y, FS, q, mode=mode)

    # Reconstruction quality
    assert np.all(np.isfinite(x_hat)), (
        f"{mode}: reconstruction contains NaN or Inf"
    )

    relative_error = np.linalg.norm(x - x_hat) / np.linalg.norm(x)

    snr_db = (
        np.inf
        if relative_error == 0
        else -20 * np.log10(relative_error)
    )

    print(
        f"\n{mode}: "
        f"processing={relative_processing:.3e}, "
        f"reconstruction={relative_error:.3e}, "
        f"SNR={snr_db:.2f} dB"
    )

    assert relative_error < 1e-3, (
        f"{mode}: reconstruction error too large "
        f"({relative_error:.3e}, SNR={snr_db:.2f} dB)"
    )
