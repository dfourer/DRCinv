# =============================================================================
# DRC parameter estimation with MEE or black-box optimization
#
# This example follows demo_drcinv.py: it loads input.mp3, applies a known DRC,
# estimates the unknown controls from the processed signal, and reconstructs
# the original signal with the estimated controls.
# =============================================================================

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from drcinv import drc_forward, drc_inverse, drcparam


# ============================================================
# User settings
# ============================================================

# Choose "mee" or "black_box".
estimation_method = "mee"

# The released MEE checkpoint estimates compression parameters only.
mode = "comp"  # "comp" or "exp" (black-box only for "exp")

# MEE settings. Set to a local .pt file for offline use; None downloads the
# checkpoint from the GitHub Release and caches it on first use.
mee_checkpoint = None
mee_device = "auto"

# Black-box settings. reference_method is intentionally explicit: the library
# does not select mRMR or mutual information on the user's behalf.
reference_method = "mrmr"  # or "mutual_information", "binary_fisher", "inverse_fisher"
black_box_optimizer = "powell"
max_evaluations = 100


# ============================================================
# Helpers
# ============================================================

def normalize(audio):
    """Peak-normalize a listening copy without changing estimation input."""

    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def print_parameters(title, values):
    names = (
        "threshold [dBFS]",
        "ratio",
        "envelope attack [s]",
        "envelope release [s]",
        "gain attack [s]",
        "gain release [s]",
        "detector order p",
    )
    print(title)
    for name, value in zip(names, values):
        print(f"  {name:22s}: {value:.8g}")


# ============================================================
# Load audio
# ============================================================

filename = Path(__file__).resolve().parent / "input.mp3"
x, Fs = sf.read(filename, dtype="float64")

if x.ndim == 2:
    x = np.mean(x, axis=1)

print(f"Sampling rate: {Fs} Hz")
print(f"Signal shape: {x.shape}")
print(f"Estimator: {estimation_method}\n")


# ============================================================
# Create an example processed signal
# ============================================================

if mode == "comp":
    q_true = np.asarray(
        [-45.0, 8.0, 0.001, 0.150, 0.0005, 0.200, 2.0]
    )
else:
    q_true = np.asarray(
        [-10.0, 4.0, 0.003, 0.050, 0.002, 0.080, 2.0]
    )

y = drc_forward(x, Fs, q_true, mode=mode)


# ============================================================
# Estimate the seven DRC parameters from y
# ============================================================

t0 = time.perf_counter()

if estimation_method == "mee":
    if mode != "comp":
        raise ValueError("The released MEE checkpoint supports compression only.")

    estimator = drcparam.MEEParameterEstimator(
        checkpoint_path=mee_checkpoint,
        device=mee_device,
    )
    estimate = estimator.estimate(y, Fs)
    q_est = estimate.as_vector()

    print("Raw normalized MEE outputs:")
    print(np.asarray(estimate.normalized))

elif estimation_method == "black_box":
    process_type = "compression" if mode == "comp" else "expansion"
    estimator = drcparam.BlackBoxParameterEstimator(
        reference_method=reference_method,  # required; no library default
        process_type=process_type,
        optimizer=black_box_optimizer,
        max_evaluations=max_evaluations,
    )
    result = estimator.estimate_result(y, Fs)
    q_est = result.parameters.as_vector()

    print(f"Objective: {result.objective:.8g}")
    print(f"Optimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")

else:
    raise ValueError("estimation_method must be 'mee' or 'black_box'.")

estimation_time = time.perf_counter() - t0

print_parameters("\nTrue parameters:", q_true)
print_parameters("\nEstimated parameters:", q_est)
print(f"\nEstimation time: {estimation_time:.3f} s")


# ============================================================
# Reconstruct with the estimated parameters
# ============================================================

x_rec = drc_inverse(y, Fs, q_est, mode=mode)
error = x - x_rec

print(f"Max reconstruction error: {np.max(np.abs(error)):.3e}")
print(f"Reconstruction RMSE:       {np.sqrt(np.mean(error**2)):.3e}")


# ============================================================
# Save listening files
# ============================================================

suffix = f"{estimation_method}_{mode}"
sf.write(f"x_listen_{suffix}.wav", 0.99 * normalize(x), Fs)
sf.write(f"y_listen_{suffix}.wav", 0.99 * normalize(y), Fs)
sf.write(f"x_rec_listen_{suffix}.wav", 0.99 * normalize(x_rec), Fs)


# ============================================================
# Plot waveform and parameter comparisons
# ============================================================

t = np.arange(len(x)) / Fs

plt.figure(figsize=(12, 4))
plt.plot(t, x, label=r"$x[n]$", linewidth=1.2)
plt.plot(t, x_rec, label=r"$\hat{x}[n]$", linewidth=1.0, alpha=0.8)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Original vs. reconstruction using estimated DRC parameters")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"x_x_hat_{suffix}.pdf", bbox_inches="tight")
plt.savefig(f"x_x_hat_{suffix}.png", bbox_inches="tight")

parameter_labels = ["L", "R", "tau_v_att", "tau_v_rel", "tau_g_att", "tau_g_rel"]
indices = np.arange(len(parameter_labels))
width = 0.38

# The six controls have different units and scales, so this figure compares
# their relative error rather than placing raw values on one shared axis.
denominator = np.maximum(np.abs(q_true[:6]), np.finfo(float).eps)
relative_error = 100.0 * (q_est[:6] - q_true[:6]) / denominator

plt.figure(figsize=(10, 4))
plt.bar(indices, relative_error, width=width)
plt.axhline(0.0, color="black", linewidth=0.8)
plt.xticks(indices, parameter_labels)
plt.ylabel("Signed relative error [%]")
plt.title("DRC parameter-estimation error")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(f"parameter_error_{suffix}.pdf", bbox_inches="tight")
plt.savefig(f"parameter_error_{suffix}.png", bbox_inches="tight")

plt.show()
