# =============================================================================
# Audio Dynamic Range Compression/Expansion
#
# Author:
#   D. Fourer
#
# Reference:
#   H. Sun, D. Fourer, and H. Maaref,
#   "Hybrid Model-Driven Inversion of Audio Dynamic Range Compression
#   and Expansion,"
#   submitted to IEEE/ACM Transactions on Audio, Speech, and Language
#   Processing, 2026.
#
# License:
#   MIT License
#
#   Copyright (c) 2026 D. Fourer
#
#   Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the
#   "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to
#   the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# =============================================================================

from drcinv import drc_forward, drc_inverse
import numpy as np
import soundfile as sf
import time

from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Load MP3
# ============================================================

filename = Path(__file__).resolve().parent / "input.mp3"

mode   = "exp"  # comp or exp
optim  = "brentq"  # brentq (faster) or hybr

x, Fs = sf.read(filename, dtype="float64")

print(f"Sampling rate: {Fs} Hz")
print(f"Signal shape: {x.shape}")

print(f"Running {mode} mode, root finder {optim}...\n")
# ============================================================
# Stereo -> mono (optional)
# ============================================================
def normalize(x):
    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x


if x.ndim == 2:
    x = np.mean(x, axis=1)


# ============================================================
# DRC parameters
# ============================================================


if mode == "comp":
    # compressor
    q = [
        -45.0,
        8.0,
        0.001,   # envelope attack 1 ms
        0.150,
        0.0005,  # gain attack 0.5 ms
        0.200,
        2,
    ]
else:
    # expander
    q = [
    -10.0,   # threshold, -15
    4.0,    # ratio 10
    0.003,
    0.050,
    0.002,
    0.080,
    2,
    ]   
# ============================================================
# Forward DRC
# ============================================================
t0 = time.perf_counter()
y = drc_forward(
    x,
    Fs,
    q,
    mode=mode
)
time_forward_new = time.perf_counter() - t0
print(f"  Forward : {time_forward_new:.3f} s")

# ============================================================
# Inverse DRC
# ============================================================
t0 = time.perf_counter()
x_rec = drc_inverse(
    y,
    Fs,
    q,
    mode=mode,
    optim=optim
)
time_inverse_new = time.perf_counter() - t0
print(f"  Inversion : {time_inverse_new:.3f} s")

# ============================================================
# Reconstruction error
# ============================================================

error = x - x_rec

print(f"Max error : {np.max(np.abs(error)):.3e}")
print(f"RMSE      : {np.sqrt(np.mean(error**2)):.3e}")


# ============================================================
# Save results
# ============================================================

#sf.write("processed.wav", y, Fs)
#sf.write("reconstructed.wav", x_rec, Fs)


sf.write("x_listen_"+ str(mode) +".wav", 0.99 * normalize(x), Fs)
sf.write("y_listen_"+ str(mode) +".wav", 0.99 * normalize(y), Fs)
sf.write("x_rec_listen_"+ str(mode) +".wav", 0.99 * normalize(x_rec), Fs)




# ============================================================
# Time axis
# ============================================================

t = np.arange(len(x)) / Fs


# ============================================================
# Figure 1: Original signal x
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Original signal $x[n]$")
plt.grid(True)
plt.tight_layout()
plt.savefig("x_" + str(mode) + ".pdf", bbox_inches="tight")

# ============================================================
# Figure 2: DRC output y
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(t, y)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("DRC output $y[n]$")
plt.grid(True)
plt.tight_layout()
plt.savefig("y_" + str(mode) + ".pdf", bbox_inches="tight")

# ============================================================
# Figure 3: Reconstructed signal x_hat
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(t, x_rec)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title(r"Reconstructed signal $\hat{x}[n]$")
plt.grid(True)
plt.tight_layout()
plt.savefig("x_hat" + str(mode) + ".pdf", bbox_inches="tight")

# ============================================================
# Figure 4: x and x_hat comparison
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(t, x, label=r"$x[n]$", linewidth=1.2)
plt.plot(t, x_rec, label=r"$\hat{x}[n]$", linewidth=1.0, alpha=0.8)

plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title(r"Original vs reconstructed: $x[n]$ and $\hat{x}[n]$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("x_x_hat_" + str(mode) + ".pdf", bbox_inches="tight")
plt.savefig("x_x_hat_" + str(mode) + ".png", bbox_inches="tight")
# ============================================================
# Figure 5: x and y comparison
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(t, x/np.max(np.abs(x)), label=r"$x[n]$", linewidth=1.2)
plt.plot(t, y/np.max(np.abs(y)), label=r"$y[n]$", linewidth=1.0, alpha=0.8)

plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title(r"Original vs processed rescaled signals: $x[n]$ and $y[n]$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("x_y_" + str(mode) + ".pdf", bbox_inches="tight")
plt.savefig("x_y_" + str(mode) + ".png", bbox_inches="tight")
# ============================================================
# Display all figures
# ============================================================

plt.show()

#debug (should be equal to 0)
print(np.max(np.abs(y) - np.abs(x)))
diff = np.abs(x) - np.abs(y)

print("Max attenuation:", np.max(diff))
print("Mean attenuation:", np.mean(diff))

rms_x = np.sqrt(np.mean(x**2))
rms_y = np.sqrt(np.mean(y**2))

print("RMS x:", rms_x)
print("RMS y:", rms_y)
print("RMS change [dB]:", 20*np.log10(rms_y / rms_x))


duration = len(x) / Fs

RTF_fwd = time_forward_new / duration
RTF_inv = time_inverse_new / duration

print(f"Audio duration  : {duration:.2f} s")
print(f"RTF (forward)   : {RTF_fwd:.4f}")
print(f"RTF (inversion) : {RTF_inv:.4f}")
print(f"Speed (inv.)    : {1 / RTF_inv:.2f} x real-time")
