import numpy as np
from drclibrary import drc_forward, drc_inverse

Fs = 44100
x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(Fs) / Fs)

# [threshold (dBFS), ratio, envelope attack/release (s),
#  gain attack/release (s), detector order]
q = [-20, 4, 0.01, 0.1, 0.01, 0.1, 2]

# Compress and reconstruct
processed = drc_forward(x, Fs, q, mode="comp")
reconstructed = drc_inverse(processed, Fs, q, mode="comp")
