# DRCinv — Audio Compression/Expansion Model-Based Inversion

[![PyPI version](https://img.shields.io/pypi/v/drcinv.svg)](https://pypi.org/project/drcinv/)
[![Documentation Status](https://readthedocs.org/projects/drcinv/badge/?version=latest)](https://drcinv.readthedocs.io/en/latest/)

**DRCinv** is a Python implementation of forward dynamic range compression and expansion, together with model-based inversion for audio signal reconstruction.

This repository accompanies the research work:

> H. Sun, D. Fourer, and H. Maaref, **“Hybrid Model-Driven Inversion of Audio Dynamic Range Compression and Expansion,”** submitted to *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2026.

**Project page:** https://hdrc-inv.github.io/
**Source code:** https://github.com/dfourer/DRCinv

> **Publication status:** The project page describes the paper as submitted. Update the citation and add the DOI when the final bibliographic details are available.

## Overview

Dynamic range compression and expansion are widely used to modify the amplitude dynamics of audio signals. DRCinv provides functions to simulate these processes and estimate the original signal from processed audio using a model-based inversion approach.

The implementation exposes two principal functions:

- `drc_forward`: apply dynamic range compression or expansion.
- `drc_inverse`: estimate the original signal from the processed signal, given the model parameters.

## Documentation

The full technical documentation, including the DRC model,
model-based inversion, API reference, and examples, is available at:

**[DRCinv Documentation — Read the Docs](https://drcinv.readthedocs.io/en/latest/)**

## Installation

Install from the project directory during development:

```bash
python -m pip install -e .
```

Once the `drcinv` distribution is published on PyPI, it can be installed with:

```bash
pip install drcinv
```

The core package requires NumPy, SciPy, and Numba. Install the optional PyTorch
dependency when using the MEE parameter estimator:

```bash
pip install "drcinv[param]"
```

## Quick start

```python
import numpy as np
from drcinv import drc_forward, drc_inverse

Fs = 44100
x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(Fs) / Fs)

# [threshold (dBFS), ratio, envelope attack/release (s),
#  gain attack/release (s), detector order]
q = [-20, 4, 0.01, 0.1, 0.01, 0.1, 2]

# Compress and reconstruct
processed = drc_forward(x, Fs, q, mode="comp")
reconstructed = drc_inverse(processed, Fs, q, mode="comp")
```

The parameter vector is `q = [L, R, tau_v_att, tau_v_rel, tau_g_att, tau_g_rel, p]`, where `L` is the threshold in dBFS, `R` is the ratio, the four `tau` values are time constants in seconds, and `p` is the detector order (1 for peak, 2 for RMS-like detection).

Set `mode="comp"` for downward compression or `mode="exp"` for downward expansion. The inverse uses the Numba-compiled bounded Brent solver; `optim="brentq"` is the only supported setting and the default.

## Estimate unknown compression parameters with MEE

`drcparam` integrates the Music Effects Encoder (MEE) from the research code.
The default model estimates the six continuous compression controls and adds
the fixed RMS-like detector order, returning the seven-element vector expected
by `drc_inverse`:

```python
import soundfile as sf
from drcinv.drcparam import MEEParameterEstimator
from drcinv import drc_inverse

y, sample_rate = sf.read("compressed.wav", always_2d=False)

# Reuse one estimator when processing multiple signals.
estimator = MEEParameterEstimator(device="auto")
parameters = estimator.estimate(y, sample_rate)
q = parameters.as_vector()
x_hat = drc_inverse(y, sample_rate, q, mode="comp")

print(parameters.as_dict())
print(parameters.normalized)
```

The input may use any native sample rate. It is converted to mono, then
center-cropped or zero-padded to the five-second duration used in training;
it is not resampled or peak-normalized. Time constants returned by the public
API are in seconds and can be passed directly to `drcinv`.

The ~337 MiB checkpoint is not included in the PyPI wheel. On first inference,
it is downloaded from the `v0.2.0` GitHub Release asset
`MEE_compression_large.pt`, verified with SHA-256, and cached under
`$XDG_CACHE_HOME/drcinv` (or `~/.cache/drcinv`). For offline use:

```python
estimator = MEEParameterEstimator(
    checkpoint_path="/path/to/MEE_compression_large.pt",
    device="cpu",
)
```

Before publishing `v0.2.0`, upload the checkpoint copied from
`pretrained/DRC/MedleyDB/MEE/MEE_large.pt` with the exact asset name above.
Its expected SHA-256 is
`c471092038d81a0f4a23ea8daec6f98afb5f0936b0924c805dc9b9343f1c41e2`.

## Black-box parameter estimation

`drcparam` also provides derivative-free estimation in AQFeature space. Each
candidate parameter vector is inverted, represented by selected audio-quality
features, and compared with a dry-audio reference point using a regularized
Mahalanobis distance.

The feature-selection method is intentionally not chosen by the library. Pass
one of `"mrmr"`, `"mutual_information"`, `"binary_fisher"`, or
`"inverse_fisher"` explicitly:

```python
from drcinv.drcparam import BlackBoxParameterEstimator

estimator = BlackBoxParameterEstimator(
    reference_method="mutual_information",  # required
    process_type="compression",
    optimizer="powell",
    max_evaluations=100,
)
result = estimator.estimate_result(y, sample_rate)
q = result.parameters.as_vector()

print(q)
print(result.objective, result.success, result.message)
```

The packaged reference points cover all four feature-selection methods for
both compression and expansion. `powell` and `differential_evolution` use
SciPy. `patternsearch` requires `drcinv[patternsearch]`; `bayesopt` requires
`drcinv[bayesopt]`.

For a one-shot call, select either estimator with `method`:

```python
from drcinv.drcparam import estimate_drc_parameters

q_mee = estimate_drc_parameters(y, sample_rate, method="mee")
q_black_box = estimate_drc_parameters(
    y,
    sample_rate,
    method="black_box",
    reference_method="mrmr",
    process_type="compression",
)
```

## Illustrations

The following plots reproduce the visual material linked from the [project webpage](https://fourer.fr/drc/). Image paths are relative to this README and assume that the corresponding PNG files are stored in `docs/`.

### Compressor

**Original vs. processed signal**

![Original and compressed signals](https://fourer.fr/drc/x_y_comp.png)

**Original vs. reconstructed signal**

![Original and reconstructed signals after compression](https://fourer.fr/drc/x_x_hat_comp.png)

### Expander

**Original vs. processed signal**

![Original and expanded signals](https://fourer.fr/drc/x_y_exp.png)

**Original vs. reconstructed signal**

![Original and reconstructed signals after expansion](https://fourer.fr/drc/x_x_hat_exp.png)

## Full audio demonstration

The repository includes [`examples/demo_drcinv.py`](examples/demo_drcinv.py), a complete audio example that reads [`examples/input.mp3`](examples/input.mp3), applies forward DRC, reconstructs the original signal, and evaluates reconstruction accuracy and processing speed.

Install the library locally and the dependencies required by the example:

```bash
python -m pip install -e .
python -m pip install soundfile matplotlib
```

Run from the repository root:

```bash
python examples/demo_drcinv.py
```

The example loads the MP3 relative to its own location, so it can find `examples/input.mp3` regardless of the working directory. Stereo audio is converted to mono by averaging the channels. The code uses a peak-normalization helper for the listening files; the actual forward and inverse computations use the unnormalized signal.

### Choose compression or expansion

At the beginning of `examples/demo_drcinv.py`, choose the processing mode and inverse solver:

```python
mode = "exp"     # "comp" (compression) or "exp" (expansion)
optim = "brentq" # Numba-compiled bounded Brent solver
```

The example defines separate parameter vectors for compression and expansion. The **compression** settings are:

```python
q = [-45.0, 8.0, 0.001, 0.150, 0.0005, 0.200, 2]
```

The **expansion** settings are:

```python
q = [-10.0, 4.0, 0.003, 0.050, 0.002, 0.080, 2]
```

### Forward processing and reconstruction

The core of the example is:

```python
from drcinv import drc_forward, drc_inverse

# x and Fs are loaded from the input audio file; q and mode are set above.
y = drc_forward(x, Fs, q, mode=mode)
x_rec = drc_inverse(y, Fs, q, mode=mode, optim=optim)

error = x - x_rec
print(f"Max error: {np.max(np.abs(error)):.3e}")
print(f"RMSE: {np.sqrt(np.mean(error**2)):.3e}")
```

Here `x` is the original audio, `y` the compressed or expanded audio, and `x_rec` the reconstructed signal. The full example also reports forward and inverse computation times, maximum and root-mean-square reconstruction errors, changes in RMS level, real-time factors (RTF), and inverse processing speed relative to real time.

### Generated audio and figures

Running the demonstration writes the following listening files to the **current working directory** (where the command is launched):

- `x_listen_comp.wav`, `y_listen_comp.wav`, `x_rec_listen_comp.wav` in compression mode;
- `x_listen_exp.wav`, `y_listen_exp.wav`, `x_rec_listen_exp.wav` in expansion mode.

It also saves individual plots for the original, processed, and reconstructed waveforms, plus comparisons between the original and reconstructed signals and between normalized original and processed signals. The comparison plots are saved as both PDF and PNG; the other plots are saved as PDF. The script calls `plt.show()` to display the figures interactively.

The PNGs under [`docs/`](docs/) are the figures included in the **Illustrations** section above; the demonstration does not automatically write to `docs/`.

### Automated tests

The demonstration is an executable example, not a pytest test. Automated tests belong in `tests/test_drcinv.py` and should contain functions named `test_*`:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/ -v
```

## Citation

If you use DRCinv in your research, please cite the following preprint:

H. Sun, D. Fourer, and H. Maaref, **"Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics,"** *arXiv:2411.04337*, 2024.

**Preprint:** [arXiv:2411.04337](https://arxiv.org/abs/2411.04337)

**Publication status:** A related manuscript is currently under review at *IEEE/ACM Transactions on Audio, Speech, and Language Processing* (minor revision).

### BibTeX

```bibtex
@misc{sun2024neural,
  title={Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics},
  author={Sun, Haoran and Fourer, Dominique and Maaref, Hichem},
  year={2024},
  eprint={2411.04337},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  doi={10.48550/arXiv.2411.04337}
}
```

## License

See the `LICENSE` file for the terms applicable to this software. Ensure that the license file is included before public distribution.

## Links

- [Research project page](https://fourer.fr/drc/)
- [GitHub repository](https://github.com/dfourer/DRCinv)
 
