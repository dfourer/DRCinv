# Model-Based Inversion

DRCinv estimates the original audio signal from its
compressed or expanded version.

## Forward model

The forward model is evaluated with:

```python
y = drc_forward(x, Fs, q, mode="comp")
```

## Inverse model

The inverse model is evaluated with:

```python
x_hat = drc_inverse(y, Fs, q, mode="comp")
```

## Numerical solvers

Two optimization methods are available:

- `optim="hybr"`
- `optim="brentq"`

## Reconstruction accuracy

The relative reconstruction error is:

\[
\varepsilon =
\frac{\|x-\hat{x}\|_2}{\|x\|_2}
\]

The reconstruction SNR is:

\[
\mathrm{SNR}_{rec} = -20\log_{10}(\varepsilon)
\]
