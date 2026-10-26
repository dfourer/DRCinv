# Examples

## Compression and reconstruction

```python
import numpy as np
from drcinv import drc_forward, drc_inverse

Fs = 44100
t = np.arange(Fs) / Fs

x = 0.8 * np.sin(2 * np.pi * 440 * t)

q = [-20, 4, 0.01, 0.1, 0.01, 0.1, 2]

y = drc_forward(x, Fs, q, mode="comp")
x_hat = drc_inverse(y, Fs, q, mode="comp")

relative_error = np.linalg.norm(x - x_hat) / np.linalg.norm(x)

print(f"Relative error: {relative_error:.3e}")
```

For expansion, use `mode="exp"` in both calls.
