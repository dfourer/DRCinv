# Dynamic Range Control Model

DRCinv supports dynamic range compression and expansion.

## Model parameters

The parameter vector is:

\[
q = [L, R, \tau_{v,a}, \tau_{v,r}, \tau_{g,a}, \tau_{g,r}, p]
\]

where:

- `L`: threshold in dBFS
- `R`: compression or expansion ratio
- `tau_v_att`: detector attack time in seconds
- `tau_v_rel`: detector release time in seconds
- `tau_g_att`: gain attack time in seconds
- `tau_g_rel`: gain release time in seconds
- `p`: detector order

## Processing modes

- `mode="comp"`: downward compression
- `mode="exp"`: downward expansion 
