# =============================================================================
# Fast Audio Dynamic Range Compression/Expansion for DRCinv
# =============================================================================

import numpy as np
from numba import njit


def _parse_q(q):
    """
    q = [L, R, tau_v_att, tau_v_rel, tau_g_att, tau_g_rel, p]

    L             : threshold in dBFS
    R             : ratio >= 1
    tau_v_att     : envelope attack time [s]
    tau_v_rel     : envelope release time [s]
    tau_g_att     : gain attack time [s]
    tau_g_rel     : gain release time [s]
    p             : detector order, 1=peak, 2=RMS-like
    """
    if isinstance(q, dict):
        return (
            q["L"],
            q["R"],
            q["tau_v_att"],
            q["tau_v_rel"],
            q["tau_g_att"],
            q["tau_g_rel"],
            q["p"],
        )

    if len(q) != 7:
        raise ValueError(
            "q must be [L, R, tau_v_att, tau_v_rel, "
            "tau_g_att, tau_g_rel, p]"
        )

    return tuple(q)


def _smoothing_coeff(Fs, tau):
    if tau <= 0:
        raise ValueError("Time constants must be strictly positive.")

    return 1.0 - np.exp(-2.2 / (Fs * tau))


_MODE_COMP = 0
_MODE_EXP = 1
_XTOL = 1.0e-14
_RTOL = 1.0e-12
_MAXITER = 100


@njit(cache=False)
def _drc_forward_kernel(
    x,
    mode_code,
    p,
    l,
    kappa,
    slope,
    beta_att,
    beta_rel,
    gamma_att,
    gamma_rel,
):
    x_tilde_prev = 0.0
    g_prev = 1.0

    y = np.empty_like(x)

    for n in range(x.size):
        xn = x[n]
        v_prev = x_tilde_prev ** (1.0 / p)

        if abs(xn) > v_prev:
            beta = beta_att
        else:
            beta = beta_rel

        beta_bar = 1.0 - beta
        x_tilde = beta * abs(xn) ** p + beta_bar * x_tilde_prev
        v = x_tilde ** (1.0 / p)

        if mode_code == _MODE_COMP:
            if v > l:
                f = kappa * v ** (-slope)
            else:
                f = 1.0
        else:
            if 0.0 < v < l:
                f = kappa * v**slope
            else:
                f = 1.0

        if f < g_prev:
            gamma = gamma_att
        else:
            gamma = gamma_rel

        gamma_bar = 1.0 - gamma
        g = gamma * f + gamma_bar * g_prev

        y[n] = xn * g

        x_tilde_prev = x_tilde
        g_prev = g

    return y


def drc_forward(x, Fs, q, mode="comp"):
    """
    Forward feedforward DRC.

    Parameters
    ----------
    x : array_like
        Input signal.
    Fs : float
        Sampling frequency [Hz].
    q : sequence or dict
        [L, R, tau_v_att, tau_v_rel,
         tau_g_att, tau_g_rel, p]
    mode : {"comp", "exp"}
        "comp" : downward compression
        "exp"  : downward expansion

    Returns
    -------
    y : ndarray
        Processed signal.
    """

    x = np.asarray(x, dtype=float)

    L, R, tau_v_att, tau_v_rel, \
        tau_g_att, tau_g_rel, p = _parse_q(q)

    if mode not in ("comp", "exp"):
        raise ValueError("mode must be 'comp' or 'exp'.")

    if R < 1:
        raise ValueError("R must satisfy R >= 1.")

    if p not in (1, 2):
        raise ValueError("p must be 1 or 2.")

    # Identity mapping
    if R == 1:
        return x.copy()

    l = 10.0 ** (L / 20.0)

    if mode == "comp":
        slope = 1.0 - 1.0 / R
        kappa = l**slope
        mode_code = _MODE_COMP
    else:
        slope = R - 1.0
        kappa = l ** (-slope)
        mode_code = _MODE_EXP

    beta_att = _smoothing_coeff(Fs, tau_v_att)
    beta_rel = _smoothing_coeff(Fs, tau_v_rel)
    gamma_att = _smoothing_coeff(Fs, tau_g_att)
    gamma_rel = _smoothing_coeff(Fs, tau_g_rel)

    return _drc_forward_kernel(
        np.ascontiguousarray(x),
        mode_code,
        int(p),
        l,
        kappa,
        slope,
        beta_att,
        beta_rel,
        gamma_att,
        gamma_rel,
    )


@njit(cache=False, inline="always")
def _zeta(
    v,
    mode_code,
    p,
    kappa,
    slope,
    gamma,
    gamma_bar,
    g_prev,
    minimum_v_p,
    beta_abs_y_p,
):
    if mode_code == _MODE_COMP:
        f = kappa * v ** (-slope)
    else:
        f = kappa * v**slope

    current_gain = gamma * f + gamma_bar * g_prev
    return current_gain**p * (v**p - minimum_v_p) - beta_abs_y_p


@njit(cache=False)
def _brentq(
    v_low,
    v_high,
    z_low,
    z_high,
    mode_code,
    p,
    kappa,
    slope,
    gamma,
    gamma_bar,
    g_prev,
    minimum_v_p,
    beta_abs_y_p,
):
    """Bounded Brent-Dekker solve matching SciPy's stopping convention."""

    if z_low == 0.0:
        return v_low
    if z_high == 0.0:
        return v_high

    x_pre = v_low
    x_cur = v_high
    x_blk = 0.0
    f_pre = z_low
    f_cur = z_high
    f_blk = 0.0
    s_pre = x_cur - x_pre
    s_cur = s_pre

    for _ in range(_MAXITER):
        opposite_signs = (
            (f_pre < 0.0 and f_cur > 0.0)
            or (f_pre > 0.0 and f_cur < 0.0)
        )

        if f_pre != 0.0 and f_cur != 0.0 and opposite_signs:
            x_blk = x_pre
            f_blk = f_pre
            s_pre = x_cur - x_pre
            s_cur = s_pre

        if abs(f_blk) < abs(f_cur):
            x_pre = x_cur
            x_cur = x_blk
            x_blk = x_pre
            f_pre = f_cur
            f_cur = f_blk
            f_blk = f_pre

        delta = 0.5 * (_XTOL + _RTOL * abs(x_cur))
        s_bis = 0.5 * (x_blk - x_cur)

        if f_cur == 0.0 or abs(s_bis) < delta:
            return x_cur

        if abs(s_pre) > delta and abs(f_cur) < abs(f_pre):
            if x_pre == x_blk:
                # Secant interpolation.
                s_try = -f_cur * (x_cur - x_pre) / (f_cur - f_pre)
            else:
                # Inverse quadratic interpolation.
                d_pre = (f_pre - f_cur) / (x_pre - x_cur)
                d_blk = (f_blk - f_cur) / (x_blk - x_cur)
                s_try = (
                    -f_cur
                    * (f_blk * d_blk - f_pre * d_pre)
                    / (d_blk * d_pre * (f_blk - f_pre))
                )

            if 2.0 * abs(s_try) < min(
                abs(s_pre), 3.0 * abs(s_bis) - delta
            ):
                s_pre = s_cur
                s_cur = s_try
            else:
                s_pre = s_bis
                s_cur = s_bis
        else:
            s_pre = s_bis
            s_cur = s_bis

        x_pre = x_cur
        f_pre = f_cur

        if abs(s_cur) > delta:
            x_cur += s_cur
        elif s_bis > 0.0:
            x_cur += delta
        else:
            x_cur -= delta

        f_cur = _zeta(
            x_cur,
            mode_code,
            p,
            kappa,
            slope,
            gamma,
            gamma_bar,
            g_prev,
            minimum_v_p,
            beta_abs_y_p,
        )

    # The reference implementation raises on non-convergence. Reaching this
    # return would require an equally pathological bracket; return the best
    # bounded iterate so the compiled kernel remains usable from nopython.
    return x_cur


@njit(cache=False)
def _drc_inverse_kernel(
    y,
    mode_code,
    p,
    l,
    kappa,
    slope,
    beta_att,
    beta_rel,
    gamma_att,
    gamma_rel,
):
    x_hat = np.empty_like(y)

    beta_att_bar = 1.0 - beta_att
    beta_rel_bar = 1.0 - beta_rel
    gamma_att_bar = 1.0 - gamma_att
    gamma_rel_bar = 1.0 - gamma_rel
    l_p = l**p
    tiny = np.finfo(np.float64).tiny

    x_tilde_prev = 0.0
    g_prev = 1.0

    for n in range(y.size):
        yn = y[n]
        abs_y = abs(yn)

        if abs_y == 0.0:
            beta = beta_rel
            beta_bar = beta_rel_bar
            v = (beta_bar * x_tilde_prev) ** (1.0 / p)

            if mode_code == _MODE_COMP:
                if v > l:
                    f = kappa * v ** (-slope)
                else:
                    f = 1.0
            else:
                if 0.0 < v < l:
                    f = kappa * v**slope
                else:
                    f = 1.0

            if f < g_prev:
                gamma = gamma_att
                gamma_bar = gamma_att_bar
            else:
                gamma = gamma_rel
                gamma_bar = gamma_rel_bar

            g_prev = gamma * f + gamma_bar * g_prev
            x_hat[n] = 0.0
            x_tilde_prev = v**p
            continue

        previous_envelope = x_tilde_prev ** (1.0 / p)
        if abs_y > g_prev * previous_envelope:
            beta = beta_att
            beta_bar = beta_att_bar
        else:
            beta = beta_rel
            beta_bar = beta_rel_bar

        if mode_code == _MODE_COMP:
            radicand = (
                (kappa / g_prev) ** (p / slope)
                - beta_bar * x_tilde_prev
            )
            if radicand < 0.0:
                gamma = gamma_att
                gamma_bar = gamma_att_bar
            else:
                attack_threshold = (
                    g_prev * (radicand / beta) ** (1.0 / p)
                )
                if abs_y > attack_threshold:
                    gamma = gamma_att
                    gamma_bar = gamma_att_bar
                else:
                    gamma = gamma_rel
                    gamma_bar = gamma_rel_bar
        else:
            radicand = (
                (g_prev / kappa) ** (p / slope)
                - beta_bar * x_tilde_prev
            )
            if radicand < 0.0:
                gamma = gamma_rel
                gamma_bar = gamma_rel_bar
            else:
                attack_threshold = (
                    g_prev * (radicand / beta) ** (1.0 / p)
                )
                if abs_y < attack_threshold:
                    gamma = gamma_att
                    gamma_bar = gamma_att_bar
                else:
                    gamma = gamma_rel
                    gamma_bar = gamma_rel_bar

        minimum_v_p = beta_bar * x_tilde_prev

        if l_p < minimum_v_p:
            active = mode_code == _MODE_COMP
        else:
            if 1.0 < g_prev:
                gamma_t = gamma_att
            else:
                gamma_t = gamma_rel

            threshold = (
                gamma_t + (1.0 - gamma_t) * g_prev
            ) * ((l_p - minimum_v_p) / beta) ** (1.0 / p)

            if mode_code == _MODE_COMP:
                active = abs_y > threshold
            else:
                active = abs_y < threshold

        if not active:
            f = 1.0
            if f < g_prev:
                gamma_inactive = gamma_att
                gamma_inactive_bar = gamma_att_bar
            else:
                gamma_inactive = gamma_rel
                gamma_inactive_bar = gamma_rel_bar

            g = gamma_inactive + gamma_inactive_bar * g_prev
            xn = yn / g
            x_tilde = beta * abs(xn) ** p + minimum_v_p

            x_hat[n] = xn
            x_tilde_prev = x_tilde
            g_prev = g
            continue

        beta_abs_y_p = beta * abs_y**p
        v_physical_min = minimum_v_p ** (1.0 / p)

        if mode_code == _MODE_COMP:
            v_low = max(l, v_physical_min)
            z_low = _zeta(
                v_low,
                mode_code,
                p,
                kappa,
                slope,
                gamma,
                gamma_bar,
                g_prev,
                minimum_v_p,
                beta_abs_y_p,
            )

            if z_low >= 0.0:
                v_star = v_low
            else:
                v_high = max(2.0 * v_low, 2.0 * l, v_physical_min + 1.0)
                z_high = _zeta(
                    v_high,
                    mode_code,
                    p,
                    kappa,
                    slope,
                    gamma,
                    gamma_bar,
                    g_prev,
                    minimum_v_p,
                    beta_abs_y_p,
                )

                for _ in range(100):
                    if z_high >= 0.0:
                        break
                    v_high *= 2.0
                    z_high = _zeta(
                        v_high,
                        mode_code,
                        p,
                        kappa,
                        slope,
                        gamma,
                        gamma_bar,
                        g_prev,
                        minimum_v_p,
                        beta_abs_y_p,
                    )

                if z_high >= 0.0:
                    v_star = _brentq(
                        v_low,
                        v_high,
                        z_low,
                        z_high,
                        mode_code,
                        p,
                        kappa,
                        slope,
                        gamma,
                        gamma_bar,
                        g_prev,
                        minimum_v_p,
                        beta_abs_y_p,
                    )
                else:
                    v_star = v_high
        else:
            v_low = max(v_physical_min, tiny)
            v_high = l
            z_low = _zeta(
                v_low,
                mode_code,
                p,
                kappa,
                slope,
                gamma,
                gamma_bar,
                g_prev,
                minimum_v_p,
                beta_abs_y_p,
            )
            z_high = _zeta(
                v_high,
                mode_code,
                p,
                kappa,
                slope,
                gamma,
                gamma_bar,
                g_prev,
                minimum_v_p,
                beta_abs_y_p,
            )

            if z_low >= 0.0:
                v_star = v_low
            elif z_high <= 0.0:
                v_star = v_high
            else:
                v_star = _brentq(
                    v_low,
                    v_high,
                    z_low,
                    z_high,
                    mode_code,
                    p,
                    kappa,
                    slope,
                    gamma,
                    gamma_bar,
                    g_prev,
                    minimum_v_p,
                    beta_abs_y_p,
                )

        input_power = (v_star**p - minimum_v_p) / beta
        input_power = max(input_power, 0.0)
        x_magnitude = input_power ** (1.0 / p)
        if yn < 0.0:
            xn = -x_magnitude
        else:
            xn = x_magnitude

        if mode_code == _MODE_COMP:
            f = kappa * v_star ** (-slope)
        else:
            f = kappa * v_star**slope

        g = gamma * f + gamma_bar * g_prev
        x_hat[n] = xn
        x_tilde_prev = v_star**p
        g_prev = g

    return x_hat


def drc_inverse(
    y,
    Fs,
    q,
    mode="comp",
    optim="brentq",
    if_normalize=False,
):
    """Invert feedforward DRC with a compiled recursive Brent kernel.

    The bounded Brent solve and the state recursion are compiled with Numba.
    Only ``optim="brentq"`` is supported.
    """

    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional signal.")

    L, R, tau_v_att, tau_v_rel, tau_g_att, tau_g_rel, p = _parse_q(q)

    if mode not in ("comp", "exp"):
        raise ValueError("mode must be 'comp' or 'exp'.")
    if optim != "brentq":
        raise ValueError("drcinv only supports optim='brentq'.")
    if R < 1:
        raise ValueError("R must satisfy R >= 1.")
    if p not in (1, 2):
        raise ValueError("p must be 1 or 2.")

    if R == 1:
        x_hat = y.copy()
    else:
        l = 10.0 ** (L / 20.0)
        if mode == "comp":
            slope = 1.0 - 1.0 / R
            kappa = l**slope
            mode_code = _MODE_COMP
        else:
            slope = R - 1.0
            kappa = l ** (-slope)
            mode_code = _MODE_EXP

        beta_att = _smoothing_coeff(Fs, tau_v_att)
        beta_rel = _smoothing_coeff(Fs, tau_v_rel)
        gamma_att = _smoothing_coeff(Fs, tau_g_att)
        gamma_rel = _smoothing_coeff(Fs, tau_g_rel)

        x_hat = _drc_inverse_kernel(
            np.ascontiguousarray(y),
            mode_code,
            int(p),
            l,
            kappa,
            slope,
            beta_att,
            beta_rel,
            gamma_att,
            gamma_rel,
        )

    if if_normalize:
        peak = np.max(np.abs(x_hat))
        if peak > 0.0:
            x_hat = x_hat / peak

    return x_hat
