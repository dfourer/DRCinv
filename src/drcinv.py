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

import numpy as np
from scipy.optimize import brentq
from scipy.optimize import root

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

    # ---------------------------------------------------------
    # Static-curve parameters
    # ---------------------------------------------------------
    l = 10.0 ** (L / 20.0)

    S_c = 1.0 - 1.0 / R
    kappa_c = l ** S_c

    S_e = R - 1.0
    kappa_e = l ** (-S_e)

    # ---------------------------------------------------------
    # Envelope smoothing coefficients
    # ---------------------------------------------------------
    beta_att = _smoothing_coeff(Fs, tau_v_att)
    beta_rel = _smoothing_coeff(Fs, tau_v_rel)

    # ---------------------------------------------------------
    # Gain smoothing coefficients
    # ---------------------------------------------------------
    gamma_att = _smoothing_coeff(Fs, tau_g_att)
    gamma_rel = _smoothing_coeff(Fs, tau_g_rel)

    # ---------------------------------------------------------
    # Initial states
    # ---------------------------------------------------------
    x_tilde_prev = 0.0
    g_prev = 1.0

    y = np.empty_like(x)

    # ---------------------------------------------------------
    # Sample-by-sample processing
    # ---------------------------------------------------------
    for n, xn in enumerate(x):

        # Previous envelope
        v_prev = x_tilde_prev ** (1.0 / p)

        # Envelope attack / release
        if abs(xn) > v_prev:
            beta = beta_att
        else:
            beta = beta_rel

        beta_bar = 1.0 - beta

        # Envelope update
        x_tilde = (
            beta * abs(xn) ** p
            + beta_bar * x_tilde_prev
        )

        v = x_tilde ** (1.0 / p)

        # -----------------------------------------------------
        # Static gain
        # -----------------------------------------------------
        if mode == "comp":

            if v > l:
                f = kappa_c * v ** (-S_c)
            else:
                f = 1.0

        else:  # expansion

            if 0.0 < v < l:
                f = kappa_e * v ** S_e
            else:
                f = 1.0

        # -----------------------------------------------------
        # Gain attack / release
        # -----------------------------------------------------
        if f < g_prev:
            gamma = gamma_att
        else:
            gamma = gamma_rel

        gamma_bar = 1.0 - gamma

        # Gain smoothing
        g = gamma * f + gamma_bar * g_prev

        # Output
        y[n] = xn * g

        # Update states
        x_tilde_prev = x_tilde
        g_prev = g

    return y


def drc_inverse(y, Fs, q, mode="comp", optim="hybr"):
    """
    Model-based inverse DRC.

    Parameters
    ----------
    y : array_like
        Observed DRC-processed signal.
    Fs : float
        Sampling frequency [Hz].
    q : sequence or dict
        [L, R, tau_v_att, tau_v_rel,
         tau_g_att, tau_g_rel, p]
    mode : {"comp", "exp"}
        "comp" : inverse downward compression
        "exp"  : inverse downward expansion

    Returns
    -------
    x_hat : ndarray
        Estimated original signal.
    """

    y = np.asarray(y, dtype=float)

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
        return y.copy()

    # ---------------------------------------------------------
    # Static-curve parameters
    # ---------------------------------------------------------
    l = 10.0 ** (L / 20.0)

    S_c = 1.0 - 1.0 / R
    kappa_c = l ** S_c

    S_e = R - 1.0
    kappa_e = l ** (-S_e)

    # ---------------------------------------------------------
    # Envelope coefficients
    # ---------------------------------------------------------
    beta_att = _smoothing_coeff(Fs, tau_v_att)
    beta_rel = _smoothing_coeff(Fs, tau_v_rel)

    # ---------------------------------------------------------
    # Gain coefficients
    # ---------------------------------------------------------
    gamma_att = _smoothing_coeff(Fs, tau_g_att)
    gamma_rel = _smoothing_coeff(Fs, tau_g_rel)

    # ---------------------------------------------------------
    # Initial states
    # ---------------------------------------------------------
    x_tilde_prev = 0.0
    g_prev = 1.0

    x_hat = np.empty_like(y)

    tiny = np.finfo(float).tiny

    # ---------------------------------------------------------
    # Sample-by-sample inversion
    # ---------------------------------------------------------
    for n, yn in enumerate(y):

        abs_y = abs(yn)

        # =====================================================
        # Special case: y[n] = 0
        # =====================================================
        if abs_y == 0.0:

            beta = beta_rel
            beta_bar = 1.0 - beta

            # x[n] = 0 => direct envelope update
            v = (
                beta_bar * x_tilde_prev
            ) ** (1.0 / p)

            # Static gain
            if mode == "comp":

                if v > l:
                    f = kappa_c * v ** (-S_c)
                else:
                    f = 1.0

            else:

                if 0.0 < v < l:
                    f = kappa_e * v ** S_e
                else:
                    f = 1.0

            # Gain phase
            if f < g_prev:
                gamma = gamma_att
            else:
                gamma = gamma_rel

            g = (
                gamma * f
                + (1.0 - gamma) * g_prev
            )

            x_hat[n] = 0.0

            x_tilde_prev = v ** p
            g_prev = g

            continue

        # =====================================================
        # Predict envelope attack / release phase
        # =====================================================
        previous_envelope = x_tilde_prev ** (1.0 / p)

        if abs_y > g_prev * previous_envelope:
            beta = beta_att
        else:
            beta = beta_rel

        beta_bar = 1.0 - beta

        # =====================================================
        # Predict gain attack / release phase
        # =====================================================
        if mode == "comp":

            radicand = (
                (kappa_c / g_prev) ** (p / S_c)
                - beta_bar * x_tilde_prev
            )

            if radicand < 0.0:

                # Attack is necessarily active
                gamma = gamma_att

            else:

                attack_threshold = (
                    g_prev
                    * (radicand / beta) ** (1.0 / p)
                )

                if abs_y > attack_threshold:
                    gamma = gamma_att
                else:
                    gamma = gamma_rel

        else:  # expansion

            radicand = (
                (g_prev / kappa_e) ** (p / S_e)
                - beta_bar * x_tilde_prev
            )

            if radicand < 0.0:

                # Attack condition cannot be satisfied
                gamma = gamma_rel

            else:

                attack_threshold = (
                    g_prev
                    * (radicand / beta) ** (1.0 / p)
                )

                if abs_y < attack_threshold:
                    gamma = gamma_att
                else:
                    gamma = gamma_rel

        gamma_bar = 1.0 - gamma

        # =====================================================
        # Determine active / inactive static region
        # =====================================================
        minimum_v_p = beta_bar * x_tilde_prev

        if l ** p < minimum_v_p:

            # v[n] > l necessarily
            if mode == "comp":
                active = True
            else:
                active = False

        else:

            # At v=l, f(l)=1
            if 1.0 < g_prev:
                gamma_T = gamma_att
            else:
                gamma_T = gamma_rel

            gamma_T_bar = 1.0 - gamma_T

            T = (
                gamma_T
                + gamma_T_bar * g_prev
            ) * (
                (l ** p - minimum_v_p) / beta
            ) ** (1.0 / p)

            if mode == "comp":
                active = abs_y > T
            else:
                active = abs_y < T

        # =====================================================
        # Inactive region
        # =====================================================
        if not active:

            f = 1.0

            # Exact gain phase for f=1
            if f < g_prev:
                gamma_inactive = gamma_att
            else:
                gamma_inactive = gamma_rel

            g = (
                gamma_inactive
                + (1.0 - gamma_inactive) * g_prev
            )

            xn = yn / g

            x_tilde = (
                beta * abs(xn) ** p
                + beta_bar * x_tilde_prev
            )

            x_hat[n] = xn

            x_tilde_prev = x_tilde
            g_prev = g

            continue

        # =====================================================
        # Active region: characteristic equation
        # =====================================================
        def zeta(v):

            if mode == "comp":
                f = kappa_c * v ** (-S_c)
            else:
                f = kappa_e * v ** S_e

            current_gain = (
                gamma * f
                + gamma_bar * g_prev
            )

            return (
                current_gain ** p
                * (v ** p - beta_bar * x_tilde_prev)
                - beta * abs_y ** p
            )

        # Physical lower bound
        v_physical_min = (
            beta_bar * x_tilde_prev
        ) ** (1.0 / p)

        # =====================================================
        # Root search / best approximate candidate
        #
        # zeta is strictly increasing on the admissible interval.
        # Therefore:
        #   - if a sign change exists -> unique root
        #   - otherwise the closest admissible candidate is at
        #     one of the interval boundaries.
        # =====================================================
        if mode == "comp":

            v_low = max(l, v_physical_min)

            z_low = zeta(v_low)

            if z_low >= 0.0:

                # No admissible zero below because zeta is increasing.
                # Best approximate candidate = lower boundary.
                v_star = v_low

            else:

                # Find an upper bracket
                v_high = max(
                    2.0 * v_low,
                    2.0 * l,
                    v_physical_min + 1.0
                )

                for _ in range(100):

                    if zeta(v_high) >= 0.0:
                        break

                    v_high *= 2.0

                if zeta(v_high) >= 0.0:
                    if optim == "brentq":

                        v_star = brentq(
                            zeta,
                            v_low,
                            v_high,
                            xtol=1e-14,
                            rtol=1e-12,
                            maxiter=100
                        )
                    
                    elif optim == "hybr":
                    
                        v_init = ( beta * (abs_y / (gamma + gamma_bar * g_prev))**p + beta_bar * x_tilde_prev )**(1.0 / p)
                    
                        sol = root(
                            lambda v: np.array([zeta(v[0])]),
                            x0=np.array([v_init]),
                            method="hybr",
                            options={
                                "xtol": 1e-10,
                                "maxfev": 100
                            }
                        )
                    
                        v_star = float(sol.x[0])
                    
                    else:
                        raise ValueError("optim must be 'brentq' or 'hybr'.")   								
						

                else:

                    # Extremely pathological numerical case
                    v_star = v_high

        else:  # expansion

            v_low = max(v_physical_min, tiny)
            v_high = l

            z_low = zeta(v_low)
            z_high = zeta(v_high)

            if z_low >= 0.0:

                # Best approximate admissible candidate
                v_star = v_low

            elif z_high <= 0.0:

                # Best approximate admissible candidate
                v_star = v_high

            else:

                if optim == "brentq":

                    v_star = brentq(
                        zeta,
                        v_low,
                        v_high,
                        xtol=1e-14,
                        rtol=1e-12,
                        maxiter=100
                    )
                
                elif optim == "hybr":
                    v_init = ( beta * (abs_y / (gamma + gamma_bar * g_prev))**p + beta_bar * x_tilde_prev )**(1.0 / p)
                
                    sol = root(
                        lambda v: np.array([zeta(v[0])]),
                        x0=np.array([v_init]),
                        method="hybr",
                        options={
                            "xtol": 1e-10,
                            "maxfev": 100
                        }
                    )
                
                    v_star = float(sol.x[0])
                
                else:
                    raise ValueError("optim must be 'brentq' or 'hybr'.")

        # =====================================================
        # Recover x[n]
        # =====================================================
        input_power = (
            v_star ** p
            - beta_bar * x_tilde_prev
        ) / beta

        # Numerical safeguard
        input_power = max(input_power, 0.0)

        x_magnitude = input_power ** (1.0 / p)

        xn = np.copysign(x_magnitude, yn)

        # Static gain at recovered envelope
        if mode == "comp":
            f = kappa_c * v_star ** (-S_c)
        else:
            f = kappa_e * v_star ** S_e

        # Gain state
        g = (
            gamma * f
            + gamma_bar * g_prev
        )

        # Store reconstructed sample
        x_hat[n] = xn

        # Recursive states
        x_tilde_prev = v_star ** p
        g_prev = g

    return x_hat
