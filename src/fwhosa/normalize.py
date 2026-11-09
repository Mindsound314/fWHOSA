"""Turning accumulated moments into estimates.

Moments over an averaging domain (a window, a set of trials, or everything),
in row-scaled units with weight total ``n``::

    S   = sum W1 W2 conj(W3)      Q12 = sum |W1 W2|^2
    P_m = sum |W_m|^2             (m = 1, 2, 3, gathered at each triad's rows)

Bicoherence (both forms are invariant to per-row scaling)::

    paired_moment     b^2 = |S|^2 / (Q12 * P3)          bounded by 1 (Kim & Powers 1979)
    product_of_powers b^2 = n |S|^2 / (P1 * P2 * P3)    can exceed 1 (Brillinger-Hinich line)

Bispectrum (physical units restored with the row scales r)::

    raw             B = (S / n) * r1 r2 r3
    log_density     B / D(lambda)                      Newman, Pidde & Stefanovska (2021), Eq. 45
    linear_density  B / (D(lambda) * f1 * f2)          the same density per Hz^2

with ``lambda = f1 / (f1 + f2)`` and ``D`` computed from the exact response fCWT
applies (peak gain included), so a flat bispectrum of density ``k3 / fs**2``
comes out as ``k3 / fs**2`` in ``linear_density``.
"""
from __future__ import annotations

import math

import numpy as np

from .transform import FCWT_PEAK_GAIN

PAIRED_BOUND_TOL = 1e-5


def dpsi(lam: np.ndarray, fb: float, gain: float = FCWT_PEAK_GAIN) -> np.ndarray:
    """D(lambda) = int int g(r1) g(r2) g(lambda r1 + (1 - lambda) r2) dr1 dr2 for fCWT's Morlet.

    Closed form for the Gaussian response; exact to well below 1e-6 for
    ``fb >= 1``, where the truncation at r <= 0 and r >= 2 is negligible.
    """
    lam = np.asarray(lam, dtype=float)
    a = 2.0 * math.pi ** 2 * fb ** 2
    return gain ** 3 * math.pi / (a * np.sqrt(1.0 + lam ** 2 + (1.0 - lam) ** 2))


def dpsi_quadrature(lam: float, fb: float, gain: float = FCWT_PEAK_GAIN, n: int = 2001) -> float:
    """Direct 2-D integration of the same quantity, used to test :func:`dpsi`."""
    a = 2.0 * math.pi ** 2 * fb ** 2
    half = 12.0 / math.sqrt(2.0 * a)               # +-12 response SDs around r = 1
    r = np.linspace(max(1e-9, 1.0 - half), min(2.0, 1.0 + half), n)

    def g(x):
        return np.where((x > 0) & (x <= 2.0), gain * np.exp(-a * (x - 1.0) ** 2), 0.0)

    R1, R2 = np.meshgrid(r, r, indexing="ij")
    f = g(R1) * g(R2) * g(lam * R1 + (1.0 - lam) * R2)
    return float(np.trapezoid(np.trapezoid(f, r, axis=1), r))


def bispectrum(S, n, scale_prod, f1, f2, normalization: str, fb: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        B = (S / n) * scale_prod
    if normalization == "raw":
        return B
    lam = f1 / (f1 + f2)
    B = B / dpsi(lam, fb)
    if normalization == "log_density":
        return B
    if normalization == "linear_density":
        return B / (f1 * f2)
    raise ValueError("normalization must be 'raw', 'log_density' or 'linear_density'")


def bicoherence(S, Q12, P1, P2, P3, n, denominator: str, squared: bool) -> np.ndarray:
    num = S.real ** 2 + S.imag ** 2
    if denominator == "paired_moment":
        den = Q12 * P3
    elif denominator == "product_of_powers":
        den = P1 * P2 * P3 / np.asarray(n, dtype=float)
    else:
        raise ValueError("denominator must be 'paired_moment' or 'product_of_powers'")
    with np.errstate(invalid="ignore", divide="ignore"):
        b2 = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    if denominator == "paired_moment":
        finite = np.isfinite(b2)
        worst = float(np.max(b2[finite])) if finite.any() else 0.0
        if worst > 1.0 + PAIRED_BOUND_TOL:
            raise FloatingPointError(
                f"paired-moment bicoherence reached {worst:.6f} > 1; Cauchy-Schwarz bounds it by 1, "
                "so this indicates a numerical or implementation error, not a result")
    return b2 if squared else np.sqrt(b2)
