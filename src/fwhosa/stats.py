"""Surrogate null models and multiplicity-corrected tests.

Surrogates rerun the exact estimator and settings on data whose coupling has
been destroyed, so the null always matches the estimate it tests. Stationary
Fourier surrogates ("phase", "iaaft") preserve each channel's spectrum but also
remove genuine nonstationarity; "circular_shift" (cross estimators only) keeps
every channel intact and breaks only their alignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .result import HOSResult

METHODS = ("phase", "iaaft", "circular_shift")
CORRECTIONS = ("max-stat", "fdr", "none")


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same amplitude spectrum, independent uniform Fourier phases (per row if 2-D)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    ph = rng.uniform(0.0, 2.0 * np.pi, X.shape)
    ph[..., 0] = 0.0
    if n % 2 == 0:
        ph[..., -1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=n, axis=-1)


def iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = 100) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate (Schreiber & Schmitz 1996)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x)
    for r, row in enumerate(x):
        amp = np.abs(np.fft.rfft(row))
        sorted_vals = np.sort(row)
        s = rng.permutation(row)
        for _ in range(n_iter):
            S = np.fft.rfft(s)
            s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=row.size)
            s_new = sorted_vals[np.argsort(np.argsort(s))]
            if np.array_equal(s_new, s):
                break
            s = s_new
        out[r] = s
    return out


def circular_shift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent random circular shift of each row, at least 10% of its length from zero."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n = x.shape[-1]
    lo = max(1, n // 10)
    shifts = rng.integers(lo, n - lo + 1, size=x.shape[0])
    return np.stack([np.roll(row, int(s)) for row, s in zip(x, shifts)])


@dataclass
class NullTestResult:
    """Estimate plus inference; ``result.values`` is never modified."""

    result: HOSResult
    p_values: np.ndarray          # corrected for multiplicity (NaN where the estimate is invalid)
    p_uncorrected: np.ndarray
    threshold: np.ndarray | None  # per-cell critical value of the statistic (max-stat only)
    significant: np.ndarray       # p_values <= alpha and valid
    method: str
    correction: str
    n_surrogates: int
    alpha: float


def _statistic(res: HOSResult) -> np.ndarray:
    v = np.asarray(res.values)
    return np.abs(v).astype(np.float64) if np.iscomplexobj(v) else v.astype(np.float64)


def null_test(estimator, *args, method: str = "phase", n_surrogates: int = 200, alpha: float = 0.05,
              correction: str = "max-stat", seed=None, max_bytes: int = 2 * 1024 ** 3,
              **kwargs) -> NullTestResult:
    """Surrogate test for any fwhosa estimator.

    Call as ``null_test(fw.bicoherence, x, fs, band=..., expectation=..., ...)`` or
    ``null_test(fw.cross_bicoherence, x, y, z, fs, ...)``. The statistic is the
    bicoherence, or |B| for bispectra. "max-stat" controls the family-wise error
    rate with the maximum of per-cell standardised statistics; "fdr" applies
    Benjamini-Hochberg to per-cell p-values.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")
    *signals, fs = args
    cross = len(signals) == 3
    if method == "circular_shift" and not cross:
        raise ValueError("circular_shift only breaks alignment between channels; use it with cross estimators")
    rng = np.random.default_rng(seed)

    observed = estimator(*signals, fs, **kwargs)
    obs = _statistic(observed)
    valid = observed.valid
    if correction == "max-stat" and n_surrogates * obs.size * 4 > max_bytes:
        raise MemoryError("max-stat needs every surrogate map in memory; reduce n_surrogates, "
                          "the triad plan, or use correction='fdr'")

    keep = np.empty((n_surrogates,) + obs.shape, np.float32) if correction == "max-stat" else None
    exceed = np.zeros(obs.shape, np.int64)
    for s in range(n_surrogates):
        if method == "circular_shift":
            sur = [signals[0]] + [circular_shift(np.asarray(sig), rng).reshape(np.shape(sig))
                                  for sig in signals[1:]]
        else:
            fn = phase_randomize if method == "phase" else iaaft
            sur = [fn(np.asarray(sig), rng).reshape(np.shape(sig)) for sig in signals]
        stat = _statistic(estimator(*sur, fs, **kwargs))
        exceed += stat >= obs
        if keep is not None:
            keep[s] = stat

    p_unc = (1.0 + exceed) / (n_surrogates + 1.0)
    p_unc = np.where(valid, p_unc, np.nan)
    threshold = None
    if correction == "none":
        p = p_unc
    elif correction == "fdr":
        p = np.full(obs.shape, np.nan)
        flat = p_unc[valid]
        order = np.argsort(flat)
        m = flat.size
        adj = flat[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(adj, 1.0)
        p[valid] = out
    else:
        # Standardise with all n + 1 maps, observed included, so the observed map and
        # the surrogates are treated identically and stay exchangeable under the null.
        pooled = np.concatenate([obs[None].astype(np.float32), keep], axis=0)
        mu = pooled.mean(axis=0, dtype=np.float64)
        sd = pooled.std(axis=0, dtype=np.float64)
        del pooled
        sd = np.where(sd > 0, sd, np.inf)
        z_obs = (obs - mu) / sd
        z_sur = (keep - mu) / sd
        maxima = np.where(valid[None], z_sur, -np.inf).reshape(n_surrogates, -1).max(axis=1)
        p = (1.0 + (maxima[:, None] >= z_obs.reshape(1, -1)).sum(axis=0)) / (n_surrogates + 1.0)
        p = np.where(valid, p.reshape(obs.shape), np.nan)
        z_crit = np.quantile(maxima, 1.0 - alpha)
        threshold = mu + z_crit * np.where(np.isfinite(sd), sd, 0.0)
    significant = np.where(valid, p <= alpha, False)
    return NullTestResult(result=observed, p_values=p, p_uncorrected=p_unc, threshold=threshold,
                          significant=significant, method=method, correction=correction,
                          n_surrogates=int(n_surrogates), alpha=float(alpha))
