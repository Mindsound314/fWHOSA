"""Triad plans (which frequency triplets to compute) and window plans (how to average in time)."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .transform import Grid, Wavelet

_EPS = 1e-9


@dataclass(frozen=True)
class TriadPlan:
    """Triads ``(f1, f2, f1 + f2)`` as row indices into a harmonic transform grid.

    ``kind="auto"`` keeps only ``f1 <= f2`` (the auto bispectrum is symmetric);
    ``kind="cross"`` keeps every ordered pair, because channel order matters.
    """

    grid: Grid
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    kind: str
    pair_freqs: np.ndarray = field(repr=False)
    margin_sd: float = 3.0
    sum_cap_hz: float = math.inf

    @property
    def n(self) -> int:
        return int(self.i.size)

    @property
    def f1(self) -> np.ndarray:
        return self.grid.freqs[self.i]

    @property
    def f2(self) -> np.ndarray:
        return self.grid.freqs[self.j]

    @property
    def f3(self) -> np.ndarray:
        return self.grid.freqs[self.k]


def _on_grid(f: float, df: float) -> int:
    q = f / df
    k = int(round(q))
    if abs(q - k) > 1e-6:
        raise ValueError(f"{f} Hz is not a multiple of df = {df} Hz")
    return k


def build_triad_plan(band, df: float, fs: float, wavelet: Wavelet, kind: str, *,
                     margin_sd: float = 3.0, lowpass: float | None = None,
                     triads=None) -> TriadPlan:
    """Plan the triads for pair axes ``band = (f_lo, f_hi)`` on a grid of spacing ``df``.

    A triad is kept only if its sum frequency leaves ``margin_sd`` spectral SDs of
    the wavelet below Nyquist and below ``lowpass`` (the acquisition low-pass, if
    given). ``triads`` optionally restricts the plan to explicit ``(f1, f2)`` pairs,
    which must lie on the grid.
    """
    if kind not in ("auto", "cross"):
        raise ValueError("kind must be 'auto' or 'cross'")
    if df <= 0:
        raise ValueError("df must be positive")
    try:
        f_lo, f_hi = (float(band[0]), float(band[1]))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("band must be a pair (f_lo, f_hi) in Hz") from exc
    if not (0 < f_lo < f_hi):
        raise ValueError("band must satisfy 0 < f_lo < f_hi")

    k_lo = int(round(f_lo / df))
    if k_lo < 1:
        raise ValueError("band's lower edge must be at least df")
    if abs(k_lo * df - f_lo) > 1e-6:
        warnings.warn(f"band's lower edge {f_lo} Hz rounded to {k_lo * df} Hz, a multiple of df",
                      stacklevel=3)
    k_hi = int(math.floor(f_hi / df + _EPS))

    ceiling = fs / 2.0 if lowpass is None else min(fs / 2.0, float(lowpass))
    cap = ceiling / (1.0 + margin_sd * wavelet.relative_spectral_sd)
    k_cap = int(math.floor(cap / df + _EPS))
    # fCWT's request ends one bin above the last row; keep that request at or below Nyquist.
    k_cap = min(k_cap, int(math.floor(fs / 2.0 / df + _EPS)) - 1)

    if triads is None:
        pair_k = np.arange(k_lo, k_hi + 1)
        a, b = np.meshgrid(pair_k, pair_k, indexing="ij")
        keep = (a + b) <= k_cap
        if kind == "auto":
            keep &= a <= b
        ka, kb = a[keep], b[keep]
    else:
        pairs = [(_on_grid(float(p), df), _on_grid(float(q), df)) for p, q in triads]
        if not pairs:
            raise ValueError("triads is empty")
        ka = np.array([p for p, _ in pairs])
        kb = np.array([q for _, q in pairs])
        if np.any(ka < 1) or np.any(kb < 1):
            raise ValueError("explicit triad frequencies must be positive")
        too_high = (ka + kb) > k_cap
        if np.any(too_high):
            raise ValueError(f"{int(too_high.sum())} explicit triad(s) have sums above the "
                             f"{cap:.2f} Hz margin below Nyquist / low-pass")
        k_lo = int(min(ka.min(), kb.min()))
        pair_k = np.unique(np.concatenate([ka, kb]))

    if ka.size == 0:
        raise ValueError(
            f"no triads: every sum frequency in band {band} exceeds {cap:.2f} Hz, the "
            f"{margin_sd}-SD spectral margin below Nyquist (fs={fs}) or the low-pass"
        )
    k_top = int((ka + kb).max())
    grid = Grid(df=float(df), k0=k_lo, n=k_top - k_lo + 1)

    sigma_lo = k_lo * df * wavelet.relative_spectral_sd
    if df > 2.0 * sigma_lo:
        warnings.warn(
            f"df = {df} Hz is coarser than twice the wavelet's spectral SD at {k_lo * df} Hz "
            f"({sigma_lo:.3f} Hz); components falling between bins are attenuated", stacklevel=3)

    return TriadPlan(grid=grid, i=(ka - k_lo).astype(np.int32), j=(kb - k_lo).astype(np.int32),
                     k=(ka + kb - k_lo).astype(np.int32), kind=kind,
                     pair_freqs=pair_k * float(df), margin_sd=float(margin_sd), sum_cap_hz=float(cap))


@dataclass(frozen=True)
class WindowPlan:
    """Windows as sums of equal blocks of ``block`` samples.

    A window spans ``per_window`` consecutive blocks and windows start every
    ``stride`` blocks, so overlapping windows reuse block sums instead of
    recomputing the shared samples.
    """

    block: int
    per_window: int
    stride: int

    @property
    def window_len(self) -> int:
        return self.block * self.per_window

    @property
    def step_len(self) -> int:
        return self.block * self.stride

    def n_windows(self, n_blocks: int) -> int:
        if n_blocks < self.per_window:
            return 0
        return (n_blocks - self.per_window) // self.stride + 1


def make_window_plan(window_s: float, step_s: float, fs: float) -> WindowPlan:
    wlen = int(round(window_s * fs))
    step = int(round(step_s * fs))
    if wlen < 1 or step < 1:
        raise ValueError("window and step must each span at least one sample")
    block = math.gcd(wlen, step)
    return WindowPlan(block=block, per_window=wlen // block, stride=step // block)
