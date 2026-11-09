"""Analytic Morlet continuous wavelet transform on a harmonic frequency grid.

Wavelet convention (fCWT 0.1.18's, read from its C++ source)
--------------------------------------------------------------
For a row with centre frequency ``f``, the transform multiplies the signal's
spectrum by

    g(r) = c * exp(-2 * pi**2 * fb**2 * (r - 1)**2),   r = nu / f > 0,

with peak gain ``c = sqrt(2*pi) * pi**(-1/4)`` at every scale (analytic wavelet,
positive frequencies only). ``fb`` is the number of cycles per envelope standard
deviation: the envelope SD is ``fb / f`` seconds and the spectral SD is
``f / (2*pi*fb)`` Hz. ``fb = 2`` corresponds to omega0 = 4*pi (about 12.6) in
Torrence-Compo terms. A real cosine of amplitude ``A`` gives ``|W| = c * A / 2``.

Two implementations
-------------------
``"exact"`` (default) evaluates ``g`` at every FFT bin, so coefficients do not
depend on how long the transformed stretch is; segmented and whole-recording
results agree to the guard tolerance.

``"fcwt"`` calls fCWT's low-level API (``pip install fwhosa[fcwt]``). It is fast,
but fCWT samples its mother wavelet with a floor index whose step shrinks with
the FFT length, so its coefficients shift by up to about 0.6% with segment or
recording length at frequencies whose scale is not an integer number of samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Peak gain of the Morlet frequency response (fCWT's ``sqrt(2*pi) * IPI4``).
FCWT_PEAK_GAIN: float = math.sqrt(2.0 * math.pi) * 0.75112554446

FB_RANGE = (1.0, 10.0)
TRANSFORMS = ("exact", "fcwt")


@dataclass(frozen=True)
class Wavelet:
    """Analytic Morlet wavelet with bandwidth ``fb`` (cycles per envelope SD)."""

    fb: float = 2.0

    def __post_init__(self) -> None:
        lo, hi = FB_RANGE
        if not (lo <= float(self.fb) <= hi):
            raise ValueError(
                f"fb={self.fb} is outside the supported range [{lo}, {hi}]; below 1 the "
                "response is truncated at r = 0 and r = 2, above 10 frequency grids "
                "undersample it badly."
            )

    @property
    def relative_spectral_sd(self) -> float:
        """Spectral SD as a fraction of the centre frequency, 1 / (2*pi*fb)."""
        return 1.0 / (2.0 * math.pi * self.fb)

    def envelope_sd(self, f):
        """Temporal SD of the amplitude envelope at frequency ``f``, in seconds."""
        return self.fb / np.asarray(f, dtype=float)

    def response(self, r):
        """Frequency response g(r), including its peak gain."""
        r = np.asarray(r, dtype=float)
        a = 2.0 * math.pi ** 2 * self.fb ** 2
        return np.where(r > 0, FCWT_PEAK_GAIN * np.exp(-a * (r - 1.0) ** 2), 0.0)


@dataclass(frozen=True)
class Grid:
    """Harmonic frequency grid ``f_m = (k0 + m) * df`` for ``m = 0 .. n-1``."""

    df: float
    k0: int
    n: int

    @property
    def freqs(self) -> np.ndarray:
        return (self.k0 + np.arange(self.n)) * self.df


def guard_sd(tol: float) -> float:
    """Envelope SDs beyond which a Gaussian envelope's two-sided area is below ``tol``.

    Solves ``erfc(k / sqrt(2)) = tol`` for ``k`` by bisection.
    """
    if not (0.0 < tol < 1.0):
        raise ValueError("guard tolerance must be in (0, 1)")
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid / math.sqrt(2.0)) > tol:
            lo = mid
        else:
            hi = mid
    return hi


def cwt(x, fs: float, grid: Grid, wavelet: Wavelet, nthreads: int = 1, method: str = "exact",
        pad: int = 0) -> np.ndarray:
    """CWT of a 1-D signal on ``grid``; returns ``(grid.n, x.size)`` complex64, rows ascending.

    ``pad`` is the number of zeros the exact transform appends beyond the wavelet
    support so circular convolution cannot wrap; callers pass the guard length.
    Samples within the guard of the stretch's ends see zero context and must be
    discarded or flagged by the caller.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("cwt expects a 1-D signal")
    if method == "exact":
        return _cwt_exact(x, fs, grid, wavelet, nthreads, pad)
    if method == "fcwt":
        return _cwt_fcwt(x, fs, grid, wavelet, nthreads)
    raise ValueError(f"transform must be one of {TRANSFORMS}")


def _cwt_exact(x, fs, grid, wavelet, nthreads, pad, rows_per_chunk: int = 8):
    import scipy.fft as sfft

    n = x.size
    L = sfft.next_fast_len(n + max(int(pad), 1) * 2)
    X = sfft.fft(np.asarray(x, dtype=np.float64), n=L, workers=nthreads)
    nu = sfft.fftfreq(L, 1.0 / fs)
    pos = np.flatnonzero(nu > 0)
    a = 2.0 * math.pi ** 2 * wavelet.fb ** 2
    reach = 12.0 / math.sqrt(2.0 * a)            # response is below 1e-30 beyond 12 SDs
    out = np.empty((grid.n, n), dtype=np.complex64)
    freqs = grid.freqs
    for r0 in range(0, grid.n, rows_per_chunk):
        rows = freqs[r0:r0 + rows_per_chunk]
        # float64 throughout: a loud stretch anywhere in the record sets the scale of every
        # FFT bin, so single precision would cost quiet stretches ~1e-4 relative accuracy.
        Z = np.zeros((rows.size, L), dtype=np.complex128)
        for m, f in enumerate(rows):
            band = pos[(nu[pos] > f * (1.0 - reach)) & (nu[pos] < f * (1.0 + reach))]
            r = nu[band] / f
            Z[m, band] = X[band] * (FCWT_PEAK_GAIN * np.exp(-a * (r - 1.0) ** 2))
        out[r0:r0 + rows.size] = sfft.ifft(Z, axis=1, workers=nthreads)[:, :n]
    return out


def _cwt_fcwt(x, fs, grid, wavelet, nthreads):
    try:
        import fcwt  # imported lazily: fcwt imports matplotlib at load time
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("transform='fcwt' needs the optional fCWT package: pip install fwhosa[fcwt]") from exc

    x = np.ascontiguousarray(x, dtype=np.float32)
    fs_int = int(round(fs))
    ratio = fs_int / float(fs)              # fCWT needs an integer fs; scale frequencies instead
    f_start = grid.freqs[0] * ratio
    f_request_top = (grid.freqs[0] + grid.n * grid.df) * ratio   # fCWT's top bin is f1 - df
    if f_request_top > fs_int / 2.0 + 1e-9:
        raise ValueError(f"grid top {grid.freqs[-1]:.3f} Hz is too close to Nyquist for fCWT's linear grid")
    morlet = fcwt.Morlet(float(wavelet.fb))
    scales = fcwt.Scales(morlet, fcwt.FCWT_LINFREQS, fs_int, float(f_start), float(f_request_top), int(grid.n))
    engine = fcwt.FCWT(morlet, int(max(1, nthreads)), False, True)
    out = np.zeros((grid.n, x.size), dtype=np.complex64)
    engine.cwt(x, scales, out)
    got = np.zeros(grid.n, dtype=np.float32)
    scales.getFrequencies(got)
    got = got[::-1].astype(np.float64) / ratio
    if not np.allclose(got, grid.freqs, rtol=0.0, atol=1e-3 * grid.df):
        worst = float(np.max(np.abs(got - grid.freqs)))
        raise RuntimeError(f"fCWT returned frequencies off the intended grid by up to {worst:.2e} Hz")
    return np.ascontiguousarray(out[::-1])
