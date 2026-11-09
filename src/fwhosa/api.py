"""Public estimators: bispectrum, bicoherence and their three-channel cross forms."""
from __future__ import annotations

import math
import os
import re
import warnings
from importlib import metadata

import numpy as np

from . import normalize
from ._pipeline import run
from ._version import __version__
from .engine import resolve_backend
from .plan import WindowPlan, build_triad_plan, make_window_plan
from .result import HOSResult
from .transform import FCWT_PEAK_GAIN, TRANSFORMS, Wavelet

EXPECTATIONS = ("time", "trials", "all")


def _parse_bytes(limit) -> int:
    if isinstance(limit, (int, np.integer)):
        return int(limit)
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)B?\s*", str(limit).upper())
    if not m:
        raise ValueError(f"cannot parse memory_limit={limit!r}; use bytes or e.g. '2GB'")
    return int(float(m.group(1)) * 1024 ** "_KMGT".index(m.group(2) or "_"))


def _as_signal(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1-D (one recording) or 2-D (trials x time); got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


def _estimate(kind: str, raw_signals, fs, *, band, expectation, df, fb, window, step, normalization,
              denominator, squared, backend, transform, min_dof, conditioning_floor, margin_sd,
              lowpass, guard_tol, memory_limit, triads, nthreads, out) -> HOSResult:
    if expectation not in EXPECTATIONS:
        raise ValueError(f"expectation must be one of {EXPECTATIONS}")
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}")
    fs = float(fs)
    if not (fs > 0 and math.isfinite(fs)):
        raise ValueError("fs must be a positive number")
    names = ("x",) if len(raw_signals) == 1 else ("x", "y", "z")
    signals = [_as_signal(s, nm) for s, nm in zip(raw_signals, names)]
    if any(s.shape != signals[0].shape for s in signals):
        raise ValueError("x, y and z must have identical shapes")
    ndim = signals[0].ndim
    if expectation == "time" and ndim != 1:
        raise ValueError("expectation='time' takes one continuous recording (1-D); "
                         "use expectation='trials' or 'all' for trials x time arrays")
    if expectation == "trials" and ndim != 2:
        raise ValueError("expectation='trials' takes a trials x time array (2-D)")
    signals = [s if s.ndim == 2 else s[None, :] for s in signals]

    wavelet = Wavelet(fb)
    cross = kind.startswith("cross")
    plan = build_triad_plan(band, df, fs, wavelet, "cross" if cross else "auto",
                            margin_sd=margin_sd, lowpass=lowpass, triads=triads)
    if expectation == "all":
        if window is not None or step is not None:
            raise ValueError("window and step do not apply to expectation='all'")
        wplan = WindowPlan(block=max(1, int(round(0.25 * fs))), per_window=1, stride=1)
    else:
        if window is None:
            raise ValueError(f"window (seconds) is required for expectation='{expectation}'")
        if window <= 0 or (step is not None and step <= 0):
            raise ValueError("window and step must be positive")
        wplan = make_window_plan(window, window / 2.0 if step is None else step, fs)

    backend_used = resolve_backend(backend)
    nthreads = int(nthreads or os.cpu_count() or 1)
    sums = run(signals, fs, plan, wavelet, expectation, wplan, backend_used,
               _parse_bytes(memory_limit), nthreads, guard_tol, transform=transform)

    f1, f2, f3 = plan.f1, plan.f2, plan.f3
    if kind.endswith("bispectrum"):
        values = normalize.bispectrum(sums.S, sums.n, sums.scale_prod[None, :], f1, f2,
                                      normalization, fb).astype(np.complex64)
    else:
        values = normalize.bicoherence(sums.S, sums.Q12, sums.P1, sums.P2, sums.P3, sums.n,
                                       denominator, squared).astype(np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        energy = (sums.P1 / sums.n) * (sums.P2 / sums.n) * (sums.P3 / sums.n)
    dof = sums.duration_s * np.sqrt(f1 ** 2 + f2 ** 2 + f3 ** 2) / (2.0 * math.sqrt(math.pi) * fb)
    flags = {
        "boundary": sums.boundary,
        "conditioning": ~(energy >= conditioning_floor),
        "dof": np.broadcast_to(dof < min_dof, values.shape).copy(),
        "undefined": ~np.isfinite(values),
    }
    valid = ~np.logical_or.reduce(list(flags.values()))
    if flags["dof"][0].any():
        warnings.warn(f"{int(flags['dof'][0].sum())} of {plan.n} triads have fewer than min_dof={min_dof} "
                      "effective independent samples per estimate and are flagged invalid; lengthen "
                      "the window or raise the band's lower edge", stacklevel=3)

    times = (np.full(1, np.nan) if expectation == "all"
             else (sums.starts + wplan.window_len / 2.0) / fs)
    params = dict(
        fwhosa=__version__, fcwt=_version("fCWT"), numpy=np.__version__, numba=_version("numba"),
        kind=kind, fs=fs, band=list(map(float, band)) if triads is None else None,
        triads_explicit=triads is not None, df=float(df), fb=float(fb),
        wavelet="analytic Morlet (fCWT convention), response c*exp(-2*pi^2*fb^2*(r-1)^2)",
        gain_convention=f"fCWT peak gain c = {FCWT_PEAK_GAIN:.6f}, included in D_psi",
        normalization=normalization if kind.endswith("bispectrum") else None,
        density_measure={"raw": "none (triple-product average, physical units^3)",
                         "log_density": "per d(log f1) d(log f2), Newman et al. 2021 Eq. 45",
                         "linear_density": "per Hz^2 (log density / (f1 f2))"}.get(normalization)
        if kind.endswith("bispectrum") else None,
        denominator=None if kind.endswith("bispectrum") else denominator,
        squared=None if kind.endswith("bispectrum") else bool(squared),
        expectation=expectation, weighting="pooled samples (equal-length trials)",
        window_s=None if expectation == "all" else wplan.window_len / fs,
        step_s=None if expectation == "all" else wplan.step_len / fs,
        backend=backend_used,
        precision="float32 coefficients; float64 block sums" if backend_used != "cuda"
        else "float32 coefficients and block sums; float64 window and trial sums",
        margin_sd=float(margin_sd), sum_cap_hz=plan.sum_cap_hz, lowpass=lowpass,
        guard_tol=float(guard_tol), min_dof=float(min_dof), conditioning_floor=float(conditioning_floor),
        dof_model="white-noise approximation: duration * sqrt(f1^2+f2^2+f3^2) / (2 sqrt(pi) fb)",
        **sums.info,
    )
    if out is not None:
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=values.dtype, shape=values.shape)
        mm[:] = values
        mm.flush()
        values = mm
    return HOSResult(kind=kind, values=values, f1=f1, f2=f2, f3=f3, times=times, valid=valid,
                     flags=flags, dof=dof, pair_freqs=plan.pair_freqs, params=params)


_COMMON_DOC = """
    Parameters common to all estimators
    -----------------------------------
    fs : float
        Sampling rate in Hz.
    band : (float, float)
        Range of the f1 and f2 axes in Hz. Sums f1 + f2 must leave ``margin_sd``
        spectral SDs of the wavelet below Nyquist (and below ``lowpass``); triads
        that do not are left out of the plan.
    expectation : {"time", "trials", "all"}
        Required. "time": sliding windows over one continuous recording (1-D input).
        "trials": windows at matching latencies, summed across trials
        (trials x time input). "all": one estimate over everything. Complex
        moments are summed before any ratio is formed.
    df : float
        Grid spacing in Hz; the lower band edge is rounded to a multiple of df so
        every f1 + f2 falls exactly on a grid row.
    fb : float
        fCWT Morlet bandwidth in cycles per envelope SD (envelope SD = fb / f s).
    window, step : float
        Window length and step in seconds (step defaults to window / 2).
    backend : {"auto", "reference", "cpu", "cuda"}
        "auto" picks CUDA when CuPy and a GPU are present, else the Numba CPU
        engine, and records its choice. "cuda" fails loudly if unavailable.
    transform : {"exact", "fcwt"}
        "exact" (default) evaluates the Morlet response at every FFT bin, so
        coefficients do not depend on segment or recording length. "fcwt" uses
        fCWT (optional extra); faster, but its coefficients shift by up to about
        0.6% with transformed length.
    min_dof : float
        Estimates with fewer effective independent samples (white-noise
        approximation, see ``result.dof``) are flagged invalid. Provisional
        default pending the window-rule calibration study.
    conditioning_floor : float
        Estimates whose mean powers multiply to less than this, in units of each
        row's recording-wide RMS, are flagged (recording-relative conditioning).
    margin_sd, lowpass : float
        Spectral margin below Nyquist, and the acquisition low-pass in Hz.
    guard_tol : float
        Envelope-tail tolerance that sets edge guards and segment context.
    memory_limit : int or str
        Budget for coefficients and block moments, e.g. "2GB"; larger inputs are
        processed in guard-banded segments.
    triads : sequence of (f1, f2), optional
        Explicit triads on the grid instead of the full band.
    out : path, optional
        Write ``values`` to a memory-mapped .npy file at this path.
"""


def bicoherence(x, fs, *, band, expectation, df=0.5, fb=2.0, window=None, step=None,
                denominator="paired_moment", squared=True, backend="auto", transform="exact", min_dof=2.0,
                conditioning_floor=1e-6, margin_sd=3.0, lowpass=None, guard_tol=1e-4,
                memory_limit="2GB", triads=None, nthreads=None, out=None) -> HOSResult:
    """Wavelet bicoherence of one signal.

    denominator : {"paired_moment", "product_of_powers"}
        "paired_moment" (default) is bounded by 1 (Kim & Powers 1979);
        "product_of_powers" can legitimately exceed 1 and is never clipped.
    squared : bool
        Return b^2 (default) or |b|.
    """
    return _estimate("bicoherence", (x,), fs, band=band, expectation=expectation, df=df, fb=fb,
                     window=window, step=step, normalization=None, denominator=denominator,
                     squared=squared, backend=backend, transform=transform, min_dof=min_dof,
                     conditioning_floor=conditioning_floor, margin_sd=margin_sd, lowpass=lowpass,
                     guard_tol=guard_tol, memory_limit=memory_limit, triads=triads,
                     nthreads=nthreads, out=out)


def bispectrum(x, fs, *, band, expectation, df=0.5, fb=2.0, window=None, step=None,
               normalization="linear_density", backend="auto", transform="exact", min_dof=2.0, conditioning_floor=1e-6,
               margin_sd=3.0, lowpass=None, guard_tol=1e-4, memory_limit="2GB", triads=None,
               nthreads=None, out=None) -> HOSResult:
    """Wavelet bispectrum of one signal.

    normalization : {"raw", "log_density", "linear_density"}
        "raw" is the triple-product average in physical units cubed;
        "log_density" divides by D_psi (Newman, Pidde & Stefanovska 2021, Eq. 45);
        "linear_density" (default) further divides by f1 * f2, giving a density
        per Hz^2 comparable with a Fourier bispectrum.
    """
    return _estimate("bispectrum", (x,), fs, band=band, expectation=expectation, df=df, fb=fb,
                     window=window, step=step, normalization=normalization, denominator=None,
                     squared=True, backend=backend, transform=transform, min_dof=min_dof,
                     conditioning_floor=conditioning_floor, margin_sd=margin_sd, lowpass=lowpass,
                     guard_tol=guard_tol, memory_limit=memory_limit, triads=triads,
                     nthreads=nthreads, out=out)


def cross_bicoherence(x, y, z, fs, *, band, expectation, df=0.5, fb=2.0, window=None, step=None,
                      denominator="paired_moment", squared=True, backend="auto", transform="exact", min_dof=2.0,
                      conditioning_floor=1e-6, margin_sd=3.0, lowpass=None, guard_tol=1e-4,
                      memory_limit="2GB", triads=None, nthreads=None, out=None) -> HOSResult:
    """Cross-bicoherence of x at f1, y at f2 and z at f1 + f2 (ordered; see :func:`bicoherence`)."""
    return _estimate("cross_bicoherence", (x, y, z), fs, band=band, expectation=expectation, df=df,
                     fb=fb, window=window, step=step, normalization=None, denominator=denominator,
                     squared=squared, backend=backend, transform=transform, min_dof=min_dof,
                     conditioning_floor=conditioning_floor, margin_sd=margin_sd, lowpass=lowpass,
                     guard_tol=guard_tol, memory_limit=memory_limit, triads=triads,
                     nthreads=nthreads, out=out)


def cross_bispectrum(x, y, z, fs, *, band, expectation, df=0.5, fb=2.0, window=None, step=None,
                     normalization="linear_density", backend="auto", transform="exact", min_dof=2.0,
                     conditioning_floor=1e-6, margin_sd=3.0, lowpass=None, guard_tol=1e-4,
                     memory_limit="2GB", triads=None, nthreads=None, out=None) -> HOSResult:
    """Cross-bispectrum of x at f1, y at f2 and z at f1 + f2 (ordered; see :func:`bispectrum`)."""
    return _estimate("cross_bispectrum", (x, y, z), fs, band=band, expectation=expectation, df=df,
                     fb=fb, window=window, step=step, normalization=normalization, denominator=None,
                     squared=True, backend=backend, transform=transform, min_dof=min_dof,
                     conditioning_floor=conditioning_floor, margin_sd=margin_sd, lowpass=lowpass,
                     guard_tol=guard_tol, memory_limit=memory_limit, triads=triads,
                     nthreads=nthreads, out=out)


for _f in (bicoherence, bispectrum, cross_bicoherence, cross_bispectrum):
    _f.__doc__ = (_f.__doc__ or "") + _COMMON_DOC
