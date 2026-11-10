"""
Cross Time‑Varying Bicoherence (TVCrossBiCoh) – GPU & Streaming
================================================================

Cross‑channel generalisation of fCWBicoherence_gpu:

    b_xyz^2(f1,f2) =
        | E[ W_x(f1,t) W_y(f2,t) W_z^*(f1+f2,t) ] |^2
        -------------------------------------------------
        E[|W_x(f1,t) W_y(f2,t)|^2] · E[|W_z(f1+f2,t)|^2]

(“Hinich” form, the default), or the HOSA‑style denominator

        E[|W_x(f1,t)|^2] · E[|W_y(f2,t)|^2] · E[|W_z(f1+f2,t)|^2].

Implements time‑varying cross‑bicoherence over sliding windows, with
optional GPU acceleration and streaming to disk.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numba as nb

try:
    import cupy as cp  # optional
except Exception:  # GPU optional
    cp = None  # type: ignore

import fcwt
from fCWBispectrum_gpu import _build_lookup  # reuse LUT helper


__all__ = ["compute", "plot_slice"]


# -----------------------------------------------------------------------------
# 1)  Core engines
# -----------------------------------------------------------------------------


@nb.njit(parallel=True, fastmath=True)
def _tv_cross_bicoherence_cpu(
    coef1: np.ndarray,
    coef2: np.ndarray,
    coef3: np.ndarray,
    lut: np.ndarray,
    starts: np.ndarray,
    wlen: int,
    eps: float,
    invalid_to_nan: bool,
    squared: bool,
    denom: str,
) -> np.ndarray:
    """
    CPU engine: cross‑bicoherence for three CWT coefficient arrays.

    coef1, coef2, coef3 : (nbins, N)
        Wavelet coefficients for x(t), y(t), z(t).
    lut : (nbins, nbins)
        Lookup table giving k with f_k ≈ f_i+f_j or -1 if out of band.
    starts : (n_wins,)
        Start indices of analysis windows.
    wlen : int
        Window length (samples).
    """
    nbins, _ = coef1.shape
    n_wins = starts.size

    out = np.empty((nbins, nbins, n_wins), np.float32)
    fill = np.float32(np.nan if invalid_to_nan else 0.0)
    out[:] = fill

    for w in nb.prange(n_wins):
        s = starts[w]
        win1 = coef1[:, s : s + wlen]
        win2 = coef2[:, s : s + wlen]
        win3 = coef3[:, s : s + wlen]

        # mean power per bin for each channel
        power1 = np.empty(nbins, np.float32)
        power2 = np.empty(nbins, np.float32)
        power3 = np.empty(nbins, np.float32)

        for i in range(nbins):
            acc1 = 0.0
            acc2 = 0.0
            acc3 = 0.0
            for t in range(wlen):
                z1 = win1[i, t]
                z2 = win2[i, t]
                z3 = win3[i, t]
                acc1 += z1.real * z1.real + z1.imag * z1.imag
                acc2 += z2.real * z2.real + z2.imag * z2.imag
                acc3 += z3.real * z3.real + z3.imag * z3.imag
            power1[i] = acc1 / wlen
            power2[i] = acc2 / wlen
            power3[i] = acc3 / wlen

        for i in range(nbins):
            wi1 = win1[i]
            for j in range(nbins):
                k = lut[i, j]
                if k < 0:
                    continue  # stays fill (NaN or 0)

                wj2 = win2[j]
                wk3 = win3[k]

                # numerator: E[ wi1 * wj2 * conj(wk3) ]
                acc_r = 0.0
                acc_i = 0.0
                for t in range(wlen):
                    z = wi1[t] * wj2[t] * wk3[t].conjugate()
                    acc_r += z.real
                    acc_i += z.imag
                num_r = acc_r / wlen
                num_i = acc_i / wlen
                num_abs2 = num_r * num_r + num_i * num_i

                # denominator
                p3 = power3[k]
                if denom == "hosa":
                    den = power1[i] * power2[j] * p3 + eps
                else:  # "hinich" (default)
                    acc = 0.0
                    for t in range(wlen):
                        z = wi1[t] * wj2[t]
                        acc += z.real * z.real + z.imag * z.imag
                    p12 = acc / wlen
                    den = p12 * p3 + eps

                if squared:
                    val = num_abs2 / den
                else:
                    val = math.sqrt(num_abs2 / den)

                out[i, j, w] = val

    return out


def _tv_cross_bicoherence_gpu(
    win1_gpu: "cp.ndarray",
    win2_gpu: "cp.ndarray",
    win3_gpu: "cp.ndarray",
    lut_dev: "cp.ndarray",
    eps: float,
    invalid_to_nan: bool,
    squared: bool,
    denom: str,
) -> "cp.ndarray":
    """
    GPU engine for a single window (three channels).
    """
    nbins, _ = win1_gpu.shape

    # per‑bin powers
    p1 = cp.mean(cp.abs(win1_gpu) ** 2, axis=1)  # (nbins,)
    p2 = cp.mean(cp.abs(win2_gpu) ** 2, axis=1)  # (nbins,)
    p3 = cp.mean(cp.abs(win3_gpu) ** 2, axis=1)  # (nbins,)

    X1 = win1_gpu[:, None, :]   # (nbins, 1, wlen)
    X2 = win2_gpu[None, :, :]   # (1, nbins, wlen)

    idx = lut_dev               # (nbins, nbins)
    mask = idx < 0
    idx_safe = cp.maximum(idx, 0)

    X3 = win3_gpu[idx_safe].copy()  # (nbins, nbins, wlen)
    if invalid_to_nan:
        X3 = cp.where(mask[..., None], cp.nan + 1j * cp.nan, X3)
    else:
        X3[mask] = 0

    num = cp.mean(X1 * X2 * cp.conj(X3), axis=2)  # (nbins, nbins)

    p3_mat = p3[idx_safe]  # (nbins, nbins)
    if not invalid_to_nan:
        # keep denom finite where we set X3=0
        p3_mat = cp.where(mask, 1.0, p3_mat)

    eps_dtype = cp.asarray(eps, dtype=num.real.dtype)

    if denom == "hosa":
        P1 = p1[:, None]
        P2 = p2[None, :]
        den = P1 * P2 * p3_mat + eps_dtype
    else:  # "hinich"
        p12 = cp.mean(cp.abs(X1 * X2) ** 2, axis=2)
        den = p12 * p3_mat + eps_dtype

    if squared:
        bic = cp.abs(num) ** 2 / den
    else:
        bic = cp.abs(num) / cp.sqrt(den)

    return bic


# -----------------------------------------------------------------------------
# 2)  Public API
# -----------------------------------------------------------------------------


def compute(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sig3: np.ndarray,
    fs: int | np.int32,
    f0: float = 1.0,
    f1: float | None = None,
    nbins: int = 256,
    *,
    win_len: int | None = None,
    step: int | None = None,
    use_gpu: bool = False,
    dump_dir: str | Path | None = None,
    dump_every: float | None = None,
    nthreads: int = 8,
    c0: float = 6.0,           # kept for parity; unused
    eps: float = 1e-12,
    normalize: str = "unit_var",
    invalid_to_nan: bool = False,
    squared: bool = True,
    denom: str = "hinich",
    clip_to_unit: bool = True,
    warn_on_clip: bool = True,
) -> Tuple[np.ndarray | List[str], np.ndarray, np.ndarray]:
    """
    Time‑varying **cross‑bicoherence** cube B(f₁,f₂,t) for three channels.

    Parameters
    ----------
    sig1, sig2, sig3 : np.ndarray
        1‑D real signals x(t), y(t), z(t). **Must have identical length**.
    fs : int
        Sampling rate (Hz).
    f0, f1 : float
        CWT low / high cutoff (Hz). If f1 is None, defaults to fs/2 - 1.
    nbins : int
        Number of frequency bins between f0 and f1.
    win_len : int or None
        Window length in samples for bicoherence estimation.
        Default = fs // 2.
    step : int or None
        Hop size in samples between windows. Default = win_len // 2.
    use_gpu : bool
        If True and CuPy is available, use the GPU implementation.
    dump_dir : str or Path or None
        If not None, stream results to .npy chunks in this directory instead
        of keeping the full cube in RAM.
    dump_every : float or None
        Approximate chunk duration in seconds when streaming. Ignored if
        dump_dir is None.
    nthreads : int
        Thread count passed to fcwt.cwt.
    c0 : float
        Kept for API symmetry; not used here.
    eps : float
        Small positive constant to stabilise denominators.
    normalize : {"unit_var", "none"}
        Per-channel amplitude normalisation applied BEFORE the CWT.
        Cross-bicoherence is invariant to a constant rescaling of each
        input (numerator and denominator both scale as the sixth power of
        amplitude and it cancels EXACTLY in real arithmetic), so this does
        not change the estimand. Its sole purpose is numerical: MNE stores
        EEG in volts, so raw amplitudes are O(1e-5) and the band-limited
        denominator power1*power2*p3 is O(1e-30..1e-34), far below any
        sane eps. With such inputs the "+ eps" term dominates the
        denominator and the estimator silently returns num/eps -- an
        UNNORMALISED bispectrum magnitude -- instead of a bicoherence.
        "unit_var" rescales each channel to unit standard deviation, which
        places the denominator near unity regardless of the input's
        physical scale or reference (CAR, CSD, ...), so eps acts only as a
        genuine divide-by-zero guard. "none" preserves the legacy
        behaviour. Default "unit_var".
    invalid_to_nan : bool
        If True, out‑of‑band (i,j) entries are NaN. If False, they are 0.
    squared : bool
        If True (default), return b². If False, return |b|.
    denom : {"hinich", "hosa"}
        Choice of normalisation.
    clip_to_unit : bool
        If True, clamp finite values to [0,1].
    warn_on_clip : bool
        If True, issue a summary warning about clipping.

    Returns
    -------
    B : np.ndarray or list[str]
        If dump_dir is None: B has shape (nbins, nbins, n_windows).
        If dump_dir is not None: list of .npy file paths with chunks.
    freqs : np.ndarray
        Frequency axis (Hz), shape (nbins,).
    tau : np.ndarray
        Time centres of windows (seconds), shape (n_windows,).
    """
    if sig1.ndim != 1 or sig2.ndim != 1 or sig3.ndim != 1:
        raise ValueError("All input signals must be 1‑D.")

    if not (sig1.shape[0] == sig2.shape[0] == sig3.shape[0]):
        raise ValueError("sig1, sig2, sig3 must have the same length (N_samples).")

    if denom not in ("hinich", "hosa"):
        raise ValueError("denom must be 'hinich' or 'hosa'.")

    if normalize not in ("unit_var", "none"):
        raise ValueError("normalize must be 'unit_var' or 'none'.")

    x = np.asarray(sig1, dtype=np.float32, order="C")
    y = np.asarray(sig2, dtype=np.float32, order="C")
    z = np.asarray(sig3, dtype=np.float32, order="C")

    # ------------------------------------------------------------------
    # Per-channel amplitude normalisation (numerical only; see `normalize`
    # in the docstring). Bicoherence is invariant to a constant rescaling
    # of each channel, so dividing by the per-channel std does not change
    # the estimand -- it only lifts the denominator away from `eps` so the
    # result is a true bicoherence rather than an eps-floored bispectrum.
    # Computed in float64 to avoid a second float32 rounding of the scale.
    # ------------------------------------------------------------------
    if normalize == "unit_var":
        # asarray may return the caller's own array (when already float32 &
        # C-contiguous); copy first so we never mutate the input in place.
        x = x.copy(); y = y.copy(); z = z.copy()
        for _s in (x, y, z):
            _sd = float(np.std(_s.astype(np.float64)))
            if _sd > 0.0 and np.isfinite(_sd):
                _s /= np.float32(_sd)
            # a constant or all-zero channel is left untouched; its
            # denominator is genuinely ~0 and eps correctly guards it.

    fs = int(fs)
    f1 = f1 or fs / 2 - 1
    nbins = int(nbins)

    N = x.size
    win_len = win_len or fs // 2
    step = step or win_len // 2

    if win_len < 1:
        raise ValueError("win_len must be at least 1 sample.")
    if step < 1:
        raise ValueError("step must be at least 1 sample.")

    # CWT for each signal
    freqs, coef1 = fcwt.cwt(x, fs, f0, f1, nbins, nthreads=int(nthreads))
    if freqs[0] > freqs[-1]:
        freqs = freqs[::-1].copy()
        coef1 = coef1[::-1, :].copy()

    freqs2, coef2 = fcwt.cwt(y, fs, f0, f1, nbins, nthreads=int(nthreads))
    if freqs2[0] > freqs2[-1]:
        freqs2 = freqs2[::-1].copy()
        coef2 = coef2[::-1, :].copy()

    freqs3, coef3 = fcwt.cwt(z, fs, f0, f1, nbins, nthreads=int(nthreads))
    if freqs3[0] > freqs3[-1]:
        freqs3 = freqs3[::-1].copy()
        coef3 = coef3[::-1, :].copy()

    if freqs2.shape != freqs.shape or not np.allclose(freqs2, freqs):
        raise RuntimeError("Frequency grid mismatch between sig1 and sig2.")
    if freqs3.shape != freqs.shape or not np.allclose(freqs3, freqs):
        raise RuntimeError("Frequency grid mismatch between sig1 and sig3.")

    # LUT & windows
    lut = _build_lookup(freqs)
    if N < win_len:
        starts = np.arange(0, dtype=np.int64)  # no windows
    else:
        starts = np.arange(0, N - win_len + 1, step, dtype=np.int64)
    tau = (starts + win_len / 2) / fs

    # Degenerate case: no windows
    if starts.size == 0:
        if dump_dir is None:
            B = np.empty((nbins, nbins, 0), np.float32)
            return B, freqs, tau
        else:
            return [], freqs, tau

    # streaming setup
    if dump_dir is None:
        chunk_windows = starts.size
    else:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_every = dump_every or 5.0
        # approximate: number of windows per chunk
        chunk_windows = max(1, int(np.ceil(dump_every * fs / step)))

    use_gpu = bool(use_gpu and cp is not None)
    gpu_full = use_gpu and dump_dir is None
    gpu_stream = use_gpu and dump_dir is not None

    if gpu_full:
        coef1_gpu = cp.asarray(coef1)
        coef2_gpu = cp.asarray(coef2)
        coef3_gpu = cp.asarray(coef3)
        lut_dev = cp.asarray(lut)

    file_list: List[str] = []
    chunks: List[np.ndarray] = []

    # clipping stats
    total_low = 0
    total_high = 0
    total_finite = 0

    def _maybe_clip(B_chunk: np.ndarray) -> np.ndarray:
        nonlocal total_low, total_high, total_finite
        if not clip_to_unit:
            return B_chunk
        finite = np.isfinite(B_chunk)
        if not np.any(finite):
            return B_chunk
        finite_vals = B_chunk[finite]
        below = np.count_nonzero(finite_vals < 0.0)
        above = np.count_nonzero(finite_vals > 1.0)
        if below or above:
            np.clip(B_chunk, 0.0, 1.0, out=B_chunk)
        total_low += below
        total_high += above
        total_finite += finite_vals.size
        return B_chunk

    for idx0 in range(0, starts.size, chunk_windows):
        sl = slice(idx0, min(idx0 + chunk_windows, starts.size))
        abs_starts = starts[sl]

        if gpu_full:
            local = abs_starts - starts[0]
            bic_chunks = []
            for s in local:
                bic = _tv_cross_bicoherence_gpu(
                    coef1_gpu[:, s : s + win_len],
                    coef2_gpu[:, s : s + win_len],
                    coef3_gpu[:, s : s + win_len],
                    lut_dev,
                    eps,
                    invalid_to_nan,
                    squared,
                    denom,
                )
                bic_chunks.append(bic)
            B_chunk = cp.stack(bic_chunks, axis=2).get()
            B_chunk = _maybe_clip(B_chunk)

        elif gpu_stream:
            t0 = int(abs_starts[0])
            t1 = int(abs_starts[-1] + win_len)
            slice1 = cp.asarray(coef1[:, t0:t1])
            slice2 = cp.asarray(coef2[:, t0:t1])
            slice3 = cp.asarray(coef3[:, t0:t1])
            lut_dev = cp.asarray(lut)

            local = abs_starts - t0
            bic_chunks = []
            for s in local:
                bic = _tv_cross_bicoherence_gpu(
                    slice1[:, s : s + win_len],
                    slice2[:, s : s + win_len],
                    slice3[:, s : s + win_len],
                    lut_dev,
                    eps,
                    invalid_to_nan,
                    squared,
                    denom,
                )
                bic_chunks.append(bic)
            B_chunk = cp.stack(bic_chunks, axis=2).get()
            del slice1, slice2, slice3
            cp.get_default_memory_pool().free_all_blocks()
            B_chunk = _maybe_clip(B_chunk)

        else:
            local = abs_starts - starts[0]
            B_chunk = _tv_cross_bicoherence_cpu(
                coef1,
                coef2,
                coef3,
                lut,
                local,
                win_len,
                eps,
                invalid_to_nan,
                squared,
                denom,
            )
            B_chunk = _maybe_clip(B_chunk)

        if dump_dir is None:
            chunks.append(B_chunk)
        else:
            fname = dump_dir / f"C_{len(file_list):04d}.npy"
            np.save(fname, B_chunk)
            file_list.append(str(fname))
            del B_chunk

    # emit one warning at the end
    if warn_on_clip and clip_to_unit and (total_low > 0 or total_high > 0):
        pct = 100.0 * (total_low + total_high) / max(1, total_finite)
        warnings.warn(
            (
                "Cross‑bicoherence clipped to [0,1]: "
                f"{total_low} values < 0 and {total_high} values > 1 "
                f"(out of {total_finite} finite, {pct:.4f}% clipped)."
            ),
            RuntimeWarning,
        )

    if dump_dir is None:
        if chunks:
            B = np.concatenate(chunks, axis=2).astype(np.float32, copy=False)
        else:
            B = np.empty((nbins, nbins, 0), np.float32)
        return B, freqs, tau
    else:
        return file_list, freqs, tau


# -----------------------------------------------------------------------------
# 3)  Plotting helper
# -----------------------------------------------------------------------------


import matplotlib.pyplot as plt


def plot_slice(C: np.ndarray, freqs: np.ndarray, tau: np.ndarray, *, w: int = 0):
    """
    Quick visualisation of a single time slice.

    For cross‑bicoherence, interpret the axes as:
        x‑axis: f₁ on channel 1
        y‑axis: f₂ on channel 2
    """
    if w < 0 or w >= C.shape[2]:
        raise IndexError("slice index out of range")
    vmin = 0 if np.nanmin(C) >= 0 else float(np.nanmin(C))
    vmax = 1 if np.nanmax(C) <= 1 else float(np.nanmax(C))
    plt.imshow(
        C[:, :, w],
        extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("f₁ (Hz, channel 1)")
    plt.ylabel("f₂ (Hz, channel 2)")
    plt.title(f"Cross‑bicoherence – window {w}/{C.shape[2]-1}  (t={tau[w]:.2f} s)")
    plt.colorbar(label="Coh.")
    plt.show()
