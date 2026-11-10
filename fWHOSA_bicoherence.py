"""
Time‑Varying Bicoherence (TVBiCoh) – outer‑window CWT, WenCh‑style
==================================================================

Companion to fWHOSA (wavelet bispectrum) but for **bicoherence**:

    b^2(f1,f2) = | E[X(f1) X(f2) X*(f1+f2)] |^2
                 --------------------------------
                 E[|X(f1) X(f2)|^2] · E[|X(f1+f2)|^2]        (Hinich default)

Key points
----------
* Uses the same OUTER/INNER windowing scheme as fWHOSA:
    - OUTER window (outer_win_s): CWT segment length
    - INNER window (inner_win_s): averaging window for bicoherence inside each OUTER

* No Newman D_psi normalisation: for bicoherence the wavelet‑kernel factors
  cancel between numerator and denominator.

* CPU and optional GPU:
    - CPU core is Numba‑accelerated.
    - GPU core uses CuPy, window‑by‑window, like fCWBicoherence_gpu.

* Options:
    - denom = "hinich" (default) or "hosa"
    - squared=True → b^2 (standard); squared=False → b
    - invalid_to_nan: treat out‑of‑band (i,j) where f1+f2 is not representable
      as NaN (True) or 0 (False).
    - clip_to_unit: clamp finite values to [0,1], with a one‑shot warning.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import numba as nb
import fcwt

try:
    import cupy as cp  # optional
except Exception:
    cp = None  # type: ignore

import matplotlib.pyplot as plt

__all__ = ["compute", "plot_slice"]


# -----------------------------------------------------------------------------
# 1)  LUT for triads: K[i,j] = k with f_k ≈ f_i + f_j
# -----------------------------------------------------------------------------

def _build_lookup(freqs: np.ndarray) -> np.ndarray:
    """
    Lookup K[i,j] giving index k with f_k ≈ f_i + f_j, or -1 if out of range.
    """
    nb = int(len(freqs))
    if freqs[0] > freqs[-1]:
        f_min, f_max = float(freqs[-1]), float(freqs[0])
    else:
        f_min, f_max = float(freqs[0]), float(freqs[-1])

    lut = np.full((nb, nb), -1, dtype=np.int32)

    for i in range(nb):
        fi = float(freqs[i])
        f3_target = fi + freqs
        valid_mask = (f3_target >= f_min) & (f3_target <= f_max)
        valid_indices = np.where(valid_mask)[0]
        for j in valid_indices:
            f3 = float(f3_target[j])
            # nearest bin for f3
            lut[i, j] = int(np.argmin(np.abs(freqs - f3)))

    return lut


# -----------------------------------------------------------------------------
# 2)  Bicoherence engines (CPU / GPU) for ONE CWT segment
# -----------------------------------------------------------------------------

@nb.njit(parallel=True, fastmath=True)
def _tv_bicoherence_cpu(
    coef: np.ndarray,
    lut: np.ndarray,
    starts: np.ndarray,
    wlen: int,
    eps: float,
    invalid_to_nan: bool,
    squared: bool,
    denom_mode: int,
) -> np.ndarray:
    """
    CPU time‑varying bicoherence engine for a single OUTER CWT segment.

    coef   : (nbins, T_seg) complex64
    lut    : (nbins, nbins) int32 (k index or -1)
    starts : (n_wins,) int64 – start indices of INNER windows (relative to segment)
    wlen   : INNER window length (samples)
    denom_mode : 0 → Hinich, 1 → HOSA
    """
    nbins = coef.shape[0]
    n_wins = starts.size

    out = np.empty((nbins, nbins, n_wins), np.float32)
    fill = np.float32(np.nan if invalid_to_nan else 0.0)
    out[:] = fill

    for w in nb.prange(n_wins):
        s = int(starts[w])
        win = coef[:, s: s + wlen]  # (nbins, wlen)

        # mean power per bin over this INNER window
        power = np.empty(nbins, np.float32)
        for i in range(nbins):
            acc = 0.0
            for t in range(wlen):
                z = win[i, t]
                acc += z.real * z.real + z.imag * z.imag
            power[i] = acc / wlen

        for i in range(nbins):
            wi = win[i]
            for j in range(i, nbins):
                k = lut[i, j]
                if k < 0:
                    # out-of-band triad, leave as fill (NaN or 0)
                    continue

                # numerator: E[ wi * wj * conj(wk) ]
                acc_r = 0.0
                acc_i = 0.0
                for t in range(wlen):
                    z = wi[t] * win[j, t] * win[k, t].conjugate()
                    acc_r += z.real
                    acc_i += z.imag
                num_r = acc_r / wlen
                num_i = acc_i / wlen
                num_abs2 = num_r * num_r + num_i * num_i

                # denominator parts
                p3 = power[k]
                if denom_mode == 1:  # HOSA: product of powers
                    den = power[i] * power[j] * p3 + eps
                else:  # Hinich: E[|X1 X2|^2] * E[|X3|^2]
                    acc = 0.0
                    for t in range(wlen):
                        z = wi[t] * win[j, t]
                        acc += z.real * z.real + z.imag * z.imag
                    p12 = acc / wlen
                    den = p12 * p3 + eps

                if squared:
                    val = num_abs2 / den
                else:
                    val = math.sqrt(num_abs2 / den)

                out[i, j, w] = np.float32(val)
                out[j, i, w] = np.float32(val)

    return out


def _tv_bicoherence_gpu(
    win_gpu: "cp.ndarray",
    lut_dev: "cp.ndarray",
    eps: float,
    invalid_to_nan: bool,
    squared: bool,
    denom_mode: int,
) -> "cp.ndarray":
    """
    GPU bicoherence for a single INNER window within one OUTER segment.

    win_gpu : (nbins, wlen) complex64
    lut_dev : (nbins, nbins) int32 on device
    """
    nbins, _ = win_gpu.shape

    # mean power per bin
    power = cp.mean(cp.abs(win_gpu) ** 2, axis=1)  # (nbins,)

    X1 = win_gpu[:, None, :]       # (nbins, 1, wlen)
    X2 = win_gpu[None, :, :]       # (1, nbins, wlen)
    idx = lut_dev                  # (nbins, nbins)
    mask = idx < 0
    idx_safe = cp.maximum(idx, 0)  # use bin 0 for invalids, then mask

    X3 = win_gpu[idx_safe].copy()  # (nbins, nbins, wlen)

    if invalid_to_nan:
        X3 = cp.where(mask[..., None], cp.nan + 1j * cp.nan, X3)
    else:
        X3[mask] = 0

    num = cp.mean(X1 * X2 * cp.conj(X3), axis=2)  # (nbins, nbins)

    p3 = power[idx_safe]  # (nbins, nbins)
    if not invalid_to_nan:
        # Where we force numerator to 0, keep a finite denom
        p3 = cp.where(mask, 1.0, p3)

    eps_dtype = cp.asarray(eps, dtype=num.real.dtype)

    if denom_mode == 1:  # HOSA
        p1 = power[:, None]
        p2 = power[None, :]
        den = p1 * p2 * p3 + eps_dtype
    else:                # Hinich
        p12 = cp.mean(cp.abs(X1 * X2) ** 2, axis=2)  # (nbins, nbins)
        den = p12 * p3 + eps_dtype

    if squared:
        bic = cp.abs(num) ** 2 / den
    else:
        bic = cp.abs(num) / cp.sqrt(den)

    # Enforce symmetry
    i, j = cp.triu_indices(nbins, 1)
    bic[j, i] = bic[i, j]

    return bic


# -----------------------------------------------------------------------------
# 3)  Main entry point – OUTER CWT window + INNER bicoherence windows
# -----------------------------------------------------------------------------

def compute(
    signal: np.ndarray,
    fs: int,
    *,
    f0: float = 1.0,
    f1: float | None = None,
    nbins: int = 256,
    inner_win_s: float = 1.0,
    inner_step_s: float | None = None,
    outer_win_s: float = 5.0,
    use_gpu: bool = True,
    dump_dir: str | None = None,
    nthreads: int = 8,
    c0: float = 6.0,              # kept for parity with other codes; unused here
    eps: float = 1e-12,
    invalid_to_nan: bool = False,
    squared: bool = True,
    denom: str = "hinich",
    clip_to_unit: bool = True,
    warn_on_clip: bool = True,
) -> Tuple[np.ndarray | List[str], np.ndarray, np.ndarray]:
    """
    Time‑varying **bicoherence** with explicit OUTER CWT segments.

    Time scales
    -----------
    outer_win_s : float
        Length (seconds) of the CWT analysis (OUTER) window.
        CWT is computed on these segments only. Default 5.0 s.

    inner_win_s : float
        Length (seconds) of the bicoherence averaging window INSIDE each
        outer window. Determines time resolution of bicoherence. Default 1.0 s.

    inner_step_s : float or None
        Step (seconds) between inner windows. If None, defaults to
        50% overlap (inner_win_s / 2).

    Parameters
    ----------
    signal : 1‑D array_like
        Input time series.
    fs : int
        Sampling rate (Hz).
    f0, f1 : float
        CWT min / max frequencies (Hz). If f1 is None, uses fs/2.
    nbins : int
        Number of CWT frequency bins.
    use_gpu : bool
        Use CuPy GPU kernels if available.
    dump_dir : str or None
        If not None, stream segments to disk as .npy files instead of
        keeping the full cube in memory.
    nthreads : int
        Threads for fcwt.cwt and Numba.
    c0 : float
        Unused here; kept only for API parity with fWHOSA / bispectrum code.
    eps : float
        Small positive constant to stabilise denominators.
    invalid_to_nan : bool
        If True, out‑of‑band (i,j) triads are NaN; if False, they are 0.
    squared : bool
        If True (default), return b^2; if False, return b.
    denom : {"hinich", "hosa"}
        Denominator style:
          "hinich":  E|X1 X2|^2 · E|X3|^2  (robust, default)
          "hosa"  :  E|X1|^2 · E|X2|^2 · E|X3|^2
    clip_to_unit : bool
        If True, clamp finite values to [0,1].
    warn_on_clip : bool
        If True, emit a one‑shot RuntimeWarning summarising clipping stats.

    Returns
    -------
    C : np.ndarray or list[str]
        If dump_dir is None:
            C has shape (nbins, nbins, n_windows), dtype float32.
        If dump_dir is not None:
            list of .npy file paths (one per OUTER segment).
    freqs : np.ndarray
        Frequency axis (Hz), shape (nbins,).
    tau : np.ndarray
        Time centres of all INNER windows (seconds), shape (n_windows,).
    """
    # -----------------------------
    # 0) Pre‑checks & time grid
    # -----------------------------
    sig = np.asarray(signal, dtype=np.float32).ravel()
    N = sig.size
    if N == 0:
        raise ValueError("Input signal is empty.")

    fs = int(fs)
    if f1 is None:
        f1 = fs / 2.0

    # Inner window (samples)
    inner_len = int(round(inner_win_s * fs))
    if inner_len < 1:
        inner_len = 1

    # Inner step (samples)
    if inner_step_s is None:
        inner_step = max(1, inner_len // 2)  # 50% overlap
    else:
        inner_step = int(round(inner_step_s * fs))
        if inner_step < 1:
            inner_step = 1

    # Outer window (samples)
    outer_len = int(round(outer_win_s * fs))
    if outer_len < inner_len:
        outer_len = inner_len

    # Not enough samples for even one inner window: just build freqs & return empty
    if N < inner_len:
        freqs, _ = fcwt.cwt(sig, int(fs), float(f0), float(f1), int(nbins), int(nthreads))
        if freqs[0] > freqs[-1]:
            freqs = freqs[::-1].copy()
        C_empty = np.zeros((len(freqs), len(freqs), 0), dtype=np.float32)
        tau_empty = np.zeros((0,), dtype=np.float32)
        return C_empty, freqs, tau_empty

    # Global INNER starts and time centres
    starts = np.arange(0, N - inner_len + 1, inner_step, dtype=np.int64)
    n_wins = starts.size
    tau = (starts + inner_len / 2.0) / float(fs)

    # Denominator mode
    d = denom.lower()
    if d == "hinich":
        denom_mode = 0
    elif d == "hosa":
        denom_mode = 1
    else:
        raise ValueError("denom must be 'hinich' or 'hosa'")

    # GPU availability
    gpu_avail = bool(use_gpu and (cp is not None))

    # Output streaming / buffering
    if dump_dir is not None:
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        file_list: List[str] = []
    else:
        chunks: List[np.ndarray] = []

    freqs_global: np.ndarray | None = None
    lut: np.ndarray | None = None
    lut_dev = None  # GPU LUT

    # Clipping stats
    total_low = 0
    total_high = 0
    total_finite = 0

    def _maybe_clip(chunk: np.ndarray) -> np.ndarray:
        nonlocal total_low, total_high, total_finite
        if not clip_to_unit:
            return chunk
        finite = np.isfinite(chunk)
        if not np.any(finite):
            return chunk
        vals = chunk[finite]
        below = np.count_nonzero(vals < 0.0)
        above = np.count_nonzero(vals > 1.0)
        if below or above:
            np.clip(chunk, 0.0, 1.0, out=chunk)
        total_low += below
        total_high += above
        total_finite += vals.size
        return chunk

    # -----------------------------
    # 1) Group INNER windows into OUTER segments
    # -----------------------------
    i = 0
    seg_counter = 0

    while i < n_wins:
        # First inner window in this OUTER segment
        s0 = int(starts[i])
        max_start = s0 + (outer_len - inner_len)

        # All inner windows with start <= max_start go into this segment
        j = int(np.searchsorted(starts, max_start, side="right"))
        if j <= i:
            j = i + 1
        j_end = min(j, n_wins) - 1

        s_idx = starts[i: j_end + 1]  # inner window starts in this OUTER segment

        # Segment coverage [seg_start, seg_end)
        seg_start = int(s_idx[0])
        seg_end = int(s_idx[-1] + inner_len)
        if seg_end > N:
            seg_end = N

        # Slice the signal for this OUTER window
        sig_seg = sig[seg_start:seg_end]

        # -------------------------
        # 1a) CWT on this OUTER segment
        # -------------------------
        if freqs_global is None:
            freqs_seg, coef_seg = fcwt.cwt(
                sig_seg, int(fs), float(f0), float(f1), int(nbins), int(nthreads)
            )
            # Ensure low→high order
            if freqs_seg[0] > freqs_seg[-1]:
                freqs_seg = freqs_seg[::-1].copy()
                coef_seg = coef_seg[::-1].copy()

            freqs_global = freqs_seg.astype(np.float32, copy=True)
            lut = _build_lookup(freqs_global)

            if gpu_avail and lut_dev is None:
                lut_dev = cp.asarray(lut)
        else:
            # Subsequent segments: we assume fcwt uses same frequency grid
            freqs_seg, coef_seg = fcwt.cwt(
                sig_seg, int(fs), float(f0), float(f1), int(nbins), int(nthreads)
            )
            if freqs_seg[0] > freqs_seg[-1]:
                freqs_seg = freqs_seg[::-1].copy()
                coef_seg = coef_seg[::-1].copy()
            # we ignore freqs_seg and rely on freqs_global

        # Local INNER starts inside this segment
        local_starts = (s_idx - seg_start).astype(np.int64)

        # -------------------------
        # 1b) Bicoherence on INNER windows within this OUTER segment
        # -------------------------
        if gpu_avail:
            if lut_dev is None and lut is not None:
                lut_dev = cp.asarray(lut)

            d_slice = cp.asarray(coef_seg)
            nbins_loc = d_slice.shape[0]
            n_loc = local_starts.size

            bic_chunks = []
            for s_local in local_starts:
                w = int(s_local)
                win_gpu = d_slice[:, w: w + inner_len]
                bic = _tv_bicoherence_gpu(
                    win_gpu, lut_dev, eps, invalid_to_nan, squared, denom_mode
                )
                bic_chunks.append(bic)

            B_chunk = cp.stack(bic_chunks, axis=2).get().astype(np.float32, copy=False)
            del d_slice
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU engine directly on this segment
            B_chunk = _tv_bicoherence_cpu(
                coef_seg.astype(np.complex64, copy=False),
                lut,
                local_starts,
                inner_len,
                eps,
                invalid_to_nan,
                squared,
                denom_mode,
            )

        B_chunk = _maybe_clip(B_chunk)

        # Store or dump
        if dump_dir is not None:
            fn = dump_path / f"C_{seg_counter:05d}.npy"
            np.save(fn, B_chunk)
            file_list.append(str(fn))
        else:
            chunks.append(B_chunk)

        seg_counter += 1
        # Advance to next group of INNER windows
        i = j_end + 1

    # -----------------------------
    # 2) Final assembly
    # -----------------------------
    if freqs_global is None:
        # Should only happen if N < inner_len, but we guarded above
        freqs_global = np.zeros((0,), dtype=np.float32)

    # Emit one clipping warning at the end
    if warn_on_clip and clip_to_unit and (total_low > 0 or total_high > 0):
        pct = 100.0 * (total_low + total_high) / max(1, total_finite)
        warnings.warn(
            f"Bicoherence clipped to [0,1]: {total_low} values < 0 and "
            f"{total_high} values > 1 (out of {total_finite} finite, {pct:.4f}% clipped).",
            RuntimeWarning,
        )

    if dump_dir is not None:
        return file_list, freqs_global, tau

    if len(chunks) == 0:
        C_all = np.zeros((len(freqs_global), len(freqs_global), 0), dtype=np.float32)
    else:
        C_all = np.concatenate(chunks, axis=2)

    return C_all, freqs_global, tau


# -----------------------------------------------------------------------------
# 4)  Convenience plotting
# -----------------------------------------------------------------------------

def plot_slice(C: np.ndarray, freqs: np.ndarray, tau: np.ndarray, *, w: int = 0):
    """
    Plot a single time slice C[:,:,w] of the bicoherence cube.

    C : (nbins, nbins, n_windows), float32
    freqs : (nbins,)
    tau : (n_windows,)
    w : int
        Window index.
    """
    if w < 0 or w >= C.shape[2]:
        raise IndexError("slice index out of range")

    vmin = 0.0 if np.nanmin(C) >= 0 else float(np.nanmin(C))
    vmax = 1.0 if np.nanmax(C) <= 1 else float(np.nanmax(C))

    plt.imshow(
        C[:, :, w],
        extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("f₁ (Hz)")
    plt.ylabel("f₂ (Hz)")
    plt.title(f"Bicoherence – inner window {w}/{C.shape[2]-1}  (t = {tau[w]:.2f} s)")
    plt.colorbar(label="Coh.")
    plt.show()
