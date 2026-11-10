"""
Time-Varying Bispectrum (TVB) – GPU & Streaming (V5, outer-window CWT)
=====================================================================

Fast wavelet-bispectrum implementation with explicit outer (CWT) and
inner (bispectrum) time scales.

Mental model
------------
1) Signal in
2) Build LUT and D_psi(lambda) (Newman normalisation)
3) For each OUTER window (e.g. 5 s):
     - compute CWT on that 5 s segment
     - inside that, slide INNER windows (e.g. 1 s) and compute
       the triple product
4) Stack all inner-window slices into a cube B(f1,f2,t)

Key parameters
--------------
- outer_win_s (default 5.0 s)
    Length of the big analysis window on which CWT is computed.
- inner_win_s (default 1.0 s)
    Length of the bispectrum averaging window, sliding inside each
    outer window (time resolution of B).

We never store CWT for the full record, only for one 5 s outer segment
at a time, so RAM usage is controlled by outer_win_s and nbins.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import fcwt
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
from functools import lru_cache

try:
    import cupy as cp
except ImportError:
    cp = None

__all__ = ["compute", "plot_slice"]

# -----------------------------------------------------------------------------
# 1)  Newman Normalization (D_psi) – cached
# -----------------------------------------------------------------------------

# Key: (nbins, freq_start, freq_end, bandwidth_c0)
_D_CACHE: dict[tuple, np.ndarray] = {}


@nb.njit(fastmath=True)
def _morlet_hat_val(r: float, fb: float) -> float:
    """
    Morlet wavelet shape in positive frequency domain:

        psi_hat(r) ∝ exp( -2 * (pi * fb * (r-1))^2 )

    Overall constant is dropped (cancels between CWT and D_psi).
    """
    if r <= 0:
        return 0.0
    # 19.7392088 ≈ 2 * pi^2
    return np.exp(-19.7392088 * (fb * (r - 1)) ** 2)


def _compute_D_integrand(r2: float, r1: float, lam: float, fb: float) -> float:
    """
    Integrand for Newman's D_psi(lambda):

        D_psi(lambda) =
            ∫∫ psî(r1) psî(r2) psî(lambda r1 + (1-lambda) r2)
                / (r1 r2) dr1 dr2
    """
    p1 = _morlet_hat_val(r1, fb)
    p2 = _morlet_hat_val(r2, fb)
    p3 = _morlet_hat_val(lam * r1 + (1.0 - lam) * r2, fb)
    return (p1 * p2 * p3) / (r1 * r2)


@lru_cache(maxsize=32)
def _get_D_interpolator(fb: float):
    """
    Compute and cache D_psi(lambda) for bandwidth fb.
    """
    print(f"[Info] Computing D_psi(lambda) curve for c0={fb}...")

    lam_grid = np.linspace(0.01, 0.99, 40)
    D_vals = []
    # Truncated r-domain (sufficient for fCWT Morlet)
    limit_low, limit_high = 0.1, 3.0

    for lam in lam_grid:
        val, _ = dblquad(
            _compute_D_integrand,
            limit_low,
            limit_high,
            lambda x: limit_low,
            lambda x: limit_high,
            args=(lam, fb),
            epsabs=1e-3,
            epsrel=1e-3,
        )
        D_vals.append(val)

    # Clamp to nearest interior value at endpoints
    return interp1d(
        lam_grid,
        D_vals,
        kind="cubic",
        bounds_error=False,
        fill_value=(D_vals[0], D_vals[-1]),
    )


def _build_norm_matrix_cached(freqs: np.ndarray, fb: float) -> np.ndarray:
    """
    Build or retrieve N[i,j] = D_psi(lambda(i,j)),
    where lambda = f1 / (f1 + f2).
    """
    key = (int(len(freqs)), float(freqs[0]), float(freqs[-1]), float(fb))

    if key in _D_CACHE:
        return _D_CACHE[key]

    D_interp = _get_D_interpolator(fb)

    F1, F2 = np.meshgrid(freqs, freqs, indexing="ij")
    Sum_F = F1 + F2

    Lambda_mat = np.divide(F1, Sum_F, out=np.zeros_like(Sum_F), where=Sum_F != 0.0)
    Norm_mat = D_interp(Lambda_mat).astype(np.float32)

    _D_CACHE[key] = Norm_mat
    return Norm_mat



# -----------------------------------------------------------------------------
# 2)  Bispectrum Engines (CPU / GPU)
# -----------------------------------------------------------------------------

def _build_lookup(freqs: np.ndarray) -> np.ndarray:
    """
    Lookup K[i,j] giving index k with f_k ≈ f_i + f_j, or -1 if out of range.
    """
    nb = len(freqs)
    if freqs[0] > freqs[-1]:
        f_min, f_max = freqs[-1], freqs[0]
    else:
        f_min, f_max = freqs[0], freqs[-1]

    lut = np.full((nb, nb), -1, dtype=np.int32)

    for i in range(nb):
        fi = freqs[i]
        f3_target = fi + freqs
        valid_mask = (f3_target >= f_min) & (f3_target <= f_max)
        valid_indices = np.where(valid_mask)[0]
        for j in valid_indices:
            f3 = f3_target[j]
            lut[i, j] = int(np.argmin(np.abs(freqs - f3)))

    return lut


def _tv_bispec_gpu(
    coef_gpu: "cp.ndarray",
    lut: np.ndarray,
    starts: np.ndarray,
    wlen: int,
) -> np.ndarray:
    """
    GPU triple-product engine for one OUTER CWT segment.

    coef_gpu : (nbins, T_seg)
    lut      : (nbins, nbins) (CPU array)
    starts   : (n_wins,) start indices of INNER windows (relative to segment)
    wlen     : inner window length (samples)
    """
    nbins, T_seg = coef_gpu.shape
    n_wins = int(starts.size)

    out = cp.empty((nbins, nbins, n_wins), dtype=cp.complex64)

    d_lut = cp.asarray(lut)
    valid_mask = d_lut >= 0
    lut_clip = cp.clip(d_lut, 0, nbins - 1)

    for w in range(n_wins):
        s = int(starts[w])
        win = coef_gpu[:, s : s + wlen]  # (nbins, wlen)

        W_k = win[lut_clip]  # (nbins, nbins, wlen)
        W_k[~valid_mask] = 0

        TP = (
            win[:, None, :] *        # (nbins, 1, wlen)
            win[None, :, :] *        # (1, nbins, wlen)
            cp.conj(W_k)             # (nbins, nbins, wlen)
        )
        out[..., w] = cp.mean(TP, axis=2)

    return cp.asnumpy(out)


@nb.njit(parallel=True, fastmath=True)
def _tv_bispec_cpu(
    coeff: np.ndarray, lut: np.ndarray, starts: np.ndarray, wlen: int
) -> np.ndarray:
    """
    CPU triple-product engine for one OUTER CWT segment.

    coeff : (nbins, T_seg)
    lut   : (nbins, nbins)
    starts: (n_wins,)
    wlen  : int
    """
    nbins = coeff.shape[0]
    n_wins = starts.size
    out = np.zeros((nbins, nbins, n_wins), np.complex64)

    for w in nb.prange(n_wins):
        s = int(starts[w])
        win = coeff[:, s : s + wlen]
        for i in range(nbins):
            for j in range(i, nbins):
                k = lut[i, j]
                if k >= 0:
                    acc = 0.0j
                    for t in range(wlen):
                        acc += win[i, t] * win[j, t] * np.conj(win[k, t])
                    val = acc / wlen
                    out[i, j, w] = val
                    out[j, i, w] = val
    return out


# -----------------------------------------------------------------------------
# 3)  Main Entry Point (outer-window CWT, inner-window bispectrum)
# -----------------------------------------------------------------------------

def compute(
    signal: np.ndarray,
    fs: int,
    *,
    f0: float = 1.0,
    f1: float | None = None,
    nbins: int = 256,
    c0: float = 2.0,
    inner_win_s: float = 1.0,
    inner_step_s: float | None = None,
    outer_win_s: float = 5.0,
    use_gpu: bool = True,
    dump_dir: str | None = None,
    nthreads: int = 8,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute time-varying wavelet bispectrum with an OUTER CWT window.

    Time scales
    -----------
    outer_win_s : float
        Length (seconds) of the CWT analysis window. CWT is computed on
        5 s segments that slide along the record. Default 5.0 s.

    inner_win_s : float
        Length (seconds) of the bispectrum averaging window INSIDE each
        outer window. Default 1.0 s.

    inner_step_s : float or None
        Step (seconds) between inner windows. If None, defaults to
        50% overlap (inner_win_s / 2).

    We group INNER windows into OUTER segments such that all inner
    windows in a segment live within a single CWT support of at most
    outer_win_s seconds. We then compute CWT only on that segment,
    compute bispectrum for those inner windows, discard CWT, and move on.

    Returns
    -------
    B : np.ndarray or list[str]
        If dump_dir is None:
            B has shape (nbins, nbins, n_windows) (complex64).
        If dump_dir is not None:
            list of .npy file paths, one per outer segment.
    freqs : np.ndarray
        Frequency axis (Hz), shape (nbins,).
    tau : np.ndarray
        Time centres of inner windows (seconds), shape (n_windows,).
    """
    # -----------------------------
    # 0) Time grid and sanity checks
    # -----------------------------
    sig = np.asarray(signal, dtype=np.float32).ravel()
    N = sig.size
    if N == 0:
        raise ValueError("Input signal is empty.")

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

    # Not enough samples for even one inner window
    if N < inner_len:
        freqs, _ = fcwt.cwt(sig, int(fs), f0, f1, int(nbins), int(nthreads))
        if freqs[0] > freqs[-1]:
            freqs = freqs[::-1].copy()
        B_empty = np.zeros((len(freqs), len(freqs), 0), dtype=np.complex64)
        tau_empty = np.zeros((0,), dtype=np.float32)
        return B_empty, freqs, tau_empty

    # Global inner window starts
    starts = np.arange(0, N - inner_len + 1, inner_step, dtype=np.int64)
    n_wins = starts.size
    tau = (starts + inner_len / 2.0) / float(fs)

    # -----------------------------
    # 1) Pre-allocation for outputs
    # -----------------------------
    gpu_avail = bool(use_gpu and (cp is not None))

    if dump_dir is not None:
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        file_list: list[str] = []
    else:
        chunks: list[np.ndarray] = []

    freqs_global: np.ndarray | None = None
    lut: np.ndarray | None = None
    norm_mat: np.ndarray | None = None
    normalization_c0 = c0

    # -----------------------------
    # 2) Group inner windows into OUTER CWT segments
    # -----------------------------
    i = 0
    while i < n_wins:
        # First inner window in this outer segment
        s0 = int(starts[i])
        max_start = s0 + (outer_len - inner_len)

        # All inner windows with start <= max_start go into this segment
        j = int(np.searchsorted(starts, max_start, side="right"))
        if j <= i:
            j = i + 1
        j_end = min(j, n_wins) - 1

        s_idx = starts[i : j_end + 1]  # inner window starts in this segment

        # Segment coverage in samples [seg_start, seg_end)
        seg_start = int(s_idx[0])
        seg_end = int(s_idx[-1] + inner_len)
        if seg_end > N:
            seg_end = N

        # Slice the signal for this outer window
        sig_seg = sig[seg_start:seg_end]

        # -------------------------
        # 2a) CWT on this outer segment
        # -------------------------
        if freqs_global is None:
            # First segment: also infer D_psi bandwidth default if needed
            try:
                freqs_seg, coef_seg = fcwt.cwt(
                    sig_seg, int(fs), f0, f1, int(nbins), int(nthreads)
                )
            except TypeError:
                detected_default = 2.0
                if normalize:
                    print("[Warning] fCWT wrapper does not support custom c0.")
                    print(f"          -> Forcing normalization c0={detected_default} "
                          "to match engine default.")
                freqs_seg, coef_seg = fcwt.cwt(
                    sig_seg, int(fs), f0, f1, int(nbins), int(nthreads)
                )
                normalization_c0 = detected_default

            # Ensure low→high frequency order
            if freqs_seg[0] > freqs_seg[-1]:
                freqs_seg = freqs_seg[::-1].copy()
                coef_seg = coef_seg[::-1].copy()

            freqs_global = freqs_seg
            lut = _build_lookup(freqs_global)

            if normalize:
                norm_mat = _build_norm_matrix_cached(freqs_global, normalization_c0)

        else:
            # Subsequent segments: we already know freqs_global, lut, norm_mat
            # Just compute CWT and assume fCWT uses same frequency grid
            freqs_seg, coef_seg = fcwt.cwt(
                sig_seg, int(fs), f0, f1, int(nbins), int(nthreads)
            )
            if freqs_seg[0] > freqs_seg[-1]:
                freqs_seg = freqs_seg[::-1].copy()
                coef_seg = coef_seg[::-1].copy()
            # We ignore freqs_seg and rely on freqs_global

        # Local inner window starts inside this segment
        local_starts = (s_idx - seg_start).astype(np.int64)

        # -------------------------
        # 2b) Triple product on INNER windows inside this outer segment
        # -------------------------
        if gpu_avail:
            d_slice = cp.asarray(coef_seg)
            b_chunk = _tv_bispec_gpu(
                d_slice,
                lut,
                local_starts,
                inner_len,
            )
            del d_slice
            cp.get_default_memory_pool().free_all_blocks()
        else:
            b_chunk = _tv_bispec_cpu(coef_seg, lut, local_starts, inner_len)
        

        # Newman normalization
        if normalize:
            b_chunk /= norm_mat[..., None]

        # Store or dump
        if dump_dir is not None:
            fn = dump_path / f"B_{i:05d}.npy"
            np.save(fn, b_chunk)
            file_list.append(str(fn))
        else:
            chunks.append(b_chunk)

        # Advance to next group of inner windows
        i = j_end + 1

    # -----------------------------
    # 3) Final assembly
    # -----------------------------
    if freqs_global is None:
        # Should only happen if we bailed early
        freqs_global = np.zeros((0,), dtype=np.float32)

    if dump_dir is not None:
        return file_list, freqs_global, tau

    if len(chunks) == 0:
        B_all = np.zeros((len(freqs_global), len(freqs_global), 0),
                         dtype=np.complex64)
    else:
        B_all = np.concatenate(chunks, axis=2)

    return B_all, freqs_global, tau


# -----------------------------------------------------------------------------
# 4)  Convenience plotting
# -----------------------------------------------------------------------------

def plot_slice(B: np.ndarray, freqs: np.ndarray, tau: np.ndarray, *, w: int = 0):
    """
    Plot |B(f1,f2, t_w)| in dB for a single inner window index w.
    """
    if w < 0 or w >= B.shape[2]:
        raise IndexError("slice index out of range")

    slice_db = 20.0 * np.log10(np.abs(B[:, :, w]) + 1e-12)

    plt.imshow(
        slice_db,
        extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.xlabel("f₁ (Hz)")
    plt.ylabel("f₂ (Hz)")
    plt.title(
        f"|Bispectrum| – inner window {w}/{B.shape[2]-1} "
        f"(t = {tau[w]:.2f} s)"
    )
    plt.colorbar(label="dB")
    plt.show()
