"""
Time‑Varying Bispectrum (TVB) – GPU & Streaming
==============================================
Fast wavelet‑bispectrum implementation that can **optionally**
run the triple‑product on the GPU and/or stream chunks of the
output cube to disk to avoid RAM / VRAM overflow.

Usage
-----
>>> files, freqs, tau = compute(sig, fs, use_gpu=True,
...                             dump_dir="bispec", dump_every=5)
>>> # files = ["bispec/B_0000.npy", "bispec/B_0001.npy", ...]

API summary
-----------
compute(..., use_gpu=True, dump_dir=Path(...), dump_every=5.0)
  * *No streaming* → (B, freqs, tau) where B has shape (nb, nb, nWins)
  * *Streaming*   → (file_list, freqs, tau) – list of dumped .npy files

Author: Ali Alsuwaidi 
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import numba as nb
import fcwt

try:
    import cupy as cp  # optional
except ImportError:  # keep API even if CuPy missing
    cp = None  # type: ignore

__all__ = ["compute", "plot_slice"]

# -----------------------------------------------------------------------------
# 1)  Helper utilities
# -----------------------------------------------------------------------------

def _build_lookup(freqs: np.ndarray) -> np.ndarray:
    """
    Return an (nbins × nbins) int32 LUT mapping (i, j) → k such that
    freqs[k] ≈ freqs[i] + freqs[j].  Out-of-band (i.e., sums outside
    [min(freqs), max(freqs)]) are marked with -1.
    """
    nb = len(freqs)
    # determine band edges regardless of monotonic direction
    f_min, f_max = (freqs[-1], freqs[0]) if freqs[0] > freqs[-1] else (freqs[0], freqs[-1])

    lut = np.full((nb, nb), -1, dtype=np.int32)
    for i in range(nb):
        fi = freqs[i]
        for j in range(nb):
            f3 = fi + freqs[j]
            if f3 < f_min or f3 > f_max:
                continue
            # nearest-bin match on the existing (possibly non-uniform) grid
            lut[i, j] = int(np.argmin(np.abs(freqs - f3)))
    return lut



@nb.njit(parallel=True, fastmath=True)
def _tv_bispec_cpu(coeff: np.ndarray, lut: np.ndarray,
                   starts: np.ndarray, wlen: int) -> np.ndarray:
    nbins, _ = coeff.shape
    out = np.zeros((nbins, nbins, starts.size), np.complex64)

    for w in nb.prange(starts.size):
        s = starts[w]
        win = coeff[:, s:s + wlen]
        for i in range(nbins):
            x1 = win[i]
            for j in range(i, nbins):
                k = lut[i, j]
                if k < 0:
                    continue
                val = np.mean(x1 * win[j] * np.conj(win[k]))
                out[i, j, w] = val
                out[j, i, w] = val
    return out


def _tv_bispec_gpu(coef_gpu: "cp.ndarray", lut: np.ndarray,
                   starts: np.ndarray, wlen: int) -> np.ndarray:
    """Same idea, but on CuPy arrays – ≫10× faster than Python loops."""
    nbins, _ = coef_gpu.shape
    out = cp.empty((nbins, nbins, starts.size), dtype=cp.complex64)

    # ---- hoist invariant GPU data ------------------------------------
    lut_dev = cp.asarray(lut)
    valid = lut_dev >= 0                  # mask for in-band pairs
    idx_safe = cp.maximum(lut_dev, 0)     # safe indices for -1 entries
    tri_i, tri_j = cp.triu_indices(nbins, 1)

    for w, s in enumerate(starts):
        win = coef_gpu[:, s:s + wlen]                 # (nbins, wlen)
        X1, X2 = win[:, None, :], win[None, :, :]
        X3 = win[idx_safe]                            # (nbins, nbins, wlen)
        X3[~valid] = 0
        out[..., w] = cp.mean(X1 * X2 * cp.conj(X3), axis=2)

        # enforce symmetry once per slice
        out[tri_j, tri_i, w] = out[tri_i, tri_j, w]

    return cp.asnumpy(out)



# -----------------------------------------------------------------------------
# 2)  Public API
# -----------------------------------------------------------------------------

def compute(signal: np.ndarray,
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
            c0: float = 6.0) -> Tuple[np.ndarray | List[str], np.ndarray, np.ndarray]:
    """Compute a time‑varying bispectrum cube, optionally streamed / GPU‑ised.

    Parameters
    ----------
    signal : 1‑D ndarray (float32 recommended)
    fs     : sample‑rate (Hz)
    f0, f1 : band limits (Hz).  f1 defaults to Nyquist.
    nbins  : number of frequency bins.
    win_len: window length (samples).  Defaults to fs//2.
    step   : hop size (samples).  Defaults to win_len//2.
    use_gpu: run inner loop on CuPy / CUDA (if available).
    dump_dir: path ⇒ stream chunks to disk instead of one big cube.
    dump_every: seconds per chunk (default 5 s if dump_dir given).

    Returns
    -------
    *No streaming* – ``B, freqs, tau`` where B has shape (nb, nb, nWins)
    *With streaming* – ``file_list, freqs, tau`` (list of .npy paths)
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be 1‑D")

    sig = np.asarray(signal, dtype=np.float32, order="C")
    fs = int(fs)
    f1 = f1 or fs / 2 - 1
    nbins = int(nbins)
    win_len = win_len or fs // 2
    if win_len > sig.size:
        raise ValueError(f"win_len ({win_len}) exceeds signal length ({sig.size}).")
    step = step or win_len // 2

    # ---- Continuous Wavelet Transform ---------------------------------
    freqs, coef = fcwt.cwt(sig, fs, f0, f1, nbins, nthreads=int(nthreads))
    if freqs[0] > freqs[-1]:
       freqs = freqs[::-1].copy()       # make it C-contiguous again
       coef  = coef[::-1, :].copy()

    # ---- lookup and window geometry -----------------------------------
    lut = _build_lookup(freqs)
    starts = np.arange(0, sig.size - win_len + 1, step, dtype=np.int64)
    tau = (starts + win_len / 2) / fs

    # ---- Chunk configuration ------------------------------------------
    if dump_dir is None:
        chunk_windows = starts.size       # whole cube
    else:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_every = dump_every or 5.0
        chunk_windows = max(1, int(np.ceil(dump_every * fs / step)))

    # What kind of GPU usage is requested?
    use_gpu = bool(use_gpu and cp is not None)
    gpu_full = use_gpu and dump_dir is None          # whole‑cube on GPU
    gpu_stream = use_gpu and dump_dir is not None    # per‑chunk GPU

    if gpu_full:
        coef_gpu = cp.asarray(coef)  # one copy, fits only if RAM/VRAM allows

    file_list: List[str] = []
    chunks: List[np.ndarray] = []

    for idx0 in range(0, starts.size, chunk_windows):
        sl = slice(idx0, min(idx0 + chunk_windows, starts.size))
        abs_starts = starts[sl]           # absolute indices in coef

        if gpu_full:
            local_starts = abs_starts - starts[0]
            B_chunk = _tv_bispec_gpu(coef_gpu, lut, local_starts, win_len)

        elif gpu_stream:
            # copy only the needed time‑slice of coef to the GPU
            t0 = int(abs_starts[0])
            t1 = int(abs_starts[-1] + win_len)
            coef_slice_gpu = cp.asarray(coef[:, t0:t1])
            local_starts = abs_starts - t0
            B_chunk = _tv_bispec_gpu(coef_slice_gpu, lut, local_starts, win_len)
            del coef_slice_gpu
            cp.get_default_memory_pool().free_all_blocks()

        else:  # CPU path
            local_starts = abs_starts - starts[0]
            B_chunk = _tv_bispec_cpu(coef, lut, local_starts, win_len)

        # ---- save / accumulate ---------------------------------------
        if dump_dir is None:
            chunks.append(B_chunk)
        else:
            fname = dump_dir / f"B_{len(file_list):04d}.npy"
            np.save(fname, B_chunk)
            file_list.append(str(fname))
            del B_chunk

    # ---- return -------------------------------------------------------
    if dump_dir is None:
        B = np.concatenate(chunks, axis=2) if chunks else np.empty((nbins, nbins, 0))
        return B, freqs, tau
    else:
        return file_list, freqs, tau


# -----------------------------------------------------------------------------
# 3)  Quick plotting helper ----------------------------------------------------
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

def plot_slice(B: np.ndarray, freqs: np.ndarray, tau: np.ndarray, *, w: int = 0):
    """Plot |B| slice for window *w*."""
    if w < 0 or w >= B.shape[2]:
        raise IndexError("slice index out of range")
    slice_db = 20 * np.log10(np.abs(B[:, :, w]) + 1e-12)
    plt.imshow(slice_db,
               extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
               origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel('f₁ (Hz)'); plt.ylabel('f₂ (Hz)')
    plt.title(f'|Bispectrum| – window {w}/{B.shape[2]-1}  (t={tau[w]:.2f} s)')
    plt.colorbar(label='dB')
    plt.show()
