"""Triad moment engines.

Every backend implements one contract::

    block_moments(W1, W2, W3, I, J, K, block, n_blocks) -> (S, Q12)

where ``W1, W2, W3`` are C-contiguous complex64 coefficient arrays of shape
``(n_freqs, n_samples)`` (the same object three times for auto estimators),
``I, J, K`` are int32 row indices of the triads, and for every triad ``t`` and
block ``b`` covering samples ``[b*block, (b+1)*block)``::

    S[t, b]   = sum_n W1[I[t], n] * W2[J[t], n] * conj(W3[K[t], n])   (complex128)
    Q12[t, b] = sum_n |W1[I[t], n] * W2[J[t], n]|**2                   (float64)

Per-row powers are backend-independent and computed by :func:`row_powers`.
"""
from __future__ import annotations

import numpy as np

BACKENDS = ("auto", "reference", "cpu", "cuda")


def cuda_available() -> bool:
    try:
        import cupy as cp  # noqa: F401

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - any import or driver failure means no CUDA
        return False


def resolve_backend(name: str) -> str:
    """Map a backend request to a concrete backend; ``"cuda"`` fails loudly when absent."""
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if name == "auto":
        return "cuda" if cuda_available() else "cpu"
    if name == "cuda" and not cuda_available():
        raise RuntimeError("backend='cuda' requested but CuPy or a CUDA device is not available; "
                           "install the [gpu] extra or use backend='cpu'")
    return name


def get_block_moments(name: str):
    if name == "reference":
        from .reference import block_moments
    elif name == "cpu":
        from .cpu import block_moments
    elif name == "cuda":
        from .cuda import block_moments
    else:
        raise ValueError(f"unknown backend {name!r}; resolve 'auto' first")
    return block_moments


def row_powers(W: np.ndarray, block: int, n_blocks: int) -> np.ndarray:
    """Per-row block sums of |W|**2, shape ``(n_freqs, n_blocks)``, float64."""
    used = W[:, : block * n_blocks]
    p = used.real.astype(np.float64) ** 2 + used.imag.astype(np.float64) ** 2
    return p.reshape(W.shape[0], n_blocks, block).sum(axis=2)
