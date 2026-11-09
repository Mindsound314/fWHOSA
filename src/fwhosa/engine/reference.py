"""Readable NumPy reference engine, in complex128.

This is the estimator as the equations state it; the fast backends must match
it. It favours clarity over speed.
"""
from __future__ import annotations

import numpy as np


def block_moments(W1, W2, W3, I, J, K, block, n_blocks):
    T = int(np.asarray(I).size)
    S = np.empty((T, n_blocks), dtype=np.complex128)
    Q12 = np.empty((T, n_blocks), dtype=np.float64)
    for b in range(n_blocks):
        cols = slice(b * block, (b + 1) * block)
        w1 = W1[I, cols].astype(np.complex128)      # (T, block): W1 at f1 for every triad
        w2 = W2[J, cols].astype(np.complex128)      # W2 at f2
        w3 = W3[K, cols].astype(np.complex128)      # W3 at f1 + f2
        pair = w1 * w2
        S[:, b] = np.sum(pair * np.conj(w3), axis=1)
        Q12[:, b] = np.sum(np.abs(pair) ** 2, axis=1)
    return S, Q12
