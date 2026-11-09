"""Numba CPU engine: parallel over triads, float64 accumulation."""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath={"reassoc", "contract", "nsz", "arcp"}, cache=False)
def _kernel(W1, W2, W3, I, J, K, block, n_blocks, S_re, S_im, Q12):
    for t in nb.prange(I.shape[0]):
        i = I[t]
        j = J[t]
        k = K[t]
        for b in range(n_blocks):
            sr = 0.0
            si = 0.0
            q = 0.0
            start = b * block
            for n in range(start, start + block):
                a = W1[i, n]
                c = W2[j, n]
                d = W3[k, n]
                ar = np.float64(a.real)
                ai = np.float64(a.imag)
                cr = np.float64(c.real)
                ci = np.float64(c.imag)
                dr = np.float64(d.real)
                di = np.float64(d.imag)
                pr = ar * cr - ai * ci          # (W1 * W2).real
                pi = ar * ci + ai * cr          # (W1 * W2).imag
                sr += pr * dr + pi * di         # (W1 * W2 * conj(W3)).real
                si += pi * dr - pr * di         # (W1 * W2 * conj(W3)).imag
                q += pr * pr + pi * pi
            S_re[t, b] = sr
            S_im[t, b] = si
            Q12[t, b] = q


def block_moments(W1, W2, W3, I, J, K, block, n_blocks):
    I = np.ascontiguousarray(I, dtype=np.int32)
    J = np.ascontiguousarray(J, dtype=np.int32)
    K = np.ascontiguousarray(K, dtype=np.int32)
    T = I.size
    S_re = np.empty((T, n_blocks), dtype=np.float64)
    S_im = np.empty((T, n_blocks), dtype=np.float64)
    Q12 = np.empty((T, n_blocks), dtype=np.float64)
    _kernel(W1, W2, W3, I, J, K, int(block), int(n_blocks), S_re, S_im, Q12)
    return S_re + 1j * S_im, Q12
