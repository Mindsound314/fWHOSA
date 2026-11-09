"""CuPy CUDA engine: one thread per (triad, block), float32 products, float64 results.

Each thread loops over its block's samples, so device memory scales with the
number of triads and blocks, not with triads x window length as in the legacy
code. Block sums are short (one block), so float32 accumulation stays within
the declared tolerances; window and trial sums happen later in float64.

CI runners have no GPU, so this backend is validated only by running
``pytest -m cuda`` on a CUDA machine; do that before relying on it.
"""
from __future__ import annotations

import numpy as np

_SOURCE = r"""
extern "C" __global__
void triad_block_moments(const float* W1, const float* W2, const float* W3,
                         const int* I, const int* J, const int* K,
                         const long long row_stride, const int block,
                         const int n_blocks, const int T,
                         float* S_re, float* S_im, float* Q12)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)T * n_blocks;
    if (idx >= total) return;
    int t = (int)(idx / n_blocks);
    int b = (int)(idx - (long long)t * n_blocks);
    long long offset = 2LL * (long long)b * block;          /* interleaved re, im */
    const float* a = W1 + (long long)I[t] * row_stride + offset;
    const float* c = W2 + (long long)J[t] * row_stride + offset;
    const float* d = W3 + (long long)K[t] * row_stride + offset;
    float sr = 0.f, si = 0.f, q = 0.f;
    for (int n = 0; n < block; ++n) {
        float ar = a[2 * n], ai = a[2 * n + 1];
        float cr = c[2 * n], ci = c[2 * n + 1];
        float dr = d[2 * n], di = d[2 * n + 1];
        float pr = ar * cr - ai * ci;
        float pi = ar * ci + ai * cr;
        sr += pr * dr + pi * di;
        si += pi * dr - pr * di;
        q  += pr * pr + pi * pi;
    }
    S_re[idx] = sr;
    S_im[idx] = si;
    Q12[idx] = q;
}
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp

        _KERNEL = cp.RawKernel(_SOURCE, "triad_block_moments")
    return _KERNEL


def block_moments(W1, W2, W3, I, J, K, block, n_blocks):
    import cupy as cp

    d1 = cp.asarray(np.ascontiguousarray(W1, dtype=np.complex64))
    d2 = d1 if W2 is W1 else cp.asarray(np.ascontiguousarray(W2, dtype=np.complex64))
    d3 = d1 if W3 is W1 else (d2 if W3 is W2 else cp.asarray(np.ascontiguousarray(W3, dtype=np.complex64)))
    dI = cp.asarray(np.asarray(I, dtype=np.int32))
    dJ = cp.asarray(np.asarray(J, dtype=np.int32))
    dK = cp.asarray(np.asarray(K, dtype=np.int32))
    T = int(dI.size)
    total = T * int(n_blocks)
    S_re = cp.empty(total, dtype=cp.float32)
    S_im = cp.empty(total, dtype=cp.float32)
    Q12 = cp.empty(total, dtype=cp.float32)
    threads = 256
    grid = (total + threads - 1) // threads
    _kernel()((grid,), (threads,),
              (d1.view(cp.float32), d2.view(cp.float32), d3.view(cp.float32), dI, dJ, dK,
               np.int64(2 * d1.shape[1]), np.int32(block), np.int32(n_blocks), np.int32(T),
               S_re, S_im, Q12))
    S = (S_re.astype(cp.float64) + 1j * S_im.astype(cp.float64)).reshape(T, n_blocks)
    return cp.asnumpy(S), cp.asnumpy(Q12.astype(cp.float64).reshape(T, n_blocks))
