"""Tests 2, 3, 4 and 11: the moment engines against an independent oracle and exact answers."""
import numpy as np
import pytest

from fwhosa import normalize
from fwhosa.engine import cuda_available, get_block_moments, row_powers


def oracle(W1, W2, W3, triads, denominator):
    """Brute-force complex128 loops; shares no code with the package."""
    out = []
    n = W1.shape[1]
    for i, j, k in triads:
        S = 0j
        Q = P1 = P2 = P3 = 0.0
        for t in range(n):
            a, b, c = complex(W1[i, t]), complex(W2[j, t]), complex(W3[k, t])
            S += a * b * c.conjugate()
            Q += abs(a * b) ** 2
            P1 += abs(a) ** 2
            P2 += abs(b) ** 2
            P3 += abs(c) ** 2
        out.append(abs(S) ** 2 / (Q * P3) if denominator == "paired_moment"
                   else n * abs(S) ** 2 / (P1 * P2 * P3))
    return np.array(out)


def engine_bicoherence(backend, W1, W2, W3, triads, denominator, block):
    I, J, K = (np.array(c, dtype=np.int32) for c in zip(*triads))
    n_blocks = W1.shape[1] // block
    S, Q = get_block_moments(backend)(W1, W2, W3, I, J, K, block, n_blocks)
    P1 = row_powers(W1, block, n_blocks)[I].sum(1)
    P2 = row_powers(W2, block, n_blocks)[J].sum(1)
    P3 = row_powers(W3, block, n_blocks)[K].sum(1)
    return normalize.bicoherence(S.sum(1), Q.sum(1), P1, P2, P3, block * n_blocks, denominator, True)


def random_coefs(rng, rows=12, n=600):
    return (rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))).astype(np.complex64)


TRIADS = [(0, 2, 3), (1, 1, 3), (2, 5, 8), (4, 6, 11), (3, 3, 7)]
BACKENDS = ["reference", "cpu"] + (["cuda"] if cuda_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("denominator", ["paired_moment", "product_of_powers"])
def test_engine_matches_oracle(rng, backend, denominator):
    W1, W2, W3 = random_coefs(rng), random_coefs(rng), random_coefs(rng)
    got = engine_bicoherence(backend, W1, W2, W3, TRIADS, denominator, block=60)
    want = oracle(W1, W2, W3, TRIADS, denominator)
    np.testing.assert_allclose(got, want, rtol=1e-4 if backend == "cuda" else 1e-9)


@pytest.mark.parametrize("backend", ["reference", "cpu"])
def test_exact_biphase(rng, backend):
    n, beta = 800, 1.1
    phi1, phi2 = rng.uniform(0, 2 * np.pi, (2, n))
    W = np.zeros((4, n), np.complex64)
    W[0], W[1], W[2] = np.exp(1j * phi1), np.exp(1j * phi2), np.exp(1j * (phi1 + phi2 - beta))
    S, Q = get_block_moments(backend)(W, W, W, np.array([0], np.int32), np.array([1], np.int32),
                                      np.array([2], np.int32), 100, 8)
    np.testing.assert_allclose(S.sum() / n, np.exp(1j * beta), atol=1e-6)
    P3 = row_powers(W, 100, 8)[2].sum()
    b2 = normalize.bicoherence(S.sum(1), Q.sum(1), None, None, P3, n, "paired_moment", True)
    np.testing.assert_allclose(b2, 1.0, atol=1e-6)


def test_analytical_null_is_one_over_n(rng):
    n, reps = 50, 400
    vals = []
    for _ in range(reps):
        W = np.exp(1j * rng.uniform(0, 2 * np.pi, (3, n))).astype(np.complex64)
        vals.append(engine_bicoherence("cpu", W, W, W, [(0, 1, 2)], "paired_moment", block=n)[0])
    assert abs(np.mean(vals) * n - 1.0) < 0.1


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="needs CuPy and a CUDA device")
def test_cuda_raw_moments_match_reference(rng):
    W1, W2, W3 = random_coefs(rng, n=6000), random_coefs(rng, n=6000), random_coefs(rng, n=6000)
    I, J, K = (np.array(c, dtype=np.int32) for c in zip(*TRIADS))
    S0, Q0 = get_block_moments("reference")(W1, W2, W3, I, J, K, 200, 30)
    S1, Q1 = get_block_moments("cuda")(W1, W2, W3, I, J, K, 200, 30)
    np.testing.assert_allclose(S1, S0, rtol=0, atol=1e-4 * np.abs(S0).max())
    np.testing.assert_allclose(Q1, Q0, rtol=1e-4)
