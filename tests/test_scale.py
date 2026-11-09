"""Test 5: scale laws, and merging separately scaled recordings."""
import numpy as np
import pytest
from conftest import qpc_signal

import fwhosa as fw
from fwhosa import normalize
from fwhosa.engine import get_block_moments, row_powers

KW = dict(band=(4, 20), df=1.0, expectation="time", window=2.0, step=1.0, backend="cpu")


@pytest.mark.parametrize("denominator", ["paired_moment", "product_of_powers"])
def test_bicoherence_is_unit_free(rng, denominator):
    x = qpc_signal(128, 30, 7, 9, rng)
    a = fw.bicoherence(x, 128, denominator=denominator, **KW)
    b = fw.bicoherence(2e-5 * x, 128, denominator=denominator, **KW)     # EEG in volts
    c = fw.bicoherence(1e6 * x, 128, denominator=denominator, **KW)
    np.testing.assert_allclose(b.values, a.values, atol=1e-5)
    np.testing.assert_allclose(c.values, a.values, atol=1e-5)


def test_raw_cross_bispectrum_scales_as_abc(rng):
    x, y, z = (qpc_signal(128, 20, 7, 9, rng) for _ in range(3))
    kw = dict(KW, expectation="all", window=None, step=None)
    base = fw.cross_bispectrum(x, y, z, 128, normalization="raw", **kw).values
    scaled = fw.cross_bispectrum(3.0 * x, 1e-5 * y, -2.0 * z, 128, normalization="raw", **kw).values
    np.testing.assert_allclose(scaled, 3.0 * 1e-5 * -2.0 * base, rtol=1e-4, atol=1e-12)


def test_split_runs_reproduce_joint_run_after_reconciliation(rng):
    """Moments from separately scaled halves, restored to common units, equal the joint moments."""
    n, block = 2400, 100
    W = (rng.standard_normal((10, n)) + 1j * rng.standard_normal((10, n))).astype(np.complex64)
    W[:, n // 2:] *= 7.0                                   # the second half is much louder
    I, J, K = (np.array(v, np.int32) for v in ([1, 2, 0], [3, 2, 4], [4, 4, 4]))
    engine = get_block_moments("cpu")

    def moments(Wpart, rms):
        Ws = (Wpart / rms[:, None]).astype(np.complex64)
        S, Q = engine(Ws, Ws, Ws, I, J, K, block, Wpart.shape[1] // block)
        P = row_powers(Ws, block, Wpart.shape[1] // block).sum(1)
        r = rms
        # restore to common (unscaled) units before merging
        return S.sum(1) * r[I] * r[J] * r[K], Q.sum(1) * (r[I] * r[J]) ** 2, P * r ** 2

    rms_all = np.sqrt((np.abs(W.astype(np.complex128)) ** 2).mean(1))
    joint = moments(W, rms_all)
    halves = [moments(W[:, s], np.sqrt((np.abs(W[:, s].astype(np.complex128)) ** 2).mean(1)))
              for s in (slice(0, n // 2), slice(n // 2, n))]
    merged = [halves[0][m] + halves[1][m] for m in range(3)]
    b_joint = normalize.bicoherence(joint[0], joint[1], None, None, joint[2][K], n, "paired_moment", True)
    b_merge = normalize.bicoherence(merged[0], merged[1], None, None, merged[2][K], n, "paired_moment", True)
    np.testing.assert_allclose(b_merge, b_joint, rtol=1e-6)
