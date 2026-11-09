"""Test 7: each denominator keeps its own behaviour; nothing is clipped."""
import numpy as np

import fwhosa as fw
from fwhosa import normalize
from fwhosa.engine import get_block_moments, row_powers


def test_four_sample_fixture():
    W = np.zeros((3, 4), np.complex64)
    W[:, 3] = 1.0                                  # each row is [0, 0, 0, 1]
    I, J, K = (np.array([m], np.int32) for m in (0, 1, 2))
    S, Q = get_block_moments("reference")(W, W, W, I, J, K, 4, 1)
    P = row_powers(W, 4, 1)[:, 0]
    args = (S[:, 0], Q[:, 0], P[I], P[J], P[K], 4)
    np.testing.assert_allclose(normalize.bicoherence(*args, "product_of_powers", True), 4.0)
    np.testing.assert_allclose(normalize.bicoherence(*args, "paired_moment", True), 1.0)


def test_bursty_signal_product_form_exceeds_one_unclipped(rng):
    fs, seconds = 128, 60
    t = np.arange(fs * seconds) / fs
    env = np.zeros(t.size)
    for c in rng.uniform(0, seconds, 40):
        env += np.exp(-0.5 * ((t - c) / 0.08) ** 2)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    x = env * (np.cos(2 * np.pi * 7 * t + p1) + np.cos(2 * np.pi * 9 * t + p2)
               + 0.6 * np.cos(2 * np.pi * 16 * t + p1 + p2)) + 0.2 * rng.standard_normal(t.size)
    kw = dict(band=(4, 20), df=1.0, expectation="time", window=1.0, step=0.5, backend="cpu")
    prod = fw.bicoherence(x, fs, denominator="product_of_powers", **kw)
    paired = fw.bicoherence(x, fs, denominator="paired_moment", **kw)
    assert np.nanmax(prod.values) > 1.5                      # legitimate, and not clipped
    assert np.nanmax(paired.values) <= 1.0 + 1e-5
