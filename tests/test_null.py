"""Test 12: surrogate tests hold their false-positive rate after multiplicity control."""
import numpy as np
import pytest
from conftest import diffusing_qpc

import fwhosa as fw

KW = dict(band=(6, 16), df=1.0, expectation="all", backend="cpu")


@pytest.mark.slow
def test_family_wise_error_rate_on_gaussian_noise():
    rng = np.random.default_rng(2024)
    realizations, hits = 40, 0
    for r in range(realizations):
        x = rng.standard_normal(100 * 10)
        out = fw.null_test(fw.bicoherence, x, 100, method="phase", n_surrogates=39,
                           correction="max-stat", seed=r, **KW)
        hits += bool(out.significant.any())
    assert hits / realizations <= 0.15, hits      # nominal 0.05; 40 draws leave room for noise


@pytest.mark.slow
def test_mean_complex_bispectrum_vanishes_for_gaussian_noise():
    rng = np.random.default_rng(7)
    vals = np.array([fw.bispectrum(rng.standard_normal(1000), 100, normalization="raw", **KW).values[0]
                     for _ in range(60)])
    mean, se = vals.mean(axis=0), vals.std(axis=0) / np.sqrt(len(vals))
    assert np.median(np.abs(mean) / se) < 2.0


def test_coupling_is_detected_and_uncoupled_is_not(rng):
    for coupled in (True, False):
        x = diffusing_qpc(100, 30, 7, 9, rng, coupled=coupled)
        out = fw.null_test(fw.bicoherence, x, 100, method="phase", n_surrogates=49, seed=1, **KW)
        t = np.flatnonzero((out.result.f1 == 7) & (out.result.f2 == 9))[0]
        assert out.significant[0, t] == coupled
    assert np.allclose(out.result.values, fw.bicoherence(x, 100, **KW).values)   # estimate untouched


def test_fdr_and_none_corrections_run(rng):
    x = rng.standard_normal(1000)
    for correction in ("fdr", "none"):
        out = fw.null_test(fw.bicoherence, x, 100, n_surrogates=19, correction=correction, seed=3, **KW)
        assert out.p_values.shape == out.result.values.shape
