"""Test 6: a process with a flat bispectrum comes out flat, at the right level, with no phase error.

i.i.d. centred Exp(1) noise has third cumulant k3 = 2 at lag (0, 0) only, so its
bispectral density is k3 / fs**2 everywhere in the bifrequency plane.
"""
import numpy as np
import pytest

import fwhosa as fw


@pytest.mark.slow
def test_flat_bispectrum_recovered():
    fs, seconds = 100, 300
    x = np.random.default_rng(11).exponential(1.0, fs * seconds) - 1.0      # known mean: exactly 0
    kw = dict(band=(5, 20), df=0.5, expectation="all", backend="cpu")
    lin = fw.bispectrum(x, fs, normalization="linear_density", **kw)
    log = fw.bispectrum(x, fs, normalization="log_density", **kw)
    v = lin.values[0].astype(np.complex128)
    ratio = v.real / (2.0 / fs ** 2)
    assert abs(np.median(ratio) - 1.0) < 0.05, np.median(ratio)
    assert np.median(np.abs(v.imag) / np.abs(v.real)) < 0.15
    np.testing.assert_allclose(log.values[0], lin.values[0] * lin.f1 * lin.f2, rtol=1e-5)
    slope = np.polyfit(np.log(lin.f1 * lin.f2), np.log(np.abs(v.real)), 1)[0]
    assert abs(slope) < 0.1, slope                       # flat in linear-frequency units
