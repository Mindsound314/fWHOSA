"""Test 1: the transform against an independent time-domain convolution with the stated wavelet."""
import importlib.util
import math

import numpy as np
import pytest

from fwhosa.transform import FCWT_PEAK_GAIN, Grid, Wavelet, cwt

HAS_FCWT = importlib.util.find_spec("fcwt") is not None


def direct_cwt_at(x, fs, f, fb, samples):
    """W(t0) = sum_n x[n] psi(t0 - t_n) / fs with the time-domain wavelet of g(r)."""
    t = np.arange(x.size) / fs
    amp = FCWT_PEAK_GAIN * f / (fb * math.sqrt(2.0 * math.pi))
    out = []
    for s in samples:
        tau = t[s] - t
        psi = amp * np.exp(2j * np.pi * f * tau) * np.exp(-(tau * f) ** 2 / (2.0 * fb ** 2))
        out.append(np.sum(x * psi) / fs)
    return np.array(out)


@pytest.mark.parametrize("fb", [1.5, 2.0, 4.0])
def test_exact_transform_matches_time_domain_convolution(rng, fb):
    fs, grid = 128, Grid(df=0.5, k0=8, n=40)                   # 4.0 .. 23.5 Hz
    x = rng.standard_normal(4096)
    W = cwt(x, fs, grid, Wavelet(fb), pad=2048)
    samples = np.arange(1500, 2600, 97)                        # interior samples
    for row in (0, 13, 39):
        ref = direct_cwt_at(x, fs, grid.freqs[row], fb, samples)
        err = np.max(np.abs(W[row, samples] - ref)) / np.max(np.abs(ref))
        assert err < 1e-5, (fb, row, err)


@pytest.mark.parametrize("method,tol", [("exact", 1e-5)] + ([("fcwt", 3e-3)] if HAS_FCWT else []))
def test_cosine_amplitude_phase_and_frequency_label(method, tol):
    fs, grid = 128, Grid(df=0.5, k0=8, n=40)
    f, amp, phi = 10.0, 3.0, 0.7
    t = np.arange(8192) / fs
    W = cwt(amp * np.cos(2 * np.pi * f * t + phi), fs, grid, Wavelet(2.0), method=method, pad=1024)
    mid = slice(2048, 6144)
    row = int(np.argmax(np.abs(W[:, mid]).mean(axis=1)))
    assert grid.freqs[row] == f
    np.testing.assert_allclose(np.abs(W[row, mid]), FCWT_PEAK_GAIN * amp / 2, rtol=tol)
    phase_err = np.angle(W[row, mid] * np.exp(-1j * (2 * np.pi * f * t[mid] + phi)))
    assert np.max(np.abs(phase_err)) < tol


@pytest.mark.skipif(not HAS_FCWT, reason="optional fCWT not installed")
def test_fcwt_agrees_with_exact_within_its_documented_quantization(rng):
    fs, grid = 128, Grid(df=0.5, k0=8, n=40)
    x = rng.standard_normal(4096)
    exact = cwt(x, fs, grid, Wavelet(2.0), pad=1024)
    fast = cwt(x, fs, grid, Wavelet(2.0), method="fcwt")
    mid = slice(1024, 3072)
    err = np.linalg.norm(fast[:, mid] - exact[:, mid]) / np.linalg.norm(exact[:, mid])
    assert err < 1e-2, err
