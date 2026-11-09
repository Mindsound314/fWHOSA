"""Test 15 (light): the effective-DOF model predicts the null bicoherence level.

The full window-rule calibration (several bandwidths, noise colours and window
shapes, chosen on one simulation set and validated on another) lives in
``benchmarks/calibrate_window_rule.py``; this test pins the model it relies on.
"""
import numpy as np
import pytest

import fwhosa as fw


@pytest.mark.slow
def test_null_mean_tracks_one_over_dof():
    x = np.random.default_rng(3).standard_normal(160 * 240)
    res = fw.bicoherence(x, 160, band=(8, 20), df=1.0, expectation="time", window=4.0, step=2.0,
                         backend="cpu")
    keep = res.dof >= 5
    ratio = np.nanmean(np.where(res.valid, res.values, np.nan), axis=0)[keep] * res.dof[keep]
    assert 0.75 < np.median(ratio) < 1.25, np.median(ratio)
