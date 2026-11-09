"""Test 9: guard-banded segments reproduce the whole-recording result, with no periodic modulation."""
import numpy as np
import pytest

import fwhosa as fw

KW = dict(band=(4, 20), df=1.0, expectation="time", window=2.0, step=0.5, backend="cpu", guard_tol=1e-6)


@pytest.mark.parametrize("estimator", [fw.bicoherence, fw.bispectrum])
def test_segmented_equals_whole_recording(rng, estimator):
    x = rng.standard_normal(128 * 60)
    whole = estimator(x, 128, memory_limit="1GB", **KW)
    assert not whole.params["segmented"]
    for limit in ("1MB", "2MB"):                           # two different segment lengths
        seg = estimator(x, 128, memory_limit=limit, **KW)
        assert seg.params["segmented"] and seg.params["core_blocks"] < whole.params["core_blocks"]
        ok = whole.valid & seg.valid
        assert ok.mean() > 0.8
        scale = np.abs(whole.values[ok]).max()
        diff = np.abs(seg.values - whole.values)
        assert np.max(diff[ok]) <= 1e-4 * scale, (limit, np.max(diff[ok]) / scale)
        # no modulation at the segment period: the per-window error is flat and tiny
        per_window = np.where(ok, diff, 0).max(axis=1) / scale
        assert per_window.max() <= 1e-4
