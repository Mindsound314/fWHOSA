"""Test 16: recording-relative conditioning can change masks but never values."""
import numpy as np

import fwhosa as fw

KW = dict(band=(4, 20), df=1.0, expectation="time", window=2.0, step=1.0, backend="cpu")


def test_appended_burst_changes_masks_not_values(rng):
    fs = 128
    quiet = 1e-2 * rng.standard_normal(fs * 40)
    burst = 10.0 * rng.standard_normal(fs * 20)            # 1000x louder
    alone = fw.bicoherence(quiet, fs, **KW)
    joined = fw.bicoherence(np.concatenate([quiet, burst]), fs, **KW)
    # The burst is real context: its wavelet tail reaches back in time, scaled by the
    # 1000x amplitude ratio. Compare windows ending 10 envelope SDs (at 4 Hz) before it.
    ends = alone.times + 1.0
    w = np.flatnonzero(ends <= 40 - 10 * 2.0 / 4.0)
    w = w[alone.valid[w].any(axis=1)]
    np.testing.assert_allclose(joined.values[w], alone.values[w], atol=1e-5)
    assert not alone.flags["conditioning"][w].any()
    assert joined.flags["conditioning"][w].any()            # the quiet stretch is now flagged
