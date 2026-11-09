"""Test 13: invalid input raises; degenerate input is flagged, never silently reinterpreted."""
import numpy as np
import pytest

import fwhosa as fw
from fwhosa.engine import cuda_available

KW = dict(band=(4, 20), df=1.0, window=2.0, backend="cpu")


def test_channels_by_time_array_is_not_flattened(rng):
    x = rng.standard_normal((8, 1280))
    with pytest.raises(ValueError, match="one continuous recording"):
        fw.bicoherence(x, 128, expectation="time", **KW)


def test_expectation_is_required(rng):
    with pytest.raises(TypeError):
        fw.bicoherence(rng.standard_normal(1280), 128, band=(4, 20))


@pytest.mark.parametrize("bad", [
    dict(expectation="trials"),                                   # 1-D input
    dict(expectation="sometimes"),
    dict(expectation="time", window=None),
    dict(expectation="all"),                                      # window given
    dict(expectation="time", window=-1.0),
    dict(expectation="time", fb=0.5),
    dict(expectation="time", band=(20, 4)),
    dict(expectation="time", band=(45, 60)),                      # sums beyond the margin
    dict(expectation="time", triads=[(4.3, 7.0)]),                # off the grid
])
def test_bad_settings_raise(rng, bad):
    kw = {**KW, **bad}
    with pytest.raises(ValueError):
        fw.bicoherence(rng.standard_normal(1280), 128, **kw)


def test_nan_raises(rng):
    x = rng.standard_normal(1280)
    x[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fw.bicoherence(x, 128, expectation="time", **KW)


def test_mismatched_cross_shapes_raise(rng):
    with pytest.raises(ValueError, match="identical shapes"):
        fw.cross_bicoherence(rng.standard_normal(1280), rng.standard_normal(1281),
                             rng.standard_normal(1280), 128, expectation="time", **KW)


@pytest.mark.skipif(cuda_available(), reason="a GPU is present")
def test_explicit_cuda_fails_loudly(rng):
    with pytest.raises(RuntimeError, match="cuda"):
        fw.bicoherence(rng.standard_normal(1280), 128, expectation="time", **{**KW, "backend": "cuda"})


def test_all_zero_input_is_flagged_not_crashing():
    res = fw.bicoherence(np.zeros(1280), 128, expectation="time", **KW)
    assert not res.valid.any()
    assert res.flags["conditioning"].all()
