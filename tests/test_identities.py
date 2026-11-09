"""Test 8: cross with identical inputs equals auto; swapping x and y transposes the map."""
import numpy as np
from conftest import qpc_signal

import fwhosa as fw

KW = dict(band=(4, 20), df=1.0, expectation="time", window=2.0, step=1.0, backend="cpu")


def _lookup(res):
    return {(float(a), float(b)): res.values[:, t] for t, (a, b) in enumerate(zip(res.f1, res.f2))}


def test_cross_of_identical_inputs_equals_auto(rng):
    x = qpc_signal(128, 30, 7, 9, rng)
    auto = fw.bicoherence(x, 128, **KW)
    cross = _lookup(fw.cross_bicoherence(x, x.copy(), x.copy(), 128, **KW))
    for t, (a, b) in enumerate(zip(auto.f1, auto.f2)):
        np.testing.assert_allclose(cross[(float(a), float(b))], auto.values[:, t], atol=1e-6)


def test_swapping_x_and_y_transposes(rng):
    x, y, z = (qpc_signal(128, 30, 7, 9, rng) for _ in range(3))
    xy = _lookup(fw.cross_bicoherence(x, y, z, 128, **KW))
    yx = _lookup(fw.cross_bicoherence(y, x, z, 128, **KW))
    for (a, b), v in xy.items():
        np.testing.assert_allclose(yx[(b, a)], v, atol=1e-6)
