"""Test 14: outputs match stored arrays from the reference backend; any change needs a changelog entry."""
import sys
from pathlib import Path

import numpy as np
import pytest

from fwhosa.engine import cuda_available

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(DATA))
from make_golden import compute  # noqa: E402

BACKENDS = ["reference", "cpu"] + (["cuda"] if cuda_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_golden(backend):
    golden = np.load(DATA / "golden_v0.1.npz")
    got = compute(backend)
    for name, arr in got.items():
        ref = golden[name]
        tol = 1e-4 if backend == "cuda" else 1e-5
        scale = np.nanmax(np.abs(ref))
        np.testing.assert_allclose(arr, ref, rtol=tol, atol=tol * scale, err_msg=name)
