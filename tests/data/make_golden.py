"""Regenerate tests/data/golden_v0.1.npz with the NumPy reference backend.

Run only when a deliberate change to the estimators is made, and record it in
CHANGELOG.md; test_golden.py compares every backend against these arrays.
"""
from pathlib import Path

import numpy as np

import fwhosa as fw


def signals():
    rng = np.random.default_rng(20261109)
    fs = 128
    t = np.arange(fs * 20) / fs
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    x = (np.cos(2 * np.pi * 7 * t + p1) + np.cos(2 * np.pi * 9 * t + p2)
         + 0.6 * np.cos(2 * np.pi * 16 * t + p1 + p2) + 0.5 * rng.standard_normal(t.size))
    y = rng.standard_normal(t.size)
    z = rng.standard_normal(t.size)
    return fs, x, y, z


def compute(backend):
    fs, x, y, z = signals()
    common = dict(band=(4, 16), df=1.0, backend=backend)
    return dict(
        bicoherence=fw.bicoherence(x, fs, expectation="time", window=2.0, step=1.0, **common).values,
        bispectrum=fw.bispectrum(x, fs, expectation="all", normalization="linear_density", **common).values,
        cross_bicoherence=fw.cross_bicoherence(x, y, z, fs, expectation="all", **common).values,
    )


if __name__ == "__main__":
    out = Path(__file__).with_name("golden_v0.1.npz")
    np.savez_compressed(out, **compute("reference"))
    print(f"wrote {out}")
