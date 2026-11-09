"""Result object, expectation modes, explicit triads and output options."""
import numpy as np
from conftest import qpc_signal

import fwhosa as fw

KW = dict(band=(4, 16), df=1.0, backend="cpu")


def test_save_load_round_trip(tmp_path, rng):
    res = fw.bicoherence(qpc_signal(128, 20, 7, 9, rng), 128, expectation="time", window=2.0, **KW)
    res.save(tmp_path / "r.npz")
    back = fw.HOSResult.load(tmp_path / "r.npz")
    np.testing.assert_array_equal(back.values, res.values)
    np.testing.assert_array_equal(back.valid, res.valid)
    assert back.params["denominator"] == "paired_moment" and back.kind == "bicoherence"
    assert set(back.flags) == set(res.flags)


def test_matrix_orientation_for_cross(rng):
    x, y, z = (qpc_signal(128, 20, 7, 9, rng) for _ in range(3))
    res = fw.cross_bicoherence(x, y, z, 128, expectation="all", **KW)
    M = res.matrix(0)
    t = np.flatnonzero((res.f1 == 5) & (res.f2 == 8))[0]
    a, b = list(res.pair_freqs).index(5.0), list(res.pair_freqs).index(8.0)
    assert M[a, b] == res.values[0, t]                       # rows are f1, columns f2
    ax = res.plot(0)
    assert ax.get_xlabel().startswith("f₁")


def test_trials_expectation_pools_moments(rng):
    fs, n_trials = 128, 30
    t = np.arange(fs * 3) / fs
    trials = []
    for _ in range(n_trials):                                # phases vary by trial, coupling holds
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        trials.append(np.cos(2 * np.pi * 7 * t + p1) + np.cos(2 * np.pi * 9 * t + p2)
                      + 0.6 * np.cos(2 * np.pi * 16 * t + p1 + p2) + 0.5 * rng.standard_normal(t.size))
    res = fw.bicoherence(np.array(trials), fs, expectation="trials", window=1.0, step=0.5, **KW)
    assert res.values.shape == (res.n_windows, res.n_triads)
    tri = np.flatnonzero((res.f1 == 7) & (res.f2 == 9))[0]
    mid = res.n_windows // 2
    assert res.values[mid, tri] > 0.8
    assert np.median(res.values[mid]) < 0.3
    assert res.flags["boundary"][0].any()                    # epoch edges are guarded


def test_explicit_triads_and_memmap(tmp_path, rng):
    x = qpc_signal(128, 20, 7, 9, rng)
    res = fw.bispectrum(x, 128, band=(4, 16), df=1.0, backend="cpu", expectation="time", window=2.0,
                        triads=[(7, 9), (5, 6)], out=tmp_path / "v.npy")
    assert res.n_triads == 2 and list(res.f3) == [16.0, 11.0]
    np.testing.assert_array_equal(np.load(tmp_path / "v.npy"), np.asarray(res.values))


def test_params_record_provenance(rng):
    res = fw.bispectrum(rng.standard_normal(2560), 128, expectation="all", **KW)
    for key in ("fwhosa", "fcwt", "fb", "gain_convention", "density_measure", "backend",
                "row_scales", "guard_tol", "sum_cap_hz", "dof_model"):
        assert key in res.params, key
