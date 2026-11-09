"""Window-rule calibration study, null half (blueprint test 15).

Checks the effective-DOF model on noise. For each wavelet width, window length and noise
colour it measures, per triad, the null mean of b^2 times the model's DOF, which is 1 where
the model holds. Selection and validation seeds are disjoint, so `min_dof` can be chosen
on one set and confirmed on the other. Windows flagged `boundary` are left out. The power
half (coupled carriers against surrogate thresholds) is not automated yet; `coupled()`
generates its signals.

Usage: python benchmarks/calibrate_window_rule.py [--seeds 20]
Prints a table; nothing here runs in CI.
"""
import argparse
import itertools
import warnings

import numpy as np

import fwhosa as fw

FS = 160


def noise(rng, n, colour):
    w = rng.standard_normal(n)
    if colour == "white":
        return w
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / FS)
    W[1:] /= f[1:] ** (0.5 if colour == "pink" else 1.0)   # 1/f or 1/f^2 power
    W[0] = 0
    return np.fft.irfft(W, n)


def coupled(rng, n, fa=8.0, fb_=11.0):
    t = np.arange(n) / FS
    p1, p2 = (np.cumsum(rng.normal(0, 0.05, n)) for _ in range(2))
    return (np.cos(2 * np.pi * fa * t + p1) + np.cos(2 * np.pi * fb_ * t + p2)
            + 0.6 * np.cos(2 * np.pi * (fa + fb_) * t + p1 + p2))


def run(seeds, split):
    rows = []
    for fbw, window, colour in itertools.product((1.5, 2.0, 3.0), (1.0, 2.0, 4.0), ("white", "pink", "brown")):
        ratios, dofs = [], None
        for s in seeds:
            rng = np.random.default_rng(10_000 * split + s)
            x = noise(rng, FS * 120, colour)
            res = fw.bicoherence(x, FS, band=(4, 20), df=1.0, fb=fbw, expectation="time",
                                 window=window, step=window, min_dof=0.0, backend="auto")
            v = np.where(res.flags["boundary"], np.nan, res.values)
            ratios.append(np.nanmean(v, axis=0) * res.dof)
            dofs = res.dof
        rows.append(dict(fb=fbw, window=window, colour=colour, dof=dofs,
                         ratio=np.mean(ratios, axis=0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    seeds = range(args.seeds)
    select = run(seeds, split=0)
    validate = run(seeds, split=1)
    print("condition                     model fit (median null b^2 x dof, by dof bin)")
    for a, b in zip(select, validate):
        bins = [(0, 2), (2, 5), (5, 10), (10, np.inf)]
        fit = []
        for lo, hi in bins:
            sel = (a["dof"] >= lo) & (a["dof"] < hi)
            fit.append("  -  " if not sel.any() else
                       f"{np.median(a['ratio'][sel]):.2f}/{np.median(b['ratio'][sel]):.2f}")
        print(f"fb={a['fb']:<3} win={a['window']:<3} {a['colour']:<5}  " + "  ".join(fit))
    print("\nEach cell: selection-set / validation-set median of (null mean b^2 * dof) for dof bins "
          "[0,2) [2,5) [5,10) [10,inf). Values near 1 mean the DOF model holds; choose min_dof "
          "as the lowest bin edge where both sets stay near 1, then confirm power with coupled().")


if __name__ == "__main__":
    main()
