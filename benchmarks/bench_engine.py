"""Throughput of the estimator on one channel: seconds per channel-minute and triad-samples per second.

Usage: python benchmarks/bench_engine.py [--backend cpu|cuda] [--minutes 1]
"""
import argparse
import time
import warnings

import numpy as np

import fwhosa as fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--fs", type=float, default=160.0)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    fs = args.fs
    x = np.random.default_rng(0).standard_normal(int(args.minutes * 60 * fs))
    kw = dict(band=(2, 30), df=0.5, expectation="time", window=2.0, step=0.5, backend=args.backend)
    fw.bicoherence(x[: int(10 * fs)], fs, **kw)                       # warm-up (JIT, kernels)
    t0 = time.perf_counter()
    res = fw.bicoherence(x, fs, **kw)
    dt = time.perf_counter() - t0
    triad_samples = res.n_triads * x.size
    print(f"backend={res.params['backend']} transform={res.params['transform']} triads={res.n_triads} "
          f"samples={x.size}: {dt:.2f} s total, {dt / args.minutes:.2f} s per channel-minute, "
          f"{triad_samples / dt:.3g} triad-samples/s")


if __name__ == "__main__":
    main()
