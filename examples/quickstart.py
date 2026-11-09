"""fwhosa quick start: detect quadratic phase coupling, test it, and plot it.

Run: python examples/quickstart.py   (needs matplotlib for the figure)
"""
import numpy as np

import fwhosa as fw

fs, seconds = 160, 60
rng = np.random.default_rng(0)
t = np.arange(fs * seconds) / fs

# Carriers at 7 and 13 Hz whose phases drift; a 20 Hz component locked to their sum.
# (Fixed-phase sinusoids would be a degenerate test case; see the README's pitfalls.)
p1 = np.cumsum(rng.normal(0, 0.05, t.size))
p2 = np.cumsum(rng.normal(0, 0.05, t.size))
x = (np.cos(2 * np.pi * 7 * t + p1) + np.cos(2 * np.pi * 13 * t + p2)
     + 0.6 * np.cos(2 * np.pi * 20 * t + p1 + p2) + 0.5 * rng.standard_normal(t.size))
x *= 20e-6                                     # volts, as MNE stores EEG; units do not matter

res = fw.bicoherence(x, fs, band=(4, 30), df=0.5, expectation="all")
best = int(np.nanargmax(np.where(res.valid[0], res.values[0], np.nan)))
print(f"{res.n_triads} triads; strongest b^2 = {res.values[0, best]:.2f} at "
      f"({res.f1[best]:.1f}, {res.f2[best]:.1f}) -> {res.f3[best]:.1f} Hz; backend {res.params['backend']}")

out = fw.null_test(fw.bicoherence, x, fs, band=(4, 30), df=0.5, expectation="all",
                   method="phase", n_surrogates=99, correction="max-stat", seed=1)
sig = out.significant[0]
print(f"{int(sig.sum())} triads significant after family-wise correction, spanning "
      f"f1 {res.f1[sig].min():.1f}-{res.f1[sig].max():.1f} Hz and f2 {res.f2[sig].min():.1f}-"
      f"{res.f2[sig].max():.1f} Hz: one coupled triad, spread over the wavelet's bandwidth")

tv = fw.bicoherence(x, fs, band=(4, 30), df=0.5, expectation="time", window=4.0, step=1.0)
print(f"time-varying: {tv.n_windows} windows of 4 s; median b^2 at (7, 13): "
      f"{np.median(tv.values[:, (tv.f1 == 7) & (tv.f2 == 13)]):.2f}")

try:
    import matplotlib.pyplot as plt

    res.plot(0)
    plt.savefig("quickstart_bicoherence.png", dpi=120, bbox_inches="tight")
    print("saved quickstart_bicoherence.png")
except ImportError:
    print("install matplotlib to save the figure")
