# fwhosa

[![tests](https://github.com/Mindsound314/fWHOSA/actions/workflows/ci.yml/badge.svg)](https://github.com/Mindsound314/fWHOSA/actions/workflows/ci.yml)

**Fast wavelet higher-order spectral analysis.** Wavelet bispectrum, bicoherence and
their three-channel cross forms on one validated engine, with stated density measures,
surrogate null tests and a known-answer test suite.

```python
import fwhosa as fw

res = fw.bicoherence(x, fs=160, band=(4, 30), expectation="time", window=2.0, step=0.5)
res.values          # (n_windows, n_triads): b^2 for every triad (f1, f2, f1 + f2)
res.valid           # which estimates to trust, and res.flags says why not
res.plot(window=0)  # f1 horizontal, f2 vertical
```

Status: v0.1.0. The NumPy reference and Numba CPU backends pass the full test suite.
The CuPy CUDA backend is implemented to the same contract; run `pytest -m cuda` on your
GPU before relying on it, because CI runners have no GPU.

## Install

```bash
pip install fwhosa              # numpy, scipy, numba
pip install "fwhosa[gpu]"       # + CuPy for CUDA 12.x (Blackwell GPUs need CUDA 12.8+)
pip install "fwhosa[plot]"      # + matplotlib for HOSResult.plot
pip install "fwhosa[fcwt]"      # + optional fCWT transform (Linux: apt install libfftw3-single3)
```

Before the first PyPI release: `pip install "git+https://github.com/Mindsound314/fWHOSA"`. From a clone: `pip install -e ".[dev]"` and `pytest`.

## Quick start

```python
import numpy as np
import fwhosa as fw

# One continuous recording: sliding windows
res = fw.bicoherence(x, fs, band=(4, 30), df=0.5, expectation="time", window=2.0, step=0.5)

# Epochs (trials x time): moments summed across trials at each latency
res = fw.bicoherence(epochs, fs, band=(4, 30), expectation="trials", window=1.0)

# One map over everything, as a bispectral density per Hz^2
res = fw.bispectrum(x, fs, band=(4, 30), expectation="all", normalization="linear_density")

# Three channels: x at f1, y at f2, z at f1 + f2 (order matters)
res = fw.cross_bicoherence(x, y, z, fs, band=(4, 30), expectation="all")

# Surrogate test with family-wise error control; res.values is never modified
out = fw.null_test(fw.bicoherence, x, fs, band=(4, 30), expectation="all",
                   method="iaaft", n_surrogates=200, correction="max-stat")
out.significant, out.p_values

res.save("sub-01_C3.npz"); res = fw.HOSResult.load("sub-01_C3.npz")
```

`expectation` has no default on purpose: whether you average over time, over trials or
over everything is a scientific choice, not something to infer from an array's shape.

See `examples/quickstart.py` for a runnable walk-through.

## What is computed

**Wavelet.** An analytic Morlet wavelet in fCWT's convention. For a row centred at `f`,
the signal's spectrum is multiplied by

    g(r) = c * exp(-2 * pi^2 * fb^2 * (r - 1)^2),   r = nu / f > 0,   c = sqrt(2*pi) * pi^(-1/4)

`fb` is the number of cycles per envelope standard deviation: the envelope SD is `fb / f`
seconds (0.5 s at 4 Hz for the default `fb = 2`) and the spectral SD is `f / (2*pi*fb)`.
`fb = 2` corresponds to omega0 = 4*pi (about 12.6) in Torrence-Compo terms. A cosine of
amplitude `A` gives `|W| = c * A / 2`.

**Triad plan.** `band = (f_lo, f_hi)` names the f1 and f2 axes. Frequencies sit on a
harmonic grid `f = k * df`, so every sum `f1 + f2` falls exactly on a grid row. A triad is
kept only if its sum leaves `margin_sd` (default 3) spectral SDs below Nyquist and below
the acquisition low-pass (`lowpass=`). At `fs = 160` Hz this caps sums near 64.6 Hz, so a
4-30 Hz pair axis fits and a 4-40 Hz one does not. Auto estimators store only `f1 <= f2`.
`triads=[(f1, f2), ...]` restricts the plan to chosen pairs.

**Moments.** Over each averaging domain, with rows scaled by their recording-wide RMS:

    S = sum W1 W2 conj(W3),   Q12 = sum |W1 W2|^2,   P_m = sum |W_m|^2,   n = number of samples

| Estimator | Option | Definition | Range |
| --- | --- | --- | --- |
| bicoherence | `denominator="paired_moment"` (default) | `|S|^2 / (Q12 P3)` | 0 to 1 (Kim & Powers 1979) |
| bicoherence | `denominator="product_of_powers"` | `n |S|^2 / (P1 P2 P3)` | can exceed 1; never clipped |
| bispectrum | `normalization="raw"` | `S / n`, in physical units cubed | |
| bispectrum | `normalization="log_density"` | `raw / D(lambda)` | Newman, Pidde & Stefanovska (2021), Eq. 45 |
| bispectrum | `normalization="linear_density"` (default) | `raw / (D(lambda) f1 f2)` | per Hz^2 |

`lambda = f1 / (f1 + f2)` and `D(lambda) = c^3 pi / (2 pi^2 fb^2 sqrt(1 + lambda^2 + (1 - lambda)^2))`,
the exact integral for this wavelet. A process with a flat bispectrum of density
`k3 / fs^2` comes out as `k3 / fs^2` in `linear_density` (`tests/test_flat.py`). A log density grows in
proportion to f1 f2 for the same process; that is the measure, not an error.

**Expectation.** `"time"`: windows of `window` seconds every `step` seconds over one 1-D
recording. `"trials"`: windows at matching latencies, with moments summed across trials
before any ratio is formed. `"all"`: one estimate per triad. Windows are sums of blocks
of `gcd(window, step)` samples, so overlapping windows cost no extra work.

## Validity flags

Values are never overwritten. `res.valid` is `False` wherever any of four flags in `res.flags` is set:

* `boundary`: the window lies within the wavelet's reach of a recording or epoch edge
  (`guard_tol`, default 1e-4 of the envelope's area).
* `conditioning`: the three mean powers multiply to less than `conditioning_floor`
  (default 1e-6) in units of each row's recording-wide RMS. This is recording-relative:
  appending a loud burst elsewhere can flag a quiet stretch without changing its values.
* `dof`: fewer than `min_dof` effective independent samples (`res.dof`, a white-noise
  approximation, `duration * sqrt(f1^2 + f2^2 + f3^2) / (2 sqrt(pi) fb)`). On white, pink
  and brown noise the null mean of b^2 is 0.79-0.85 times `1 / dof` for dof 2 to 5 and
  0.93-0.96 times it above 10, for `fb` 1.5 to 3 and 1- to 4-s windows
  (`benchmarks/results/window_rule_null.txt`). The default `min_dof = 2` marks where the
  approximation starts to hold; whether a stricter threshold buys power is not yet calibrated.
* `undefined`: a zero denominator.

## Statistics

`fw.null_test(estimator, *signals, fs, method=..., correction=...)` reruns the same
estimator and settings on surrogates. Methods: `"phase"` (Fourier phase randomisation),
`"iaaft"` (also preserves the amplitude distribution), `"circular_shift"` (cross
estimators: keeps each channel, breaks their alignment). Corrections: `"max-stat"`
(family-wise error via the maximum standardised statistic, standardised over the observed
and surrogate maps together), `"fdr"` (Benjamini-Hochberg) or `"none"`. On Gaussian noise
the max-stat test's family-wise error rate measured 0.04 at a nominal 0.05 over 100
independent records.

## Pitfalls for EEG

1. **Raw b^2 is not coupling strength.** Its null level is about `1 / dof` and grows at low
   frequency: on pure noise with 1-s windows, mean b^2 is 0.56 at 2-4 Hz and 0.16 at 13-20 Hz.
   Test against surrogates and report `res.dof`.
2. **Window length must suit the lowest frequency.** With `fb = 2` the envelope SD is `2 / f`
   seconds, so a 1-s window holds one or two independent samples in delta and theta.
3. **Fixed-phase sinusoids are a degenerate null.** Any fixed phases give a constant biphase,
   so phase-randomised surrogates of pure sinusoids also score b^2 near 1 under time
   averaging. Validate with carriers whose phases drift, or with trials.
4. **Common sources.** Identical inputs make cross-bicoherence equal auto-bicoherence, so a
   cross map between electrodes cannot by itself show interaction between distinct sources;
   volume conduction produces it too.
5. **Non-sinusoidal rhythms.** An arch-shaped mu rhythm has harmonics that create
   bicoherence at (f, f, 2f) without any coupling between separate oscillators.
6. **Sums need headroom.** Check `res.params["sum_cap_hz"]` against your sampling rate and
   your amplifier's low-pass.
7. **Loud artifacts reach their neighbours.** A burst 1000x louder than its surroundings
   still contributes through the wavelet's tail several envelope SDs away.
8. **One coupled triad lights up a neighbourhood.** Every triad whose wavelets capture the
   coupled components rightly rejects "no coupling", so significance spreads over about
   three spectral SDs around the true triad (about +-1.7 Hz at 7 Hz with `fb = 2`). Report
   the peak and the extent, not the count of significant cells.
9. **`transform="fcwt"` is fast but length-dependent.** fCWT samples its mother wavelet with
   a floor index whose step shrinks with the FFT length, so its coefficients shift by up to
   about 0.6% with segment or recording length. The default `"exact"` transform does not.

## Backends and performance

`backend="auto"` picks CUDA when CuPy and a GPU are present, otherwise the Numba CPU
engine, and records its choice in `res.params["backend"]`. `"cuda"` fails loudly when
unavailable; `"reference"` is the readable NumPy implementation the others must match.

On two CPU cores the Numba engine processes about 1e8 triad-samples per second: 0.15 s per
channel-minute for 1,653 triads at 160 Hz. Surrogate inference multiplies that by the
number of surrogates, which is what the CUDA backend is for. Measure your own machine with
`python benchmarks/bench_engine.py --backend cuda`.

Recordings that exceed `memory_limit` are processed in segments with guard bands, and the
segmented result matches the whole-recording one to the guard tolerance (`tests/test_boundary.py`).

## Validation

`pytest` runs the known-answer suite; `pytest -m cuda` adds the GPU checks.

| Tests | What they establish |
| --- | --- |
| `test_cwt.py` | The transform equals a time-domain convolution with the stated wavelet; amplitude `c A / 2`, phase and frequency labels |
| `test_engine.py` | Every backend matches an independent complex128 oracle; exact biphase recovery; the analytical `1/N` null |
| `test_scale.py` | Bicoherence is unit-free (volts or microvolts); raw `B(ax, by, cz) = abc B(x, y, z)`; split runs merge exactly |
| `test_flat.py` | A flat bispectrum comes out flat at `k3 / fs^2`, with the imaginary part at noise level |
| `test_denominators.py` | The paired form stays within 1; the product form keeps legitimate values above 1 |
| `test_identities.py` | Cross with identical inputs equals auto; swapping x and y transposes the map |
| `test_boundary.py` | Segmented processing reproduces the whole-recording result to 1e-4 |
| `test_dpsi.py` | The closed-form `D(lambda)` equals direct integration for `fb` from 1 to 10 |
| `test_null.py` | The surrogate test holds its error rate and detects genuine coupling |
| `test_validation.py` | Bad input raises instead of being silently reinterpreted |
| `test_golden.py` | Outputs match stored reference arrays |
| `test_dof.py`, `test_mask.py`, `test_api.py` | The DOF model, recording-relative masks, results and options |

## Differences from the legacy modules

* Bicoherence no longer collapses on inputs in volts (a fixed `eps` used to dominate).
* No outer-window edge artifacts: whole-recording transforms, or guard-banded segments.
* Normalisations have descriptive names and are never clipped.
* The bispectrum's measure is stated, and `D(lambda)` includes the wavelet's gain and is exact.
* `fb` is a real parameter; the old `c0` never reached `fcwt.cwt`, which hard-codes `Morlet(2.0)`.
* Inputs are validated; a channels x time array is never flattened into one signal.
* Cross plots label their axes correctly.

## Citing

Please cite the software (see `CITATION.cff`) and the methods it implements:
Kim & Powers (1979), *IEEE Trans. Plasma Sci.* 7:120-131, for the paired-moment
bicoherence; Newman, Pidde & Stefanovska (2021), *Appl. Comput. Harmon. Anal.* 51:171-224,
for the wavelet bispectral density; Arts & van den Broek (2022), *Nat. Comput. Sci.*, for
fCWT and its wavelet convention.

## License

BSD 3-Clause. The optional fCWT extra is Apache-2.0 and links FFTW, which is GPL; fwhosa
itself neither includes nor requires either.
