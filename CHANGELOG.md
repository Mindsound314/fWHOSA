# Changelog

## 0.1.0 (unreleased)

First public release, replacing the five legacy modules (`fCWBispectrum_gpu.py`,
`fWHOSA_v0.1.py`, `fWHOSA_bicoherence.py`, `fWHOSA_crossbispec_v0_1.py`,
`fWHOSA_crossbicoh_v0_1.py`, tagged `v0.0-legacy`).

### Estimators
- One engine behind `bispectrum`, `bicoherence`, `cross_bispectrum`, `cross_bicoherence`.
- Bicoherence denominators renamed `paired_moment` (bounded, Kim & Powers 1979) and
  `product_of_powers` (unbounded); nothing is clipped.
- Bispectrum measures `raw`, `log_density` (Newman et al. 2021, Eq. 45) and
  `linear_density`; `D(lambda)` is the exact closed form and includes the wavelet's gain.
- `expectation` is required: `"time"`, `"trials"` or `"all"`.

### Correctness fixes relative to the legacy code
- Row scaling replaces the fixed `eps` that collapsed bicoherence on inputs in volts.
- Whole-recording transforms, or guard-banded segments, replace 5-s outer windows that
  produced edge artifacts every 4.5 s.
- `fb` is a real parameter; `c0` never reached `fcwt.cwt`, and a `c0` other than 2 rescaled
  the legacy bispectrum by 9.1x (c0 = 6) to about 19,000x (c0 = 10).
- Inputs are validated; 2-D arrays are no longer flattened by `.ravel()`.
- Cross plots label f1 and f2 on the correct axes.

### New
- `transform="exact"` (default): a length-independent FFT implementation of fCWT's Morlet
  convention. `transform="fcwt"` remains available as an optional extra.
- Validity flags (`boundary`, `conditioning`, `dof`, `undefined`) and an effective-DOF model.
- `null_test` with phase, IAAFT and circular-shift surrogates and max-stat or FDR control.
- `HOSResult` with provenance, `matrix`, `plot`, `save` and `load`.
- Known-answer test suite; golden arrays in `tests/data/golden_v0.1.npz`.

### Known limitations
- The CUDA backend is implemented to the engine contract but has not yet been validated on
  hardware in CI; run `pytest -m cuda` on your GPU before relying on it.
- `min_dof = 2` rests on the null half of the window-rule calibration; the power half is
  still to be run.
- Effective DOF uses a white-noise approximation (within about 20% of simulation for dof >= 2).
