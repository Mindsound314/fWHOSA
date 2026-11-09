"""fwhosa: fast wavelet higher-order spectral analysis.

Four estimators share one engine and one result type::

    import fwhosa as fw
    res = fw.bicoherence(x, fs, band=(4, 30), expectation="time", window=2.0)

See the README for the estimators' definitions, the density measures and the
pitfalls (window length versus frequency, surrogate thresholds, common sources).
"""
from ._version import __version__
from .api import bicoherence, bispectrum, cross_bicoherence, cross_bispectrum
from .engine import cuda_available
from .result import HOSResult
from .stats import NullTestResult, circular_shift, iaaft, null_test, phase_randomize
from .transform import Wavelet

__all__ = [
    "__version__",
    "bicoherence",
    "bispectrum",
    "cross_bicoherence",
    "cross_bispectrum",
    "null_test",
    "NullTestResult",
    "HOSResult",
    "Wavelet",
    "phase_randomize",
    "iaaft",
    "circular_shift",
    "cuda_available",
]
