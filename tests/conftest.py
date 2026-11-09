import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: statistical tests that take tens of seconds")
    config.addinivalue_line("markers", "cuda: needs CuPy and a CUDA device (run locally, not in CI)")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def diffusing_qpc(fs, seconds, fa, fb, rng, diffusion=0.05, noise=0.5, coupled=True):
    """Carriers whose phases random-walk; the sum component is phase-locked only if coupled.

    Fixed-phase sinusoids are a degenerate null: any fixed phases give a constant
    biphase, so phase-randomised surrogates of them also score b^2 near 1.
    """
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    p1 = np.cumsum(rng.normal(0, diffusion, n))
    p2 = np.cumsum(rng.normal(0, diffusion, n))
    p3 = p1 + p2 if coupled else np.cumsum(rng.normal(0, diffusion * np.sqrt(2), n))
    x = (np.cos(2 * np.pi * fa * t + p1) + np.cos(2 * np.pi * fb * t + p2)
         + 0.6 * np.cos(2 * np.pi * (fa + fb) * t + p3))
    return x + noise * rng.standard_normal(n)


def qpc_signal(fs, seconds, fa, fb, rng, noise=0.5, amps=(1.0, 1.0, 0.6)):
    """Quadratically phase-coupled triad fa + fb -> fa + fb in white noise."""
    t = np.arange(int(round(seconds * fs))) / fs
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    x = (amps[0] * np.cos(2 * np.pi * fa * t + p1) + amps[1] * np.cos(2 * np.pi * fb * t + p2)
         + amps[2] * np.cos(2 * np.pi * (fa + fb) * t + p1 + p2))
    return x + noise * rng.standard_normal(t.size)
