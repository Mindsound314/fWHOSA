"""Test 10: the closed-form D_psi agrees with direct integration across bandwidths."""
import numpy as np
import pytest

from fwhosa.normalize import dpsi, dpsi_quadrature


@pytest.mark.parametrize("fb", [1.0, 2.0, 5.0, 10.0])
def test_closed_form_matches_quadrature(fb):
    for lam in np.linspace(0.05, 0.95, 7):
        exact = float(dpsi(lam, fb))
        quad = dpsi_quadrature(lam, fb)
        assert abs(exact / quad - 1.0) < 1e-6, (fb, lam, exact, quad)


@pytest.mark.parametrize("fb", [1.0, 2.0, 10.0])
def test_symmetric_positive_finite(fb):
    lam = np.linspace(0.0, 1.0, 101)
    d = dpsi(lam, fb)
    assert np.all(np.isfinite(d)) and np.all(d > 0)
    np.testing.assert_allclose(d, d[::-1], rtol=1e-12)
