"""
Integration tests for the opt-in Legendre angular-distribution feature.

Neutron emission is isotropic by default. When ``use_legendre_angular=True`` is
passed, discrete-level emission is weighted by the per-level Legendre angular
distributions read from the (alpha,n) data files, with a per-level fallback to
isotropic emission for any level lacking tabulated coefficients.

These tests assert three invariants:
  1. The default (flag absent / False) path is exactly isotropic.
  2. Enabling Legendre only reshapes the spectrum; the total yield is conserved.
  3. Targets with no tabulated angular data fall back to an identical spectrum.
"""

import numpy as np
import pytest

from alphanso.transport import Transport
from alphanso.parsers import get_angular_distributions
from alphanso.utils import (
    _accumulate_spectrum_legendre,
    _legendre_antideriv_grid,
)

# Be-9 carries Legendre angular data (levels 0, 1, 2); Al-27 does not, so it
# exercises the isotropic fallback path through the same code.
_TARGET_WITH_ANGULAR = {"Be-9": 1.0}
_TARGET_WITHOUT_ANGULAR = {"Al-27": 1.0}


def _beam_config(matdef, beam_energy, use_legendre_angular=None):
    config = {
        "calc_type": "beam",
        "matdef": matdef,
        "beam_energy": beam_energy,
        "calculate_gammas": False,
        "save_data_files": False,
    }
    if use_legendre_angular is not None:
        config["use_legendre_angular"] = use_legendre_angular
    return config


class TestLegendreAngular:

    @pytest.mark.integration
    def test_default_matches_explicit_isotropic(self):
        """Omitting the flag must be identical to use_legendre_angular=False."""
        default = Transport.calculate(_beam_config(_TARGET_WITH_ANGULAR, 3.5))
        explicit = Transport.calculate(
            _beam_config(_TARGET_WITH_ANGULAR, 3.5, use_legendre_angular=False))

        assert default["an_yield"] == explicit["an_yield"]
        np.testing.assert_array_equal(
            np.array(default["an_spectrum"]), np.array(explicit["an_spectrum"]))

    @pytest.mark.integration
    def test_legendre_conserves_total_yield(self):
        """Angular weighting reshapes the spectrum but must not change the yield."""
        iso = Transport.calculate(
            _beam_config(_TARGET_WITH_ANGULAR, 3.5, use_legendre_angular=False))
        leg = Transport.calculate(
            _beam_config(_TARGET_WITH_ANGULAR, 3.5, use_legendre_angular=True))

        assert np.isclose(iso["an_yield"], leg["an_yield"], rtol=1e-9)

    @pytest.mark.integration
    def test_legendre_changes_spectrum_when_data_present(self):
        """With tabulated coefficients, the spectrum shape must actually change."""
        iso = Transport.calculate(
            _beam_config(_TARGET_WITH_ANGULAR, 3.5, use_legendre_angular=False))
        leg = Transport.calculate(
            _beam_config(_TARGET_WITH_ANGULAR, 3.5, use_legendre_angular=True))

        iso_spec = np.array(iso["an_spectrum"])
        leg_spec = np.array(leg["an_spectrum"])
        assert np.max(np.abs(iso_spec - leg_spec)) > 1e-6
        # Both remain normalized spectra.
        assert np.isclose(iso_spec.sum(), leg_spec.sum(), rtol=1e-9)

    @pytest.mark.integration
    def test_fallback_identical_without_angular_data(self):
        """A target with no angular data yields a spectrum identical to isotropic."""
        assert get_angular_distributions(13027) is None  # Al-27, sanity check

        iso = Transport.calculate(
            _beam_config(_TARGET_WITHOUT_ANGULAR, 8.0, use_legendre_angular=False))
        leg = Transport.calculate(
            _beam_config(_TARGET_WITHOUT_ANGULAR, 8.0, use_legendre_angular=True))

        assert np.isclose(iso["an_yield"], leg["an_yield"], rtol=1e-12)
        np.testing.assert_allclose(
            np.array(iso["an_spectrum"]), np.array(leg["an_spectrum"]),
            rtol=0, atol=1e-12)

    @pytest.mark.integration
    def test_parser_returns_expected_structure(self):
        """get_angular_distributions returns {level_idx: {E_alpha: [coeffs]}}."""
        ang = get_angular_distributions(4009)  # Be-9
        assert ang is not None
        assert all(isinstance(k, int) for k in ang)
        # Each level maps incident energies (MeV) to a Legendre coefficient list
        # whose leading term is the normalization a_0 = 1.
        for level_dist in ang.values():
            for coeffs in level_dist.values():
                assert len(coeffs) >= 1
                assert coeffs[0] == pytest.approx(1.0)


class TestLegendreKernel:
    """Unit tests for the vectorized Legendre accumulation math (no data needed)."""

    def test_isotropic_coeffs_reproduce_uniform_emission(self):
        """coeffs = [1, 0, ...] must reproduce uniform (isotropic) binning exactly."""
        rng = np.random.default_rng(1)
        n_steps, n_levels = 5, 3
        enmin = rng.uniform(0.1, 5.0, (n_steps, n_levels))
        enmax = enmin + rng.uniform(0.2, 2.0, (n_steps, n_levels))
        yield_matrix = rng.uniform(0.0, 2.0, (n_steps, n_levels))
        valid = rng.random((n_steps, n_levels)) > 0.3
        vi, vj = np.where(valid)

        bins = np.linspace(0.0, 10.0, 51)
        b_lo = np.minimum(bins[:-1], bins[1:])
        b_hi = np.maximum(bins[:-1], bins[1:])

        # Reference: uniform density y / (enmax - enmin) over each band.
        width = (enmax - enmin)
        ref = np.zeros(len(b_lo))
        for i, j in zip(vi, vj):
            dens = yield_matrix[i, j] / width[i, j]
            lo = np.maximum(b_lo, enmin[i, j])
            hi = np.minimum(b_hi, enmax[i, j])
            ref += dens * np.maximum(0.0, hi - lo)

        coeffs = np.zeros((len(vi), 3))
        coeffs[:, 0] = 1.0
        got = _accumulate_spectrum_legendre(
            b_lo, b_hi, vi, vj, yield_matrix, enmin, enmax, coeffs)

        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)

    def test_antiderivative_matches_numerical_integral(self):
        """F(mu2) - F(mu1) must equal the integral of the Legendre series."""
        from numpy.polynomial import legendre as L

        coeffs = np.array([1.0, 0.3, -0.2, 0.1])
        mu = np.linspace(-1.0, 1.0, 200001)
        f = np.zeros_like(mu)
        for l, a in enumerate(coeffs):
            c = np.zeros(l + 1)
            c[l] = 1.0
            f += (2 * l + 1) / 2.0 * a * L.legval(mu, c)

        cut = 0.4
        mask = mu <= cut
        numerical = np.trapezoid(f[mask], mu[mask])
        analytic = (_legendre_antideriv_grid(coeffs[None, :], np.array([[cut]]))[0, 0]
                    - _legendre_antideriv_grid(coeffs[None, :], np.array([[-1.0]]))[0, 0])

        assert np.isclose(numerical, analytic, atol=1e-5)
