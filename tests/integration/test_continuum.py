"""
Sanity checks for the MT=91 continuum (alpha,n) channel.
"""

import numpy as np
import pytest
from alphanso.parsers import (
    get_an_xs, get_continuum_info, get_branching_info, get_stopping_power)
from alphanso.transport import Transport
from alphanso.utils import rebin_xs


EBINS = np.linspace(0, 14, 200)


def _integrate(zaid, e_alpha, target_mass, product_mass, cont_xs=None, cont_dist=None):
    an_xs_b = rebin_xs(get_an_xs(zaid), EBINS)
    sp_b = rebin_xs(get_stopping_power(zaid), EBINS)
    q, levels, bdata = get_branching_info(zaid)
    ep = sorted(bdata.keys())
    return Transport._integrate_over_ebins(
        e_alpha, EBINS, an_xs_b, sp_b, bdata, levels,
        q, product_mass, target_mass, ep,
        continuum_xs=cont_xs, continuum_dist=cont_dist)


class TestContinuumParsing:

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid", [4009, 8017, 8018])
    def test_continuum_info_present(self, zaid):
        """Be-9, O-17, O-18 all have MT=91 in ENDF."""
        cont_xs, cont_dist = get_continuum_info(zaid)
        assert cont_xs is not None, f"ZAID {zaid}: expected continuum XS"
        assert cont_dist is not None, f"ZAID {zaid}: expected continuum distribution"

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid", [4009, 8017, 8018])
    def test_continuum_dist_units(self, zaid):
        """Distribution energies should be in MeV (not eV): incident <20, E_out <20."""
        _, cont_dist = get_continuum_info(zaid)
        inc_energies = sorted(cont_dist.keys())
        assert inc_energies[0] < 20.0, f"ZAID {zaid}: incident energies look like eV not MeV"
        for pairs in cont_dist.values():
            e_outs = [p[0] for p in pairs]
            assert max(e_outs) < 20.0, f"ZAID {zaid}: E_out looks like eV not MeV"

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid", [4009, 8017, 8018])
    def test_continuum_dist_normalized(self, zaid):
        """Each tabulated f(E_out | E_in) should integrate to ~1 over its full range."""
        _, cont_dist = get_continuum_info(zaid)
        for e_in, pairs in cont_dist.items():
            e_outs = np.array([p[0] for p in pairs])
            probs = np.array([p[1] for p in pairs])
            integral = float(np.trapezoid(probs, e_outs))
            assert 0.5 < integral < 2.0, (
                f"ZAID {zaid} at E_in={e_in:.3f} MeV: integral={integral:.3f}, expected ~1.0")


class TestContinuumPhysics:

    @pytest.mark.integration
    def test_yield_conserved_be9_high_energy(self):
        """Total yield must be identical with and without continuum (f_91 is a redistribution)."""
        zaid = 4009
        cont_xs, cont_dist = get_continuum_info(zaid)
        r_no = _integrate(zaid, 8.0, 9.0121831, 12.0)
        r_with = _integrate(zaid, 8.0, 9.0121831, 12.0, cont_xs, cont_dist)
        assert abs(r_no[0] - r_with[0]) / r_no[0] < 1e-10, (
            f"Yield changed: {r_no[0]:.6e} -> {r_with[0]:.6e}")

    @pytest.mark.integration
    def test_yield_conserved_o17_high_energy(self):
        """Total yield must be identical with and without continuum for O-17."""
        zaid = 8017
        cont_xs, cont_dist = get_continuum_info(zaid)
        r_no = _integrate(zaid, 10.0, 16.9991315, 19.9924402)
        r_with = _integrate(zaid, 10.0, 16.9991315, 19.9924402, cont_xs, cont_dist)
        assert abs(r_no[0] - r_with[0]) / r_no[0] < 1e-10, (
            f"Yield changed: {r_no[0]:.6e} -> {r_with[0]:.6e}")

    @pytest.mark.integration
    def test_spectral_shift_be9(self):
        """
        Above the Be-9 MT=91 threshold, the continuum must shift spectral weight
        from high-energy bins toward low-energy bins.

        Be-9 at 8 MeV: continuum neutrons peak at 0-1.6 MeV.
        The low-energy region must gain weight and high-energy must lose.
        """
        zaid = 4009
        cont_xs, cont_dist = get_continuum_info(zaid)
        r_no = _integrate(zaid, 8.0, 9.0121831, 12.0)
        r_with = _integrate(zaid, 8.0, 9.0121831, 12.0, cont_xs, cont_dist)

        delta = r_with[1] - r_no[1]

        low_mask = EBINS[:-1] < 2.0
        high_mask = EBINS[:-1] > 6.0

        assert delta[low_mask].sum() > 0, "Low-energy bins should gain weight from continuum"
        assert delta[high_mask].sum() < 0, "High-energy bins should lose weight to continuum"

    @pytest.mark.integration
    def test_spectral_shift_o17(self):
        """O-17 at 10 MeV: continuum must shift weight toward low-energy bins."""
        zaid = 8017
        cont_xs, cont_dist = get_continuum_info(zaid)
        r_no = _integrate(zaid, 10.0, 16.9991315, 19.9924402)
        r_with = _integrate(zaid, 10.0, 16.9991315, 19.9924402, cont_xs, cont_dist)

        delta = r_with[1] - r_no[1]

        low_mask = EBINS[:-1] < 3.0
        high_mask = EBINS[:-1] > 7.0

        assert delta[low_mask].sum() > 0, "Low-energy bins should gain weight from continuum"
        assert delta[high_mask].sum() < 0, "High-energy bins should lose weight to continuum"

    @pytest.mark.integration
    def test_no_effect_below_threshold_be9(self):
        """Below the Be-9 MT=91 threshold (~5.69 MeV), continuum must have zero effect."""
        zaid = 4009
        cont_xs, cont_dist = get_continuum_info(zaid)
        e_below = 5.0
        r_no = _integrate(zaid, e_below, 9.0121831, 12.0)
        r_with = _integrate(zaid, e_below, 9.0121831, 12.0, cont_xs, cont_dist)

        assert np.allclose(r_no[1], r_with[1], atol=1e-20), (
            "Spectrum should be unchanged below the MT=91 threshold")

    @pytest.mark.integration
    def test_f_matrix_row_sums_be9(self):
        """f_matrix rows must sum to 1 for steps within the tabulated energy range."""
        from alphanso.utils import _preprocess_continuum_dist
        _, cont_dist = get_continuum_info(4009)
        e_steps = np.array([6.0, 7.0, 8.0, 10.0, 12.0])
        f_mat = _preprocess_continuum_dist(cont_dist, e_steps, EBINS)
        for i, e in enumerate(e_steps):
            row_sum = f_mat[i].sum()
            assert abs(row_sum - 1.0) < 1e-6, (
                f"f_matrix row at E={e} MeV sums to {row_sum:.6f}, expected 1.0")
