"""
Sanity checks for the non-discrete (alpha,n) channels: the MT=91 continuum and
multi-particle channels such as MT=22 and MT=28.
"""

import numpy as np
import pytest
from alphanso.constants import ALPH_MASS, ANEUT_MASS
from alphanso.parsers import (
    get_an_xs, get_continuum_channels, get_branching_info, get_stopping_power)
from alphanso.transport import Transport
from alphanso.utils import (
    rebin_xs, _interpolate_continuum_table, _preprocess_continuum_dist)


EBINS = np.linspace(0, 14, 200)


def _channel(zaid, mt=91):
    matches = [c for c in get_continuum_channels(zaid) if c.mt == mt]
    return matches[0] if matches else None


def _integrate(zaid, e_alpha, target_mass, product_mass, continua=None):
    an_xs_b = rebin_xs(get_an_xs(zaid), EBINS)
    sp_b = rebin_xs(get_stopping_power(zaid), EBINS)
    q, levels, bdata = get_branching_info(zaid)
    ep = sorted(bdata.keys())
    return Transport._integrate_over_ebins(
        e_alpha, EBINS, an_xs_b, sp_b, bdata, levels,
        q, product_mass, target_mass, ep,
        continua=continua)


def _mean_cm_energy(channel, e_alpha):
    e_out, f = _interpolate_continuum_table(channel.f, e_alpha, channel.unit_base, True)
    return np.trapezoid(f * e_out, e_out) / np.trapezoid(f, e_out)


class TestContinuumParsing:

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid", [4009, 8017, 8018])
    def test_continuum_info_present(self, zaid):
        """Be-9, O-17, O-18 all have MT=91 in ENDF."""
        channel = _channel(zaid)
        assert channel is not None, f"ZAID {zaid}: expected continuum channel"
        assert channel.xs, f"ZAID {zaid}: expected continuum XS"
        assert channel.f is not None, f"ZAID {zaid}: expected continuum distribution"

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid", [4009, 8017, 8018])
    def test_continuum_dist_units(self, zaid):
        """Distribution energies should be in MeV (not eV): incident <20, E_out <20."""
        channel = _channel(zaid)
        inc_energies = sorted(channel.f.keys())
        assert inc_energies[0] < 20.0, f"ZAID {zaid}: incident energies look like eV not MeV"
        for pairs in channel.f.values():
            e_outs = [p[0] for p in pairs]
            assert max(e_outs) < 20.0, f"ZAID {zaid}: E_out looks like eV not MeV"

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid", [4009, 8017, 8018])
    def test_continuum_dist_normalized(self, zaid):
        """Each tabulated f(E_out | E_in) should integrate to ~1 over its full range."""
        channel = _channel(zaid)
        for e_in, pairs in channel.f.items():
            e_outs = np.array([p[0] for p in pairs])
            probs = np.array([p[1] for p in pairs])
            integral = float(np.trapezoid(probs, e_outs))
            assert 0.5 < integral < 2.0, (
                f"ZAID {zaid} at E_in={e_in:.3f} MeV: integral={integral:.3f}, expected ~1.0")

    @pytest.mark.integration
    def test_q_value_is_continuum_start(self):
        """Be-9 MT=91 Q must be QI (-3.93983 MeV), not the ground-state <Q> (+5.70117 MeV)."""
        channel = _channel(4009)
        assert channel.q_value == pytest.approx(-3.93983)

    @pytest.mark.integration
    def test_q_value_matches_threshold(self):
        """The continuum Q must reproduce the MT=91 cross-section threshold."""
        channel = _channel(4009)
        energies = sorted(channel.xs.keys())
        first = next(i for i, e in enumerate(energies) if channel.xs[e] > 0.0)
        threshold = energies[first - 1]
        product_mass = 12.0
        expected = -channel.q_value * (ANEUT_MASS + product_mass) / (
            ANEUT_MASS + product_mass - ALPH_MASS)
        assert threshold == pytest.approx(expected, rel=2e-3)

    @pytest.mark.integration
    def test_kalbach_mann_fields(self):
        """Be-9 MT=91 is Kalbach-Mann in the CM frame, on a unit base, with r tabulated."""
        channel = _channel(4009)
        assert channel.frame == 'centerOfMass'
        assert channel.unit_base
        assert channel.r is not None
        assert sorted(channel.r.keys()) == sorted(channel.f.keys())

    @pytest.mark.integration
    def test_uncorrelated_distribution_parsed(self):
        """F-19 stores MT=91 as an uncorrelated isotropic CM distribution."""
        channel = _channel(9019)
        assert channel is not None
        assert channel.f is not None
        assert channel.r is None
        assert channel.frame == 'centerOfMass'

    @pytest.mark.integration
    def test_unspecified_distribution(self):
        """B-10 leaves the MT=91 distribution unspecified but still has its continuum Q."""
        channel = _channel(5010)
        assert channel is not None
        assert channel.f is None
        assert channel.q_value == pytest.approx(-5.82611)


class TestContinuumInterpolation:

    @pytest.mark.integration
    def test_unit_base_interpolates_range(self):
        """Between two tables the E_out range is interpolated, not merged."""
        channel = _channel(4009)
        e_out_lo = np.array(channel.f[6.0])[:, 0]
        e_out_hi = np.array(channel.f[6.5])[:, 0]
        e_out, f = _interpolate_continuum_table(channel.f, 6.25, True, True)
        assert e_out[-1] == pytest.approx(0.5 * (e_out_lo[-1] + e_out_hi[-1]))
        assert np.trapezoid(f, e_out) == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.integration
    def test_unit_base_near_threshold(self):
        """Just above threshold the Be-9 continuum must be narrow, not a mix of 0 and 0.2 MeV."""
        channel = _channel(4009)
        e_out, _ = _interpolate_continuum_table(channel.f, 5.75, True, True)
        assert e_out[-1] < 0.05


class TestContinuumPhysics:

    @pytest.mark.integration
    def test_yield_conserved_be9_high_energy(self):
        """Total yield must be identical with and without continuum (f_91 is a redistribution)."""
        zaid = 4009
        channel = _channel(zaid)
        r_no = _integrate(zaid, 8.0, 9.0121831, 12.0)
        r_with = _integrate(zaid, 8.0, 9.0121831, 12.0, [channel])
        assert abs(r_no[0] - r_with[0]) / r_no[0] < 1e-10, (
            f"Yield changed: {r_no[0]:.6e} -> {r_with[0]:.6e}")

    @pytest.mark.integration
    def test_yield_conserved_o17_high_energy(self):
        """Total yield must be identical with and without continuum for O-17."""
        zaid = 8017
        channel = _channel(zaid)
        r_no = _integrate(zaid, 10.0, 16.9991315, 19.9924402)
        r_with = _integrate(zaid, 10.0, 16.9991315, 19.9924402, [channel])
        assert abs(r_no[0] - r_with[0]) / r_no[0] < 1e-10, (
            f"Yield changed: {r_no[0]:.6e} -> {r_with[0]:.6e}")

    @pytest.mark.integration
    def test_spectral_shift_be9(self):
        """
        Above the Be-9 MT=91 threshold, the continuum must shift spectral weight
        from high-energy bins toward low-energy bins.
        """
        zaid = 4009
        channel = _channel(zaid)
        r_no = _integrate(zaid, 8.0, 9.0121831, 12.0)
        r_with = _integrate(zaid, 8.0, 9.0121831, 12.0, [channel])

        delta = r_with[1] - r_no[1]

        low_mask = EBINS[:-1] < 2.0
        high_mask = EBINS[:-1] > 6.0

        assert delta[low_mask].sum() > 0, "Low-energy bins should gain weight from continuum"
        assert delta[high_mask].sum() < 0, "High-energy bins should lose weight to continuum"

    @pytest.mark.integration
    def test_spectral_shift_o17(self):
        """O-17 at 10 MeV: continuum must shift weight toward low-energy bins."""
        zaid = 8017
        channel = _channel(zaid)
        r_no = _integrate(zaid, 10.0, 16.9991315, 19.9924402)
        r_with = _integrate(zaid, 10.0, 16.9991315, 19.9924402, [channel])

        delta = r_with[1] - r_no[1]

        low_mask = EBINS[:-1] < 3.0
        high_mask = EBINS[:-1] > 7.0

        assert delta[low_mask].sum() > 0, "Low-energy bins should gain weight from continuum"
        assert delta[high_mask].sum() < 0, "High-energy bins should lose weight to continuum"

    @pytest.mark.integration
    def test_no_effect_below_threshold_be9(self):
        """Below the Be-9 MT=91 threshold (~5.69 MeV), continuum must have zero effect."""
        zaid = 4009
        channel = _channel(zaid)
        e_below = 5.0
        r_no = _integrate(zaid, e_below, 9.0121831, 12.0)
        r_with = _integrate(zaid, e_below, 9.0121831, 12.0, [channel])

        assert np.allclose(r_no[1], r_with[1], atol=1e-20), (
            "Spectrum should be unchanged below the MT=91 threshold")

    @pytest.mark.integration
    def test_f_matrix_row_sums_be9(self):
        """f_matrix rows must sum to 1 for steps within the tabulated energy range."""
        channel = _channel(4009)
        e_steps = np.array([6.0, 7.0, 8.0, 10.0, 12.0])
        f_mat = _preprocess_continuum_dist(channel, e_steps, EBINS, 9.0121831)
        for i, e in enumerate(e_steps):
            row_sum = f_mat[i].sum()
            assert abs(row_sum - 1.0) < 1e-6, (
                f"f_matrix row at E={e} MeV sums to {row_sum:.6f}, expected 1.0")

    @pytest.mark.integration
    def test_isotropic_cm_to_lab_mean_energy(self):
        """For isotropic CM emission the lab mean energy is <E'> + E_c exactly."""
        zaid, target_mass = 9019, 18.9984032
        channel = _channel(zaid)
        e_alpha = 12.0
        bins = np.linspace(0.0, 14.0, 2801)
        row = _preprocess_continuum_dist(
            channel, np.array([e_alpha]), bins, target_mass, e_spacing=0.0, n_sub=16)[0]
        mean_lab = np.sum(row * 0.5 * (bins[:-1] + bins[1:]))
        e_c = ALPH_MASS * ANEUT_MASS / (ALPH_MASS + target_mass) ** 2 * e_alpha
        assert mean_lab == pytest.approx(_mean_cm_energy(channel, e_alpha) + e_c, rel=1e-3)

    @pytest.mark.integration
    def test_cm_to_lab_spreads_low_energy_continuum(self):
        """
        Just above the Be-9 threshold the CM continuum ends below 0.1 MeV, but the
        center-of-mass motion carries lab energies up to (sqrt(E') + sqrt(E_c))^2.
        """
        zaid, target_mass = 4009, 9.0121831
        channel = _channel(zaid)
        e_alpha = 6.0
        bins = np.linspace(0.0, 2.0, 401)
        row = _preprocess_continuum_dist(
            channel, np.array([e_alpha]), bins, target_mass, e_spacing=0.0)[0]
        e_out, _ = _interpolate_continuum_table(channel.f, e_alpha, True, True)
        e_c = ALPH_MASS * ANEUT_MASS / (ALPH_MASS + target_mass) ** 2 * e_alpha
        lab_max = (np.sqrt(e_out[-1]) + np.sqrt(e_c)) ** 2
        above_cm = row[bins[:-1] >= e_out[-1]].sum()
        assert above_cm > 0.3
        assert row[bins[:-1] >= lab_max].sum() == 0.0

    @pytest.mark.integration
    def test_coarse_alpha_grid_matches_exact(self):
        """Evaluating the shape every 20 keV must match evaluating it at every step."""
        zaid, target_mass = 4009, 9.0121831
        channel = _channel(zaid)
        e_steps = np.linspace(9.0, 5.7, 700)
        exact = _preprocess_continuum_dist(channel, e_steps, EBINS, target_mass, e_spacing=0.0)
        coarse = _preprocess_continuum_dist(channel, e_steps, EBINS, target_mass)
        weights = np.ones(len(e_steps))
        spec_exact = weights @ exact
        spec_coarse = weights @ coarse
        assert np.abs(spec_coarse - spec_exact).sum() / spec_exact.sum() < 2e-3

    @pytest.mark.integration
    def test_box_fallback_uses_continuum_q(self):
        """
        Without a tabulated distribution the box must end at the two-body maximum
        for the continuum Q, not the ground-state Q.
        """
        zaid = 5010
        channel = _channel(zaid)
        target_mass, product_mass = 10.0129369, 13.0057386
        e_alpha = 9.8
        _, spec_with, *_ = _integrate(zaid, e_alpha, target_mass, product_mass, [channel])
        _, spec_no, *_ = _integrate(zaid, e_alpha, target_mass, product_mass)
        delta = spec_with - spec_no

        term1 = np.sqrt(ALPH_MASS * ANEUT_MASS * e_alpha) / (ANEUT_MASS + product_mass)
        term2 = ALPH_MASS * ANEUT_MASS * e_alpha / (ANEUT_MASS + product_mass) ** 2
        term3 = (product_mass * e_alpha + product_mass * channel.q_value
                 - ALPH_MASS * e_alpha) / (ANEUT_MASS + product_mass)
        enmax = (term1 + np.sqrt(term2 + term3)) ** 2
        assert delta[EBINS[:-1] > enmax + EBINS[1]].max() <= 0.0
        assert delta[EBINS[1:] < enmax].sum() > 0.0


def _raw_xs(zaid, mt):
    import xml.etree.ElementTree as ET
    from alphanso.parsers import _find_gnds_xml, _get_cross_section_from_reaction
    root = ET.parse(_find_gnds_xml(zaid, None)).getroot()
    xs = _get_cross_section_from_reaction(root.find(f".//reaction[@ENDF_MT='{mt}']"))
    energies = sorted(xs.keys())
    return lambda e: np.interp(e, energies, [xs[k] for k in energies], left=0.0, right=0.0)


def _channel_xs(channel, e):
    energies = sorted(channel.xs.keys())
    return np.interp(e, energies, [channel.xs[k] for k in energies], left=0.0, right=0.0)


class TestMultiParticleChannels:

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid, mts", [
        (4009, {91, 22}), (5010, {91, 28, 22}), (8018, {91, 16, 22})])
    def test_channels_found(self, zaid, mts):
        """Every non-discrete neutron channel of the evaluation is read."""
        assert {c.mt for c in get_continuum_channels(zaid)} == mts

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid", [6013, 26056])
    def test_no_channels_without_levels_and_total(self, zaid):
        """SOURCES targets (C-13) and TENDL files without MT=201 and levels get none."""
        assert get_continuum_channels(zaid) == []

    @pytest.mark.integration
    def test_multiplicity_applied(self):
        """O-18 (alpha,2n) produces two neutrons per reaction."""
        channel = _channel(8018, 16)
        assert _channel_xs(channel, 12.0) == pytest.approx(2.0 * _raw_xs(8018, 16)(12.0))

    @pytest.mark.integration
    def test_be9_breakup_capped_below_table_threshold(self):
        """
        Be-9 MT=22 jumps from 0 at 2.4 MeV to 0.025 b at 4.64 MeV, so interpolation
        gives ~0.024 b at 4.5 MeV where MT=201 leaves only ~0.0003 b beyond the
        levels. The cap removes it and leaves the channel unchanged above, apart
        from rounding in the evaluated partials.
        """
        channel = _channel(4009, 22)
        assert _raw_xs(4009, 22)(4.5) > 0.02
        assert _channel_xs(channel, 4.5) < 1e-3
        for e in (4.7, 5.5, 7.0):
            assert _channel_xs(channel, e) == pytest.approx(_raw_xs(4009, 22)(e), rel=1e-3)

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid, mt, e_alpha, share", [
        (4009, 22, 5.5, 0.317), (5010, 28, 5.5, 0.633)])
    def test_channel_share_of_total(self, zaid, mt, e_alpha, share):
        """Be-9 breakup is ~32% and B-10 (alpha,np) ~63% of neutron production at 5.5 MeV."""
        total = _raw_xs(zaid, 201)(e_alpha)
        assert _channel_xs(_channel(zaid, mt), e_alpha) / total == pytest.approx(share, abs=0.01)

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid, e_alpha, target_mass, product_mass", [
        (4009, 5.49, 9.0121831, 12.0),
        (4009, 8.0, 9.0121831, 12.0),
        (5010, 5.49, 10.0129369, 13.0057386),
        (8018, 7.0, 17.9991596, 20.9938467)])
    def test_yield_conserved_all_channels(self, zaid, e_alpha, target_mass, product_mass):
        """All channels together only redistribute the total yield."""
        r_no = _integrate(zaid, e_alpha, target_mass, product_mass)
        r_with = _integrate(zaid, e_alpha, target_mass, product_mass,
                            get_continuum_channels(zaid))
        assert r_with[0] == pytest.approx(r_no[0], rel=1e-10)

    @pytest.mark.integration
    @pytest.mark.parametrize("zaid, target_mass, product_mass", [
        (4009, 9.0121831, 12.0), (5010, 10.0129369, 13.0057386)])
    def test_breakup_raises_low_energy_share(self, zaid, target_mass, product_mass):
        """At Am-241 energy the breakup channels move weight below 1.5 MeV."""
        r_no = _integrate(zaid, 5.49, target_mass, product_mass)
        r_with = _integrate(zaid, 5.49, target_mass, product_mass,
                            get_continuum_channels(zaid))
        low = EBINS[1:] <= 1.5
        assert r_with[1][low].sum() / r_with[0] > r_no[1][low].sum() / r_no[0] + 0.05
