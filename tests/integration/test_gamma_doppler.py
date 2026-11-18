"""
Tests for Doppler broadening of the alpha-induced gamma lines.

Covers the recoil-slowing and line-shape mathematics on synthetic data, the
RIPL-3 lifetime data, and the binned spectrum of all four calculation types.
SaG4n/Geant4 emits unshifted lines, so physics checks use analytic results and
published line widths instead.
"""

import logging
import os

import numpy as np
import pytest

from alphanso import __main__ as cli
from alphanso import gamma_doppler as gd
from alphanso import gamma_production as gp
from alphanso.data_manager import get_data_dir
from alphanso.parsers import get_level_half_lives
from alphanso.transport import Transport
from tests.integration.configs import SANDWICH_CONFIGS

DATA_ROOT = str(get_data_dir())
GAMMA_ROOT = os.path.join(DATA_ROOT, "gamma")

requires_gamma_data = pytest.mark.skipif(
    not os.path.isdir(GAMMA_ROOT),
    reason="gamma data not installed",
)

SPECTRUM_KEYS = {
    "gamma_energy_bins",
    "gamma_spectrum",
    "gamma_spectrum_absolute",
    "gamma_recoil_density",
    "gamma_recoil_density_assumed",
}


def _scheme(levels):
    """Build a LevelScheme from {level: [(E_gamma, weight, next_level), ...]} with energies in MeV."""
    energies = sorted(levels)
    index = {energy: i for i, energy in enumerate(energies)}
    gammas = [[(e_gamma, weight, index.get(next_level)) for e_gamma, weight, next_level in levels[energy]]
              for energy in energies]
    return gp.LevelScheme(energies, gammas)


def _line_fwhm_kev(results, energy, half_window=0.3):
    edges = np.array(results["gamma_energy_bins"])
    centers = 0.5 * (edges[1:] + edges[:-1])
    spectrum = np.array(results["gamma_spectrum_absolute"])
    window = np.abs(centers - energy) < half_window
    above = centers[window][spectrum[window] >= spectrum[window].max() / 2.0]
    return (above.max() - above.min()) * 1e3


def _fraction_near(results, energy, half_width_kev=2.5):
    """Fraction of a line's intensity within half_width_kev of its rest-frame energy."""
    edges = np.array(results["gamma_energy_bins"])
    centers = 0.5 * (edges[1:] + edges[:-1])
    spectrum = np.array(results["gamma_spectrum_absolute"])
    lines = np.array(results["gamma_lines"])
    intensity = lines[np.argmin(np.abs(lines[:, 0] - energy)), 1]
    return spectrum[np.abs(centers - energy) * 1e3 <= half_width_kev].sum() / intensity


@pytest.fixture(scope="module")
def be9():
    return Transport.calculate(
        {"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0, "density": 1.85})


@pytest.fixture(scope="module")
def carbon13():
    return Transport.calculate(
        {"calc_type": "beam", "matdef": {"C-13": 1.0}, "beam_energy": 8.0, "density": 2.26})


@pytest.fixture(scope="module")
def puo2_by_density():
    matdef = {"Pu-239": 239.052 / 271.05, "O": 32.0 / 271.05}
    return {
        density: Transport.beam_problem([[5.15, 1.0]], matdef, density=density)
        for density in (2.3, 11.46)
    }


@pytest.fixture(scope="module")
def sandwich():
    return Transport.calculate(SANDWICH_CONFIGS[0])


class TestRecoilStopping:

    def test_lss_matches_textbook_formula(self):
        # S_e = 3.83e-15 Z1^(7/6) Z2 / (Z1^(2/3) + Z2^(2/3))^(3/2) sqrt(E[keV]/M1) eV cm^2
        z, mass, z2, energy_kev = 10, 20.0, 8, 500.0
        speed = np.sqrt(2.0 * energy_kev * 1e-3 / (mass * gd.AMU_TO_MEV))
        expected = (3.83e-15 * z ** (7 / 6) * z2 / (z ** (2 / 3) + z2 ** (2 / 3)) ** 1.5
                    * np.sqrt(energy_kev / mass))
        assert gd.electronic_stopping(z, z2, np.array([speed]))[0] == pytest.approx(expected, rel=2e-3)

    def test_electronic_stopping_saturates_above_v0_z_two_thirds(self):
        z = 6
        limit = z ** (2 / 3) / gd.BOHR_SPEED_INVERSE
        low, at_limit, above = gd.electronic_stopping(z, 4, np.array([0.5 * limit, limit, 2.0 * limit]))
        assert low == pytest.approx(0.5 * at_limit)
        assert above == pytest.approx(at_limit)

    def test_zbl_reduced_stopping(self):
        # Reduced energy 1 gives s_n = ln(2.1383) / (2 * 1.20914) = 0.3143
        z, mass, z2, mass2 = 10, 20.0, 8, 16.0
        screening = z ** 0.23 + z2 ** 0.23
        energy_kev = z * z2 * (mass + mass2) * screening / (32.53 * mass2)
        speed = np.sqrt(2.0 * energy_kev * 1e-3 / (mass * gd.AMU_TO_MEV))
        expected = 8.462e-15 * z * z2 * mass * 0.3143 / ((mass + mass2) * screening)
        assert gd.nuclear_stopping(z, mass, z2, mass2, np.array([speed]))[0] == pytest.approx(expected, rel=1e-3)

    def test_transit_times_scale_inversely_with_atom_density(self):
        edges = np.linspace(0.0, 0.02, 21)
        composition = ((8, 16.0, 1.0),)
        dense = gd.transit_times(10, 20.0, edges, gd.RecoilMedium(composition, 1e23))
        thin = gd.transit_times(10, 20.0, edges, gd.RecoilMedium(composition, 5e22))
        assert np.isinf(dense[0])
        assert thin[1:] == pytest.approx(2.0 * dense[1:])

    def test_slowing_time_magnitude(self):
        # Solids stop MeV recoils within about a picosecond.
        speed = np.sqrt(2.0 * 1.0 / (20.0 * gd.AMU_TO_MEV))
        edges = np.linspace(0.0, speed, 201)
        medium = gd.RecoilMedium(((8, 16.0, 2 / 3), (94, 239.0, 1 / 3)), 7.6e22)
        half_speed = gd.transit_times(10, 20.0, edges, medium)[100:].sum()
        assert 1e-13 < half_speed < 1e-12


class TestEmissionHistogram:

    def test_zero_lifetime_emits_on_arrival(self):
        arrivals = np.array([0.0, 1.0, 2.0])
        assert gd.emission_histogram(arrivals, 0.0, np.array([np.inf, 1.0, 1.0])).tolist() == [0.0, 1.0, 2.0]

    def test_long_lifetime_emits_at_rest(self):
        arrivals = np.array([0.0, 1.0, 2.0])
        emitted = gd.emission_histogram(arrivals, 1e6, np.array([np.inf, 1.0, 1.0]))
        assert emitted[0] == pytest.approx(3.0, rel=1e-5)

    def test_population_is_conserved(self):
        arrivals = np.random.default_rng(1).random(50)
        transit = np.full(50, 2.0)
        transit[0] = np.inf
        assert gd.emission_histogram(arrivals, 7.0, transit).sum() == pytest.approx(arrivals.sum())

    def test_exponential_decay_while_slowing(self):
        # All nuclei start mid-way through the top bin of a uniform clock:
        # the fraction reaching the bottom bin is exp(-T / tau).
        transit = np.full(101, 0.1)
        transit[0] = np.inf
        arrivals = np.zeros(101)
        arrivals[-1] = 1.0
        emitted = gd.emission_histogram(arrivals, 4.0, transit)
        assert emitted[0] == pytest.approx(np.exp(-(100 * 0.1 - 0.05) / 4.0), rel=1e-9)


class TestCascade:

    def test_cascade_delay_lowers_the_emission_speed(self):
        # 2 MeV level (lifetime 1) feeds a 1 MeV level (lifetime 1).
        scheme = _scheme({2.0: [(1.0, 1.0, 1.0)], 1.0: [(0.999, 1.0, None)]})
        transit = np.full(51, 0.1)
        transit[0] = np.inf
        start = np.zeros(51)
        start[-1] = 1.0
        lines = gd.cascade_line_histograms(scheme, {1: start}, np.array([1.0, 1.0]), transit)
        speeds = np.arange(51)
        first, second = lines[1.0], lines[0.999]
        assert first.sum() == pytest.approx(1.0)
        assert second.sum() == pytest.approx(1.0)
        assert (second * speeds).sum() < (first * speeds).sum()
        # A sum of two exponentials: the mean delay of the second gamma is twice that of the first.
        assert 50 - (second * speeds).sum() == pytest.approx(2 * (50 - (first * speeds).sum()), rel=0.05)

    def test_branching_matches_expected_lines(self):
        scheme = _scheme({
            2.0: [(1.0, 25.0, 1.0), (2.0, 75.0, None)],
            1.0: [(1.001, 1.0, None)],
        })
        transit = np.full(10, 1.0)
        transit[0] = np.inf
        start = np.zeros(10)
        start[-1] = 1.0
        lines = gd.cascade_line_histograms(scheme, {1: start}, np.array([0.5, 0.5]), transit)
        totals = {energy: histogram.sum() for energy, histogram in lines.items()}
        assert totals == pytest.approx(scheme.expected_lines(1))


class TestDepositLine:

    def test_single_speed_is_a_uniform_box(self):
        edges = np.linspace(0.9, 1.1, 201)
        spectrum = np.zeros(200)
        gd.deposit_line(spectrum, edges, 1.0, np.array([2.0]), np.array([0.05]))
        lorentz = 1.0 / np.sqrt(1.0 - 0.05 ** 2)
        centers = 0.5 * (edges[1:] + edges[:-1])
        inside = (centers > lorentz * 0.95 + 1e-3) & (centers < lorentz * 1.05 - 1e-3)
        assert spectrum.sum() == pytest.approx(2.0)
        assert spectrum[inside] == pytest.approx(2.0 * 1e-3 / (lorentz * 0.1))
        assert spectrum[(centers < lorentz * 0.95 - 1e-3) | (centers > lorentz * 1.05 + 1e-3)].max() == 0.0

    def test_at_rest_goes_to_the_containing_bin(self):
        spectrum = np.zeros(3)
        gd.deposit_line(spectrum, np.array([0.0, 1.0, 2.0, 3.0]), 1.5, np.array([4.0]), np.array([0.0]))
        assert spectrum.tolist() == [0.0, 4.0, 0.0]

    def test_at_rest_on_the_last_edge_stays_in_range(self):
        spectrum = np.zeros(2)
        gd.deposit_line(spectrum, np.array([0.0, 1.0, 2.0]), 2.0, np.array([1.0]), np.array([0.0]))
        assert spectrum.tolist() == [0.0, 1.0]

    def test_non_uniform_edges_conserve_intensity(self):
        edges = np.array([0.5, 0.99, 0.995, 1.0, 1.02, 1.5])
        spectrum = np.zeros(5)
        gd.deposit_line(spectrum, edges, 1.0, np.array([0.5, 0.25, 0.25]), np.array([0.0, 0.01, 0.02]))
        assert spectrum.sum() == pytest.approx(1.0)

    def test_outside_the_grid_is_not_binned(self):
        spectrum = np.zeros(1)
        gd.deposit_line(spectrum, np.array([0.0, 1.0]), 5.0, np.array([1.0, 1.0]), np.array([0.0, 0.01]))
        assert spectrum[0] == 0.0


class TestAgainstMonteCarlo:

    def test_deterministic_shape_matches_sampling_of_the_same_model(self):
        # A level with T1/2 = 0.73 ps populated by Ne-20 recoils slowing down in PuO2.
        z, mass, energy, lifetime = 10, 20.0, 1.634, 0.73e-12 / np.log(2.0)
        low, high = np.array([0.006]), np.array([0.014])
        medium = gd.RecoilMedium(((8, 16.0, 2 / 3), (94, 239.0, 1 / 3)), 7.64e22)
        edges = energy + 2.5e-4 + np.linspace(-0.03, 0.03, 121)     # the line is mid-bin

        speed_edges = np.linspace(0.0, high[0] * (1.0 + 1e-9), gd.NUM_SPEED_BINS + 1)
        speeds = 0.5 * (speed_edges[:-1] + speed_edges[1:])
        speeds[0] = 0.0
        arrivals = gd._speed_kernel(speed_edges, low, high)[0]
        emitted = gd.emission_histogram(
            arrivals, lifetime, gd.transit_times(z, mass, speed_edges, medium))
        deterministic = np.zeros(120)
        gd.deposit_line(deterministic, edges, energy, emitted, speeds)

        # Sample the initial speed, the decay time and the emission angle.
        rng = np.random.default_rng(20261104)
        samples = 2_000_000
        table = np.geomspace(1e-5, 0.015, 20000)
        stopping = gd.recoil_stopping_power(z, mass, table, medium.composition) * 1e-6
        clock_rate = mass * gd.AMU_TO_MEV / (medium.atom_density * stopping * gd.SPEED_OF_LIGHT)
        clock = np.concatenate(([0.0], np.cumsum(0.5 * (clock_rate[1:] + clock_rate[:-1]) * np.diff(table))))
        initial = np.sqrt(rng.uniform(low[0] ** 2, high[0] ** 2, samples))
        emission_clock = np.interp(initial, table, clock) - rng.exponential(lifetime, samples)
        beta = np.interp(emission_clock, clock, table, left=0.0)
        lab = energy / np.sqrt(1.0 - beta ** 2) * (1.0 + beta * rng.uniform(-1.0, 1.0, samples))
        sampled = np.histogram(lab, bins=edges)[0] / samples

        assert deterministic.sum() == pytest.approx(1.0)
        assert np.cumsum(deterministic) == pytest.approx(np.cumsum(sampled), abs=5e-3)
        # The sharp peak holds the same fraction of the line.
        central = np.abs(0.5 * (edges[1:] + edges[:-1]) - energy) < 2e-3
        assert deterministic[central].sum() == pytest.approx(sampled[central].sum(), abs=0.01)


class TestInputs:

    def test_default_grid_is_1_kev_over_0_to_15_mev(self):
        edges = gd.parse_gamma_energy_bins()
        assert edges[0] == 0.0 and edges[-1] == 15.0
        assert np.diff(edges) == pytest.approx(1e-3)

    def test_three_element_list_is_shorthand(self):
        assert gd.parse_gamma_energy_bins([0.0, 2.0, 5]).tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_three_element_array_is_explicit_edges(self):
        assert gd.parse_gamma_energy_bins(np.array([0.0, 2.0, 5.0])).tolist() == [0.0, 2.0, 5.0]

    def test_descending_edges_are_returned_ascending(self):
        assert gd.parse_gamma_energy_bins([3.0, 2.0, 1.0, 0.0]).tolist() == [0.0, 1.0, 2.0, 3.0]

    @pytest.mark.parametrize("bins", [[1.0], [0.0, 1.0, 1.0, 2.0], np.array([0.0, 2.0, 1.0])])
    def test_invalid_edges_are_rejected(self, bins):
        with pytest.raises(ValueError):
            gd.parse_gamma_energy_bins(bins)

    def test_assumed_density_uses_the_default_atom_density(self):
        medium, density, assumed = gd.recoil_medium({8016: 2 / 3, 94239: 1 / 3})
        assert assumed
        assert medium.atom_density == gd.DEFAULT_ATOM_DENSITY
        assert density == pytest.approx(12.0, rel=0.02)

    def test_explicit_density(self):
        medium, density, assumed = gd.recoil_medium({8016: 2 / 3, 94239: 1 / 3}, 11.46)
        assert not assumed
        assert density == 11.46
        assert medium.atom_density == pytest.approx(7.64e22, rel=0.01)

    @pytest.mark.parametrize("density", [0.0, -1.0])
    def test_non_positive_density_is_rejected(self, density):
        with pytest.raises(ValueError):
            gd.recoil_medium({4009: 1.0}, density)

    def test_assumed_density_warns_once(self, monkeypatch, caplog):
        monkeypatch.setattr(gd, "_warned_density", False)
        with caplog.at_level(logging.WARNING, logger=gd.logger.name):
            gd.recoil_medium({4009: 1.0})
            gd.recoil_medium({4009: 1.0})
        assert len([r for r in caplog.records if "No density given" in r.message]) == 1

    @pytest.mark.parametrize("field", ["density", "target_density"])
    def test_cli_rejects_non_positive_density(self, field):
        config = {"calc_type": "beam", "beam_energy": 5.0, field: -2.0}
        with pytest.raises(ValueError, match=field):
            cli._validate_and_normalize_config(config, "test")


@requires_gamma_data
class TestLifetimeData:

    def test_ripl_half_lives(self):
        carbon = dict(get_level_half_lives(6, 12))
        assert carbon[0.0] is None
        assert carbon[4.43982] == pytest.approx(4.224e-14)
        assert dict(get_level_half_lives(8, 16))[6.12989] == pytest.approx(1.84e-11)
        assert dict(get_level_half_lives(10, 21))[0.350727] == pytest.approx(7.13e-12)

    def test_untabulated_nuclide(self):
        assert get_level_half_lives(6, 200) == []

    def test_levels_are_matched_to_the_scheme(self):
        target = gp.get_gamma_target_data(4009, GAMMA_ROOT)
        channel = next(ch for ch in target.channels if ch.label == "MT51")
        lifetimes = gd._level_lifetimes(channel.scheme, 6, 12)
        assert lifetimes[channel.level] == pytest.approx(4.224e-14 / np.log(2.0))


@requires_gamma_data
class TestRecoilKinematics:

    def test_be9_4p44_mev_channel(self):
        # Hand calculation at 5 MeV: b_cm = 0.0159, b' = 0.0081, so the unslowed
        # 4.44 MeV line extends to about +/-107 keV.
        target = gp.get_gamma_target_data(4009, GAMMA_ROOT)
        channel = next(ch for ch in target.channels if ch.label == "MT51")
        low, high = gd.recoil_speed_range(channel, target, np.array([5.0]))
        assert low[0] == pytest.approx(0.0159 - 0.0081, abs=2e-4)
        assert high[0] == pytest.approx(0.0159 + 0.0081, abs=2e-4)
        assert channel.residual == (6, 12)

    def test_multi_particle_channel_uses_ejectile_momentum(self):
        target = gp.get_gamma_target_data(4009, GAMMA_ROOT)
        channel = next(ch for ch in target.channels if ch.kind == gp.GammaChannel.KIND_WALK)
        low, high = gd.recoil_speed_range(channel, target, np.array([8.0]))
        beta_cm = np.sqrt(2.0 * 8.0 / (gd.ALPH_MASS * gd.AMU_TO_MEV)) * gd.ALPH_MASS / (gd.ALPH_MASS + 9.0)
        assert high[0] > beta_cm > low[0]


@requires_gamma_data
class TestBeamSpectrum:

    def test_on_by_default_and_conserves_the_yield(self, be9):
        assert SPECTRUM_KEYS <= set(be9)
        assert sum(be9["gamma_spectrum_absolute"]) == pytest.approx(be9["gamma_yield"], rel=1e-9)
        assert sum(be9["gamma_spectrum"]) == pytest.approx(1.0)

    def test_grid_is_ascending_and_matches_the_spectrum(self, be9):
        edges = np.array(be9["gamma_energy_bins"])
        assert np.all(np.diff(edges) > 0.0)
        assert len(be9["gamma_spectrum"]) == edges.size - 1
        assert len(be9["gamma_spectrum_absolute"]) == edges.size - 1

    def test_density_is_recorded(self, be9):
        assert be9["gamma_recoil_density"] == 1.85
        assert be9["gamma_recoil_density_assumed"] is False

    def test_assumed_density_is_recorded(self):
        results = Transport.calculate({"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0})
        assert results["gamma_recoil_density_assumed"] is True
        assert results["gamma_recoil_density"] == pytest.approx(8e22 * 9.012 / 6.022e23, rel=1e-3)

    def test_doppler_leaves_rest_frame_results_unchanged(self, be9):
        off = Transport.calculate(
            {"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0, "gamma_doppler": False})
        assert not SPECTRUM_KEYS & set(off)
        for key in off:
            assert off[key] == be9[key], key

    def test_no_spectrum_without_gammas(self):
        results = Transport.calculate(
            {"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0, "calculate_gammas": False})
        assert not SPECTRUM_KEYS & set(results)

    def test_repeat_runs_are_identical(self, be9):
        repeat = Transport.calculate(
            {"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0, "density": 1.85})
        assert repeat["gamma_spectrum_absolute"] == be9["gamma_spectrum_absolute"]

    def test_config_is_not_modified_and_can_be_reused(self):
        bins = [0.0, 6.0, 601]
        config = {"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0, "gamma_energy_bins": bins}
        first = Transport.calculate(config)
        assert config["gamma_energy_bins"] is bins and bins == [0.0, 6.0, 601]
        assert len(first["gamma_energy_bins"]) == 601
        assert Transport.calculate(config)["gamma_spectrum"] == first["gamma_spectrum"]

    def test_custom_grid(self):
        edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        results = Transport.calculate({
            "calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0, "gamma_energy_bins": edges})
        assert results["gamma_energy_bins"] == edges
        assert sum(results["gamma_spectrum_absolute"]) == pytest.approx(results["gamma_yield"], rel=1e-9)
        # Most of the yield is the 4.44 MeV line, whose width is well below the bin width.
        assert results["gamma_spectrum"][4] > 0.8

    def test_intensity_outside_the_grid_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger=gd.logger.name):
            results = Transport.calculate({
                "calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0,
                "gamma_energy_bins": [0.0, 3.0, 301]})
        assert "outside gamma_energy_bins" in caplog.text
        assert sum(results["gamma_spectrum_absolute"]) < results["gamma_yield"]

    def test_no_lines_gives_an_empty_spectrum(self):
        results = Transport.calculate({"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 1.0})
        assert results["gamma_yield"] == 0.0
        assert sum(results["gamma_spectrum_absolute"]) == 0.0
        assert sum(results["gamma_spectrum"]) == 0.0

    def test_material_without_gamma_data(self):
        results = Transport.calculate({"calc_type": "beam", "matdef": {"Bi-209": 1.0}, "beam_energy": 8.0})
        assert sum(results["gamma_spectrum_absolute"]) == 0.0

    def test_zero_mass_fraction_target(self):
        results = Transport.calculate(
            {"calc_type": "beam", "matdef": {"Be-9": 1.0, "C-13": 0.0}, "beam_energy": 5.0})
        assert sum(results["gamma_spectrum_absolute"]) == pytest.approx(results["gamma_yield"], rel=1e-9)

    def test_photon_product_continuum_is_binned_smoothly(self):
        # Mg-25 goes through the TENDL F02 channel. Its photon continuum is tabulated on a
        # coarse grid, which must not appear as a comb of narrow lines in the 1 keV spectrum.
        results = Transport.calculate(
            {"calc_type": "beam", "matdef": {"Mg-25": 1.0}, "beam_energy": 8.0, "density": 1.74})
        spectrum = np.array(results["gamma_spectrum_absolute"])
        assert spectrum.sum() == pytest.approx(results["gamma_yield"], rel=1e-3)
        assert spectrum.sum() <= results["gamma_yield"]
        continuum = spectrum[8000:10000]
        assert np.count_nonzero(continuum) == continuum.size


@requires_gamma_data
class TestLineShapes:

    def test_am_be_4p44_mev_width(self):
        # Janout, Pospisil and Vobecky, J. Radioanal. Chem. 56, 71 (1980) measured
        # FWHM = 90 +/- 4 keV for 9Be(a,n g)12C in an Am-Be source. This model gives
        # about 110 keV: it assumes isotropic neutron emission and alphas that reach
        # the Be at full energy, whereas in a powder compact they first lose energy
        # in the actinide oxide grains.
        results = Transport.calculate(
            {"calc_type": "homogeneous", "matdef": {"Am-241": 0.02, "Be-9": 0.98}, "density": 1.9})
        assert 85.0 < _line_fwhm_kev(results, 4.438) < 125.0

    def test_short_lived_line_barely_depends_on_density(self):
        # C-12 4.44 MeV has T1/2 = 42 fs, much shorter than the slowing time.
        widths = [
            _line_fwhm_kev(Transport.calculate(
                {"calc_type": "beam", "matdef": {"Be-9": 1.0}, "beam_energy": 5.0, "density": density}), 4.438)
            for density in (1.0, 2.0)]
        assert widths[0] > 80.0
        assert widths[1] == pytest.approx(widths[0], rel=0.1)

    def test_long_lived_line_stays_sharp(self, carbon13):
        # O-16 6.13 MeV: T1/2 = 18 ps.
        assert _fraction_near(carbon13, 6.1286) > 0.9

    @pytest.mark.parametrize("energy", [6.9155, 7.1152])
    def test_short_lived_lines_from_the_same_nucleus_are_broad(self, carbon13, energy):
        # O-16 6.92 and 7.12 MeV: T1/2 = 4.7 and 8.3 fs.
        lines = np.array(carbon13["gamma_lines"])
        nearest = lines[np.argmin(np.abs(lines[:, 0] - energy)), 0]
        assert abs(nearest - energy) < 2e-3
        assert _fraction_near(carbon13, nearest) < 0.25

    def test_comparable_lifetime_responds_to_density(self, puo2_by_density):
        # Ne-20 1634 keV: T1/2 = 0.73 ps, comparable to the slowing time.
        assert _fraction_near(puo2_by_density[11.46], 1.6336) > (
            1.5 * _fraction_near(puo2_by_density[2.3], 1.6336))

    def test_short_lived_line_in_oxide_ignores_density(self, puo2_by_density):
        # Ne-21 1396 keV: T1/2 = 52 fs.
        assert _fraction_near(puo2_by_density[11.46], 1.396) == pytest.approx(
            _fraction_near(puo2_by_density[2.3], 1.396), abs=0.05)

    def test_long_lived_line_in_oxide_is_sharp(self, puo2_by_density):
        # Ne-21 351 keV: T1/2 = 7.1 ps.
        assert _fraction_near(puo2_by_density[11.46], 0.35072) > 0.95


@requires_gamma_data
@pytest.mark.integration
class TestOtherCalculationTypes:

    def test_homogeneous(self):
        results = Transport.calculate({"calc_type": "homogeneous", "matdef": {"Am-241": 0.02, "Be-9": 0.98}})
        assert SPECTRUM_KEYS <= set(results)
        assert sum(results["gamma_spectrum_absolute"]) == pytest.approx(results["gamma_yield"], rel=1e-9)

    def test_interface_uses_the_target_density(self):
        results = Transport.calculate({
            "calc_type": "interface",
            "source_matdef": {"Am-241": 1.0},
            "source_density": 13.67,
            "target_matdef": {"Be-9": 1.0},
            "target_density": 1.85,
        })
        assert results["gamma_recoil_density"] == 1.85
        assert results["gamma_recoil_density_assumed"] is False
        assert sum(results["gamma_spectrum_absolute"]) == pytest.approx(results["gamma_yield"], rel=1e-9)

    def test_interface_can_be_disabled(self):
        results = Transport.calculate({
            "calc_type": "interface",
            "source_matdef": {"Am-241": 1.0},
            "source_density": 13.67,
            "target_matdef": {"Be-9": 1.0},
            "gamma_doppler": False,
        })
        assert not SPECTRUM_KEYS & set(results)

    def test_sandwich_regions_add_up(self, sandwich):
        total = np.array(sandwich["gamma_spectrum_target"]) + np.sum(sandwich["gamma_spectrum_layers"], axis=0)
        assert total == pytest.approx(np.array(sandwich["gamma_spectrum_absolute"]))
        assert total.sum() == pytest.approx(sandwich["gamma_yield"], rel=1e-6)
        assert sum(sandwich["gamma_spectrum"]) == pytest.approx(1.0)

    def test_sandwich_spectra_are_not_negative(self, sandwich):
        for spectrum in sandwich["gamma_spectrum_layers"]:
            assert min(spectrum) >= 0.0

    def test_sandwich_records_densities(self, sandwich):
        assert sandwich["gamma_recoil_density_layers"] == [2.70]
        assert sandwich["gamma_recoil_density_assumed"] is True

    def test_sandwich_target_density(self):
        results = Transport.calculate({**SANDWICH_CONFIGS[0], "target_density": 1.85})
        assert results["gamma_recoil_density_target"] == 1.85
        assert results["gamma_recoil_density_assumed"] is False

    def test_sandwich_can_be_disabled(self):
        results = Transport.calculate({**SANDWICH_CONFIGS[0], "gamma_doppler": False})
        assert not any(key.startswith(("gamma_spectrum", "gamma_recoil", "gamma_energy")) for key in results)
