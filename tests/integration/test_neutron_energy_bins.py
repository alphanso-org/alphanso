"""Regression tests for custom neutron energy grids through the public API."""

import numpy as np
import pytest

from alphanso.transport import Transport


pytestmark = pytest.mark.integration


@pytest.fixture
def beam_config():
    return {
        "calc_type": "beam",
        "matdef": {"Be-9": 1.0},
        "beam_energy": 3.5,
        "calculate_gammas": False,
    }


@pytest.fixture(scope="module")
def default_result():
    config = {
        "calc_type": "beam",
        "matdef": {"Be-9": 1.0},
        "beam_energy": 3.5,
        "calculate_gammas": False,
    }
    result = Transport.calculate(config)
    assert "neutron_energy_bins" not in config
    assert len(result["an_spectrum"]) == 100
    assert sum(result["an_spectrum"]) == pytest.approx(1.0)
    return result


def assert_grid_and_spectrum(result, reference, edges, reference_edge_indices):
    """Compare custom bins with sums of the corresponding default bins."""
    for key in ("neutron_energy_bins", "spectrum_energy_bins"):
        np.testing.assert_allclose(result[key], edges, rtol=1e-12, atol=1e-14)
    assert result["an_yield"] == pytest.approx(reference["an_yield"], rel=1e-12)
    for key in ("an_spectrum", "an_spectrum_absolute"):
        expected = [
            sum(reference[key][start:stop])
            for start, stop in zip(reference_edge_indices[:-1], reference_edge_indices[1:])
        ]
        np.testing.assert_allclose(result[key], expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "bins", [None, [0, 15, 101], [15, 0, 101]],
    ids=["none", "ascending-shorthand", "descending-shorthand"],
)
def test_default_grid_equivalents(beam_config, default_result, bins):
    """Omitted, None, and either shorthand order produce the same spectrum."""
    original = bins.copy() if bins is not None else None
    beam_config["neutron_energy_bins"] = bins

    result = Transport.calculate(beam_config)

    assert_grid_and_spectrum(result, default_result, np.linspace(0, 15, 101), range(101))
    assert beam_config["neutron_energy_bins"] is bins
    assert bins == original


@pytest.mark.parametrize(
    "bins",
    [[0, 15, 3], [15, 0, 3], np.array([0, 7.5, 15]), np.array([15, 7.5, 0])],
    ids=["ascending-shorthand", "descending-shorthand", "ascending-array", "descending-array"],
)
def test_three_edge_grids_are_repeatable(beam_config, default_result, bins):
    """Three-item lists expand once; three-item arrays retain their explicit edges."""
    original = bins.copy()
    beam_config["neutron_energy_bins"] = bins

    for _ in range(2):
        result = Transport.calculate(beam_config)

        assert_grid_and_spectrum(result, default_result, [0, 7.5, 15], [0, 50, 100])
        assert beam_config["neutron_energy_bins"] is bins
        np.testing.assert_array_equal(bins, original)


@pytest.mark.parametrize(
    "bins", [[0, 1.5, 4.5, 15], [15, 4.5, 1.5, 0]],
    ids=["ascending", "descending"],
)
def test_nonuniform_explicit_lists(beam_config, default_result, bins):
    """Lists of other lengths preserve explicit edges and align the output spectrum."""
    original = bins.copy()
    beam_config["neutron_energy_bins"] = bins

    result = Transport.calculate(beam_config)

    assert_grid_and_spectrum(result, default_result, [0, 1.5, 4.5, 15], [0, 10, 30, 100])
    assert beam_config["neutron_energy_bins"] is bins
    assert bins == original
