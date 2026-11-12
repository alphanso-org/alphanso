"""Unit tests for alphanso.utils helpers."""

import pytest

from alphanso.utils import matdef_to_zaids
from alphanso.atomic_data_loader import (
    get_atomic_mass,
    get_natural_abundance,
    get_natural_isotopes,
)

O_ISOTOPES = get_natural_isotopes(8)


def test_natural_element_atom_fractions_match_abundances():
    """Expanding a natural element must recover its true isotopic atom fractions.

    A mass-fraction input of a natural element is split among its isotopes by
    mass; dividing those mass fractions back out by isotope mass must return the
    tabulated atom abundances. Splitting by atom abundance instead (the previous
    behavior) biases the atom fractions toward the lighter isotopes.
    """
    _, atom_fractions = matdef_to_zaids({"O": 1.0})

    for iso in O_ISOTOPES:
        assert atom_fractions[iso] == pytest.approx(get_natural_abundance(iso))


def test_natural_oxygen_matches_individual_isotopics():
    """Natty-O and the equivalent per-isotope mass split give identical atoms.

    Calling an element by name must be equivalent to listing its isotopes with
    the mass fractions the element itself implies (the natty-O vs individual
    isotopics check).
    """
    weights = {
        iso: get_natural_abundance(iso) * get_atomic_mass(iso) for iso in O_ISOTOPES
    }
    total = sum(weights.values())
    isotopic_matdef = {f"O-{iso % 1000}": weights[iso] / total for iso in O_ISOTOPES}

    _, natural_af = matdef_to_zaids({"O": 1.0})
    _, isotopic_af = matdef_to_zaids(isotopic_matdef)

    assert natural_af.keys() == isotopic_af.keys()
    for iso in O_ISOTOPES:
        assert natural_af[iso] == pytest.approx(isotopic_af[iso])


def test_natural_element_mass_fractions_are_mass_weighted():
    """Isotope mass fractions are abundance*mass, normalized, and sum to one."""
    mass_fractions, _ = matdef_to_zaids({"O": 1.0})

    expected = {
        iso: get_natural_abundance(iso) * get_atomic_mass(iso) for iso in O_ISOTOPES
    }
    norm = sum(expected.values())

    assert sum(mass_fractions.values()) == pytest.approx(1.0)
    for iso in O_ISOTOPES:
        assert mass_fractions[iso] == pytest.approx(expected[iso] / norm)

    assert mass_fractions[8018] > get_natural_abundance(8018)
    assert mass_fractions[8016] < get_natural_abundance(8016)


def test_single_isotope_input_is_unchanged():
    """An explicit isotope is passed through without isotopic expansion."""
    mass_fractions, atom_fractions = matdef_to_zaids({"O-16": 1.0})

    assert mass_fractions == {8016: 1.0}
    assert atom_fractions == {8016: 1.0}
