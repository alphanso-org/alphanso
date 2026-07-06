"""
Tests for the alpha-induced gamma production module.

Covers the level-scheme cascade mathematics on synthetic data, the parsed
channel structure of packaged nuclear data, and end-to-end gamma output of
the transport calculations (values validated against SaG4n/Geant4; see the
gamma source-term validation report).
"""

import os

import pytest

from alphanso import gamma_production as gp
from alphanso.data_manager import get_data_dir
from alphanso.transport import Transport

GAMMA_ROOT = os.path.join(str(get_data_dir()), "gamma")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(GAMMA_ROOT),
    reason="gamma data not installed",
)


def _scheme(rows):
    """Build a LevelScheme from (E_level_keV, E_gamma_keV, weight) rows."""
    import tempfile
    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False) as f:
        for row in rows:
            f.write(" %f %f %f\n" % row)
        path = f.name
    try:
        return gp.LevelScheme.from_file(path)
    finally:
        os.unlink(path)


class TestLevelScheme:

    def test_single_level_single_gamma(self):
        s = _scheme([(1000.0, 999.9, 1.0)])
        lines = s.expected_lines(0)
        assert lines == {0.9999: 1.0}

    def test_cascade_via_next_linking(self):
        # 2 MeV level decays to the 1 MeV level (gamma 1.0 MeV), which decays
        # to ground (gamma 0.9999 MeV): expect both lines with intensity 1.
        s = _scheme([
            (1000.0, 999.9, 1.0),
            (2000.0, 1000.1, 1.0),
        ])
        lines = {round(e, 4): i for e, i in s.expected_lines(1).items()}
        assert lines[1.0001] == pytest.approx(1.0)
        assert lines[0.9999] == pytest.approx(1.0)

    def test_branching_probabilities(self):
        # 2 MeV level: 75% direct to ground, 25% via the 1 MeV level.
        s = _scheme([
            (1000.0, 1000.0, 1.0),
            (2000.0, 1000.0, 25.0),
            (2000.0, 2000.0, 75.0),
        ])
        lines = {round(e, 4): i for e, i in s.expected_lines(1).items()}
        assert lines[2.0] == pytest.approx(0.75)
        assert lines[1.0] == pytest.approx(0.25 * 2)

    def test_zero_weight_level_uses_last_gamma(self):
        # All weights zero: Geant4's cumulative sampling picks the last row.
        s = _scheme([
            (2000.0, 1500.0, 0.0),
            (2000.0, 2000.0, 0.0),
        ])
        lines = s.expected_lines(0)
        assert lines == {2.0: 1.0}

    def test_find_level_tolerances(self):
        s = _scheme([(1000.0, 1000.0, 1.0), (2000.0, 2000.0, 1.0)])
        assert s.find_level(1.0) == 0            # exact
        assert s.find_level(1.015) == 0          # within 20 keV
        assert s.find_level(1.05) is None        # outside 20 keV
        assert s.highest_level_at_or_below(1.5) == 0
        assert s.highest_level_at_or_below(2.5) == 1
        assert s.highest_level_at_or_below(0.5) is None

    def test_walk_spends_budget_down_the_scheme(self):
        s = _scheme([(1000.0, 1000.0, 1.0), (3000.0, 3000.0, 1.0)])
        # 4.5 MeV budget: 3 MeV level, then 1 MeV level, then stop.
        lines = {round(e, 4): i for e, i in gp._walk_lines(s, 4.5).items()}
        assert lines[3.0] == pytest.approx(1.0)
        assert lines[1.0] == pytest.approx(1.0)
        assert gp._walk_lines(s, 0.5) == {}


class TestPackagedChannelData:

    def test_be9_channels(self):
        t = gp.get_gamma_target_data(4009, GAMMA_ROOT)
        assert t is not None
        by_label = {ch.label: ch for ch in t.channels}
        # Discrete (alpha,n1) at the C-12 4.4389 MeV level
        assert by_label["MT51"].e_excitation == pytest.approx(4.439, abs=1e-3)
        assert 4.438 in [round(e, 3) for e in by_label["MT51"].lines]
        # Hoyle-state channel cascades 3.215 + 4.438
        mt52 = by_label["MT52"].lines
        assert {round(e, 3) for e in mt52} == {3.215, 4.438}
        # Continuum channel emits the fixed 9.641 MeV level line
        assert [round(e, 3) for e in by_label["MT91"].lines] == [9.637]

    def test_b10_mt54_uses_highest_level_below(self):
        # MT54 (E_x = 6.364 MeV) has no matching N-13 level within 20 keV;
        # the ParticleHP no-distribution rule emits from the highest level
        # below, 3.547 MeV.
        t = gp.get_gamma_target_data(5010, GAMMA_ROOT)
        by_label = {ch.label: ch for ch in t.channels}
        assert [round(e, 3) for e in by_label["MT54"].lines] == [3.547]
        # MT91 (MF6 rule, no level within 20 keV) must be absent.
        assert "MT91" not in by_label

    def test_li6_has_no_line_channels(self):
        # B-9 is particle-unbound with no level scheme: only the
        # multi-particle walk channels exist and produce no lines at
        # these energies.
        t = gp.get_gamma_target_data(3006, GAMMA_ROOT)
        assert t is not None
        assert all(ch.kind == gp.GammaChannel.KIND_WALK for ch in t.channels)

    def test_unknown_target_returns_none(self):
        assert gp.get_gamma_target_data(83209, GAMMA_ROOT) is None


class TestTransportGammaOutput:

    def test_be9_10mev_beam(self):
        # Validated against SaG4n/Geant4 (JENDLTENDL01): physical gamma
        # yield 1.90e-4 per alpha; alphanso reproduces within a few percent.
        r = Transport.beam_problem([[10.0, 1.0]], {4009: 1.0})
        assert 1.6e-4 < r["gamma_yield"] < 2.2e-4
        lines = dict(r["gamma_lines"])
        top = max(lines, key=lines.get)
        assert round(top, 3) == 4.438

    def test_gammas_can_be_disabled(self):
        r = Transport.beam_problem([[10.0, 1.0]], {4009: 1.0},
                                   calculate_gammas=False)
        assert "gamma_yield" not in r

    def test_below_threshold_no_gammas(self):
        r = Transport.beam_problem([[1.0, 1.0]], {4009: 1.0})
        assert r["gamma_yield"] == 0.0
        assert r["gamma_lines"] == []

    def test_gamma_yield_equals_line_sum(self):
        r = Transport.beam_problem([[10.0, 1.0]], {8018: 1.0})
        line_sum = sum(i for _, i in r["gamma_lines"])
        assert r["gamma_yield"] == pytest.approx(line_sum, rel=1e-12)

    def test_material_without_gamma_data(self):
        # Bi-209 has no gamma channel data; the calculation must still run
        # and report zero gamma yield.
        r = Transport.beam_problem([[8.0, 1.0]], {83209: 1.0})
        assert r["gamma_yield"] == 0.0
