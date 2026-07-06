"""
Alpha-induced gamma-ray production (source term).

This module computes the prompt gamma-ray source term produced while alpha
particles slow down in a material, using the same evaluated-data channels
that drive the (alpha,n) neutron calculation.  It is a deterministic
re-implementation of the final-state sampling performed by Geant4
ParticleHP (as used by SaG4n) on the JENDL/AN-2005 based ``JENDLTENDL01``
charged-particle library, with explicit energy-conservation guards.

Physics model
-------------
For a target nuclide the library provides a set of exit channels under
``gamma/Alpha/Inelastic``:

``F01``  (alpha,n) final states, one section per ENDF MT:
    - MT=50..90: discrete two-body channels leaving the residual nucleus in
      level ``i`` with excitation ``E_x = QI(MT50) - QI(MT)``.  The residual
      de-excites through the gamma cascade of its level scheme
      (``gamma/Gammas/z{Z}.a{A}``, G4NDL format).
    - MT=91: lumped continuum.  ParticleHP assigns the *fixed* excitation
      ``E_x = QI(MT50) - QI(MT91)``; the same rule is used here, but the
      cascade is only emitted where that excitation is kinematically
      affordable (``E_cm + QI(MT50) >= E_x``).  SaG4n omits this guard,
      which produces energy-non-conserving gammas (e.g. Li-6).

``F04``/``F06``/``F10``/... multi-particle channels ((alpha,2n), (alpha,n+alpha),
    (alpha,n+p), ...): the mean residual excitation is obtained from energy
    balance, ``E_x = E_cm + Q - <E_ejectiles>``, with the channel Q-value
    taken from the data file and the mean ejectile energy from the MF6
    energy distributions.  The excitation is spent walking down the residual
    level scheme, emitting the cascade of the highest reachable level at
    each pass (the ParticleHP "BaseFS" algorithm).  Note: Geant4 computes
    this energy budget from a neutron-projectile binding-energy bookkeeping
    that adds the alpha projectile's binding energy (+28.3 MeV) for
    (alpha,np)/(alpha,2n) channels; that defect is corrected here by using
    the evaluated Q-value, which correctly suppresses these channels near
    threshold.

``F02`` photon-production files: targets evaluated with explicit photon data
    (JENDL: F-19, Na-23, Al-27, Si-28/29/30) carry MF12/13 photon
    multiplicities with MF14/15 line and continuum spectra; TENDL-derived
    ``.z`` files carry an inclusive MF6 representation with an explicit
    photon product.  Both are integrated as additional gamma channels.

Cascade expectation values
--------------------------
Geant4 samples one gamma per level according to the level-scheme branching
probabilities and follows the chain to the ground state.  Here the *expected*
line intensities per de-excitation are computed once per level by recursion,
so a single thick-target integral per channel yields the deterministic line
spectrum.  Level matching uses ParticleHP's tolerances (exact within 1 keV,
otherwise nearest within 20 keV, otherwise no gamma emission), and levels
whose branching weights are all zero emit their last-listed gamma (Geant4's
sampling behaviour for such data).

Thick-target integration
------------------------
Every channel contributes ``integral( sigma_ch(E) / S(E) * L_ch(E, E_gamma) dE )``
per source alpha, where ``S`` is the material stopping power and ``L_ch`` the
line vector described above; the integral shares the alpha slowing-down grid
of the (alpha,n) calculation (`Transport._integrate_over_ebins`).
"""

import gzip
import logging
import os
import zlib
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ParticleHP level-matching tolerances (G4SaG4nParticleHPInelasticCompFS::GetLevelFromQI)
LEVEL_MATCH_EXACT_MEV = 1.0e-3     # 1 keV
LEVEL_MATCH_MAX_MEV = 20.0e-3      # 20 keV
# Tolerance of the "nothing known on hadron" level search (CompFS line ~586)
NO_DIST_LEVEL_TOL_MEV = 5.0e-3     # 5 keV
# Tolerance for the kinematic-affordability guard on the MT=91 fixed excitation
AFFORDABILITY_TOL_MEV = 25.0e-3
# Bucket width used to cache multi-particle-channel level walks
WALK_BUDGET_BUCKET_MEV = 0.02

ALPHA_ZA = (2, 4)

# Ejectile content of each ParticleHP inelastic final-state directory
# (from the G4ParticleHPInelastic channel registration).  Keys are directory
# names; values are lists of (Z, A) of the light ejectiles.  F02 is the
# inclusive (n,X) channel; its residual is taken as the physical one for a
# single removed neutron.
_N, _P, _D, _T, _H, _A = (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)
FS_EJECTILES: Dict[str, List[Tuple[int, int]]] = {
    'F01': [_N],
    'F02': [_N],
    'F03': [_N, _N, _D],
    'F04': [_N, _N],
    'F05': [_N, _N, _N],
    'F06': [_N, _A],
    'F07': [_N, _A, _A, _A],
    'F08': [_N, _N, _A],
    'F09': [_N, _N, _N, _A],
    'F10': [_N, _P],
    'F11': [_N, _A, _A],
    'F12': [_N, _N, _A, _A],
    'F13': [_N, _D],
    'F14': [_N, _T],
    'F15': [_N, _H],
    'F16': [_N, _D, _A, _A],
    'F17': [_N, _T, _A, _A],
    'F18': [_N, _N, _N, _N],
    'F19': [_N, _N, _P],
    'F20': [_N, _N, _N, _P],
    'F21': [_N, _P, _P],
    'F22': [_N, _P, _A],
    'F23': [_P],
    'F24': [_D],
    'F25': [_T],
    'F26': [_H],
    'F27': [_A],
    'F28': [_A, _A],
    'F29': [_A, _A, _A],
    'F30': [_P, _P],
    'F31': [_P, _A],
    'F32': [_D, _A, _A],
    'F33': [_T, _A, _A],
    'F34': [_P, _D],
    'F35': [_P, _T],
    'F36': [_D, _A],
}

# Mass code of the photon product in MF6 blocks
_PHOTON_MASS_CODE = 0.0


def _read_tokens(path: str) -> List[str]:
    """Read a ParticleHP data file (optionally gzip/zlib compressed) into tokens."""
    with open(path, 'rb') as f:
        raw = f.read()
    if path.endswith('.z') or raw[:2] in (b'\x1f\x8b', b'\x78\x9c', b'\x78\x01', b'\x78\xda'):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            raw = zlib.decompress(raw)
    return raw.decode('ascii', errors='replace').split()


class _TokenStream:
    """Sequential typed reader over a ParticleHP token list."""

    def __init__(self, tokens: List[str], path: str):
        self._t = tokens
        self._i = 0
        self._path = path

    def eof(self) -> bool:
        return self._i >= len(self._t)

    def integer(self) -> int:
        tok = self._next()
        val = float(tok)
        ival = int(round(val))
        if abs(val - ival) > 1e-9:
            raise ValueError(
                f"Expected integer token, got '{tok}' at position {self._i - 1} in {self._path}")
        return ival

    def real(self) -> float:
        return float(self._next())

    def skip(self, count: int) -> None:
        self._i += count
        if self._i > len(self._t):
            raise ValueError(f"Unexpected end of data in {self._path}")

    def reals(self, count: int) -> np.ndarray:
        out = np.array(self._t[self._i:self._i + count], dtype=float)
        if out.size != count:
            raise ValueError(f"Unexpected end of data in {self._path}")
        self._i += count
        return out

    def _next(self) -> str:
        if self.eof():
            raise ValueError(f"Unexpected end of data in {self._path}")
        tok = self._t[self._i]
        self._i += 1
        return tok


class LevelScheme:
    """
    De-excitation level scheme of one nucleus (G4NDL ``Inelastic/Gammas`` format).

    Each file row is ``E_level(keV)  E_gamma(keV)  weight``; rows are grouped
    into levels, each gamma is linked to the level it feeds (energy matching,
    replicating G4ParticleHPDeExGammas), and expected line intensities per
    de-excitation are evaluated by recursion over the branching weights.
    """

    def __init__(self, level_energies: List[float],
                 gammas_per_level: List[List[Tuple[float, float, Optional[int]]]]):
        # gammas_per_level[i] = [(gamma_energy_MeV, weight, next_level_idx or None), ...]
        self.level_energies = level_energies
        self.gammas_per_level = gammas_per_level
        self._expected_cache: Dict[int, Dict[float, float]] = {}

    @classmethod
    def from_file(cls, path: str) -> "LevelScheme":
        tokens = _read_tokens(path)
        if len(tokens) % 3 != 0:
            raise ValueError(f"Malformed level-scheme file (token count) {path}")
        rows = np.array(tokens, dtype=float).reshape(-1, 3)
        rows[:, 0] *= 1e-3  # keV -> MeV
        rows[:, 1] *= 1e-3

        eps = 1e-8  # 0.01 keV grouping tolerance, as in G4
        level_energies: List[float] = []
        gamma_rows: List[List[Tuple[float, float]]] = []
        current = None
        for elev, egam, weight in rows:
            if current is None or abs(elev - current) > eps:
                level_energies.append(float(elev))
                gamma_rows.append([])
                current = elev
            gamma_rows[-1].append((float(egam), float(weight)))

        # Link each gamma to the level it feeds (G4ParticleHPDeExGammas::Init)
        gammas_per_level: List[List[Tuple[float, float, Optional[int]]]] = []
        for i, gammas in enumerate(gamma_rows):
            linked = []
            elev = level_energies[i]
            for egam, weight in gammas:
                best = elev - egam - eps
                target: Optional[int] = None
                for j, ej in enumerate(level_energies):
                    diff = abs(elev - (ej + egam))
                    if diff < best:
                        best = diff
                        target = j
                if target is not None and level_energies[target] == elev:
                    target = target - 1 if target > 0 else None
                linked.append((egam, weight, target))
            gammas_per_level.append(linked)

        return cls(level_energies, gammas_per_level)

    def find_level(self, e_excitation: float) -> Optional[int]:
        """Level index for an excitation energy (ParticleHP GetLevelFromQI rules)."""
        best_idx = None
        best_diff = None
        for i, elev in enumerate(self.level_energies):
            diff = abs(e_excitation - elev)
            if diff < LEVEL_MATCH_EXACT_MEV:
                return i
            if diff < LEVEL_MATCH_MAX_MEV and (best_diff is None or diff < best_diff):
                best_diff = diff
                best_idx = i
        return best_idx

    def highest_level_at_or_below(self, energy: float) -> Optional[int]:
        """Highest level with E_level < energy (ParticleHP BaseFS walk rule)."""
        idx = None
        for i, elev in enumerate(self.level_energies):
            if elev < energy:
                idx = i
        return idx

    @property
    def lowest_level_energy(self) -> float:
        return self.level_energies[0]

    def expected_lines(self, level_idx: int) -> Dict[float, float]:
        """
        Expected gamma line intensities for one de-excitation of ``level_idx``.

        Returns {gamma_energy_MeV: expected_count_per_deexcitation}.
        """
        if level_idx in self._expected_cache:
            return self._expected_cache[level_idx]
        # Iterative resolution to protect against malformed (cyclic) data.
        self._expected_cache[level_idx] = self._expected_lines_impl(level_idx, frozenset())
        return self._expected_cache[level_idx]

    def _expected_lines_impl(self, level_idx: int, visiting: frozenset) -> Dict[float, float]:
        if level_idx in visiting:
            logger.warning("Cyclic gamma cascade detected at level %d; truncating", level_idx)
            return {}
        gammas = self.gammas_per_level[level_idx]
        if not gammas:
            return {}
        weights = np.array([w for _, w, _ in gammas], dtype=float)
        total = weights.sum()
        if total > 0:
            probs = weights / total
        else:
            # G4's cumulative sampling selects the last entry when all
            # weights are zero.
            probs = np.zeros(len(gammas))
            probs[-1] = 1.0
        out: Dict[float, float] = defaultdict(float)
        for (egam, _w, nxt), p in zip(gammas, probs):
            if p <= 0.0:
                continue
            if egam > 0.0:
                out[egam] += p
            if nxt is not None:
                for e_next, i_next in self._expected_lines_impl(
                        nxt, visiting | {level_idx}).items():
                    out[e_next] += p * i_next
        return dict(out)


class _MF6Product:
    """One product block of a ParticleHP MF6 (dataType 6) section."""

    def __init__(self, mass_code: float, mass: float, law: int, q_gs: float, q_as: float,
                 yield_e: np.ndarray, yield_v: np.ndarray,
                 mean_e_in: Optional[np.ndarray], mean_e_out: Optional[np.ndarray],
                 spectra: Optional[List[Tuple[float, np.ndarray, np.ndarray]]]):
        self.mass_code = mass_code
        self.mass = mass
        self.law = law
        self.q_gs = q_gs  # MeV
        self.q_as = q_as  # MeV (QI)
        self.yield_e = yield_e  # MeV
        self.yield_v = yield_v
        self.mean_e_in = mean_e_in    # MeV, incident grid of the energy distribution
        self.mean_e_out = mean_e_out  # MeV, mean outgoing energy at each incident point
        # law-1 spectra: [(E_in_MeV, e_out_MeV[], pdf_per_MeV[]), ...]
        self.spectra = spectra

    def multiplicity(self, e: np.ndarray) -> np.ndarray:
        if self.yield_e.size == 0:
            return np.zeros_like(e)
        return np.interp(e, self.yield_e, self.yield_v,
                         left=0.0, right=float(self.yield_v[-1]))

    def mean_energy(self, e: np.ndarray) -> np.ndarray:
        """Mean outgoing energy (MeV) at incident energies ``e`` (law 1 only)."""
        if self.mean_e_in is None or self.mean_e_in.size == 0:
            return np.zeros_like(e)
        return np.interp(e, self.mean_e_in, self.mean_e_out,
                         left=float(self.mean_e_out[0]), right=float(self.mean_e_out[-1]))


class _FSSection:
    """One (infoType, dataType) section of a ParticleHP final-state file."""

    def __init__(self, data_type: int, mt: int):
        self.data_type = data_type
        self.mt = mt
        self.qi: Optional[float] = None          # MeV
        self.lr: Optional[int] = None
        self.xs_e: Optional[np.ndarray] = None   # MeV
        self.xs_v: Optional[np.ndarray] = None   # barns
        self.target_mass: Optional[float] = None  # neutron-mass units
        self.products: List[_MF6Product] = []


def _parse_hp_vector(ts: _TokenStream) -> Tuple[np.ndarray, np.ndarray]:
    """Read a G4ParticleHPVector: total, interpolation manager, (x, y) pairs."""
    total = ts.integer()
    n_ranges = ts.integer()
    ts.skip(2 * n_ranges)
    flat = ts.reals(2 * total)
    return flat[0::2], flat[1::2]


def _parse_mf6_product(ts: _TokenStream, want_spectra: bool) -> _MF6Product:
    mass_code = ts.real()
    mass = ts.real()
    ts.integer()          # isomer flag
    law = ts.integer()
    q_gs = ts.real() * 1e-6
    q_as = ts.real() * 1e-6
    ye, yv = _parse_hp_vector(ts)
    ye = ye * 1e-6  # eV -> MeV

    mean_e_in = mean_e_out = None
    spectra = None
    if law == 1:
        ts.integer()                 # target code (ZA)
        ts.integer()                 # angular representation
        ts.integer()                 # interpolation scheme
        n_energies = ts.integer()
        n_ranges = ts.integer()
        ts.skip(2 * n_ranges)
        e_in_list, mean_list = [], []
        spectra = [] if want_spectra else None
        for _ in range(n_energies):
            e_in = ts.real() * 1e-6
            n_out = ts.integer()
            n_discrete = ts.integer()
            n_ang = ts.integer()
            block = ts.reals(n_out * (1 + n_ang)).reshape(n_out, 1 + n_ang)
            e_out = block[:, 0] * 1e-6         # eV -> MeV
            f0 = block[:, 1]
            # The first n_discrete rows are discrete emission probabilities;
            # the remainder is a continuum density per eV.
            disc_e = e_out[:n_discrete]
            disc_p = f0[:n_discrete]
            cont_e = e_out[n_discrete:]
            cont_pdf = f0[n_discrete:] * 1e6   # per eV -> per MeV
            cont_norm = np.trapezoid(cont_pdf, cont_e) if cont_e.size > 1 else 0.0
            norm = float(disc_p.sum() + cont_norm)
            if norm > 0:
                mean = float(disc_p @ disc_e)
                if cont_norm > 0:
                    mean += float(np.trapezoid(cont_pdf * cont_e, cont_e))
                mean /= norm
            else:
                mean = 0.0
            e_in_list.append(e_in)
            mean_list.append(mean)
            if want_spectra:
                spectra.append((e_in, disc_e, disc_p, cont_e, cont_pdf))
        mean_e_in = np.array(e_in_list)
        mean_e_out = np.array(mean_list)
    elif law == 2:
        n_energies = ts.integer()
        n_ranges = ts.integer()
        ts.skip(2 * n_ranges)
        for _ in range(n_energies):
            ts.real()                # incident energy
            ts.integer()             # temperature/representation flag
            n_coeff = ts.integer()
            ts.skip(n_coeff)
    else:
        raise ValueError(f"Unsupported MF6 distribution law {law}")

    return _MF6Product(mass_code, mass, law, q_gs, q_as, ye, yv,
                       mean_e_in, mean_e_out, spectra)


FORMAT_COMPFS = 'compfs'
FORMAT_BASEFS = 'basefs'


def parse_fs_file(path: str, fmt: str, want_spectra: bool = False) -> List[_FSSection]:
    """
    Parse a ParticleHP final-state file.

    Two on-disk layouts exist, mirroring the two Geant4 reader classes:
      - ``compfs`` (F01): every section header carries ``infoType dataType
        sfType dummy`` and dataType-3 sections carry their own QI/LR values.
      - ``basefs`` (F02, F04, F06, F10, ...): section headers are
        ``infoType dataType`` and a single ``Q dummy`` pair follows the first
        header; the file describes one reaction channel.

    Returns the list of sections in file order.
    """
    ts = _TokenStream(_read_tokens(path), path)
    sections: List[_FSSection] = []
    file_q: Optional[float] = None
    while not ts.eof():
        ts.integer()                 # infoType
        data_type = ts.integer()
        if fmt == FORMAT_COMPFS:
            mt = ts.integer()
            ts.integer()             # dummy
        else:
            mt = 0
            if file_q is None:
                file_q = ts.real() * 1e-6
                ts.integer()         # dummy
        sec = _FSSection(data_type, mt)
        if data_type == 3:
            if fmt == FORMAT_COMPFS:
                sec.qi = ts.real() * 1e-6
                sec.lr = ts.integer()
            else:
                sec.qi = file_q
            total = ts.integer()
            flat = ts.reals(2 * total)
            sec.xs_e = flat[0::2] * 1e-6
            sec.xs_v = flat[1::2]
        elif data_type == 6:
            sec.target_mass = ts.real()
            ts.integer()             # frame flag
            n_products = ts.integer()
            for _ in range(n_products):
                sec.products.append(_parse_mf6_product(ts, want_spectra))
        else:
            raise ValueError(f"Unsupported dataType {data_type} in {path}")
        sections.append(sec)
    return sections


class GammaChannel:
    """A single gamma-producing reaction channel of one target nuclide."""

    KIND_DISCRETE = 'discrete'
    KIND_CONTINUUM = 'continuum'
    KIND_WALK = 'walk'
    KIND_PHOTON_DATA = 'photon-data'

    def __init__(self, kind: str, label: str, xs_e: np.ndarray, xs_v: np.ndarray):
        self.kind = kind
        self.label = label
        self.xs_e = xs_e            # MeV
        self.xs_v = xs_v            # barns
        # discrete / continuum
        self.lines: Dict[float, float] = {}
        self.e_excitation: Optional[float] = None
        self.q_ground: Optional[float] = None   # QI of the ground-state channel (MeV)
        # walk
        self.scheme: Optional[LevelScheme] = None
        self.q_value: Optional[float] = None    # true channel Q (MeV)
        self.products: List[_MF6Product] = []
        self.walk_cache: Dict[int, Dict[float, float]] = {}
        # photon-data
        self.photon_lines_by_e: Optional[List[Tuple[float, Dict[float, float]]]] = None
        self.photon_multiplicity: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.photon_spectra: Optional[List[Tuple[float, np.ndarray, np.ndarray]]] = None

    def cross_section(self, e: np.ndarray) -> np.ndarray:
        return np.interp(e, self.xs_e, self.xs_v, left=0.0, right=0.0)


class GammaTargetData:
    """All gamma-producing channels for one target ZAID."""

    def __init__(self, zaid: int, target_a: float, channels: List[GammaChannel]):
        self.zaid = zaid
        self.target_a = target_a
        self.channels = channels

    @property
    def cm_factor(self) -> float:
        """Lab -> CM kinetic-energy conversion factor for the incident alpha."""
        return self.target_a / (self.target_a + 4.002602)


def _element_file(directory: str, z: int, a: int) -> Optional[str]:
    """Locate ``{z}_{a}_<Element>`` (or ``.z``) inside a channel directory."""
    if not os.path.isdir(directory):
        return None
    prefix = f"{z}_{a}_"
    plain, compressed = None, None
    for fname in os.listdir(directory):
        if not fname.startswith(prefix):
            continue
        stem = fname[len(prefix):]
        if '_' in stem or stem.endswith('.orig') or stem.endswith('~'):
            continue
        if fname.endswith('.z'):
            compressed = compressed or os.path.join(directory, fname)
        else:
            plain = plain or os.path.join(directory, fname)
    return plain or compressed


@lru_cache(maxsize=None)
def _load_level_scheme(gammas_dir: str, z: int, a: int) -> Optional[LevelScheme]:
    path = os.path.join(gammas_dir, f"z{z}.a{a}")
    if not os.path.exists(path):
        return None
    try:
        return LevelScheme.from_file(path)
    except (ValueError, OSError) as exc:
        logger.warning("Failed to parse level scheme %s: %s", path, exc)
        return None


def _build_f01_channels(path: str, z: int, a: int,
                        gammas_dir: str) -> List[GammaChannel]:
    """
    Channels from an F01 file: discrete (alpha,n_i) levels and MT=91 continuum.

    The residual level is selected with the rule ParticleHP applies to the
    section: channels carrying their own MF6 block use GetLevelFromQI
    (nearest level within 20 keV, otherwise no gammas); channels without
    distribution data take the "nothing known on hadron" branch, which picks
    the highest level below ``E_x + 5 keV`` and never rejects.
    """
    sections = parse_fs_file(path, FORMAT_COMPFS)
    xs_by_mt = {s.mt: s for s in sections if s.data_type == 3}
    mf6_mts = {s.mt for s in sections if s.data_type == 6}
    if 50 not in xs_by_mt:
        logger.debug("F01 file %s has no MT=50 section; skipping", path)
        return []
    qi0 = xs_by_mt[50].qi
    res_z, res_a = z + 2, a + 3
    scheme = _load_level_scheme(gammas_dir, res_z, res_a)
    if scheme is None:
        logger.debug("No level scheme for residual z%d.a%d; no (alpha,n) "
                     "discrete gammas for %s", res_z, res_a, path)
        return []

    channels = []
    for mt, sec in sorted(xs_by_mt.items()):
        if mt == 50 or not (50 < mt <= 91):
            continue
        e_exc = qi0 - sec.qi
        if e_exc <= 0:
            continue
        kind = GammaChannel.KIND_CONTINUUM if mt == 91 else GammaChannel.KIND_DISCRETE
        if mt in mf6_mts:
            level = scheme.find_level(e_exc)
        else:
            level = scheme.highest_level_at_or_below(
                e_exc + NO_DIST_LEVEL_TOL_MEV)
        if level is None:
            logger.debug("MT=%d of %s: excitation %.4f MeV matches no level of "
                         "z%d.a%d; channel emits no gammas",
                         mt, os.path.basename(path), e_exc, res_z, res_a)
            continue
        lines = scheme.expected_lines(level)
        if not lines:
            continue
        ch = GammaChannel(kind, f"MT{mt}", sec.xs_e, sec.xs_v)
        ch.lines = lines
        ch.e_excitation = e_exc
        ch.q_ground = qi0
        channels.append(ch)
    return channels


def _build_multiparticle_channels(path: str, fs_dir: str, z: int, a: int,
                                  gammas_dir: str) -> List[GammaChannel]:
    """
    Channels from a multi-particle final-state file (F02, F04, F06, F10, ...).

    Produces a level-walk channel for the residual de-excitation and, when the
    MF6 block carries an explicit photon product (TENDL-derived files), a
    photon-data channel integrating that product directly.
    """
    ejectiles = FS_EJECTILES.get(fs_dir)
    if ejectiles is None:
        logger.debug("Unknown final-state directory %s; skipping %s", fs_dir, path)
        return []
    res_z = z + ALPHA_ZA[0] - sum(zz for zz, _ in ejectiles)
    res_a = a + ALPHA_ZA[1] - sum(aa for _, aa in ejectiles)
    if res_z <= 0 or res_a <= 0 or res_a < res_z:
        return []

    sections = parse_fs_file(path, FORMAT_BASEFS, want_spectra=True)
    xs = next((s for s in sections if s.data_type == 3), None)
    mf6 = next((s for s in sections if s.data_type == 6), None)
    if xs is None or xs.xs_e is None:
        return []
    products = mf6.products if mf6 is not None else []

    channels: List[GammaChannel] = []
    photon = next((p for p in products
                   if p.mass_code == _PHOTON_MASS_CODE and p.mass == 0.0), None)
    if photon is not None and photon.spectra is not None:
        ch = GammaChannel(GammaChannel.KIND_PHOTON_DATA, f"{fs_dir}-photon",
                          xs.xs_e, xs.xs_v)
        ch.photon_multiplicity = (photon.yield_e, photon.yield_v)
        ch.photon_spectra = photon.spectra
        channels.append(ch)

    scheme = _load_level_scheme(gammas_dir, res_z, res_a)
    if scheme is not None:
        ch = GammaChannel(GammaChannel.KIND_WALK, fs_dir, xs.xs_e, xs.xs_v)
        ch.scheme = scheme
        ch.q_value = xs.qi
        ch.products = products
        channels.append(ch)
    return channels


def _walk_lines(scheme: LevelScheme, budget: float) -> Dict[float, float]:
    """Expected lines from the BaseFS level walk for one excitation budget."""
    out: Dict[float, float] = defaultdict(float)
    guard = 0
    while budget >= scheme.lowest_level_energy and guard < 100:
        guard += 1
        idx = scheme.highest_level_at_or_below(budget)
        if idx is None:
            break
        for e, i in scheme.expected_lines(idx).items():
            out[e] += i
        budget -= scheme.level_energies[idx]
    return dict(out)


def get_gamma_target_data(zaid: int, gamma_data_root: str) -> Optional[GammaTargetData]:
    """
    Load all gamma-production channels for ``zaid`` from a gamma data root
    (a directory containing ``Alpha/Inelastic/...`` and ``Gammas/``).

    Returns None when no data exists for the target.
    """
    return _get_gamma_target_data_cached(int(zaid), os.path.abspath(gamma_data_root))


@lru_cache(maxsize=None)
def _get_gamma_target_data_cached(zaid: int, root: str) -> Optional[GammaTargetData]:
    z, a = zaid // 1000, zaid % 1000
    inelastic = os.path.join(root, 'Alpha', 'Inelastic')
    gammas_dir = os.path.join(root, 'Gammas')
    if not os.path.isdir(inelastic):
        logger.warning("Gamma data directory %s not found; gamma production "
                       "disabled for ZAID %d", inelastic, zaid)
        return None

    channels: List[GammaChannel] = []
    f01 = _element_file(os.path.join(inelastic, 'F01'), z, a)
    if f01:
        try:
            channels.extend(_build_f01_channels(f01, z, a, gammas_dir))
        except ValueError as exc:
            logger.warning("Failed to parse %s: %s", f01, exc)

    for fs_dir in sorted(FS_EJECTILES):
        if fs_dir == 'F01':
            continue
        path = _element_file(os.path.join(inelastic, fs_dir), z, a)
        if not path:
            continue
        try:
            channels.extend(_build_multiparticle_channels(path, fs_dir, z, a, gammas_dir))
        except ValueError as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
            continue

    if not channels:
        return None
    return GammaTargetData(zaid, float(a), channels)


def compute_gamma_production(e_steps: np.ndarray, de: np.ndarray,
                             stopping: np.ndarray,
                             target: GammaTargetData) -> Tuple[float, Dict[float, float]]:
    """
    Thick-target gamma production for one slowing-down grid.

    Args:
        e_steps: alpha energies at the integration midpoints (MeV, descending)
        de: energy-step widths (MeV)
        stopping: stopping power at the midpoints (MeV cm^2 / atom)
        target: channel data from :func:`get_gamma_target_data`

    Returns:
        (gamma_yield, lines): total gammas per source alpha and the discrete
        line dictionary {gamma_energy_MeV: intensity_per_alpha}.
    """
    lines: Dict[float, float] = defaultdict(float)
    ok = stopping > 1e-30
    if not np.any(ok):
        return 0.0, {}
    e_ok = e_steps[ok]
    path_weight = de[ok] / stopping[ok]     # atoms/cm^2 traversed in each step
    cm = target.cm_factor

    for ch in target.channels:
        sigma = ch.cross_section(e_ok) * 1e-24    # barns -> cm^2
        pop = sigma * path_weight                 # reactions per source alpha
        active = pop > 0.0
        if not np.any(active):
            continue

        if ch.kind == GammaChannel.KIND_DISCRETE:
            total = float(pop[active].sum())
            for e_g, i_g in ch.lines.items():
                lines[e_g] += total * i_g

        elif ch.kind == GammaChannel.KIND_CONTINUUM:
            # ParticleHP fixed-excitation rule with an affordability guard
            affordable = (e_ok * cm + ch.q_ground + AFFORDABILITY_TOL_MEV
                          >= ch.e_excitation)
            total = float(pop[active & affordable].sum())
            if total > 0:
                for e_g, i_g in ch.lines.items():
                    lines[e_g] += total * i_g

        elif ch.kind == GammaChannel.KIND_WALK:
            scheme = ch.scheme
            e_act = e_ok[active]
            budget = e_act * cm + ch.q_value
            for prod in ch.products:
                budget = budget - prod.multiplicity(e_act) * prod.mean_energy(e_act)
            pop_act = pop[active]
            feasible = budget >= scheme.lowest_level_energy
            if not np.any(feasible):
                continue
            buckets = np.round(budget[feasible] / WALK_BUDGET_BUCKET_MEV).astype(int)
            pops = pop_act[feasible]
            for bucket in np.unique(buckets):
                sel = buckets == bucket
                w = float(pops[sel].sum())
                if bucket not in ch.walk_cache:
                    ch.walk_cache[bucket] = _walk_lines(
                        scheme, bucket * WALK_BUDGET_BUCKET_MEV)
                for e_g, i_g in ch.walk_cache[bucket].items():
                    lines[e_g] += w * i_g

        elif ch.kind == GammaChannel.KIND_PHOTON_DATA:
            _accumulate_photon_data_channel(ch, e_ok, pop, lines)

    total_yield = float(sum(lines.values()))
    return total_yield, dict(lines)


# ---------------------------------------------------------------------------
# Explicit photon-product (TENDL F02) support
# ---------------------------------------------------------------------------

def _accumulate_photon_data_channel(ch: GammaChannel, e_ok: np.ndarray,
                                    pop: np.ndarray,
                                    lines: Dict[float, float]) -> None:
    """
    Integrate an explicit MF6 photon product: reactions x multiplicity, with
    each tabulated spectrum (discrete lines plus continuum) applied to the
    reactions occurring in its incident-energy panel.  Continuum densities
    are collapsed onto their own grid points with trapezoid weights.
    """
    mult = np.interp(e_ok, ch.photon_multiplicity[0], ch.photon_multiplicity[1],
                     left=0.0, right=float(ch.photon_multiplicity[1][-1]))
    photons = pop * mult
    if photons.sum() <= 0:
        return
    e_grid = np.array([s[0] for s in ch.photon_spectra])
    idx = np.clip(np.searchsorted(e_grid, e_ok, side='right') - 1,
                  0, len(e_grid) - 1)
    for panel in np.unique(idx):
        w = float(photons[idx == panel].sum())
        if w <= 0:
            continue
        _, disc_e, disc_p, cont_e, cont_pdf = ch.photon_spectra[panel]
        cont_norm = float(np.trapezoid(cont_pdf, cont_e)) if cont_e.size > 1 else 0.0
        norm = float(disc_p.sum()) + cont_norm
        if norm <= 0:
            continue
        for e_g, p_g in zip(disc_e, disc_p):
            if p_g > 0 and e_g > 0:
                lines[round(float(e_g), 6)] += w * p_g / norm
        if cont_norm > 0:
            weights = np.zeros_like(cont_e)
            d = np.diff(cont_e)
            weights[:-1] += 0.5 * d * cont_pdf[:-1]
            weights[1:] += 0.5 * d * cont_pdf[1:]
            weights /= norm
            for e_g, frac in zip(cont_e, weights):
                if frac > 0 and e_g > 0:
                    lines[round(float(e_g), 6)] += w * frac


def clear_caches() -> None:
    """Drop all cached parsed data (for tests)."""
    _get_gamma_target_data_cached.cache_clear()
    _load_level_scheme.cache_clear()
