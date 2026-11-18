"""
Doppler broadening of the alpha-induced gamma lines.

Every reaction leaves the residual nucleus recoiling, so a gamma emitted before
the recoil stops is Doppler shifted.  Turning the rest-frame lines of
``gamma_production`` into a binned spectrum takes three ingredients:

Recoil speed
    Reactions are tallied by channel and alpha energy.  Neutron emission is
    isotropic in the centre of mass, as in the neutron spectrum calculation, so
    the squared lab speed of the recoil is uniform between (b_cm - b')^2 and
    (b_cm + b')^2.  Two-body kinematics give b' for the F01 channels; for
    multi-particle channels it is estimated from the mean ejectile energies,
    assuming uncorrelated ejectiles.

Emission time
    Each level decays with its RIPL-3 lifetime, and delays add along a cascade.
    Levels without a tabulated lifetime, and explicit TENDL photon lines, which
    have no level identity, are treated as emitting immediately.

Recoil slowing
    Lindhard-Scharff electronic stopping (held constant above the velocity
    v0 * Z^(2/3), where it stops growing) plus ZBL universal nuclear stopping,
    combined over the material by Bragg additivity.  Only the speed matters, so
    deflections by nuclear collisions are irrelevant.

Isotropic emission from a nucleus at speed b fills the energy interval
E0 * gamma * [1 - b, 1 + b] uniformly (cos(theta) is uniform in the nucleus
frame), so the 4-pi spectrum depends on the speed alone.  A line is therefore a
sum of such boxes over the speed distribution at emission, plus a spike at E0
from nuclei that have already stopped.  The total intensity of each line is
taken from the rest-frame calculation, so broadening only redistributes it.

The photon continuum of the TENDL channels is already continuous and is binned
directly, without broadening.
"""

import logging
from collections import defaultdict
from functools import lru_cache
from typing import NamedTuple, Optional, Tuple

import numpy as np

from .atomic_data_loader import atomic_data
from .constants import ALPH_MASS, AMU_TO_MEV, ANEUT_MASS, AVOGADRO_NUM
from .gamma_production import (
    LEVEL_MATCH_MAX_MEV,
    GammaChannel,
    LevelScheme,
    photon_panel_emission,
    walk_budget,
    walk_levels,
)
from .parsers import get_level_half_lives

logger = logging.getLogger(__name__)

# Linear [start, stop, num_points] grid: 1 keV bins over the range of the neutron grid
DEFAULT_GAMMA_ENERGY_BINS = [0.0, 15.0, 15001]

# Assumed when no density is given.  Atom density varies far less between solids
# than mass density does: about 5e22 (U, Pu metal) to 1.4e23 atoms/cm^3 (BeO).
DEFAULT_ATOM_DENSITY = 8.0e22  # atoms/cm^3

ALPHA_GROUP_MEV = 0.05         # width of the alpha-energy groups of the reaction tally
NUM_SPEED_BINS = 200           # bin 0 is treated as "at rest"
OUTSIDE_GRID_WARNING_FRACTION = 0.01

SPEED_OF_LIGHT = 2.99792458e10   # cm/s
BOHR_SPEED_INVERSE = 137.036     # c / v0
LSS_PREFACTOR = 1.915e-14        # 8 pi e^2 a0, eV cm^2

_warned_density = False


class RecoilMedium(NamedTuple):
    """Stopping medium: (Z, atomic mass, atom fraction) per nuclide, and atoms/cm^3."""
    composition: Tuple[Tuple[int, float, float], ...]
    atom_density: float


def parse_gamma_energy_bins(raw_bins=None) -> np.ndarray:
    """
    Convert a ``gamma_energy_bins`` input to ascending bin edges (MeV).

    Follows the ``neutron_energy_bins`` rules: three-element lists are
    ``[start, stop, num_points]`` shorthand, everything else is explicit edges.
    """
    if raw_bins is None:
        raw_bins = DEFAULT_GAMMA_ENERGY_BINS
    if isinstance(raw_bins, list) and len(raw_bins) == 3:
        edges = np.linspace(raw_bins[0], raw_bins[1], int(raw_bins[2]))
    else:
        edges = np.asarray(raw_bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("gamma_energy_bins must contain at least two bin edges")
    if edges[0] > edges[-1]:
        edges = edges[::-1]
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("gamma_energy_bins must be strictly monotonic")
    return edges


def recoil_medium(atom_fractions: dict,
                  density: Optional[float] = None) -> Tuple[RecoilMedium, float, bool]:
    """
    Describe a material for recoil slowing.

    Args:
        atom_fractions: {ZAID: atom fraction} of the material
        density: density of the solid in g/cm^3; DEFAULT_ATOM_DENSITY is assumed when None

    Returns:
        (medium, density in g/cm^3, whether the density was assumed)

    Raises:
        ValueError: If the density is not positive
    """
    global _warned_density
    if density is not None and density <= 0:
        raise ValueError(f"Density must be positive, got {density}")
    composition = tuple(
        (zaid // 1000, float(atomic_data.get_atomic_mass(zaid)), float(fraction))
        for zaid, fraction in atom_fractions.items())
    mean_mass = sum(mass * fraction for _, mass, fraction in composition)
    if mean_mass <= 0.0:
        return RecoilMedium(composition, DEFAULT_ATOM_DENSITY), 0.0, density is None
    assumed = density is None
    if assumed:
        density = DEFAULT_ATOM_DENSITY * mean_mass / AVOGADRO_NUM
        if not _warned_density:
            _warned_density = True
            logger.warning(
                "No density given for gamma Doppler broadening; assuming %.1e atoms/cm^3 "
                "for recoil slowing. Set 'density' or 'target_density' to override.",
                DEFAULT_ATOM_DENSITY)
    return RecoilMedium(composition, density * AVOGADRO_NUM / mean_mass), float(density), assumed


def electronic_stopping(z: int, z2: int, speed: np.ndarray) -> np.ndarray:
    """Lindhard-Scharff electronic stopping of ion Z in target atom Z2 (eV cm^2/atom)."""
    velocity = np.minimum(speed * BOHR_SPEED_INVERSE, z ** (2 / 3))     # units of v0
    return LSS_PREFACTOR * z ** (7 / 6) * z2 / (z ** (2 / 3) + z2 ** (2 / 3)) ** 1.5 * velocity


def nuclear_stopping(z: int, mass: float, z2: int, mass2: float, speed: np.ndarray) -> np.ndarray:
    """ZBL universal nuclear stopping of ion (Z, mass) in target atom (Z2, mass2) (eV cm^2/atom)."""
    energy_kev = 0.5 * mass * AMU_TO_MEV * speed ** 2 * 1e3
    screening = z ** 0.23 + z2 ** 0.23
    reduced = 32.53 * mass2 * energy_kev / (z * z2 * (mass + mass2) * screening)
    reduced_stopping = np.where(
        reduced <= 30.0,
        np.log1p(1.1383 * reduced) / (2.0 * (reduced + 0.01321 * reduced ** 0.21226
                                             + 0.19593 * np.sqrt(reduced))),
        np.log(np.maximum(reduced, 30.0)) / (2.0 * np.maximum(reduced, 30.0)))
    return 8.462e-15 * z * z2 * mass * reduced_stopping / ((mass + mass2) * screening)


def recoil_stopping_power(z: int, mass: float, speed: np.ndarray, composition) -> np.ndarray:
    """
    Stopping cross section of a recoil in a material (eV cm^2/atom).

    Args:
        z: Atomic number of the recoil
        mass: Mass of the recoil (u)
        speed: Recoil speed, v/c
        composition: (Z, atomic mass, atom fraction) per nuclide of the material
    """
    stopping = np.zeros_like(speed)
    for z2, mass2, fraction in composition:
        stopping += fraction * (electronic_stopping(z, z2, speed)
                                + nuclear_stopping(z, mass, z2, mass2, speed))
    return stopping


def transit_times(z: int, mass: float, speed_edges: np.ndarray,
                  medium: RecoilMedium) -> np.ndarray:
    """Time (s) a slowing recoil spends in each speed bin; infinite in the at-rest bin."""
    centers = 0.5 * (speed_edges[:-1] + speed_edges[1:])
    stopping = recoil_stopping_power(z, mass, centers, medium.composition) * 1e-6   # MeV cm^2
    # dt = dE / (n S v) with dE = M c^2 b db and v = b c
    times = (mass * AMU_TO_MEV * np.diff(speed_edges)
             / (medium.atom_density * stopping * SPEED_OF_LIGHT))
    times[0] = np.inf
    return times


def emission_histogram(arrivals: np.ndarray, lifetime: float, transit: np.ndarray) -> np.ndarray:
    """
    Speed distribution at emission for nuclei entering a level.

    Args:
        arrivals: Population entering the level per speed bin, at mid-bin
        lifetime: Mean lifetime of the level (s); 0 emits immediately
        transit: Transit time of each speed bin (s), see :func:`transit_times`
    """
    if lifetime <= 0.0:
        return arrivals.copy()
    survive_bin = np.exp(-transit / lifetime).tolist()
    survive_half = np.exp(-0.5 * transit / lifetime).tolist()
    entering = arrivals.tolist()
    emitted = [0.0] * len(entering)
    carried = 0.0
    for j in range(len(entering) - 1, -1, -1):
        leaving = carried * survive_bin[j] + entering[j] * survive_half[j]
        emitted[j] = carried + entering[j] - leaving
        carried = leaving
    return np.array(emitted)


def cascade_line_histograms(scheme: LevelScheme, arrivals: dict, lifetimes: np.ndarray,
                            transit: np.ndarray) -> dict:
    """
    Follow populations down a level scheme.

    Args:
        scheme: Level scheme of the recoiling nucleus
        arrivals: {level index: population per speed bin}; updated in place
        lifetimes: Mean lifetime per level (s)
        transit: Transit time of each speed bin (s)

    Returns:
        {E_gamma_MeV: intensity per speed bin at emission}
    """
    lines = {}
    for level in np.argsort(scheme.level_energies)[::-1].tolist():
        incoming = arrivals.get(level)
        if incoming is None:
            continue
        emitted = emission_histogram(incoming, lifetimes[level], transit)
        for e_gamma, probability, next_level in scheme.branches(level):
            if e_gamma > 0.0:
                lines[e_gamma] = lines.get(e_gamma, 0.0) + probability * emitted
            if next_level is not None:
                arrivals[next_level] = arrivals.get(next_level, 0.0) + probability * emitted
    return lines


def deposit_line(spectrum: np.ndarray, edges: np.ndarray, energy: float,
                 intensity: np.ndarray, speeds: np.ndarray) -> None:
    """
    Add one line to a binned spectrum.

    Args:
        spectrum: Spectrum to update
        edges: Ascending bin edges (MeV)
        energy: Rest-frame gamma energy (MeV)
        intensity: Intensity per speed bin
        speeds: Speed of each bin, v/c; zero means at rest
    """
    moving = (intensity > 0.0) & (speeds > 0.0)
    if np.any(moving):
        beta = speeds[moving]
        lorentz = energy / np.sqrt(1.0 - beta ** 2)
        low, high = lorentz * (1.0 - beta), lorentz * (1.0 + beta)
        first = max(int(np.searchsorted(edges, low.min(), side='right')) - 1, 0)
        last = min(int(np.searchsorted(edges, high.max(), side='left')), edges.size - 1)
        if last > first:
            window = edges[first:last + 1, np.newaxis]
            spectrum[first:last] += np.diff(
                np.clip((window - low) / (high - low), 0.0, 1.0) @ intensity[moving])
    at_rest = intensity[speeds <= 0.0].sum()
    if at_rest > 0.0 and edges[0] <= energy <= edges[-1]:
        spectrum[min(int(np.searchsorted(edges, energy, side='right')) - 1, spectrum.size - 1)] += at_rest


class ReactionTally:
    """
    Reactions behind gamma lines, by channel and alpha-energy group.

    Each thread-pool task fills its own tally; the tasks are merged in submission
    order so that results do not depend on thread scheduling.
    """

    def __init__(self, target=None, scale: float = 1.0):
        self._target = target
        self._scale = scale
        self.reactions = {}         # channel -> (target, reactions per alpha-energy group)
        self.photon_panels = {}     # photon-data channel -> photons per spectrum panel

    def add_reactions(self, channel: GammaChannel, e_alpha: np.ndarray,
                      reactions: np.ndarray) -> None:
        groups = (e_alpha / ALPHA_GROUP_MEV).astype(int)
        counts = np.bincount(groups, weights=reactions) * self._scale
        self._add_counts(channel, self._target, counts)

    def add_photon_panel(self, channel: GammaChannel, panel: int, photons: float) -> None:
        panels = self.photon_panels.setdefault(channel, np.zeros(len(channel.photon_spectra)))
        panels[panel] += photons * self._scale

    def merge(self, other: "ReactionTally") -> None:
        for channel, (target, counts) in other.reactions.items():
            self._add_counts(channel, target, counts)
        for channel, panels in other.photon_panels.items():
            if channel in self.photon_panels:
                self.photon_panels[channel] += panels
            else:
                self.photon_panels[channel] = panels.copy()

    def _add_counts(self, channel, target, counts):
        entry = self.reactions.get(channel)
        if entry is None:
            self.reactions[channel] = (target, counts.copy())
            return
        stored = entry[1]
        if counts.size > stored.size:
            stored = np.concatenate((stored, np.zeros(counts.size - stored.size)))
            self.reactions[channel] = (target, stored)
        stored[:counts.size] += counts


def recoil_speed_range(channel: GammaChannel, target, e_alpha: np.ndarray):
    """Lowest and highest lab recoil speed (v/c) for reactions at alpha energies ``e_alpha``."""
    residual_a = channel.residual[1]
    beta_cm = (np.sqrt(2.0 * e_alpha / (ALPH_MASS * AMU_TO_MEV))
               * ALPH_MASS / (ALPH_MASS + target.target_a))
    if channel.kind in (GammaChannel.KIND_DISCRETE, GammaChannel.KIND_CONTINUUM):
        available = np.maximum(
            e_alpha * target.cm_factor + channel.q_ground - channel.e_excitation, 0.0)
        kinetic = available * ANEUT_MASS / (ANEUT_MASS + residual_a)
        beta_relative = np.sqrt(2.0 * kinetic / (residual_a * AMU_TO_MEV))
    else:
        # Mean squared momentum of uncorrelated ejectiles (MeV^2)
        momentum_squared = np.zeros_like(e_alpha)
        for product in channel.products:
            if product.mass > 0.0:
                momentum_squared += (2.0 * product.mass * ANEUT_MASS * AMU_TO_MEV
                                     * product.multiplicity(e_alpha) * product.mean_energy(e_alpha))
        beta_relative = np.sqrt(np.maximum(momentum_squared, 0.0)) / (residual_a * AMU_TO_MEV)
    return np.abs(beta_cm - beta_relative), beta_cm + beta_relative


def _speed_kernel(speed_edges: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Fraction of recoils per speed bin (columns) for each alpha-energy group (rows)."""
    width = np.maximum(high ** 2 - low ** 2, 1e-30)[:, np.newaxis]
    cumulative = np.clip((speed_edges ** 2 - low[:, np.newaxis] ** 2) / width, 0.0, 1.0)
    return np.diff(cumulative, axis=1)


@lru_cache(maxsize=None)
def _level_lifetimes(scheme: LevelScheme, z: int, a: int) -> np.ndarray:
    """Mean lifetime (s) per level of a scheme, matched to RIPL-3 by energy; 0 when unknown."""
    tabulated = get_level_half_lives(z, a)
    lifetimes = np.zeros(len(scheme.level_energies))
    if not tabulated:
        return lifetimes
    energies = np.array([energy for energy, _ in tabulated])
    half_lives = [half_life for _, half_life in tabulated]
    for i, energy in enumerate(scheme.level_energies):
        nearest = int(np.argmin(np.abs(energies - energy)))
        if abs(energies[nearest] - energy) < LEVEL_MATCH_MAX_MEV and half_lives[nearest]:
            lifetimes[i] = half_lives[nearest] / np.log(2.0)
    return lifetimes


def _photon_channel_shapes(channel, panels, shape, speeds, spectrum, edges, remaining, shapes):
    """Discrete lines of an explicit photon product get ``shape``; its continuum is binned."""
    for panel, photons in enumerate(panels.tolist()):
        emission = photon_panel_emission(channel, panel) if photons > 0.0 else None
        if emission is None:
            continue
        disc_e, disc_p, norm, cont_e, cont_frac = emission
        for e_gamma, probability in zip(disc_e, disc_p):
            if probability > 0 and e_gamma > 0:
                shapes[round(float(e_gamma), 6)].append(
                    (photons * probability / norm * shape, speeds))
        emitted = (cont_frac > 0) & (cont_e > 0)
        if not np.any(emitted):
            continue
        # The collapsed continuum points stand for the cells between their midpoints.
        boundaries = np.concatenate((cont_e[:1], 0.5 * (cont_e[:-1] + cont_e[1:]), cont_e[-1:]))
        cell_photons = photons * np.where(emitted, cont_frac, 0.0)
        spectrum += np.diff(np.interp(edges, boundaries,
                                      np.concatenate(([0.0], np.cumsum(cell_photons)))))
        for e_gamma, amount in zip(cont_e[emitted], cell_photons[emitted]):
            key = round(float(e_gamma), 6)
            remaining[key] = remaining.get(key, 0.0) - amount


def _residual_line_shapes(residual, entries, photon_panels, medium,
                          spectrum, edges, remaining, shapes):
    """Collect the line shapes of one recoiling nucleus and bin its photon continuum."""
    z, a = residual
    groups = []
    for channel, target, reactions in entries:
        occupied = np.nonzero(reactions)[0]
        if occupied.size == 0:
            continue
        e_alpha = (occupied + 0.5) * ALPHA_GROUP_MEV
        low, high = recoil_speed_range(channel, target, e_alpha)
        groups.append((channel, target, occupied, e_alpha, reactions[occupied], low, high))
    if not groups:
        return
    max_speed = max(max(float(group[6].max()) for group in groups), 1e-6)
    speed_edges = np.linspace(0.0, max_speed * (1.0 + 1e-9), NUM_SPEED_BINS + 1)
    speeds = 0.5 * (speed_edges[:-1] + speed_edges[1:])
    speeds[0] = 0.0

    scheme = None
    arrivals = defaultdict(lambda: np.zeros(NUM_SPEED_BINS))
    for channel, target, occupied, e_alpha, reactions, low, high in groups:
        kernel = _speed_kernel(speed_edges, low, high)
        if channel.kind in (GammaChannel.KIND_DISCRETE, GammaChannel.KIND_CONTINUUM):
            scheme = channel.scheme
            arrivals[channel.level] += reactions @ kernel
        elif channel.kind == GammaChannel.KIND_WALK:
            scheme = channel.scheme
            budgets = walk_budget(channel, e_alpha, target.cm_factor)
            for group, budget, weight, row in zip(occupied.tolist(), budgets.tolist(),
                                                  reactions, kernel):
                visited = channel.walk_level_cache.get(group)
                if visited is None:
                    visited = walk_levels(scheme, budget)
                    channel.walk_level_cache[group] = visited
                for level in visited:
                    arrivals[level] += weight * row
        else:
            initial = reactions @ kernel
            if initial.sum() > 0.0:
                _photon_channel_shapes(channel, photon_panels[channel], initial / initial.sum(),
                                       speeds, spectrum, edges, remaining, shapes)

    if scheme is not None and arrivals:
        transit = transit_times(z, float(a), speed_edges, medium)
        lifetimes = _level_lifetimes(scheme, z, a)
        for e_gamma, histogram in cascade_line_histograms(
                scheme, dict(arrivals), lifetimes, transit).items():
            shapes[e_gamma].append((histogram, speeds))


def doppler_spectrum(tally: ReactionTally, lines: dict, energy_bins: np.ndarray,
                     medium: RecoilMedium) -> np.ndarray:
    """
    Doppler-broadened gamma spectrum of one beam problem.

    Args:
        tally: Reactions recorded while computing ``lines``
        lines: Rest-frame lines {E_gamma_MeV: intensity}
        energy_bins: Ascending bin edges (MeV)
        medium: Material the recoils slow down in

    Returns:
        Intensity per energy bin, in the units of ``lines``
    """
    spectrum = np.zeros(energy_bins.size - 1)
    remaining = dict(lines)
    shapes = defaultdict(list)

    by_residual = defaultdict(list)
    for channel, (target, reactions) in tally.reactions.items():
        by_residual[channel.residual].append((channel, target, reactions))
    for residual, entries in by_residual.items():
        _residual_line_shapes(residual, entries, tally.photon_panels, medium,
                              spectrum, energy_bins, remaining, shapes)

    for energy, intensity in remaining.items():
        # Only rounding is left of a line that was purely collapsed continuum.
        if intensity <= 1e-9 * lines.get(energy, 0.0):
            continue
        parts = shapes.get(energy)
        total = sum(histogram.sum() for histogram, _ in parts) if parts else 0.0
        if total > 0.0:
            for histogram, speeds in parts:
                deposit_line(spectrum, energy_bins, energy, histogram * (intensity / total), speeds)
        else:
            deposit_line(spectrum, energy_bins, energy, np.array([intensity]), np.zeros(1))

    # TENDL photon continua have weak tails past the default grid; a coarse or narrow
    # user grid is what this is meant to catch.
    total_lines = sum(lines.values())
    if total_lines > 0.0 and spectrum.sum() < total_lines * (1.0 - OUTSIDE_GRID_WARNING_FRACTION):
        logger.warning("%.3g%% of the gamma intensity lies outside gamma_energy_bins",
                       100.0 * (1.0 - spectrum.sum() / total_lines))
    return spectrum
