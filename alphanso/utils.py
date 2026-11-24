"""
Miscellaneous utility functions for ALPHANSO.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import interp1d

from .constants import ALPH_MASS, ANEUT_MASS
from .parsers import get_stopping_power
from .atomic_data_loader import (
    get_atomic_mass,
    get_natural_abundance,
    get_atomic_number,
    get_natural_isotopes
)


def matdef_to_zaids(matdef_input):
    """
    Convert material definition to ZAID format and calculate atom fractions.

    Args:
        matdef_input: Dict with isotope names/ZAIDs as keys and mass fractions as values
                     Examples: {'Al-27': 1.0}, {13027: 1.0}, {'C': 1.0} (natural element)
                     Natural elements can be specified as 'C', 'O', or as ZAIDs like 6000, 8000

    Returns:
        tuple: (mass_fractions_dict, atom_fractions_dict) with ZAIDs as keys
    """
    zaid_fractions = {}
    for key, value in matdef_input.items():
        if isinstance(key, str):
            parts = key.split('-')
            z = get_atomic_number(parts[0])
            if z is not None:
                if len(parts) == 2:
                    zaid = z * 1000 + int(parts[1])
                else:
                    zaid = z * 1000
                zaid_fractions[zaid] = float(value)
        else:
            zaid_fractions[int(key)] = float(value)

    mass_fractions = {}
    for zaid, value in zaid_fractions.items():
        a = zaid % 1000
        if a == 0:
            z = zaid // 1000
            natural_isos = get_natural_isotopes(z)
            if natural_isos:
                total_mass_weight = sum(
                    (get_natural_abundance(iso) or 0) * (get_atomic_mass(iso) or 0)
                    for iso in natural_isos
                )
                if total_mass_weight > 0:
                    for iso_zaid in natural_isos:
                        abundance = get_natural_abundance(iso_zaid)
                        iso_mass = get_atomic_mass(iso_zaid)
                        if abundance and abundance > 0 and iso_mass and iso_mass > 0:
                            mass_fractions[iso_zaid] = (
                                value * (abundance * iso_mass) / total_mass_weight
                            )
        else:
            mass_fractions[zaid] = value

    total_mass = sum(mass_fractions.values())
    if total_mass > 0:
        for zaid in mass_fractions:
            mass_fractions[zaid] /= total_mass

    atom_fractions = {}
    total_atoms = 0.0
    for zaid, mass_frac in mass_fractions.items():
        atomic_mass = get_atomic_mass(zaid)
        if atomic_mass is not None and atomic_mass > 0:
            atom_frac = mass_frac / atomic_mass
            atom_fractions[zaid] = atom_frac
            total_atoms += atom_frac

    if total_atoms > 0:
        for zaid in atom_fractions:
            atom_fractions[zaid] /= total_atoms

    return mass_fractions, atom_fractions


def rebin_xs(xs_dict, ebins, extrapolate=False):
    """Rebin a cross section dictionary to a new energy grid via linear interpolation."""
    energies = np.array(sorted(xs_dict.keys()))
    cross_sections = np.array([xs_dict[e] for e in energies])

    if extrapolate:
        fill_left = cross_sections[0]
        fill_right = cross_sections[-1]
    else:
        fill_left = 0
        fill_right = 0

    interpolated_cross_sections = np.interp(
        ebins, energies, cross_sections, left=fill_left, right=fill_right)

    new_xs_dict = dict(zip(ebins, interpolated_cross_sections))

    return new_xs_dict


def get_composite_stopping(mass_fractions, data_dir=None):
    """Calculate composite stopping power via Bragg-Kleeman weighting.

    Args:
        mass_fractions: Dict with ZAIDs as keys and mass fractions as values
        data_dir: Optional data directory path

    Returns:
        Dict with energies as keys and composite stopping powers as values
    """
    atom_fractions = {}
    total_atoms = 0.0

    for zaid, mass_frac in mass_fractions.items():
        atomic_mass = get_atomic_mass(zaid)
        if atomic_mass is not None and atomic_mass > 0:
            atom_frac = mass_frac / atomic_mass
            atom_fractions[zaid] = atom_frac
            total_atoms += atom_frac

    if total_atoms > 0:
        for zaid in atom_fractions:
            atom_fractions[zaid] /= total_atoms

    stopping_data_dict = {}
    all_energies = set()

    for zaid, afrac in atom_fractions.items():
        stopping_data = get_stopping_power(zaid, data_dir)
        if stopping_data is not None:
            stopping_data_dict[zaid] = (stopping_data, afrac)
            all_energies.update(stopping_data.keys())

    if not all_energies:
        return {}

    sorted_energies = sorted(all_energies)

    composite_stopping = {}

    for energy in sorted_energies:
        total_stopping = 0.0

        for zaid, (stopping_data, afrac) in stopping_data_dict.items():
            if energy in stopping_data:
                sp_value = stopping_data[energy]
            else:
                energies_list = sorted(stopping_data.keys())
                sp_values = [stopping_data[e] for e in energies_list]
                sp_value = np.interp(energy, energies_list, sp_values)

            total_stopping += sp_value * afrac

        composite_stopping[energy] = total_stopping

    return composite_stopping


def rebin_endf_spectrum(
    endf_spectrum: List[Tuple[float, float]],
    neutron_energy_bins: np.ndarray
) -> np.ndarray:
    """
    Rebin ENDF group integrals spectrum to neutron energy bins.

    Args:
        endf_spectrum: List of (energy [MeV], intensity [fraction]) tuples
        neutron_energy_bins: Energy bin edges [MeV]

    Returns:
        Spectrum values for each bin [len(neutron_energy_bins)-1], normalized
    """
    nng = len(neutron_energy_bins) - 1
    spectrum = np.zeros(nng)

    if not endf_spectrum:
        return spectrum

    endf_energies = np.array([e for e, _ in endf_spectrum])
    endf_intensities = np.array([i for _, i in endf_spectrum])

    total_intensity = np.sum(endf_intensities)
    if total_intensity > 0:
        endf_intensities = endf_intensities / total_intensity
    else:
        return spectrum

    interp_func = interp1d(
        endf_energies,
        endf_intensities,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    for n in range(nng):
        e_low = min(neutron_energy_bins[n], neutron_energy_bins[n + 1])
        e_high = max(neutron_energy_bins[n], neutron_energy_bins[n + 1])

        if e_low < 0 or e_high <= e_low:
            continue

        bin_center = (e_low + e_high) / 2.0
        bin_width = e_high - e_low

        intensity_at_center = float(interp_func(bin_center))
        spectrum[n] = intensity_at_center

    total = np.sum(spectrum)
    if total > 0:
        spectrum = spectrum / total

    return spectrum


_ALPHA_BREAKUP_ENERGY = 28.29566


def _kalbach_separation_energy(z_c, a_c, z_x, a_x, breakup_energy):
    """
    Separation energy (MeV) of a particle from a compound nucleus, Kalbach systematics.

    Uses the mass formula of Kalbach, Phys. Rev. C 37, 2350 (1988), Eq. 4, as given
    in the ENDF-6 Formats Manual, File 6, LAW=1, LANG=2.

    Args:
        z_c: int - Compound nucleus charge
        a_c: int - Compound nucleus mass number
        z_x: int - Charge of the nucleus left after the particle separates
        a_x: int - Mass number of the nucleus left after the particle separates
        breakup_energy: float - Binding energy of the particle itself (MeV)

    Returns:
        float - Separation energy in MeV
    """
    n_c = a_c - z_c
    n_x = a_x - z_x
    return (15.68 * (a_c - a_x)
            - 28.07 * ((n_c - z_c) ** 2 / a_c - (n_x - z_x) ** 2 / a_x)
            - 18.56 * (a_c ** (2.0 / 3.0) - a_x ** (2.0 / 3.0))
            + 33.22 * ((n_c - z_c) ** 2 / a_c ** (4.0 / 3.0)
                       - (n_x - z_x) ** 2 / a_x ** (4.0 / 3.0))
            - 0.717 * (z_c ** 2 / a_c ** (1.0 / 3.0) - z_x ** 2 / a_x ** (1.0 / 3.0))
            + 1.211 * (z_c ** 2 / a_c - z_x ** 2 / a_x)
            - breakup_energy)


def _kalbach_slope_alpha_n(e_alpha, e_out_cm, target_z, target_a):
    """
    Kalbach-Mann slope parameter a for (alpha,n), from the Kalbach systematics.

    ENDF-6 Formats Manual, File 6, LAW=1, LANG=2. For an incident alpha Ma = 0, so
    the X3 term vanishes and a = C1 X1 + C2 X1^3.

    Args:
        e_alpha: float - Incident alpha energy in the lab (MeV)
        e_out_cm: ndarray - Neutron emission energy in the center of mass (MeV)
        target_z: int - Target charge
        target_a: int - Target mass number

    Returns:
        ndarray - Slope parameter a, same shape as e_out_cm
    """
    z_c, a_c = target_z + 2, target_a + 4
    a_b = target_a + 3
    s_a = _kalbach_separation_energy(z_c, a_c, target_z, target_a, _ALPHA_BREAKUP_ENERGY)
    s_b = _kalbach_separation_energy(z_c, a_c, z_c, a_b, 0.0)
    e_a = e_alpha * target_a / (target_a + 4) + s_a
    e_b = np.asarray(e_out_cm) * (a_b + 1) / a_b + s_b
    x1 = min(e_a, 130.0) * e_b / e_a
    return 0.04 * x1 + 1.8e-6 * x1 ** 3


def _interpolate_continuum_table(tables, e_alpha, unit_base, density):
    """
    Interpolate a tabulated y(E_out | E_in) to one incident energy.

    Between the two bracketing incident energies the tables are mixed linearly in
    E_in. With unit_base, each table is first mapped onto [0, 1] over its own
    E_out range, mixed there, and mapped back onto the interpolated range
    (GNDS interpolationQualifier "unitbase"); density rescales y with the range so
    that it stays normalized, which "unitbase-unscaled" quantities skip.

    Args:
        tables: dict - {E_in_MeV: [(E_out_MeV, y), ...]}
        e_alpha: float - Incident alpha energy (MeV)
        unit_base: bool - Interpolate on a unit base
        density: bool - y is a density in E_out

    Returns:
        tuple (e_out, y) of ndarrays describing a piecewise-linear function, or
        None below the first tabulated incident energy
    """
    inc = sorted(tables.keys())
    if e_alpha < inc[0]:
        return None

    idx = int(np.searchsorted(inc, e_alpha, side='right')) - 1
    idx = max(0, min(idx, len(inc) - 2))
    e_lo = inc[idx]
    e_hi = inc[min(idx + 1, len(inc) - 1)]
    t = (e_alpha - e_lo) / (e_hi - e_lo) if e_hi > e_lo else 0.0
    t = max(0.0, min(t, 1.0))

    lo = np.asarray(tables[e_lo], dtype=float)
    hi = np.asarray(tables[e_hi], dtype=float)
    eo_lo, y_lo = lo[:, 0], lo[:, 1]
    eo_hi, y_hi = hi[:, 0], hi[:, 1]

    if not unit_base:
        e_out = np.unique(np.concatenate([eo_lo, eo_hi]))
        y = ((1.0 - t) * np.interp(e_out, eo_lo, y_lo, left=0.0, right=0.0)
             + t * np.interp(e_out, eo_hi, y_hi, left=0.0, right=0.0))
        return e_out, y

    w_lo = max(eo_lo[-1] - eo_lo[0], 1e-30)
    w_hi = max(eo_hi[-1] - eo_hi[0], 1e-30)
    x_lo = (eo_lo - eo_lo[0]) / w_lo
    x_hi = (eo_hi - eo_hi[0]) / w_hi
    x = np.unique(np.concatenate([x_lo, x_hi]))
    u_lo = np.interp(x, x_lo, y_lo)
    u_hi = np.interp(x, x_hi, y_hi)
    if density:
        u_lo = u_lo * w_lo
        u_hi = u_hi * w_hi
    u = (1.0 - t) * u_lo + t * u_hi
    e_min = (1.0 - t) * eo_lo[0] + t * eo_hi[0]
    width = (1.0 - t) * w_lo + t * w_hi
    y = u / width if density else u
    return e_min + x * width, y


def _continuum_lab_cdf(e_out, f, edges):
    """
    Cumulative integral of a lab-frame piecewise-linear energy density at given energies.

    Args:
        e_out: ndarray - Outgoing energies (MeV)
        f: ndarray - Density at e_out (1/MeV)
        edges: ndarray - Energies at which to evaluate the cumulative integral (MeV)

    Returns:
        ndarray - Integral of f from 0 to each edge
    """
    cdf = np.zeros(len(e_out))
    cdf[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * np.diff(e_out))
    return np.interp(edges, e_out, cdf, left=0.0, right=cdf[-1])


def _continuum_cm_cdf(e_out, f, r_table, e_alpha, e_cm_neutron,
                      target_z, target_a, edges, n_sub=4):
    """
    Cumulative lab-frame integral of a center-of-mass energy-angle distribution.

    Each panel of the piecewise-linear f(E') is split into n_sub sub-panels and
    represented by its midpoint E' with the sub-panel integral as weight. A neutron
    emitted with CM energy E' at CM cosine mu has lab energy
    E = E' + E_c + 2 sqrt(E' E_c) mu, where E_c is the energy of a neutron moving
    with the center of mass, so each node spreads over
    [(sqrt(E') - sqrt(E_c))^2, (sqrt(E') + sqrt(E_c))^2] following its angular
    distribution. Kalbach-Mann emission has
    p(mu) = a / (2 sinh a) (cosh(a mu) + r sinh(a mu)); without r it is isotropic.

    Args:
        e_out: ndarray - CM outgoing energies (MeV)
        f: ndarray - CM energy density at e_out (1/MeV)
        r_table: tuple or None - (e_out, r) precompound fraction, None for isotropic
        e_alpha: float - Incident alpha energy (MeV)
        e_cm_neutron: float - E_c (MeV)
        target_z: int - Target charge
        target_a: int - Target mass number
        edges: ndarray - Ascending lab energies at which to evaluate (MeV)
        n_sub: int - Sub-panels per tabulated panel

    Returns:
        ndarray - Lab-frame yield below each edge
    """
    de = np.diff(e_out)
    frac = (np.arange(n_sub) + 0.5) / n_sub
    nodes = (e_out[:-1, np.newaxis] + de[:, np.newaxis] * frac).ravel()
    weights = np.interp(nodes, e_out, f) * np.repeat(de / n_sub, n_sub)
    keep = weights > 0.0
    nodes = nodes[keep]
    weights = weights[keep]
    result = np.zeros(len(edges))
    if len(nodes) == 0:
        return result

    root_c = np.sqrt(e_cm_neutron)
    root_n = np.sqrt(nodes)
    lab_min = np.min((root_n - root_c) ** 2)
    lab_max = np.max((root_n + root_c) ** 2)
    result[edges >= lab_max] = weights.sum()
    active = (edges > lab_min) & (edges < lab_max)
    if not np.any(active):
        return result

    half_span = np.maximum(2.0 * root_n * root_c, 1e-300)
    mu = np.clip((edges[active, np.newaxis] - (nodes + e_cm_neutron)) / half_span,
                 -1.0, 1.0)

    if r_table is None:
        cdf = 0.5 * (mu + 1.0)
    else:
        a = _kalbach_slope_alpha_n(e_alpha, nodes, target_z, target_a)
        r = np.interp(nodes, r_table[0], r_table[1])
        sinh_a = np.sinh(a)
        x = np.exp(a * mu)
        cdf = (0.5 * ((1.0 + r) * x - (1.0 - r) / x)
               + sinh_a - r * np.cosh(a)) / (2.0 * sinh_a)

    result[active] = np.clip(cdf, 0.0, 1.0) @ weights
    return result


def _preprocess_continuum_dist(channel, e_steps_mev, neutron_bins, target_mass_amu,
                               e_spacing=0.02, n_sub=4):
    """
    Lab-frame continuum neutron spectrum shape on the output bins, per alpha step.

    Interpolates the channel's tabulated distribution to each incident alpha energy
    (on a unit base when the evaluation asks for it) and, for a center-of-mass
    distribution, transforms it to the lab using its angular distribution. The
    shape is evaluated at alpha energies about e_spacing apart and linearly
    interpolated between them, since it changes little over one slowing-down step.

    Args:
        channel: ContinuumChannel - MT=91 channel with a tabulated f
        e_steps_mev: ndarray - Incident alpha energies at each integration step (MeV)
        neutron_bins: ndarray - Neutron output energy bin edges (MeV)
        target_mass_amu: float - Target mass (amu)
        e_spacing: float - Alpha energy spacing at which the shape is evaluated (MeV)
        n_sub: int - Sub-panels per tabulated CM panel, see _continuum_cm_cdf

    Returns:
        ndarray - Shape (n_steps, n_bins). Each nonzero row sums to 1.
    """
    n_steps = len(e_steps_mev)
    n_bins = len(neutron_bins) - 1
    f_matrix = np.zeros((n_steps, n_bins))

    if channel is None or not channel.f:
        return f_matrix

    edges, inverse = np.unique(np.asarray(neutron_bins, dtype=float), return_inverse=True)
    i_lo = np.minimum(inverse[:-1], inverse[1:])
    i_hi = np.maximum(inverse[:-1], inverse[1:])
    target_z, target_a = channel.zaid // 1000, channel.zaid % 1000
    in_cm = channel.frame == 'centerOfMass'
    cm_factor = ALPH_MASS * ANEUT_MASS / (ALPH_MASS + target_mass_amu) ** 2

    def shape_at(e_alpha):
        table = _interpolate_continuum_table(channel.f, e_alpha, channel.unit_base, True)
        if table is None:
            return None
        e_out, f = table
        if in_cm:
            r_table = None
            if channel.r:
                r_table = _interpolate_continuum_table(
                    channel.r, e_alpha, channel.unit_base, False)
            cdf = _continuum_cm_cdf(
                e_out, f, r_table, e_alpha, cm_factor * e_alpha,
                target_z, target_a, edges, n_sub=n_sub)
        else:
            cdf = _continuum_lab_cdf(e_out, f, edges)
        row = np.maximum(0.0, cdf[i_hi] - cdf[i_lo])
        row_sum = row.sum()
        return row / row_sum if row_sum > 0.0 else None

    e_steps = np.asarray(e_steps_mev, dtype=float)
    e_first = min(channel.f.keys())
    above = np.nonzero(e_steps >= e_first)[0]
    if len(above) == 0:
        return f_matrix

    e_lo_eval, e_hi_eval = e_steps[above].min(), e_steps[above].max()
    n_eval = len(above)
    if e_spacing > 0.0:
        n_eval = max(2, int(np.ceil((e_hi_eval - e_lo_eval) / e_spacing)) + 1)
    if n_eval >= len(above):
        for i in above:
            row = shape_at(e_steps[i])
            if row is not None:
                f_matrix[i] = row
        return f_matrix

    e_eval = np.linspace(e_lo_eval, e_hi_eval, n_eval)
    rows = np.zeros((n_eval, n_bins))
    for k, e_alpha in enumerate(e_eval):
        row = shape_at(e_alpha)
        if row is not None:
            rows[k] = row
    e_q = e_steps[above]
    k = np.clip(np.searchsorted(e_eval, e_q, side='right') - 1, 0, n_eval - 2)
    t = ((e_q - e_eval[k]) / (e_eval[k + 1] - e_eval[k]))[:, np.newaxis]
    f_matrix[above] = (1.0 - t) * rows[k] + t * rows[k + 1]
    sums = f_matrix[above].sum(axis=1)
    f_matrix[above[sums > 0.0]] /= sums[sums > 0.0, np.newaxis]

    return f_matrix


def _accumulate_spectrum_continuum_box(
        b_lo, b_hi, cont_yield, e_steps,
        q_value, product_mass, target_mass_amu,
        aneut_mass, alph_mass):
    """
    Accumulate continuum channel spectrum using a kinematic box fallback.

    Used when no tabulated neutron energy distribution is available for MT=91.
    Distributes each alpha step's yield uniformly over [0, enmax_91] computed
    from the same two-body kinematic formula as discrete levels, with the Q-value
    at the start of the continuum.

    Args:
        b_lo: ndarray - Lower edges of neutron energy bins (MeV)
        b_hi: ndarray - Upper edges of neutron energy bins (MeV)
        cont_yield: ndarray - Continuum yield integrand at each alpha step
        e_steps: ndarray - Alpha energies at each step (MeV)
        q_value: float - Q-value at the start of the continuum, ENDF QI (MeV)
        product_mass: float - Product nucleus mass (amu)
        target_mass_amu: float - Target nucleus mass (amu)
        aneut_mass: float - Neutron mass (amu)
        alph_mass: float - Alpha particle mass (amu)

    Returns:
        ndarray - Accumulated spectrum, shape (len(b_lo),)
    """
    term1 = np.sqrt(alph_mass * aneut_mass * e_steps) / (aneut_mass + product_mass)
    term2 = (alph_mass * aneut_mass * e_steps) / (aneut_mass + product_mass) ** 2
    term3 = (product_mass * e_steps + product_mass * q_value -
             alph_mass * e_steps) / (aneut_mass + product_mass)
    sqrt_arg = term2 + term3

    valid = (cont_yield > 0.0) & (sqrt_arg >= 0.0)
    if not np.any(valid):
        return np.zeros(len(b_lo))

    y_v = cont_yield[valid]
    sqrt_val_v = np.sqrt(sqrt_arg[valid])
    enmax_v = (term1[valid] + sqrt_val_v) ** 2

    kinematic_valid = enmax_v > 0.0
    enmax_v = enmax_v[kinematic_valid]
    y_v = y_v[kinematic_valid]

    if len(y_v) == 0:
        return np.zeros(len(b_lo))

    density_v = y_v / enmax_v

    ov_upper = np.minimum(b_hi[:, np.newaxis], enmax_v[np.newaxis, :])
    ov_lower = b_lo[:, np.newaxis]
    overlap = np.maximum(0.0, ov_upper - ov_lower)
    return overlap @ density_v


def _interpolate_legendre_coeffs(
        ang_dist: Optional[Dict[float, List[float]]],
        e_alpha_mev) -> np.ndarray:
    """
    Linearly interpolate Legendre angular distribution coefficients at alpha energies.

    Args:
        ang_dist: dict, optional - {alpha_energy_MeV: [a_0, a_1, ..., a_L]}
        e_alpha_mev: float or ndarray - Alpha particle energies in MeV, shape (n,)

    Returns:
        ndarray - Interpolated coefficients, shape (n, L+1), zero-padded to the longest
        tabulated expansion. Energies outside the tabulated range, or all energies when
        ang_dist is None or empty, get the isotropic row [1, 0, ...].
    """
    e = np.atleast_1d(np.asarray(e_alpha_mev, dtype=float))
    if not ang_dist:
        coeffs = np.zeros((len(e), 1))
        coeffs[:, 0] = 1.0
        return coeffs

    energies = np.array(sorted(ang_dist.keys()))
    max_len = max(len(c) for c in ang_dist.values())
    table = np.zeros((len(energies), max_len))
    for row, e_key in enumerate(energies):
        c = ang_dist[e_key]
        table[row, :len(c)] = c

    coeffs = np.empty((len(e), max_len))
    for l in range(max_len):
        coeffs[:, l] = np.interp(e, energies, table[:, l])

    outside = (e < energies[0]) | (e > energies[-1])
    coeffs[outside] = 0.0
    coeffs[outside, 0] = 1.0
    return coeffs


def _legendre_antideriv_grid(coeffs, mu):
    """
    Evaluate the antiderivative of a Legendre-expanded angular distribution on a grid.

    Computes F(mu) = (1/2) * sum_l a_l * (P_{l+1}(mu) - P_{l-1}(mu)), with P_{-1} = 1,
    using the three-term recurrence for Legendre polynomials. The integral of the
    normalized distribution (a_0 = 1) from mu1 to mu2 is F(mu2) - F(mu1); an isotropic
    distribution ([1, 0, ...]) therefore gives F(mu2) - F(mu1) = (mu2 - mu1) / 2, which
    reproduces the uniform-in-mu result used by :func:`_accumulate_spectrum`.

    Args:
        coeffs: ndarray - Legendre coefficients, shape (n_valid, L+1) with a_0 = 1
        mu: ndarray - Evaluation points, shape (nng, n_valid)

    Returns:
        ndarray - Antiderivative values, shape (nng, n_valid)
    """
    n_coeffs = coeffs.shape[1]
    a = coeffs.T  # (L+1, n_valid), so a[l] broadcasts against mu's trailing axis
    p_prev = np.ones_like(mu)   # P_{l-1}, initialized to P_{-1} = 1
    p_curr = np.ones_like(mu)   # P_l,     initialized to P_0    = 1
    p_next = mu                 # P_{l+1}, initialized to P_1    = mu
    result = 0.5 * a[0][np.newaxis, :] * (p_next - p_prev)
    for l in range(1, n_coeffs):
        p_new = ((2 * l + 1) * mu * p_next - l * p_curr) / (l + 1)
        result = result + 0.5 * a[l][np.newaxis, :] * (p_new - p_curr)
        p_prev = p_curr
        p_curr = p_next
        p_next = p_new
    return result


def _accumulate_spectrum_legendre(
        b_lo, b_hi, valid_i, valid_j, yield_matrix, enmin, enmax, coeffs_padded):
    """
    Accumulate neutron spectrum using Legendre angular distribution weighting.

    For each valid (alpha_step, level) pair, distributes the yield into neutron energy
    bins using the analytic integral of the Legendre-expanded CM-frame angular
    distribution, mapped to lab-frame energy via E_n = mid + half_w * cos(theta_CM)
    where mid = (enmax + enmin) / 2 and half_w = (enmax - enmin) / 2. Pairs whose
    coefficients are [1, 0, ...] reduce exactly to isotropic emission.

    Args:
        b_lo: ndarray - Lower edges of neutron energy bins, shape (nng,)
        b_hi: ndarray - Upper edges of neutron energy bins, shape (nng,)
        valid_i: ndarray - Alpha step indices of valid (step, level) pairs, shape (n_valid,)
        valid_j: ndarray - Level indices of valid (step, level) pairs, shape (n_valid,)
        yield_matrix: ndarray - Yield weights, shape (n_steps, n_levels)
        enmin: ndarray - Minimum lab neutron energy, shape (n_steps, n_levels)
        enmax: ndarray - Maximum lab neutron energy, shape (n_steps, n_levels)
        coeffs_padded: ndarray - Legendre coefficients, shape (n_valid, max_L+1),
            padded with zeros. Use [1, 0, 0, ...] for the isotropic fallback.

    Returns:
        ndarray - Accumulated spectrum, shape (nng,)
    """
    nng = len(b_lo)
    n_valid = len(valid_i)
    if n_valid == 0:
        return np.zeros(nng)

    enmin_k = enmin[valid_i, valid_j]           # (n_valid,)
    enmax_k = enmax[valid_i, valid_j]
    y_k = yield_matrix[valid_i, valid_j]
    half_w = (enmax_k - enmin_k) * 0.5
    mid = (enmax_k + enmin_k) * 0.5

    # Clip each neutron energy bin to each pair's kinematic band -> (nng, n_valid)
    e_lo = np.maximum(b_lo[:, np.newaxis], enmin_k[np.newaxis, :])
    e_hi = np.minimum(b_hi[:, np.newaxis], enmax_k[np.newaxis, :])
    overlap = e_hi > e_lo

    inv_half_w = (1.0 / half_w)[np.newaxis, :]
    mid_row = mid[np.newaxis, :]
    mu1 = (e_lo - mid_row) * inv_half_w
    mu2 = (e_hi - mid_row) * inv_half_w

    integral = (_legendre_antideriv_grid(coeffs_padded, mu2)
                - _legendre_antideriv_grid(coeffs_padded, mu1))
    contrib = np.where(overlap, y_k[np.newaxis, :] * integral, 0.0)
    return contrib.sum(axis=1)
