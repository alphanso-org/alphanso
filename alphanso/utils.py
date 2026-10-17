"""
Miscellaneous utility functions for ALPHANSO.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import interp1d

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
                total_abundance = sum(
                    get_natural_abundance(iso) or 0 for iso in natural_isos
                )
                if total_abundance > 0:
                    for iso_zaid in natural_isos:
                        abundance = get_natural_abundance(iso_zaid)
                        if abundance and abundance > 0:
                            mass_fractions[iso_zaid] = value * (abundance / total_abundance)
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


def _preprocess_continuum_dist(
        continuum_dist: Optional[Dict[float, List[Tuple[float, float]]]],
        e_steps_mev: np.ndarray,
        neutron_bins: np.ndarray) -> np.ndarray:
    """
    Pre-process a tabulated continuum energy distribution onto the output energy bin grid.

    For each incident alpha energy step, bi-linearly interpolates f(E_out | E_alpha)
    between the two bracketing tabulated incident energies and integrates over each
    output energy bin using the trapezoidal rule.

    Args:
        continuum_dist: {incident_energy_MeV: [(E_out_MeV, prob_1/MeV), ...]}, or None
        e_steps_mev: ndarray - Incident alpha energies at each integration step (MeV)
        neutron_bins: ndarray - Neutron output energy bin edges (MeV)

    Returns:
        ndarray - Shape (n_steps, n_bins). Each row gives the fraction of continuum
        yield deposited in each output bin. Rows sum to <= 1.
    """
    n_steps = len(e_steps_mev)
    n_bins = len(neutron_bins) - 1
    f_matrix = np.zeros((n_steps, n_bins))

    if continuum_dist is None or len(continuum_dist) == 0:
        return f_matrix

    inc_energies = sorted(continuum_dist.keys())
    e_min_tab = inc_energies[0]

    b_lo = np.minimum(neutron_bins[:-1], neutron_bins[1:])
    b_hi = np.maximum(neutron_bins[:-1], neutron_bins[1:])

    for i in range(n_steps):
        e_alpha = e_steps_mev[i]
        if e_alpha < e_min_tab:
            continue

        idx = int(np.searchsorted(inc_energies, e_alpha, side='right')) - 1
        idx = min(idx, len(inc_energies) - 2)

        e_lo = inc_energies[idx]
        e_hi_tab = inc_energies[idx + 1]
        pairs_lo = continuum_dist[e_lo]
        pairs_hi = continuum_dist[e_hi_tab]

        eo_lo = np.array([p[0] for p in pairs_lo])
        fp_lo = np.array([p[1] for p in pairs_lo])
        eo_hi = np.array([p[0] for p in pairs_hi])
        fp_hi = np.array([p[1] for p in pairs_hi])

        merged_e = np.unique(np.concatenate([eo_lo, eo_hi]))
        f_lo_interp = np.interp(merged_e, eo_lo, fp_lo, left=0.0, right=0.0)
        f_hi_interp = np.interp(merged_e, eo_hi, fp_hi, left=0.0, right=0.0)

        t = (e_alpha - e_lo) / (e_hi_tab - e_lo) if e_hi_tab > e_lo else 0.0
        t = max(0.0, min(t, 1.0))
        f_blend = f_lo_interp + t * (f_hi_interp - f_lo_interp)

        cdf = np.zeros(len(merged_e))
        cdf[1:] = np.cumsum(
            0.5 * (f_blend[:-1] + f_blend[1:]) * np.diff(merged_e)
        )
        cdf_lo = np.interp(b_lo, merged_e, cdf, left=0.0, right=cdf[-1])
        cdf_hi = np.interp(b_hi, merged_e, cdf, left=0.0, right=cdf[-1])
        bin_integrals = np.maximum(0.0, cdf_hi - cdf_lo)

        valid_bins = b_hi > b_lo
        f_matrix[i, valid_bins] = bin_integrals[valid_bins]

        row_sum = f_matrix[i].sum()
        if row_sum > 0.0:
            f_matrix[i] /= row_sum

    return f_matrix


def _accumulate_spectrum_continuum_box(
        b_lo, b_hi, cont_yield, e_steps,
        q_value, product_mass, target_mass_amu,
        aneut_mass, alph_mass):
    """
    Accumulate continuum channel spectrum using a kinematic box fallback.

    Used when no tabulated neutron energy distribution is available for MT=91.
    Distributes each alpha step's yield uniformly over [0, enmax_91] computed
    from the same two-body kinematic formula as discrete levels with level_energy=0.

    Args:
        b_lo: ndarray - Lower edges of neutron energy bins (MeV)
        b_hi: ndarray - Upper edges of neutron energy bins (MeV)
        cont_yield: ndarray - Continuum yield integrand at each alpha step
        e_steps: ndarray - Alpha energies at each step (MeV)
        q_value: float - Ground-state Q-value used as continuum Q approximation (MeV)
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
        e_alpha_mev: float) -> Optional[np.ndarray]:
    """
    Linearly interpolate Legendre angular distribution coefficients at a given alpha energy.

    Args:
        ang_dist: dict, optional - {alpha_energy_MeV: [a_0, a_1, ..., a_L]}
        e_alpha_mev: float - Alpha particle energy in MeV

    Returns:
        ndarray, optional - Interpolated Legendre coefficients, or None if outside
        the tabulated energy range or if ang_dist is None
    """
    if ang_dist is None:
        return None

    energies = sorted(ang_dist.keys())
    if not energies or e_alpha_mev < energies[0] or e_alpha_mev > energies[-1]:
        return None

    idx = int(np.searchsorted(energies, e_alpha_mev, side='right')) - 1
    idx = min(idx, len(energies) - 2)

    e_lo = energies[idx]
    e_hi = energies[idx + 1]
    c_lo = ang_dist[e_lo]
    c_hi = ang_dist[e_hi]

    max_len = max(len(c_lo), len(c_hi))
    arr_lo = np.zeros(max_len)
    arr_hi = np.zeros(max_len)
    arr_lo[:len(c_lo)] = c_lo
    arr_hi[:len(c_hi)] = c_hi

    t = (e_alpha_mev - e_lo) / (e_hi - e_lo) if e_hi > e_lo else 0.0
    return arr_lo + t * (arr_hi - arr_lo)


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
