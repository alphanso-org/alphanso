import logging
import numpy as np
import math
import os
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
from scipy.special import erf
from collections import defaultdict
from .constants import AVOGADRO_NUM, ANEUT_MASS, ALPH_MASS
from .atomic_data_loader import atomic_data
from .parsers import (
    get_an_xs,
    get_branching_info,
    get_continuum_info,
    get_decay_spectrum,
    get_stopping_power,
    get_gamma_cascade_info,
    load_delayed_neutron_data)
from .data_manager import ensure_data
from .output_files import normalize_results_payload, write_results_yaml
from .utils import (
    rebin_xs,
    get_composite_stopping,
    matdef_to_zaids,
    rebin_endf_spectrum,
    _accumulate_spectrum_continuum_box,
    _preprocess_continuum_dist,
)

logger = logging.getLogger(__name__)


def _accumulate_spectrum(b_lo, b_hi, y_flat, w_flat, enmin_flat, enmax_flat):
    ov_upper = np.minimum(b_hi[:, None], enmax_flat[None, :])
    ov_lower = np.maximum(b_lo[:, None], enmin_flat[None, :])
    overlap = np.maximum(0.0, ov_upper - ov_lower)
    return np.sum((y_flat / w_flat)[None, :] * overlap, axis=1)


def _build_range_table(energies, stops):
    de = np.diff(energies)
    increments = 0.5 * (1.0 / stops[:-1] + 1.0 / stops[1:]) * de
    range_table = np.empty(len(energies))
    range_table[0] = 0.0
    np.cumsum(increments, out=range_table[1:])
    return range_table


def _reverse_spectrum_results(results: dict) -> dict:
    """Reverse spectrum arrays to output in increasing energy order."""
    spectrum_keys = ['an_spectrum', 'sf_spectrum', 'combined_spectrum',
                     'an_spectrum_absolute', 'delayedn_spectrum']
    bin_keys = ['neutron_energy_bins', 'spectrum_energy_bins']

    for key in spectrum_keys:
        if key in results and results[key] is not None:
            results[key] = results[key][::-1]

    for key in bin_keys:
        if key in results and results[key] is not None:
            results[key] = results[key][::-1]

    if 'spectrum_layers' in results and results['spectrum_layers'] is not None:
        results['spectrum_layers'] = [
            s[::-1] if s is not None else None
            for s in results['spectrum_layers']
        ]

    return results


def _gamma_line_pairs(line_items) -> list:
    """Convert gamma line pairs to built-in list pairs."""
    return [
        [float(energy), float(intensity)]
        for energy, intensity in line_items
    ]


class Transport(object):
    @staticmethod
    def calculate(config: dict) -> dict:
        """
        Main entry point that dispatches to the appropriate calculation type based on config.

        Args:
            config: dict - Configuration dictionary with 'calc_type' and calculation-specific parameters

        Returns:
            dict - Results dictionary with yields, spectra, and diagnostic information

        Raises:
            ValueError: If calc_type is not one of 'beam', 'homogeneous', 'interface', or 'sandwich'
        """

        ensure_data()

        # Convert neutron_energy_bins shorthand [start, stop, num_points] to full array
        if 'neutron_energy_bins' in config and isinstance(
                config['neutron_energy_bins'], list) and len(
                config['neutron_energy_bins']) == 3:
            config['neutron_energy_bins'] = np.linspace(
                config['neutron_energy_bins'][0],
                config['neutron_energy_bins'][1],
                int(config['neutron_energy_bins'][2]))

        calc_type = config.get('calc_type')

        if calc_type == 'beam':
            results = _reverse_spectrum_results(Transport._calculate_beam(config))
        elif calc_type == 'homogeneous':
            results = _reverse_spectrum_results(Transport._calculate_homogeneous(config))
        elif calc_type == 'interface':
            results = _reverse_spectrum_results(Transport._calculate_interface(config))
        elif calc_type == 'sandwich':
            results = _reverse_spectrum_results(Transport._calculate_sandwich(config))
        else:
            raise ValueError(f"Unknown calculation type: {calc_type}")

        results = normalize_results_payload(results)

        # Save results to files if output_dir is specified
        output_dir = config.get('output_dir')
        save_data_files = config.get('save_data_files', True)
        if output_dir and save_data_files:
            write_results_yaml(results, output_dir)

        return results

    @staticmethod
    def _calculate_beam(config: dict) -> dict:
        """
        Calculate neutron production from alpha beam incident on thick target.

        Args:
            config: dict - Configuration containing:
                - matdef: dict - Target material definition
                - beam_energy: float OR beam_intensities: list - Alpha beam specification
                - num_alpha_groups, min_alpha_energy, max_alpha_energy: optional - Alpha energy grid params
                - neutron_energy_bins: ndarray, optional - Neutron energy grid
                - an_xs_data_dir, stopping_power_data_dir: str, optional
                - calculate_gammas: bool, optional - Enable gamma calculation (default: True)
                - gamma_data_dir: str, optional - Gamma cascade data directory

        Returns:
            dict - Results containing:
                - an_yield: float - Neutrons per incident alpha
                - an_spectrum: list - Normalized spectrum
                - an_spectrum_absolute: list - Absolute spectrum
                - neutron_energy_bins: list - Energy bin edges
                - gamma_yield: float, optional - Gammas per incident alpha (if calculate_gammas=True)
                - gamma_lines: list, optional - Discrete gamma lines

        Raises:
            ValueError: If neither 'beam_energy' nor 'beam_intensities' is specified
        """

        matdef = config['matdef']

        num_alpha_groups = config.get('num_alpha_groups')
        min_alpha_energy = config.get('min_alpha_energy')
        max_alpha_energy = config.get('max_alpha_energy')
        neutron_energy_bins = config.get('neutron_energy_bins')
        an_xs_data_dir = config.get('an_xs_data_dir')
        stopping_power_data_dir = config.get('stopping_power_data_dir')

        calculate_gammas = config.get('calculate_gammas', True)
        gamma_data_dir = config.get('gamma_data_dir')

        if 'beam_intensities' in config:
            energies = config['beam_intensities']
        elif 'beam_energy' in config:
            beam_energy = config['beam_energy']
            energies = [[beam_energy, 1.0]]
        else:
            raise ValueError(
                "Beam calculation must specify either 'beam_energy' or 'beam_intensities'")

        beam_results = Transport.beam_problem(
            energies, matdef,
            data_dir=None,
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            neutron_energy_bins=neutron_energy_bins,
            an_xs_data_dir=an_xs_data_dir,
            stopping_power_data_dir=stopping_power_data_dir,
            calculate_gammas=calculate_gammas,
            gamma_data_dir=gamma_data_dir
        )

        p_total = beam_results['neutron_yield']
        total_spectrum = beam_results['neutron_spectrum']
        neutron_energy_bins = beam_results['neutron_energy_bins']

        results = {
            'an_yield': float(p_total),
            'neutron_energy_bins': neutron_energy_bins.tolist() if neutron_energy_bins is not None else None
        }

        if total_spectrum is not None:
            results['an_spectrum'] = total_spectrum.tolist()
            results['an_spectrum_absolute'] = (total_spectrum * p_total).tolist()
            results['spectrum_energy_bins'] = neutron_energy_bins.tolist(
            ) if neutron_energy_bins is not None else None

        if calculate_gammas and 'gamma_yield' in beam_results:
            results['gamma_yield'] = float(beam_results['gamma_yield'])
            results['gamma_lines'] = beam_results['gamma_lines']

        return results

    @staticmethod
    def _calculate_homogeneous(config: dict) -> dict:
        """
        Calculate neutron production from alpha emitters uniformly mixed in target material.

        Args:
            config: dict - Configuration containing:
                - matdef: dict - Material definition (sources and targets)
                - num_alpha_groups, min_alpha_energy, max_alpha_energy: optional - Alpha energy grid params
                - neutron_energy_bins: ndarray, optional - Neutron energy grid
                - an_xs_data_dir, stopping_power_data_dir, decay_data_dir: str, optional

        Returns:
            dict - Results containing:
                - an_yield, sf_yield, combined_yield: float - Individual and combined yields
                - an_spectrum, sf_spectrum, combined_spectrum: list - Normalized spectra
                - neutron_energy_bins: list - Energy bin edges
                - average_energy, average_energy_an, average_energy_sf: float - Average energies
                - sf_contributors: list - SF nuclide information (if any)
        """

        matdef = config['matdef']

        num_alpha_groups = config.get('num_alpha_groups')
        min_alpha_energy = config.get('min_alpha_energy')
        max_alpha_energy = config.get('max_alpha_energy')
        neutron_energy_bins = config.get('neutron_energy_bins')
        an_xs_data_dir = config.get('an_xs_data_dir')
        stopping_power_data_dir = config.get('stopping_power_data_dir')
        decay_data_dir = config.get('decay_data_dir')

        calculate_gammas = config.get('calculate_gammas', True)
        gamma_data_dir = config.get('gamma_data_dir')

        result = Transport.homogeneous_problem(
            matdef,
            data_dir=None,
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            neutron_energy_bins=neutron_energy_bins,
            an_xs_data_dir=an_xs_data_dir,
            stopping_power_data_dir=stopping_power_data_dir,
            decay_data_dir=decay_data_dir,
            calculate_gammas=calculate_gammas,
            gamma_data_dir=gamma_data_dir
        )

        return result

    @staticmethod
    def _integrate_over_ebins(
            e_alpha,
            neutron_energy_bins,
            an_xs,
            stopping_power,
            branching_ratios,
            energy_levels,
            q_value,
            product_mass,
            target_mass_amu,
            ep_branching,
            gamma_cascades=None,
            continuum_xs=None,
            continuum_dist=None):
        """
        Calculate neutron yield and energy spectrum from alpha slowing down in target.

        Args:
            e_alpha: float - Initial alpha energy (MeV)
            neutron_energy_bins: ndarray - Neutron energy grid for spectrum calculation
            an_xs: dict - (\alpha,n) cross sections {energy: cross_section}
            stopping_power: dict - Alpha stopping power {energy: stopping_power}
            branching_ratios: dict - Branching ratios {energy: [fractions]}
            energy_levels: list - Product nucleus energy levels (MeV)
            q_value: float - Reaction Q-value (MeV)
            product_mass: float - Product nucleus mass (amu)
            target_mass_amu: float - Target nucleus mass (amu)
            ep_branching: list - Alpha energy grid for branching ratio interpolation
            gamma_cascades: dict, optional - Gamma cascade data {level_idx: [(final, E_gamma, prob), ...]}
            continuum_xs: dict, optional - MT=91 continuum cross sections
                {energy_MeV: cross_section_barns}. If None, no continuum channel is added.
            continuum_dist: dict, optional - MT=91 neutron energy distribution
                {incident_energy_MeV: [(E_out_MeV, prob_1/MeV), ...]}. If None and
                continuum_xs is not None, falls back to a kinematic box over [0, enmax_91].

        Returns:
            tuple - (an_yield: float, spectrum: ndarray, gamma_yield: float, gamma_lines: list)
            an_yield includes both discrete and continuum contributions.
        """

        nng = len(neutron_energy_bins) - 1
        spectrum = np.zeros(nng)

        e_grid = np.linspace(e_alpha, 0.0, 2000)
        e_alpha_steps = 0.5 * (e_grid[:-1] + e_grid[1:])
        de = e_grid[:-1] - e_grid[1:]

        ee_cs = np.array(list(an_xs.keys()))
        cx_cs = np.array(list(an_xs.values()))
        sort_idx_cs = np.argsort(ee_cs)
        ee_cs = ee_cs[sort_idx_cs]
        cx_cs = cx_cs[sort_idx_cs]

        ee_sp = np.array(list(stopping_power.keys()))
        sp_vals = np.array(list(stopping_power.values()))
        sort_idx_sp = np.argsort(ee_sp)
        ee_sp = ee_sp[sort_idx_sp]
        sp_vals = sp_vals[sort_idx_sp]

        cs_cm2_grid = np.interp(
            e_alpha_steps,
            ee_cs,
            cx_cs,
            left=0.0,
            right=0.0) * 1e-24
        sp_grid = np.interp(e_alpha_steps, ee_sp, sp_vals, left=0.0, right=0.0)

        valid_mask = (sp_grid > 1e-30) & (cs_cm2_grid > 0)

        if not np.any(valid_mask):
            return 0.0, spectrum, 0.0, []

        e_steps_valid = e_alpha_steps[valid_mask]
        de_valid = de[valid_mask]
        prob_steps = cs_cm2_grid[valid_mask] / sp_grid[valid_mask]

        if branching_ratios and energy_levels:
            idx = np.searchsorted(
                ep_branching,
                e_steps_valid,
                side='right') - 1
            idx = np.clip(idx, 0, len(ep_branching) - 1)

            num_levels = len(energy_levels)
            br_table = np.zeros((len(ep_branching), num_levels))
            for k_idx, e_key in enumerate(ep_branching):
                br_vals = branching_ratios[e_key]
                length = min(len(br_vals), num_levels)
                br_table[k_idx, :length] = br_vals[:length]

            br_matrix = br_table[idx]

        else:
            num_levels = len(energy_levels) if energy_levels else 1
            br_matrix = np.ones((len(e_steps_valid), num_levels))
            if energy_levels is None:
                energy_levels = [0.0]

        E_alpha = e_steps_valid[:, np.newaxis]
        E_level = np.array(energy_levels)[np.newaxis, :]
        Q_level = q_value - E_level

        Threshold = np.where(Q_level < 0, -
                             Q_level *
                             (ANEUT_MASS +
                              product_mass) /
                             (ANEUT_MASS +
                              product_mass -
                              ALPH_MASS), 0.0)

        kinematics_mask = (E_alpha >= Threshold) & (br_matrix > 0)

        term1 = np.sqrt(ALPH_MASS * ANEUT_MASS * E_alpha) / \
            (ANEUT_MASS + product_mass)
        term2 = (ALPH_MASS * ANEUT_MASS * E_alpha) / \
            ((ANEUT_MASS + product_mass) ** 2)
        term3 = (product_mass * E_alpha + product_mass * Q_level -
                 ALPH_MASS * E_alpha) / (ANEUT_MASS + product_mass)
        sqrt_arg = term2 + term3

        valid_physics = (sqrt_arg >= 0) & kinematics_mask

        sqrt_val = np.sqrt(np.maximum(sqrt_arg, 0))

        senmax = term1 + sqrt_val
        enmax = senmax * senmax

        e90 = -Q_level * product_mass / (product_mass - ALPH_MASS)

        cond_e90 = E_alpha <= e90
        senmin = np.where(cond_e90, term1 - sqrt_val, -term1 + sqrt_val)
        enmin = senmin * senmin

        valid_physics &= (enmax > enmin)

        yield_matrix = prob_steps[:, np.newaxis] * \
            br_matrix * de_valid[:, np.newaxis]

        cont_yield_steps = None
        if continuum_xs is not None:
            cont_xs_e = np.array(sorted(continuum_xs.keys()))
            cont_xs_v = np.array([continuum_xs[e] for e in cont_xs_e])
            sigma91_cm2 = np.interp(
                e_steps_valid, cont_xs_e, cont_xs_v, left=0.0, right=0.0) * 1e-24
            sigma_total_cm2 = cs_cm2_grid[valid_mask]
            f_91 = np.minimum(sigma91_cm2 / sigma_total_cm2, 1.0)
            if np.any(f_91 > 0.0):
                yield_matrix = yield_matrix * (1.0 - f_91[:, np.newaxis])
                cont_yield_steps = prob_steps * de_valid * f_91

        width_matrix = enmax - enmin
        width_matrix[width_matrix <= 0] = 1e-30

        if not np.any(valid_physics):
            return 0.0, spectrum, 0.0, []

        y_flat = yield_matrix[valid_physics]
        w_flat = width_matrix[valid_physics]
        enmin_flat = enmin[valid_physics]
        enmax_flat = enmax[valid_physics]

        bin_edges = neutron_energy_bins
        b_lo = np.minimum(bin_edges[:-1], bin_edges[1:])
        b_hi = np.maximum(bin_edges[:-1], bin_edges[1:])

        spectrum = _accumulate_spectrum(b_lo, b_hi, y_flat, w_flat, enmin_flat, enmax_flat)

        if cont_yield_steps is not None:
            if continuum_dist is not None:
                f_matrix = _preprocess_continuum_dist(
                    continuum_dist, e_steps_valid, neutron_energy_bins)
                spectrum = spectrum + np.einsum('i,ij->j', cont_yield_steps, f_matrix)
            else:
                spectrum = spectrum + _accumulate_spectrum_continuum_box(
                    b_lo, b_hi, cont_yield_steps, e_steps_valid,
                    q_value, product_mass, target_mass_amu,
                    ANEUT_MASS, ALPH_MASS)

        if gamma_cascades is not None:
            gamma_yield, gamma_lines = Transport._calculate_gamma_spectrum(
                yield_matrix,
                valid_physics,
                energy_levels,
                gamma_cascades,
            )
        else:
            gamma_yield, gamma_lines = 0.0, []

        return (np.sum(spectrum), spectrum,
                gamma_yield, gamma_lines)

    @staticmethod
    def _calculate_gamma_spectrum(
        yield_matrix: np.ndarray,
        valid_physics: np.ndarray,
        energy_levels: list,
        gamma_cascades: dict,
    ):
        """
        Calculate gamma ray yield and lines from nuclear de-excitation.

        This method computes gamma ray production from excited nuclear levels populated
        by (alpha,n) reactions. Each populated level de-excites by emitting gamma rays
        according to the cascade transitions defined in gamma_cascades.

        Args:
            yield_matrix: ndarray - Population rate for each level [alpha_steps, levels]
                Units: neutrons/s or per incident alpha depending on context
            valid_physics: ndarray - Boolean mask for physically allowed transitions [alpha_steps, levels]
            energy_levels: list - Excited state energies in MeV, index corresponds to level
            gamma_cascades: dict - Gamma transition data {level_idx: [(final_idx, E_gamma, prob), ...]}

        Returns:
            tuple: (total_gamma_yield, gamma_lines)
                - total_gamma_yield: float - Total gamma ray production rate
                - gamma_lines: list - [[energy_MeV, intensity], ...] sorted discrete gamma lines

        Algorithm:
            1. Extract valid level populations from yield_matrix using valid_physics mask
            2. For each populated level:
                - Lookup gamma transitions from gamma_cascades
                - For each transition (i -> f, E_gamma, prob):
                    - Accumulate: gamma_intensity[E_gamma] += population[i] * prob
            3. Convert accumulated intensities to sorted line list
        """
        if gamma_cascades is None:
            return 0.0, []

        gamma_lines_dict = defaultdict(float)

        valid_yields = yield_matrix[valid_physics]

        if len(valid_yields) == 0:
            return 0.0, []

        num_levels = len(energy_levels)

        level_populations = np.zeros(num_levels)
        for level_idx in range(num_levels):
            level_mask = valid_physics[:, level_idx] if level_idx < valid_physics.shape[1] else np.zeros(
                valid_physics.shape[0], dtype=bool)
            if np.any(level_mask):
                level_populations[level_idx] = np.sum(
                    yield_matrix[level_mask, level_idx])

        active_populations = level_populations.copy()
        
        for _ in range(num_levels + 1):
            new_populations = np.zeros(num_levels)
            any_moved = False
            
            for level_idx in range(1, num_levels):
                pop = active_populations[level_idx]
                if pop <= 0:
                    continue
                    
                if level_idx not in gamma_cascades:
                    energy = energy_levels[level_idx]
                    if energy > 0:
                        gamma_lines_dict[round(energy, 6)] += pop
                    any_moved = True
                    continue
                
                transitions = gamma_cascades[level_idx]
                for final_idx, gamma_energy, transition_prob in transitions:
                    if transition_prob <= 0:
                        continue
                        
                    if gamma_energy > 0:
                        gamma_lines_dict[round(gamma_energy, 6)] += pop * transition_prob
                    
                    if final_idx > 0:
                        new_populations[final_idx] += pop * transition_prob
                    any_moved = True
            
            active_populations = new_populations
            if not any_moved or np.sum(active_populations) <= 0:
                break

        gamma_lines = _gamma_line_pairs(sorted(gamma_lines_dict.items()))
        total_gamma_yield = sum(intensity for _, intensity in gamma_lines)

        return total_gamma_yield, gamma_lines

    @staticmethod
    def beam_problem(
            energies,
            matdef,
            data_dir=None,
            num_alpha_groups=None,
            min_alpha_energy=None,
            max_alpha_energy=None,
            neutron_energy_bins=None,
            an_xs_data_dir=None,
            stopping_power_data_dir=None,
            calculate_gammas=False,
            gamma_data_dir=None):
        """
        Calculate neutron production from alpha beam incident on thick target.

        Args:
            energies: list - [[energy_MeV, intensity], ...] pairs for incident alphas
            matdef: dict - Target material definition {ZAID: mass_fraction}
            data_dir: str, optional - Legacy data directory (deprecated)
            num_alpha_groups: int, optional - Number of alpha energy groups (default: 15000)
            min_alpha_energy: float, optional - Minimum alpha energy in MeV (default: 1e-11)
            max_alpha_energy: float, optional - Maximum alpha energy in MeV (default: 15)
            neutron_energy_bins: ndarray, optional - Neutron spectrum energy grid
            an_xs_data_dir: str, optional - (\alpha,n) cross section data directory
            stopping_power_data_dir: str, optional - Stopping power data directory
            calculate_gammas: bool, optional - Enable gamma calculation (default: False)
            gamma_data_dir: str, optional - Directory containing gamma cascade data

        Returns:
            dict - Results dictionary containing:
                - neutron_yield: float - Neutron yield per incident alpha
                - neutron_spectrum: ndarray - Normalized neutron spectrum
                - neutron_energy_bins: ndarray - Energy bins for neutrons
                - gamma_yield: float - Gamma yield per incident alpha (if calculate_gammas=True)
                - gamma_lines: list - [[energy_MeV, intensity], ...] discrete gamma lines
        """

        if num_alpha_groups is None:
            num_alpha_groups = 15000
        if min_alpha_energy is None:
            min_alpha_energy = 1e-11
        if max_alpha_energy is None:
            max_alpha_energy = 15
        ebins = np.linspace(min_alpha_energy, max_alpha_energy, num_alpha_groups + 1)
        if neutron_energy_bins is None:
            neutron_energy_bins = np.linspace(15.0, 0.0, 101)

        mass_fractions, atom_fractions = matdef_to_zaids(matdef)

        stopping_data_source = stopping_power_data_dir if stopping_power_data_dir is not None else data_dir
        an_xs_data_source = an_xs_data_dir if an_xs_data_dir is not None else data_dir
        gamma_data_source = gamma_data_dir if gamma_data_dir is not None else an_xs_data_source

        stopping_power = get_composite_stopping(
            mass_fractions, stopping_data_source)
        stopping_binned = rebin_xs(stopping_power, ebins, extrapolate=True)

        total_spectrum = np.zeros(len(neutron_energy_bins) - 1)
        p_total = 0

        if calculate_gammas:
            total_gamma_yield = 0.0
            total_gamma_lines = defaultdict(float)

        target_data_list = []
        for zaid, afrac in atom_fractions.items():
            an_xs_data = get_an_xs(zaid, an_xs_data_source)
            if an_xs_data is None:
                continue

            an_xs_binned = rebin_xs(an_xs_data, ebins)

            target_mass_amu = atomic_data.get_atomic_mass(zaid)
            if target_mass_amu is None:
                logger.warning(
                    f"Target mass not found for ZAID {zaid}. Skipping this target.")
                continue

            z = zaid // 1000
            a = zaid % 1000
            product_zaid = (z + 2) * 1000 + (a + 3)
            product_mass = atomic_data.get_atomic_mass(product_zaid)
            if product_mass is None:
                logger.warning(
                    f"Product mass not found for ZAID {product_zaid}. Skipping this target.")
                continue

            q_value, level_energies, branching_data = get_branching_info(
                zaid, an_xs_data_source)
            if q_value is None or level_energies is None or branching_data is None or len(
                    level_energies) == 0:
                logger.warning(
                    f"No branching data available for target ZAID {zaid}. Skipping this target.")
                continue

            gamma_cascades = None
            if calculate_gammas:
                gamma_cascades = get_gamma_cascade_info(
                    product_zaid,
                    data_dir=gamma_data_source,
                    level_energies=level_energies
                )

            continuum_xs_data, continuum_dist_data = get_continuum_info(
                zaid, an_xs_data_source)

            target_data_list.append({
                'zaid': zaid,
                'afrac': afrac,
                'an_xs_binned': an_xs_binned,
                'target_mass_amu': target_mass_amu,
                'product_mass': product_mass,
                'q_value': q_value,
                'level_energies': level_energies,
                'branching_data': branching_data,
                'energy_keys': sorted(branching_data.keys()),
                'gamma_cascades': gamma_cascades,
                'continuum_xs': continuum_xs_data,
                'continuum_dist': continuum_dist_data,
            })

        def _worker(e, intensity, t_data):
            min_xs_energy = min(t_data['energy_keys'])
            if e < min_xs_energy:
                return None

            energy_keys = t_data['energy_keys']
            alpha_energy_index = np.searchsorted(energy_keys, e, side='right') - 1
            if alpha_energy_index < 0:
                alpha_energy_index = 0
            if alpha_energy_index >= len(energy_keys):
                alpha_energy_index = len(energy_keys) - 1

            closest_energy = energy_keys[alpha_energy_index]
            f_branching = np.array(t_data['branching_data'][closest_energy])

            valid_levels = []
            valid_bf_columns = []
            for level_idx, level_energy in enumerate(t_data['level_energies']):
                q_eff = t_data['q_value'] - level_energy
                if q_eff < 0:
                    threshold = -q_eff * (ANEUT_MASS + t_data['product_mass']) / t_data['target_mass_amu']
                else:
                    threshold = 0
                if e >= threshold and level_energy >= 0 and level_energy < 50:
                    valid_levels.append(level_energy)
                    valid_bf_columns.append(level_idx)

            if not valid_levels:
                return None

            scale = t_data['afrac'] * intensity
            p, spectrum, gamma_y, gamma_lines = Transport._integrate_over_ebins(
                e,
                neutron_energy_bins,
                t_data['an_xs_binned'],
                stopping_binned,
                t_data['branching_data'],
                valid_levels,
                t_data['q_value'],
                t_data['product_mass'],
                t_data['target_mass_amu'],
                t_data['energy_keys'],
                gamma_cascades=t_data['gamma_cascades'] if calculate_gammas else None,
                continuum_xs=t_data.get('continuum_xs'),
                continuum_dist=t_data.get('continuum_dist'),
            )
            return {
                'p': p * scale,
                'spectrum': spectrum * scale,
                'gamma_y': gamma_y * scale,
                'gamma_lines': _gamma_line_pairs(
                    (eg, ig * scale) for eg, ig in gamma_lines
                ),
            }

        futures = []
        with ThreadPoolExecutor() as pool:
            for e_i_pair in energies:
                e = e_i_pair[0]
                intensity = e_i_pair[1]
                if intensity == 0:
                    continue
                for t_data in target_data_list:
                    futures.append(pool.submit(_worker, e, intensity, t_data))

        for future in futures:
            result = future.result()
            if result is None:
                continue
            p_total += result['p']
            total_spectrum += result['spectrum']
            if calculate_gammas:
                total_gamma_yield += result['gamma_y']
                for eg, ig in result['gamma_lines']:
                    total_gamma_lines[eg] += ig

        if np.sum(total_spectrum) > 0:
            normalized_spectrum = total_spectrum / np.sum(total_spectrum)
        else:
            normalized_spectrum = total_spectrum

        results = {
            'neutron_yield': p_total,
            'neutron_spectrum': normalized_spectrum,
            'neutron_energy_bins': neutron_energy_bins
        }

        if calculate_gammas:
            results['gamma_yield'] = total_gamma_yield
            results['gamma_lines'] = _gamma_line_pairs(sorted(total_gamma_lines.items()))

        return results

    @staticmethod
    def homogeneous_problem(
            matdef,
            data_dir=None,
            num_alpha_groups=None,
            min_alpha_energy=None,
            max_alpha_energy=None,
            neutron_energy_bins=None,
            an_xs_data_dir=None,
            stopping_power_data_dir=None,
            decay_data_dir=None,
            calculate_gammas=False,
            gamma_data_dir=None):
        """
        Calculate neutron production from uniform mixture of alpha emitters in target material.

        Args:
            matdef: dict - Material definition {ZAID: mass_fraction}
            data_dir: str, optional - Legacy data directory (deprecated)
            num_alpha_groups: int, optional - Number of alpha energy groups (default: 15000)
            min_alpha_energy: float, optional - Minimum alpha energy in MeV (default: 1e-11)
            max_alpha_energy: float, optional - Maximum alpha energy in MeV (default: 15)
            neutron_energy_bins: ndarray, optional - Neutron spectrum energy grid
            an_xs_data_dir: str, optional - (alpha,n) cross section data directory
            stopping_power_data_dir: str, optional - Stopping power data directory
            decay_data_dir: str, optional - Decay spectrum data directory
            calculate_gammas: bool, optional - Enable gamma calculation (default: False)
            gamma_data_dir: str, optional - Directory containing gamma cascade data

        Returns:
            dict - Complete results including (alpha,n) and SF contributions:
                'an_yield': (alpha,n) neutron yield (n/s/g)
                'sf_yield': SF neutron yield (n/s/g)
                'combined_yield': Combined yield (n/s/g)
                'an_spectrum': Normalized (alpha,n) spectrum
                'sf_spectrum': Normalized SF spectrum
                'combined_spectrum': Combined normalized spectrum
                'neutron_energy_bins': Energy bin edges (MeV)
                'average_energy': Average neutron energy (MeV)
                'average_energy_an': Average (alpha,n) energy (MeV)
                'average_energy_sf': Average SF energy (MeV)
                'sf_contributors': SF nuclide information (if any SF nuclides present)
        """
        mass_fractions, atom_fractions = matdef_to_zaids(matdef)

        spectrum = defaultdict(float)

        for zaid, wtfrac in mass_fractions.items():
            atomic_mass = atomic_data.get_atomic_mass(zaid)
            atoms_per_gram = AVOGADRO_NUM / atomic_mass * wtfrac

            a_per_sec_per_atom, alpha_spectrum = get_decay_spectrum(
                zaid,
                data_dir=decay_data_dir if decay_data_dir is not None else data_dir,
                decay_mode='alpha'
            )

            for line in alpha_spectrum:
                e = line[0]
                i = line[1] * a_per_sec_per_atom * atoms_per_gram
                spectrum[e] += i

        energies = [[e, float(i)] for e, i in spectrum.items()]

        if neutron_energy_bins is None:
            neutron_energy_bins = np.linspace(15.0, 0.0, 101)

        decay_data_source = decay_data_dir if decay_data_dir is not None else data_dir

        beam_results = Transport.beam_problem(
            energies, mass_fractions, data_dir=data_dir,
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            neutron_energy_bins=neutron_energy_bins,
            an_xs_data_dir=an_xs_data_dir, stopping_power_data_dir=stopping_power_data_dir,
            calculate_gammas=calculate_gammas,
            gamma_data_dir=gamma_data_dir)

        p_total_an = beam_results['neutron_yield']
        spectrum_an = beam_results['neutron_spectrum']
        neutron_energy_bins = beam_results['neutron_energy_bins']

        gamma_yield_an = beam_results.get('gamma_yield', 0.0)
        gamma_lines_an = beam_results.get('gamma_lines', [])

        p_total_sf = 0.0
        spectrum_sf = np.zeros(len(neutron_energy_bins) - 1)
        sf_contributors = []
        p_total_delayedn = 0.0
        spectrum_delayedn = np.zeros(len(neutron_energy_bins) - 1)

        for zaid, wtfrac in mass_fractions.items():
            atomic_mass = atomic_data.get_atomic_mass(zaid)
            atoms_per_gram = AVOGADRO_NUM / atomic_mass * wtfrac

            sf_strength, endf_spectrum, sf_params = get_decay_spectrum(
                zaid,
                data_dir=decay_data_source,
                decay_mode='sf',
                return_params=True
            )

            if sf_strength == 0.0 or not sf_params:
                continue

            watt_a = sf_params.get('watt_a', 0.0)
            watt_b = sf_params.get('watt_b', 0.0)
            nubar = sf_params.get('nubar', 0.0)
            endf_avg_energy = sf_params.get('avg_energy', 0.0)

            sf_yield_nuclide = sf_strength * atoms_per_gram
            p_total_sf += sf_yield_nuclide

            if watt_a > 0.0 and watt_b > 0.0:
                watt_spec = Transport._calculate_watt_spectrum(
                    neutron_energy_bins,
                    watt_a,
                    watt_b,
                    normalize=True
                )
                spectrum_sf += sf_yield_nuclide * watt_spec
            elif endf_spectrum and len(endf_spectrum) > 0:
                endf_spec_binned = rebin_endf_spectrum(
                    endf_spectrum,
                    neutron_energy_bins
                )
                spectrum_sf += sf_yield_nuclide * endf_spec_binned
            else:
                logger.warning(
                    f"ZAID {zaid}: Has SF yield but missing Watt parameters (a={watt_a}, b={watt_b}) and no ENDF spectrum. ")

            z = zaid // 1000
            a = zaid % 1000
            symbol = atomic_data.get_element_symbol(z)
            sf_contributors.append({
                'zaid': int(zaid),
                'name': f"{symbol}-{a}",
                'sf_yield': float(sf_yield_nuclide),
                'sf_branching': float(sf_params.get('sf_branching', 0.0)),
                'nubar': float(nubar),
                'watt_a': float(watt_a),
                'watt_b': float(watt_b),
                'endf_avg_energy': float(endf_avg_energy)
            })

            dn_data = load_delayed_neutron_data(zaid)
            if dn_data and nubar > 0.0:
                nu_delayed = dn_data['nu_delayed']
                fission_rate = sf_yield_nuclide / nubar
                delayedn_yield_nuclide = fission_rate * nu_delayed
                p_total_delayedn += delayedn_yield_nuclide
                dn_tuples = [
                    (e, s * 0.05)
                    for e, s in zip(dn_data['energy_grid_MeV'], dn_data['spectrum_per_MeV'])
                ]
                spectrum_delayedn += delayedn_yield_nuclide * rebin_endf_spectrum(
                    dn_tuples, neutron_energy_bins
                )
                sf_contributors[-1]['nu_delayed']     = float(nu_delayed)
                sf_contributors[-1]['delayedn_yield'] = float(delayedn_yield_nuclide)
            else:
                sf_contributors[-1]['nu_delayed']     = 0.0
                sf_contributors[-1]['delayedn_yield'] = 0.0

        if np.sum(spectrum_delayedn) > 0:
            spectrum_delayedn_normalized = spectrum_delayedn / np.sum(spectrum_delayedn)
        else:
            spectrum_delayedn_normalized = spectrum_delayedn

        if np.sum(spectrum_sf) > 0:
            spectrum_sf_normalized = spectrum_sf / np.sum(spectrum_sf)
        else:
            spectrum_sf_normalized = spectrum_sf

        p_total_combined = p_total_an + p_total_sf

        if p_total_combined > 0:
            spectrum_combined = (
                p_total_an * spectrum_an +
                p_total_sf * spectrum_sf_normalized
            ) / p_total_combined
        else:
            spectrum_combined = spectrum_an

        def calc_avg_energy(spec, ebins):
            """Calculate average energy from spectrum (for non-Watt spectra)."""
            bin_centers = 0.5 * (ebins[:-1] + ebins[1:])
            return float(np.sum(spec * bin_centers)
                         ) if np.sum(spec) > 0 else 0.0

        avg_energy_sf = 0.0
        if sf_contributors:
            total_sf_yield = 0.0
            weighted_ebar_sum = 0.0
            for contrib in sf_contributors:
                watt_a_contrib = contrib['watt_a']
                watt_b_contrib = contrib['watt_b']
                sf_yield_contrib = contrib['sf_yield']
                endf_avg_contrib = contrib.get('endf_avg_energy', 0.0)

                if watt_a_contrib > 0.0 and watt_b_contrib > 0.0 and sf_yield_contrib > 0.0:
                    ebar_contrib = 0.25 * watt_a_contrib * watt_a_contrib * \
                        watt_b_contrib + 1.5 * watt_a_contrib
                    weighted_ebar_sum += ebar_contrib * sf_yield_contrib
                    total_sf_yield += sf_yield_contrib
                elif endf_avg_contrib > 0.0 and sf_yield_contrib > 0.0:
                    weighted_ebar_sum += endf_avg_contrib * sf_yield_contrib
                    total_sf_yield += sf_yield_contrib

            if total_sf_yield > 0:
                avg_energy_sf = weighted_ebar_sum / total_sf_yield
            else:
                avg_energy_sf = calc_avg_energy(
                    spectrum_sf_normalized, neutron_energy_bins)
        else:
            avg_energy_sf = calc_avg_energy(
                spectrum_sf_normalized, neutron_energy_bins)

        avg_energy_an = calc_avg_energy(spectrum_an, neutron_energy_bins)
        avg_energy_total = calc_avg_energy(
            spectrum_combined, neutron_energy_bins)

        result = {
            'an_yield': float(p_total_an),
            'sf_yield': float(p_total_sf),
            'combined_yield': float(p_total_combined),
            'an_spectrum': spectrum_an.tolist() if isinstance(
                spectrum_an,
                np.ndarray) else spectrum_an,
            'sf_spectrum': spectrum_sf_normalized.tolist(),
            'combined_spectrum': spectrum_combined.tolist() if isinstance(
                spectrum_combined,
                np.ndarray) else spectrum_combined,
            'neutron_energy_bins': neutron_energy_bins.tolist(),
            'average_energy_an': avg_energy_an,
            'average_energy_sf': avg_energy_sf,
            'average_energy': avg_energy_total,
        }

        if calculate_gammas:
            result['gamma_yield'] = float(gamma_yield_an)
            result['gamma_lines'] = gamma_lines_an

        if sf_contributors:
            result['sf_contributors'] = sf_contributors

        result['delayedn_strength'] = float(p_total_delayedn)
        result['delayedn_spectrum'] = spectrum_delayedn_normalized.tolist()

        return result

    @staticmethod
    def _yield_integration(e_alpha, an_xs_binned, stopping_binned):
        """
        Calculate total neutron yield from alpha slowing down (simplified integration).

        Args:
            e_alpha: float - Initial alpha energy (MeV)
            an_xs_binned: dict - Binned (\alpha,n) cross sections {energy: cross_section}
            stopping_binned: dict - Binned stopping power {energy: stopping_power}

        Returns:
            float - Total neutron yield per incident alpha
        """
        ebins = np.array(list(an_xs_binned.keys()))
        cs_values = np.array(list(an_xs_binned.values()))
        sp_values = np.array(list(stopping_binned.values()))

        alpha_idx = np.searchsorted(ebins, e_alpha, side='right') - 1
        if alpha_idx < 0 or alpha_idx >= len(ebins) - 1:
            return 0.0

        de = np.diff(ebins[:alpha_idx + 1])
        cs_avg = (cs_values[:alpha_idx] + cs_values[1:alpha_idx + 1]) / 2
        sp_avg = (sp_values[:alpha_idx] + sp_values[1:alpha_idx + 1]) / 2

        cs_avg_cm2 = cs_avg * 1e-24

        sp_avg = np.where(sp_avg > 1e-20, sp_avg, 1e-20)

        integrand = cs_avg_cm2 / sp_avg
        p = np.sum(integrand * de)

        return p

    @staticmethod
    def _calculate_watt_spectrum(
        neutron_energy_bins: np.ndarray,
        watt_a: float,
        watt_b: float,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Calculate Watt fission spectrum using analytical integration.

        Watt spectrum: f(E) = C * exp(-E/a) * sinh(sqrt(bE))

        where:
            C = normalization constant
            a = Watt parameter [MeV]
            b = Watt parameter [1/MeV]
            E = neutron energy [MeV]

        Integration over energy bin [E_low, E_high]:
        integral of f(E) dE = (1/sqrt{pi ba}) * [exp terms] + (1/2) * [erf terms]

        Args:
            neutron_energy_bins: Energy bin edges [MeV], can be ascending or descending
            watt_a: Watt parameter a [MeV], must be > 0
            watt_b: Watt parameter b [1/MeV], must be > 0
            normalize: If True, normalize spectrum to sum to 1.0

        Returns:
            Spectrum values for each bin [len(neutron_energy_bins)-1]
            Returns zeros if invalid parameters
        """
        nng = len(neutron_energy_bins) - 1
        spectrum = np.zeros(nng)

        if watt_a <= 0 or watt_b <= 0:
            logger.warning(
                f"Invalid Watt parameters: a={watt_a}, b={watt_b}. Returning zero spectrum."
            )
            return spectrum

        spba = np.sqrt(np.pi * watt_b * watt_a)

        sa = np.sqrt(watt_a)
        sbaso4 = np.sqrt(watt_b * watt_a * watt_a / 4.0)

        for n in range(nng):
            e_low = min(neutron_energy_bins[n], neutron_energy_bins[n + 1])
            e_high = max(neutron_energy_bins[n], neutron_energy_bins[n + 1])

            if e_low < 0 or e_high <= e_low:
                continue

            if e_low == 0:
                e_low = 1e-10

            seh = np.sqrt(e_high)
            sel = np.sqrt(e_low)

            c1 = (sel - sbaso4) / sa
            c2 = (seh - sbaso4) / sa
            c3 = (sel + sbaso4) / sa
            c4 = (seh + sbaso4) / sa

            eft = 0.0
            if c1 < 0.0 and c2 < 0.0:
                eft = 0.5 * (erf(-c1) - erf(-c2))
            elif c1 < 0.0 and c2 >= 0.0:
                eft = 0.5 * (erf(-c1) + erf(c2))
            elif c1 >= 0.0 and c2 >= 0.0:
                eft = 0.5 * (erf(c2) - erf(c1))

            eft += 0.5 * (erf(c4) - erf(c3))

            c1s = c1 * c1
            c2s = c2 * c2
            c3s = c3 * c3
            c4s = c4 * c4

            exp_contrib = (1.0 / spba) * (
                np.exp(-c1s) - np.exp(-c2s) - np.exp(-c3s) + np.exp(-c4s)
            )

            spectrum[n] = eft + exp_contrib

        if normalize:
            total = np.sum(spectrum)
            if total > 0:
                spectrum = spectrum / total
            else:
                logger.warning("Watt spectrum sum is zero, cannot normalize")

        return spectrum

    def _integrate_inverse_stopping(
            stopping_power,
            max_energy=15.0):
        """
        Calculate range integral for alpha particles in material.

        Computes integral of 1/stopping_power(E) dE over energy bins for range calculations.

        Args:
            stopping_power: dict - Stopping power data {energy_MeV: stopping_power_MeV_cm}
            max_energy: float, optional - Maximum energy for integration (MeV)

        Returns:
            ndarray - Array of [energy_center_MeV, integral_cm] pairs
        """
        max_energy = min(max_energy, max(stopping_power.keys()))

        step_size = 0.1
        energies = np.arange(max_energy, 0, -step_size)
        energies = np.append(energies, 0.0)
        energies = np.array(sorted(energies), float)

        sp_e = np.array(sorted(stopping_power.keys()), float)
        sp_v = np.array([stopping_power[e] for e in sp_e], float)
        left_val = sp_v[0]
        right_val = sp_v[-1]
        stops = np.interp(energies, sp_e, sp_v, left=left_val, right=right_val)

        e_bin_edges = energies
        e_bin_centers = 0.5 * (e_bin_edges[:-1] + e_bin_edges[1:])

        inv_stops = 1.0 / stops
        integrals = 0.5 * (inv_stops[:-1] + inv_stops[1:]) * np.diff(e_bin_edges)
        stopping_integral = np.column_stack([e_bin_centers, integrals])
        return stopping_integral

    @staticmethod
    def degrade_alpha_energy_through_layer(
        alpha_energies,
        layer_matdef,
        layer_density,
        layer_thickness,
        stopping_power_data_dir=None,
        num_alpha_groups=None,
        min_alpha_energy=None,
        max_alpha_energy=None,
        n_angular_bins=20
    ):
        """
        Calculate exit energies of alphas after traversing intermediate layer with angular dependence.

        Args:
            alpha_energies: list - [(energy, intensity), ...] pairs for incident alphas
            layer_matdef: dict - Layer material definition {ZAID: mass_fraction}
            layer_density: float - Layer density (g/cm^3)
            layer_thickness: float - Layer thickness (cm)
            stopping_power_data_dir: str, optional - Path to stopping power data
            num_alpha_groups: int, optional - Number of alpha energy groups (default: 15000)
            min_alpha_energy: float, optional - Minimum alpha energy in MeV (default: 1e-11)
            max_alpha_energy: float, optional - Maximum alpha energy in MeV (default: 15)
            n_angular_bins: int - Number of angular bins for path length distribution (default: 20)

        Returns:
            list - [(energy, intensity), ...] pairs for alphas exiting the layer
                Alphas that stop in the layer are filtered out
        """

        if not alpha_energies or len(alpha_energies) == 0:
            return []

        mass_fractions, atom_fractions = matdef_to_zaids(layer_matdef)

        avg_atomic_mass = sum(atomic_data.get_atomic_mass(zaid) * afrac
                              for zaid, afrac in atom_fractions.items())
        atom_density = layer_density * AVOGADRO_NUM / avg_atomic_mass

        stopping = get_composite_stopping(
            mass_fractions, data_dir=stopping_power_data_dir)

        stopping_abs = {e: s * atom_density for e, s in stopping.items()}

        max_energy = max([e for e, i in alpha_energies])
        max_energy = min(max_energy, max(stopping_abs.keys()))

        if num_alpha_groups is None:
            num_alpha_groups = 15000
        if min_alpha_energy is None:
            min_alpha_energy = 1e-11
        if max_alpha_energy is None:
            max_alpha_energy = 15
        ebins = np.linspace(min_alpha_energy, max_alpha_energy, num_alpha_groups + 1)
        energies = np.array(sorted(ebins), float)
        energies = energies[energies <= max_energy]
        if energies[-1] < max_energy:
            energies = np.append(energies, max_energy)

        sp_e = np.array(sorted(stopping_abs.keys()), float)
        sp_v = np.array([stopping_abs[e] for e in sp_e], float)
        left_val = sp_v[0]
        right_val = sp_v[-1]
        stops = np.interp(energies, sp_e, sp_v, left=left_val, right=right_val)

        range_table = _build_range_table(energies, stops)

        cos_theta = np.linspace(1.0, 0.01, n_angular_bins + 1)
        cos_theta_centers = 0.5 * (cos_theta[:-1] + cos_theta[1:])

        d_omega = 2.0 * np.pi * (cos_theta[:-1] - cos_theta[1:])
        d_omega_normalized = d_omega / (2.0 * np.pi)

        path_lengths = layer_thickness / cos_theta_centers

        alpha_arr = np.array(alpha_energies)
        e_in_arr = alpha_arr[:, 0]
        intensity_arr = alpha_arr[:, 1]
        valid_alpha = (e_in_arr > 0) & (intensity_arr > 0)
        e_in_arr = e_in_arr[valid_alpha]
        intensity_arr = intensity_arr[valid_alpha]

        if len(e_in_arr) == 0:
            return []

        range_in_arr = np.interp(e_in_arr, energies, range_table)
        range_out = range_in_arr[:, None] - path_lengths[None, :]
        valid_mask = range_in_arr[:, None] >= path_lengths[None, :]
        e_out_all = np.interp(range_out.ravel(), range_table, energies).reshape(range_out.shape)
        valid_mask &= (e_out_all > 0) & (e_out_all < e_in_arr[:, None])

        e_out_valid = e_out_all[valid_mask]
        intensity_out_valid = (intensity_arr[:, None] * d_omega_normalized[None, :])[valid_mask]

        if len(e_out_valid) == 0:
            return []

        e_out_binned = np.round(e_out_valid, 4)
        unique_energies, inverse = np.unique(e_out_binned, return_inverse=True)
        intensities = np.zeros(len(unique_energies))
        np.add.at(intensities, inverse, intensity_out_valid)

        keep = intensities > 0
        return list(zip(unique_energies[keep].tolist(), intensities[keep].tolist()))

    @staticmethod
    def sandwich_alpha_term_bc(
        source_matdef,
        source_density,
        intermediate_layers,
        num_alpha_groups=None,
        min_alpha_energy=None,
        max_alpha_energy=None,
        stopping_power_data_dir=None,
        n_angular_bins=20
    ):
        """
        Calculate alpha spectrum at final interface after degradation through all intermediate layers.

        Composes interface emission with sequential energy degradation through multiple layers
        to obtain alpha spectrum entering target material.

        Args:
            source_matdef: dict - Source material definition {ZAID: mass_fraction}
            source_density: float - Source density (g/cm^3)
            intermediate_layers: list - Layer dicts containing:
                - 'matdef': dict - Layer material definition
                - 'thickness': float - Layer thickness (cm)
                - 'density': float - Layer density (g/cm^3)
            num_alpha_groups: int, optional - Number of alpha energy groups (default: 15000)
            min_alpha_energy: float, optional - Minimum alpha energy in MeV (default: 1e-11)
            max_alpha_energy: float, optional - Maximum alpha energy in MeV (default: 15)
            stopping_power_data_dir: str, optional - Path to stopping power data
            n_angular_bins: int - Number of angular bins for degradation (default: 20)

        Returns:
            list - [(energy_MeV, intensity), ...] pairs for alpha spectrum at final interface
        """
        alpha_spectrum = Transport.interface_alpha_term(
            source_matdef=source_matdef,
            source_density=source_density,
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            stopping_power_data_dir=stopping_power_data_dir
        )

        for layer in intermediate_layers:
            alpha_spectrum = Transport.degrade_alpha_energy_through_layer(
                alpha_energies=alpha_spectrum,
                layer_matdef=layer['matdef'],
                layer_density=layer['density'],
                layer_thickness=layer['thickness'],
                stopping_power_data_dir=stopping_power_data_dir,
                num_alpha_groups=num_alpha_groups,
                min_alpha_energy=min_alpha_energy,
                max_alpha_energy=max_alpha_energy,
                n_angular_bins=n_angular_bins
            )
            if len(alpha_spectrum) == 0:
                logger.warning("All alphas stopped in intermediate layers")
                break

        return alpha_spectrum

    @staticmethod
    def interface_alpha_term(source_matdef,
                             source_density,
                             num_alpha_groups=None,
                             min_alpha_energy=None,
                             max_alpha_energy=None,
                             stopping_power_data_dir=None,
                             decay_data_dir=None):
        """
        Calculate alpha spectrum at interface from volumetric source emission.

        Integrates alpha emission from semi-infinite source region to compute
        spectrum arriving at interface with target material.

        Args:
            source_matdef: dict - Source material definition {ZAID: mass_fraction}
            source_density: float - Source density (g/cm^3)
            num_alpha_groups: int, optional - Number of alpha energy groups (default: 15000)
            min_alpha_energy: float, optional - Minimum alpha energy in MeV (default: 1e-11)
            max_alpha_energy: float, optional - Maximum alpha energy in MeV (default: 15)
            stopping_power_data_dir: str, optional - Path to stopping power data

        Returns:
            list - [(energy_MeV, intensity), ...] pairs for alpha spectrum at interface
        """
        mass_fractions, atom_fractions = matdef_to_zaids(source_matdef)
        avg_atomic_mass = sum(atomic_data.get_atomic_mass(zaid) * afrac
                              for zaid, afrac in atom_fractions.items())
        atom_density = source_density * AVOGADRO_NUM / avg_atomic_mass

        energies = Transport.get_bulk_alpha_source(source_matdef, decay_data_dir=decay_data_dir)
        volumetric_energies = [[e, i * source_density] for e, i in energies]
        max_energy = max([e[0] for e in volumetric_energies])
        stopping = get_composite_stopping(
            mass_fractions, data_dir=stopping_power_data_dir)
        stopping = {e: s * atom_density for e, s in stopping.items()}
        stopping_integral = Transport._integrate_inverse_stopping(
            stopping, max_energy=max_energy)
        interface_energies = {}
        for e_i in volumetric_energies:
            ev = e_i[0]
            Sv = e_i[1]
            mask = stopping_integral[:, 0] <= ev
            if np.any(mask):
                for eg_ig in stopping_integral[mask]:
                    eg = eg_ig[0]
                    ig = Sv / 4 * eg_ig[1]
                    if eg in interface_energies:
                        interface_energies[eg] += ig
                    else:
                        interface_energies[eg] = ig
        interface_energies_list = [(e, i)
                                   for e, i in interface_energies.items()]
        interface_energies_list.sort(key=lambda x: x[0])
        return interface_energies_list

    @staticmethod
    def _calculate_interface(config: dict) -> dict:
        """
        Calculate neutron production from source material in contact with target material.

        Solves interface geometry where semi-infinite source region contacts target,
        accounting for volumetric alpha emission and slowing down in source.

        Args:
            config: dict - Configuration containing:
                - source_matdef: dict - Source material definition
                - source_density: float - Source density (g/cm^3)
                - target_matdef: dict - Target material definition
                - num_alpha_groups, min_alpha_energy, max_alpha_energy: optional - Alpha energy grid params
                - neutron_energy_bins: ndarray, optional - Neutron energy grid
                - an_xs_data_dir, stopping_power_data_dir: str, optional

        Returns:
            dict - Results containing:
                - an_yield: float - Neutron yield (n/s/cm^2)
                - an_spectrum: list - Normalized spectrum
                - an_spectrum_absolute: list - Absolute spectrum
                - neutron_energy_bins: list
                - spectrum_energy_bins: list
        """
        num_alpha_groups = config.get('num_alpha_groups')
        min_alpha_energy = config.get('min_alpha_energy')
        max_alpha_energy = config.get('max_alpha_energy')
        neutron_energy_bins = config.get('neutron_energy_bins')
        an_xs_data_dir = config.get('an_xs_data_dir')
        stopping_power_data_dir = config.get('stopping_power_data_dir')
        decay_data_dir = config.get('decay_data_dir')
        source_matdef = config['source_matdef']
        source_density = config.get('source_density')

        calculate_gammas = config.get('calculate_gammas', True)
        gamma_data_dir = config.get('gamma_data_dir')

        interface_spectrum = Transport.interface_alpha_term(
            source_matdef=source_matdef,
            source_density=source_density,
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            stopping_power_data_dir=stopping_power_data_dir,
            decay_data_dir=decay_data_dir
        )
        beam_results = Transport.beam_problem(
            interface_spectrum, config['target_matdef'],
            data_dir=None,
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            neutron_energy_bins=neutron_energy_bins,
            an_xs_data_dir=an_xs_data_dir,
            stopping_power_data_dir=stopping_power_data_dir,
            calculate_gammas=calculate_gammas,
            gamma_data_dir=gamma_data_dir
        )

        p_total = beam_results['neutron_yield']
        total_spectrum = beam_results['neutron_spectrum']
        neutron_energy_bins = beam_results['neutron_energy_bins']

        absolute_spectrum = None
        if total_spectrum is not None and p_total > 0:
            absolute_spectrum = total_spectrum * p_total

        results = {
            'an_yield': float(p_total),
            'an_spectrum': total_spectrum.tolist() if total_spectrum is not None else None,
            'an_spectrum_absolute': absolute_spectrum.tolist() if absolute_spectrum is not None else None,
            'neutron_energy_bins': neutron_energy_bins.tolist() if neutron_energy_bins is not None else None,
            'spectrum_energy_bins': neutron_energy_bins.tolist() if neutron_energy_bins is not None else None}

        if calculate_gammas and 'gamma_yield' in beam_results:
            results['gamma_yield'] = float(beam_results['gamma_yield'])
            results['gamma_lines'] = beam_results['gamma_lines']

        return results

    @staticmethod
    def _calculate_ab_spectrum_volumetric(
        source_matdef,
        source_density,
        stopping_power_data_dir=None,
        decay_data_dir=None,
        min_alpha_energy=1e-11,
        max_alpha_energy=15
    ):
        """
        Calculate alpha spectrum entering first layer from volumetric source emission.

        Computes hemisphere emission from source into first intermediate layer
        using volumetric integration with geometric factor 0.25.

        Args:
            source_matdef: dict - Source material definition {ZAID: mass_fraction}
            source_density: float - Source density (g/cm^3)
            stopping_power_data_dir: str, optional - Path to stopping power data
            decay_data_dir: str, optional - Path to decay data
            min_alpha_energy: float, optional - Minimum alpha energy (MeV, default: 1e-11)
            max_alpha_energy: float, optional - Maximum alpha energy (MeV, default: 15)

        Returns:
            dict - Sparse dictionary {energy_MeV: intensity} for AB alpha spectrum
        """

        source_mass_fractions, source_atom_fractions = matdef_to_zaids(
            source_matdef)
        source_avg_mass = sum(atomic_data.get_atomic_mass(zaid) * afrac
                              for zaid, afrac in source_atom_fractions.items())
        source_atom_density = source_density * AVOGADRO_NUM / source_avg_mass

        VOLUMETRIC_DEA = 0.016
        eamin = max(min_alpha_energy, 0.001)
        num_vol_groups = int(round((max_alpha_energy - eamin) / VOLUMETRIC_DEA))
        energy_grid = np.linspace(eamin, max_alpha_energy, num_vol_groups + 1)
        dea = (max_alpha_energy - eamin) / num_vol_groups

        source_stopping = get_composite_stopping(
            source_mass_fractions, stopping_power_data_dir)
        source_stopping = rebin_xs(
            source_stopping, energy_grid, extrapolate=True)

        scxa = np.array([source_stopping.get(e, 1e-30) for e in energy_grid])
        scxa = np.maximum(scxa, 1e-30)

        source_isotopes = []
        for zaid, faq in source_atom_fractions.items():
            alam, alpha_spectrum = get_decay_spectrum(
                zaid, data_dir=decay_data_dir)
            if alam > 0 and alpha_spectrum:
                source_isotopes.append({
                    'alam': alam,
                    'faq': faq,
                    'alpha_lines': alpha_spectrum
                })

        GEOMETRIC_FACTOR_AB = 0.25
        astab = np.zeros(num_vol_groups)

        for isotope in source_isotopes:
            fact = isotope['alam'] * isotope['faq'] * GEOMETRIC_FACTOR_AB
            ps = np.zeros(num_vol_groups)

            p = np.zeros(num_vol_groups)
            for m in range(num_vol_groups):
                p[m] = fact * (1.0 / scxa[m] + 1.0 / scxa[m + 1]) * dea / 2.0

            for eala, fal in isotope['alpha_lines']:
                eala = min(eala, max_alpha_energy)
                for m in range(num_vol_groups):
                    if eala >= energy_grid[m + 1]:
                        fm = 1.0
                    elif eala > energy_grid[m]:
                        fm = (eala - energy_grid[m]) / dea
                    else:
                        fm = 0.0
                    ps[m] += fal * p[m] * fm

            astab += ps

        return {
            (energy_grid[m] + energy_grid[m + 1]) / 2.0: astab[m]
            for m in range(num_vol_groups) if astab[m] > 0
        }

    @staticmethod
    def _calculate_itrans_single_layer(
        layer_stopping,
        layer_thickness,
        layer_atom_density,
        n_angular_bins,
        num_alpha_groups,
        dea
    ):
        """
        Calculate energy degradation matrix for single layer traversal.

        Computes exit energy groups for alphas traversing layer at different angles
        using range-energy relationships.

        Args:
            layer_stopping: ndarray - Stopping power at energy grid (MeV*cm^2)
            layer_thickness: float - Layer thickness (cm)
            layer_atom_density: float - Atom density (atoms/cm^3)
            n_angular_bins: int - Number of angular bins
            num_alpha_groups: int - Number of alpha energy groups
            dea: float - Alpha energy bin width (MeV)

        Returns:
            ndarray - Energy degradation matrix [n_angular_bins, num_alpha_groups]
                itrans[i_ang, ig] = exit energy group after traversal at angle i_ang
        """
        theta = np.linspace(0, np.pi / 2, n_angular_bins + 1)
        itrans = np.zeros((n_angular_bins, num_alpha_groups), dtype=int)

        safe_stopping = np.maximum(layer_stopping, 1e-30)
        inv_stopping = 1.0 / safe_stopping

        bin_ranges = (dea / 2.0) * \
            (inv_stopping[:-1] + inv_stopping[1:]) / layer_atom_density
        cum_range = np.concatenate(([0.0], np.cumsum(bin_ranges)))

        for i_ang in range(n_angular_bins):
            cos_theta_i = np.cos(theta[i_ang])
            if cos_theta_i <= 0:
                itrans[i_ang, :] = num_alpha_groups
                continue

            D = layer_thickness / cos_theta_i
            required_range = cum_range[:num_alpha_groups] + D
            indices = np.searchsorted(cum_range, required_range, side='left')
            itrans[i_ang, :] = np.minimum(indices - 1, num_alpha_groups)

        return itrans

    @staticmethod
    def _compose_itrans_matrices(itrans1, itrans2):
        """
        Compose two energy degradation matrices for sequential layer traversal.

        Combines degradation effects of two layers: alpha exits first layer at
        degraded energy, then exits second layer at further degraded energy.

        Args:
            itrans1: ndarray - First layer degradation matrix
            itrans2: ndarray - Second layer degradation matrix

        Returns:
            ndarray - Composed degradation matrix
                result[ang, ig] = itrans2[ang, itrans1[ang, ig]]
        """
        n_angular_bins, num_alpha_groups = itrans1.shape
        out_of_bounds = itrans1 >= num_alpha_groups
        clamped = np.minimum(itrans1, num_alpha_groups - 1)
        result = itrans2[np.arange(n_angular_bins)[:, np.newaxis], clamped]
        result[out_of_bounds] = num_alpha_groups - 1
        return result

    @staticmethod
    def _calculate_bc_spectrum_volumetric(
        source_matdef,
        source_density,
        intermediate_layers,
        stopping_power_data_dir=None,
        decay_data_dir=None,
        n_angular_bins=40,
        min_alpha_energy=1e-11,
        max_alpha_energy=15
    ):
        """
        Calculate alpha spectrum entering target after degradation through all intermediate layers.

        Computes volumetric integration with angular dependence and sequential energy
        degradation through multiple layers using geometric factor 0.125.

        Args:
            source_matdef: dict - Source material definition {ZAID: mass_fraction}
            source_density: float - Source density (g/cm^3)
            intermediate_layers: list - Layer dicts containing:
                - 'matdef': dict - Layer material definition
                - 'density': float - Layer density (g/cm^3)
                - 'thickness': float - Layer thickness (cm)
            stopping_power_data_dir: str, optional - Path to stopping power data
            decay_data_dir: str, optional - Path to decay data
            n_angular_bins: int, optional - Number of angular bins (default: 40)
            min_alpha_energy: float, optional - Minimum alpha energy (MeV, default: 1e-11)
            max_alpha_energy: float, optional - Maximum alpha energy (MeV, default: 15)

        Returns:
            dict - Sparse dictionary {energy_MeV: intensity} for BC alpha spectrum
        """

        source_mass_fractions, source_atom_fractions = matdef_to_zaids(
            source_matdef)
        source_avg_mass = sum(atomic_data.get_atomic_mass(zaid) * afrac
                              for zaid, afrac in source_atom_fractions.items())
        source_atom_density = source_density * AVOGADRO_NUM / source_avg_mass

        VOLUMETRIC_DEA = 0.016
        eamin = max(min_alpha_energy, 0.001)
        num_vol_groups = int(round((max_alpha_energy - eamin) / VOLUMETRIC_DEA))
        energy_grid = np.linspace(eamin, max_alpha_energy, num_vol_groups + 1)
        dea = (max_alpha_energy - eamin) / num_vol_groups

        source_stopping = get_composite_stopping(
            source_mass_fractions, stopping_power_data_dir)
        source_stopping = rebin_xs(
            source_stopping, energy_grid, extrapolate=True)

        scxa = np.array([source_stopping.get(e, 1e-30) for e in energy_grid])
        scxa = np.maximum(scxa, 1e-30)

        theta = np.linspace(0, np.pi / 2, n_angular_bins + 1)

        itrans_cumulative = np.zeros(
            (n_angular_bins, num_vol_groups), dtype=int)
        for i_ang in range(n_angular_bins):
            itrans_cumulative[i_ang, :] = np.arange(num_vol_groups)

        for layer in intermediate_layers:
            layer_matdef = layer['matdef']
            layer_density = layer['density']
            layer_thickness = layer['thickness']

            layer_mass_fractions, layer_atom_fractions = matdef_to_zaids(
                layer_matdef)
            layer_avg_mass = sum(
                atomic_data.get_atomic_mass(zaid) * afrac for zaid,
                afrac in layer_atom_fractions.items())
            layer_atom_density = layer_density * AVOGADRO_NUM / layer_avg_mass

            layer_stopping = get_composite_stopping(
                layer_mass_fractions, stopping_power_data_dir)
            layer_stopping = rebin_xs(
                layer_stopping, energy_grid, extrapolate=True)

            layer_stopping_array = np.array(
                [layer_stopping.get(e, 1e-30) for e in energy_grid])
            layer_stopping_array = np.maximum(layer_stopping_array, 1e-30)

            layer_itrans = Transport._calculate_itrans_single_layer(
                layer_stopping_array,
                layer_thickness,
                layer_atom_density,
                n_angular_bins,
                num_vol_groups,
                dea
            )

            itrans_cumulative = Transport._compose_itrans_matrices(
                itrans_cumulative, layer_itrans)

        source_isotopes = []
        for zaid, faq in source_atom_fractions.items():
            alam, alpha_spectrum = get_decay_spectrum(
                zaid, data_dir=decay_data_dir)
            if alam > 0 and alpha_spectrum:
                source_isotopes.append({
                    'alam': alam,
                    'faq': faq,
                    'alpha_lines': alpha_spectrum
                })

        GEOMETRIC_FACTOR_BC = 0.125
        astbc = np.zeros(num_vol_groups)

        d_omega = np.cos(2 * theta[:-1]) - np.cos(2 * theta[1:])

        p_base = (1.0 / scxa[:-1] + 1.0 / scxa[1:]) * (dea / 2.0)

        for isotope in source_isotopes:
            fact = isotope['alam'] * isotope['faq'] * GEOMETRIC_FACTOR_BC
            p_iso = fact * p_base

            for eala, fal in isotope['alpha_lines']:
                eala = min(eala, max_alpha_energy)
                fm = np.zeros(num_vol_groups)
                idx = np.searchsorted(energy_grid, eala) - 1

                if idx < 0:
                    continue

                fm[:idx] = 1.0
                if idx < num_vol_groups:
                    fm[idx] = (eala - energy_grid[idx]) / dea

                source_term = fal * p_iso * fm
                safe_indices = np.minimum(itrans_cumulative, num_vol_groups)
                req_energies = energy_grid[safe_indices]
                can_penetrate = (eala >= req_energies)
                angular_factor = np.sum(
                    d_omega[:, np.newaxis] * can_penetrate, axis=0)
                astbc += source_term * angular_factor

        return {
            (energy_grid[m] + energy_grid[m + 1]) / 2.0: astbc[m]
            for m in range(num_vol_groups) if astbc[m] > 0
        }

    @staticmethod
    def _calculate_sandwich(config: dict) -> dict:
        """
        Calculate neutron production from multi-layer sandwich geometry A|B1|B2|...|Bn|C.

        Uses volumetric formulation with exact interface treatment and sequential
        energy degradation through intermediate layers.

        Args:
            config: dict - Configuration containing:
                - source_matdef: dict - Source material (Region A)
                - source_density: float - Source density (g/cm^3)
                - intermediate_layers: list - Layer dicts with matdef, density, thickness
                - target_matdef: dict - Target material (Region C)
                - num_alpha_groups: int, optional - Alpha energy groups (default: 400)
                - min_alpha_energy: float, optional - Min alpha energy (MeV, default: 1e-7)
                - max_alpha_energy: float, optional - Max alpha energy (MeV, default: 6.5)
                - n_angular_bins: int, optional - Angular bins (default: 40)
                - neutron_energy_bins: ndarray, optional - Neutron spectrum grid
                - an_xs_data_dir, stopping_power_data_dir, decay_data_dir: str, optional

        Returns:
            dict - Results containing:
                - an_yield: float - Total neutron yield (n/s/cm^2)
                - yield_target: float - Target contribution (n/s/cm^2)
                - yield_layers: list - Per-layer breakdown (n/s/cm^2)
                - yield_ab_b, yield_bc_b, yield_bc_c: float - Individual terms (n/s/cm^2)
                - an_spectrum: list - Normalized spectrum
                - an_spectrum_absolute: list - Absolute spectrum
                - neutron_energy_bins: list
                - spectrum_layers: list - Per-layer spectra
        """
        an_xs_data_dir = config.get('an_xs_data_dir')
        stopping_power_data_dir = config.get('stopping_power_data_dir')
        decay_data_dir = config.get('decay_data_dir')
        n_angular_bins = config.get('n_angular_bins', 40)
        neutron_energy_bins = config.get('neutron_energy_bins')

        calculate_gammas = config.get('calculate_gammas', True)
        gamma_data_dir = config.get('gamma_data_dir')

        num_alpha_groups = config.get('num_alpha_groups', 15000)
        min_alpha_energy = config.get('min_alpha_energy', 1e-11)
        max_alpha_energy = config.get('max_alpha_energy', 15)

        source_matdef = config['source_matdef']
        source_density = config['source_density']
        target_matdef = config['target_matdef']
        intermediate_layers = config['intermediate_layers']

        first_layer = intermediate_layers[0]
        last_layer = intermediate_layers[-1]

        ab_alpha_dict = Transport._calculate_ab_spectrum_volumetric(
            source_matdef=source_matdef,
            source_density=source_density,
            stopping_power_data_dir=stopping_power_data_dir,
            decay_data_dir=decay_data_dir,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy
        )
        ab_alpha_list = [[energy, intensity]
                         for energy, intensity in ab_alpha_dict.items()]

        bc_alpha_dict = Transport._calculate_bc_spectrum_volumetric(
            source_matdef=source_matdef,
            source_density=source_density,
            intermediate_layers=intermediate_layers,
            stopping_power_data_dir=stopping_power_data_dir,
            decay_data_dir=decay_data_dir,
            n_angular_bins=n_angular_bins,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy
        )
        bc_alpha_list = [[energy, intensity]
                         for energy, intensity in bc_alpha_dict.items()]

        results_ab_b = Transport.beam_problem(
            ab_alpha_list, first_layer['matdef'],
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            neutron_energy_bins=neutron_energy_bins,
            an_xs_data_dir=an_xs_data_dir,
            stopping_power_data_dir=stopping_power_data_dir,
            calculate_gammas=calculate_gammas,
            gamma_data_dir=gamma_data_dir
        )
        yield_ab_b = results_ab_b['neutron_yield']
        spectrum_ab_b = results_ab_b['neutron_spectrum']
        en_bins = results_ab_b['neutron_energy_bins']
        gamma_yield_ab_b = results_ab_b.get('gamma_yield', 0.0)
        gamma_lines_ab_b = results_ab_b.get('gamma_lines', [])

        results_bc_b = Transport.beam_problem(
            bc_alpha_list, last_layer['matdef'],
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            neutron_energy_bins=neutron_energy_bins,
            an_xs_data_dir=an_xs_data_dir,
            stopping_power_data_dir=stopping_power_data_dir,
            calculate_gammas=calculate_gammas,
            gamma_data_dir=gamma_data_dir
        )
        yield_bc_b = results_bc_b['neutron_yield']
        spectrum_bc_b = results_bc_b['neutron_spectrum']
        gamma_yield_bc_b = results_bc_b.get('gamma_yield', 0.0)
        gamma_lines_bc_b = results_bc_b.get('gamma_lines', [])

        results_bc_c = Transport.beam_problem(
            bc_alpha_list, target_matdef,
            num_alpha_groups=num_alpha_groups,
            min_alpha_energy=min_alpha_energy,
            max_alpha_energy=max_alpha_energy,
            neutron_energy_bins=neutron_energy_bins,
            an_xs_data_dir=an_xs_data_dir,
            stopping_power_data_dir=stopping_power_data_dir,
            calculate_gammas=calculate_gammas,
            gamma_data_dir=gamma_data_dir
        )
        yield_bc_c = results_bc_c['neutron_yield']
        spectrum_bc_c = results_bc_c['neutron_spectrum']
        gamma_yield_bc_c = results_bc_c.get('gamma_yield', 0.0)
        gamma_lines_bc_c = results_bc_c.get('gamma_lines', [])

        yield_per_layer = []
        spectrum_per_layer = []
        gamma_yield_per_layer = []
        gamma_lines_per_layer = []

        for i in range(len(intermediate_layers)):
            layer_matdef = intermediate_layers[i]['matdef']

            if i == 0:
                yield_entering = yield_ab_b
                spectrum_entering = spectrum_ab_b
                gamma_yield_entering = gamma_yield_ab_b
                gamma_lines_entering = gamma_lines_ab_b
            else:
                entering_alpha_dict = Transport._calculate_bc_spectrum_volumetric(
                    source_matdef=source_matdef,
                    source_density=source_density,
                    intermediate_layers=intermediate_layers[:i],
                    stopping_power_data_dir=stopping_power_data_dir,
                    decay_data_dir=decay_data_dir,
                    n_angular_bins=n_angular_bins,
                    min_alpha_energy=min_alpha_energy,
                    max_alpha_energy=max_alpha_energy
                )
                entering_alpha_list = [[e, inten]
                                       for e, inten in entering_alpha_dict.items()]
                results_entering = Transport.beam_problem(
                    entering_alpha_list, layer_matdef,
                    num_alpha_groups=num_alpha_groups,
                    min_alpha_energy=min_alpha_energy,
                    max_alpha_energy=max_alpha_energy,
                    neutron_energy_bins=neutron_energy_bins,
                    an_xs_data_dir=an_xs_data_dir,
                    stopping_power_data_dir=stopping_power_data_dir,
                    calculate_gammas=calculate_gammas,
                    gamma_data_dir=gamma_data_dir
                )
                yield_entering = results_entering['neutron_yield']
                spectrum_entering = results_entering['neutron_spectrum']
                gamma_yield_entering = results_entering.get('gamma_yield', 0.0)
                gamma_lines_entering = results_entering.get('gamma_lines', [])

            if i == len(intermediate_layers) - 1:
                yield_exiting = yield_bc_b
                spectrum_exiting = spectrum_bc_b
                gamma_yield_exiting = gamma_yield_bc_b
                gamma_lines_exiting = gamma_lines_bc_b
            else:
                exiting_alpha_dict = Transport._calculate_bc_spectrum_volumetric(
                    source_matdef=source_matdef,
                    source_density=source_density,
                    intermediate_layers=intermediate_layers[:i + 1],
                    stopping_power_data_dir=stopping_power_data_dir,
                    decay_data_dir=decay_data_dir,
                    n_angular_bins=n_angular_bins,
                    min_alpha_energy=min_alpha_energy,
                    max_alpha_energy=max_alpha_energy
                )
                exiting_alpha_list = [[e, inten]
                                      for e, inten in exiting_alpha_dict.items()]
                results_exiting = Transport.beam_problem(
                    exiting_alpha_list, layer_matdef,
                    num_alpha_groups=num_alpha_groups,
                    min_alpha_energy=min_alpha_energy,
                    max_alpha_energy=max_alpha_energy,
                    neutron_energy_bins=neutron_energy_bins,
                    an_xs_data_dir=an_xs_data_dir,
                    stopping_power_data_dir=stopping_power_data_dir,
                    calculate_gammas=calculate_gammas,
                    gamma_data_dir=gamma_data_dir
                )
                yield_exiting = results_exiting['neutron_yield']
                spectrum_exiting = results_exiting['neutron_spectrum']
                gamma_yield_exiting = results_exiting.get('gamma_yield', 0.0)
                gamma_lines_exiting = results_exiting.get('gamma_lines', [])

            net_yield_in_layer = yield_entering - yield_exiting
            yield_per_layer.append(net_yield_in_layer)

            if spectrum_entering is not None and spectrum_exiting is not None:
                abs_entering = spectrum_entering * yield_entering
                abs_exiting = spectrum_exiting * yield_exiting
                net_spectrum_in_layer = abs_entering - abs_exiting
                spectrum_per_layer.append(net_spectrum_in_layer)
            else:
                spectrum_per_layer.append(None)

            net_gamma_yield = gamma_yield_entering - gamma_yield_exiting
            gamma_yield_per_layer.append(net_gamma_yield)

            dict_entering = dict(gamma_lines_entering)
            dict_exiting = dict(gamma_lines_exiting)
            net_lines_dict = defaultdict(float)
            all_energies = set(dict_entering.keys()) | set(dict_exiting.keys())
            for e in all_energies:
                net = dict_entering.get(e, 0.0) - dict_exiting.get(e, 0.0)
                if net > 1e-20:
                    net_lines_dict[e] = net
            gamma_lines_per_layer.append(_gamma_line_pairs(sorted(net_lines_dict.items())))

        an_yield = yield_bc_c + sum(yield_per_layer)

        total_spectrum = None
        if spectrum_bc_c is not None:
            total_spectrum = spectrum_bc_c * yield_bc_c  # Convert to absolute

        for s in spectrum_per_layer:
            if s is not None:
                if total_spectrum is None:
                    total_spectrum = s.copy()
                else:
                    total_spectrum += s

        normalized_spectrum = None
        if total_spectrum is not None and np.sum(total_spectrum) > 0:
            normalized_spectrum = total_spectrum / np.sum(total_spectrum)

        gamma_yield_total = None

        if calculate_gammas:
            gamma_yield_total = gamma_yield_bc_c + sum(gamma_yield_per_layer)

        results = {
            'an_yield': float(an_yield),
            'yield_target': float(yield_bc_c),
            'yield_layers': [
                float(y) for y in yield_per_layer],
            'yield_ab_b': float(yield_ab_b),
            'yield_bc_b': float(yield_bc_b),
            'yield_bc_c': float(yield_bc_c),
            'an_spectrum': normalized_spectrum.tolist() if normalized_spectrum is not None else None,
            'an_spectrum_absolute': total_spectrum.tolist() if total_spectrum is not None else None,
            'spectrum_energy_bins': en_bins.tolist() if en_bins is not None else None,
            'neutron_energy_bins': en_bins.tolist() if en_bins is not None else None,
            'spectrum_layers': [
                s.tolist() if s is not None else None for s in spectrum_per_layer]}

        if calculate_gammas:
            results['gamma_yield'] = float(gamma_yield_total)
            results['gamma_yield_target'] = float(gamma_yield_bc_c)
            results['gamma_yield_layers'] = [float(gy) for gy in gamma_yield_per_layer]
            results['gamma_lines_target'] = _gamma_line_pairs(gamma_lines_bc_c)

            final_lines_dict = defaultdict(float)
            for e, i in gamma_lines_bc_c:
                final_lines_dict[e] += i

            for layer_lines in gamma_lines_per_layer:
                for e, i in layer_lines:
                    final_lines_dict[e] += i

            results['gamma_lines'] = _gamma_line_pairs(sorted(final_lines_dict.items()))

        return results

    @staticmethod
    def get_bulk_alpha_source(matdef, decay_data_dir=None):
        """
        Extract alpha decay spectrum from material composition.

        Calculates combined alpha emission spectrum from all alpha-emitting isotopes
        in material based on decay data.

        Args:
            matdef: dict - Material definition {ZAID: mass_fraction}
            decay_data_dir: str, optional - Path to decay data directory

        Returns:
            list - [(energy_MeV, intensity_per_s_per_g), ...] pairs for alpha source spectrum
        """

        mass_fractions, atom_fractions = matdef_to_zaids(matdef)

        spectrum = defaultdict(float)

        for zaid, wtfrac in mass_fractions.items():
            atomic_mass = atomic_data.get_atomic_mass(zaid)
            atoms_per_gram = AVOGADRO_NUM / atomic_mass * wtfrac

            a_per_sec_per_atom, alpha_spectrum = get_decay_spectrum(
                zaid, data_dir=decay_data_dir)

            for line in alpha_spectrum:
                e = line[0]
                i = line[1] * a_per_sec_per_atom * atoms_per_gram
                spectrum[e] += i

        energies = [[e, float(i)] for e, i in spectrum.items()]
        return energies
