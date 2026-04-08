import xml.etree.ElementTree as ET
import numpy as np
import os
import math
import re
import yaml
import logging
from collections import defaultdict
from typing import Optional, Dict, Tuple, List
from scipy.interpolate import interp1d

from .data_manager import get_data_dir
from .sources_parsers import (
    get_sources_an_xs,
    get_sources_stopping_power,
    get_sources_branching_info,
    get_sources_decay_data)
from .constants import ALPH_MASS, AMU_TO_MEV, ANEUT_MASS, AVOGADRO_NUM, ZALP, ALPH
from .atomic_data_loader import atomic_data
from .atomic_data_loader import get_atomic_mass as get_atomic_mass_from_db
from .data_manager import get_data_dir

logger = logging.getLogger(__name__)


def _default_data_root():
    return str(get_data_dir())

def _load_sources_overrides():
    """Load the sources_overrides.yaml configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    overrides_path = os.path.join(current_dir, "data", "sources_overrides.yaml")

    if os.path.exists(overrides_path):
        try:
            with open(overrides_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load sources_overrides.yaml: {e}")
            return None
    return None


_SOURCES_OVERRIDES = _load_sources_overrides()


def _should_use_sources_for_an_xs(zaid: int) -> bool:
    """Check if a ZAID should use SOURCES data for (alpha,n) cross sections."""
    if _SOURCES_OVERRIDES is None:
        return False

    use_sources_zaids = _SOURCES_OVERRIDES.get('an_xs', {}).get('use_sources_tape_zaids', [])
    return zaid in use_sources_zaids


def _should_use_sources_for_stopping(zaid: int) -> bool:
    """Check if a ZAID should use SOURCES data for stopping power."""
    if _SOURCES_OVERRIDES is None:
        return False

    z = zaid // 1000
    default_z_threshold = _SOURCES_OVERRIDES.get('stopping', {}).get('default_sources_for_z_gt', 999)
    return z > default_z_threshold


def _get_sources_an_xs_dir() -> Optional[str]:
    """Get the directory for SOURCES (alpha,n) cross section data."""
    if _SOURCES_OVERRIDES is None:
        return None

    tape_path = _SOURCES_OVERRIDES.get('an_xs', {}).get('tape')
    if tape_path:
        return os.path.join(_default_data_root(), os.path.dirname(tape_path))
    return None


def _get_sources_stopping_dir() -> Optional[str]:
    """Get the directory for SOURCES stopping power data."""
    if _SOURCES_OVERRIDES is None:
        return None

    tape_path = _SOURCES_OVERRIDES.get('stopping', {}).get('tape')
    if tape_path:
        return os.path.join(_default_data_root(), os.path.dirname(tape_path))
    return None


def _get_endf_filename(zaid: int) -> str:
    """Convert ZAID to ENDF filename format (a-ZZZ_Element_AAA.endf.gnds.xml)."""
    z = zaid // 1000
    a = zaid % 1000

    symbol = atomic_data.get_element_symbol(z)

    return f"a-{z:03d}_{symbol}_{a:03d}.endf.gnds.xml"


def _find_gnds_xml(zaid: int, data_dir: Optional[os.PathLike]) -> Optional[str]:
    """Return the path to the GNDS XML file for zaid, or None if not found.

    Handles default ENDF/JENDL/TENDL directory layouts and TENDL filename variants.

    Args:
        zaid: ZAID in ZZZAAA format
        data_dir: Directory to search. None uses the default ENDF/JENDL/TENDL hierarchy.

    Returns:
        Absolute path to the first matching file, or None.
    """
    z = zaid // 1000
    a = zaid % 1000
    symbol = atomic_data.get_element_symbol(z)

    if data_dir is None:
        data_root = _default_data_root()
        candidates = [
            os.path.join(data_root, 'an_xs', "ENDF", _get_endf_filename(zaid)),
            os.path.join(data_root, 'an_xs', "JENDL", f'{zaid}.xml'),
            os.path.join(data_root, 'an_xs', "TENDL", f'{zaid}.xml'),
        ]
    else:
        try:
            data_dir_str = str(data_dir).lower()
        except (TypeError, AttributeError):
            data_dir_str = str(data_dir)
        if "tendl-" in data_dir_str or "tendl" in data_dir_str:
            candidates = [
                os.path.join(data_dir, f"a-{symbol}{a:03d}.tendl.gnds.xml"),
                os.path.join(data_dir, f"a_{z:03d}-{symbol}-{a:03d}.xml"),
                os.path.join(data_dir, f"{symbol.upper()}{a:03d}.xml"),
                os.path.join(data_dir, f"{symbol.capitalize()}{a:03d}.xml"),
                os.path.join(data_dir, f"{symbol}{a:03d}.xml"),
                os.path.join(data_dir, f"{zaid}.xml"),
            ]
        elif "endf" in data_dir_str:
            candidates = [os.path.join(data_dir, _get_endf_filename(zaid))]
        else:
            candidates = [os.path.join(data_dir, f'{zaid}.xml')]

    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def get_an_xs(
        zaid: int, data_dir: Optional[os.PathLike] = None) -> Optional[Dict[float, float]]:
    """
    Get the (a,n) reaction cross section for a given ZAID.

    Args:
        zaid: ZAID of the nucleus (ZZZAAA format),
        data_dir: [Optional: Defaults to JENDL/TENDL data] Directory containing the data.

    Returns:
        {energy (MeV): cross section (barns)}: (a,n) reaction cross section,
        None: If the cross section data or file is not found.

    Raises:
        ValueError: If the ZAID is not a valid ZZZAAA formatted ZAID,
        FileNotFoundError: If no cross section files are found in the data directory,
        ValueError: If the cross section data format is not supported.
    """

    if zaid >= 1e6:
        raise ValueError(f"ZAID {zaid} is not a valid ZZZAAA formatted ZAID.")

    z = zaid // 1000
    a = zaid % 1000
    symbol = atomic_data.get_element_symbol(z)

    if data_dir is None and _should_use_sources_for_an_xs(zaid):
        sources_dir = _get_sources_an_xs_dir()
        if sources_dir:
            logger.info(f"Using SOURCES data for ZAID {zaid} based on sources_overrides.yaml")
            return get_sources_an_xs(z, a, symbol, sources_dir)

    if data_dir == "sources" or (data_dir is not None and "sources" in str(
            data_dir)):
        return get_sources_an_xs(z, a, symbol, data_dir)

    found_path = _find_gnds_xml(zaid, data_dir)

    if found_path is None and data_dir is not None:
        try:
            data_dir_str = str(data_dir).lower()
        except (TypeError, AttributeError):
            data_dir_str = str(data_dir)
        if "tendl-" in data_dir_str or "tendl" in data_dir_str:
            try:
                for fname in os.listdir(data_dir):
                    if not fname.lower().endswith('.xml'):
                        continue
                    if symbol.lower() in fname.lower() and f"{a:03d}" in fname:
                        found_path = os.path.join(data_dir, fname)
                        break
            except OSError:
                pass

    if found_path is None:
        if data_dir is not None:
            try:
                data_dir_str = str(data_dir).lower()
            except (TypeError, AttributeError):
                data_dir_str = str(data_dir)
            if "tendl-" not in data_dir_str and "tendl" not in data_dir_str and "endf" not in data_dir_str:
                return _get_an_xs_xml(os.path.join(data_dir, f"{zaid}.xml"))
        return None

    return _get_an_xs_xml(found_path)


def get_stopping_power(
        zaid: int, data_dir: Optional[os.PathLike] = None) -> Dict[float, float]:
    """
    Get the stopping power for a given ZAID.

    Args:
        zaid: ZAID of the nucleus (ZZZAAA format),
        data_dir: [Optional: Defaults to ASTAR/SRIM data] Directory containing the data.

    Returns:
        Dictionary {energy (MeV): total stopping power (MeV cm^2)}: Stopping power.

    Raises:
        ValueError: If the ZAID is not a valid ZZZAAA formatted ZAID,
        ValueError: If the data directory is not supported.
    """

    if zaid >= 1e6:
        raise ValueError(f"ZAID {zaid} is not a valid ZZZAAA formatted ZAID.")

    atomic_mass = atomic_data.get_atomic_mass(zaid)

    if data_dir is None and (_should_use_sources_for_an_xs(zaid) or _should_use_sources_for_stopping(zaid)):
        sources_dir = _get_sources_stopping_dir()
        if sources_dir:
            logger.info(f"Using SOURCES stopping power for ZAID {zaid} based on sources_overrides.yaml")
            return get_sources_stopping_power(zaid, sources_dir, atomic_mass=atomic_mass)

    if data_dir == "sources" or (
            data_dir is not None and "sources" in str(data_dir)):
        return get_sources_stopping_power(
            zaid, data_dir, atomic_mass=atomic_mass)

    z = zaid // 1000
    if data_dir is None:
        data_root = _default_data_root()
        if z > 92:
            sources_dir = os.path.join(
                data_root, "stopping", "sources")
            if os.path.exists(sources_dir):
                return get_sources_stopping_power(
                    zaid, sources_dir, atomic_mass=atomic_mass)

        astar_path = os.path.join(
            data_root, "stopping", "ASTAR", f"{z}.txt")
        srim_path = os.path.join(
            data_root, "stopping", "SRIM", f"{z}.txt")
        if os.path.exists(astar_path):
            return _get_stopping_power_astar(astar_path, atomic_mass)
        elif os.path.exists(srim_path):
            return _get_stopping_power_srim(srim_path, atomic_mass)
        else:
            logger.warning(f"No stopping power data found for ZAID {zaid}")
            return {}
    else:
        return _get_stopping_power_detect_format(zaid, data_dir)


def get_branching_info(zaid: int,
                       data_dir: Optional[os.PathLike] = None) -> Optional[Tuple[float,
                                                                                 List[Tuple[float,
                                                                                            float]]]]:
    """
    Get the branching ratios and Q-values for a given ZAID.

    Args:
        source_zaid: ZAID of the source nucleus (ZZZAAA format),
        target_zaid: ZAID of the target nucleus (ZZZAAA format),
        data_dir: [Optional: Defaults to ENDF data] Directory containing the data.

    Returns:
        q_value: Q-value for the alpha decay (MeV),
        level_energies: List of level energies (MeV),
        branching_data: Dictionary of branching fractions {energy: branching_fractions}

    Raises:
        ValueError: If the ZAID is not a valid ZZZAAA formatted ZAID,
        FileNotFoundError: If no branching data is found in the data directory,
        ValueError: If the branching data format is not supported.
    """

    if zaid >= 1e6:
        raise ValueError(f"ZAID {zaid} is not a valid ZZZAAA formatted ZAID.")

    z = zaid // 1000
    a = zaid % 1000
    symbol = atomic_data.get_element_symbol(z)

    if data_dir is None and _should_use_sources_for_an_xs(zaid):
        sources_dir = _get_sources_an_xs_dir()
        if sources_dir:
            logger.info(f"Using SOURCES branching data for ZAID {zaid} based on sources_overrides.yaml")
            s4c_key = f"{z:04d}{a*10:04d}"
            return get_sources_branching_info(s4c_key, sources_dir)

    if data_dir == "sources" or (
            data_dir is not None and "sources" in str(data_dir)):
        s4c_key = f"{z:04d}{a*10:04d}"
        return get_sources_branching_info(s4c_key, data_dir)

    found_path = _find_gnds_xml(zaid, data_dir)

    if not found_path:
        logger.warning(
            f"(a,n) cross sections file not found for {zaid}, cannot compute branching fractions.")
        return {}, {}, 0.0

    tree = ET.parse(found_path)
    root = tree.getroot()

    if "tendl" in found_path.lower():
        tendl_branch = _get_tendl_branching_info(root, zaid)
        if tendl_branch is not None:
            return tendl_branch

    try:
        level_energies, level_cross_sections, q_value = _get_endf_level_data(
            root)
    except (KeyError, ValueError, ET.ParseError):
        level_energies, level_cross_sections, q_value = {}, {}, 0.0

    if not level_energies or not level_cross_sections:
        tendl_path = os.path.join(os.path.dirname(
            found_path), "TENDL-2023", f"a-{symbol}{a:03d}.tendl.gnds.xml")
        if os.path.exists(tendl_path):
            try:
                ttree = ET.parse(tendl_path)
                troot = ttree.getroot()
                tendl_branch = _get_tendl_branching_info(troot, zaid)
                if tendl_branch is not None:
                    return tendl_branch
            except Exception:
                pass

        if found_path and os.path.exists(found_path):
            logger.info(
                f"JENDL file {found_path} exists but lacks level data (MT=50-59). Using default ground-state branching.")

            energy_grid = []
            for reaction in root.findall(".//reaction"):
                xys = reaction.find(".//crossSection//XYs1d")
                if xys is not None and xys.find("values") is not None:
                    txt = xys.find("values").text or ""
                    try:
                        arr = np.array([float(x) for x in txt.split()])
                        if arr.size >= 2 and arr.size % 2 == 0:
                            energies_eV = arr[0::2]
                            energy_grid = energies_eV / 1e6
                            break
                    except Exception:
                        continue

            if energy_grid is None or len(energy_grid) == 0:
                energy_grid = np.arange(0.1, 15.1, 0.1)

            level_energies = {0: 0.0}
            level_cross_sections = {0: {float(e): 1.0 for e in energy_grid}}
            q_value = 0.0

            branching_data = {float(e): [1.0] for e in energy_grid}
            return q_value, [0.0], branching_data

        raise ValueError(f"No level data found in ENDF file {found_path}")

    branching_data = _calculate_branching_fractions(level_cross_sections)

    sorted_levels = sorted(level_energies.items())
    level_energies_list = [energy for _, energy in sorted_levels]

    num_energy_levels = len(level_energies_list)

    for energy_key in list(branching_data.keys()):
        arr = branching_data[energy_key]
        try:
            arr_np = np.asarray(arr, dtype=float)
        except Exception:
            branching_data.pop(energy_key, None)
            continue
        if arr_np.ndim == 0:
            arr_np = np.array([float(arr_np)])
        arr_np = arr_np[:num_energy_levels]
        if arr_np.size < num_energy_levels:
            arr_np = np.pad(arr_np, (0, num_energy_levels -
                            arr_np.size), mode='constant', constant_values=0.0)
        branching_data[energy_key] = arr_np.tolist()

    if not branching_data:
        return q_value, level_energies_list, {0.0: [1.0]}

    try:
        z = zaid // 1000
        if z == 12:
            energies = sorted(branching_data.keys(),
                              key=lambda e: abs(float(e) - 5.0))
            if energies:
                e_key = energies[0]
                fractions = np.asarray(branching_data[e_key], dtype=float)
                num_levels = len(level_energies_list)
                preview = min(5, num_levels)
                le_preview = ", ".join(
                    f"{level_energies_list[i]:.3f}" for i in range(preview))
                fr_preview = ", ".join(
                    f"{fractions[i]:.3f}" for i in range(preview))

    except Exception:
        pass

    return q_value, level_energies_list, branching_data


def _get_ground_state_cascade(level_energies: List[float]) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Create fallback gamma cascade assuming all levels decay directly to ground state.

    This is a simplified physics model used when detailed gamma cascade data is unavailable.
    Each excited level is assumed to emit a single gamma ray of energy E_gamma = E_level
    and transition directly to the ground state with 100% branching ratio.

    Args:
        level_energies: List of nuclear level energies in MeV (index 0 is ground state at 0.0 MeV)

    Returns:
        Dictionary mapping level index to list of transitions:
        {level_idx: [(final_level_idx, gamma_energy_MeV, transition_probability), ...]}
    """
    cascades = {}

    for i, energy in enumerate(level_energies):
        if i == 0:
            cascades[0] = []
            continue

        if energy > 0:
            cascades[i] = [(0, energy, 1.0)]
        else:
            cascades[i] = []

    return cascades


def _parse_endf_gamma_cascades(filepath: os.PathLike, level_energies: List[float]) -> Optional[Dict[int, List[Tuple[int, float, float]]]]:
    """
    Parse gamma cascade data from ENDF/ENSDF nuclear structure files.
    

    Args:
        filepath: Path to ENDF/ENSDF XML file
        level_energies: List of level energies to validate against (MeV)

    Returns:
        Dictionary mapping level index to gamma transitions, or None if parsing fails:
        {level_idx: [(final_level_idx, gamma_energy_MeV, transition_probability), ...]}
    """
    try:
        root = ET.parse(filepath).getroot()
    except Exception as e:
        logger.debug(f"Failed to parse ENSDF/ENDF gamma cascades for {filepath}: {e}")
        return None

    num_levels = len(level_energies)
    cascades: Dict[int, List[Tuple[int, float, float]]] = {
        idx: [] for idx in range(num_levels)
    }
    level_energy_map = {round(energy, 6): idx for idx, energy in enumerate(level_energies)}
    max_level_energy = max(level_energies) if level_energies else 0.0

    def _parse_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
            return float(match.group(0)) if match else None

    def _energy_to_mev(value: Optional[float], unit: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        if unit:
            unit_norm = unit.strip().lower()
            if unit_norm in ("ev", "electronvolt", "electronvolts"):
                return value / 1e6
            if unit_norm in ("kev", "kiloelectronvolt", "kiloelectronvolts"):
                return value / 1e3
            if unit_norm in ("mev", "megaelectronvolt", "megaelectronvolts"):
                return value
            if unit_norm in ("gev", "gigaelectronvolt", "gigaelectronvolts"):
                return value * 1e3
        if max_level_energy > 0.0:
            if value <= max_level_energy * 10.0:
                return value
            if value <= max_level_energy * 1e4:
                return value / 1e3
        return value / 1e6 if value > 1e3 else value

    def _find_index_by_energy(energy_mev: Optional[float]) -> Optional[int]:
        if energy_mev is None:
            return None
        rounded = round(energy_mev, 6)
        mapped = level_energy_map.get(rounded)
        if mapped is not None:
            return mapped
        best_idx = None
        best_diff = None
        for idx, energy in enumerate(level_energies):
            diff = abs(energy - energy_mev)
            if best_diff is None or diff < best_diff:
                best_idx = idx
                best_diff = diff
        if best_diff is not None and best_diff <= 1e-3:
            return best_idx
        return None

    def _parse_level_ref(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        value_str = value.strip()
        if value_str.lower() in ("gs", "g.s.", "ground", "groundstate", "ground_state"):
            return 0
        if value_str.isdigit():
            return int(value_str)
        match = re.search(r"(?:_e)?(\d+)", value_str)
        return int(match.group(1)) if match else None

    def _extract_value_and_unit(elem, name: str) -> Tuple[Optional[str], Optional[str]]:
        for child in elem:
            if child.tag.rsplit('}', 1)[-1] == name:
                value_attr = child.get("value")
                if value_attr:
                    return value_attr, child.get("unit")
                if child.text and child.text.strip():
                    return child.text.strip(), child.get("unit")
                for sub in child:
                    if sub.tag.rsplit('}', 1)[-1] == "double":
                        sub_value = sub.get("value")
                        if sub_value:
                            return sub_value, sub.get("unit") or child.get("unit")
        return None, None

    level_elem_indices = {}
    for level_elem in root.iter():
        if level_elem.tag.rsplit('}', 1)[-1] != "level":
            continue
        level_idx = _parse_level_ref(level_elem.get("index") or level_elem.get("number") or level_elem.get("id"))
        energy_val = _parse_float(level_elem.get("energy"))
        energy_unit = level_elem.get("unit") or level_elem.get("energyUnit")
        if energy_val is None:
            energy_val_str, energy_unit = _extract_value_and_unit(level_elem, "energy")
            energy_val = _parse_float(energy_val_str)
        energy_mev = _energy_to_mev(energy_val, energy_unit)
        if level_idx is not None and 0 <= level_idx < num_levels:
            level_elem_indices[level_elem] = level_idx
        elif energy_mev is not None:
            mapped = _find_index_by_energy(energy_mev)
            if mapped is not None:
                level_elem_indices[level_elem] = mapped

    transitions: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    processed_gamma = set()

    def _add_transition(
        initial_idx: Optional[int],
        final_idx: Optional[int],
        gamma_energy_mev: Optional[float],
        intensity: Optional[float]
    ) -> None:
        if initial_idx is None or final_idx is None:
            return
        if initial_idx < 0 or initial_idx >= num_levels:
            return
        if final_idx < 0 or final_idx >= num_levels:
            return
        if initial_idx == final_idx:
            return
        if gamma_energy_mev is None and 0 <= initial_idx < num_levels and 0 <= final_idx < num_levels:
            gamma_energy_mev = level_energies[initial_idx] - level_energies[final_idx]
        if gamma_energy_mev is None or gamma_energy_mev <= 0.0:
            return
        intensity_val = intensity if intensity is not None else 1.0
        if intensity_val <= 0.0:
            return
        transitions[initial_idx].append((final_idx, gamma_energy_mev, intensity_val))

    for level_elem, level_idx in level_elem_indices.items():
        for gamma_elem in level_elem.iter():
            if gamma_elem.tag.rsplit('}', 1)[-1] != "gamma":
                continue
            processed_gamma.add(id(gamma_elem))
            final_idx = _parse_level_ref(gamma_elem.get("finalLevel") or gamma_elem.get("finalLevelIndex")
                                         or gamma_elem.get("finalLevelId"))
            if final_idx is None:
                final_energy_val = _parse_float(gamma_elem.get("finalLevelEnergy") or gamma_elem.get("finalEnergy"))
                final_energy_mev = _energy_to_mev(final_energy_val, gamma_elem.get("finalEnergyUnit"))
                final_idx = _find_index_by_energy(final_energy_mev)
            gamma_energy_val = _parse_float(gamma_elem.get("energy"))
            gamma_energy_unit = gamma_elem.get("unit") or gamma_elem.get("energyUnit")
            if gamma_energy_val is None:
                gamma_energy_str, gamma_energy_unit = _extract_value_and_unit(gamma_elem, "energy")
                gamma_energy_val = _parse_float(gamma_energy_str)
            gamma_energy_mev = _energy_to_mev(gamma_energy_val, gamma_energy_unit)
            intensity_val = _parse_float(gamma_elem.get("intensity") or gamma_elem.get("probability"))
            if intensity_val is None:
                intensity_str, _ = _extract_value_and_unit(gamma_elem, "intensity")
                intensity_val = _parse_float(intensity_str)
            _add_transition(level_idx, final_idx, gamma_energy_mev, intensity_val)

    for gamma_elem in root.iter():
        if gamma_elem.tag.rsplit('}', 1)[-1] != "gamma":
            continue
        if id(gamma_elem) in processed_gamma:
            continue
        initial_idx = _parse_level_ref(gamma_elem.get("initialLevel") or gamma_elem.get("initialLevelIndex")
                                       or gamma_elem.get("initialLevelId"))
        if initial_idx is None:
            initial_energy_val = _parse_float(gamma_elem.get("initialLevelEnergy") or gamma_elem.get("initialEnergy"))
            initial_energy_mev = _energy_to_mev(initial_energy_val, gamma_elem.get("initialEnergyUnit"))
            initial_idx = _find_index_by_energy(initial_energy_mev)
        final_idx = _parse_level_ref(gamma_elem.get("finalLevel") or gamma_elem.get("finalLevelIndex")
                                     or gamma_elem.get("finalLevelId"))
        if final_idx is None:
            final_energy_val = _parse_float(gamma_elem.get("finalLevelEnergy") or gamma_elem.get("finalEnergy"))
            final_energy_mev = _energy_to_mev(final_energy_val, gamma_elem.get("finalEnergyUnit"))
            final_idx = _find_index_by_energy(final_energy_mev)
        gamma_energy_val = _parse_float(gamma_elem.get("energy"))
        gamma_energy_unit = gamma_elem.get("unit") or gamma_elem.get("energyUnit")
        if gamma_energy_val is None:
            gamma_energy_str, gamma_energy_unit = _extract_value_and_unit(gamma_elem, "energy")
            gamma_energy_val = _parse_float(gamma_energy_str)
        gamma_energy_mev = _energy_to_mev(gamma_energy_val, gamma_energy_unit)
        if gamma_energy_mev is None and initial_idx is not None and final_idx is not None:
            gamma_energy_mev = level_energies[initial_idx] - level_energies[final_idx]
        intensity_val = _parse_float(gamma_elem.get("intensity") or gamma_elem.get("probability"))
        if intensity_val is None:
            intensity_str, _ = _extract_value_and_unit(gamma_elem, "intensity")
            intensity_val = _parse_float(intensity_str)
        _add_transition(initial_idx, final_idx, gamma_energy_mev, intensity_val)

    if not transitions:
        logger.debug(f"No ENSDF/ENDF gamma cascade transitions found in {filepath}")
        return None

    for level_idx, transition_list in transitions.items():
        total_intensity = sum(item[2] for item in transition_list)
        if total_intensity <= 0.0:
            continue
        cascades[level_idx] = [
            (final_idx, gamma_energy, intensity / total_intensity)
            for final_idx, gamma_energy, intensity in transition_list
        ]

    return cascades


def _parse_ripl3_gamma_cascades(filepath: os.PathLike, target_a: int, level_energies: List[float]) -> Optional[Dict[int, List[Tuple[int, float, float]]]]:
    """
    Parse gamma cascade data from RIPL-3 nuclear level scheme files.

    Args:
        filepath: Path to RIPL-3 .dat file (e.g., z006.dat for carbon)
        target_a: Mass number of target isotope (e.g., 12 for C-12)
        level_energies: List of level energies (MeV) to match against

    Returns:
        Dictionary mapping level index to gamma transitions, or None if parsing fails:
        {level_idx: [(final_level_idx, gamma_energy_MeV, transition_probability), ...]}
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logger.debug(f"Failed to read RIPL-3 file {filepath}: {e}")
        return None

    num_levels = len(level_energies)
    cascades: Dict[int, List[Tuple[int, float, float]]] = {idx: [] for idx in range(num_levels)}
    level_energy_map = {round(energy, 6): idx for idx, energy in enumerate(level_energies)}

    i = 0
    while i < len(lines):
        line = lines[i]
        if len(line) < 10:
            i += 1
            continue

        try:
            a_val = int(line[5:10].strip())
        except (ValueError, IndexError):
            i += 1
            continue

        if a_val != target_a:
            i += 1
            continue

        i += 1
        ripl_levels = {}

        while i < len(lines):
            line = lines[i]
            if len(line) < 20:
                i += 1
                continue

            if line[0] not in [' ', '\t']:
                break

            try:
                nl = int(line[0:3].strip())
                elv = float(line[4:14].strip())
                ng = int(line[34:37].strip()) if len(line) > 36 and line[34:37].strip() else 0
            except (ValueError, IndexError):
                i += 1
                continue

            ripl_levels[nl] = elv

            if ng > 0:
                for _ in range(ng):
                    i += 1
                    if i >= len(lines):
                        break
                    gamma_line = lines[i]
                    if len(gamma_line) < 50:
                        continue

                    try:
                        nf = int(gamma_line[39:43].strip())
                        eg = float(gamma_line[44:54].strip())
                        pg = float(gamma_line[55:65].strip())

                        elv_rounded = round(elv, 6)
                        if elv_rounded in level_energy_map:
                            initial_idx = level_energy_map[elv_rounded]
                        else:
                            initial_idx = None
                            for idx, e in enumerate(level_energies):
                                if abs(e - elv) < 0.01:
                                    initial_idx = idx
                                    break

                        if nf in ripl_levels:
                            final_energy = ripl_levels[nf]
                            final_rounded = round(final_energy, 6)
                            if final_rounded in level_energy_map:
                                final_idx = level_energy_map[final_rounded]
                            else:
                                final_idx = None
                                for idx, e in enumerate(level_energies):
                                    if abs(e - final_energy) < 0.01:
                                        final_idx = idx
                                        break
                        else:
                            final_idx = None

                        if initial_idx is not None and final_idx is not None and initial_idx != final_idx:
                            if pg > 0.0 and eg > 0.0:
                                cascades[initial_idx].append((final_idx, eg, pg))
                    except (ValueError, IndexError):
                        continue

            i += 1
        break

    if any(len(transitions) > 0 for transitions in cascades.values()):
        return cascades
    return None


def get_gamma_cascade_info(
    product_zaid: int,
    data_dir: Optional[os.PathLike] = None,
    level_energies: Optional[List[float]] = None
) -> Optional[Dict[int, List[Tuple[int, float, float]]]]:
    """
    Get gamma cascade transition data for product nucleus from (alpha,n) reaction.

    Args:
        product_zaid: ZAID of product nucleus (e.g., 6012 for C-12 from Be-9(alpha,n))
        data_dir: Optional directory containing gamma cascade data files
        level_energies: Optional list of level energies (MeV) for validation and fallback

    Returns:
        Dictionary mapping level index to list of gamma transitions:
        {level_idx: [(final_level_idx, gamma_energy_MeV, transition_probability), ...]}

        Returns None if no level energy information is available.
    """
    if level_energies is None or len(level_energies) == 0:
        logger.warning(f"No level energies provided for product ZAID {product_zaid}, cannot calculate gamma cascades")
        return None

    if level_energies[0] != 0.0:
        logger.warning(f"Level energies should start with ground state at 0.0 MeV for ZAID {product_zaid}")

    cascades = None
    z = product_zaid // 1000
    a = product_zaid % 1000

    if data_dir is None:
        data_root = _default_data_root()
        levels_dir = os.path.join(data_root, "levels")
        decay_dir = os.path.join(data_root, "decay", "ENDFBVIII")
    else:
        levels_dir = data_dir
        decay_dir = data_dir

    ripl3_path = os.path.join(levels_dir, f"z{z:03d}.dat")
    if os.path.exists(ripl3_path):
        cascades = _parse_ripl3_gamma_cascades(ripl3_path, a, level_energies)
        if cascades is not None:
            logger.debug(f"Loaded gamma cascade data for ZAID {product_zaid} from RIPL-3")
            return cascades

    symbol = atomic_data.get_element_symbol(z)
    possible_paths = [
        os.path.join(decay_dir, f"{product_zaid}.xml"),
        os.path.join(decay_dir, f"{z:03d}{a:03d}.xml"),
        os.path.join(decay_dir, f"a-{z:03d}_{symbol}_{a:03d}.endf.gnds.xml"),
    ]

    for filepath in possible_paths:
        if os.path.exists(filepath):
            cascades = _parse_endf_gamma_cascades(filepath, level_energies)
            if cascades is not None:
                logger.debug(f"Loaded gamma cascade data for ZAID {product_zaid} from {filepath}")
                return cascades

    logger.debug(f"Using ground-state fallback gamma cascade model for product ZAID {product_zaid}")
    cascades = _get_ground_state_cascade(level_energies)

    return cascades


def _calculate_sfnu_from_cumulative_dist(cum_dist: List[float]) -> float:
    """
    Calculate average neutron multiplicity from cumulative distribution.

    Args:
        cum_dist: List of cumulative probabilities [Pr[n<=0], Pr[n<=1], ..., Pr[n<=9]]

    Returns:
        Average neutron multiplicity
    """
    if not cum_dist or len(cum_dist) < 2:
        return 0.0

    discrete_probs = [cum_dist[0]]
    for i in range(1, len(cum_dist)):
        discrete_probs.append(cum_dist[i] - cum_dist[i-1])

    avg_sfnu = sum(i * p for i, p in enumerate(discrete_probs))

    return avg_sfnu


def _load_sf_data_from_yaml(zaid: int,
                            data_dir: Optional[os.PathLike] = None) -> Dict[str, float]:
    """
    Load spontaneous fission data from sf.yaml database.

    Args:
        zaid: ZAID identifier
        data_dir: Directory containing decay data (will look in data_dir/../decay/sf.yaml)

    Returns:
        Dictionary with SF parameters:
        {
            'sfnu': float - Average neutron multiplicity (calculated from dist if needed),
            'watt1': float - Watt parameter a [MeV],
            'watt2': float - Watt parameter b [1/MeV],
            'width': float - Gaussian width parameter,
            'sfyield': float - SF yield
        }
        Returns empty dict {} if not found.
    """
    try:
        if data_dir is not None:
            data_dir_str = str(data_dir)
            if data_dir_str.endswith(
                    'ENDFBVIII') or data_dir_str.endswith('gnds'):
                yaml_path = os.path.join(
                    os.path.dirname(data_dir), 'sf.yaml')
            else:
                yaml_path = os.path.join(data_dir, 'sf.yaml')
        else:
            yaml_path = os.path.join(
                _default_data_root(), 'decay', 'sf.yaml')

        if not os.path.exists(yaml_path):
            return {}

        with open(yaml_path, 'r') as f:
            sf_data = yaml.safe_load(f)

        if zaid not in sf_data:
            return {}

        sf_entry = sf_data[zaid]
        if not isinstance(sf_entry, dict):
            return {}

        result = {
            'width': float(sf_entry.get('width', 0.0)),
            'watt1': float(sf_entry.get('watt1', 0.0)),
            'watt2': float(sf_entry.get('watt2', 0.0)),
            'sfyield': float(sf_entry.get('sfyield', 0.0))
        }

        if 'sfnu' in sf_entry:
            result['sfnu'] = float(sf_entry['sfnu'])
        elif 'sfnu_dist' in sf_entry:
            cum_dist = sf_entry['sfnu_dist']
            if isinstance(cum_dist, list):
                result['sfnu'] = _calculate_sfnu_from_cumulative_dist(cum_dist)
                logger.debug(f"ZAID {zaid}: Calculated sfnu={result['sfnu']:.3f} from cumulative distribution")
            else:
                result['sfnu'] = 0.0
        else:
            result['sfnu'] = 0.0

        if result['watt1'] == 0.0 and result['watt2'] == 0.0 and result['sfnu'] > 0.0:
            logger.warning(
                f"ZAID {zaid}: SF data available but Watt spectrum parameters are zero. "
                f"Spectrum calculation not possible."
            )

        return result

    except Exception as e:
        logger.debug(f"Could not load SF data from YAML for ZAID {zaid}: {e}")
        return {}


def load_delayed_neutron_data(zaid: int) -> dict:
    """
    Load delayed neutron yield and spectrum for a nuclide from the bundled library.

    Args:
        zaid: ZAID identifier (ZZZAAA format)

    Returns:
        {'nu_delayed', 'average_energy_MeV', 'energy_grid_MeV', 'spectrum_per_MeV'}:
            Delayed neutron yield and 200-bin spectrum (0-10 MeV),
        {}: If the nuclide is outside Z=89-106 or has no data file.
    """
    z = zaid // 1000
    a = zaid % 1000
    if z < 89 or z > 106:
        return {}
    yaml_path = get_data_dir() / "delayed_neutron" / "spectra" / f"dn_spectrum_{z}_{a}_sf.yaml"
    if not yaml_path.exists():
        logger.warning(f"ZAID {zaid}: No delayed neutron data at {yaml_path}")
        return {}
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    required = ('nu_delayed', 'energy_MeV', 'spectrum_per_MeV')
    if not all(k in data for k in required):
        logger.warning(f"ZAID {zaid}: Delayed neutron YAML missing required keys")
        return {}
    return {
        'nu_delayed':         float(data['nu_delayed']),
        'average_energy_MeV': float(data.get('average_energy_MeV', 0.0)),
        'energy_grid_MeV':    list(data['energy_MeV']),
        'spectrum_per_MeV':   list(data['spectrum_per_MeV']),
    }


def _parse_endf_sf_data(filepath: str, zaid: int,
                        data_dir: Optional[os.PathLike] = None):
    """
    Parse SF data from ENDF/B-VIII decay file documentation.

    Args:
        filepath: Path to ENDF XML file
        zaid: ZAID identifier
        data_dir: Directory containing decay data (for nubar.yaml fallback)

    Returns:
        Tuple of (sf_strength, spectrum, params) where:
        - sf_strength: SF neutron emission rate [neutrons/s/atom]
        - spectrum: List of (energy [MeV], intensity [fraction]) tuples from group integrals,
                    or empty list if group integrals not available
        - params: Dict with 'nubar', 'decay_constant', 'sf_branching', 'avg_energy'
                 (watt_a and watt_b set to 0.0 as unavailable)
    """
    import xml.etree.ElementTree as ET
    import re

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        halflife_elem = root.find('.//halflife/double[@label="eval"]')
        if halflife_elem is None:
            return 0.0, [], {}

        halflife_s = float(halflife_elem.get('value'))
        decay_constant = np.log(2) / halflife_s if halflife_s > 0 else 0.0

        sf_mode = root.find(".//*decayMode[@mode='SF']")
        if sf_mode is None:
            return 0.0, [], {}

        sf_br_elem = sf_mode.find(".//probability/double[@label='BR']")
        if sf_br_elem is None:
            return 0.0, [], {}

        sf_branching = float(sf_br_elem.get('value'))

        nubar = 0.0
        spectrum = []
        avg_energy = 0.0

        doc_elem = root.find(".//endfCompatible")
        if doc_elem is not None and doc_elem.text:
            doc_text = doc_elem.text

            nubar_match = re.search(
                r'NEUTRONS PER SPONTANEOUS FISSION.*?TOTAL\s*=\s*(\d+\.\d+)',
                doc_text,
                re.DOTALL | re.IGNORECASE
            )
            if nubar_match:
                nubar = float(nubar_match.group(1))

            group_integrals_pattern = re.compile(
                r'^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+([\d\.]+[Ee][\+\-]?\d+)\s+[\d\.Ee][\+\-]?\d+',
                re.MULTILINE)

            group_section_match = re.search(
                r'GROUP\s+ENERGY\s+RANGE.*?SPECTRUM.*?RSD.*?\n.*?\n(.*?)(?=\*{10,}|\Z)',
                doc_text,
                re.DOTALL | re.IGNORECASE)

            if group_section_match:
                group_section = group_section_match.group(
                    1) if group_section_match.lastindex and group_section_match.lastindex >= 1 else group_section_match.group(0)
                matches = group_integrals_pattern.findall(group_section)

                if matches:
                    total_integral = 0.0
                    weighted_energy_sum = 0.0

                    for match in matches:
                        group_num = int(match[0])
                        e_low = float(match[1])
                        e_high = float(match[2])
                        integral = float(match[3])

                        e_center = (e_low + e_high) / 2.0
                        bin_width = e_high - e_low

                        spectrum.append((e_center, integral))

                        total_integral += integral
                        weighted_energy_sum += e_center * integral

                    if total_integral > 0:
                        spectrum = [(e, i / total_integral)
                                    for e, i in spectrum]
                        avg_energy = weighted_energy_sum / total_integral if total_integral > 0 else 0.0
                    else:
                        spectrum = []

        if nubar > 0 and sf_branching > 0:
            sf_strength = decay_constant * sf_branching * nubar
        else:
            sf_strength = 0.0

        params = {
            'decay_constant': decay_constant,
            'sf_branching': sf_branching,
            'nubar': nubar,
            'watt_a': 0.0,
            'watt_b': 0.0,
            'avg_energy': avg_energy,
        }

        return sf_strength, spectrum, params

    except Exception as e:
        logger.warning(f"Failed to parse ENDF SF data for ZAID {zaid}: {e}")
        return 0.0, [], {}


def _get_sf_data_with_yaml_nubar(zaid: int, return_params: bool = False):
    """
    Get SF data using ENDF for decay parameters and YAML for SF data (nubar and Watt parameters).

    Args:
        zaid: ZAID identifier
        return_params: If True, return params dict

    Returns:
        Same format as get_decay_spectrum for SF mode
    """
    import xml.etree.ElementTree as ET

    endf_dir = os.path.join(_default_data_root(), 'decay', 'ENDFBVIII')
    filepath = os.path.join(endf_dir, f"{zaid}.xml")

    if not os.path.exists(filepath):
        if return_params:
            return 0.0, [], {}
        else:
            return 0.0, []

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        halflife_elem = root.find('.//halflife/double[@label="eval"]')
        if halflife_elem is None:
            if return_params:
                return 0.0, [], {}
            else:
                return 0.0, []

        halflife_s = float(halflife_elem.get('value'))
        decay_constant = np.log(2) / halflife_s if halflife_s > 0 else 0.0

        sf_mode = root.find(".//*decayMode[@mode='SF']")
        if sf_mode is None:
            if return_params:
                return 0.0, [], {}
            else:
                return 0.0, []

        sf_br_elem = sf_mode.find(".//probability/double[@label='BR']")
        if sf_br_elem is None:
            if return_params:
                return 0.0, [], {}
            else:
                return 0.0, []

        sf_branching = float(sf_br_elem.get('value'))

        sf_yaml_data = _load_sf_data_from_yaml(zaid, data_dir=None)

        if not sf_yaml_data or sf_yaml_data.get('sfnu', 0.0) == 0.0:
            logger.warning(f"ZAID {zaid}: No SF data found in YAML database")
            if return_params:
                return 0.0, [], {}
            else:
                return 0.0, []

        nubar = sf_yaml_data['sfnu']
        watt_a = sf_yaml_data.get('watt1', 0.0)
        watt_b = sf_yaml_data.get('watt2', 0.0)

        sf_strength = decay_constant * sf_branching * nubar

        spectrum = []

        params = {
            'decay_constant': decay_constant,
            'sf_branching': sf_branching,
            'nubar': nubar,
            'watt_a': watt_a,
            'watt_b': watt_b,
            'avg_energy': 0.0,
        }

        logger.info(
            f"ZAID {zaid}: Using YAML SF data: nubar={nubar:.3f}, watt_a={watt_a:.3f}, watt_b={watt_b:.3f}")

        if return_params:
            return sf_strength, spectrum, params
        else:
            return sf_strength, spectrum

    except Exception as e:
        logger.warning(
            f"Failed to get SF data with YAML for ZAID {zaid}: {e}")
        if return_params:
            return 0.0, [], {}
        else:
            return 0.0, []


def get_decay_spectrum(
    zaid: int,
    data_dir: Optional[os.PathLike] = None,
    decay_mode: str = 'alpha',
    return_params: bool = False
):
    """
    Get decay spectrum and strength for a given nuclide.

    Args:
        zaid: ZAID identifier (ZZZAAA format)
        data_dir: Data directory
            - None: Use default ENDF/B-VIII decay data
            - "sources4a" or "sources4c": Use SOURCES tape5 (use sources4c for SF with Watt parameters)
            - "yaml": Use ENDF for decay data but nubar from YAML database (SF mode only)
            - Path: Custom directory
        decay_mode: Decay mode to extract
            - 'alpha': Alpha particle emission
            - 'sf': Spontaneous fission neutron emission (requires sources4c for Watt parameters)
        return_params: If True, return additional parameters dict
            For SF mode: {'nubar', 'watt_a', 'watt_b', 'decay_constant', 'sf_branching'}

    Returns:
        If return_params=False:
            - decay_strength: Emission rate per atom [particles/s/atom or neutrons/s/atom]
            - spectrum: List of (energy [MeV], intensity [fraction]) tuples
        If return_params=True:
            - decay_strength: Emission rate per atom
            - spectrum: List of (energy [MeV], intensity [fraction]) tuples
            - params: Dictionary of decay parameters
    """
    if data_dir == "sources" or (
            data_dir is not None and "sources" in str(data_dir)):
        return get_sources_decay_data(
            zaid, data_dir, decay_mode, return_params)

    if data_dir == "yaml" and decay_mode == 'sf':
        return _get_sf_data_with_yaml_nubar(zaid, return_params)

    if data_dir is None:
        data_dir = os.path.join(
            _default_data_root(),
            "decay",
            'ENDFBVIII')

    filepath = os.path.join(data_dir, f"{zaid}.xml")

    if not os.path.exists(filepath):
        if return_params:
            return 0.0, [], {}
        else:
            return 0.0, []

    if decay_mode == 'alpha':
        result = _parse_gnds_decay_data(filepath)
        if return_params:
            return result[0], result[1], {}
        else:
            return result

    elif decay_mode == 'sf':
        result = _parse_endf_sf_data(filepath, zaid, data_dir)
        sf_strength, spectrum, params = result

        sf_yaml_data = _load_sf_data_from_yaml(zaid, data_dir)

        if sf_yaml_data:
            if params.get('nubar', 0.0) == 0.0 and sf_yaml_data.get('sfnu', 0.0) > 0.0:
                params['nubar'] = sf_yaml_data['sfnu']
                decay_const = params.get('decay_constant', 0.0)
                sf_br = params.get('sf_branching', 0.0)
                if decay_const > 0 and sf_br > 0:
                    sf_strength = decay_const * sf_br * params['nubar']
                    logger.info(
                        f"ZAID {zaid}: Using nubar={params['nubar']:.3f} from YAML (not found in ENDF)")

            if params.get('watt_a', 0.0) == 0.0 and params.get('watt_b', 0.0) == 0.0:
                if sf_yaml_data.get('watt1', 0.0) > 0.0 and sf_yaml_data.get('watt2', 0.0) > 0.0:
                    params['watt_a'] = sf_yaml_data['watt1']
                    params['watt_b'] = sf_yaml_data['watt2']
                    logger.info(
                        f"ZAID {zaid}: Using Watt parameters from YAML: a={params['watt_a']:.3f}, b={params['watt_b']:.3f}")

        if params.get('sf_branching', 0.0) > 0:
            if sf_strength == 0.0 or params.get('nubar', 0.0) == 0.0:
                logger.warning(
                    f"SF data incomplete for ZAID {zaid}; has SF branching, but missing nubar. ")
            elif params.get('watt_a', 0.0) == 0.0 or params.get('watt_b', 0.0) == 0.0:
                logger.warning(
                    f"ZAID {zaid}: Has nubar={params.get('nubar', 0.0):.3f} but no Watt spectrum parameters available. ")

        if return_params:
            return sf_strength, spectrum, params
        else:
            return sf_strength, spectrum

    else:
        raise ValueError(f"Unknown decay_mode: {decay_mode}")


def _get_tendl_branching_info(
        root, zaid: int) -> Optional[Tuple[float, List[float], Dict[float, List[float]]]]:
    """
    TENDL fallback - returns hardcoded ground-state branching.

    Args:
        root: Root element of the TENDL XML file
        zaid: ZAID identifier

    Returns:
        Tuple of (q_value, level_energies, branching_data) or None if parsing fails
    """
    reactions = root.findall(".//reaction")
    if not reactions:
        return None

    energy_grid = None
    for reaction in reactions:
        xys = reaction.find(".//crossSection//XYs1d")
        if xys is not None and xys.find("values") is not None:
            txt = xys.find("values").text or ""
            try:
                arr = np.array([float(x) for x in txt.split()])
                if arr.size >= 2 and arr.size % 2 == 0:
                    energies_eV = arr[0::2]
                    energy_grid = energies_eV / 1e6
                    break
            except Exception:
                continue

    if energy_grid is None:
        energy_grid = np.arange(0.1, 15.1, 0.1)

    z = zaid // 1000
    a = zaid % 1000
    target_zaid = zaid
    product_zaid = (z + 2) * 1000 + (a + 3)

    m_target = get_atomic_mass_from_db(target_zaid)
    m_product = get_atomic_mass_from_db(product_zaid)

    if m_target is not None and m_product is not None:
        q_amu = (m_target + ALPH_MASS) - (m_product + ANEUT_MASS)
        q_value = q_amu * AMU_TO_MEV
    else:
        q_value = 0

    level_energies = [0.0]

    branching_data = {float(e): [1.0] for e in energy_grid}

    try:
        z = zaid // 1000
        if z == 12:
            energies = sorted(branching_data.keys(),
                              key=lambda e: abs(float(e) - 5.0))
            if energies:
                e_key = energies[0]
                fractions = np.asarray(branching_data[e_key], dtype=float)

    except Exception:
        pass

    return q_value, level_energies, branching_data


def _parse_gnds_decay_data(
        file_path: str) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Parse a GNDS XML decay data file to extract alpha decay strength and spectrum.

    Args:
        file_path (str): Path to the GNDS XML file.

    Returns:
        Tuple[float, List[Tuple[float, float]]]:
            - Alpha decay strength (decays/second/atom),
            - Alpha spectrum as a list of (energy [MeV], intensity [fraction]).
    """

    tree = ET.parse(file_path)
    root = tree.getroot()

    alpha_decay_strength = 0.0
    alpha_spectrum = []

    for nuclide in root.findall(".//nuclide"):
        for nucleus in nuclide.findall(".//nucleus"):
            halflife_element = nucleus.find(".//halflife/double")
            if halflife_element is not None:
                try:
                    half_life = float(halflife_element.get("value"))
                    if half_life > 0:
                        decay_constant = np.log(2) / half_life
                    else:
                        decay_constant = 0.0
                except (TypeError, ValueError):
                    logger.error("Error: Invalid or missing half-life value.")
                    decay_constant = 0.0
            else:
                decay_constant = 0.0

            decay_data = nucleus.find("decayData")
            if decay_data is not None:
                for decay_mode in decay_data.findall(".//decayMode"):
                    mode = decay_mode.get("mode")
                    if mode == "alpha":
                        branching_ratio_element = decay_mode.find(
                            ".//probability/double")
                        if branching_ratio_element is not None:
                            try:
                                branching_ratio = float(
                                    branching_ratio_element.get("value"))
                            except (TypeError, ValueError):
                                logger.error(
                                    "Invalid or missing branching ratio value.")
                                branching_ratio = 0.0
                        else:
                            logger.error("Branching ratio element not found.")
                            branching_ratio = 0.0

                        alpha_decay_strength = decay_constant * branching_ratio

                        spectra = decay_mode.find(".//spectra")
                        if spectra is not None:
                            for spectrum in spectra.findall(".//spectrum"):
                                if spectrum.get("label") == "alpha":
                                    for discrete in spectrum.findall(
                                            ".//discrete"):
                                        intensity_element = discrete.find(
                                            ".//intensity")
                                        if intensity_element is not None:
                                            intensity = float(
                                                intensity_element.get("value"))
                                        else:
                                            intensity = 0.0

                                        energy_element = discrete.find(
                                            ".//energy")
                                        if energy_element is not None:
                                            energy_mev = float(
                                                energy_element.get("value")) / 1e6
                                        else:
                                            energy_mev = 0.0

                                        alpha_spectrum.append(
                                            (energy_mev, intensity))

    return alpha_decay_strength, alpha_spectrum


def _get_an_xs_jendl_tendl(z: int, a: int, symbol: str) -> Dict[float, float]:
    """
    Get the (a,n) reaction cross section data for a given ZAID from JENDL if available, else TENDL.

    Args:
        z: Atomic number
        a: Atomic mass number
        symbol: Element symbol

    Returns:
        {energy (MeV): cross_section (barns)}: (a,n) reaction cross section
    """

    data_dir = os.path.join(_default_data_root(), "an_xs")

    jendl_5 = _get_an_xs_xml(os.path.join(
        data_dir, f"a_{z:03d}-{symbol}-{a:03d}.xml"))
    if jendl_5 is not None:
        return jendl_5

    jendl_1 = _get_an_xs_xml(os.path.join(data_dir, f"{symbol}-{a:03d}.xml"))
    if jendl_1 is not None:
        return jendl_1

    tendl_file = os.path.join(data_dir, "TENDL-2023",
                              f"a-{symbol}{a:03d}.tendl.gnds.xml")
    if os.path.exists(tendl_file):
        tendl_2023 = _get_an_xs_xml(tendl_file)
        if tendl_2023 is not None:
            return tendl_2023

    return None


def _get_an_xs_xml(filepath: os.PathLike) -> Optional[Dict[float, float]]:
    """
    Get the (a,n) reaction cross section data for a given ZAID from an XML file.

    Args:
        filepath: Path to the XML file.

    Returns:
        {energy (MeV): cross_section (barns)}: (a,n) reaction cross section
    """

    if not os.path.exists(filepath):
        return None

    tree = ET.parse(filepath)
    root = tree.getroot()

    reactions_node = root.find('reactions')
    incomplete_reactions = root.find('incompleteReactions')
    all_reactions = []
    if reactions_node is not None:
        all_reactions.extend(list(reactions_node))
    if incomplete_reactions is not None:
        all_reactions.extend(list(incomplete_reactions))
    reactions_node = all_reactions

    neutron_producing_mt = [
        4, 11] + list(range(16, 26)) + [28, 29, 30] + list(range(32, 39)) + [41, 42, 44, 45]
    cross_sections = []
    for reaction in reactions_node:
        mt_number = reaction.get('ENDF_MT')
        if mt_number and int(mt_number) == 201:
            return _get_cross_section_from_reaction(reaction)
        elif mt_number and int(mt_number) in neutron_producing_mt:
            cross_sections.append(_get_cross_section_from_reaction(reaction))
    if len(cross_sections) == 0:
        return None
    if len(cross_sections) == 1:
        return cross_sections[0]
    return _sum_cross_sections(cross_sections)


def _get_cross_section_from_reaction(
        reaction: ET.Element) -> Dict[float, float]:
    """
    Get the cross section from a reaction element.

    Args:
        reaction: Reaction element.

    Returns:
        {energy (MeV): cross_section (barns)}: Cross section.

    Raises:
        ValueError: If the reaction is badly formed.
    """
    xs = reaction.find("crossSection")
    xys1d = xs.find("XYs1d")

    if xys1d is None:
        regions1d = xs.find("regions1d")
        if regions1d is None:
            return None
        function1ds = regions1d.find("function1ds")
        if function1ds is None:
            return None

        all_xys1d = function1ds.findall("XYs1d")
        if all_xys1d:
            all_energies = []
            all_cross_sections = []

            for xys_elem in all_xys1d:
                values_elem = xys_elem.find("values")
                if values_elem is not None:
                    xystring = values_elem.text.strip()
                    if xystring:
                        values = np.array([float(x) for x in xystring.split()])
                        if len(values) % 2 == 0:
                            energies_eV = values[::2]
                            cross_sections_barns = values[1::2]
                            all_energies.extend(energies_eV)
                            all_cross_sections.extend(cross_sections_barns)

            if all_energies:
                energies_MeV = np.array(all_energies) / 1e6
                return dict(zip(energies_MeV, all_cross_sections))

        xys1d = function1ds.find("XYs1d")
        if xys1d is None:
            xys1d = function1ds.find(".//XYs1d")

    if xys1d is None:
        raise ValueError(f"No XYs1d data found in cross section.")

    values_elem = xys1d.find("values")
    if values_elem is None:
        raise ValueError(f"No values found in XYs1d element.")

    xystring = values_elem.text.strip()
    if not xystring:
        raise ValueError(f"Empty values in XYs1d element.")

    try:
        values = np.array([float(x) for x in xystring.split()])
    except Exception:
        return None
    if values.ndim == 0 or values.size < 2 or (values.size % 2) != 0:
        return None

    energies_eV = values[0::2]
    cross_sections_barns = values[1::2]
    energies_MeV = energies_eV / 1e6
    return dict(zip(energies_MeV, cross_sections_barns))


def _sum_cross_sections(
        cross_sections: List[Dict[float, float]]) -> Dict[float, float]:
    """
    Sum a list of cross sections using interpolation-based approach.

    Creates interpolators for each level, then sums the interpolated values to get total cross-section at each energy, returns discretized result.

    Args:
        cross_sections: List of cross section dictionaries {energy: xs_value}

    Returns:
        Dictionary with summed cross sections on common energy grid
    """
    if not cross_sections:
        return {}

    if len(cross_sections) == 1:
        return cross_sections[0]

    interpolators = []
    for xs_dict in cross_sections:
        if not xs_dict:
            continue
        energies = sorted(xs_dict.keys())
        xs_values = [xs_dict[e] for e in energies]
        interp_func = interp1d(
            energies,
            xs_values,
            kind='linear',
            bounds_error=False,
            fill_value=0.0)
        interpolators.append((energies, interp_func))

    if not interpolators:
        return {}

    all_energies = set()
    for energies, _ in interpolators:
        all_energies.update(energies)

    if not all_energies:
        return {}

    common_energy_grid = sorted(all_energies)

    total_cs = np.zeros(len(common_energy_grid))

    for _, interp_func in interpolators:
        level_cs = interp_func(common_energy_grid)
        level_cs = np.maximum(level_cs, 0.0)
        total_cs += level_cs

    return {float(e): float(s) for e, s in zip(common_energy_grid, total_cs)}


def _get_stopping_power_astar(
        filepath: os.PathLike, atomic_mass: float) -> Optional[Dict[float, float]]:
    """
    Get the stopping power data for a given ZAID from ASTAR data.

    Args:
        filepath: Path to the ASTAR data file,
        atomic_mass: Atomic mass of the element.

    Returns:
        Dictionary {energy (MeV): total stopping power (MeV cm^2)}, or None if the file is not found.
    """

    if not os.path.exists(filepath):
        return None

    energy_list = []
    stopping_power_list = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines[8:]:
            if line.strip() and not line.startswith(('ASTAR', 'Kinetic', 'Energy', 'MeV')):
                columns = line.split()
                energy = float(columns[0])
                total_stopping_power = float(columns[3])

                converted_stopping_power = total_stopping_power * atomic_mass / AVOGADRO_NUM
                energy_list.append(energy)
                stopping_power_list.append(converted_stopping_power)

    return dict(zip(energy_list, stopping_power_list))


def _get_stopping_power_detect_format(
        zaid: int, data_dir: os.PathLike) -> Optional[Dict[float, float]]:
    """
    Auto-detect whether the data directory contains ASTAR or SRIM format files
    and call the appropriate helper function.

    Args:
        z: Atomic number
        a: Atomic mass number
        symbol: Element symbol
        data_dir: Path to the data directory

    Returns:
        Dictionary {energy (MeV): total stopping power (MeV cm^2)}, or None if detection fails
    """

    z = zaid // 1000
    a = zaid % 1000
    atomic_mass = atomic_data.get_atomic_mass(zaid)
    symbol = atomic_data.get_element_symbol(z)

    for filename in os.listdir(data_dir):
        if filename.endswith(('.txt', '.dat')):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith(
                            'ASTAR:') or first_line.startswith('ATIMA:'):
                        astar_file = os.path.join(data_dir, f"{z}.txt")
                        if os.path.exists(astar_file):
                            return _get_stopping_power_astar(
                                astar_file, atomic_mass)
                    elif first_line.startswith('# From SRIM'):
                        srim_file = os.path.join(
                            data_dir, f"{z}.txt")
                        if os.path.exists(srim_file):
                            return _get_stopping_power_srim(
                                srim_file, atomic_mass)
                    break
            except Exception:
                continue

    raise ValueError(
        f"No stopping power data found for ZAID {z:03d}{a:03d} in {data_dir}.")


def _get_stopping_power_srim(
        filepath: os.PathLike, atomic_mass: float) -> Optional[Dict[float, float]]:
    """
    Get the stopping power data for a given ZAID from SRIM data.

    Args:
        filepath: Path to the SRIM data file,
        atomic_mass: Atomic mass of the element.

    Returns:
        Dictionary {energy (MeV): total stopping power (MeV cm^2)}, or None if the file is not found.
    """

    if not os.path.exists(filepath):
        return None

    energy_list = []
    stopping_power_list = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines[4:]:
            if line.strip() and not line.startswith(('ASTAR', 'Kinetic', 'Energy', 'MeV')):
                columns = line.split()
                if columns[1] == 'keV':
                    energy = float(columns[0]) / 1_000.0
                else:
                    energy = float(columns[0])

                total_stopping_power = (
                    float(columns[2]) + float(columns[3])) * atomic_mass / AVOGADRO_NUM

                energy_list.append(energy)
                stopping_power_list.append(total_stopping_power)

    return dict(zip(energy_list, stopping_power_list))


def _get_continuum_dist_from_reaction(
        reaction: ET.Element) -> Optional[Dict[float, List[Tuple[float, float]]]]:
    """
    Extract the neutron energy distribution from an MT=91 continuum reaction element.

    Handles KalbachMann (<KalbachMann><f><XYs2d>) and direct <XYs2d> formats.
    Returns None when the distribution element is <unspecified>.

    Args:
        reaction: ET.Element - MT=91 reaction element from a GNDS XML file

    Returns:
        {incident_energy_MeV: [(E_out_MeV, prob_1/MeV), ...]}, or None if no tabulated data
    """
    product = reaction.find(".//product[@pid='n']")
    if product is None:
        return None
    distribution = product.find("distribution")
    if distribution is None:
        return None

    xys2d = None
    kalbach = distribution.find("KalbachMann")
    if kalbach is not None:
        f_elem = kalbach.find("f")
        if f_elem is not None:
            xys2d = f_elem.find("XYs2d")
    else:
        xys2d = distribution.find("XYs2d")

    if xys2d is None:
        return None

    func1ds = xys2d.find("function1ds")
    if func1ds is None:
        return None

    result = {}
    for xys1d in func1ds.findall("XYs1d"):
        e_ev_str = xys1d.get("outerDomainValue")
        if e_ev_str is None:
            continue
        values_elem = xys1d.find("values")
        if values_elem is None or not values_elem.text:
            continue
        raw = [float(x) for x in values_elem.text.split()]
        if len(raw) < 2 or len(raw) % 2 != 0:
            continue
        e_out_arr = raw[::2]
        prob_arr = raw[1::2]
        pairs = [(e_out_arr[k] / 1e6, prob_arr[k] * 1e6) for k in range(len(e_out_arr))]
        result[float(e_ev_str) / 1e6] = pairs

    return result if result else None


def get_continuum_info(
        zaid: int,
        data_dir: Optional[os.PathLike] = None
) -> Tuple[Optional[Dict[float, float]], Optional[Dict[float, List[Tuple[float, float]]]]]:
    """
    Get cross section and energy distribution for the MT=91 continuum (alpha,n) reaction.

    Reads the MT=91 continuum reaction from the same GNDS XML files used for cross
    sections. Returns (None, None) when SOURCES data is in use or no MT=91 reaction
    is present. Returns (continuum_xs, None) when cross section data exists but the
    neutron energy distribution element is <unspecified>.

    Args:
        zaid: int - Target nucleus ZAID (ZZZAAA format)
        data_dir: os.PathLike, optional - Directory containing nuclear data files.
            Defaults to the standard ENDF data directory.

    Returns:
        Tuple of (continuum_xs, continuum_dist) where continuum_xs is
        {energy_MeV: cross_section_barns} and continuum_dist is
        {incident_energy_MeV: [(E_out_MeV, prob_1/MeV), ...]}.
        Returns (None, None) if no MT=91 data is found.

    Raises:
        ValueError: If zaid is not a valid ZZZAAA formatted ZAID
    """
    if zaid >= 1e6:
        raise ValueError(f"ZAID {zaid} is not a valid ZZZAAA formatted ZAID.")

    if data_dir is None and _should_use_sources_for_an_xs(zaid):
        return None, None

    if data_dir == "sources" or (
            data_dir is not None and "sources" in str(data_dir)):
        return None, None

    found_path = _find_gnds_xml(zaid, data_dir)

    if not found_path:
        return None, None

    try:
        tree = ET.parse(found_path)
        root = tree.getroot()
    except ET.ParseError:
        return None, None

    reaction = root.find(".//reaction[@ENDF_MT='91']")
    if reaction is None:
        return None, None

    continuum_xs = _get_cross_section_from_reaction(reaction)
    if continuum_xs is None:
        return None, None

    continuum_dist = _get_continuum_dist_from_reaction(reaction)
    return continuum_xs, continuum_dist


def _calculate_branching_fractions(
        level_cross_sections: Dict[int, Dict[float, float]]) -> Dict[float, np.ndarray]:
    """
    Calculate branching fractions for a given set of level cross sections.

    Args:
        level_cross_sections: Dictionary of level cross sections {level_idx: {energy: cross_section}}

    Returns:
        Dictionary of branching fractions {energy: branching_fractions}
    """

    if not level_cross_sections:
        return {}

    all_energies = set()
    for level_cs in level_cross_sections.values():
        all_energies.update(level_cs.keys())

    sorted_energies = sorted(all_energies)
    branching_data = {}

    for energy in sorted_energies:
        level_values = []
        total_cs = 0.0

        max_level = max(level_cross_sections.keys())
        for level_idx in range(max_level + 1):
            if level_idx in level_cross_sections:
                level_cs = level_cross_sections[level_idx]
                if energy in level_cs:
                    cs_value = level_cs[energy]
                    level_values.append(cs_value)
                    total_cs += cs_value
                else:
                    energies = sorted(level_cs.keys())
                    if energies and min(energies) <= energy <= max(energies):
                        cs_values = [level_cs[e] for e in energies]
                        try:
                            interp_func = interp1d(
                                energies, cs_values, kind='linear', bounds_error=False, fill_value=0.0)
                            cs_value = float(interp_func(energy))
                            level_values.append(cs_value)
                            total_cs += cs_value
                        except Exception:
                            level_values.append(0.0)
                    else:
                        level_values.append(0.0)
            else:
                level_values.append(0.0)

        if total_cs > 0:
            branching_fractions = np.array(level_values) / total_cs
            branching_data[energy] = branching_fractions

    return branching_data


def _get_endf_level_data(
        root) -> Tuple[Dict[int, float], Dict[int, Dict[float, float]], float]:
    """
    Extract level energies, cross sections, and Q-value from ENDF root element.

    Args:
        root: Root element of the ENDF XML file

    Returns:
        level_energies: Dictionary of level energies {level_idx: energy (MeV)}
        level_cross_sections: Dictionary of level cross sections {level_idx: {energy: cross_section}}
        ground_state_q_value: Q-value for the ground state (MeV)
    """

    level_energies = {}
    level_cross_sections = {}
    ground_state_q_value = 0.0

    level_energies[0] = 0.0

    for nuclide in root.findall(".//nuclide"):
        nuclide_id = nuclide.get('id', '')
        if '_e' in nuclide_id:
            match = re.search(r'_e(\d+)', nuclide_id)
            if match:
                level_num = int(match.group(1))
                energy_elem = nuclide.find(".//energy/double")
                if energy_elem is not None:
                    energy_ev = float(energy_elem.get('value', 0))
                    level_energies[level_num] = energy_ev / 1e6

    mt50_reaction = root.find(".//reaction[@ENDF_MT='50']")
    if mt50_reaction is not None:
        cs_data = _get_cross_section_from_reaction(mt50_reaction)
        if cs_data:
            level_cross_sections[0] = cs_data

        q_elem = mt50_reaction.find(".//Q/constant1d")
        if q_elem is not None:
            q_value_ev = float(q_elem.get('value', 0))
            ground_state_q_value = q_value_ev / 1e6

    for level_idx in range(1, 41):
        mt = 50 + level_idx
        reaction = root.find(f".//reaction[@ENDF_MT='{mt}']")
        if reaction is not None:
            cs_data = _get_cross_section_from_reaction(reaction)
            if cs_data:
                level_cross_sections[level_idx] = cs_data

    return level_energies, level_cross_sections, ground_state_q_value
