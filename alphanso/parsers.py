import glob
import gzip
import xml.etree.ElementTree as ET
import numpy as np
import os
import math
import re
import yaml
import zlib as _zlib
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

_G4HPDATA_ENV = "G4PARTICLEHPDATA"
_XML_ATTR_MT = "ENDF_MT"
_XML_TAG_MULT_SUM = "multiplicitySum"
_XML_TAG_REACTION = "reaction"


def _default_data_root():
    return str(get_data_dir())

def _load_data_overrides():
    """Load the data_overrides.yaml configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    overrides_path = os.path.join(current_dir, "data", "data_overrides.yaml")

    if os.path.exists(overrides_path):
        try:
            with open(overrides_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load data_overrides.yaml: {e}")
            return None
    return None


_DATA_OVERRIDES = _load_data_overrides()


def _should_use_sources_for_an_xs(zaid: int) -> bool:
    """Check if a ZAID should use SOURCES data for (alpha,n) cross sections."""
    if _DATA_OVERRIDES is None:
        return False

    use_sources_zaids = _DATA_OVERRIDES.get('an_xs', {}).get('use_sources_tape_zaids', [])
    return zaid in use_sources_zaids


def _should_use_tendl_for_an_xs(zaid: int) -> bool:
    """Check if a ZAID should use TENDL instead of JENDL for (alpha,n) cross sections."""
    if _DATA_OVERRIDES is None:
        return False

    use_tendl_zaids = _DATA_OVERRIDES.get('an_xs', {}).get('use_tendl_zaids', [])
    return zaid in use_tendl_zaids


def _should_use_sources_for_stopping(zaid: int) -> bool:
    """Check if a ZAID should use SOURCES data for stopping power."""
    if _DATA_OVERRIDES is None:
        return False

    z = zaid // 1000
    default_z_threshold = _DATA_OVERRIDES.get('stopping', {}).get('default_sources_for_z_gt', 999)
    return z > default_z_threshold


def _get_sources_an_xs_dir() -> Optional[str]:
    """Get the directory for SOURCES (alpha,n) cross section data."""
    if _DATA_OVERRIDES is None:
        return None

    tape_path = _DATA_OVERRIDES.get('an_xs', {}).get('tape')
    if tape_path:
        return os.path.join(_default_data_root(), os.path.dirname(tape_path))
    return None


def _get_sources_stopping_dir() -> Optional[str]:
    """Get the directory for SOURCES stopping power data."""
    if _DATA_OVERRIDES is None:
        return None

    tape_path = _DATA_OVERRIDES.get('stopping', {}).get('tape')
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
        if _should_use_tendl_for_an_xs(zaid):
            candidates = [
                os.path.join(data_root, 'an_xs', "TENDL", f'{zaid}.xml'),
            ]
        else:
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
            logger.info(f"Using SOURCES data for ZAID {zaid} based on data_overrides.yaml")
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
            logger.info(f"Using SOURCES stopping power for ZAID {zaid} based on data_overrides.yaml")
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
            logger.info(f"Using SOURCES branching data for ZAID {zaid} based on data_overrides.yaml")
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
            tq, tl, td = tendl_branch
            return _extend_branching_with_jendltendl01(zaid, tq, tl, td, data_dir)

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
                    tq, tl, td = tendl_branch
                    return _extend_branching_with_jendltendl01(zaid, tq, tl, td, data_dir)
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

    q_value, level_energies_list, branching_data = _extend_branching_with_jendltendl01(
        zaid, q_value, level_energies_list, branching_data, data_dir
    )

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


_PROTON_MASS_AMU: float = 1.00782503207

_DAUGHTER_LEVEL_ENERGIES_MEV: Dict[int, List[float]] = {
    6012: [0.0, 4.4389, 7.6542, 9.6410, 10.3531, 12.7126],
    7014: [0.0, 2.3126, 3.9478, 4.9153, 5.1059, 5.6910, 5.8340, 6.2035, 6.4460],
    8016: [0.0, 6.0497, 6.9171, 7.1169, 7.6170, 8.8720, 9.8450, 9.5850, 10.3560],
    10020: [0.0, 1.6337, 4.2479, 4.9669, 5.6213, 5.7884, 6.7207, 7.0424, 7.1542],
}


def _compute_neutron_sep_energy(product_zaid: int) -> Optional[float]:
    """
    Compute the neutron separation energy for a nucleus using tabulated atomic masses.

    Uses atomic masses so that electron binding energies cancel exactly.

    Args:
        product_zaid: ZAID of the nucleus in ZZZAAA format.

    Returns:
        Neutron separation energy in MeV, or None if masses are unavailable.
    """
    z = product_zaid // 1000
    a = product_zaid % 1000
    if a < 2:
        return None
    daughter_zaid = z * 1000 + (a - 1)
    m_product = get_atomic_mass_from_db(product_zaid)
    m_daughter = get_atomic_mass_from_db(daughter_zaid)
    m_n = ANEUT_MASS
    if m_product is None or m_daughter is None:
        return None
    return (m_daughter + m_n - m_product) * AMU_TO_MEV


def _compute_proton_sep_energy(product_zaid: int) -> Optional[float]:
    """
    Compute the proton separation energy for a nucleus using tabulated atomic masses.

    Uses atomic masses so that electron binding energies cancel exactly.

    Args:
        product_zaid: ZAID of the nucleus in ZZZAAA format.

    Returns:
        Proton separation energy in MeV, or None if masses are unavailable.
    """
    z = product_zaid // 1000
    a = product_zaid % 1000
    if z < 1 or a < 2:
        return None
    daughter_zaid = (z - 1) * 1000 + (a - 1)
    m_product = get_atomic_mass_from_db(product_zaid)
    m_daughter = get_atomic_mass_from_db(daughter_zaid)
    m_h1 = get_atomic_mass_from_db(1001)
    if m_product is None or m_daughter is None or m_h1 is None:
        return None
    return (m_daughter + m_h1 - m_product) * AMU_TO_MEV


def _flatten_cascade(
    start_idx: int,
    cascades: Dict[int, List[Tuple[int, float, float]]],
    level_energies: List[float],
) -> List[Tuple[float, float]]:
    """
    Propagate a cascade from start_idx to the ground state and collect all emitted gammas.

    Iterates the full cascade chain so that intermediate levels are followed
    rather than stopping after a single transition.

    Args:
        start_idx: Level index to start from.
        cascades: Gamma transition dict {level_idx: [(final_idx, E_gamma_MeV, prob)]}.
        level_energies: Level energies in MeV.

    Returns:
        List of (gamma_energy_MeV, cumulative_probability) pairs.
    """
    gammas: Dict[float, float] = defaultdict(float)
    stack = [(start_idx, 1.0)]
    max_depth = (len(level_energies) + 1) * 4
    depth = 0
    while stack and depth < max_depth:
        depth += 1
        idx, pop = stack.pop()
        if idx == 0 or pop <= 1e-15:
            continue
        transitions = cascades.get(idx, [])
        if not transitions:
            e = level_energies[idx] if idx < len(level_energies) else 0.0
            if e > 0.0:
                gammas[round(e, 6)] += pop
            continue
        for final_idx, gamma_e, prob in transitions:
            if gamma_e > 0.0:
                gammas[round(gamma_e, 6)] += pop * prob
            if final_idx > 0:
                stack.append((final_idx, pop * prob))
    return list(gammas.items())


def _reroute_unbound_levels(
    product_zaid: int,
    cascades: Dict[int, List[Tuple[int, float, float]]],
    level_energies: List[float],
    data_dir: Optional[os.PathLike],
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Replace cascade entries for proton-unbound product levels with gamma lines
    from the daughter nucleus reached by proton emission.

    When an (alpha,n) product level has excitation energy above the proton
    separation energy, the level decays by proton emission rather than by
    gamma emission.  ALPHANSO's gamma-only cascade model produces zero yield
    for such levels.  This function routes the population through the daughter
    nucleus so that gammas from the daughter de-excitation are included.

    Population is distributed uniformly across all daughter levels energetically
    accessible from the available energy E_avail = E_level - S_p.  The daughter
    cascade is then fully propagated to the ground state and the resulting gamma
    lines are embedded directly as transitions from the unbound level to the
    product ground state (final_idx = 0), preventing double-counting in the
    iterative cascade loop of _calculate_gamma_spectrum.

    Scaling without renormalization is intentional: probability not assigned to
    any transition represents population that reaches the daughter ground state
    without emitting a gamma.

    Args:
        product_zaid: ZAID of the primary (alpha,n) product.
        cascades: Gamma cascade dict for the product nucleus.
        level_energies: Level energies of the product nucleus in MeV.
        data_dir: Data directory passed through to get_gamma_cascade_info.

    Returns:
        Modified cascade dict (new object; input is not modified).
    """
    s_p = _compute_proton_sep_energy(product_zaid)
    if s_p is None or s_p <= 0.0:
        return cascades

    unbound = [i for i, e in enumerate(level_energies) if i > 0 and e > s_p]
    if not unbound:
        return cascades

    z = product_zaid // 1000
    a = product_zaid % 1000
    daughter_zaid = (z - 1) * 1000 + (a - 1)

    daughter_levels = _DAUGHTER_LEVEL_ENERGIES_MEV.get(daughter_zaid)
    if daughter_levels is None:
        daughter_gnds = _find_gnds_xml(daughter_zaid, data_dir)
        if daughter_gnds is not None:
            try:
                root = ET.parse(daughter_gnds).getroot()
                lev_dict, _, _ = _get_endf_level_data(root)
                if lev_dict:
                    max_l = max(lev_dict.keys())
                    daughter_levels = [lev_dict.get(i, 0.0) for i in range(max_l + 1)]
            except Exception:
                pass

    if daughter_levels is None or len(daughter_levels) < 2:
        return cascades

    daughter_cascades = get_gamma_cascade_info(
        daughter_zaid,
        data_dir=data_dir,
        level_energies=daughter_levels,
    )
    if daughter_cascades is None:
        return cascades

    modified: Dict[int, List[Tuple[int, float, float]]] = {
        k: list(v) for k, v in cascades.items()
    }

    for i in unbound:
        e_avail = level_energies[i] - s_p
        accessible = [j for j, e in enumerate(daughter_levels) if j > 0 and e <= e_avail]
        if not accessible:
            modified[i] = []
            continue
        pop_each = 1.0 / len(accessible)
        gamma_accum: Dict[float, float] = defaultdict(float)
        for j in accessible:
            for e_gamma, prob in _flatten_cascade(j, daughter_cascades, daughter_levels):
                gamma_accum[e_gamma] += pop_each * prob
        modified[i] = [
            (0, e_gamma, p)
            for e_gamma, p in gamma_accum.items()
            if p > 0.0
        ]

    return modified


_GAMMA_CASCADE_CORRECTIONS: Dict[int, list] = {
    8016: [
        {
            'type': 'zero_level',
            'from_energy_mev': 6.048,
            'tolerance': 0.005,
        },
        {
            'type': 'scale_branch',
            'from_energy_mev': 8.869,
            'to_energy_mev': 0.0,
            'scale': 0.05,
            'tolerance': 0.010,
        },
    ],
    10021: [
        {
            'type': 'scale_branch',
            'from_energy_mev': 2.795,
            'to_energy_mev': 0.0,
            'scale': 1.0 / 300.0,
            'tolerance': 0.003,
        },
        {
            'type': 'scale_branch',
            'from_energy_mev': 2.789,
            'to_energy_mev': 0.351,
            'scale': 1.0 / 13.0,
            'tolerance': 0.003,
        },
    ],
}


def _apply_gamma_cascade_corrections(
    product_zaid: int,
    cascades: Dict[int, List[Tuple[int, float, float]]],
    level_energies: List[float],
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Apply known corrections to RIPL-3 gamma cascade data for specific product nuclei.

    Corrections address three classes of error documented against SaG4n/GEANT4:
      - E0 transitions that cannot emit a single photon (set branch to zero).
      - K-forbidden transitions whose RIPL-3 branch is orders of magnitude too large.
      - Levels whose particle-decay widths dominate but are not carried in RIPL-3,
        causing the gamma/total ratio to be assigned as 100%.

    Scaling a branch without renormalising is intentional: the probability not
    assigned to any gamma transition represents population lost to particle decay.

    Args:
        product_zaid: ZAID of the product nucleus.
        cascades: Gamma cascade dict {level_idx: [(final_idx, E_gamma_MeV, prob)]}.
        level_energies: List of level energies in MeV, index 0 is the ground state.

    Returns:
        Corrected cascade dict (new object; input is not modified).
    """
    corrections = _GAMMA_CASCADE_CORRECTIONS.get(product_zaid)
    if not corrections:
        return cascades

    def _find_idx(e_mev: float, tol: float) -> Optional[int]:
        for i, e in enumerate(level_energies):
            if abs(e - e_mev) <= tol:
                return i
        return None

    modified: Dict[int, List[Tuple[int, float, float]]] = {
        k: list(v) for k, v in cascades.items()
    }

    for corr in corrections:
        from_idx = _find_idx(corr['from_energy_mev'], corr['tolerance'])
        if from_idx is None:
            continue

        if corr['type'] == 'zero_level':
            modified[from_idx] = []

        elif corr['type'] == 'scale_branch':
            to_idx = _find_idx(corr['to_energy_mev'], corr['tolerance'])
            if to_idx is None:
                continue
            scale = corr['scale']
            modified[from_idx] = [
                (f, e, p * scale if f == to_idx else p)
                for f, e, p in modified.get(from_idx, [])
            ]

    return modified


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
            cascades = _reroute_unbound_levels(product_zaid, cascades, level_energies, data_dir)
            return _apply_gamma_cascade_corrections(product_zaid, cascades, level_energies)

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
                cascades = _reroute_unbound_levels(product_zaid, cascades, level_energies, data_dir)
                return _apply_gamma_cascade_corrections(product_zaid, cascades, level_energies)

    logger.debug(f"Using ground-state fallback gamma cascade model for product ZAID {product_zaid}")
    cascades = _get_ground_state_cascade(level_energies)
    cascades = _reroute_unbound_levels(product_zaid, cascades, level_energies, data_dir)
    return _apply_gamma_cascade_corrections(product_zaid, cascades, level_energies)


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
    Parse branching data from a TENDL GNDS XML file.

    Reads all discrete level cross sections (MT=50 through MT=90) using the same
    _get_endf_level_data / _calculate_branching_fractions pipeline as the JENDL
    path.  Level energies absent from nuclide elements are recovered from the
    per-reaction Q-values (see _get_endf_level_data).  This makes TENDL files
    contribute higher discrete levels that JENDL truncates to the MT=91 continuum.

    Falls back to ground-state-only branching when no discrete level cross
    sections are found, preserving the previous behaviour for targets where the
    TENDL file does not carry discrete channel data.

    Args:
        root: Root element of the TENDL GNDS XML file.
        zaid: ZAID identifier of the target nucleus.

    Returns:
        Tuple of (q_value_MeV, level_energies_MeV, branching_data) or None if
        parsing fails entirely.
    """
    z = zaid // 1000
    a = zaid % 1000
    product_zaid = (z + 2) * 1000 + (a + 3)

    m_target = get_atomic_mass_from_db(zaid)
    m_product = get_atomic_mass_from_db(product_zaid)
    if m_target is not None and m_product is not None:
        q_value = ((m_target + ALPH_MASS) - (m_product + ANEUT_MASS)) * AMU_TO_MEV
    else:
        q_value = 0.0

    lev_dict, level_cross_sections, _ = _get_endf_level_data(root)

    if len(level_cross_sections) > 1:
        max_l = max(lev_dict.keys()) if lev_dict else 0
        level_energies_list = [lev_dict.get(i, 0.0) for i in range(max_l + 1)]
        branching_data = _calculate_branching_fractions(level_cross_sections)
        if branching_data:
            return q_value, level_energies_list, branching_data

    reactions = root.findall(".//reaction")
    energy_grid = None
    for reaction in reactions:
        xys = reaction.find(".//crossSection//XYs1d")
        if xys is not None and xys.find("values") is not None:
            txt = xys.find("values").text or ""
            try:
                arr = np.array([float(x) for x in txt.split()])
                if arr.size >= 2 and arr.size % 2 == 0:
                    energy_grid = arr[0::2] / 1e6
                    break
            except Exception:
                continue
    if energy_grid is None:
        energy_grid = np.arange(0.1, 15.1, 0.1)

    branching_data = {float(e): [1.0] for e in energy_grid}
    return q_value, [0.0], branching_data


_JENDLTENDL01_NAME_MAP = {
    3006: '3_6_Lithium',
    3007: '3_7_Lithium',
    4009: '4_9_Berylium',
    5010: '5_10_Boron',
    5011: '5_11_Boron',
    6012: '6_12_Carbon',
    6013: '6_13_Carbon',
    7014: '7_14_Nitrogen',
    7015: '7_15_Nitrogen',
    8017: '8_17_Oxygen',
    8018: '8_18_Oxygen',
}


def _parse_jendltendl01_channel_xs(filepath):
    """
    Parse a JENDLTENDL01 Fx file and return the per-channel cross section.

    Returns (q_mev, e_mev_array, xs_barns_array).
    """
    with open(filepath) as fh:
        tokens = fh.read().split()

    v2 = float(tokens[2])
    if abs(v2) < 1000:
        q_ev = float(tokens[4])
        n_pts = int(tokens[6])
        data_start = 7
    else:
        q_ev = v2
        n_pts = int(tokens[4])
        data_start = 5

    e_list = []
    xs_list = []
    for i in range(n_pts):
        e_list.append(float(tokens[data_start + 2 * i]))
        xs_list.append(float(tokens[data_start + 2 * i + 1]))

    return q_ev / 1e6, np.array(e_list) / 1e6, np.array(xs_list)


def _parse_jendltendl01_f01_sections(
        filepath: str
) -> List[Tuple[int, float, np.ndarray, np.ndarray]]:
    """
    Parse the multi-section JENDLTENDL01 F01 file.

    The F01 file contains stacked MT sections: a total cross section (MT=4),
    per-level discrete cross sections (MT=50-90), and a continuum section
    (MT=91).  Each MT section appears twice: first as a cross-section table,
    then as a secondary-distribution table.  Only the cross-section occurrences
    are returned.

    The XS occurrence of each MT section has the token pattern:
        MT  0
        Q_eV  0  N_pts
        E1 XS1  E2 XS2  ...  (N_pts pairs)

    The distribution occurrence has a different header (second token != 0), so
    it is skipped automatically.

    Returns:
        List of (mt, q_ev, e_mev_array, xs_barns_array) for each discrete
        level MT (50-90 inclusive).  MT=4 (total) and MT=91 (continuum) are
        excluded.
    """
    with open(filepath) as fh:
        tokens = fh.read().split()

    result = []
    i = 0
    while i < len(tokens) - 4:
        try:
            mt = int(tokens[i])
        except ValueError:
            i += 1
            continue
        if tokens[i + 1] != '0':
            i += 1
            continue
        if not (50 <= mt <= 90):
            i += 1
            continue
        try:
            q_ev = float(tokens[i + 2])
            zero_check = tokens[i + 3]
            n_pts = int(tokens[i + 4])
        except (ValueError, IndexError):
            i += 1
            continue
        if zero_check != '0':
            i += 1
            continue
        if n_pts <= 0 or i + 5 + 2 * n_pts > len(tokens):
            i += 1
            continue
        e_list = []
        xs_list = []
        for k in range(n_pts):
            e_list.append(float(tokens[i + 5 + 2 * k]))
            xs_list.append(float(tokens[i + 5 + 2 * k + 1]))
        result.append((
            mt,
            q_ev,
            np.array(e_list) / 1e6,
            np.array(xs_list),
        ))
        i += 5 + 2 * n_pts

    return result


def _get_jendltendl01_extra_levels(
        zaid: int,
        jendl_level_energies: List[float],
        data_dir: Optional[os.PathLike] = None
) -> Optional[Tuple[List[float], Dict[int, Dict[float, float]]]]:
    """
    Parse JENDLTENDL01 Fx files for channels with level energies above the JENDL ceiling.

    Only levels whose derived excitation energy exceeds the highest JENDL level
    (by more than a tolerance) are returned, avoiding double-counting.

    Args:
        zaid: Target ZAID in ZZZAAA format.
        jendl_level_energies: List of level energies (MeV) already covered by JENDL.
        data_dir: Override for the data root directory.

    Returns:
        (extra_level_energies_mev, extra_level_xs) where extra_level_xs is
        {level_idx: {alpha_energy_mev: xs_barns}}, with indices continuing from
        len(jendl_level_energies). Returns None if no JENDLTENDL01 data is available.
    """
    if zaid not in _JENDLTENDL01_NAME_MAP:
        return None

    name = _JENDLTENDL01_NAME_MAP[zaid]
    if data_dir is None:
        jt_dir = os.path.join(_default_data_root(), 'an_xs', 'JENDLTENDL01')
    else:
        jt_dir = os.path.join(str(data_dir), 'an_xs', 'JENDLTENDL01')

    f01_path = os.path.join(jt_dir, 'F01', name)
    if not os.path.exists(f01_path):
        return None

    try:
        q_gs, _, _ = _parse_jendltendl01_channel_xs(f01_path)
    except Exception:
        return None

    jendl_ceiling = max(jendl_level_energies) if jendl_level_energies else 0.0
    tolerance = 0.1

    z_target = zaid // 1000
    a_target = zaid % 1000
    product_zaid = (z_target + 2) * 1000 + (a_target + 3)
    s_n = _compute_neutron_sep_energy(product_zaid)
    s_n_limit = s_n - 0.2 if s_n is not None else float('inf')

    extra_energies = []
    extra_xs_dict = {}
    next_idx = len(jendl_level_energies)

    try:
        f01_sections = _parse_jendltendl01_f01_sections(f01_path)
    except Exception:
        f01_sections = []

    for mt, q_ev, e_mev, xs_barns in f01_sections:
        q_k = q_ev / 1e6
        e_level = q_gs - q_k
        if e_level <= jendl_ceiling + tolerance:
            continue
        if e_level >= s_n_limit:
            continue
        if any(abs(e_level - e_ex) < tolerance for e_ex in extra_energies):
            continue
        xs_at_e = {float(e): float(xs) for e, xs in zip(e_mev, xs_barns)}
        extra_energies.append(e_level)
        extra_xs_dict[next_idx] = xs_at_e
        next_idx += 1

    for fx in range(1, 37):
        fxdir = f'F{fx:02d}'
        if fxdir == 'F01':
            continue
        path = os.path.join(jt_dir, fxdir, name)
        if not os.path.exists(path):
            continue
        try:
            q_k, e_mev, xs_barns = _parse_jendltendl01_channel_xs(path)
        except Exception:
            continue

        e_level = q_gs - q_k
        if e_level <= jendl_ceiling + tolerance:
            continue
        if e_level >= s_n_limit:
            continue

        if any(abs(e_level - e_ex) < tolerance for e_ex in extra_energies):
            continue

        xs_at_e = {float(e): float(xs) for e, xs in zip(e_mev, xs_barns)}
        extra_energies.append(e_level)
        extra_xs_dict[next_idx] = xs_at_e
        next_idx += 1

    if not extra_energies:
        return None

    return extra_energies, extra_xs_dict


def _extend_branching_with_jendltendl01(
        zaid: int,
        q_value: float,
        level_energies: List[float],
        branching_data: Dict[float, List[float]],
        data_dir: Optional[os.PathLike] = None
) -> Tuple[float, List[float], Dict[float, List[float]]]:
    """
    Extend JENDL branching data with higher-level channels from JENDLTENDL01.

    The extra JENDLTENDL01 channels (above the JENDL level ceiling) are added to the
    level list and their per-energy cross sections are folded into the branching
    fractions.  Cross sections for all channels (JENDL + extra) are summed to form
    the new total, so fractions are renormalised consistently.

    Args:
        zaid: Target ZAID.
        q_value: Existing ground-state Q-value (MeV).
        level_energies: Level energies already in the JENDL result (MeV).
        branching_data: Existing branching fractions {E_MeV: [fraction_per_level]}.
        data_dir: Data root override.

    Returns:
        (q_value, extended_level_energies, extended_branching_data)
    """
    result = _get_jendltendl01_extra_levels(zaid, level_energies, data_dir)
    if result is None:
        return q_value, level_energies, branching_data

    extra_level_energies, extra_level_xs = result
    n_jendl = len(level_energies)
    extended_levels = list(level_energies) + extra_level_energies

    extended_branching = {}
    for e_alpha, fracs in branching_data.items():
        existing_xs = list(fracs)
        total_existing = sum(existing_xs)

        extra_xs_at_e = []
        for idx in sorted(extra_level_xs.keys()):
            xs_dict = extra_level_xs[idx]
            es = sorted(xs_dict.keys())
            if es and e_alpha >= min(es):
                if e_alpha in xs_dict:
                    extra_xs_at_e.append(xs_dict[e_alpha])
                else:
                    vals = [xs_dict[ee] for ee in es]
                    try:
                        f = interp1d(es, vals, kind='linear',
                                     bounds_error=False, fill_value=0.0)
                        extra_xs_at_e.append(float(f(e_alpha)))
                    except Exception:
                        extra_xs_at_e.append(0.0)
            else:
                extra_xs_at_e.append(0.0)

        extra_total = sum(extra_xs_at_e)
        grand_total = total_existing + extra_total

        if grand_total <= 0:
            extended_branching[e_alpha] = (
                list(fracs) + [0.0] * len(extra_level_energies)
            )
            continue

        if total_existing > 0:
            scale = total_existing / grand_total
            new_jendl = [f * scale for f in existing_xs]
        else:
            new_jendl = list(existing_xs)

        new_extra = [x / grand_total for x in extra_xs_at_e]
        extended_branching[e_alpha] = new_jendl + new_extra

    return q_value, extended_levels, extended_branching


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


def _get_mt91_continuum_xs(root) -> Optional[Dict[float, float]]:
    mt91 = root.find(".//reaction[@ENDF_MT='91']")
    if mt91 is None:
        return None
    try:
        xs = _get_cross_section_from_reaction(mt91)
        return xs if xs else None
    except Exception:
        return None


def _parse_ripl3_higher_levels(
        filepath: os.PathLike,
        product_a: int,
        min_energy_mev: float,
        max_energy_mev: float,
) -> List[float]:
    try:
        with open(filepath, 'r') as fh:
            lines = fh.readlines()
    except Exception:
        return []

    result: List[float] = []
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
        if a_val != product_a:
            i += 1
            continue

        i += 1
        while i < len(lines):
            line = lines[i]
            if len(line) < 14:
                i += 1
                continue
            if line[0] not in (' ', '\t'):
                break
            try:
                a_check = int(line[5:10].strip())
                if a_check != product_a:
                    break
            except (ValueError, IndexError):
                pass
            try:
                elv = float(line[4:14].strip())
                ng = int(line[34:37].strip()) if len(line) > 36 and line[34:37].strip() else 0
            except (ValueError, IndexError):
                i += 1
                continue
            if min_energy_mev < elv < max_energy_mev:
                result.append(elv)
            i += ng + 1
        break

    return result


def _extend_level_xs_with_mt91_continuum(
        zaid: int,
        q_value: float,
        level_energies: Dict[int, float],
        level_cross_sections: Dict[int, Dict[float, float]],
        root,
        data_dir: Optional[os.PathLike],
) -> Tuple[Dict[int, float], Dict[int, Dict[float, float]]]:
    xs91 = _get_mt91_continuum_xs(root)
    if not xs91:
        return level_energies, level_cross_sections

    z_target = zaid // 1000
    a_target = zaid % 1000
    product_a = a_target + 3
    product_z = z_target + 2
    product_zaid = product_z * 1000 + product_a

    if data_dir is None:
        data_root = _default_data_root()
    else:
        data_root = str(data_dir)

    ripl3_path = os.path.join(data_root, 'levels', f'z{product_z:03d}.dat')
    if not os.path.exists(ripl3_path):
        return level_energies, level_cross_sections

    s_n = _compute_neutron_sep_energy(product_zaid)
    max_level_e = (s_n - 0.1) if s_n is not None else 20.0

    jendl_ceiling = max(level_energies.values()) if level_energies else 0.0
    higher_levels = _parse_ripl3_higher_levels(
        ripl3_path, product_a, jendl_ceiling, max_level_e
    )
    if not higher_levels:
        return level_energies, level_cross_sections

    existing_energies = list(level_energies.values())
    dedup_tolerance = 0.01
    higher_levels = [
        e for e in higher_levels
        if not any(abs(e - ex) < dedup_tolerance for ex in existing_energies)
    ]
    if not higher_levels:
        return level_energies, level_cross_sections

    next_idx = max(level_energies.keys()) + 1
    new_level_energies: Dict[int, float] = dict(level_energies)
    new_level_cross_sections: Dict[int, Dict[float, float]] = dict(level_cross_sections)
    extra_indices: List[int] = []

    for e_level in higher_levels:
        if e_level <= q_value:
            threshold = 0.0
        else:
            threshold = (e_level - q_value) * (a_target + 4) / a_target
        xs_at: Dict[float, float] = {}
        for e_alpha, xs_val in xs91.items():
            if e_alpha < threshold or xs_val <= 0.0:
                continue
            xs_at[e_alpha] = xs_val
        if xs_at:
            new_level_energies[next_idx] = e_level
            new_level_cross_sections[next_idx] = xs_at
            extra_indices.append(next_idx)
            next_idx += 1

    if not extra_indices:
        return level_energies, level_cross_sections

    for idx in extra_indices:
        xs_dict = new_level_cross_sections[idx]
        for e_alpha in list(xs_dict.keys()):
            count = sum(
                1 for j in extra_indices
                if j in new_level_cross_sections and e_alpha in new_level_cross_sections[j]
            )
            if count > 0:
                xs_dict[e_alpha] /= count

    return new_level_energies, new_level_cross_sections


def get_mt91_extra_levels(
        zaid: int,
        q_value: float,
        jendl_level_energies: List[float],
        data_dir: Optional[os.PathLike] = None,
) -> Optional[List[Tuple[float, float]]]:
    """
    Return RIPL-3 discrete levels above the JENDL ceiling for use in gamma calculation.

    These are the product-nucleus levels that receive population from the MT=91
    continuum (alpha,n) channel.  Each entry is (level_energy_MeV, threshold_MeV)
    where threshold is the minimum incident alpha energy needed to reach that level.

    Args:
        zaid: Target ZAID.
        q_value: Ground-state Q-value for the (alpha,n) reaction (MeV).
        jendl_level_energies: Level energies already present in JENDL data (MeV).
        data_dir: Data directory override.

    Returns:
        List of (level_energy_MeV, threshold_MeV) tuples, or None if no RIPL-3
        data is found or no extra levels exist above the JENDL ceiling.
    """
    z_target = zaid // 1000
    a_target = zaid % 1000
    product_a = a_target + 3
    product_z = z_target + 2
    product_zaid = product_z * 1000 + product_a

    if data_dir is None:
        data_root = _default_data_root()
    else:
        data_root = str(data_dir)

    ripl3_path = os.path.join(data_root, 'levels', f'z{product_z:03d}.dat')
    if not os.path.exists(ripl3_path):
        return None

    s_n = _compute_neutron_sep_energy(product_zaid)
    max_level_e = (s_n - 0.1) if s_n is not None else 20.0

    jendl_ceiling = max(jendl_level_energies) if jendl_level_energies else 0.0
    higher_levels = _parse_ripl3_higher_levels(
        ripl3_path, product_a, jendl_ceiling, max_level_e
    )
    if not higher_levels:
        return None

    existing_energies = list(jendl_level_energies)
    dedup_tolerance = 0.01
    higher_levels = [
        e for e in higher_levels
        if not any(abs(e - ex) < dedup_tolerance for ex in existing_energies)
    ]
    if not higher_levels:
        return None

    result = []
    for e_level in higher_levels:
        if e_level <= q_value:
            threshold = 0.0
        else:
            threshold = (e_level - q_value) * (a_target + 4) / a_target
        result.append((e_level, threshold))
    return result


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
            if level_idx not in level_energies and ground_state_q_value != 0.0:
                q_elem = reaction.find(".//Q/constant1d")
                if q_elem is not None:
                    try:
                        q_mt_mev = float(q_elem.get('value', 0)) / 1e6
                        derived = ground_state_q_value - q_mt_mev
                        if derived > 0.0:
                            level_energies[level_idx] = derived
                    except (TypeError, ValueError):
                        pass

    return level_energies, level_cross_sections, ground_state_q_value


def _open_possibly_compressed(filepath: str) -> str:
    """Return the decoded text content of a file that may be gzip, raw zlib, or plain text.

    Args:
        filepath: Path to the file to open.

    Returns:
        Decoded string contents of the file.

    Raises:
        OSError: If the file cannot be opened by any method.
    """
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as fh:
            return fh.read()
    except (OSError, EOFError, gzip.BadGzipFile):
        pass
    try:
        with open(filepath, 'rb') as fh:
            raw = fh.read()
        return _zlib.decompress(raw).decode('utf-8', errors='replace')
    except (_zlib.error, Exception):
        pass
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        return fh.read()


def _find_f02_file(target_zaid: int, data_dir: Optional[str] = None) -> Optional[str]:
    """Return the path to the G4TENDL F02 photon-production file for target_zaid, or None.

    Searches the bundled data directory first, then falls back to the directory
    pointed to by the G4PARTICLEHPDATA environment variable.

    Args:
        target_zaid: ZAID in ZZZAAA format.
        data_dir: Override for the data root directory. None uses the default.

    Returns:
        Absolute path to the first matching F02 file, or None if not found.
    """
    z = target_zaid // 1000
    a = target_zaid % 1000
    prefix = f"{z}_{a}_"

    if data_dir is None:
        data_root = _default_data_root()
    else:
        data_root = str(data_dir)

    for search_root in [data_root, os.environ.get(_G4HPDATA_ENV, "")]:
        if not search_root:
            continue
        matches = glob.glob(os.path.join(search_root, "Alpha/Inelastic/F02", prefix + "*"))
        if matches:
            return matches[0]

    return None


def _parse_f02_file(
        filepath: str,
) -> Optional[Tuple[Dict[float, float], Dict[float, List[Tuple[float, float]]]]]:
    """Parse a G4TENDL F02 photon-production file.

    The file format has three sections:
      Section 1 (0 0 n header): total inelastic cross section table (skipped).
      Section 2 (n 1 n 2 header): photon multiplicity per reaction vs alpha energy.
      Section 3 (per alpha-energy records): discrete gamma spectrum entries.

    Args:
        filepath: Path to the F02 file (may be gzip-compressed, raw zlib-compressed,
                  or plain text).

    Returns:
        Tuple of (mult_dict, spectrum) where mult_dict maps alpha energy in MeV to
        photon multiplicity, and spectrum maps alpha energy in MeV to a list of
        (gamma_energy_MeV, weight) pairs. Returns None if parsing fails.
    """
    try:
        content = _open_possibly_compressed(filepath)
    except Exception as exc:
        logger.debug("Failed to open F02 file %s: %s", filepath, exc)
        return None

    try:
        tokens = content.split()
        floats = []
        for t in tokens:
            try:
                floats.append(float(t))
            except ValueError:
                floats.append(None)

        n = len(floats)
        i = 0

        section1_found = False
        while i < n - 2:
            if (floats[i] == 0.0 and floats[i + 1] == 0.0
                    and floats[i + 2] is not None
                    and 2 <= floats[i + 2] <= 2000):
                n_pairs = int(floats[i + 2])
                i += 3 + 2 * n_pairs
                section1_found = True
                break
            i += 1

        if not section1_found:
            return None

        mult_dict = {}
        while i < n - 3:
            f0 = floats[i]
            f1 = floats[i + 1]
            f2 = floats[i + 2]
            f3 = floats[i + 3]
            if (f0 is not None and f1 == 1.0 and f2 is not None
                    and f2 == f0 and f3 == 2.0 and 2 <= f0 <= 500):
                n_mult = int(f0)
                i += 4
                for _ in range(n_mult):
                    if i + 1 >= n:
                        break
                    e_ev = floats[i]
                    mult = floats[i + 1]
                    if e_ev is not None and mult is not None and e_ev > 0:
                        mult_dict[e_ev / 1e6] = mult
                    i += 2
                break
            i += 1

        if not mult_dict:
            return None

        spectrum = {}
        while i < n - 3:
            f0 = floats[i]
            f1 = floats[i + 1]
            f2 = floats[i + 2]
            f3 = floats[i + 3]
            if (f0 is not None and f0 > 1e3
                    and f1 is not None and 1.0 <= f1 <= 200.0
                    and f1 == round(f1)
                    and f2 == 0.0 and f3 == 2.0):
                e_alpha_mev = f0 / 1e6
                n_gammas = int(f1)
                i += 4
                pairs = []
                for _ in range(n_gammas):
                    if i + 2 >= n:
                        break
                    e_g = floats[i]
                    weight = floats[i + 1]
                    angle = floats[i + 2]
                    if (e_g is not None and weight is not None
                            and angle == 0.0 and e_g >= 0.0 and weight > 1e-30):
                        pairs.append((e_g / 1e6, weight))
                    i += 3
                if pairs:
                    spectrum[e_alpha_mev] = pairs
            else:
                i += 1

        return mult_dict, spectrum

    except Exception as exc:
        logger.debug("Failed to parse F02 file %s: %s", filepath, exc)
        return None


def get_f02_gamma_data(
        target_zaid: int,
        data_dir: Optional[str] = None
) -> Optional[Tuple[Dict[float, float], Dict[float, List[Tuple[float, float]]]]]:
    """Return parsed F02 gamma data for target_zaid, or None if no file is found.

    Args:
        target_zaid: ZAID in ZZZAAA format.
        data_dir: Override for the data root directory. None uses the default.

    Returns:
        Tuple of (mult_dict, spectrum) as returned by _parse_f02_file, or None
        if no F02 file exists for this target or if parsing fails.
    """
    f02_path = _find_f02_file(target_zaid, data_dir)
    if f02_path is None:
        return None
    result = _parse_f02_file(f02_path)
    if result is None:
        logger.warning("F02 file found for ZAID %d but could not be parsed: %s",
                       target_zaid, f02_path)
        return None
    return result


_AN_PRIMARY_MTS = frozenset(
    [2, 4, 5, 11, 91, 201]
    + list(range(50, 92))
)


def _get_multiplicity_from_sum(elem: ET.Element) -> Optional[Dict[float, float]]:
    """Extract a photon multiplicity table from a GNDS multiplicitySum XML element.

    Args:
        elem: XML element containing a multiplicitySum node.

    Returns:
        Dict mapping alpha energy in MeV to photon multiplicity, or None if no
        valid data is found.
    """
    for xys in elem.iter():
        vals_elem = None
        for child in xys.iter():
            if 'values' in child.tag:
                vals_elem = child
                break
        if vals_elem is not None and vals_elem.text and vals_elem.text.strip():
            vals = vals_elem.text.split()
            if len(vals) >= 4 and len(vals) % 2 == 0:
                try:
                    floats = [float(v) for v in vals]
                except ValueError:
                    continue
                ee = np.array(floats[0::2]) / 1e6
                mv = np.array(floats[1::2])
                if np.any(mv > 0):
                    return dict(zip(ee, mv))
    return None


def _find_tendl_xml(target_zaid: int) -> Optional[str]:
    """Return the path to the TENDL GNDS XML file for target_zaid, or None if absent.

    Args:
        target_zaid: ZAID in ZZZAAA format.

    Returns:
        Absolute path to the TENDL XML file, or None if not found.
    """
    data_root = _default_data_root()
    path = os.path.join(data_root, 'an_xs', 'TENDL', f'{target_zaid}.xml')
    return path if os.path.exists(path) else None


def _load_tendl_gamma_channels(
        target_zaid: int,
        exclude_mts: Optional[frozenset] = None,
) -> List[Tuple[Dict[float, float], Dict[float, float]]]:
    """Load per-MT gamma production channels from the TENDL GNDS XML for target_zaid.

    For each reaction MT that has a multiplicitySum element and a cross section,
    returns a (xs_dict, mult_dict) pair where xs_dict maps alpha energy in MeV to
    cross section in barns and mult_dict maps alpha energy in MeV to photon multiplicity.

    Args:
        target_zaid: ZAID in ZZZAAA format.
        exclude_mts: Set of ENDF MT numbers to skip. None includes all MTs.

    Returns:
        List of (xs_dict, mult_dict) pairs, one per MT that has both cross section
        and multiplicity data. Returns an empty list if the XML file is absent or
        if no qualifying MTs are found.
    """
    xml_path = _find_tendl_xml(target_zaid)
    if xml_path is None:
        return []
    try:
        tree = ET.parse(xml_path)
    except Exception as exc:
        logger.debug("Failed to parse TENDL XML ZAID %d: %s", target_zaid, exc)
        return []
    root = tree.getroot()

    mult_by_mt = {}
    for elem in root.iter():
        if _XML_TAG_MULT_SUM not in elem.tag:
            continue
        mt_str = elem.get(_XML_ATTR_MT)
        if mt_str is None:
            continue
        try:
            mt = int(mt_str)
        except ValueError:
            continue
        if exclude_mts is not None and mt in exclude_mts:
            continue
        mult = _get_multiplicity_from_sum(elem)
        if mult is not None:
            mult_by_mt[mt] = mult

    if not mult_by_mt:
        return []

    xs_by_mt = {}
    for elem in root.iter():
        if _XML_TAG_REACTION not in elem.tag:
            continue
        mt_str = elem.get(_XML_ATTR_MT)
        if mt_str is None:
            continue
        try:
            mt = int(mt_str)
        except ValueError:
            continue
        if mt not in mult_by_mt:
            continue
        try:
            xs = _get_cross_section_from_reaction(elem)
        except Exception:
            xs = None
        if xs:
            xs_by_mt[mt] = xs

    return [
        (xs_by_mt[mt], mult)
        for mt, mult in mult_by_mt.items()
        if mt in xs_by_mt
    ]


def get_secondary_gamma_channels(
        target_zaid: int,
) -> List[Tuple[Dict[float, float], Dict[float, float]]]:
    """Return TENDL gamma channels that are NOT covered by the primary F02*sigma_an calculation.

    Excludes MTs in _AN_PRIMARY_MTS: elastic (MT=2), (alpha,n) channels already
    captured by F02 (MT=4, 50-91), and MT=5/11/201 bookkeeping entries.
    All other reaction channels — including mixed neutron+gamma channels such as
    (alpha,na) MT=22, (alpha,np) MT=28, (alpha,2n) MT=16 — are included, since
    their gamma yields are not present in the F02 data.

    Args:
        target_zaid: ZAID in ZZZAAA format.

    Returns:
        List of (xs_dict, mult_dict) pairs for non-primary gamma-producing MTs.
    """
    return _load_tendl_gamma_channels(target_zaid, exclude_mts=_AN_PRIMARY_MTS)


def get_tendl_all_gamma_channels(
        target_zaid: int,
) -> List[Tuple[Dict[float, float], Dict[float, float]]]:
    """Return all TENDL gamma production channels for target_zaid (no MT filtering).

    Used when replacing F02*sigma_an with a full TENDL per-channel sum.

    Args:
        target_zaid: ZAID in ZZZAAA format.

    Returns:
        List of (xs_dict, mult_dict) pairs for all MTs with gamma multiplicity data.
    """
    return _load_tendl_gamma_channels(target_zaid)


_TENDL_PRIMARY_RATIO_LO = 1.5
_TENDL_PRIMARY_RATIO_HI = 3.0
_TENDL_PRIMARY_MT4_THRESH = 0.1
_TENDL_PRIMARY_E_REF = 8.0


def tendl_is_primary_gamma_mode(
        target_zaid: int,
        f02_mult_dict: Dict[float, float],
        an_xs_dict: Dict[float, float],
) -> bool:
    """Return True if the TENDL per-channel gamma sum should replace F02*sigma_an.

    The check evaluates at a reference alpha energy of _TENDL_PRIMARY_E_REF MeV:
      1. The ratio TENDL_total(xs*mult) / F02(sigma_an*mult) must be in
         (_TENDL_PRIMARY_RATIO_LO, _TENDL_PRIMARY_RATIO_HI).
      2. The MT=4 (alpha,n inelastic) multiplicity at the reference energy must
         exceed _TENDL_PRIMARY_MT4_THRESH, indicating TENDL has adequate coverage
         of the primary channel across the slowing-down integral range.

    Currently returns False for all benchmarked targets; the infrastructure is
    retained for future targets with more complete TENDL gamma data.

    Args:
        target_zaid: ZAID in ZZZAAA format.
        f02_mult_dict: Dict mapping alpha energy in MeV to F02 photon multiplicity.
        an_xs_dict: Dict mapping alpha energy in MeV to sigma_an in barns.

    Returns:
        True if TENDL per-channel mode is valid for this target, False otherwise.
    """
    xml_path = _find_tendl_xml(target_zaid)
    if xml_path is None:
        return False

    an_e = np.array(sorted(an_xs_dict.keys()))
    an_v = np.array([an_xs_dict[e] for e in an_e])
    an_ref = float(np.interp(_TENDL_PRIMARY_E_REF, an_e, an_v, left=0.0, right=0.0))

    f02_e = np.array(sorted(f02_mult_dict.keys()))
    f02_v = np.array([f02_mult_dict[e] for e in f02_e])
    f02_m_ref = float(np.interp(_TENDL_PRIMARY_E_REF, f02_e, f02_v))
    f02_xm = an_ref * f02_m_ref
    if f02_xm < 1e-10:
        return False

    try:
        tree = ET.parse(xml_path)
    except Exception:
        return False
    root = tree.getroot()

    mult_by_mt = {}
    for elem in root.iter():
        if _XML_TAG_MULT_SUM not in elem.tag:
            continue
        mt_str = elem.get(_XML_ATTR_MT)
        if mt_str is None:
            continue
        try:
            mt = int(mt_str)
        except ValueError:
            continue
        mult = _get_multiplicity_from_sum(elem)
        if mult is not None:
            mult_by_mt[mt] = mult

    xs_by_mt = {}
    for elem in root.iter():
        if _XML_TAG_REACTION not in elem.tag:
            continue
        mt_str = elem.get(_XML_ATTR_MT)
        if mt_str is None:
            continue
        try:
            mt = int(mt_str)
        except ValueError:
            continue
        if mt not in mult_by_mt:
            continue
        try:
            xs = _get_cross_section_from_reaction(elem)
        except Exception:
            xs = None
        if xs:
            xs_by_mt[mt] = xs

    tendl_total = 0.0
    mt4_mult_ref = 0.0
    for mt, mult in mult_by_mt.items():
        if mt not in xs_by_mt:
            continue
        me = np.array(sorted(mult.keys()))
        mv = np.array([mult[e] for e in me])
        m_val = float(np.interp(_TENDL_PRIMARY_E_REF, me, mv, left=0.0, right=mv[-1]))
        xs = xs_by_mt[mt]
        xe = np.array(sorted(xs.keys()))
        xv = np.array([xs[e] for e in xe])
        x_val = float(np.interp(_TENDL_PRIMARY_E_REF, xe, xv))
        tendl_total += x_val * m_val
        if mt == 4:
            mt4_mult_ref = m_val

    ratio = tendl_total / f02_xm
    return (
        _TENDL_PRIMARY_RATIO_LO < ratio < _TENDL_PRIMARY_RATIO_HI
        and mt4_mult_ref > _TENDL_PRIMARY_MT4_THRESH
    )
