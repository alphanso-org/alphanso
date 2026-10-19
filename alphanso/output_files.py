from pathlib import Path

import numpy as np
import yaml


# (neutron unit, gamma unit) for the yields of each calculation type.
_YIELD_UNITS = {
    "beam": ("neutrons per incident alpha", "gammas per incident alpha"),
    "homogeneous": ("n/s/g", "gammas/s/g"),
    "interface": ("n/s/cm^2", "gammas/s/cm^2"),
    "sandwich": ("n/s/cm^2", "gammas/s/cm^2"),
}

_YIELD_NOTES = {
    "beam": "Yields are per incident alpha when beam_intensities sum to 1.",
    "homogeneous": "Yields are per gram of material.",
    "interface": "Yields are per cm^2 of interface area.",
    "sandwich": "Yields are per cm^2 of interface area.",
}

_ENERGY_BINS = "MeV (bin edges, ascending)"
_NORMALIZED_SPECTRUM = "fraction of neutrons in each energy bin (sums to 1)"

_SF_CONTRIBUTOR_UNITS = {
    "zaid": "identifier",
    "name": "identifier",
    "sf_yield": "n/s/g",
    "sf_branching": "fraction of decays that are spontaneous fission",
    "nubar": "neutrons per fission",
    "watt_a": "MeV",
    "watt_b": "1/MeV",
    "endf_avg_energy": "MeV",
    "nu_delayed": "neutrons per fission",
    "delayedn_yield": "n/s/g",
}


def result_units(calc_type: str) -> dict:
    """Return the unit of each top-level results key for a calculation type."""
    neutrons, gammas = _YIELD_UNITS[calc_type]
    gamma_lines = f"[energy (MeV), intensity ({gammas})] pairs"
    return {
        "an_yield": neutrons,
        "sf_yield": neutrons,
        "combined_yield": neutrons,
        "delayedn_strength": neutrons,
        "yield_target": neutrons,
        "yield_layers": f"{neutrons}, one value per intermediate layer",
        "yield_ab_b": neutrons,
        "yield_bc_b": neutrons,
        "yield_bc_c": neutrons,
        "an_spectrum": _NORMALIZED_SPECTRUM,
        "sf_spectrum": _NORMALIZED_SPECTRUM,
        "combined_spectrum": _NORMALIZED_SPECTRUM,
        "delayedn_spectrum": _NORMALIZED_SPECTRUM,
        "an_spectrum_absolute": f"{neutrons} per energy bin",
        "spectrum_layers": f"{neutrons} per energy bin, one spectrum per intermediate layer",
        "neutron_energy_bins": _ENERGY_BINS,
        "spectrum_energy_bins": _ENERGY_BINS,
        "average_energy": "MeV",
        "average_energy_an": "MeV",
        "average_energy_sf": "MeV",
        "gamma_yield": gammas,
        "gamma_yield_target": gammas,
        "gamma_yield_layers": f"{gammas}, one value per intermediate layer",
        "gamma_lines": gamma_lines,
        "gamma_lines_target": gamma_lines,
        "sf_contributors": "one entry per spontaneous-fission nuclide",
    }


def results_units_header(results: dict, calc_type: str) -> str:
    """Build a YAML comment block documenting the units of the keys in ``results``."""
    units = result_units(calc_type)
    lines = [f"# ALPHANSO results (calc_type: {calc_type})", "#", "# Units:"]
    for key in results:
        if key not in units:
            continue
        lines.append(f"#   {key}: {units[key]}")
        if key == "sf_contributors":
            lines.extend(f"#     {field}: {unit}" for field, unit in _SF_CONTRIBUTOR_UNITS.items())
    lines.extend(["#", f"# {_YIELD_NOTES[calc_type]}"])
    return "\n".join(lines) + "\n"


def normalize_results_payload(value):
    """Recursively convert results payloads to built-in YAML-safe Python types."""
    if isinstance(value, np.ndarray):
        return normalize_results_payload(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: normalize_results_payload(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize_results_payload(item) for item in value]

    return value


def write_results_yaml(results: dict, output_dir, filename: str = "results.yaml",
                       calc_type: str = None) -> Path:
    """Write a normalized results dictionary to a YAML file and return the file path.

    When ``calc_type`` is given, the file starts with a comment block listing
    the units of each result. YAML loaders ignore the comments.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_results = normalize_results_payload(results)
    results_file = output_path / filename
    with open(results_file, "w") as f:
        if calc_type in _YIELD_UNITS:
            f.write(results_units_header(normalized_results, calc_type))
        yaml.safe_dump(
            normalized_results,
            f,
            default_flow_style=True,
            indent=2,
            sort_keys=False,
        )

    return results_file
