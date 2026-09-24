"""Canonical ALPHANSO calculation metadata and numerical defaults.

This module is intentionally independent of the GUI.  The transport engine,
command-line adapters, and graphical interface all import these definitions so
that a release cannot silently present different defaults in different places.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_NUM_ALPHA_GROUPS = 15_000
DEFAULT_MIN_ALPHA_ENERGY = 1.0e-11
DEFAULT_MAX_ALPHA_ENERGY = 15.0
DEFAULT_NEUTRON_ENERGY_BINS = (15.0, 0.0, 101)
DEFAULT_N_ANGULAR_BINS = 40
DEFAULT_CALCULATE_GAMMAS = True
DEFAULT_SAVE_DATA_FILES = True


CALCULATION_TYPES: dict[str, dict[str, str]] = {
    "beam": {
        "label": "Beam",
        "description": (
            "Define a mono- or polyenergetic alpha beam and thick target material."
        ),
        "short_description": "Mono- or polyenergetic beam",
        "visual": "beam",
    },
    "homogeneous": {
        "label": "Homogeneous",
        "description": (
            "Define the uniform source mixture by isotope and mass fraction."
        ),
        "short_description": "Uniform source and target mixture",
        "visual": "mix",
    },
    "interface": {
        "label": "Interface",
        "description": (
            "Define the alpha-emitting source and the adjoining target region."
        ),
        "short_description": "Planar source-target boundary",
        "visual": "interface",
    },
    "sandwich": {
        "label": "Sandwich",
        "description": (
            "Build the source, intermediate layers, and final target region."
        ),
        "short_description": "Source, layers, and target",
        "visual": "sandwich",
    },
}

CONFIG_FIELD_GROUPS: dict[str, dict[str, str]] = {
    "Data sources": {
        "description": (
            "Leave a path blank to use the dataset installed with ALPHANSO. "
            "Set a directory only when overriding that data source."
        ),
    },
}


COMMON_CONFIG_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "key": "min_alpha_energy",
        "label": "Minimum alpha energy",
        "kind": "number",
        "default": DEFAULT_MIN_ALPHA_ENERGY,
        "group": "Alpha energy grid",
        "unit": "MeV",
        "minimum": 0,
        "step": "any",
        "advanced": True,
    },
    {
        "key": "max_alpha_energy",
        "label": "Maximum alpha energy",
        "kind": "number",
        "default": DEFAULT_MAX_ALPHA_ENERGY,
        "group": "Alpha energy grid",
        "unit": "MeV",
        "minimum": 0,
        "step": 0.1,
        "greater_than": "min_alpha_energy",
        "advanced": True,
    },
    {
        "key": "num_alpha_groups",
        "label": "Alpha energy groups",
        "kind": "integer",
        "default": DEFAULT_NUM_ALPHA_GROUPS,
        "group": "Alpha energy grid",
        "minimum": 100,
        "step": 100,
        "advanced": True,
    },
    {
        "key": "neutron_energy_bins",
        "label": "Neutron energy bins",
        "kind": "range_points",
        "default": list(DEFAULT_NEUTRON_ENERGY_BINS),
        "group": "Neutron energy grid",
        "unit": "MeV",
        "parts": (
            {"label": "start", "kind": "number", "step": 0.1},
            {"label": "stop", "kind": "number", "step": 0.1},
            {"label": "num_points", "kind": "integer", "minimum": 2, "step": 1},
        ),
        "advanced": True,
    },
    {
        "key": "n_angular_bins",
        "label": "Angular bins",
        "kind": "integer",
        "default": DEFAULT_N_ANGULAR_BINS,
        "group": "Transport discretization",
        "minimum": 1,
        "step": 1,
        "applies_to": ("sandwich",),
        "advanced": True,
    },
    {
        "key": "calculate_gammas",
        "label": "Calculate prompt gamma production",
        "kind": "boolean",
        "default": DEFAULT_CALCULATE_GAMMAS,
        "group": "Physics",
        "description": "Calculate prompt gamma yield and discrete lines.",
    },
    {
        "key": "output_dir",
        "label": "Output directory",
        "kind": "string",
        "default": "",
        "group": "Output",
        "placeholder": "alphanso_output/run_name",
        "browse": "directory",
        "omit_when_empty": True,
    },
    {
        "key": "save_data_files",
        "label": "Save data files",
        "kind": "boolean",
        "default": DEFAULT_SAVE_DATA_FILES,
        "group": "Output",
        "description": "Write results.yaml when an output directory is set.",
    },
    {
        "key": "an_xs_data_dir",
        "label": "(alpha,n) cross-section data directory",
        "kind": "string",
        "default": "",
        "group": "Data sources",
        "placeholder": "path/to/an_xs_data",
        "browse": "directory",
        "advanced": True,
        "omit_when_empty": True,
    },
    {
        "key": "stopping_power_data_dir",
        "label": "Stopping-power data directory",
        "kind": "string",
        "default": "",
        "group": "Data sources",
        "placeholder": "path/to/stopping_power_data",
        "browse": "directory",
        "advanced": True,
        "omit_when_empty": True,
    },
    {
        "key": "decay_data_dir",
        "label": "Decay data directory",
        "kind": "string",
        "default": "",
        "group": "Data sources",
        "placeholder": "path/to/decay_data",
        "browse": "directory",
        "advanced": True,
        "omit_when_empty": True,
    },
    {
        "key": "gamma_data_dir",
        "label": "Gamma-production data directory",
        "kind": "string",
        "default": "",
        "group": "Data sources",
        "placeholder": "path/to/gamma_data",
        "browse": "directory",
        "advanced": True,
        "omit_when_empty": True,
    },
)


def common_config_defaults() -> dict[str, Any]:
    """Return a new mapping of all shared calculation defaults."""
    return {
        field["key"]: deepcopy(field["default"])
        for field in COMMON_CONFIG_FIELDS
    }


def configuration_schema() -> dict[str, Any]:
    """Return the serializable schema consumed by external interfaces."""
    return {
        "calculation_types": deepcopy(CALCULATION_TYPES),
        "field_groups": deepcopy(CONFIG_FIELD_GROUPS),
        "common_fields": deepcopy(COMMON_CONFIG_FIELDS),
    }


def validate_common_config(config: dict[str, Any]) -> list[str]:
    """Validate fields shared by all ALPHANSO calculation types."""
    errors: list[str] = []
    groups = config.get("num_alpha_groups")
    if groups is not None and (
        isinstance(groups, bool) or not isinstance(groups, int) or groups < 100
    ):
        errors.append("Alpha energy groups must be an integer of at least 100.")

    minimum = config.get("min_alpha_energy")
    maximum = config.get("max_alpha_energy")
    if minimum is not None and (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math_is_finite(minimum)
        or minimum < 0
    ):
        errors.append("Minimum alpha energy must be a non-negative number.")
    if maximum is not None and (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math_is_finite(maximum)
    ):
        errors.append("Maximum alpha energy must be a finite number.")
    if (
        isinstance(minimum, (int, float))
        and not isinstance(minimum, bool)
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and maximum <= minimum
    ):
        errors.append("Maximum alpha energy must be greater than minimum alpha energy.")

    bins = config.get("neutron_energy_bins")
    if bins is not None:
        if not isinstance(bins, list) or len(bins) != 3:
            errors.append("Neutron energy bins must be [start, stop, points].")
        else:
            start, stop, points = bins
            bounds_are_numeric = all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math_is_finite(item)
                for item in (start, stop)
            )
            if not bounds_are_numeric or float(start) == float(stop):
                errors.append("Neutron energy bounds must be distinct numeric values.")
            if isinstance(points, bool) or not isinstance(points, int) or points < 2:
                errors.append("Neutron energy points must be an integer of at least 2.")

    if config.get("calc_type") == "sandwich":
        angular_bins = config.get("n_angular_bins")
        if angular_bins is not None and (
            isinstance(angular_bins, bool)
            or not isinstance(angular_bins, int)
            or angular_bins < 1
        ):
            errors.append("Angular bins must be a positive integer.")
    return errors


def math_is_finite(value: int | float) -> bool:
    """Avoid importing NumPy into this lightweight metadata module."""
    return value not in (float("inf"), float("-inf")) and value == value
