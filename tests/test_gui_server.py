"""Focused tests for the local ALPHANSO GUI adapter."""

import ast
import json
from importlib.metadata import metadata
from pathlib import Path
from types import SimpleNamespace

import yaml

from alphanso.configuration import (
    CALCULATION_TYPES,
    DEFAULT_MAX_ALPHA_ENERGY,
    DEFAULT_MIN_ALPHA_ENERGY,
    DEFAULT_N_ANGULAR_BINS,
    DEFAULT_NEUTRON_ENERGY_BINS,
    DEFAULT_NUM_ALPHA_GROUPS,
)
from alphanso.gui_server import (
    _DesktopAPI,
    _bootstrap_payload,
    _gui_display_name,
    _server_url,
    config_for_transport,
    example_catalog,
    saved_configuration_catalog,
    validate_config,
    write_saved_configuration_catalog,
)


REPOSITORY_ROOT = Path(__file__).parents[1]


def _stringify_mapping_keys(value):
    if isinstance(value, dict):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def test_every_bundled_example_is_valid():
    examples = example_catalog()

    assert examples
    assert {example["config"]["calc_type"] for example in examples} == {
        "beam", "homogeneous", "interface", "sandwich"
    }
    for example in examples:
        assert validate_config(example["config"]) == [], example["id"]


def test_numeric_zaids_are_restored_after_json_encoding():
    prepared = config_for_transport({
        "calc_type": "homogeneous",
        "matdef": {"92235": 0.5, "92238": 0.35, "8000": 0.15},
    })

    assert prepared["matdef"] == {92235: 0.5, 92238: 0.35, 8000: 0.15}


def test_catalog_matches_every_repository_example():
    catalog = {
        example["source"]: example["config"]
        for example in example_catalog()
    }
    expected_sources = set()
    example_root = REPOSITORY_ROOT / "example_usage"

    for path in sorted((*example_root.glob("*.yaml"), *example_root.glob("*.yml"))):
        source = f"example_usage/{path.name}"
        expected_sources.add(source)
        assert catalog[source] == yaml.safe_load(path.read_text(encoding="utf-8"))

    for script_path in sorted(example_root.glob("*.py")):
        if script_path.name == "__init__.py":
            continue
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                variable_name = node.targets[0].id
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                variable_name = node.target.id
                value = node.value
            else:
                continue
            if not variable_name.endswith("_config") or value is None:
                continue
            try:
                config = ast.literal_eval(value)
            except (TypeError, ValueError):
                continue
            source = f"example_usage/{script_path.name}::{variable_name}"
            expected_sources.add(source)
            assert catalog[source] == _stringify_mapping_keys(config)

    assert set(catalog) == expected_sources


def test_server_url_uses_the_bound_port():
    server = SimpleNamespace(server_address=("127.0.0.1", 49152))
    assert _server_url(server, "127.0.0.1") == "http://localhost:49152"


def test_gui_product_name_is_metadata_driven_without_a_tagline():
    assert _gui_display_name({"display_name": "ALPHANSO"}) == "ALPHANSO"
    assert _gui_display_name({"display_name": "ALPHANSO GUI"}) == "ALPHANSO"


def test_bootstrap_uses_distribution_metadata_and_canonical_schema():
    payload = _bootstrap_payload()
    installed = metadata("alphanso")

    assert payload["package"]["name"] == installed["Name"]
    assert payload["package"]["version"] == installed["Version"]
    assert payload["package"]["summary"] == installed["Summary"]
    assert payload["configuration"]["calculation_types"] == CALCULATION_TYPES

    fields = {
        field["key"]: field
        for field in payload["configuration"]["common_fields"]
    }
    assert fields["num_alpha_groups"]["default"] == DEFAULT_NUM_ALPHA_GROUPS
    assert fields["min_alpha_energy"]["default"] == DEFAULT_MIN_ALPHA_ENERGY
    assert fields["max_alpha_energy"]["default"] == DEFAULT_MAX_ALPHA_ENERGY
    assert fields["neutron_energy_bins"]["default"] == list(
        DEFAULT_NEUTRON_ENERGY_BINS
    )
    assert fields["n_angular_bins"]["default"] == DEFAULT_N_ANGULAR_BINS
    assert {
        "min_alpha_energy",
        "max_alpha_energy",
        "num_alpha_groups",
        "neutron_energy_bins",
        "n_angular_bins",
    } <= {
        field["key"]
        for field in payload["configuration"]["common_fields"]
        if field.get("advanced")
    }
    for key in (
        "an_xs_data_dir",
        "stopping_power_data_dir",
        "decay_data_dir",
        "gamma_data_dir",
    ):
        assert fields[key]["browse"] == "directory"
        assert fields[key]["placeholder"].startswith("path/to/")
    assert payload["configuration"]["field_groups"]["Data sources"]["description"]


def test_workspace_defaults_are_built_from_the_bundled_examples():
    payload = _bootstrap_payload()
    defaults = payload["workspace_defaults"]
    examples = {
        example["config"]["calc_type"]: example["config"]
        for example in payload["examples"]
        if example["kind"] == "YAML"
    }

    assert defaults["name"] == examples["homogeneous"]["name"]
    assert defaults["beam_matdef"] == examples["beam"]["matdef"]
    assert defaults["homogeneous_matdef"] == examples["homogeneous"]["matdef"]
    assert defaults["source_matdef"] == examples["interface"]["source_matdef"]
    assert defaults["target_matdef"] == examples["interface"]["target_matdef"]
    assert defaults["intermediate_layers"] == examples["sandwich"]["intermediate_layers"]


def test_desktop_api_uses_native_folder_dialog(tmp_path):
    calls = []

    class FakeWindow:
        def create_file_dialog(self, dialog, *, directory):
            calls.append((dialog, directory))
            return [tmp_path / "custom-data"]

    api = _DesktopAPI("folder-dialog")
    api._bind_window(FakeWindow())

    assert api.choose_directory(str(tmp_path)) == str(tmp_path / "custom-data")
    assert calls == [("folder-dialog", str(tmp_path))]


def test_saved_configurations_persist_outside_the_browser_origin(tmp_path):
    path = tmp_path / "saved-configurations.json"
    configurations = [{
        "id": "config-1",
        "updated": "2026-08-04T12:00:00Z",
        "config": {"name": "Persistent model", "calc_type": "homogeneous"},
    }]

    assert write_saved_configuration_catalog(configurations, path) == configurations
    assert saved_configuration_catalog(path) == configurations
    assert json.loads(path.read_text(encoding="utf-8")) == configurations


def test_gui_sources_are_ascii_and_bundle_offline_latex():
    asset_root = REPOSITORY_ROOT / "alphanso" / "gui_assets"
    for filename in ("index.html", "app.js", "styles.css"):
        (asset_root / filename).read_text(encoding="ascii")

    index = (asset_root / "index.html").read_text(encoding="ascii")
    script = (asset_root / "app.js").read_text(encoding="ascii")
    assert "/vendor/katex/katex.min.css" in index
    assert "/vendor/katex/katex.min.js" in index
    assert r"\alpha\text{-n}" in script
    assert 'data-chart-action="zoom-in"' in script
    assert 'data-chart-action="reset"' in script
    assert 'data-spectrum-mode="normalized"' in script
    assert 'data-spectrum-mode="absolute"' in script
    assert "an_spectrum_absolute" in script
    assert 'valueKind === "duration"' in script
    assert r"\mathsf{" in script
    assert "ALPHANSO GUI" not in index
    assert "<title>ALPHANSO</title>" in index
    assert '`v${productVersion}`' in script
    assert "chevron" not in index
    assert "advanced-label" not in index
    assert index.count("data-accordion-action") == 2
