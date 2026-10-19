from pathlib import Path

import numpy as np
import pytest
import yaml

from alphanso import __main__ as cli
from alphanso.output_files import write_results_yaml
from alphanso.transport import Transport

REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_portable_yaml(path: Path, expected: dict):
    text = path.read_text()
    assert "!!python" not in text
    assert "numpy" not in text
    assert "!!binary" not in text
    assert yaml.safe_load(text) == expected


def test_transport_calculate_writes_only_results_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr("alphanso.transport.ensure_data", lambda: None)
    monkeypatch.setattr(
        Transport,
        "_calculate_beam",
        staticmethod(lambda config: {"an_yield": 1.23, "neutron_energy_bins": [0.0, 1.0]}),
    )

    output_dir = tmp_path / "python-output"
    results = Transport.calculate({
        "calc_type": "beam",
        "matdef": {"Be-9": 1.0},
        "beam_energy": 5.0,
        "output_dir": str(output_dir),
    })

    assert not (output_dir / "output.yaml").exists()
    text = (output_dir / "results.yaml").read_text()
    assert text.startswith("# ALPHANSO results (calc_type: beam)\n")
    assert yaml.safe_load(text) == results


def test_transport_calculate_normalizes_results_and_yaml_output(tmp_path, monkeypatch):
    monkeypatch.setattr("alphanso.transport.ensure_data", lambda: None)
    monkeypatch.setattr(
        Transport,
        "_calculate_beam",
        staticmethod(
            lambda config: {
                "an_yield": np.float64(1.23),
                "gamma_lines": [(3.2, np.float64(1.5))],
                "metadata": {
                    "pair": (np.float64(2.5), np.int64(3)),
                    "values": np.array([np.float64(4.5), np.float64(5.5)]),
                },
            }
        ),
    )

    output_dir = tmp_path / "python-output"
    results = Transport.calculate({
        "calc_type": "beam",
        "matdef": {"Be-9": 1.0},
        "beam_energy": 5.0,
        "output_dir": str(output_dir),
    })

    expected = {
        "an_yield": 1.23,
        "gamma_lines": [[3.2, 1.5]],
        "metadata": {
            "pair": [2.5, 3],
            "values": [4.5, 5.5],
        },
    }

    assert results == expected
    _assert_portable_yaml(output_dir / "results.yaml", expected)


def test_cli_read_out_writes_only_results_yaml(tmp_path):
    output_dir = cli.read_out(
        [
            {
                "source": "demo",
                "_result": {"an_yield": 4.56, "an_spectrum": [0.25, 0.75]},
            }
        ],
        tmp_path,
    )

    source_dir = Path(output_dir) / "demo"
    assert not (source_dir / "output.yaml").exists()
    assert yaml.safe_load((source_dir / "results.yaml").read_text()) == {
        "an_yield": 4.56,
        "an_spectrum": [0.25, 0.75],
    }


def test_cli_read_out_normalizes_non_native_results_payload(tmp_path):
    output_dir = cli.read_out(
        [
            {
                "source": "demo",
                "_result": {
                    "gamma_lines": [(3.2, np.float64(1.5))],
                    "nested": {"pair": (np.int64(2), np.float64(3.5))},
                },
            }
        ],
        tmp_path,
    )

    expected = {
        "gamma_lines": [[3.2, 1.5]],
        "nested": {"pair": [2, 3.5]},
    }

    source_dir = Path(output_dir) / "demo"
    _assert_portable_yaml(source_dir / "results.yaml", expected)


def test_cli_read_out_writes_numbered_results_for_grouped_configs(tmp_path):
    output_dir = cli.read_out(
        [
            {"source": "demo", "_result": {"an_yield": 1.0}},
            {"source": "demo", "_result": {"an_yield": 2.0}},
        ],
        tmp_path,
    )

    source_dir = Path(output_dir) / "demo"
    assert not (source_dir / "output_1.yaml").exists()
    assert not (source_dir / "output_2.yaml").exists()
    assert yaml.safe_load((source_dir / "results_1.yaml").read_text()) == {"an_yield": 1.0}
    assert yaml.safe_load((source_dir / "results_2.yaml").read_text()) == {"an_yield": 2.0}


def test_results_yaml_header_documents_units(tmp_path):
    results = {
        "an_yield": 1.0,
        "an_spectrum_absolute": [0.25, 0.75],
        "gamma_lines": [[0.511, 2.0]],
        "undocumented": 3,
    }

    text = write_results_yaml(results, tmp_path, calc_type="interface").read_text()
    header = [line for line in text.splitlines() if line.startswith("#")]

    assert header[0] == "# ALPHANSO results (calc_type: interface)"
    assert "#   an_yield: n/s/cm^2" in header
    assert "#   an_spectrum_absolute: n/s/cm^2 per energy bin" in header
    assert "#   gamma_lines: [energy (MeV), intensity (gammas/s/cm^2)] pairs" in header
    assert not any("undocumented" in line for line in header)
    assert yaml.safe_load(text) == results


def test_results_yaml_has_no_header_without_calc_type(tmp_path):
    text = write_results_yaml({"an_yield": 1.0}, tmp_path).read_text()

    assert not text.startswith("#")
    assert yaml.safe_load(text) == {"an_yield": 1.0}


def test_cli_read_out_writes_units_header(tmp_path):
    output_dir = cli.read_out(
        [
            {
                "source": "demo",
                "calc_type": "homogeneous",
                "_result": {
                    "an_yield": 4.56,
                    "sf_contributors": [{"zaid": 94240, "sf_yield": 1.0}],
                },
            }
        ],
        tmp_path,
    )

    lines = (Path(output_dir) / "demo" / "results.yaml").read_text().splitlines()
    assert lines[0] == "# ALPHANSO results (calc_type: homogeneous)"
    assert "#   an_yield: n/s/g" in lines
    assert "#     sf_yield: n/s/g" in lines


@pytest.mark.integration
@pytest.mark.parametrize("calc_type", ["beam", "homogeneous", "interface", "sandwich"])
def test_example_results_yaml_documents_every_unit(tmp_path, calc_type):
    config_path = REPO_ROOT / "example_usage" / f"example_config_{calc_type}.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["output_dir"] = str(tmp_path)

    results = Transport.calculate(config)

    text = (tmp_path / "results.yaml").read_text()
    header = [line for line in text.splitlines() if line.startswith("#")]
    for key in results:
        assert any(line.startswith(f"#   {key}: ") for line in header), key
    for field in results.get("sf_contributors", [{}])[0]:
        assert any(line.startswith(f"#     {field}: ") for line in header), field

    # Absolute spectra are per energy bin, so they sum to the total yield.
    if "an_spectrum_absolute" in results:
        assert sum(results["an_spectrum_absolute"]) == pytest.approx(results["an_yield"])

    assert yaml.safe_load(text) == results


def test_real_example_beam_results_yaml_is_portable(tmp_path):
    config_path = REPO_ROOT / "example_usage" / "example_config_beam.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["output_dir"] = str(tmp_path / "beam-example")

    results = Transport.calculate(config)

    assert "gamma_lines" in results
    assert all(isinstance(line, list) and len(line) == 2 for line in results["gamma_lines"])
    _assert_portable_yaml(Path(config["output_dir"]) / "results.yaml", results)
