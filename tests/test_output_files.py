from pathlib import Path

import numpy as np
import yaml

from alphanso import __main__ as cli
from alphanso.transport import Transport


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
    assert yaml.safe_load((output_dir / "results.yaml").read_text()) == results


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


def test_real_example_beam_results_yaml_is_portable(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "example_usage" / "example_config_beam.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["output_dir"] = str(tmp_path / "beam-example")

    results = Transport.calculate(config)

    assert "gamma_lines" in results
    assert all(isinstance(line, list) and len(line) == 2 for line in results["gamma_lines"])
    _assert_portable_yaml(Path(config["output_dir"]) / "results.yaml", results)
