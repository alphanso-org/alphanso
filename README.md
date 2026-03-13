# ALPHANSO: ALPHA Neutron SOurces

Open-source Python package for modeling ($\alpha$,n) neutron source terms.

![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)
![PyPI](https://img.shields.io/pypi/v/alphanso)

## Motivation

Legacy ($\alpha$,n) codes such as SOURCES-4C are written in FORTRAN 77 with nuclear data from the 1980s and have not been maintained since 2002. ALPHANSO provides a modern, open-source replacement built on up-to-date evaluated nuclear data libraries stored in the GNDS format. It covers all naturally occurring target nuclides and agrees well with experimental data.

Applications include reactor design, nuclear safeguards, radioactive waste management, nuclear astrophysics, and dark matter detection experiments.

## Key Features

- **Multiple Geometry Types**: Beam, homogeneous, interface, and sandwich configurations
- **Accurate Physics**: Up-to-date nuclear data with customizable data sources
- **Complete Output**: Neutron yields and energy spectra
- **Command-Line & Python API**: Use via CLI or integrate into Python workflows

## Installation

### From PyPI

```bash
pip install alphanso
```

On first use, ALPHANSO automatically downloads ~1.1 GB of nuclear data files and caches them locally (e.g., `~/.local/share/alphanso/` on Linux). To pre-download the data:

```bash
alphanso download-data
```

### From Source

```bash
git clone https://github.com/alphanso-org/alphanso.git
cd alphanso
pip install -e .
```

When installed from source, ALPHANSO uses the bundled `alphanso/data/` directory directly (no download needed).

### Custom Data Location

Set the `ALPHANSO_DATA_DIR` environment variable to override the data directory:

```bash
export ALPHANSO_DATA_DIR=/path/to/nuclear/data
```

Use `alphanso data-info` to check the current data paths and status.

## Quick Start

### Command-Line Interface

Run ALPHANSO with a YAML configuration file:

```bash
alphanso path/to/config.yaml
```

Or equivalently:

```bash
python3 -m alphanso path/to/config.yaml
```

Results are saved to `alphanso_output/<config_name>/`. You can specify a custom output directory as a second CLI argument: `alphanso config.yaml my_output/`.

#### Example YAML Configurations

**Beam** — monoenergetic alpha beam on a thick target:

```yaml
name: "Be-9 Beam Calculation"
calc_type: "beam"
matdef:
  Be-9: 1.0
beam_energy: 5.0
```

**Homogeneous** — uniform mixture of alpha emitters and targets:

```yaml
name: "Homogeneous Source"
calc_type: "homogeneous"
matdef:
  Pu-239: 0.3
  Pu-238: 0.2
  Be-9: 0.5
```

**Interface** — planar interface between source and target regions:

```yaml
name: "Interface Geometry"
calc_type: "interface"
source_matdef:
  Pu-238: 1.0
source_density: 19.8
target_matdef:
  Be-9: 1.0
```

**Sandwich** — multi-layer geometry with intermediate layers:

```yaml
name: "Multi-Layer Sandwich"
calc_type: "sandwich"
source_matdef:
  Pu-238: 1.0
source_density: 19.8
target_matdef:
  Be-9: 1.0
intermediate_layers:
  - matdef: { C-13: 1.0 }
    density: 2.26
    thickness: 1.0e-4
  - matdef: { Al-27: 1.0 }
    density: 2.70
    thickness: 1.0e-4
```

### Python API

```python
from alphanso.transport import Transport

config = {
    "calc_type": "beam",
    "matdef": {"Be-9": 1.0},
    "beam_energy": 5.0
}

results = Transport.calculate(config)
print(f"Neutron yield: {results['an_yield']:.3e} n/alpha")
```

`Transport.calculate()` accepts optional keys in the config dict:

- `name` (str): Label for identification in CLI output
- `output_dir` (str): Directory to save output files. When specified, saves `output.yaml` (full config + results) and `results.yaml` (results only).
- `save_data_files` (bool, default: `True`): Whether to save output files when `output_dir` is specified. Set to `False` to skip file saving.

## Running Tests

```bash
pytest
```

## Calculation Types

ALPHANSO supports four calculation geometries, each designed for specific physical scenarios.

### 1. Beam Calculations (`calc_type: "beam"`)

Monoenergetic or polyenergetic alpha beam incident on a thick target.

**Required Parameters**:

- `matdef` (dict): Target material composition (ZAID or element names with mass fractions)
- `beam_energy` (float): Alpha beam energy in MeV (monoenergetic)
- `beam_intensities` (list, optional): List of [energy, intensity] pairs for polyenergetic beams (use instead of `beam_energy`)

**Optional Parameters**:

- `num_alpha_groups` (int, default: `15000`): Number of alpha energy groups
- `min_alpha_energy` (float, default: `1e-11`): Minimum alpha energy in MeV
- `max_alpha_energy` (float, default: `15`): Maximum alpha energy in MeV
- `neutron_energy_bins` (list): Custom neutron energy bins in MeV (default: 0→15 MeV, 101 points). Accepts a 3-element shorthand `[start, stop, num_points]` which is expanded via `np.linspace`.
- `an_xs_data_dir` (str): Path to ($\alpha$,n) cross-section data directory
- `stopping_power_data_dir` (str): Path to stopping power data directory

**Results**:

- `an_yield`: Neutron production rate (n/$\alpha$)
- `an_spectrum`: Normalized neutron energy spectrum
- `an_spectrum_absolute`: Absolute neutron spectrum
- `neutron_energy_bins`: Energy bins for spectrum (MeV)

---

### 2. Homogeneous Calculations (`calc_type: "homogeneous"`)

Uniform mixture of alpha-emitting isotopes and target materials.

**Required Parameters**:

- `matdef` (dict): Material composition including both alpha emitters and targets

**Optional Parameters**:

- `num_alpha_groups` (int, default: `15000`): Number of alpha energy groups
- `min_alpha_energy` (float, default: `1e-11`): Minimum alpha energy in MeV
- `max_alpha_energy` (float, default: `15`): Maximum alpha energy in MeV
- `neutron_energy_bins` (list): Custom neutron energy bins in MeV (default: 0→15 MeV, 101 points). Accepts a 3-element shorthand `[start, stop, num_points]`.
- `an_xs_data_dir` (str): Path to cross-section data
- `stopping_power_data_dir` (str): Path to stopping power data
- `decay_data_dir` (str): Path to decay/branching data

**Results**:

- `an_yield`: Alpha-n neutron yield (n/s/g)
- `sf_yield`: Spontaneous fission neutron yield (n/s/g)
- `combined_yield`: Combined (alpha-n + SF) yield (n/s/g) - default output
- `an_spectrum`: Normalized alpha-n neutron spectrum
- `sf_spectrum`: Normalized spontaneous fission neutron spectrum
- `combined_spectrum`: Combined normalized neutron spectrum
- `neutron_energy_bins`: Energy bins (MeV)

---

### 3. Interface Calculations (`calc_type: "interface"`)

Planar interface between an alpha-emitting source region and a target region.

**Required Parameters**:

- `source_matdef` (dict): Alpha source material composition
- `source_density` (float): Source density in g/cm^3
- `target_matdef` (dict): Target material composition

**Optional Parameters**:

- `num_alpha_groups` (int, default: `15000`): Number of alpha energy groups
- `min_alpha_energy` (float, default: `1e-11`): Minimum alpha energy in MeV
- `max_alpha_energy` (float, default: `15`): Maximum alpha energy in MeV
- `neutron_energy_bins` (list): Custom neutron energy bins in MeV (default: 0→15 MeV, 101 points). Accepts a 3-element shorthand `[start, stop, num_points]`.
- `an_xs_data_dir` (str): Cross-section data path
- `stopping_power_data_dir` (str): Stopping power data path
- `decay_data_dir` (str): Decay data path

**Results**:

- `an_yield`: Neutron yield per cm^2 of interface (n/s/cm^2)
- `an_spectrum`: Normalized spectrum
- `an_spectrum_absolute`: Absolute spectrum (n/s/cm^2/MeV)
- `neutron_energy_bins`: Energy bins (MeV)

---

### 4. Sandwich Calculations (`calc_type: "sandwich"`)

Multi-layer sandwich geometry with volumetric formulation. Alpha source (Region A) -> Intermediate layers (Region B1, B2, ..., Bn) -> Target (Region C).

**Required Parameters**:

- `source_matdef` (dict): Alpha source material composition (Region A)
- `source_density` (float): Source density in g/cm^3
- `target_matdef` (dict): Target material composition (Region C)
- `intermediate_layers` (list of dicts): One or more intermediate layers, each containing:
  - `matdef` (dict): Layer material composition
  - `density` (float): Layer density in g/cm^3
  - `thickness` (float): Layer thickness in cm

**Optional Parameters**:

- `n_angular_bins` (int, default: `40`): Angular bins for integration
- `num_alpha_groups` (int, default: `15000`): Number of alpha energy groups
- `min_alpha_energy` (float, default: `1e-11`): Minimum alpha energy in MeV
- `max_alpha_energy` (float, default: `15`): Maximum alpha energy in MeV
- `neutron_energy_bins` (list): Custom neutron energy bins in MeV (default: 0→15 MeV, 101 points). Accepts a 3-element shorthand `[start, stop, num_points]`.
- `an_xs_data_dir` (str): Cross-section data path
- `stopping_power_data_dir` (str): Stopping power data path
- `decay_data_dir` (str): Decay data path

**Results**:

- `an_yield`: Total neutron yield (n/s/cm^2)
- `yield_target`: Yield in target region (n/s/cm^2)
- `yield_layers`: Per-layer yield breakdown (list of n/s/cm^2)
- `yield_ab_b`: Yield at AB interface in first layer material
- `yield_bc_b`: Yield at BC interface in last layer material
- `yield_bc_c`: Yield at BC interface in target material
- `an_spectrum`: Normalized total spectrum
- `an_spectrum_absolute`: Absolute total spectrum
- `spectrum_layers`: Per-layer spectrum contributions
- `neutron_energy_bins`: Energy bins (MeV)

## Material Definition Format

Materials are defined using dictionaries with isotope/element identifiers as keys and mass fractions as values.

**Supported Formats**:

- Element-mass names: `"Be-9"`, `"C-13"`, `"Al-27"`, `"Pu-238"`
- Natural element symbols: `"C"`, `"O"` (auto-expands to natural isotopes using abundance data)
- ZAID integers: `4009` (Be-9), `6013` (C-13), `94238` (Pu-238)
- Natural element ZAIDs: `6000` (natural C), `8000` (natural O) — AAA=000 convention

> **Note**: ZAID keys in Python dicts must be integers, not strings. Using string keys like `"92235"` will cause entries to be silently dropped.

**Example**:

```yaml
matdef:
  Pu-238: 0.90
  Pu-239: 0.05
  Pu-240: 0.05
```

Or in Python:

```python
matdef = {94238: 0.90, 94239: 0.05, 94240: 0.05}
```

Mass fractions should sum to 1.0 for single-phase materials, but can exceed 1.0 for compounds.

## Output Files

When using the CLI, results are saved to `alphanso_output/<config_name>/`:

- **`output.yaml`**: Complete input configuration and full results (for reproducibility)
- **`results.yaml`**: Clean results-only output (for easy parsing)

## Custom Data Sources

ALPHANSO allows you to use custom nuclear data:

```yaml
an_xs_data_dir: "/path/to/cross_sections"
stopping_power_data_dir: "/path/to/stopping_data"
decay_data_dir: "/path/to/decay_data"
```

Paths can be absolute or relative to the project root. If not specified, ALPHANSO uses built-in default data.

To integrate new data formats, extend the parsers in `alphanso/parsers.py`.

## Citation

If you use ALPHANSO in your research, please cite:

```bibtex
@article{rawal2026alphanso,
  title     = {{ALPHANSO}: Open-Source Modeling of ($\alpha$,n) Neutron Source Terms},
  author    = {Rawal, Divit and Nelson, Anthony J. and Zywiec, William and Siefman, Daniel},
  journal   = {Nuclear Instruments and Methods in Physics Research Section A},
  year      = {2026},
  note      = {Preprint}
}
```

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes and push to your fork
4. Open a pull request against `main`

Please open an issue first for bug reports or feature requests.

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.

## Authors

- Divit Rawal - [divit.rawal@berkeley.edu](mailto:divit.rawal@berkeley.edu)
- Anthony J. Nelson - [nelson254@llnl.gov](mailto:nelson254@llnl.gov)
- William Zywiec - [zywiec1@llnl.gov](mailto:zywiec1@llnl.gov)
- Daniel Siefman - [daniel.siefman@berkeley.edu](mailto:daniel.siefman@berkeley.edu)

## Support

For questions, issues, or feature requests, please [open an issue](https://github.com/alphanso-org/alphanso/issues) on GitHub or contact the authors.
