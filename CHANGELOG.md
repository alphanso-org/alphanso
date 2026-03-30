# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.1] - 2026-03-27

### Changed

- Removed `numba` dependency; replaced `@njit`-decorated loops with vectorized NumPy operations
- Unified CLI and Python file output around `results.yaml`
- Removed the duplicate `output.yaml` artifact
- Updated citation with arXiv preprint link ([arXiv:2603.17719](https://arxiv.org/abs/2603.17719))

## [1.0.0] - 2026-03-11

### Added

- Initial open-source release of ALPHANSO
- Four geometry types: beam, homogeneous, interface, and sandwich configurations
- Command-line interface (`alphanso` CLI) with YAML configuration files
- Python API via `Transport.calculate()`
- Support for multiple nuclear data libraries (cross-sections, stopping powers, decay data) in GNDS format
- Bundled default nuclear data for all naturally occurring target nuclides
- Polyenergetic beam support
- Custom neutron energy binning
- YAML output files (`output.yaml`, `results.yaml`) for reproducibility and downstream parsing
- Example configurations in `example_usage/`
