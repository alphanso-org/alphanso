# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- MT=91 continuum (alpha,n) channel support: cross sections and tabulated neutron energy distributions are parsed from GNDS XML files and folded into yield and spectrum calculations
- Kinematic box fallback for continuum channel when no tabulated neutron energy distribution is available
- New alpha-induced gamma source-term calculation (`alphanso/gamma_production.py`), a
  deterministic re-implementation of the Geant4 ParticleHP final-state sampling used by
  SaG4n on the JENDL/AN-2005 based JENDLTENDL01 library, with energy-conservation
  guards. Gamma production is evaluated on the same alpha slowing-down grid as the
  (alpha,n) calculation and covers discrete (alpha,n_i) residual de-excitation, the
  lumped MT=91 continuum, multi-particle channels ((alpha,2n), (alpha,n+p),
  (alpha,n+alpha), ...) and explicit TENDL photon-production products
- New `gamma/` nuclear-data subtree (data-v1.3.0): JENDL/AN-2005 exit-channel files in
  Geant4 ParticleHP format for 17 light targets, TENDL-2017 channel files for Z <= 30,
  and G4NDL 4.7 residual level schemes (`Gammas/z{Z}.a{A}`)
- Gamma outputs validated against SaG4n (Geant4 11.2.2, JENDLTENDL01) for Li-6, Li-7,
  Be-9, B-10, B-11, C-13, O-17 and O-18 at 1-10 MeV: gamma-per-neutron agreement within
  ~5% for all cases dominated by physical (energy-conserving) emission
- `gamma_yield`/`gamma_lines` are now asserted by the integration test suite, and a
  dedicated test module covers the level-scheme cascade math and channel parsing

### Changed

- `calculate_gammas` now defaults to `True` in `Transport.beam_problem` and
  `Transport.homogeneous_problem` (already the default in `Transport.calculate`)
- The RIPL-3/ENDF gamma-cascade pipeline in `transport.py` has been replaced by the
  new data-driven source term (known defects of the old path: JENDL level-scheme
  ceiling, zero yield for particle-unbound residuals, unphysical E0/K-forbidden
  branches)
- Nuclear data version bumped to 1.3.0 (adds the `gamma/` subtree; the release
  tarball checksum must be filled in `data_manager._EXPECTED_SHA256` on upload)

## [1.0.1] - 2026-03-27

### Changed

- Removed `numba` dependency; replaced `@njit`-decorated loops with vectorized NumPy operations
- Unified CLI and Python file output around `results.yaml`
- Removed the duplicate `output.yaml` artifact
- Removed binned gamma spectrum output (`gamma_spectrum`, `gamma_energy_bins`, `gamma_spectrum_layers`); gamma results now use discrete line pairs only
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
