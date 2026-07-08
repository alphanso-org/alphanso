# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-07-08

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
  Geant4 ParticleHP format, TENDL-2017 channel files, and G4NDL 4.7 residual level
  schemes (`Gammas/z{Z}.a{A}`). Gamma coverage matches the (alpha,n) cross-section
  coverage exactly (every target with an_xs data has gamma data), rather than an
  arbitrary Z cut
- Gamma outputs validated against SaG4n (Geant4 11.2.2, JENDLTENDL01): the discrete
  (alpha,n) light targets (Li-6..O-18, 1-10 MeV) agree to ~5% in gamma-per-neutron;
  the medium/heavy targets that go through the lumped inclusive (F02) channel (Mg, Na,
  Cl, Ca, ...) are order-of-magnitude estimates, since neither code resolves the
  individual physical sub-channels there and SaG4n itself produces convention/energy-
  non-conserving artifacts for them
- `gamma_yield`/`gamma_lines` are now asserted by the integration test suite, and a
  dedicated test module covers the level-scheme cascade math, channel parsing, the
  F02 inclusive-channel residual convention, and per-channel energy conservation

### Changed

- `calculate_gammas` now defaults to `True` in `Transport.beam_problem` and
  `Transport.homogeneous_problem` (already the default in `Transport.calculate`)
- The RIPL-3/ENDF gamma-cascade pipeline in `transport.py` has been replaced by the
  new data-driven source term (known defects of the old path: JENDL level-scheme
  ceiling, zero yield for particle-unbound residuals, unphysical E0/K-forbidden
  branches)
- Nuclear data version bumped to 1.3.0 (adds the `gamma/` subtree). The release
  tarball (`alphanso-data-v1.3.0.tar.gz`) must be uploaded to the `data-v1.3.0`
  GitHub release; its SHA-256 is pinned in `data_manager._EXPECTED_SHA256`

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
