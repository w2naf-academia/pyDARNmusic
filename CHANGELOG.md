# Changelog

All notable changes to pyDARNmusic are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
major version is `0`, breaking changes increment the **minor** version.

## [0.2.0] - 2026-08-02

### Removed

- **BREAKING**: `boxcarFilter()` has been removed from `pyDARNmusic.utils.musicUtils` and
  from the top-level `pyDARNmusic` namespace. `from pyDARNmusic import boxcarFilter` now
  raises `ImportError`.

  The function was an unused, crude in-Python reimplementation of a median despeckle. It was
  not called anywhere in this package or in DARNtids, and it was not part of the processing
  chain that produced the published MSTID index. Despeckling of SuperDARN fitacf is performed
  upstream by the `fitexfilter` binary (RST `FilterRadarScan`), before the data ever reach
  pyDARNmusic — see the DARNtids pipeline documentation, Stage 0.

  **Migration**: no action is expected for any known consumer. If you were calling
  `boxcarFilter()`, despeckle upstream with `fitexfilter` instead; the two are not
  equivalent, and results from the removed function should not be treated as comparable to
  `fitexfilter` output.

## [0.1.0]

Initial packaged release on PyPI.

Python 3 reimplementation of the MUSIC algorithm for SuperDARN MSTID detection, migrated
from DaViTpy to PyDARN/pyDARNio by Francis Hassan Tholley (2023 MS thesis), with subsequent
HDF5 migration and updates.
