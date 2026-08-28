# Changelog

All notable changes to pyDARNmusic are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
major version is `0`, breaking changes increment the **minor** version.

## [Unreleased]

### Fixed

- **`load_fitacf()` now works with pydarnio 2.0.** pydarnio 2.0 removed the class-based
  `SDarnRead` reader and the `pydarnio.exceptions.dmap_exceptions` submodule, so every
  FITACF read raised `AttributeError` on a fresh install. The loader now calls
  `pydarnio.read_fitacf(stream, mode='lax')` and catches `OSError` where it previously
  caught `EmptyFileError`. Record format is unchanged, so callers see the same list of
  `dict` records with the same `time.*` keys. (#5)

### Added

- **`load_fitacf(..., on_corrupt=...)`**, either `'skip'` (default) or `'keep'`. In `lax`
  mode pydarnio 2.0 reports corruption by returning a byte offset instead of raising, which
  silently changed a corrupt file from "dropped" under 1.x to "partially loaded". `'skip'`
  restores the 1.x behavior and is the setting that reproduces previously published results;
  `'keep'` retains the records that parsed before the corruption. Both log the byte offset,
  which pydarnio 1.x did not report.
- **`tests/`**, covering the read path and the pydarnio API surface this package depends on.
  There was previously no test directory, and therefore nothing to catch the pydarnio 2.0
  break at install time. The tests read the committed `TestData/` FITACF tree and skip when
  it is unavailable.

### Changed

- **`pydarnio` is pinned to `>=2.0`** in `pyproject.toml` and `requirements.txt`. Both
  previously declared a bare `pydarnio`, which is the root cause of the break: pip resolved
  to whatever was current at install time.

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
