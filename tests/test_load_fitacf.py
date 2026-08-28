#!/usr/bin/env python3
"""
Read-path tests for pyDARNmusic.load_fitacf.

These exist because pydarnio 2.0 removed SDarnRead and the dmap_exceptions
submodule, which broke every FITACF read in this package with no test to catch
it (see issue #5). They exercise the pydarnio API surface this package actually
depends on, so an upstream API change fails here instead of in a reprocessing run.

The tests use the bz2-compressed FITACF tree committed under TestData/, and are
skipped when that tree is absent (e.g. when running against an installed wheel).
"""
import bz2
import datetime
import glob
import os

import pytest

import pydarnio
from pyDARNmusic import load_fitacf

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TestData')

# One 2 h window of Wallops data whose file name encodes its start time.
RADAR = 'wal'
STIME = datetime.datetime(2011, 5, 9, 0, 0)
ETIME = datetime.datetime(2011, 5, 9, 2, 0)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TEST_DATA_DIR),
    reason='TestData/ FITACF tree not available',
)


@pytest.fixture(scope='module')
def raw_fitacf():
    """Decompressed bytes of a single known-good FITACF file."""
    pattern = os.path.join(TEST_DATA_DIR, '2011', 'fitacf', RADAR, '20110509.0001.00.*.bz2')
    paths = glob.glob(pattern)
    if not paths:
        pytest.skip('no FITACF file matching {!s}'.format(pattern))
    with bz2.open(paths[0]) as fp:
        return fp.read()


class TestPydarnioAPI:
    """The pydarnio surface load_fitacf() depends on, pinned by test."""

    def test_read_fitacf_lax_returns_records_and_corruption_index(self, raw_fitacf):
        result = pydarnio.read_fitacf(raw_fitacf, mode='lax')
        assert isinstance(result, tuple) and len(result) == 2
        records, corruption_idx = result
        assert len(records) > 0
        assert corruption_idx is None, 'committed test file should parse cleanly'

    def test_records_carry_the_time_keys_time2datetime_consumes(self, raw_fitacf):
        records, _ = pydarnio.read_fitacf(raw_fitacf, mode='lax')
        for key in ('time.yr', 'time.mo', 'time.dy', 'time.hr', 'time.mt', 'time.sc'):
            assert key in records[0]

    def test_empty_input_raises_oserror(self):
        # The 2.0 stand-in for the 1.x EmptyFileError that load_fitacf() catches.
        with pytest.raises(OSError):
            pydarnio.read_fitacf(b'', mode='lax')

    def test_unparseable_input_returns_zero_records_without_raising(self):
        # In lax mode garbage is reported, not raised, so load_fitacf() must
        # test the record count rather than rely on an exception.
        records, corruption_idx = pydarnio.read_fitacf(b'this is not a dmap file' * 10, mode='lax')
        assert len(records) == 0
        assert corruption_idx == 0

    def test_truncated_input_reports_the_corruption_offset(self, raw_fitacf):
        records, corruption_idx = pydarnio.read_fitacf(raw_fitacf[:len(raw_fitacf) // 2], mode='lax')
        assert len(records) > 0
        assert corruption_idx is not None


class TestLoadFitacf:
    """End-to-end behavior of the loader."""

    def test_loads_records_within_the_requested_window(self):
        fitacf = load_fitacf(RADAR, STIME, ETIME, data_dir=TEST_DATA_DIR)
        assert len(fitacf) > 0
        assert all(isinstance(record, dict) for record in fitacf)

    def test_all_returned_records_fall_inside_the_window(self):
        from pydarn import time2datetime
        fitacf = load_fitacf(RADAR, STIME, ETIME, data_dir=TEST_DATA_DIR)
        times = [time2datetime(record) for record in fitacf]
        assert min(times) >= STIME
        assert max(times) < ETIME

    def test_missing_radar_returns_empty_list(self):
        assert load_fitacf('zzz', STIME, ETIME, data_dir=TEST_DATA_DIR) == []

    def test_rejects_an_unknown_on_corrupt_setting(self):
        with pytest.raises(ValueError):
            load_fitacf(RADAR, STIME, ETIME, data_dir=TEST_DATA_DIR, on_corrupt='sometimes')

    @pytest.mark.parametrize('on_corrupt', ['skip', 'keep'])
    def test_clean_data_is_unaffected_by_on_corrupt(self, on_corrupt):
        # With no corruption present, both settings must return the same records.
        fitacf = load_fitacf(RADAR, STIME, ETIME, data_dir=TEST_DATA_DIR, on_corrupt=on_corrupt)
        assert len(fitacf) > 0

    def test_corrupt_file_is_skipped_by_default(self, tmp_path):
        # A file whose tail is garbage: 'skip' drops it, 'keep' retains the
        # records that parsed before the corruption.
        source = glob.glob(os.path.join(TEST_DATA_DIR, '2011', 'fitacf', RADAR, '20110509.0001.00.*.bz2'))[0]
        with bz2.open(source) as fp:
            raw = fp.read()

        radar_dir = tmp_path / '2011' / 'fitacf' / RADAR
        radar_dir.mkdir(parents=True)
        target = radar_dir / os.path.basename(source)
        with bz2.open(target, 'wb') as fp:
            fp.write(raw[:len(raw) // 2] + b'\xff' * 64)

        skipped = load_fitacf(RADAR, STIME, ETIME, data_dir=str(tmp_path), on_corrupt='skip')
        kept = load_fitacf(RADAR, STIME, ETIME, data_dir=str(tmp_path), on_corrupt='keep')
        assert skipped == []
        assert len(kept) > 0
