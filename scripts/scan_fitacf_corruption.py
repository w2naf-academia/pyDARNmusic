#!/usr/bin/env python3
"""
Measure the FITACF corruption rate in an archive, and what load_fitacf(on_corrupt='keep')
would recover over the default 'skip'.

This is the script behind the numbers quoted in the load_fitacf() docstring Notes.
Rerun it against a different archive before quoting a rate for that archive.

Usage
-----
    python scripts/scan_fitacf_corruption.py [N] [--root DIR]

N is the sample size (default 2000). The sample is drawn with a fixed seed, so a
rerun over an unchanged archive reproduces the same file list.

Reported for each file: whether pydarnio reads it clean, reports partial corruption
('lax' mode returns a byte offset), parses no records at all, or raises. Empty files
raise OSError and are skipped by load_fitacf() before on_corrupt is consulted, so they
are counted separately from the corruption that on_corrupt actually governs.

Result on the archive this package was built for, 2026-08-28, /data/sd-data_fitexfilter,
pydarnio 2.0: population 238,997 winter files, sample 2,000, of which 1,995 clean,
5 empty, 0 partially corrupt.
"""
import argparse
import bz2
import glob
import os
import random
import time
from collections import Counter

import pydarnio

# The ten North American radars of the 2026 MSTID climatology paper.
RADARS = ['pgr', 'sas', 'kap', 'gbr', 'cvw', 'cve', 'fhw', 'fhe', 'bks', 'wal']
# Northern Hemisphere winter months, matching the paper's analysis season.
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
YEARS = range(2010, 2022)
SEED = 20260828


def find_files(root):
    """Every winter FITACF file for the radars and years above."""
    paths = []
    for year in YEARS:
        for radar in RADARS:
            pattern = os.path.join(root, str(year), 'fitacf', radar, '*.bz2')
            for path in glob.glob(pattern):
                try:
                    month = int(os.path.basename(path)[4:6])
                except ValueError:
                    continue
                if month in WINTER_MONTHS:
                    paths.append(path)
    return paths


def classify(path):
    """Read one file and report how pydarnio handled it."""
    row = {'path': path, 'status': None, 'records': 0, 'corruption_idx': None, 'bytes': 0}
    try:
        with bz2.open(path) as fp:
            raw = fp.read()
    except Exception:
        row['status'] = 'bz2_error'
        return row

    row['bytes'] = len(raw)
    try:
        records, corruption_idx = pydarnio.read_fitacf(raw, mode='lax')
    except OSError:
        # Empty or unreadable; load_fitacf() skips these regardless of on_corrupt.
        row['status'] = 'empty_or_unreadable'
        return row
    except Exception:
        row['status'] = 'read_error'
        return row

    row['records'] = len(records)
    row['corruption_idx'] = corruption_idx
    if len(records) == 0:
        row['status'] = 'no_records'
    elif corruption_idx is not None:
        row['status'] = 'corrupt_partial'
    else:
        row['status'] = 'clean'
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('n', nargs='?', type=int, default=2000, help='sample size')
    parser.add_argument('--root', default='/data/sd-data_fitexfilter',
                        help='root of the FITACF tree')
    args = parser.parse_args()

    paths = find_files(args.root)
    if not paths:
        raise SystemExit('no files found under {!r}'.format(args.root))

    random.seed(SEED)
    random.shuffle(paths)
    sample = paths[:args.n]

    print('population: {:,} winter files, {} radars, {}-{}'.format(
          len(paths), len(RADARS), min(YEARS), max(YEARS)), flush=True)
    print('sampling:   {:,}'.format(len(sample)), flush=True)

    rows = []
    started = time.time()
    for i, path in enumerate(sample):
        rows.append(classify(path))
        if (i + 1) % 200 == 0:
            print('  {}/{}  {:.1f}s'.format(i + 1, len(sample), time.time() - started), flush=True)

    counts = Counter(row['status'] for row in rows)
    print('\nelapsed {:.1f}s for {:,} files'.format(time.time() - started, len(rows)))
    print('--- status counts ---')
    for status, count in counts.most_common():
        print('  {:22s} {:6d}  {:6.2f}%'.format(status, count, 100 * count / len(rows)))

    clean = sum(row['records'] for row in rows if row['status'] == 'clean')
    partial = sum(row['records'] for row in rows if row['status'] == 'corrupt_partial')
    print("\nrecords under on_corrupt='skip': {:,}".format(clean))
    print("records under on_corrupt='keep': {:,}".format(clean + partial))
    if clean:
        print('recovered by keep: {:,} records ({:+.3f}%)'.format(partial, 100 * partial / clean))


if __name__ == '__main__':
    main()
