#!/usr/bin/env python3
"""
Measure the transmit-frequency swing inside MSTID analysis windows, archive-wide.

This is the reporting-only measurement asked for in pyDARNmusic issue #4, which asks
whether checkDataQuality() should reject a window whose transmit frequency moves too
far. The unmerged check is preserved on tag archive/data_qc; it computes

    tfreq   = np.array(prm['tfreq'])[prm_tf]
    ptp_kHz = np.ptp(tfreq)
    if ptp_kHz > max_dFreq_kHz:   # archived default 500 kHz
        good_period = False

This script computes that same peak-to-peak value for every window, rejects nothing,
and reports the distribution by radar, UT hour, winter, and month. Enabling the check
is a method change that moves every value in the published MSTID index, so it is a
decision for the PI; this script exists to inform that decision, not to make it.

What it replicates, and where it stops
--------------------------------------
The record set per window matches what checkDataQuality() would see:

* files are selected as pyDARNmusic.load_fitacf() selects them, and a file that
  pydarnio reports as corrupt is discarded whole, matching on_corrupt='skip' (the
  setting the published index was computed under);
* records with an unsupported control program ID are dropped, as musicArray() drops
  them before prm is built;
* records are then restricted to [window start, window end), as the archived mask
  does once its prm_sTime / prm_eTime bug is fixed.

It does not build a musicArray, so it does not apply beam or gate limits. Those do
not touch prm['tfreq'], which is recorded per beam sounding before any limit is set.

Windows are the 2 h grid that darntids.mongo_tools.generate_mongo_list() tiles from
midnight. All twelve windows of each day are measured by default, because the driver
in the DARNtids working tree, run_DARNtids.py, sets slt_range=None. Two filters
narrow that:
--slt-range applies generate_mongo_list()'s solar-local-time gate, computed with that
function's own pyEphem apparent-solar-time formula, and --ut-hours takes a list of
window-start hours, for the 14,16,18,20 the climatology figures plot.

Usage
-----
    python scripts/scan_tx_frequency.py [--n N] [--root DIR] [--csv FILE]
    python scripts/scan_tx_frequency.py --ut-hours 14,16,18,20 --radars cvw

With no --n the whole selection is scanned, which is roughly 22,000 radar-days and
takes about an hour on 32 processes. --n samples that many radar-days with a fixed
seed, so a rerun over an unchanged archive reproduces the same day list. The CSV
carries one row per window, so the distribution can be re-sliced without rereading
the archive.

Result on the archive this package was built for
-----------------------------------------------
Run 2026-08-28 against /data/sd-data_fitexfilter, ten North American radars, the
2010 to 2021 winter months (11, 12, 1, 2, 3, 4), no SLT or UT gate: 21,750
radar-days, 235,663 files read, 446,926,123 beam soundings, 228,444 windows, of
which 225,072 carried a usable tfreq. Every sounding carried one; the count of
soundings with tfreq missing was zero.

Peak-to-peak within a window, all windows pooled: median 283 kHz, p90 2,794 kHz,
p99 11,048 kHz, max 22,346 kHz. 95.3% of windows exceed 100 kHz, because the radars
frequency-hop within a band, so a threshold anywhere near that measures hopping
rather than band changes. At the archived 500 kHz default, 17.9% of all windows are
rejected, and 19.0% of the UT 14/16/18/20 windows the climatology figures plot.

That loss has structure, which is the result that matters more than its size:

* by radar, over the paper's UT hours: sas 40.0%, pgr 39.1%, cvw 24.1%, cve 23.8%,
  bks 17.3%, gbr 15.8%, kap 11.3%, wal 7.5%, fhe 3.6%, fhw 3.5%;
* by UT hour within a radar: cvw and cve reject 91 to 92% of their 14 UTC windows
  against under 4% of every other hour, and fhe and bks spike the same way at 00
  and 12 UTC. Each radar changes band twice a day, at its own hour;
* by winter, over the paper's UT hours: 35.8% in 2013 and 37.6% in 2014 against
  6.4% in 2018 and 2.7% in 2021. Per radar the swing is wider still: pgr rejects
  83% of its 2014 windows and none of its 2017 windows.

1.6% of measured windows contain a non-positive tfreq (-9941, -1, or 0). The -1 is a
sentinel rather than a frequency, reaching 4% of records on some radar-days, and
those windows supply 8.9% of every rejection at the 500 kHz threshold. A check on
peak-to-peak alone counts a sentinel as a frequency change.
"""
import argparse
import bz2
import csv
import datetime
import glob
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict

import numpy as np

# The ten North American radars of the 2026 MSTID climatology paper.
RADARS = ['pgr', 'sas', 'kap', 'gbr', 'cvw', 'cve', 'fhw', 'fhe', 'bks', 'wal']
# Northern Hemisphere winter months, matching the paper's analysis season.
WINTER_MONTHS = [11, 12, 1, 2, 3, 4]
YEARS = list(range(2010, 2022))
SEED = 20260828

# The archived branch's default, and a ladder around it for context.
ARCHIVED_THRESHOLD_KHZ = 500
THRESHOLDS_KHZ = [100, 250, 500, 1000, 2000]

# Height used for the solar-local-time gate, matching generate_mongo_list()'s default.
SLT_HEIGHT_KM = 350.

# musicArray() refuses these control program IDs before prm is built.
# See https://superdarn.thayer.dartmouth.edu/wg-scd.html for the CPID database.
MAX_ABS_CPID = 20000
BAD_CPIDS = {8600}


def radar_coords():
    """Transmitter geographic coordinates, as darntids.general_lib.generate_radar_dict()
    reads them: straight from the pyDARN hardware.dat files."""
    from pydarn import SuperDARNRadars

    coords = {}
    for _, radar in SuperDARNRadars.radars.items():
        hdw = radar.hardware_info
        coords[hdw.abbrev] = (hdw.geographic.lat, hdw.geographic.lon)
    return coords


def solar_local_time(lat, lon, when):
    """Apparent solar time in hours, as darntids.mongo_tools.solartime() computes it."""
    import ephem

    obs = ephem.Observer()
    obs.lon = np.radians(lon)
    obs.lat = np.radians(lat)
    obs.elevation = SLT_HEIGHT_KM * 1000.
    obs.date = when

    sun = ephem.Sun()
    sun.compute(obs)
    hour_angle = obs.sidereal_time() - sun.ra
    slt = ephem.hours(hour_angle + ephem.hours('12:00')).norm
    return slt / (2. * np.pi) * 24


def day_windows(date, window_hours):
    """The window grid generate_mongo_list() tiles across one day."""
    step = datetime.timedelta(hours=window_hours)
    windows = []
    edge = date
    while edge < date + datetime.timedelta(days=1):
        windows.append((edge, edge + step))
        edge = edge + step
    return windows


def day_files(root, fit_sfx, radar, date):
    """(start time, path) for each FITACF file of one radar-day, sorted by start time."""
    pattern = os.path.join(root, date.strftime('%Y'), fit_sfx, radar,
                           '{!s}*{!s}*.{!s}.bz2'.format(date.strftime('%Y%m%d'), radar, fit_sfx))
    found = []
    for path in sorted(glob.glob(pattern)):
        try:
            started = datetime.datetime.strptime(os.path.basename(path)[:13], '%Y%m%d.%H%M')
        except ValueError:
            continue
        found.append((started, path))
    found.sort()
    return found


def files_for_windows(files, windows):
    """The subset of a day's files load_fitacf() would open for these windows.

    For one window that is the file covering the window start, which begins at or
    before it, plus every file beginning inside it.
    """
    starts = [started for started, _ in files]
    needed = set()
    for w_start, w_end in windows:
        covering = [i for i, started in enumerate(starts) if started <= w_start]
        if covering:
            needed.add(covering[-1])
        for i, started in enumerate(starts):
            if w_start < started < w_end:
                needed.add(i)
    return [files[i] for i in sorted(needed)]


def read_soundings(paths):
    """(time, tfreq) for every beam sounding prm would carry, plus read statistics."""
    import pydarnio
    from pydarn import time2datetime

    soundings = []
    stats = defaultdict(int)
    for path in paths:
        stats['files_opened'] += 1
        try:
            with bz2.open(path) as fp:
                raw = fp.read()
        except Exception:
            stats['files_unreadable'] += 1
            continue

        try:
            records, corruption_idx = pydarnio.read_fitacf(raw, mode='lax')
        except OSError:
            stats['files_empty'] += 1
            continue
        except Exception:
            stats['files_read_error'] += 1
            continue

        # load_fitacf(on_corrupt='skip') discards the whole file in both of these
        # cases, and 'skip' is the setting the published index was computed under.
        if len(records) == 0:
            stats['files_no_records'] += 1
            continue
        if corruption_idx is not None:
            stats['files_corrupt_skipped'] += 1
            continue

        for record in records:
            cpid = record.get('cp')
            if cpid is None or abs(cpid) >= MAX_ABS_CPID or int(abs(cpid)) in BAD_CPIDS:
                stats['soundings_bad_cpid'] += 1
                continue
            try:
                sounding_time = time2datetime(record)
            except Exception:
                stats['soundings_bad_time'] += 1
                continue
            soundings.append((sounding_time, record.get('tfreq')))
            stats['soundings'] += 1

    return soundings, stats


def scan_radar_day(task):
    """Measure every window of one radar-day. Runs in a worker process."""
    radar, date, cfg = task
    lat, lon = cfg['coords'][radar]

    windows = []
    for w_start, w_end in day_windows(date, cfg['window_hours']):
        slt = solar_local_time(lat, lon, w_start)
        windows.append((w_start, w_end, slt))

    if cfg['slt_range'] is not None:
        lo, hi = cfg['slt_range']
        windows = [w for w in windows if lo <= w[2] < hi]
    if cfg['ut_hours'] is not None:
        windows = [w for w in windows if w[0].hour in cfg['ut_hours']]

    stats = defaultdict(int)
    stats['radar_days'] = 1
    if not windows:
        return [], dict(stats)

    files = day_files(cfg['root'], cfg['fit_sfx'], radar, date)
    stats['files_present'] = len(files)
    if not files:
        stats['radar_days_no_files'] = 1
        return [], dict(stats)

    wanted = files_for_windows(files, [(w[0], w[1]) for w in windows])
    soundings, read_stats = read_soundings([path for _, path in wanted])
    for key, value in read_stats.items():
        stats[key] += value

    sounding_times = np.array([t for t, _ in soundings])
    rows = []
    for w_start, w_end, slt in windows:
        if len(sounding_times):
            in_window = np.logical_and(sounding_times >= w_start, sounding_times < w_end)
            tfreqs = [soundings[i][1] for i in np.flatnonzero(in_window)]
        else:
            tfreqs = []

        missing = sum(1 for f in tfreqs if f is None)
        good = np.array([f for f in tfreqs if f is not None], dtype=float)

        row = {
            'radar':        radar,
            'window_start': w_start.isoformat(),
            'ut_hour':      w_start.hour,
            'month':        w_start.month,
            'winter':       w_start.year if w_start.month >= 11 else w_start.year - 1,
            'slt':          round(float(slt), 3),
            'n_soundings':  len(tfreqs),
            'n_missing':    missing,
            'tfreq_min':    '' if not len(good) else int(good.min()),
            'tfreq_max':    '' if not len(good) else int(good.max()),
            'n_unique':     '' if not len(good) else int(len(np.unique(good))),
            'ptp_khz':      '' if not len(good) else float(np.ptp(good)),
        }
        rows.append(row)

        stats['windows'] += 1
        if not len(good):
            stats['windows_no_data'] += 1
        if missing:
            stats['windows_missing_tfreq'] += 1
        stats['soundings_missing_tfreq'] += missing

    return rows, dict(stats)


def build_tasks(args, coords):
    """One task per radar-day in the requested selection."""
    tasks = []
    for radar in args.radars:
        if radar not in coords:
            raise SystemExit('no pyDARN hardware entry for radar {!r}'.format(radar))
        for year in args.years:
            for month in args.months:
                day = datetime.datetime(year, month, 1)
                while day.month == month:
                    tasks.append((radar, day))
                    day = day + datetime.timedelta(days=1)
    tasks.sort(key=lambda t: (t[1], t[0]))
    return tasks


def percentile_line(label, values):
    if not len(values):
        return '  {:24s} {:>7s}'.format(label, 'n/a')
    pcts = np.percentile(values, [50, 75, 90, 95, 99])
    return '  {:24s} {:7d} {:8.0f} {:8.0f} {:8.0f} {:8.0f} {:8.0f} {:9.0f}'.format(
        label, len(values), pcts[0], pcts[1], pcts[2], pcts[3], pcts[4], values.max())


def report_group(title, key_name, groups):
    """Percentiles and archived-threshold exceedance for each group, largest first."""
    print('\n--- {!s} ---'.format(title))
    print('  {:24s} {:>7s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s}'.format(
        key_name, 'n', 'p50', 'p75', 'p90', 'p95', 'p99', 'max', '%>500'))
    for key in sorted(groups):
        values = np.array(groups[key], dtype=float)
        if not len(values):
            continue
        over = 100. * np.sum(values > ARCHIVED_THRESHOLD_KHZ) / len(values)
        print(percentile_line(str(key), values) + ' {:8.2f}%'.format(over))


def report_matrix(title, groups, row_keys, col_keys):
    """Exceedance rate at the archived threshold, by row and column."""
    print('\n--- {!s} ---'.format(title))
    print('  {:8s}'.format('') + ''.join('{:>12s}'.format(str(c)) for c in col_keys))
    for r in row_keys:
        cells = []
        for c in col_keys:
            values = np.array(groups.get((r, c), []), dtype=float)
            if not len(values):
                cells.append('{:>12s}'.format('.'))
            else:
                over = 100. * np.sum(values > ARCHIVED_THRESHOLD_KHZ) / len(values)
                cells.append('{:>12s}'.format('{:.1f}% ({:d})'.format(over, len(values))))
        print('  {:8s}'.format(str(r)) + ''.join(cells))


def report(rows, stats, args):
    print('\n' + '=' * 78)
    print('COVERAGE')
    print('=' * 78)
    for key in sorted(stats):
        print('  {:26s} {:>12,d}'.format(key, stats[key]))

    measured = [r for r in rows if r['ptp_khz'] != '']
    if not measured:
        print('\nno window carried a usable tfreq; nothing to report')
        return

    ptp = np.array([r['ptp_khz'] for r in measured], dtype=float)

    print('\n' + '=' * 78)
    print('PEAK-TO-PEAK TX FREQUENCY WITHIN A {:.0f} H WINDOW [kHz]'.format(args.window_hours))
    print('=' * 78)
    print('  windows measured: {:,}  (of {:,} in the selection)'.format(len(ptp), len(rows)))
    print('  {:24s} {:>7s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>9s}'.format(
        '', 'n', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'))
    print(percentile_line('all radars', ptp))

    print('\n--- exceedance by threshold ---')
    for threshold in THRESHOLDS_KHZ:
        count = int(np.sum(ptp > threshold))
        flag = '   <-- archived default' if threshold == ARCHIVED_THRESHOLD_KHZ else ''
        print('  > {:5d} kHz  {:8,d} windows  {:7.3f}%{!s}'.format(
            threshold, count, 100. * count / len(ptp), flag))

    by_radar = defaultdict(list)
    by_hour = defaultdict(list)
    by_winter = defaultdict(list)
    by_month = defaultdict(list)
    by_radar_hour = defaultdict(list)
    for row in measured:
        by_radar[row['radar']].append(row['ptp_khz'])
        by_hour[row['ut_hour']].append(row['ptp_khz'])
        by_winter[row['winter']].append(row['ptp_khz'])
        by_month[row['month']].append(row['ptp_khz'])
        by_radar_hour[(row['radar'], row['ut_hour'])].append(row['ptp_khz'])

    report_group('by radar', 'radar', by_radar)
    report_group('by UT hour of window start', 'UT hour', by_hour)
    report_group('by winter (year of the November)', 'winter', by_winter)
    report_group('by month', 'month', by_month)

    hours = sorted(by_hour)
    radars = [r for r in args.radars if r in by_radar]
    report_matrix('% of windows over {:d} kHz, by radar and UT hour (n in parentheses)'.format(
        ARCHIVED_THRESHOLD_KHZ), by_radar_hour, radars, hours)

    print('\n--- largest {:d} swings ---'.format(args.top))
    worst = sorted(measured, key=lambda r: -r['ptp_khz'])[:args.top]
    print('  {:6s} {:20s} {:>8s} {:>9s} {:>9s} {:>7s} {:>7s}'.format(
        'radar', 'window start (UT)', 'ptp kHz', 'tfreq min', 'tfreq max', 'unique', 'n'))
    for row in worst:
        print('  {:6s} {:20s} {:8.0f} {:9d} {:9d} {:7d} {:7d}'.format(
            row['radar'], row['window_start'], row['ptp_khz'],
            row['tfreq_min'], row['tfreq_max'], row['n_unique'], row['n_soundings']))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='/data/sd-data_fitexfilter',
                        help='root of the FITACF tree')
    parser.add_argument('--fit-sfx', default='fitacf',
                        help="FITACF flavor suffix, as load_fitacf() takes it")
    parser.add_argument('--radars', default=','.join(RADARS),
                        help='comma-separated radar codes')
    parser.add_argument('--years', default='{:d}-{:d}'.format(min(YEARS), max(YEARS)),
                        help='calendar years, as FIRST-LAST or a comma-separated list')
    parser.add_argument('--months', default=','.join(str(m) for m in WINTER_MONTHS),
                        help='comma-separated months')
    parser.add_argument('--window-hours', type=float, default=2.,
                        help='analysis window length')
    parser.add_argument('--slt-range', default='none',
                        help="solar-local-time gate as LO,HI, as generate_mongo_list() "
                             "applies it; 'none' (the default) measures every window")
    parser.add_argument('--ut-hours', default=None,
                        help='comma-separated window-start UT hours to keep, e.g. the '
                             '14,16,18,20 the climatology figures plot')
    parser.add_argument('--n', type=int, default=None,
                        help='sample this many radar-days, drawn with a fixed seed')
    parser.add_argument('--nprocs', type=int, default=max(1, mp.cpu_count() // 2),
                        help='worker processes')
    parser.add_argument('--csv', default=None,
                        help='write one row per window here')
    parser.add_argument('--top', type=int, default=25,
                        help='how many of the largest swings to list')
    args = parser.parse_args()

    args.radars = [r.strip() for r in args.radars.split(',') if r.strip()]
    args.months = [int(m) for m in args.months.split(',') if m.strip()]
    if '-' in args.years:
        first, last = args.years.split('-')
        args.years = list(range(int(first), int(last) + 1))
    else:
        args.years = [int(y) for y in args.years.split(',') if y.strip()]
    if args.ut_hours is not None:
        args.ut_hours = [int(h) for h in args.ut_hours.split(',') if h.strip()]
    if args.slt_range.lower() == 'none':
        args.slt_range = None
    else:
        lo, hi = args.slt_range.split(',')
        args.slt_range = (float(lo), float(hi))

    coords = radar_coords()
    tasks = build_tasks(args, coords)
    if args.n is not None:
        random.seed(SEED)
        random.shuffle(tasks)
        tasks = tasks[:args.n]
        tasks.sort(key=lambda t: (t[1], t[0]))

    cfg = {'root': args.root, 'fit_sfx': args.fit_sfx, 'coords': coords,
           'window_hours': args.window_hours, 'slt_range': args.slt_range,
           'ut_hours': args.ut_hours}
    payload = [(radar, date, cfg) for radar, date in tasks]

    print('radars:     {:d} ({!s})'.format(len(args.radars), ','.join(args.radars)))
    print('years:      {:d}-{:d}   months: {!s}'.format(
        min(args.years), max(args.years), ','.join(str(m) for m in args.months)))
    print('slt gate:   {!s}   ut hours: {!s}'.format(
        'none' if args.slt_range is None else args.slt_range,
        'all' if args.ut_hours is None else ','.join(str(h) for h in args.ut_hours)))
    print('radar-days: {:,}   workers: {:d}'.format(len(payload), args.nprocs), flush=True)

    rows = []
    totals = defaultdict(int)
    started = time.time()
    with mp.Pool(args.nprocs) as pool:
        for i, (day_rows, day_stats) in enumerate(
                pool.imap_unordered(scan_radar_day, payload, chunksize=4)):
            rows.extend(day_rows)
            for key, value in day_stats.items():
                totals[key] += value
            if (i + 1) % 500 == 0:
                elapsed = time.time() - started
                rate = (i + 1) / elapsed
                print('  {:,}/{:,} radar-days  {:.0f}s  eta {:.0f}s'.format(
                    i + 1, len(payload), elapsed, (len(payload) - i - 1) / rate), flush=True)

    print('\nelapsed {:.1f}s for {:,} radar-days'.format(time.time() - started, len(payload)))

    if args.csv:
        rows.sort(key=lambda r: (r['radar'], r['window_start']))
        with open(args.csv, 'w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ['radar'])
            writer.writeheader()
            writer.writerows(rows)
        print('wrote {:,} rows to {!s}'.format(len(rows), args.csv))

    report(rows, totals, args)


if __name__ == '__main__':
    main()
