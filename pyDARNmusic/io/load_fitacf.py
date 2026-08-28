#!/usr/bin/env python3
import os
import bz2
import glob
import datetime
import tqdm

import logging

import numpy as np

import pydarnio
from pydarn import time2datetime

def load_fitacf(radar,sTime,eTime=None,data_dir='/sd-data',fit_sfx='fitacf',on_corrupt='skip'):
    """
    Load FITACF data from multiple FITACF files by specifying a date range.

    This routine assumes bz2 compression.

    Parameters
    ----------
    radar : str
        Three-letter radar code, e.g. 'bks'.
    sTime : datetime.datetime
        Start of the requested window.
    eTime : datetime.datetime, optional
        End of the requested window. Defaults to sTime + 1 day.
    data_dir : str
        Root of the FITACF tree, laid out as <data_dir>/<year>/<fit_sfx>/<radar>/.
    fit_sfx : str
        FITACF flavor suffix, e.g. 'fitacf' or 'fitexfilter.fitacf'.
    on_corrupt : {'skip', 'keep'}
        What to do with a file that pydarnio reports as corrupt or truncated.
        'skip' (default) discards the whole file. 'keep' retains the records
        that parsed cleanly before the corruption. Either way the corruption is
        logged with its byte offset. See Notes for why 'skip' is the default and
        why 'keep' recovers less than it appears to.

    Returns
    -------
    list of dict
        FITACF records falling within [sTime, eTime).

    Notes
    -----
    Why 'skip' is the default, and what to expect from 'keep'.

    Read this before spending time on 'keep'. The records it returns are
    complete and valid, because pydarnio's 'lax' mode stops at the first byte it
    cannot parse and returns only records that parsed in full. The reason to
    leave the default alone is what happens downstream, not the quality of those
    records.

    1. 'skip' reproduces published results. The SuperDARN MSTID index published
       from this pipeline was computed under pydarnio 1.x, where a corrupt
       record raised and the existing handler dropped the whole file. 'skip'
       preserves that behavior, so a reprocessing run stays comparable to the
       archived index. 'keep' is a deliberate departure from it.

    2. Most of what 'keep' recovers is discarded anyway. The MSTID pipeline
       calls pyDARNmusic.utils.checkDataQuality(), which marks a window bad when
       any gap in it exceeds max_off_time, 10 minutes by default. SuperDARN
       FITACF files and MSTID analysis windows are both two hours, so one file
       is roughly one window. A file truncated halfway leaves an hour of gap, and
       its window is rejected under either setting; the extra records land in a
       window that is thrown out regardless. 'keep' changes the outcome only
       where the truncation falls within max_off_time of the end of the file's
       coverage, which is a narrow band.

    3. The rate is low enough that the choice is close to moot on the archive
       this package was built for. scripts/scan_fitacf_corruption.py sampled
       2,000 FITEX-filtered files across ten North American radars over the
       2010 to 2021 Northern Hemisphere winters, out of a population of 238,997:
       1,995 read clean, 5 were empty, and none were partially corrupt. Empty
       files raise OSError and are skipped before on_corrupt is consulted, so
       that setting had no effect on any file in the sample.

    Set 'keep' where retaining every valid record matters more than matching the
    published run, and expect the gain to be small. One case escapes both
    settings: a file truncated exactly on a record boundary parses clean and is
    reported as clean, which is a property of the DMAP format.
    """
    if on_corrupt not in ('skip','keep'):
        raise ValueError("on_corrupt must be 'skip' or 'keep', got {!r}".format(on_corrupt))

    if eTime is None:
        eTime = sTime + datetime.timedelta(days=1)

    sDate   = datetime.datetime(sTime.year,sTime.month,sTime.day)
    eDate   = datetime.datetime(eTime.year,eTime.month,eTime.day)

    # Create a list of days we need fitacf files from.
    dates   = [sDate]
    while dates[-1] < eDate:
        next_date   = dates[-1] + datetime.timedelta(days=1)
        dates.append(next_date)

    # Find the data files that fall in that date range.
    fitacf_paths_0    = []
    for date in dates:
        date_str        = date.strftime('%Y%m%d')
        year_str        = date.strftime('%Y')
        fpattern        = os.path.join(data_dir,year_str,fit_sfx,radar,'{!s}*{!s}*.{!s}.bz2'.format(date_str,radar,fit_sfx))
        fitacf_paths_0   += glob.glob(fpattern)


    # Sort the files by name.
    fitacf_paths_0.sort()

    # Get rid of any files we don't need.
    fitacf_paths = []
    time_deltas  = []
    for finx, fpath in enumerate(fitacf_paths_0):
        date_str    = os.path.basename(fpath)[:13]
        this_time   = datetime.datetime.strptime(date_str,'%Y%m%d.%H%M')

        # Eliminate files after the time we need
        if this_time <= eTime:
            fitacf_paths.append(fpath)
            time_deltas.append((sTime-this_time).total_seconds())

    # Eliminate files before the time that we need
    tds     = np.array(time_deltas)
    tf      = tds>=0
    if np.any(tf):
        min_inx = np.argmin(tds[tf])
        fitacf_paths = fitacf_paths[min_inx:]

    # Return and empty list if there are no files to load.
    if len(fitacf_paths) == 0:
        return []

    # Load and append each data file.
    print()
    fitacf = []
    for fitacf_path in tqdm.tqdm(fitacf_paths,desc='Loading {!s} Files'.format(fit_sfx),dynamic_ncols=True):
        tqdm.tqdm.write(fitacf_path)

        try:
            with bz2.open(fitacf_path) as fp:
                fitacf_stream = fp.read()
        except:
            msg = '{!s} BZ2 Decompression Error: {!s}'.format(datetime.datetime.now(),fitacf_path)
            logging.warning(msg)
            continue

        # pydarnio 2.0 replaced the class-based SDarnRead with module-level
        # readers. In 'lax' mode it reports corruption by returning the byte
        # index where parsing stopped, where 1.x raised; the guards below turn
        # that back into the skip-the-file behavior 1.x provided, and log the
        # offset, which 1.x did not report.
        try:
            records, corruption_idx = pydarnio.read_fitacf(fitacf_stream, mode='lax')
        except OSError:
            # 'failed to fill whole buffer'; the 2.0 stand-in for 1.x EmptyFileError.
            msg = '{!s} Empty or unreadable FITACF file: {!s}'.format(datetime.datetime.now(),fitacf_path)
            logging.warning(msg)
            continue # Skip fitacf file if empty.
        except Exception:
            msg = '{!s} pydarnio.read_fitacf() Error: {!s}'.format(datetime.datetime.now(),fitacf_path)
            logging.warning(msg)
            continue

        if len(records) == 0:
            # Nothing parsed at all, e.g. a file that is not DMAP. 1.x raised here.
            msg = '{!s} No FITACF records parsed (corruption at byte {!s}): {!s}'.format(datetime.datetime.now(),corruption_idx,fitacf_path)
            logging.warning(msg)
            continue

        if corruption_idx is not None:
            msg = '{!s} FITACF corruption at byte {!s} after {!s} records ({!s}): {!s}'.format(
                    datetime.datetime.now(),corruption_idx,len(records),on_corrupt,fitacf_path)
            logging.warning(msg)
            if on_corrupt == 'skip':
                continue

        fitacf += records

    # Remove uneeded fitacf records.
    fitacf_new = []
    for record in fitacf:
        try:
            this_time = time2datetime(record)
        except:
            print('FITACF record time conversion error - Skipping record.')
            continue
        if this_time >= sTime and this_time < eTime:
            fitacf_new.append(record)
    return fitacf_new

if __name__ == '__main__':
    output_dir  = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sTime       = datetime.datetime(2010,11,1)
    eTime       = datetime.datetime(2010,11,2)
    radar       = 'bks'
    fitacf      = load_fitacf(sTime,eTime,radar)
