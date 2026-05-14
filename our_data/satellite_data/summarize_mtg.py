"""
summarize_mtg.py — Summarise downloaded MTG FCI chunks by date.

Scans the _raw_chunks directory, parses FCI filenames, and produces
a CSV showing how many repeat cycles and chunk files are available
per date.

Usage:
    python summarize_mtg.py
    python summarize_mtg.py --raw_dir path/to/_raw_chunks
    python summarize_mtg.py --output summary.csv
"""

import os
import re
import sys
import csv
import argparse
import datetime
from collections import defaultdict
from pathlib import Path


# Default path: anchor to the project root (two levels up from this script,
# which lives at our_data/satellite_data/) so the same default works whether
# you run from the project root or from the script's own directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = str(PROJECT_ROOT / 'our_data' / 'satellite_data'
                      / 'MTG' / '_raw_chunks')


def parse_fci_filename(filename):
    """Extract metadata from an FCI L1C filename."""
    info = {
        'chunk_number': None,
        'repeat_cycle_in_day': None,
        'sensing_start': None,
        'is_body': 'CHK-BODY' in filename or 'CHK_BODY' in filename,
        'is_trailer': 'TRAIL' in filename,
    }

    match = re.search(r'_(\d{4})_(\d{4})\.nc$', filename)
    if match:
        info['repeat_cycle_in_day'] = int(match.group(1))
        info['chunk_number'] = int(match.group(2))

    match = re.search(r'_(?:OPE|DEV)_(\d{14})_(\d{14})_', filename)
    if match:
        try:
            info['sensing_start'] = datetime.datetime.strptime(
                match.group(1), '%Y%m%d%H%M%S'
            )
        except ValueError:
            pass

    return info


def nominal_time(date_ref, repeat_cycle_in_day, period_min=10):
    """Compute nominal start time from repeat cycle number."""
    base = date_ref.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + datetime.timedelta(
        minutes=(repeat_cycle_in_day - 1) * period_min
    )


def summarize(raw_dir):
    """
    Scan _raw_chunks and build a per-date summary.

    Returns:
        list of dicts, one per date, sorted chronologically.
    """
    if not os.path.isdir(raw_dir):
        print(f"ERROR: directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    nc_files = [f for f in os.listdir(raw_dir) if f.endswith('.nc')]
    print(f"Found {len(nc_files)} .nc files in {raw_dir}")

    # Group by date
    # Key: date string (YYYY-MM-DD)
    # Value: dict with sets of repeat cycles, chunk counts, etc.
    dates = defaultdict(lambda: {
        'body_files': 0,
        'trailer_files': 0,
        'repeat_cycles': set(),       # set of rc numbers
        'complete_cycles': 0,         # cycles with exactly 2 body chunks
        'incomplete_cycles': 0,       # cycles with only 1 body chunk
        'chunks_seen': defaultdict(set),  # rc_num -> set of chunk_numbers
    })

    for filename in nc_files:
        info = parse_fci_filename(filename)

        if info['sensing_start'] is None:
            continue

        date_str = info['sensing_start'].strftime('%Y-%m-%d')

        if info['is_trailer']:
            dates[date_str]['trailer_files'] += 1
            continue

        if info['is_body']:
            dates[date_str]['body_files'] += 1
            if info['repeat_cycle_in_day'] is not None:
                rc = info['repeat_cycle_in_day']
                dates[date_str]['repeat_cycles'].add(rc)
                if info['chunk_number'] is not None:
                    dates[date_str]['chunks_seen'][rc].add(
                        info['chunk_number']
                    )

    # Build summary rows
    rows = []
    for date_str in sorted(dates.keys()):
        d = dates[date_str]
        complete = 0
        incomplete = 0
        for rc, chunks in d['chunks_seen'].items():
            if len(chunks) >= 2:
                complete += 1
            else:
                incomplete += 1

        total_cycles = len(d['repeat_cycles'])

        # Expected cycles for a full day (144 = 24h × 6 per hour)
        coverage_pct = total_cycles / 144 * 100 if total_cycles > 0 else 0

        rows.append({
            'date': date_str,
            'body_files': d['body_files'],
            'trailer_files': d['trailer_files'],
            'repeat_cycles': total_cycles,
            'complete_2chunk': complete,
            'incomplete_1chunk': incomplete,
            'coverage_pct': round(coverage_pct, 1),
        })

    return rows, dates


def build_missing_timesteps(dates):
    """
    Compute the exact missing timesteps for each date.

    Compares the repeat cycles found in _raw_chunks against the full
    set of 144 possible cycles per day (00:00, 00:10, ..., 23:50).

    Args:
        dates (dict): Raw date data from summarize().

    Returns:
        dict: {
            "dates": {
                "2026-02-13": {
                    "present": 44,
                    "missing": 100,
                    "coverage_pct": 30.6,
                    "missing_times": ["00:00", "00:10", ...],
                    "incomplete_times": ["09:20"]  # only 1 of 2 chunks
                },
                ...
            },
            "summary": {
                "total_dates": 30,
                "total_present": 2774,
                "total_missing": 1546,
                "total_expected": 4320
            }
        }
    """
    ALL_CYCLES = set(range(1, 145))  # 1-indexed, 144 cycles per day

    result = {'dates': {}, 'summary': {}}
    total_present = 0
    total_missing = 0

    for date_str in sorted(dates.keys()):
        d = dates[date_str]
        present_rcs = d['repeat_cycles']
        missing_rcs = ALL_CYCLES - present_rcs

        # Convert rc numbers to HH:MM strings
        missing_times = []
        for rc in sorted(missing_rcs):
            total_min = (rc - 1) * 10
            h, m = divmod(total_min, 60)
            missing_times.append(f"{h:02d}:{m:02d}")

        # Find incomplete cycles (only 1 chunk instead of 2)
        incomplete_times = []
        for rc in sorted(present_rcs):
            chunks = d['chunks_seen'].get(rc, set())
            if len(chunks) < 2:
                total_min = (rc - 1) * 10
                h, m = divmod(total_min, 60)
                incomplete_times.append(f"{h:02d}:{m:02d}")

        n_present = len(present_rcs)
        n_missing = len(missing_rcs)
        coverage = n_present / 144 * 100

        total_present += n_present
        total_missing += n_missing

        result['dates'][date_str] = {
            'present': n_present,
            'missing': n_missing,
            'coverage_pct': round(coverage, 1),
            'missing_times': missing_times,
            'incomplete_times': incomplete_times,
        }

    total_expected = len(dates) * 144
    result['summary'] = {
        'total_dates': len(dates),
        'total_present': total_present,
        'total_missing': total_missing,
        'total_expected': total_expected,
        'overall_coverage_pct': round(
            total_present / total_expected * 100, 1
        ) if total_expected > 0 else 0,
    }

    return result


def save_missing_json(dates, output_path):
    """Build and save the missing timesteps JSON."""
    import json

    missing = build_missing_timesteps(dates)

    with open(output_path, 'w') as f:
        json.dump(missing, f, indent=2)

    s = missing['summary']
    print(f"Saved missing timesteps: {output_path}")
    print(f"  {s['total_present']}/{s['total_expected']} present "
          f"({s['overall_coverage_pct']}%), "
          f"{s['total_missing']} missing")


def print_table(rows):
    """Print a formatted table to stdout."""
    if not rows:
        print("No data found.")
        return

    header = (
        f"{'Date':<12} {'Body':>6} {'Trail':>6} {'Cycles':>7} "
        f"{'OK(2ch)':>8} {'Partial':>8} {'Coverage':>9}"
    )
    print(header)
    print("-" * len(header))

    total_body = 0
    total_cycles = 0

    for r in rows:
        total_body += r['body_files']
        total_cycles += r['repeat_cycles']
        print(
            f"{r['date']:<12} {r['body_files']:>6} {r['trailer_files']:>6} "
            f"{r['repeat_cycles']:>7} {r['complete_2chunk']:>8} "
            f"{r['incomplete_1chunk']:>8} {r['coverage_pct']:>8.1f}%"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<12} {total_body:>6} {'':>6} {total_cycles:>7}"
    )
    print(f"\n{len(rows)} dates, {total_body} body files, "
          f"{total_cycles} repeat cycles")


def save_csv(rows, output_path):
    """Save summary as CSV."""
    if not rows:
        print("No data to save.")
        return

    fieldnames = [
        'date', 'body_files', 'trailer_files', 'repeat_cycles',
        'complete_2chunk', 'incomplete_1chunk', 'coverage_pct',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Summarise downloaded FCI chunk files by date.'
    )
    parser.add_argument(
        '--raw_dir', type=str, default=DEFAULT_RAW_DIR,
        help=f'Path to _raw_chunks directory (default: {DEFAULT_RAW_DIR})',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default=str(PROJECT_ROOT / 'mtg_summary.csv'),
        help=(
            f'Output CSV path. Default lands at the project root '
            f'({PROJECT_ROOT / "mtg_summary.csv"}) so '
            f'intersect_product_coverage.py finds it regardless of '
            f'where the script is invoked from.'
        ),
    )
    parser.add_argument(
        '--missing', '-m', type=str, default='missing_timesteps.json',
        help='Output JSON with missing timesteps (default: missing_timesteps.json)',
    )

    args = parser.parse_args()

    rows, dates = summarize(args.raw_dir)
    print_table(rows)
    save_csv(rows, args.output)
    save_missing_json(dates, args.missing)