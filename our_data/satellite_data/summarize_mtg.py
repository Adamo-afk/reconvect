"""
summarize_mtg.py - Summarise downloaded MTG FCI chunks by date.

Scans the _raw_chunks directory, parses FCI filenames, and produces a
per-date CSV + missing-timesteps JSON. Coverage is measured against the
configured MTG minute filter (`products.mtg.filter` from
timestep_config.json) - the same set pipeline_msg_mtg.py uses to filter
downloads before transfer - so a fully-covered day reports 100%, not
"66.7%" against the unfiltered 144-cycle native cadence.

Usage:
    python summarize_mtg.py
    python summarize_mtg.py --raw_dir path/to/_raw_chunks
    python summarize_mtg.py --output summary.csv
"""

import os
import re
import sys
import csv
import json
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
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / 'our_data' / 'timestep_config.json'

# MTG native cadence (repeat cycle = 10 min); the file has 144 cycles
# per 24h day. The filter from timestep_config.json picks a subset of
# the cycles that align with the master grid.
NATIVE_CADENCE_MINUTES = 10
NATIVE_CYCLES_PER_DAY = 1440 // NATIVE_CADENCE_MINUTES  # 144


# =============================================================================
# Cadence config
# =============================================================================

def load_minute_filter():
    """Read the MTG minute filter + step from timestep_config.json.

    Returns (filter set, step_minutes). The filter is the minute-of-hour
    set the master grid snaps to for MTG - exactly the set
    pipeline_msg_mtg.py pre-filters on before downloading.
    """
    if not TIMESTEP_CONFIG_PATH.exists():
        print(
            f"ERROR: timestep config not found at {TIMESTEP_CONFIG_PATH}.\n"
            f"Run from the project root:\n"
            f"    python validate_timestep.py --step_minutes <N>",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = json.loads(TIMESTEP_CONFIG_PATH.read_text())
    flt = cfg.get('products', {}).get('mtg', {}).get('filter')
    if flt is None:
        print(
            f"ERROR: product 'mtg' has no minute filter in "
            f"{TIMESTEP_CONFIG_PATH}. Re-run validate_timestep.py with a "
            f"non-continuous mtg cadence in product_cadences.config.",
            file=sys.stderr,
        )
        sys.exit(2)
    return set(int(m) for m in flt), int(cfg['step_minutes'])


def rc_minute_of_hour(rc):
    """Minute-of-hour for a 1-indexed MTG repeat cycle (10-min native)."""
    return ((rc - 1) * NATIVE_CADENCE_MINUTES) % 60


def expected_rc_set(filter_minutes):
    """Set of MTG repeat-cycle numbers that fall on the configured filter.

    For a 10-min cadence and a filter of e.g. {0, 10, 30, 40}, this picks
    cycles whose minute-of-hour is in the filter - typically 4 of every
    6 cycles per hour, so 96 per day at step=15.
    """
    return {
        rc for rc in range(1, NATIVE_CYCLES_PER_DAY + 1)
        if rc_minute_of_hour(rc) in filter_minutes
    }


# =============================================================================
# Filename parsing
# =============================================================================

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


def nominal_time(date_ref, repeat_cycle_in_day, period_min=NATIVE_CADENCE_MINUTES):
    """Compute nominal start time from repeat cycle number."""
    base = date_ref.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + datetime.timedelta(
        minutes=(repeat_cycle_in_day - 1) * period_min
    )


def rc_to_hhmm(rc):
    """1-indexed repeat cycle -> 'HH:MM' string."""
    total_min = (rc - 1) * NATIVE_CADENCE_MINUTES
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}"


# =============================================================================
# Summarise
# =============================================================================

def summarize(raw_dir, filter_minutes):
    """
    Scan _raw_chunks and build a per-date summary.

    Returns (rows, dates) where:
        rows: list of per-date dicts ready for CSV / printing
        dates: raw per-date data passed to build_missing_timesteps
    """
    if not os.path.isdir(raw_dir):
        print(f"ERROR: directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    expected_rcs = expected_rc_set(filter_minutes)
    expected_per_day = len(expected_rcs)

    nc_files = [f for f in os.listdir(raw_dir) if f.endswith('.nc')]
    print(f"Found {len(nc_files)} .nc files in {raw_dir}")

    dates = defaultdict(lambda: {
        'body_files':     0,
        'trailer_files':  0,
        'repeat_cycles':  set(),       # all rc numbers present
        'chunks_seen':    defaultdict(set),  # rc_num -> set of chunk_numbers
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

        present_rcs = d['repeat_cycles']
        on_grid_present = present_rcs & expected_rcs
        off_grid_present = present_rcs - expected_rcs

        coverage_pct = (
            len(on_grid_present) / expected_per_day * 100
            if expected_per_day else 0
        )

        rows.append({
            'date':              date_str,
            'body_files':        d['body_files'],
            'trailer_files':     d['trailer_files'],
            'repeat_cycles':     len(present_rcs),       # total (audit)
            'on_grid':           len(on_grid_present),   # match the filter
            'off_grid':          len(off_grid_present),  # outside the filter
            'expected':          expected_per_day,
            'complete_2chunk':   complete,
            'incomplete_1chunk': incomplete,
            'coverage_pct':      round(coverage_pct, 1),
        })

    return rows, dates


def build_missing_timesteps(dates, filter_minutes):
    """
    Compute the exact missing on-grid timesteps for each date.

    Only on-grid cycles count - intersect_product_coverage.py snaps the
    master HHMM to the MTG filter, so off-grid native cycles are
    unreachable. They're tracked separately for diagnostics but not as
    "missing".
    """
    expected_rcs = expected_rc_set(filter_minutes)

    result = {
        'config': {'minute_filter': sorted(filter_minutes)},
        'dates': {},
        'summary': {},
    }
    total_present = 0
    total_missing = 0

    for date_str in sorted(dates.keys()):
        d = dates[date_str]
        present_rcs = d['repeat_cycles']
        on_grid_present = present_rcs & expected_rcs
        on_grid_missing = expected_rcs - present_rcs
        off_grid_present = present_rcs - expected_rcs

        missing_times = sorted(rc_to_hhmm(rc) for rc in on_grid_missing)
        off_grid_times = sorted(rc_to_hhmm(rc) for rc in off_grid_present)

        incomplete_times = []
        for rc in sorted(on_grid_present):
            chunks = d['chunks_seen'].get(rc, set())
            if len(chunks) < 2:
                incomplete_times.append(rc_to_hhmm(rc))

        n_present = len(on_grid_present)
        n_missing = len(on_grid_missing)
        coverage = (
            n_present / len(expected_rcs) * 100 if expected_rcs else 0
        )

        total_present += n_present
        total_missing += n_missing

        result['dates'][date_str] = {
            'present':          n_present,
            'missing':          n_missing,
            'off_grid_present': len(off_grid_present),
            'expected':         len(expected_rcs),
            'coverage_pct':     round(coverage, 1),
            'missing_times':    missing_times,
            'off_grid_times':   off_grid_times,
            'incomplete_times': incomplete_times,
        }

    total_expected = len(dates) * len(expected_rcs)
    result['summary'] = {
        'total_dates':           len(dates),
        'expected_per_day':      len(expected_rcs),
        'total_present':         total_present,
        'total_missing':         total_missing,
        'total_expected':        total_expected,
        'overall_coverage_pct':  round(
            total_present / total_expected * 100, 1
        ) if total_expected > 0 else 0,
    }

    return result


def save_missing_json(dates, filter_minutes, output_path):
    """Build and save the missing timesteps JSON."""
    missing = build_missing_timesteps(dates, filter_minutes)
    with open(output_path, 'w') as f:
        json.dump(missing, f, indent=2)

    s = missing['summary']
    print(f"Saved missing timesteps: {output_path}")
    print(f"  {s['total_present']}/{s['total_expected']} present "
          f"({s['overall_coverage_pct']}%), "
          f"{s['total_missing']} missing")


# =============================================================================
# Output
# =============================================================================

def print_table(rows):
    """Print a formatted table to stdout."""
    if not rows:
        print("No data found.")
        return

    header = (
        f"{'Date':<12} {'Body':>6} {'Trail':>6} {'Cycles':>7} "
        f"{'OnGrid':>7} {'OffGrid':>8} {'Exp':>5} "
        f"{'OK(2ch)':>8} {'Partial':>8} {'Coverage':>9}"
    )
    print(header)
    print("-" * len(header))

    total_body = 0
    total_cycles = 0
    total_on_grid = 0

    for r in rows:
        total_body += r['body_files']
        total_cycles += r['repeat_cycles']
        total_on_grid += r['on_grid']
        print(
            f"{r['date']:<12} {r['body_files']:>6} {r['trailer_files']:>6} "
            f"{r['repeat_cycles']:>7} {r['on_grid']:>7} {r['off_grid']:>8} "
            f"{r['expected']:>5} "
            f"{r['complete_2chunk']:>8} {r['incomplete_1chunk']:>8} "
            f"{r['coverage_pct']:>8.1f}%"
        )

    print("-" * len(header))
    overall_pct = (
        total_on_grid / (rows[0]['expected'] * len(rows)) * 100
        if rows and rows[0]['expected'] else 0
    )
    print(
        f"{'TOTAL':<12} {total_body:>6} {'':>6} "
        f"{total_cycles:>7} {total_on_grid:>7}"
        + ' ' * (8 + 1 + 5 + 1 + 8 + 1 + 8 + 1)
        + f"{overall_pct:>8.1f}%"
    )
    print(f"\n{len(rows)} dates, {total_body} body files, "
          f"{total_cycles} repeat cycles total "
          f"({total_on_grid} on grid)")


def save_csv(rows, output_path):
    """Save summary as CSV."""
    if not rows:
        print("No data to save.")
        return

    fieldnames = [
        'date', 'body_files', 'trailer_files', 'repeat_cycles',
        'on_grid', 'off_grid', 'expected',
        'complete_2chunk', 'incomplete_1chunk', 'coverage_pct',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


# =============================================================================
# CLI
# =============================================================================

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
        '--missing', '-m', type=str,
        default=str(PROJECT_ROOT / 'mtg_missing_timesteps.json'),
        help=(
            f'Output JSON with missing timesteps '
            f'(default: {PROJECT_ROOT / "mtg_missing_timesteps.json"} - '
            f'anchored to the project root, matching the nwcsaf / opera '
            f'naming convention).'
        ),
    )
    parser.add_argument(
        '--timesteps', type=str, nargs='+', default=None,
        help="Override the cadence minute filter (e.g. 00 10 30 40). "
             "Default: read from timestep_config.json.",
    )
    parser.add_argument(
        '--fill-from-datastore', action='store_true',
        help="After summarising, fetch every missing / incomplete repeat "
             "cycle from the EUMETSAT Data Store (FDHSI) into --raw_dir, "
             "then re-summarise and record what was obtained. Requires "
             "the `eumdac` package and EUMDAC credentials.",
    )
    parser.add_argument(
        '--eumdac_credentials', type=str, default=None,
        help="Two-line text file for --fill-from-datastore: EUMDAC key on "
             "line 1, secret on line 2. Falls back to the EUMDAC_KEY and "
             "EUMDAC_SECRET environment variables.",
    )
    parser.add_argument(
        '--fill-dry-run', action='store_true',
        help="With --fill-from-datastore, list what would be fetched "
             "without downloading anything.",
    )
    parser.add_argument(
        '--no-fill-incomplete', action='store_true',
        help="With --fill-from-datastore, recover only fully-missing "
             "cycles and leave one-chunk cycles alone.",
    )

    args = parser.parse_args()

    if args.timesteps is not None:
        filter_minutes = {int(m) for m in args.timesteps}
        step_src = 'CLI override'
    else:
        filter_minutes, step = load_minute_filter()
        step_src = f'timestep_config.json (step={step} min)'

    print("=" * 70)
    print("MTG Cache Summary")
    print("=" * 70)
    print(f"Raw dir        : {args.raw_dir}")
    print(f"Minute filter  : {sorted(filter_minutes)} ({step_src})")
    print(f"Expected/day   : {len(expected_rc_set(filter_minutes))} "
          f"(of {NATIVE_CYCLES_PER_DAY} native cycles)")
    print()

    rows, dates = summarize(args.raw_dir, filter_minutes)
    print_table(rows)
    save_csv(rows, args.output)
    save_missing_json(dates, filter_minutes, args.missing)

    if not args.fill_from_datastore:
        sys.exit(0)

    # ---- Backfill pass -------------------------------------------------
    # The first summary above located the gaps. Fetch them from the Data
    # Store, then summarise a SECOND time so the CSV / JSON on disk
    # describe the post-backfill state, and record what was obtained.
    from datastore_fill import collect_gaps, fill_gaps, print_report

    before_json = build_missing_timesteps(dates, filter_minutes)
    before = before_json['summary']
    gaps = collect_gaps(before_json,
                        include_incomplete=not args.no_fill_incomplete)

    report = fill_gaps(
        gaps,
        raw_dir=args.raw_dir,
        credentials_file=args.eumdac_credentials,
        dry_run=args.fill_dry_run,
    )
    print_report(report)

    if report['files_downloaded'] and not args.fill_dry_run:
        print("\nRe-summarising after backfill ...\n")
        rows, dates = summarize(args.raw_dir, filter_minutes)
        print_table(rows)
        save_csv(rows, args.output)

    after = build_missing_timesteps(dates, filter_minutes)
    report['coverage_before_pct'] = before['overall_coverage_pct']
    report['coverage_after_pct'] = after['summary']['overall_coverage_pct']
    report['missing_before'] = before['total_missing']
    report['missing_after'] = after['summary']['total_missing']
    after['datastore_fill'] = report

    with open(args.missing, 'w') as f:
        json.dump(after, f, indent=2)

    print(f"\nCoverage {report['coverage_before_pct']}% -> "
          f"{report['coverage_after_pct']}%  "
          f"({report['missing_before']} -> {report['missing_after']} missing)")
    print(f"Backfill record written into {args.missing} "
          f"under `datastore_fill`.")
