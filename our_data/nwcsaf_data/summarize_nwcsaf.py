"""
summarize_nwcsaf.py — Summarise downloaded NWCSAF files by date.

Scans the `_raw_data/` directory produced by `pipeline_nwcsaf.py`, parses
SAFNWC filenames, and produces:

    1. A per-date CSV table (file counts, coverage vs the configured cadence,
       complete vs incomplete CMIC+CTTH pairs).
    2. A JSON report listing the exact missing timesteps per date.

Expected per-day timestep count is derived from `our_data/timestep_config.json`
(the same source the download pipeline uses), so if you ran
`validate_timestep.py --step_minutes 15` the script will compare against
96 expected timesteps/day filtered by `{00, 10, 30, 40}`. Override the
cadence with `--timesteps 00 10 30 40 ...` if needed.

Usage:
    python our_data/nwcsaf_data/summarize_nwcsaf.py
    python our_data/nwcsaf_data/summarize_nwcsaf.py --raw_dir path/to/_raw_data
    python our_data/nwcsaf_data/summarize_nwcsaf.py --output summary.csv \\
        --missing missing.json
    python our_data/nwcsaf_data/summarize_nwcsaf.py --timesteps 00 30
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"

DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "_raw_data"

NWCSAF_PRODUCTS = ["cmic", "ctth"]

# Same regex as pipeline_nwcsaf.py / nwcsaf_arrange.py
FILENAME_PATTERN = re.compile(
    r'^S_NWC_(?P<product>\w+)_\w+_[\w-]+_'
    r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    r'T(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z'
    r'(?P<suffix>.*?)\.nc$'
)


# =============================================================================
# Cadence config
# =============================================================================

def load_minute_filter() -> tuple[set[int], int]:
    """
    Read the NWCSAF minute filter and the chosen step from timestep_config.json.

    Returns (minute_filter set, step_minutes).
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
    flt = cfg["products"].get("nwcsaf", {}).get("filter")
    if flt is None:
        print(
            f"ERROR: product 'nwcsaf' has no minute filter in {TIMESTEP_CONFIG_PATH}",
            file=sys.stderr,
        )
        sys.exit(2)
    return set(flt), int(cfg["step_minutes"])


# =============================================================================
# Filename parsing
# =============================================================================

def parse_nwcsaf_filename(filename: str) -> dict | None:
    """Extract product / sensing_dt / minute / is_plax from a SAFNWC filename."""
    match = FILENAME_PATTERN.match(os.path.basename(filename))
    if not match:
        return None
    try:
        sensing_dt = datetime.datetime(
            int(match['year']), int(match['month']), int(match['day']),
            int(match['hour']), int(match['minute']), int(match['second']),
        )
    except ValueError:
        return None
    return {
        'product':    match['product'].lower(),
        'sensing_dt': sensing_dt,
        'minute':     int(match['minute']),
        'is_plax':    'PLAX' in (match['suffix'] or ''),
    }


# =============================================================================
# Summarise
# =============================================================================

def summarize(raw_dir: Path,
              products: list[str],
              minute_filter: set[int]) -> tuple[list[dict], dict]:
    """
    Scan `raw_dir` and group files by date.

    Returns (rows, dates) where rows is a list of per-date summary dicts and
    dates is a richer per-date structure used for the missing-timesteps JSON.
    """
    if not raw_dir.is_dir():
        print(f"ERROR: directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    nc_files = [f for f in os.listdir(raw_dir) if f.endswith('.nc')]
    print(f"Found {len(nc_files)} .nc files in {raw_dir}")

    products_lower = {p.lower() for p in products}

    # date_str -> { product -> set of "HH:MM" timesteps present }
    dates: dict = defaultdict(lambda: {
        'per_product': defaultdict(set),  # product -> set of HHMM-on-grid keys
        'plax_skipped': 0,
        'unparsable': 0,
        'off_grid':   0,                  # minute not in filter
        'off_product': 0,                 # product not requested
    })

    for filename in nc_files:
        info = parse_nwcsaf_filename(filename)
        if info is None:
            # We can't tell which date — keep a flat counter under an
            # 'unknown' key so it shows up in the summary.
            dates['unknown']['unparsable'] += 1
            continue
        date_str = info['sensing_dt'].strftime('%Y-%m-%d')
        if info['is_plax']:
            dates[date_str]['plax_skipped'] += 1
            continue
        if info['product'] not in products_lower:
            dates[date_str]['off_product'] += 1
            continue
        if info['minute'] not in minute_filter:
            dates[date_str]['off_grid'] += 1
            continue
        hhmm = info['sensing_dt'].strftime('%H:%M')
        dates[date_str]['per_product'][info['product']].add(hhmm)

    # Build flat rows for the CSV / table
    rows = []
    expected_per_day = expected_timesteps_per_day(minute_filter)

    for date_str in sorted(d for d in dates.keys() if d != 'unknown'):
        d = dates[date_str]
        per_product_counts = {
            p: len(d['per_product'].get(p, set()))
            for p in products_lower
        }

        if len(products_lower) >= 2:
            present_all = set.intersection(*[
                d['per_product'].get(p, set()) for p in products_lower
            ])
        else:
            present_all = next(iter(d['per_product'].values()), set())
        complete_pairs = len(present_all)

        union = set()
        for p in products_lower:
            union |= d['per_product'].get(p, set())
        incomplete_pairs = len(union) - complete_pairs

        coverage = (complete_pairs / expected_per_day * 100
                    if expected_per_day else 0)

        rows.append({
            'date':            date_str,
            **{f"{p}_files": per_product_counts[p] for p in products_lower},
            'complete_pairs':   complete_pairs,
            'incomplete_pairs': incomplete_pairs,
            'expected':         expected_per_day,
            'coverage_pct':     round(coverage, 1),
            'plax_skipped':     d['plax_skipped'],
            'off_grid':         d['off_grid'],
        })

    return rows, dates


def expected_timesteps_per_day(minute_filter: set[int]) -> int:
    """Number of grid timesteps per 24h day given the minute filter."""
    return len(minute_filter) * 24  # 24 hours × minutes per hour


# =============================================================================
# Missing-timestep report
# =============================================================================

def build_missing_timesteps(dates: dict,
                            products: list[str],
                            minute_filter: set[int]) -> dict:
    """
    Compute the exact missing timesteps per date.

    For each date, derive the full set of expected `HH:MM` timestamps from
    the configured cadence (24 hours × minute_filter), then diff against
    what is present per product.

    Returns a dict ready for JSON serialisation.
    """
    products_lower = [p.lower() for p in products]

    expected_hhmm = sorted(
        f"{h:02d}:{m:02d}"
        for h in range(24)
        for m in sorted(minute_filter)
    )
    expected_set = set(expected_hhmm)

    result = {'dates': {}, 'summary': {}}
    total_present = 0   # complete pairs (or singletons if only one product)
    total_missing = 0

    for date_str in sorted(d for d in dates.keys() if d != 'unknown'):
        d = dates[date_str]
        per_product_missing = {}
        for p in products_lower:
            present = d['per_product'].get(p, set())
            missing = sorted(expected_set - present)
            per_product_missing[p] = missing

        if len(products_lower) >= 2:
            present_all = set.intersection(*[
                d['per_product'].get(p, set()) for p in products_lower
            ])
        else:
            present_all = next(iter(d['per_product'].values()), set())

        union = set()
        for p in products_lower:
            union |= d['per_product'].get(p, set())

        missing_completely = sorted(expected_set - union)
        partial = sorted(union - present_all)  # at least one product missing
        complete = sorted(present_all)

        coverage = (len(complete) / len(expected_hhmm) * 100
                    if expected_hhmm else 0)

        total_present += len(complete)
        total_missing += len(expected_hhmm) - len(complete)

        result['dates'][date_str] = {
            'expected':            len(expected_hhmm),
            'complete':            len(complete),
            'partial':             len(partial),
            'missing_completely':  len(missing_completely),
            'coverage_pct':        round(coverage, 1),
            'missing_completely_times': missing_completely,
            'partial_times':       partial,
            'per_product_missing': per_product_missing,
        }

    total_expected = len([d for d in dates if d != 'unknown']) * len(expected_hhmm)
    result['summary'] = {
        'products':              products_lower,
        'minute_filter':         sorted(minute_filter),
        'expected_per_day':      len(expected_hhmm),
        'total_dates':           len([d for d in dates if d != 'unknown']),
        'total_complete_pairs':  total_present,
        'total_missing':         total_missing,
        'total_expected':        total_expected,
        'overall_coverage_pct':  round(
            total_present / total_expected * 100, 1
        ) if total_expected > 0 else 0,
    }

    return result


# =============================================================================
# Output
# =============================================================================

def print_table(rows: list[dict], products: list[str]) -> None:
    if not rows:
        print("No data found.")
        return

    products_lower = [p.lower() for p in products]

    file_col_headers = " ".join(f"{p.upper():>6}" for p in products_lower)
    header = (
        f"{'Date':<12} {file_col_headers} "
        f"{'Complete':>9} {'Partial':>8} {'Expected':>9} {'Coverage':>9}"
    )
    print(header)
    print("-" * len(header))

    totals = {f"{p}_files": 0 for p in products_lower}
    totals.update({'complete_pairs': 0, 'incomplete_pairs': 0, 'expected': 0})

    for r in rows:
        for p in products_lower:
            totals[f"{p}_files"] += r[f"{p}_files"]
        totals['complete_pairs'] += r['complete_pairs']
        totals['incomplete_pairs'] += r['incomplete_pairs']
        totals['expected'] += r['expected']

        file_cols = " ".join(f"{r[f'{p}_files']:>6}" for p in products_lower)
        print(
            f"{r['date']:<12} {file_cols} "
            f"{r['complete_pairs']:>9} {r['incomplete_pairs']:>8} "
            f"{r['expected']:>9} {r['coverage_pct']:>8.1f}%"
        )

    print("-" * len(header))
    file_cols = " ".join(f"{totals[f'{p}_files']:>6}" for p in products_lower)
    overall_pct = (
        totals['complete_pairs'] / totals['expected'] * 100
        if totals['expected'] else 0
    )
    print(
        f"{'TOTAL':<12} {file_cols} "
        f"{totals['complete_pairs']:>9} {totals['incomplete_pairs']:>8} "
        f"{totals['expected']:>9} {overall_pct:>8.1f}%"
    )
    print(f"\n{len(rows)} dates")


def save_csv(rows: list[dict], output_path: str,
             products: list[str]) -> None:
    if not rows:
        print("No data to save.")
        return

    products_lower = [p.lower() for p in products]
    fieldnames = (
        ['date']
        + [f"{p}_files" for p in products_lower]
        + ['complete_pairs', 'incomplete_pairs', 'expected', 'coverage_pct',
           'plax_skipped', 'off_grid']
    )

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def save_missing_json(dates: dict,
                      products: list[str],
                      minute_filter: set[int],
                      output_path: str) -> None:
    payload = build_missing_timesteps(dates, products, minute_filter)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    s = payload['summary']
    print(f"Saved missing timesteps: {output_path}")
    print(f"  {s['total_complete_pairs']}/{s['total_expected']} present "
          f"({s['overall_coverage_pct']}%), "
          f"{s['total_missing']} missing")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise downloaded NWCSAF files by date and report "
                    "missing timesteps relative to the configured cadence."
    )
    parser.add_argument(
        '--raw_dir', type=str, default=str(DEFAULT_RAW_DIR),
        help=f'Path to the flat NWCSAF cache (default: {DEFAULT_RAW_DIR})',
    )
    parser.add_argument(
        '--products', type=str, nargs='+',
        default=NWCSAF_PRODUCTS, choices=NWCSAF_PRODUCTS,
        help='Products to include (default: cmic ctth)',
    )
    parser.add_argument(
        '--timesteps', type=str, nargs='+', default=None,
        help='Override the cadence minute filter (e.g. 00 10 30 40). '
             "Default: read from timestep_config.json.",
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default=str(PROJECT_ROOT / 'nwcsaf_summary.csv'),
        help='Output CSV filename (default: nwcsaf_summary.csv)',
    )
    parser.add_argument(
        '--missing', '-m', type=str,
        default=str(PROJECT_ROOT / 'nwcsaf_missing_timesteps.json'),
        help='Output JSON with missing timesteps '
             '(default: nwcsaf_missing_timesteps.json)',
    )

    args = parser.parse_args()

    if args.timesteps is not None:
        minute_filter = {int(m) for m in args.timesteps}
        step = None
        step_src = "CLI override"
    else:
        minute_filter, step = load_minute_filter()
        step_src = f"timestep_config.json (step={step} min)"

    print("=" * 70)
    print("NWCSAF Raw-Data Summary")
    print("=" * 70)
    print(f"Cache dir      : {args.raw_dir}")
    print(f"Products       : {args.products}")
    print(f"Minute filter  : {sorted(minute_filter)} ({step_src})")
    print(f"Expected/day   : {expected_timesteps_per_day(minute_filter)}")
    print()

    rows, dates = summarize(Path(args.raw_dir), args.products, minute_filter)
    print_table(rows, args.products)
    save_csv(rows, args.output, args.products)
    save_missing_json(dates, args.products, minute_filter, args.missing)
