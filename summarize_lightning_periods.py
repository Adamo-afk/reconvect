"""
summarize_lightning_periods.py — Coverage / missing-timestep report for the
3-stage lightning cascade.

Reads the outputs of `identify_lightning_periods.py`
(`our_data/lightning_periods/lightning_bins.csv`, `lightning_periods.csv`,
`lightning_periods_config.json`) and produces:

    1. Per-date CSV table: how many bins survived each stage, plus per-stage
       rejection counts.
    2. Missing-timesteps JSON: per date, the exact HH:MM bins that were
       dropped, broken down by *why* (no_data / stage2 / stage3 / stage4),
       plus an overall summary block.

Mirrors the structure of `summarize_raw_chunks.py` (MTG) and
`summarize_raw_data.py` (NWCSAF) so the three reports feel consistent.

Usage:
    python summarize_lightning_periods.py
    python summarize_lightning_periods.py --periods_dir other/path
    python summarize_lightning_periods.py --output coverage.csv \\
        --missing missing.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# =============================================================================
# Defaults
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PERIODS_DIR = PROJECT_ROOT / "our_data" / "lightning_periods"

CONFIG_NAME = "lightning_periods_config.json"
BINS_NAME = "lightning_bins.csv"
PERIODS_NAME = "lightning_periods.csv"


# =============================================================================
# Loaders
# =============================================================================

def load_config(periods_dir: Path) -> dict:
    path = periods_dir / CONFIG_NAME
    if not path.exists():
        sys.exit(
            f"ERROR: {path} not found.\n"
            f"Run from the project root:\n"
            f"    python identify_lightning_periods.py"
        )
    return json.loads(path.read_text())


def load_bins(periods_dir: Path) -> list[dict]:
    """One row per (date, bin) — same schema as lightning_bins.csv."""
    path = periods_dir / BINS_NAME
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_periods(periods_dir: Path) -> dict[str, dict]:
    """Returns {date: row_dict} from lightning_periods.csv."""
    path = periods_dir / PERIODS_NAME
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row['date']] = row
    return out


# =============================================================================
# Classification
# =============================================================================

def classify_bin(bin_row: dict, day_row: dict) -> str:
    """
    Return one of: 'active', 'no_data', 'stage2_filter',
    'stage3_filter', 'stage4_filter'.

    The order matters: a bin can fail multiple stages, but we report the
    earliest failure (most informative for the user).
    """
    if int(bin_row['n_maps_in_bin']) == 0:
        return 'no_data'
    if int(bin_row['valid_bin']) == 0:
        return 'stage2_filter'
    if int(day_row['valid_day']) == 0:
        return 'stage3_filter'
    if int(day_row['window_kept']) == 0:
        return 'stage4_filter'
    return 'active'


# =============================================================================
# Aggregate
# =============================================================================

def summarize(bins: list[dict], periods: dict[str, dict], bins_per_day: int):
    """
    Build per-date stats and per-date missing-timestep breakdowns.

    Returns (rows, dates) where:
        rows: list of per-date dicts ready for CSV / printing
        dates: dict {date: {'classification': {bucket: [HH:MM, ...]},
                            'totals': {bucket: int}}}
    """
    dates: dict = defaultdict(lambda: {
        'classification': defaultdict(list),
        'totals':         defaultdict(int),
    })

    for b in bins:
        date_str = b['date']
        day_row = periods.get(date_str)
        if day_row is None:
            # Bin without a matching day row — shouldn't happen if the cascade
            # ran cleanly, but classify as no_data and move on.
            bucket = 'no_data'
        else:
            bucket = classify_bin(b, day_row)
        dates[date_str]['classification'][bucket].append(b['bin_start_utc'])
        dates[date_str]['totals'][bucket] += 1

    rows = []
    for date_str in sorted(dates.keys()):
        t = dates[date_str]['totals']
        day_row = periods.get(date_str, {})
        n_active = t.get('active', 0)
        coverage = (n_active / bins_per_day * 100) if bins_per_day else 0.0
        rows.append({
            'date':                  date_str,
            'bins_total':            bins_per_day,
            'bins_active':           n_active,
            'bins_no_data':          t.get('no_data', 0),
            'bins_stage2':           t.get('stage2_filter', 0),
            'bins_stage3':           t.get('stage3_filter', 0),
            'bins_stage4':           t.get('stage4_filter', 0),
            'valid_day':             int(day_row.get('valid_day', 0)),
            'window_kept':           int(day_row.get('window_kept', 0)),
            'kept':                  int(day_row.get('kept', 0)),
            'coverage_pct':          round(coverage, 1),
        })

    return rows, dict(dates)


# =============================================================================
# Missing-timesteps JSON
# =============================================================================

def build_missing_json(rows: list[dict], dates: dict,
                       config: dict, bins_per_day: int) -> dict:
    result: dict = {'config': config, 'dates': {}, 'summary': {}}
    total_active = 0
    total_expected = 0
    total_buckets = defaultdict(int)

    for r in rows:
        d = r['date']
        cls = dates[d]['classification']

        # Sort the HH:MM lists for deterministic output
        bucket_lists = {k: sorted(v) for k, v in cls.items()}

        result['dates'][d] = {
            'expected':           bins_per_day,
            'active':             r['bins_active'],
            'valid_day':          bool(r['valid_day']),
            'window_kept':        bool(r['window_kept']),
            'kept':               bool(r['kept']),
            'coverage_pct':       r['coverage_pct'],
            'missing_breakdown': {
                'no_data':        bucket_lists.get('no_data', []),
                'stage2_filter':  bucket_lists.get('stage2_filter', []),
                'stage3_filter':  bucket_lists.get('stage3_filter', []),
                'stage4_filter':  bucket_lists.get('stage4_filter', []),
            },
        }
        total_active += r['bins_active']
        total_expected += bins_per_day
        total_buckets['no_data']       += r['bins_no_data']
        total_buckets['stage2_filter'] += r['bins_stage2']
        total_buckets['stage3_filter'] += r['bins_stage3']
        total_buckets['stage4_filter'] += r['bins_stage4']

    result['summary'] = {
        'n_dates':              len(rows),
        'bins_per_day':         bins_per_day,
        'total_expected':       total_expected,
        'total_active':         total_active,
        'overall_coverage_pct': round(total_active / total_expected * 100, 1)
                                if total_expected > 0 else 0,
        'totals_by_bucket': {
            'no_data':        total_buckets['no_data'],
            'stage2_filter':  total_buckets['stage2_filter'],
            'stage3_filter':  total_buckets['stage3_filter'],
            'stage4_filter':  total_buckets['stage4_filter'],
        },
    }
    return result


# =============================================================================
# Output
# =============================================================================

def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No data found.")
        return

    header = (
        f"{'Date':<12} {'Total':>6} {'Active':>7} "
        f"{'NoDat':>6} {'St2':>5} {'St3':>5} {'St4':>5} "
        f"{'D/W/K':>7} {'Coverage':>9}"
    )
    print(header)
    print('-' * len(header))

    tot = {k: 0 for k in (
        'bins_total', 'bins_active', 'bins_no_data',
        'bins_stage2', 'bins_stage3', 'bins_stage4',
    )}
    n_kept = 0
    for r in rows:
        for k in tot:
            tot[k] += r[k]
        if r['kept']:
            n_kept += 1
        flag_str = (
            f"{r['valid_day']}/"
            f"{r['window_kept']}/"
            f"{r['kept']}"
        )
        print(
            f"{r['date']:<12} {r['bins_total']:>6} {r['bins_active']:>7} "
            f"{r['bins_no_data']:>6} {r['bins_stage2']:>5} "
            f"{r['bins_stage3']:>5} {r['bins_stage4']:>5} "
            f"{flag_str:>7} {r['coverage_pct']:>8.1f}%"
        )

    print('-' * len(header))
    overall_pct = (tot['bins_active'] / tot['bins_total'] * 100
                   if tot['bins_total'] else 0)
    print(
        f"{'TOTAL':<12} {tot['bins_total']:>6} {tot['bins_active']:>7} "
        f"{tot['bins_no_data']:>6} {tot['bins_stage2']:>5} "
        f"{tot['bins_stage3']:>5} {tot['bins_stage4']:>5} "
        f"{n_kept:>7d} {overall_pct:>8.1f}%"
    )
    print()
    print(f"{len(rows)} dates analysed, {n_kept} kept by the cascade")


def save_csv(rows: list[dict], output_path: str) -> None:
    if not rows:
        print("No data to save.")
        return
    fieldnames = [
        'date', 'bins_total', 'bins_active',
        'bins_no_data', 'bins_stage2', 'bins_stage3', 'bins_stage4',
        'valid_day', 'window_kept', 'kept', 'coverage_pct',
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {output_path}")


def save_missing_json(payload: dict, output_path: str) -> None:
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    s = payload['summary']
    print(f"Saved missing timesteps: {output_path}")
    print(f"  {s['total_active']}/{s['total_expected']} active bins "
          f"({s['overall_coverage_pct']}%)")
    print(f"  Dropped: no_data={s['totals_by_bucket']['no_data']}, "
          f"stage2={s['totals_by_bucket']['stage2_filter']}, "
          f"stage3={s['totals_by_bucket']['stage3_filter']}, "
          f"stage4={s['totals_by_bucket']['stage4_filter']}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise the lightning 3-stage cascade output: per-date "
                    "coverage table + missing-timesteps JSON."
    )
    parser.add_argument(
        '--periods_dir', type=str, default=str(DEFAULT_PERIODS_DIR),
        help=f'Directory containing the cascade output '
             f'(default: {DEFAULT_PERIODS_DIR})',
    )
    parser.add_argument(
        '--output', '-o', type=str, default='lightning_summary.csv',
        help='Output CSV filename (default: lightning_summary.csv)',
    )
    parser.add_argument(
        '--missing', '-m', type=str,
        default='lightning_missing_timesteps.json',
        help='Output JSON with missing timesteps per stage '
             '(default: lightning_missing_timesteps.json)',
    )
    args = parser.parse_args()

    periods_dir = Path(args.periods_dir)
    config = load_config(periods_dir)
    bins = load_bins(periods_dir)
    periods = load_periods(periods_dir)

    bins_per_day = int(config['bins_per_day'])

    print("=" * 70)
    print("Lightning Cascade Summary")
    print("=" * 70)
    print(f"Periods dir         : {periods_dir}")
    print(f"step_minutes        : {config['step_minutes']}")
    print(f"aggregation_minutes : {config['aggregation_minutes']}")
    print(f"bins per day        : {bins_per_day}")
    print(f"Product             : {config['lightning_product']}")
    print(f"Thresholds          : map={config['map_fraction_threshold']:g}, "
          f"day={config['day_min_valid_fraction']}, "
          f"window={config['window_min_valid_days']}/"
          f"{config['window_days']} days")
    print()

    rows, dates_detail = summarize(bins, periods, bins_per_day)
    print_table(rows)

    save_csv(rows, args.output)
    payload = build_missing_json(rows, dates_detail, config, bins_per_day)
    save_missing_json(payload, args.missing)

    return 0


if __name__ == '__main__':
    sys.exit(main())
