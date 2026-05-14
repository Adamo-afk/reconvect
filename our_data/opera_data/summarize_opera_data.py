"""
summarize_opera_data.py — Coverage / missing-timestep report for the OPERA cache.

Scans `our_data/opera_data/{reflectivity,rainfall_rate}/{YYYY}/{MM}/{DD}/`,
parses the timestamp from each `.h5` filename, and produces a per-date
per-product table plus a JSON of the exact missing timestamps relative to
the cadence configured in `our_data/timestep_config.json`.

Mirrors the layout of `summarize_mtg.py` (MTG) and
`summarize_nwcsaf.py` (NWCSAF) so the three reports feel consistent.

Usage:
    python our_data/opera_data/summarize_opera_data.py
    python our_data/opera_data/summarize_opera_data.py --data_dir other/path
    python our_data/opera_data/summarize_opera_data.py \\
        --output opera_summary.csv --missing opera_missing.json
    python our_data/opera_data/summarize_opera_data.py --timesteps all
    python our_data/opera_data/summarize_opera_data.py --products opera_reflectivity
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

DEFAULT_DATA_DIR = Path(__file__).resolve().parent  # our_data/opera_data

# product_key -> (local subdirectory, native cadence in minutes if config missing)
PRODUCTS = {
    "opera_reflectivity":  {"subdir": "reflectivity",  "default_cadence": 15},
    "opera_rainfall_rate": {"subdir": "rainfall_rate", "default_cadence": 15},
}

TIMESTAMP_PATTERN_ISO = re.compile(
    r'(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})(\d{2})Z?'
)
TIMESTAMP_PATTERN_COMPACT = re.compile(r'(\d{12,14})')


# =============================================================================
# Cadence config
# =============================================================================

def load_minute_filter(product_key: str):
    """
    Returns (minute_filter set, step_minutes, native_cadence_minutes).

    minute_filter is the configured set of allowed minutes for the product;
    native_cadence_minutes is read from products[product].cadence_minutes
    (or falls back to the per-product default).
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
    prod_cfg = cfg["products"].get(product_key)
    if prod_cfg is None or prod_cfg.get("filter") is None:
        print(
            f"ERROR: product '{product_key}' has no minute filter in "
            f"{TIMESTEP_CONFIG_PATH}. Add it to product_cadences.json and "
            f"re-run validate_timestep.py.",
            file=sys.stderr,
        )
        sys.exit(2)
    return (
        set(prod_cfg["filter"]),
        int(cfg["step_minutes"]),
        int(prod_cfg["cadence_minutes"]),
    )


# =============================================================================
# Filename parsing
# =============================================================================

def parse_opera_timestamp(filename: str) -> datetime.datetime | None:
    """Supports ISO (`2026-05-11T000500Z-...`) and compact (`...20260511000500.h5`)."""
    m = TIMESTAMP_PATTERN_ISO.search(filename)
    if m:
        try:
            return datetime.datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            )
        except ValueError:
            pass
    m = TIMESTAMP_PATTERN_COMPACT.search(filename)
    if m:
        ts = m.group(1)[:12]
        try:
            return datetime.datetime.strptime(ts, '%Y%m%d%H%M')
        except ValueError:
            pass
    return None


def expected_hhmm(minute_filter: set[int]) -> list[str]:
    """Sorted list of every HH:MM the cadence expects across a 24h day."""
    return sorted(
        f"{h:02d}:{m:02d}"
        for h in range(24)
        for m in sorted(minute_filter)
    )


# =============================================================================
# Summarise
# =============================================================================

def discover_files(product_root: Path) -> dict[str, set[str]]:
    """
    Walk `{root}/{YYYY}/{MM}/{DD}/*.h5` and return {date_str: set_of_HH:MM}.
    """
    out: dict[str, set[str]] = defaultdict(set)
    if not product_root.is_dir():
        return out

    for year_dir in sorted(product_root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue
                for f in os.listdir(day_dir):
                    if not f.endswith('.h5'):
                        continue
                    ts = parse_opera_timestamp(f)
                    if ts is None:
                        continue
                    date_str = ts.strftime('%Y-%m-%d')
                    hhmm = ts.strftime('%H:%M')
                    out[date_str].add(hhmm)
    return out


def summarize(data_dir: Path,
              products: list[str],
              minute_filters: dict[str, set[int]]):
    """
    Returns (rows, per_date_detail).

    rows: list of per-date dicts ready for CSV / table
    per_date_detail: {date: {product: {'present': set, 'missing': list}}}
    """
    expected_hhmm_by_product: dict[str, list[str]] = {
        p: expected_hhmm(minute_filters[p]) for p in products
    }
    expected_set_by_product: dict[str, set[str]] = {
        p: set(expected_hhmm_by_product[p]) for p in products
    }

    present_by_product: dict[str, dict[str, set[str]]] = {}
    for p in products:
        root = data_dir / PRODUCTS[p]["subdir"]
        present_by_product[p] = discover_files(root)

    # Union of all dates seen for any product
    all_dates: set[str] = set()
    for p in products:
        all_dates.update(present_by_product[p].keys())

    rows = []
    detail: dict = {}
    for date_str in sorted(all_dates):
        row = {'date': date_str}
        date_detail: dict = {}
        kept_by_product: dict[str, set[str]] = {}
        for p in products:
            present = present_by_product[p].get(date_str, set())
            expected = expected_set_by_product[p]
            kept = present & expected
            kept_by_product[p] = kept
            off_grid = len(present - expected)
            missing = sorted(expected - present)
            coverage = (len(kept) / len(expected) * 100) if expected else 0.0
            row[f'{p}_files']         = len(present)
            row[f'{p}_on_grid']       = len(kept)
            row[f'{p}_off_grid']      = off_grid
            row[f'{p}_expected']      = len(expected)
            row[f'{p}_coverage_pct']  = round(coverage, 1)
            date_detail[p] = {
                'present_on_grid':  sorted(kept),
                'off_grid':         sorted(present - expected),
                'missing':          missing,
            }

        # Unified coverage = intersection of all requested products on the
        # minute grid. Matches the NWCSAF "complete_pairs / expected" idea
        # so `intersect_product_coverage.py` can read this CSV with the
        # default `:coverage_pct` column and no per-product override.
        if products:
            present_intersection = set.intersection(*kept_by_product.values())
            expected_intersection = set.intersection(
                *(expected_set_by_product[p] for p in products)
            )
        else:
            present_intersection = set()
            expected_intersection = set()
        unified = (
            len(present_intersection) / len(expected_intersection) * 100
            if expected_intersection else 0.0
        )
        row['complete_intersection'] = len(present_intersection)
        row['expected_intersection'] = len(expected_intersection)
        row['coverage_pct']          = round(unified, 1)

        rows.append(row)
        detail[date_str] = date_detail

    return rows, detail


# =============================================================================
# Output
# =============================================================================

def print_table(rows: list[dict], products: list[str]) -> None:
    if not rows:
        print("No data found.")
        return

    cols: list[tuple[str, str]] = [('Date', 'date')]
    for p in products:
        short = 'REFL' if p == 'opera_reflectivity' else 'RAIN'
        cols.append((f"{short} ok", f"{p}_on_grid"))
        cols.append((f"{short} exp", f"{p}_expected"))
        cols.append((f"{short} cov%", f"{p}_coverage_pct"))

    widths = {name: max(len(name), 9) for name, _ in cols}
    widths['Date'] = 12

    header = " ".join(f"{name:>{widths[name]}}" if name != 'Date'
                      else f"{name:<{widths[name]}}"
                      for name, _ in cols)
    print(header)
    print("-" * len(header))

    totals = {key: 0 for _, key in cols if key != 'date'}
    coverage_totals = {p: [0, 0] for p in products}  # [present, expected]

    for r in rows:
        parts = []
        for name, key in cols:
            if name == 'Date':
                parts.append(f"{r[key]:<{widths[name]}}")
            elif 'cov%' in name:
                parts.append(f"{r[key]:>{widths[name] - 1}.1f}%")
            else:
                parts.append(f"{r[key]:>{widths[name]}}")
        print(" ".join(parts))
        for _, key in cols:
            if key == 'date':
                continue
            if 'cov%' in key:
                continue  # don't sum %s
            totals[key] = totals.get(key, 0) + r[key]
        for p in products:
            coverage_totals[p][0] += r[f'{p}_on_grid']
            coverage_totals[p][1] += r[f'{p}_expected']

    print("-" * len(header))
    parts = [f"{'TOTAL':<{widths['Date']}}"]
    for name, key in cols:
        if key == 'date':
            continue
        if 'cov%' in name:
            for p in products:
                short = 'REFL' if p == 'opera_reflectivity' else 'RAIN'
                if f"{short} cov%" == name:
                    pres, exp = coverage_totals[p]
                    pct = (pres / exp * 100) if exp else 0.0
                    parts.append(f"{pct:>{widths[name] - 1}.1f}%")
        else:
            parts.append(f"{totals[key]:>{widths[name]}}")
    print(" ".join(parts))
    print(f"\n{len(rows)} dates analysed")


def save_csv(rows: list[dict], products: list[str], output_path: str) -> None:
    if not rows:
        print("No data to save.")
        return

    fieldnames = ['date']
    for p in products:
        fieldnames.extend([
            f'{p}_files', f'{p}_on_grid', f'{p}_off_grid',
            f'{p}_expected', f'{p}_coverage_pct',
        ])
    fieldnames.extend([
        'complete_intersection', 'expected_intersection', 'coverage_pct',
    ])

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def save_missing_json(rows: list[dict], detail: dict,
                      products: list[str],
                      minute_filters: dict[str, set[int]],
                      output_path: str) -> None:
    payload = {
        'config': {
            'products':       products,
            'minute_filters': {p: sorted(minute_filters[p]) for p in products},
        },
        'dates': {},
        'summary': {},
    }

    totals = {p: {'expected': 0, 'present': 0, 'missing': 0, 'off_grid': 0}
              for p in products}

    for r in rows:
        date_str = r['date']
        date_block: dict = {}
        for p in products:
            d = detail[date_str][p]
            date_block[p] = {
                'expected':            r[f'{p}_expected'],
                'present_on_grid':     r[f'{p}_on_grid'],
                'off_grid':            r[f'{p}_off_grid'],
                'missing':             r[f'{p}_expected'] - r[f'{p}_on_grid'],
                'coverage_pct':        r[f'{p}_coverage_pct'],
                'missing_times':       d['missing'],
                'off_grid_times':      d['off_grid'],
            }
            totals[p]['expected'] += r[f'{p}_expected']
            totals[p]['present']  += r[f'{p}_on_grid']
            totals[p]['missing']  += (r[f'{p}_expected'] - r[f'{p}_on_grid'])
            totals[p]['off_grid'] += r[f'{p}_off_grid']
        payload['dates'][date_str] = date_block

    summary = {'n_dates': len(rows), 'per_product': {}}
    for p in products:
        t = totals[p]
        coverage = (t['present'] / t['expected'] * 100) if t['expected'] else 0
        summary['per_product'][p] = {
            **t,
            'overall_coverage_pct': round(coverage, 1),
        }
    payload['summary'] = summary

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Saved missing timesteps: {output_path}")
    for p in products:
        t = summary['per_product'][p]
        print(f"  {p:22s}: {t['present']}/{t['expected']} present "
              f"({t['overall_coverage_pct']}%), {t['missing']} missing")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise downloaded OPERA radar files by date and "
                    "report missing timesteps vs the configured cadence."
    )
    parser.add_argument(
        '--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
        help=f'Local OPERA root (default: {DEFAULT_DATA_DIR})',
    )
    parser.add_argument(
        '--products', type=str, nargs='+',
        default=list(PRODUCTS.keys()),
        choices=list(PRODUCTS.keys()),
        help='Products to include (default: both)',
    )
    parser.add_argument(
        '--timesteps', type=str, nargs='+', default=None,
        help='Override the per-product minute filter (one filter applied to '
             "all chosen products; e.g. 00 15 30 45). 'all' uses the native "
             'product cadence (15 min for both reflectivity and '
             'rainfall_rate). Default: per-product filter from '
             'timestep_config.json.',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default=str(PROJECT_ROOT / 'opera_summary.csv'),
        help='Output CSV filename (default: opera_summary.csv)',
    )
    parser.add_argument(
        '--missing', '-m', type=str,
        default=str(PROJECT_ROOT / 'opera_missing_timesteps.json'),
        help='Output JSON with missing timesteps '
             '(default: opera_missing_timesteps.json)',
    )

    args = parser.parse_args()

    minute_filters: dict[str, set[int]] = {}
    step_src_msgs: dict[str, str] = {}
    for p in args.products:
        if args.timesteps is not None and args.timesteps != ['all']:
            minute_filters[p] = {int(m) for m in args.timesteps}
            step_src_msgs[p] = "CLI override"
        elif args.timesteps == ['all']:
            cadence = PRODUCTS[p]['default_cadence']
            minute_filters[p] = set(range(0, 60, cadence))
            step_src_msgs[p] = f"all native ({cadence}-min)"
        else:
            flt, step, _ = load_minute_filter(p)
            minute_filters[p] = flt
            step_src_msgs[p] = (f"timestep_config.json "
                                f"(step={step} min)")

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("OPERA Cache Summary")
    print("=" * 70)
    print(f"Data dir       : {data_dir}")
    print(f"Products       : {args.products}")
    for p in args.products:
        print(f"  {p:22s}: filter={sorted(minute_filters[p])} ({step_src_msgs[p]})")
    print()

    rows, detail = summarize(data_dir, args.products, minute_filters)
    print_table(rows, args.products)
    save_csv(rows, args.products, args.output)
    save_missing_json(rows, detail, args.products,
                      minute_filters, args.missing)

    return 0


if __name__ == "__main__":
    sys.exit(main())
