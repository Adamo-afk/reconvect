"""
summarize_lightning_data.py — Coverage / activity report for the lightning
.npy cache produced by read_kml_version2.py.

Scans `our_data/lightning_data/{density,current,occurrence}/nc4_*-Romania_*/`,
parses the (date, HHMM) from each `lightning_*_YYYYMMDD_HHMM.npy` filename,
loads the array to test for any non-zero pixel, and produces:

    1. `lightning_summary.csv` (project root by default)
       Per-date per-sub-product table:
          {p}_files / {p}_on_grid / {p}_off_grid /
          {p}_expected / {p}_coverage_pct
       Plus unified columns (any-of-three semantics):
          complete_union / expected_grid / coverage_pct

    2. `lightning_active_steps.csv` (project root by default)
       One row per (date, HH:MM) pair where at least one sub-product has
       activity, with columns:
          date,time_utc,density,current,occurrence
       Flags are 1 if the sub-product has any non-zero pixel at that step,
       0 otherwise. This is the artifact `intersect_product_coverage.py`
       consumes via `--active lightning=...`, replacing the missing-JSON
       gate for lightning.

The summary mirrors `our_data/opera_data/summarize_opera_data.py` so the
two products feel consistent in `intersect_product_coverage.py`.

Lightning does not emit a missing-timesteps JSON: the active CSV is the
sole presence gate for lightning in the cross-product intersection.

Usage:
    python our_data/lightning_data/summarize_lightning_data.py
    python our_data/lightning_data/summarize_lightning_data.py --data_dir other/path
    python our_data/lightning_data/summarize_lightning_data.py \\
        --output lightning_summary.csv --active lightning_active_steps.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"

DEFAULT_DATA_DIR = Path(__file__).resolve().parent  # our_data/lightning_data
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "lightning_summary.csv"
DEFAULT_ACTIVE_CSV = PROJECT_ROOT / "lightning_active_steps.csv"

# Lightning sub-products + their on-disk subdir + .npy filename prefix.
PRODUCTS: dict[str, dict[str, str]] = {
    "density":    {"subdir": "density",    "prefix": "lightning_density"},
    "current":    {"subdir": "current",    "prefix": "lightning_current"},
    "occurrence": {"subdir": "occurrence", "prefix": "lightning_occurrence"},
}

# `nc4_{YYYY-MM-DD}-Romania_{product}` date subdir, written by
# read_kml_version2.py. The trailing `-Romania_{product}` is unique per
# sub-product but the date is what matters here.
DATE_DIR_PATTERN = re.compile(r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_')

# `lightning_{product}_{YYYYMMDD}_{HHMM}.npy`
FILE_PATTERN = re.compile(
    r'^lightning_(?P<product>density|current|occurrence)_'
    r'(?P<date>\d{8})_(?P<hhmm>\d{4})\.npy$'
)


# =============================================================================
# Cadence config
# =============================================================================

def load_minute_filter():
    """
    Returns (minute_filter set, step_minutes, native_cadence_minutes_or_None).

    Reads `products.lightning` from timestep_config.json. Refuses to run if
    the lightning filter is missing — the whole point of this script is to
    measure coverage against the configured grid.
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
    prod_cfg = cfg["products"].get("lightning")
    if prod_cfg is None or prod_cfg.get("filter") is None:
        print(
            "ERROR: product 'lightning' has no minute filter in "
            f"{TIMESTEP_CONFIG_PATH}. Add it to product_cadences.config and "
            "re-run validate_timestep.py.",
            file=sys.stderr,
        )
        sys.exit(2)
    return (
        set(prod_cfg["filter"]),
        int(cfg["step_minutes"]),
        prod_cfg.get("cadence_minutes"),
    )


# =============================================================================
# Scanning
# =============================================================================

def has_activity(npy_path: Path) -> bool:
    """Return True iff the .npy contains at least one non-zero pixel."""
    try:
        data = np.load(npy_path)
    except Exception as e:
        print(f"  WARNING: could not read {npy_path}: {e}", file=sys.stderr)
        return False
    return bool(np.any(data != 0))


def scan_product(data_dir: Path, product: str) -> dict[str, dict[str, bool]]:
    """
    Scan one sub-product's date directories.

    Returns: {date_str: {HHMM: active_bool}}, where `active_bool` is True
    if the .npy file exists AND has at least one non-zero pixel.
    """
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    prod_root = data_dir / PRODUCTS[product]["subdir"]
    if not prod_root.is_dir():
        return out

    for entry in sorted(os.listdir(prod_root)):
        m = DATE_DIR_PATTERN.match(entry)
        if not m:
            continue
        date_str = m.group(1)
        day_dir = prod_root / entry
        if not day_dir.is_dir():
            continue

        expected_date_compact = date_str.replace('-', '')
        for filename in sorted(os.listdir(day_dir)):
            fm = FILE_PATTERN.match(filename)
            if not fm or fm.group('product') != product:
                continue
            if fm.group('date') != expected_date_compact:
                # Stray file from another date in this subdir — skip.
                continue
            hhmm = fm.group('hhmm')
            out[date_str][hhmm] = has_activity(day_dir / filename)

    return dict(out)


# =============================================================================
# Summarise
# =============================================================================

def expected_hhmm_set(minute_filter: set[int]) -> set[str]:
    """All HHMM the configured filter expects across one full day."""
    return {f"{h:02d}{m:02d}"
            for h in range(24)
            for m in minute_filter}


def summarize(activity_by_product: dict[str, dict[str, dict[str, bool]]],
              minute_filter: set[int]):
    """
    Build per-date stats.

    `activity_by_product[p][date][HHMM] = bool` from scan_product().

    Returns (rows, detail) where:
        rows: list of per-date dicts ready for CSV / table
        detail: {date: {product: {'on_grid_active': set, 'off_grid_active': set}}}
    """
    expected_set = expected_hhmm_set(minute_filter)

    all_dates: set[str] = set()
    for p in PRODUCTS:
        all_dates.update(activity_by_product[p].keys())

    rows: list[dict] = []
    detail: dict = {}
    for date_str in sorted(all_dates):
        row = {'date': date_str}
        date_detail: dict = {}

        # Per-sub-product counts
        on_grid_active_by_product: dict[str, set[str]] = {}
        on_grid_present_by_product: dict[str, set[str]] = {}
        for p in PRODUCTS:
            per_day = activity_by_product[p].get(date_str, {})
            files = set(per_day.keys())
            on_grid = files & expected_set
            off_grid = files - expected_set
            on_grid_active = {hh for hh in on_grid if per_day[hh]}
            off_grid_active = {hh for hh in off_grid if per_day[hh]}
            coverage = (len(on_grid_active) / len(expected_set) * 100
                        if expected_set else 0.0)
            row[f'{p}_files']         = len(files)
            row[f'{p}_on_grid']       = len(on_grid_active)
            row[f'{p}_off_grid']      = len(off_grid_active)
            row[f'{p}_expected']      = len(expected_set)
            row[f'{p}_coverage_pct']  = round(coverage, 1)
            on_grid_active_by_product[p] = on_grid_active
            on_grid_present_by_product[p] = on_grid
            date_detail[p] = {
                'on_grid_active':  sorted(on_grid_active),
                'off_grid_active': sorted(off_grid_active),
            }

        # Unified coverage = UNION of per-product on-grid actives
        # (any-of-three: a step counts if any sub-product fired).
        union_active = set().union(*on_grid_active_by_product.values())
        unified = (len(union_active) / len(expected_set) * 100
                   if expected_set else 0.0)
        row['complete_union']    = len(union_active)
        row['expected_grid']     = len(expected_set)
        row['coverage_pct']      = round(unified, 1)

        rows.append(row)
        detail[date_str] = date_detail

    return rows, detail


# =============================================================================
# Output
# =============================================================================

def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No data found.")
        return

    cols: list[tuple[str, str]] = [('Date', 'date')]
    for p in PRODUCTS:
        short = p[:3].upper()
        cols.append((f"{short} ok",  f"{p}_on_grid"))
        cols.append((f"{short} exp", f"{p}_expected"))
        cols.append((f"{short} cov%", f"{p}_coverage_pct"))
    cols.append(('Any cov%', 'coverage_pct'))

    widths = {name: max(len(name), 9) for name, _ in cols}
    widths['Date'] = 12

    header = " ".join(
        f"{name:>{widths[name]}}" if name != 'Date'
        else f"{name:<{widths[name]}}"
        for name, _ in cols
    )
    print(header)
    print("-" * len(header))

    coverage_totals = {p: [0, 0] for p in PRODUCTS}  # [present, expected]
    union_total = [0, 0]
    int_totals = {key: 0 for _, key in cols if key not in ('date',)
                  and 'cov%' not in key}

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
            if key == 'date' or 'cov' in key:
                continue
            int_totals[key] = int_totals.get(key, 0) + r[key]
        for p in PRODUCTS:
            coverage_totals[p][0] += r[f'{p}_on_grid']
            coverage_totals[p][1] += r[f'{p}_expected']
        union_total[0] += r['complete_union']
        union_total[1] += r['expected_grid']

    print("-" * len(header))
    parts = [f"{'TOTAL':<{widths['Date']}}"]
    for name, key in cols:
        if key == 'date':
            continue
        if 'cov%' in name:
            if name == 'Any cov%':
                pres, exp = union_total
                pct = (pres / exp * 100) if exp else 0.0
            else:
                # match name back to a product
                short = name.split(' ')[0]
                target = short.lower()
                pres, exp = coverage_totals[
                    next(p for p in PRODUCTS if p.startswith(target))
                ]
                pct = (pres / exp * 100) if exp else 0.0
            parts.append(f"{pct:>{widths[name] - 1}.1f}%")
        else:
            parts.append(f"{int_totals.get(key, 0):>{widths[name]}}")
    print(" ".join(parts))
    print(f"\n{len(rows)} dates analysed")


def save_summary_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        print("No data to save.")
        return

    fieldnames = ['date']
    for p in PRODUCTS:
        fieldnames.extend([
            f'{p}_files', f'{p}_on_grid', f'{p}_off_grid',
            f'{p}_expected', f'{p}_coverage_pct',
        ])
    fieldnames.extend(['complete_union', 'expected_grid', 'coverage_pct'])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def save_active_csv(activity_by_product: dict[str, dict[str, dict[str, bool]]],
                    minute_filter: set[int],
                    output_path: Path) -> None:
    """
    Emit `lightning_active_steps.csv`: one row per (date, HH:MM) pair where
    at least one sub-product is active, restricted to HHMM on the
    configured lightning minute filter.

    Restricting to the on-grid set is what makes this CSV a 1:1 match for
    intersect's snap output — every (date, snapped_hhmm) the master grid
    can produce is either in this CSV (kept) or not (dropped).
    """
    expected_set = expected_hhmm_set(minute_filter)

    # Gather union of active (date, HHMM) on-grid.
    pairs: dict[tuple[str, str], dict[str, int]] = {}
    for product in PRODUCTS:
        for date_str, per_day in activity_by_product[product].items():
            for hhmm, active in per_day.items():
                if hhmm not in expected_set:
                    continue
                if not active:
                    continue
                key = (date_str, hhmm)
                flags = pairs.setdefault(
                    key, {p: 0 for p in PRODUCTS}
                )
                flags[product] = 1

    sorted_pairs = sorted(pairs.items())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'time_utc'] + list(PRODUCTS.keys()))
        for (date_str, hhmm), flags in sorted_pairs:
            time_formatted = f"{hhmm[:2]}:{hhmm[2:]}"
            writer.writerow(
                [date_str, time_formatted] + [flags[p] for p in PRODUCTS]
            )

    print(f"Saved active steps: {output_path}  ({len(sorted_pairs)} rows)")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise lightning .npy cache by date + emit the "
                    "per-step active CSV consumed by "
                    "intersect_product_coverage.py."
    )
    parser.add_argument(
        '--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
        help=f'Lightning .npy root (default: {DEFAULT_DATA_DIR})',
    )
    parser.add_argument(
        '--output', '-o', type=str, default=str(DEFAULT_OUTPUT_CSV),
        help=f'Output summary CSV (default: {DEFAULT_OUTPUT_CSV})',
    )
    parser.add_argument(
        '--active', '-a', type=str, default=str(DEFAULT_ACTIVE_CSV),
        help=f'Output active-steps CSV consumed by intersect_product_coverage.py '
             f'(default: {DEFAULT_ACTIVE_CSV})',
    )

    args = parser.parse_args()

    minute_filter, step_minutes, cadence_minutes = load_minute_filter()
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("Lightning Cache Summary")
    print("=" * 70)
    print(f"Data dir        : {data_dir}")
    print(f"step_minutes    : {step_minutes}")
    print(f"native cadence  : {cadence_minutes if cadence_minutes is not None else '(continuous)'}")
    print(f"minute filter   : {sorted(minute_filter)}")
    if cadence_minutes is not None and step_minutes != cadence_minutes:
        # Surface the read_kml_version2.py / filter mismatch: read_kml
        # writes at step_minutes spacing while the filter is snapped to
        # the native cadence. Coverage_pct will reflect that gap.
        print("  NOTE: read_kml_version2.py writes at step_minutes "
              "spacing; if the lightning filter is derived from a "
              "different native cadence, on-grid coverage will look low.")
    print()

    activity_by_product = {
        p: scan_product(data_dir, p) for p in PRODUCTS
    }
    for p in PRODUCTS:
        n_dates = len(activity_by_product[p])
        n_files = sum(len(v) for v in activity_by_product[p].values())
        n_active = sum(
            sum(1 for x in v.values() if x)
            for v in activity_by_product[p].values()
        )
        print(f"  {p:10s} : {n_dates} dates, {n_files} files, {n_active} active")
    print()

    rows, _detail = summarize(activity_by_product, minute_filter)
    print_table(rows)

    save_summary_csv(rows, Path(args.output))
    save_active_csv(activity_by_product, minute_filter, Path(args.active))

    return 0


if __name__ == "__main__":
    sys.exit(main())
