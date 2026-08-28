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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Matches reproject.py's default. Scanning is I/O-bound — every array is
# read in full to test a single condition — so the useful worker count
# tracks disk throughput rather than core count.
DEFAULT_WORKERS = 6


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


def scan_day(args: tuple[str, str, str]) -> tuple[str, dict[str, bool]]:
    """Scan one (product, date) directory. Worker for the process pool.

    Must live at module level so ProcessPoolExecutor can pickle it — the
    same constraint reproject.py's workers are written under. Takes and
    returns plain types for the same reason.

    Args:
        args: (day_dir, date_str, product) — paths as str, not Path.

    Returns:
        (date_str, {HHMM: active_bool})
    """
    day_dir_str, date_str, product = args
    day_dir = Path(day_dir_str)
    expected_date_compact = date_str.replace('-', '')
    result: dict[str, bool] = {}

    for filename in sorted(os.listdir(day_dir)):
        fm = FILE_PATTERN.match(filename)
        if not fm or fm.group('product') != product:
            continue
        if fm.group('date') != expected_date_compact:
            # Stray file from another date in this subdir — skip.
            continue
        result[fm.group('hhmm')] = has_activity(day_dir / filename)

    return date_str, result


def scan_product(data_dir: Path, product: str,
                 workers: int = DEFAULT_WORKERS) -> dict[str, dict[str, bool]]:
    """
    Scan one sub-product's date directories, one worker per date.

    Returns: {date_str: {HHMM: active_bool}}, where `active_bool` is True
    if the .npy file exists AND has at least one non-zero pixel.

    Parallel per day rather than per file: the unit of work is then large
    enough to dwarf the process-dispatch overhead, while still giving
    hundreds of independent tasks. Every array is read in full only to
    test `np.any(data != 0)`, so across three products and a full archive
    this is hundreds of gigabytes of reads — serial scanning is the
    bottleneck, not the arithmetic.
    """
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    prod_root = data_dir / PRODUCTS[product]["subdir"]
    if not prod_root.is_dir():
        return out

    tasks: list[tuple[str, str, str]] = []
    for entry in sorted(os.listdir(prod_root)):
        m = DATE_DIR_PATTERN.match(entry)
        if not m:
            continue
        day_dir = prod_root / entry
        if not day_dir.is_dir():
            continue
        tasks.append((str(day_dir), m.group(1), product))

    if not tasks:
        return dict(out)

    # One worker is pointless overhead; fall back to a plain loop so the
    # single-threaded path stays debuggable.
    n_workers = max(1, min(workers, len(tasks)))
    if n_workers == 1:
        for task in tasks:
            date_str, result = scan_day(task)
            if result:
                out[date_str] = result
        return dict(out)

    print(f"  scanning {product}: {len(tasks)} date(s), {n_workers} workers")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(scan_day, t): t[1] for t in tasks}
        done = 0
        for future in as_completed(futures):
            date_str = futures[future]
            try:
                date_str, result = future.result()
                if result:
                    out[date_str] = result
            except Exception as exc:                      # noqa: BLE001
                print(f"  WARNING: {product} {date_str} failed: {exc}",
                      file=sys.stderr)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"    [{done}/{len(tasks)}]", flush=True)

    return dict(out)


# =============================================================================
# Summarise
# =============================================================================

def expected_hhmm_set(minute_filter: set[int]) -> set[str]:
    """All HHMM the configured filter expects across one full day."""
    return {f"{h:02d}{m:02d}"
            for h in range(24)
            for m in minute_filter}


def date_range(start: str | None, end: str | None) -> list[str]:
    """Every YYYY-MM-DD from start to end inclusive, or [] if unbounded.

    Declaring the expected range is what turns "no files for this date"
    into a reported gap. Inferring it from the files found cannot see a
    date absent from disk entirely — it is simply not described, and so
    never appears as missing coverage.
    """
    if not start or not end:
        return []
    from datetime import datetime, timedelta
    try:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        print(f"ERROR: --start/--end must be YYYY-MM-DD "
              f"(got {start!r}, {end!r})", file=sys.stderr)
        sys.exit(2)
    if d1 < d0:
        print(f"ERROR: --end {end} precedes --start {start}", file=sys.stderr)
        sys.exit(2)
    out, cur = [], d0
    while cur <= d1:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def summarize(activity_by_product: dict[str, dict[str, dict[str, bool]]],
              minute_filter: set[int],
              start: str | None = None,
              end: str | None = None):
    """
    Build per-date stats.

    `activity_by_product[p][date][HHMM] = bool` from scan_product().
    `start`/`end` (YYYY-MM-DD, inclusive) declare the range the cache is
    EXPECTED to cover; dates in range with no maps are reported as fully
    missing rather than omitted.

    Returns (rows, detail) where:
        rows: list of per-date dicts ready for CSV / table
        detail: {date: {product: {'on_grid_active': set, 'off_grid_active': set}}}
    """
    expected_set = expected_hhmm_set(minute_filter)

    all_dates: set[str] = set()
    for p in PRODUCTS:
        all_dates.update(activity_by_product[p].keys())

    expected_dates = date_range(start, end)
    if expected_dates:
        empty = [d for d in expected_dates if d not in all_dates]
        outside = sorted(all_dates - set(expected_dates))
        all_dates.update(expected_dates)
        print(f"Expected range : {expected_dates[0]} .. {expected_dates[-1]} "
              f"({len(expected_dates)} dates)")
        if empty:
            print(f"  {len(empty)} date(s) in range have NO maps at all "
                  f"— reported as fully missing")
        if outside:
            print(f"  WARNING: {len(outside)} date(s) on disk fall OUTSIDE "
                  f"the range and are still reported: {outside[:5]}"
                  + (" ..." if len(outside) > 5 else ""))

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

def render_chart(rows, output_path):
    """Monthly coverage chart from the per-date rows."""
    # One implementation, in the canonical summariser, so the
    # three product charts cannot drift apart.
    sys.path.insert(0, str(PROJECT_ROOT / 'our_data' / 'satellite_data'))
    from summarize_mtg import plot_monthly
    found = {r['date']: r['complete_union'] for r in rows}
    expected = {r['date']: r['expected_grid'] for r in rows}
    span = f"{min(found)} .. {max(found)}" if found else ""
    return plot_monthly(found, output_path,
                        title=f"LINET coverage  {span}",
                        per_date_expected=expected,
                        ylabel="active timesteps")


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
    parser.add_argument(
        '--chart', type=str, nargs='?', const='lightning_coverage.png', default=None,
        help="Render a monthly coverage chart (faded bars + line through "
             "the bar tops, with the cadence expectation as a dashed "
             "reference). Optional PNG path; defaults to "
             "lightning_coverage.png at the project root.",
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help="First date the cache is EXPECTED to cover (YYYY-MM-DD). "
             "Dates in range with no maps are reported as fully missing "
             "instead of being omitted. Without this the range is inferred "
             "from the files found, so an entirely absent date is invisible.",
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help="Last expected date, inclusive (YYYY-MM-DD). See --start.",
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=DEFAULT_WORKERS,
        help=f'Parallel workers for the per-date scan (default: '
             f'{DEFAULT_WORKERS}, matching reproject.py). Every .npy is '
             f'read in full to test for a non-zero pixel, so this is '
             f'disk-bound; 1 forces the serial path.',
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
        p: scan_product(data_dir, p, workers=args.workers) for p in PRODUCTS
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

    rows, _detail = summarize(activity_by_product, minute_filter,
                              start=args.start, end=args.end)
    print_table(rows)

    save_summary_csv(rows, Path(args.output))
    if args.chart:
        render_chart(rows, args.chart)
    save_active_csv(activity_by_product, minute_filter, Path(args.active))

    return 0


if __name__ == "__main__":
    sys.exit(main())
