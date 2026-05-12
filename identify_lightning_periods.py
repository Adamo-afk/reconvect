"""
identify_lightning_periods.py — 3-stage lightning sample-selection cascade.

For each native-cadence lightning map produced by read_kml_version2.py:

  Stage 1 (aggregation): bin maps into windows of --aggregation_minutes width
                         (default 60). Width must be >= step_minutes from
                         timestep_config.json AND a multiple of it.
  Stage 2 (map filter):  drop bins whose lightning_fraction < threshold
                         (default 1e-4).
  Stage 3 (day filter):  drop days where fewer than --day_min_valid_fraction
                         (default 0.20) of the bins survived stage 2.
  Stage 4 (window filter): tile the date range into non-overlapping windows
                         of --window_days (default 10). Keep an entire window
                         iff it contains >= --window_min_valid_days (default 2)
                         days that survived stage 3.

For each (date, bin) tuple that survives all four stages, also compute
per-patch activity on the standard 6x3 grid (any non-zero pixel in the 256x256
patch). This file (`lightning_patches.csv`) drives the lightning-source mode
of `extract_patch_seq_for_datasets.py`.

Lightning-fraction definition: bin_aggregated_map computed as element-wise
logical-OR across all native maps in the bin (regardless of product, since
we only care whether activity exists). Then
`lightning_fraction = nonzero_pixels / total_pixels`.

Outputs (default our_data/lightning_periods/):
    lightning_periods_config.json   # All CLI parameters used (reproducibility)
    lightning_bins.csv              # Per-bin detail (one row per bin per date)
    lightning_periods.csv           # Per-day summary + valid_day + kept flags
    lightning_patches.csv           # Per-bin per-patch activity (Option B);
                                    # schema matches patch_index.csv so
                                    # extract_patch_seq_for_datasets.py can
                                    # consume it directly in --source lightning.

Usage:
    python validate_timestep.py --step_minutes 15      # one-time
    python identify_lightning_periods.py               # all defaults
    python identify_lightning_periods.py --aggregation_minutes 30
    python identify_lightning_periods.py --map_fraction_threshold 5e-5 \\
        --day_min_valid_fraction 0.30
    python identify_lightning_periods.py --start_date 2026-03-01 \\
        --end_date 2026-04-30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


# =============================================================================
# Project paths + grid constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "lightning_periods"

# Romania grid + 6x3 patch layout (must match identify_patches.py)
GRID_WIDTH = 1536
GRID_HEIGHT = 768
PATCH_SIZE = 256
N_COLS = 6
N_ROWS = 3
N_PATCHES = N_COLS * N_ROWS  # 18
TOTAL_PIXELS = GRID_WIDTH * GRID_HEIGHT  # 1_179_648


# =============================================================================
# Defaults from the proposal diagram
# =============================================================================

DEFAULT_PRODUCT = "occurrence"
DEFAULT_AGGREGATION_MINUTES = 60
DEFAULT_MAP_FRACTION_THRESHOLD = 1e-4
DEFAULT_DAY_MIN_VALID_FRACTION = 0.20
DEFAULT_WINDOW_DAYS = 10
DEFAULT_WINDOW_MIN_VALID_DAYS = 2


# Lightning NetCDF filename: lightning_<product>_<YYYYMMDD>_<HHMM>.nc
FILENAME_PATTERN = re.compile(
    r'^lightning_(?P<product>density|current|occurrence)_'
    r'(?P<date>\d{8})_(?P<hhmm>\d{4})\.nc$'
)


# =============================================================================
# Config loading
# =============================================================================

def load_step_minutes() -> int:
    if not TIMESTEP_CONFIG_PATH.exists():
        print(
            f"ERROR: timestep config not found at {TIMESTEP_CONFIG_PATH}.\n"
            f"Run from the project root:\n"
            f"    python validate_timestep.py --step_minutes <N>",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = json.loads(TIMESTEP_CONFIG_PATH.read_text())
    return int(cfg["step_minutes"])


def validate_aggregation(aggregation_minutes: int, step_minutes: int) -> None:
    if aggregation_minutes <= 0:
        sys.exit(f"ERROR: --aggregation_minutes must be positive, got {aggregation_minutes}")
    if aggregation_minutes < step_minutes:
        sys.exit(
            f"ERROR: --aggregation_minutes ({aggregation_minutes}) cannot be "
            f"smaller than step_minutes ({step_minutes}) from timestep_config.json. "
            f"You cannot aggregate at a finer cadence than the underlying data."
        )
    if aggregation_minutes % step_minutes != 0:
        sys.exit(
            f"ERROR: --aggregation_minutes ({aggregation_minutes}) must be a "
            f"multiple of step_minutes ({step_minutes})."
        )
    if 1440 % aggregation_minutes != 0:
        sys.exit(
            f"ERROR: --aggregation_minutes ({aggregation_minutes}) must divide "
            f"1440 (24h) evenly."
        )


# =============================================================================
# Data loading
# =============================================================================

def discover_dates(lightning_dir: str, product: str,
                   start_date=None, end_date=None) -> list[str]:
    product_dir = os.path.join(lightning_dir, product)
    if not os.path.isdir(product_dir):
        sys.exit(f"ERROR: lightning product dir not found: {product_dir}")

    dir_re = re.compile(r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_')
    dates = []
    for subdir in sorted(os.listdir(product_dir)):
        m = dir_re.match(subdir)
        if not m:
            continue
        date_str = m.group(1)
        try:
            d = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            continue
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        dates.append(date_str)
    return dates


def load_date_maps(lightning_dir: str, product: str,
                   date_str: str) -> list[tuple[str, np.ndarray]]:
    """Load all native maps for a given date. Returns sorted list of (HHMM, 2D array)."""
    date_dir = os.path.join(
        lightning_dir, product, f"nc4_{date_str}-Romania_{product}"
    )
    if not os.path.isdir(date_dir):
        return []

    expected_date = date_str.replace('-', '')
    maps = []
    for filename in sorted(os.listdir(date_dir)):
        m = FILENAME_PATTERN.match(filename)
        if not m or m.group('product') != product:
            continue
        if m.group('date') != expected_date:
            continue
        hhmm = m.group('hhmm')
        try:
            with Dataset(os.path.join(date_dir, filename), 'r') as ds:
                data = ds.variables['datamap'][:]
                if hasattr(data, 'filled'):
                    data = data.filled(0)
                data = np.asarray(data)
                if data.ndim == 3:
                    data = data[0]
                maps.append((hhmm, data))
        except Exception as e:
            print(f"  WARN: could not read {filename}: {e}", file=sys.stderr)

    maps.sort(key=lambda kv: kv[0])
    return maps


# =============================================================================
# Stage 1 + 2: bin and filter per date
# =============================================================================

def process_date(maps: list[tuple[str, np.ndarray]],
                 aggregation_minutes: int,
                 step_minutes: int,
                 map_fraction_threshold: float) -> list[dict]:
    """
    Bin native maps into aggregation windows and compute per-bin aggregate.

    The native lightning convention (from read_kml_version2.py) is that a
    file with HHMM = T represents the interval [T - step_minutes, T) — i.e.
    the timestamp marks the END of the accumulation window. We assign each
    file to the bin that covers its end-timestamp T.

    Returns a list of dicts with bin_idx, agg_map (or None when invalid),
    fraction, valid_bin, n_maps_in_bin, n_nonzero.
    """
    bins_per_day = 1440 // aggregation_minutes

    bin_buckets: list[list[np.ndarray]] = [[] for _ in range(bins_per_day)]
    for hhmm, data in maps:
        t_min = int(hhmm[:2]) * 60 + int(hhmm[2:])
        # Map at HHMM=T represents [T - step, T). Place it in the bin that
        # contains T - 1 (so the map at exactly the bin boundary T = k*agg
        # lands in bin k-1).
        bin_idx = (t_min - 1) // aggregation_minutes
        if 0 <= bin_idx < bins_per_day:
            bin_buckets[bin_idx].append(data)

    bins = []
    for bin_idx, bucket in enumerate(bin_buckets):
        if bucket:
            stack = np.stack(bucket, axis=0)
            agg = (stack != 0).any(axis=0).astype(np.uint8)
        else:
            agg = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

        n_nonzero = int(agg.sum())
        fraction = n_nonzero / TOTAL_PIXELS if TOTAL_PIXELS else 0.0
        valid_bin = fraction >= map_fraction_threshold

        bins.append({
            'bin_idx':       bin_idx,
            'bin_start_min': bin_idx * aggregation_minutes,
            'agg_map':       agg if valid_bin else None,  # keep map only if needed
            'fraction':      fraction,
            'n_nonzero':     n_nonzero,
            'n_maps_in_bin': len(bucket),
            'valid_bin':     valid_bin,
        })

    return bins


# =============================================================================
# Stage 3: per-day filter
# =============================================================================

def apply_day_filter(bins_per_date: dict, bins_per_day: int,
                     day_min_valid_fraction: float) -> dict:
    """Returns {date: (valid_day_bool, n_valid_bins, valid_fraction)}."""
    out = {}
    for date_str, bins in bins_per_date.items():
        n_valid = sum(1 for b in bins if b['valid_bin'])
        frac = n_valid / bins_per_day if bins_per_day else 0.0
        out[date_str] = (frac >= day_min_valid_fraction, n_valid, frac)
    return out


# =============================================================================
# Stage 4: per-window filter (non-overlapping tiles)
# =============================================================================

def apply_window_filter(valid_day_by_date: dict,
                        window_days: int,
                        window_min_valid_days: int) -> set:
    """
    Tile the *contiguous* date range (filling in any missing days as invalid)
    into non-overlapping windows of `window_days` and keep all days inside
    each window that has >= `window_min_valid_days` valid days.
    """
    if not valid_day_by_date:
        return set()

    sorted_dates = sorted(valid_day_by_date.keys())
    start = datetime.strptime(sorted_dates[0], '%Y-%m-%d').date()
    end = datetime.strptime(sorted_dates[-1], '%Y-%m-%d').date()

    all_dates: list[str] = []
    d = start
    while d <= end:
        all_dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)

    kept = set()
    for i in range(0, len(all_dates), window_days):
        window = all_dates[i:i + window_days]
        n_valid = sum(
            1 for x in window
            if valid_day_by_date.get(x, (False, 0, 0))[0]
        )
        if n_valid >= window_min_valid_days:
            kept.update(window)

    return kept


# =============================================================================
# Per-patch activity (Option B)
# =============================================================================

def patches_with_activity(agg_map: np.ndarray | None) -> list[int]:
    """Return 1-indexed patch IDs that contain at least one non-zero pixel."""
    if agg_map is None:
        return []
    active = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            chunk = agg_map[
                r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                c * PATCH_SIZE:(c + 1) * PATCH_SIZE,
            ]
            if chunk.any():
                active.append(r * N_COLS + c + 1)
    return active


# =============================================================================
# Output
# =============================================================================

def write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2))


def write_bins_csv(path: Path, bins_per_date: dict,
                   aggregation_minutes: int) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'date', 'bin_idx', 'bin_start_utc', 'bin_end_utc',
            'n_maps_in_bin', 'n_nonzero_pixels',
            'lightning_fraction', 'valid_bin',
        ])
        for date_str in sorted(bins_per_date.keys()):
            for b in bins_per_date[date_str]:
                start_min = b['bin_start_min']
                end_min = start_min + aggregation_minutes
                w.writerow([
                    date_str, b['bin_idx'],
                    f"{start_min // 60:02d}:{start_min % 60:02d}",
                    f"{(end_min // 60) % 24:02d}:{end_min % 60:02d}",
                    b['n_maps_in_bin'], b['n_nonzero'],
                    f"{b['fraction']:.6e}",
                    int(b['valid_bin']),
                ])


def write_periods_csv(path: Path, bins_per_date: dict,
                      valid_day_by_date: dict,
                      kept_dates: set,
                      bins_per_day: int) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'date', 'n_bins_total', 'n_bins_valid',
            'day_valid_fraction', 'valid_day',
            'window_kept', 'kept',
        ])
        for date_str in sorted(bins_per_date.keys()):
            valid_day, n_valid, frac = valid_day_by_date[date_str]
            window_kept = date_str in kept_dates
            kept = window_kept and valid_day
            w.writerow([
                date_str, bins_per_day, n_valid, f"{frac:.4f}",
                int(valid_day), int(window_kept), int(kept),
            ])


def write_patches_csv(path: Path, bins_per_date: dict,
                      valid_day_by_date: dict,
                      kept_dates: set) -> int:
    """Schema matches patch_index.csv so extract_patch_seq can drop straight in."""
    n_rows = 0
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['date', 'time_utc', 'iso_timestamp']
        header += [f'patch_{p}' for p in range(1, N_PATCHES + 1)]
        w.writerow(header)
        for date_str in sorted(bins_per_date.keys()):
            if date_str not in kept_dates:
                continue
            if not valid_day_by_date[date_str][0]:
                continue
            for b in bins_per_date[date_str]:
                if not b['valid_bin']:
                    continue
                active = patches_with_activity(b['agg_map'])
                if not active:
                    continue
                start_min = b['bin_start_min']
                time_str = f"{start_min // 60:02d}:{start_min % 60:02d}"
                iso = f"{date_str}T{time_str}:00"
                row = [date_str, time_str, iso]
                row += [1 if p in active else 0 for p in range(1, N_PATCHES + 1)]
                w.writerow(row)
                n_rows += 1
    return n_rows


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="3-stage lightning sample-selection cascade for the "
                    "lightning training pipeline."
    )
    parser.add_argument('--data_root', type=str, default=str(DEFAULT_DATA_ROOT),
                        help=f'Root data directory (default: {DEFAULT_DATA_ROOT})')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f'Output dir (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--lightning_product', type=str, default=DEFAULT_PRODUCT,
                        choices=['density', 'current', 'occurrence'],
                        help=f'Lightning product to feed the cascade '
                             f'(default: {DEFAULT_PRODUCT})')
    parser.add_argument('--aggregation_minutes', type=int,
                        default=DEFAULT_AGGREGATION_MINUTES,
                        help=f'Stage 1 bin width in minutes (default: '
                             f'{DEFAULT_AGGREGATION_MINUTES}). Must be >= '
                             f'step_minutes from timestep_config.json and a '
                             f'multiple of it.')
    parser.add_argument('--map_fraction_threshold', type=float,
                        default=DEFAULT_MAP_FRACTION_THRESHOLD,
                        help=f'Stage 2 threshold on lightning_fraction per '
                             f'bin (default: {DEFAULT_MAP_FRACTION_THRESHOLD:g})')
    parser.add_argument('--day_min_valid_fraction', type=float,
                        default=DEFAULT_DAY_MIN_VALID_FRACTION,
                        help=f'Stage 3 minimum valid-bin fraction per day '
                             f'(default: {DEFAULT_DAY_MIN_VALID_FRACTION})')
    parser.add_argument('--window_days', type=int,
                        default=DEFAULT_WINDOW_DAYS,
                        help=f'Stage 4 non-overlapping window size in days '
                             f'(default: {DEFAULT_WINDOW_DAYS})')
    parser.add_argument('--window_min_valid_days', type=int,
                        default=DEFAULT_WINDOW_MIN_VALID_DAYS,
                        help=f'Stage 4 minimum valid days per window '
                             f'(default: {DEFAULT_WINDOW_MIN_VALID_DAYS})')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Restrict to dates >= YYYY-MM-DD (optional)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='Restrict to dates <= YYYY-MM-DD (optional)')

    args = parser.parse_args()

    step_minutes = load_step_minutes()
    validate_aggregation(args.aggregation_minutes, step_minutes)

    bins_per_day = 1440 // args.aggregation_minutes
    maps_per_bin = args.aggregation_minutes // step_minutes

    print("=" * 70)
    print("Lightning Periods Identification (3-stage cascade)")
    print("=" * 70)
    print(f"step_minutes              : {step_minutes} (timestep_config.json)")
    print(f"aggregation_minutes       : {args.aggregation_minutes}")
    print(f"bins per day              : {bins_per_day}")
    print(f"maps per bin              : {maps_per_bin}")
    print(f"product                   : {args.lightning_product}")
    print(f"map_fraction_threshold    : {args.map_fraction_threshold:g}")
    print(f"day_min_valid_fraction    : {args.day_min_valid_fraction}")
    print(f"window_days               : {args.window_days}")
    print(f"window_min_valid_days     : {args.window_min_valid_days}")
    print()

    lightning_dir = os.path.join(args.data_root, 'lightning_data')
    start_date = (datetime.strptime(args.start_date, '%Y-%m-%d').date()
                  if args.start_date else None)
    end_date = (datetime.strptime(args.end_date, '%Y-%m-%d').date()
                if args.end_date else None)

    dates = discover_dates(
        lightning_dir, args.lightning_product, start_date, end_date
    )
    print(f"Discovered {len(dates)} dates with {args.lightning_product} data")
    if not dates:
        return 1

    # Stages 1 + 2
    bins_per_date: dict = {}
    for date_str in dates:
        maps = load_date_maps(
            lightning_dir, args.lightning_product, date_str
        )
        bins = process_date(
            maps, args.aggregation_minutes, step_minutes,
            args.map_fraction_threshold,
        )
        bins_per_date[date_str] = bins
        n_valid = sum(1 for b in bins if b['valid_bin'])
        print(f"  {date_str}: {len(maps):>3} maps -> "
              f"{n_valid:>3}/{bins_per_day} valid bins")

    # Stage 3
    valid_day_by_date = apply_day_filter(
        bins_per_date, bins_per_day, args.day_min_valid_fraction,
    )

    # Stage 4
    kept_dates = apply_window_filter(
        valid_day_by_date,
        args.window_days,
        args.window_min_valid_days,
    )

    # Outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'step_minutes':            step_minutes,
        'aggregation_minutes':     args.aggregation_minutes,
        'bins_per_day':            bins_per_day,
        'maps_per_bin':            maps_per_bin,
        'lightning_product':       args.lightning_product,
        'map_fraction_threshold':  args.map_fraction_threshold,
        'day_min_valid_fraction':  args.day_min_valid_fraction,
        'window_days':             args.window_days,
        'window_min_valid_days':   args.window_min_valid_days,
        'start_date':              start_date.isoformat() if start_date else None,
        'end_date':                end_date.isoformat() if end_date else None,
        'created_utc':             datetime.now(timezone.utc)
                                           .isoformat(timespec='seconds'),
    }
    write_config(output_dir / 'lightning_periods_config.json', config)

    write_bins_csv(output_dir / 'lightning_bins.csv',
                   bins_per_date, args.aggregation_minutes)
    write_periods_csv(output_dir / 'lightning_periods.csv',
                      bins_per_date, valid_day_by_date,
                      kept_dates, bins_per_day)
    n_patch_rows = write_patches_csv(
        output_dir / 'lightning_patches.csv',
        bins_per_date, valid_day_by_date, kept_dates,
    )

    # Summary
    n_stage3 = sum(1 for v, _, _ in valid_day_by_date.values() if v)
    n_final = sum(
        1 for d in valid_day_by_date
        if d in kept_dates and valid_day_by_date[d][0]
    )

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Dates processed              : {len(dates)}")
    print(f"  Days passing Stage 3 (valid) : {n_stage3}")
    print(f"  Days inside kept windows     : {len(kept_dates)}")
    print(f"  Final kept dates             : {n_final}")
    print(f"  lightning_patches.csv rows   : {n_patch_rows}")
    print()
    print(f"Wrote: {output_dir}")
    print(f"  lightning_periods_config.json")
    print(f"  lightning_bins.csv")
    print(f"  lightning_periods.csv")
    print(f"  lightning_patches.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
