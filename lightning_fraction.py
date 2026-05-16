"""
Compute the fraction of non-zero (ones) pixels across the *active*
lightning maps - i.e. the ones the training pipeline will actually
consume - and emit a class-imbalance prior for train_models.py.

Formula: fraction = ones_pixels / total_pixels  (0.0 if total_pixels == 0)

By default, only `.npy` maps whose `(date, HH:MM)` appears in
`lightning_active_steps.csv` (produced by
`our_data/lightning_data/summarize_lightning_data.py`) are scanned.
This matches the gate downstream tooling applies, so the resulting
fraction reflects what the model sees - empty-window maps don't
dilute the denominator.

Pass `--scope_csv path/to/file.csv` to use a different (date, HH:MM)-
keyed CSV (`extract_patch_seq_for_datasets.py` outputs `train_data.csv`
and friends, which would scope the fraction even tighter to the train
split). Pass `--scope_csv none` to fall back to the legacy behaviour
(scan every `.npy` on disk).

Output: lightning_fraction.json (at `--data_root` by default).

Usage:
    python lightning_fraction.py
    python lightning_fraction.py --scope_csv train_data.csv
    python lightning_fraction.py --scope_csv none
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / 'our_data')
DEFAULT_SCOPE_CSV = str(PROJECT_ROOT / 'lightning_active_steps.csv')

PRODUCTS = ['density', 'current', 'occurrence']


def load_scope_set(csv_path):
    """Return the set of (date_str, HHMM) tuples to scope the scan by.

    Reads any CSV with at least `date` and `time_utc` columns
    (`lightning_active_steps.csv`, `train_data.csv`, etc.). HH:MM in
    the CSV is normalised to a 4-digit HHMM string so it matches the
    `.npy` filenames on disk. Returns `None` to mean "no scope" - the
    caller should treat that as the legacy "scan everything" mode.
    """
    if not csv_path or str(csv_path).lower() in ('none', ''):
        return None
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Scope CSV not found: {p}.\n"
            f"Pass --scope_csv none to scan every .npy on disk instead, or "
            f"run summarize_lightning_data.py to produce the default "
            f"lightning_active_steps.csv."
        )
    keys = set()
    with open(p, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{p}: empty CSV")
        for col in ('date', 'time_utc'):
            if col not in reader.fieldnames:
                raise ValueError(
                    f"{p}: missing required column {col!r} "
                    f"(have: {reader.fieldnames})"
                )
        # If the CSV has activity flag columns (density/current/occurrence
        # in lightning_active_steps.csv), only keep rows where at least
        # one flag is 1. For other CSVs (train_data.csv etc.) every row
        # counts. We distinguish the two by checking whether any of the
        # known flag columns exist.
        flag_cols = [c for c in reader.fieldnames if c in PRODUCTS]
        for row in reader:
            if flag_cols and not any(
                (row.get(c) or '').strip() == '1' for c in flag_cols
            ):
                continue
            date_str = (row.get('date') or '').strip()
            time_str = (row.get('time_utc') or '').strip()
            if not date_str or not time_str:
                continue
            hhmm = time_str.replace(':', '').zfill(4)
            keys.add((date_str, hhmm))
    return keys


def parse_filename(npy_file, product):
    """Extract (date, HHMM) from `lightning_<product>_YYYYMMDD_HHMM.npy`.

    Returns None when the filename doesn't match the expected pattern -
    such files are skipped silently to keep stray files from poisoning
    the scope filter.
    """
    name = npy_file
    if not name.endswith('.npy'):
        return None
    base = name[:-len('.npy')]
    prefix = f"lightning_{product}_"
    if not base.startswith(prefix):
        return None
    rest = base[len(prefix):]
    if len(rest) != 13 or rest[8] != '_':
        return None
    yyyymmdd, hhmm = rest[:8], rest[9:]
    if not (yyyymmdd.isdigit() and hhmm.isdigit() and len(hhmm) == 4):
        return None
    date_str = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return date_str, hhmm


def compute_fraction(data_root, scope_keys):
    """
    Compute fraction of non-zero pixels across the scoped lightning maps.

    Args:
        data_root: path to `our_data/`
        scope_keys: set of (date_str, HHMM) tuples to scan, or None to
            scan every `.npy` on disk.

    Returns:
        dict: per-product and global fractions + scope metadata.
    """
    lightning_dir = os.path.join(data_root, 'lightning_data')

    stats = {}
    global_ones = 0
    global_total = 0

    for product in PRODUCTS:
        product_dir = os.path.join(lightning_dir, product)
        if not os.path.isdir(product_dir):
            print(f"  {product}: NOT FOUND")
            continue

        ones = 0
        total = 0
        n_files = 0
        n_skipped_scope = 0

        for day_folder in sorted(os.listdir(product_dir)):
            day_path = os.path.join(product_dir, day_folder)
            if not os.path.isdir(day_path):
                continue

            for npy_file in sorted(os.listdir(day_path)):
                if not npy_file.endswith('.npy'):
                    continue

                if scope_keys is not None:
                    key = parse_filename(npy_file, product)
                    if key is None or key not in scope_keys:
                        n_skipped_scope += 1
                        continue

                filepath = os.path.join(day_path, npy_file)
                try:
                    data = np.load(filepath)
                    if data.ndim == 3:
                        data = np.squeeze(data, axis=0)
                    ones += int(np.count_nonzero(data))
                    total += int(data.size)
                    n_files += 1
                except Exception as e:
                    print(f"    ERROR {npy_file}: {e}")

        fraction = ones / total if total > 0 else 0.0

        stats[product] = {
            'ones_pixels':       ones,
            'total_pixels':      total,
            'fraction':          fraction,
            'n_files':           n_files,
            'n_skipped_scope':   n_skipped_scope,
        }

        global_ones += ones
        global_total += total

        scope_note = (
            f", {n_skipped_scope} skipped out-of-scope"
            if scope_keys is not None else ""
        )
        print(f"  {product}: {fraction:.6f} "
              f"({ones:,} / {total:,}, {n_files} files{scope_note})")

    global_fraction = global_ones / global_total if global_total > 0 else 0.0
    stats['global'] = {
        'ones_pixels':  global_ones,
        'total_pixels': global_total,
        'fraction':     global_fraction,
    }

    print(f"\n  Global: {global_fraction:.6f} "
          f"({global_ones:,} / {global_total:,})")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute fraction of non-zero pixels in the active "
                    "lightning maps. By default scopes the scan to "
                    "lightning_active_steps.csv so empty-window maps "
                    "don't dilute the denominator."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path (default: our_data/lightning_fraction.json)"
    )
    parser.add_argument(
        "--scope_csv", "-s", type=str, default=DEFAULT_SCOPE_CSV,
        help=f"Scope the scan by (date, HH:MM) pairs read from this CSV "
             f"(default: {DEFAULT_SCOPE_CSV}). Pass 'none' to scan every "
             f"`.npy` on disk (legacy behaviour). Any CSV with `date` + "
             f"`time_utc` columns works - e.g. train_data.csv to scope "
             f"the fraction to the training split."
    )

    args = parser.parse_args()
    output_path = args.output or os.path.join(
        args.data_root, 'lightning_fraction.json'
    )

    print("=" * 60)
    print("Lightning pixel fraction (active-step scoped)")
    print("=" * 60)
    print(f"  data_root  : {args.data_root}")
    print(f"  scope_csv  : {args.scope_csv}")

    scope_keys = load_scope_set(args.scope_csv)
    if scope_keys is None:
        print(f"  scope size : (none - scanning every .npy on disk)")
    else:
        print(f"  scope size : {len(scope_keys)} (date, HHMM) pairs")
    print()

    stats = compute_fraction(args.data_root, scope_keys)

    # Add scope metadata so the resulting JSON is self-describing.
    stats['_scope'] = {
        'scope_csv':  None if scope_keys is None else str(Path(args.scope_csv).resolve()),
        'n_scope_keys': None if scope_keys is None else len(scope_keys),
    }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
