"""
Compute per-class pixel fractions across the OPERA rainfall-rate maps
used for training, and emit the class-imbalance prior `train_models.py`
reads at compile time for the radar multiclass head
(`WeightedFocalCategoricalCrossentropy.class_fractions`).

Classes match `label_transform_opera_rainfall_multiclass` in
`create_datasets.py`:
    0: R < 10 mm/h
    1: 10 <= R < 20
    2: 20 <= R < 30
    3: 30 <= R < 40
    4: R >= 40

Scope works exactly like `lightning_fraction.py`: per-source by default
(`--source dbscan` scopes to `train_data_dbscan.csv` and writes
`opera_rainfall_fraction_dbscan.json`). Reusing
`lightning_fraction.load_scope_set` keeps the (date, HHMM) snap rules
identical so the prior is computed over the same timesteps the dataset
actually contains.

Usage:
    python opera_rainfall_fraction.py --source dbscan
    python opera_rainfall_fraction.py --source lightning
    python opera_rainfall_fraction.py --source dbscan --scope_csv none
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from lightning_fraction import load_scope_set

from pipeline_config import SOURCE


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / 'our_data')

# Bin edges (mm/h) — kept in lock-step with
# create_datasets.label_transform_opera_rainfall_multiclass.
CLASS_EDGES = [0.0, 10.0, 20.0, 30.0, 40.0]
CLASS_LABELS = ["<10", "10-20", "20-30", "30-40", ">=40"]
N_CLASSES = len(CLASS_LABELS)

# On-disk layout of reprojected OPERA rainfall_rate, per reproject.py:
#   our_data/reprojected_data/opera_data/rainfall_rate/
#       nc4_{YYYY-MM-DD}-Romania_rainfall_rate/
#           nc4_{YYYY-MM-DD}-Romania_{HHMM}_rainfall_rate.npy
OPERA_REPROJECTED_SUBDIR = os.path.join(
    'reprojected_data', 'opera_data', 'rainfall_rate'
)
OPERA_SHORT = 'rainfall_rate'


def parse_filename(npy_file: str):
    """Extract (date_str, HHMM) from
    `nc4_{YYYY-MM-DD}-Romania_{HHMM}_rainfall_rate.npy`.

    Returns None for any filename that doesn't match the expected
    schema so stray files (READMEs, partial downloads) are silently
    skipped — same behaviour as lightning_fraction.parse_filename.
    """
    if not npy_file.endswith('.npy'):
        return None
    base = npy_file[:-len('.npy')]
    suffix = f"_{OPERA_SHORT}"
    if not base.endswith(suffix):
        return None
    stem = base[:-len(suffix)]                            # nc4_YYYY-MM-DD-Romania_HHMM
    if not stem.startswith('nc4_'):
        return None
    rest = stem[len('nc4_'):]                             # YYYY-MM-DD-Romania_HHMM
    if '-Romania_' not in rest:
        return None
    date_part, hhmm = rest.rsplit('-Romania_', 1)
    if len(hhmm) != 4 or not hhmm.isdigit():
        return None
    # date_part is YYYY-MM-DD
    if len(date_part) != 10 or date_part[4] != '-' or date_part[7] != '-':
        return None
    return date_part, hhmm


def _bin_counts(arr: np.ndarray) -> np.ndarray:
    """Count pixels per rainfall class for one frame.

    NaNs are treated as 0 mm/h (class 0), matching the dataset
    transform in create_datasets.label_transform_opera_rainfall_multiclass.
    """
    x = np.where(np.isnan(arr), 0.0, arr)
    x = np.clip(x, 0.0, None)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    # Use the same boundaries as the dataset transform:
    counts[0] = int(np.sum(x < 10.0))
    counts[1] = int(np.sum((x >= 10.0) & (x < 20.0)))
    counts[2] = int(np.sum((x >= 20.0) & (x < 30.0)))
    counts[3] = int(np.sum((x >= 30.0) & (x < 40.0)))
    counts[4] = int(np.sum(x >= 40.0))
    return counts


def compute_class_fractions(data_root: str, scope_keys):
    """Walk reprojected OPERA rainfall_rate files and accumulate
    per-class pixel counts.

    Args:
        data_root: path to `our_data/`
        scope_keys: set of (date, HHMM) tuples, or None to scan every
            file under reprojected_data/opera_data/rainfall_rate/.

    Returns: dict with per-class fractions + raw counts + scope stats.
    """
    root = os.path.join(data_root, OPERA_REPROJECTED_SUBDIR)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"OPERA reprojected root not found: {root}\n"
            f"Run reproject.py first to produce the .npy files."
        )

    counts = np.zeros(N_CLASSES, dtype=np.int64)
    n_files = 0
    n_skipped_scope = 0
    n_errors = 0

    for day_folder in sorted(os.listdir(root)):
        day_path = os.path.join(root, day_folder)
        if not os.path.isdir(day_path):
            continue
        for npy_file in sorted(os.listdir(day_path)):
            if not npy_file.endswith('.npy'):
                continue
            if scope_keys is not None:
                key = parse_filename(npy_file)
                if key is None or key not in scope_keys:
                    n_skipped_scope += 1
                    continue
            filepath = os.path.join(day_path, npy_file)
            try:
                data = np.load(filepath)
                if data.ndim == 3:
                    data = np.squeeze(data, axis=0)
                counts += _bin_counts(data)
                n_files += 1
            except Exception as e:
                n_errors += 1
                print(f"    ERROR {npy_file}: {e}")

    total = int(counts.sum())
    fractions = (counts / total).tolist() if total > 0 else [0.0] * N_CLASSES

    per_class = {}
    for k in range(N_CLASSES):
        per_class[CLASS_LABELS[k]] = {
            'pixels':   int(counts[k]),
            'fraction': float(fractions[k]),
        }
        print(f"  class {k} ({CLASS_LABELS[k]:>5}): "
              f"{fractions[k]:.6f}  ({int(counts[k]):,} pixels)")

    print(f"\n  Total pixels: {total:,}  ({n_files} files, "
          f"{n_skipped_scope} skipped out-of-scope, {n_errors} errors)")

    return {
        'classes':         CLASS_LABELS,
        'fractions':       fractions,
        'counts':          counts.tolist(),
        'total_pixels':    total,
        'per_class':       per_class,
        'n_files':         n_files,
        'n_skipped_scope': n_skipped_scope,
        'n_errors':        n_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-class OPERA rainfall-rate pixel "
                    "fractions across the training-scope timesteps. The "
                    "resulting JSON feeds the WeightedFocalCategoricalCrossentropy "
                    "prior in train_models.py for radar multiclass modes."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path (default: "
             "our_data/opera_rainfall_fraction_<source>.json).",
    )
    parser.add_argument(
        "--scope_csv", "-s", type=str, default=None,
        help="Scope the scan by (date, HHMM) pairs read from this CSV. "
             "Defaults to our_data/train_data_<source>.csv. Pass 'none' "
             "to scan every .npy on disk.",
    )
    args = parser.parse_args()

    scope_csv = (
        args.scope_csv
        if args.scope_csv is not None
        else os.path.join(args.data_root, f"train_data_{SOURCE}.csv")
    )
    output_path = (
        args.output
        if args.output
        else os.path.join(
            args.data_root, f"opera_rainfall_fraction_{SOURCE}.json"
        )
    )

    print("=" * 60)
    print("OPERA rainfall-rate class fractions (training-scope)")
    print("=" * 60)
    print(f"  data_root  : {args.data_root}")
    print(f"  source     : {SOURCE}")
    print(f"  scope_csv  : {scope_csv}")
    print(f"  output     : {output_path}")

    scope_keys = load_scope_set(
        scope_csv, data_root=args.data_root, source=SOURCE,
    )
    if scope_keys is None:
        print(f"  scope size : (none - scanning every .npy on disk)")
    else:
        print(f"  scope size : {len(scope_keys)} (date, HHMM) pairs")
    print()

    stats = compute_class_fractions(args.data_root, scope_keys)
    stats['_scope'] = {
        'source':       SOURCE,
        'scope_csv':    None if scope_keys is None else str(Path(scope_csv).resolve()),
        'n_scope_keys': None if scope_keys is None else len(scope_keys),
    }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
