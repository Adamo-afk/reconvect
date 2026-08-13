"""
Compute the fraction of non-zero (ones) pixels across the lightning
maps used for training and emit the class-imbalance prior
`train_models.py` reads at compile time (focal-loss `ones_fraction`).

Formula: fraction = ones_pixels / total_pixels  (0.0 if total_pixels == 0)

The scope is per-source by default: `--source dbscan` scopes the scan
to `train_data_dbscan.csv` and writes
`lightning_fraction_dbscan.json`; `--source lightning` mirrors that
for `train_data_lightning.csv` and
`lightning_fraction_lightning.json`. The two tracks need separate
priors because their training distributions differ - the
OPERA-driven track samples many more "no lightning" timesteps than
the lightning-driven track, so the prior is meaningfully different.

Pass `--scope_csv path/to/file.csv` to override the auto-resolved
training CSV (e.g. `lightning_active_steps.csv` for the broader
active-step scope, or any other (date, HH:MM)-keyed CSV). Pass
`--scope_csv none` to scan every `.npy` on disk (legacy behaviour).

Usage:
    # Typical: per-source prior from the training split
    python lightning_fraction.py --source dbscan
    python lightning_fraction.py --source lightning

    # Broader / legacy scopes
    python lightning_fraction.py --source dbscan \\
        --scope_csv lightning_active_steps.csv
    python lightning_fraction.py --source dbscan --scope_csv none
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from datetime import datetime, timedelta

from pipeline_config import SOURCE


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / 'our_data')

PRODUCTS = ['density', 'current', 'occurrence']


def _load_step_minutes(data_root, source):
    """Read `step_minutes` from `sequence_meta_<source>.json`.

    Needed when the scope CSV is per-sequence (train_data_<source>.csv
    rows carry `start_utc / end_utc` instead of `time_utc`) and we
    have to expand each row's window at the run's cadence.
    """
    meta_path = Path(data_root) / f"sequence_meta_{source}.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} not found - needed to expand sequence rows "
            f"into per-timestep keys. Run "
            f"`python extract_patch_seq_for_datasets.py --source {source}` "
            f"first, or pass --scope_csv pointing at a per-timestep CSV "
            f"such as lightning_active_steps.csv."
        )
    meta = json.loads(meta_path.read_text())
    return int(meta["step_minutes"])


def _load_lightning_filter(data_root):
    """Return the set of lightning minute-of-hour slots from
    `timestep_config.json`. Returns `None` if no filter is set
    (continuous-cadence lightning), in which case no HHMM snapping is
    applied.
    """
    cfg_path = Path(data_root) / "timestep_config.json"
    if not cfg_path.is_file():
        return None
    cfg = json.loads(cfg_path.read_text())
    flt = cfg.get("products", {}).get("lightning", {}).get("filter")
    if not flt:
        return None
    return {int(m) for m in flt}


def _snap_hhmm_to_filter(hhmm, filter_minutes):
    """Snap an HHMM string to the nearest minute in `filter_minutes`.

    Same rule as `extract_patches.py._snap_hhmm_to_filter` so the keys
    here line up with the files that script writes / loads. Ties
    prefer the earlier minute so master :15 deterministically snaps
    to lightning :10 (not :20 if both existed).
    """
    h, m = int(hhmm[:2]), int(hhmm[2:])
    best = min(
        filter_minutes,
        key=lambda fm: (min(abs(fm - m), 60 - abs(fm - m)), fm),
    )
    diff = best - m
    if diff > 30:
        diff -= 60
    elif diff < -30:
        diff += 60
    total = (h * 60 + m + diff) % (24 * 60)
    return f"{total // 60:02d}{total % 60:02d}"


def _expand_sequence_rows(reader, step_minutes, filter_minutes):
    """Iterate rows with `date / start_utc / end_utc` and yield every
    `(date, HHMM)` tuple inside `[start_utc, end_utc]` at
    `step_minutes` spacing, snapped to the lightning filter so the
    HHMMs match the .npy filenames on disk (master grid :15 / :45
    snap to lightning :10 / :40 etc.).
    """
    step = timedelta(minutes=step_minutes)
    for row in reader:
        date_str = (row.get('date') or '').strip()
        start_str = (row.get('start_utc') or '').strip()
        end_str = (row.get('end_utc') or '').strip()
        if not (date_str and start_str and end_str):
            continue
        try:
            base = datetime.strptime(date_str, '%Y-%m-%d')
            sh, sm = (int(x) for x in start_str.split(':'))
            eh, em = (int(x) for x in end_str.split(':'))
        except ValueError:
            continue
        t = base.replace(hour=sh, minute=sm)
        t_end = base.replace(hour=eh, minute=em)
        # Windows don't cross midnight today; broken row -> skip.
        if t_end < t:
            continue
        while t <= t_end:
            hhmm = t.strftime('%H%M')
            if filter_minutes is not None:
                hhmm = _snap_hhmm_to_filter(hhmm, filter_minutes)
            yield date_str, hhmm
            t += step


def load_scope_set(csv_path, data_root=None, source=None):
    """Return the set of (date_str, HHMM) tuples to scope the scan by.

    Auto-detects the CSV shape:

    - **Per-timestep CSVs** (`lightning_active_steps.csv`,
      `patch_index.csv`) have `date,time_utc,...`. Each row is one
      (date, time); read as-is. When the CSV also carries the
      lightning sub-product flag columns
      (density/current/occurrence), only rows where at least one flag
      is 1 are kept.

    - **Per-sequence CSVs** (`train_data_<source>.csv` and friends
      from `extract_patch_seq_for_datasets.py`) have
      `date,reference_utc,start_utc,end_utc,...`. Each row's
      `[start_utc, end_utc]` window is expanded at `step_minutes`
      (from `sequence_meta_<source>.json`) and every produced HHMM
      is snapped to `products.lightning.filter` so it lines up with
      the filenames `read_kml_version2.py` writes (master :15
      snaps to lightning :10, etc.). `source` must be supplied for
      this path.

    Returns `None` when `csv_path` is empty or 'none' (legacy
    "scan every .npy on disk" mode).
    """
    if not csv_path or str(csv_path).lower() in ('none', ''):
        return None
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Scope CSV not found: {p}.\n"
            f"Pass --scope_csv none to scan every .npy on disk instead, or "
            f"run extract_patch_seq_for_datasets.py --source <s> to "
            f"produce the per-source train_data CSV."
        )

    with open(p, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{p}: empty CSV")
        fieldnames = set(reader.fieldnames)
        if 'date' not in fieldnames:
            raise ValueError(
                f"{p}: missing required column 'date' "
                f"(have: {reader.fieldnames})"
            )

        # Resolve the lightning filter once. The per-sequence branch
        # snaps every expanded HHMM; the per-timestep branch leaves
        # already-listed HHMMs alone (they're typically already on
        # the filter, e.g. lightning_active_steps.csv).
        eff_data_root = (
            data_root if data_root else str(p.parent)
        )
        filter_minutes = _load_lightning_filter(eff_data_root)

        # Branch 1: per-timestep CSV.
        if 'time_utc' in fieldnames:
            keys = set()
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
                keys.add((date_str, time_str.replace(':', '').zfill(4)))
            return keys

        # Branch 2: per-sequence CSV (start_utc + end_utc).
        if 'start_utc' in fieldnames and 'end_utc' in fieldnames:
            if source is None:
                raise ValueError(
                    f"{p}: per-sequence CSV requires --source so the "
                    f"matching sequence_meta_<source>.json can be read "
                    f"for step_minutes."
                )
            step_minutes = _load_step_minutes(eff_data_root, source)
            return set(_expand_sequence_rows(
                reader, step_minutes, filter_minutes,
            ))

        # Neither schema matched.
        raise ValueError(
            f"{p}: cannot determine scope schema. Expected either "
            f"`date,time_utc,...` (per-timestep) or "
            f"`date,start_utc,end_utc,...` (per-sequence). "
            f"Got columns: {reader.fieldnames}"
        )


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
        description="Compute fraction of non-zero pixels in the lightning "
                    "maps used for training. By default scopes the scan "
                    "to the train_data_<source>.csv produced by "
                    "extract_patch_seq_for_datasets.py so the focal-loss "
                    "prior matches the actual training distribution."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path (default: "
             "our_data/lightning_fraction_<source>.json)."
    )
    parser.add_argument(
        "--scope_csv", "-s", type=str, default=None,
        help="Scope the scan by (date, HH:MM) pairs read from this CSV. "
             "Defaults to our_data/train_data_<source>.csv so the prior "
             "matches the training distribution exactly. Pass 'none' to "
             "scan every `.npy` on disk (legacy behaviour). "
             "lightning_active_steps.csv at project root is also a "
             "useful broader scope when no train CSV exists yet."
    )

    args = parser.parse_args()

    # Per-source default paths. Explicit --scope_csv / --output flags
    # override either.
    scope_csv = (
        args.scope_csv
        if args.scope_csv is not None
        else os.path.join(args.data_root, f"train_data_{SOURCE}.csv")
    )
    output_path = (
        args.output
        if args.output
        else os.path.join(
            args.data_root, f"lightning_fraction_{SOURCE}.json"
        )
    )

    print("=" * 60)
    print("Lightning pixel fraction (training-scope)")
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

    stats = compute_fraction(args.data_root, scope_keys)

    # Add scope metadata so the resulting JSON is self-describing.
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
