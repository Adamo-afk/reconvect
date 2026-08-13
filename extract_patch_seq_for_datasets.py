"""
COALITION-4 continuous patch sequence extractor with temporal block split.

Analyzes patch_index.csv to find patches with temporal continuity
(2 past + current + 3 future = 6 consecutive 15-min steps), then splits
them into train/validation/test using the Czibula et al. (2024) method:

    Each day is divided into equal temporal blocks (default 6h, configurable).
    Within each block, qualifying sequences are ordered chronologically:
        - First 10%  → test
        - Next 10%   → validation
        - Remaining 80% → training

    This ensures all three splits sample from the same diurnal distribution,
    avoiding hour-based bias.

Window structure (15-min steps):
    T-30  T-15   T   T+15  T+30  T+45
     ↑     ↑     ↑     ↑     ↑     ↑
    past  past  curr  fut   fut   fut

Output:
    our_data/train_data.csv
    our_data/validation_data.csv
    our_data/test_data.csv

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python extract_sequences.py
    python extract_sequences.py --block_hours 4
    python extract_sequences.py --block_hours 8 --test_frac 0.15 --val_frac 0.15
    python extract_sequences.py --past 2 --future 3
"""

import csv
import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from pipeline_config import SOURCE


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)

N_PATCHES = 18

# STEP_MINUTES is loaded from the project-level timestep_config.json so
# sequence extraction always matches the cadence chosen by validate_timestep.py.
PROJECT_ROOT = Path(__file__).resolve().parent
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"


def _load_step_minutes():
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


STEP_MINUTES = _load_step_minutes()

# Temporal window (in steps; the wall-clock duration scales with STEP_MINUTES)
DEFAULT_PAST_STEPS = 2     # 2 × STEP_MINUTES before T
DEFAULT_FUTURE_STEPS = 3   # 3 × STEP_MINUTES after T

# Czibula split fractions
DEFAULT_TEST_FRAC = 0.10
DEFAULT_VAL_FRAC = 0.10

# Default temporal block size (hours)
DEFAULT_BLOCK_HOURS = 6


def build_block_boundaries(block_hours):
    """
    Build 24h block boundaries from a given block size.

    Args:
        block_hours: size of each block in hours (must divide 24 evenly)

    Returns:
        list[tuple]: [(0, block_hours), (block_hours, 2*block_hours), ...]
    """
    if 24 % block_hours != 0:
        raise ValueError(
            f"Block size {block_hours}h does not divide 24h evenly. "
            f"Valid values: 1, 2, 3, 4, 6, 8, 12, 24"
        )
    return [(h, h + block_hours) for h in range(0, 24, block_hours)]


# =============================================================================
# Patch index loader
# =============================================================================

def resolve_index_source(data_root, source):
    """
    Resolve the CSV path + effective step minutes for the chosen activity source.

    Returns: (csv_path, effective_step_minutes)

    - source='dbscan': reads our_data/patch_index/patch_index.csv (DBSCAN-
      driven, produced by identify_patches.py for either --source radar or
      --source opera) with the cadence from timestep_config.json
      (STEP_MINUTES).
      occurrence map produces one row at the native step_minutes cadence -
      so the effective sequence step equals step_minutes.
    """
    if source == SOURCE:
        return (
            os.path.join(data_root, 'patch_index', 'patch_index.csv'),
            STEP_MINUTES,
        )
    raise ValueError(f"Unknown source: {source}")


def load_patch_index(data_root, source='dbscan'):
    """
    Load the per-(date, time) patch-activity index for the chosen source.

    Returns:
        dict: {(date_str, time_str): [sorted list of active patch numbers]}
    """
    csv_path, _ = resolve_index_source(data_root, source)
    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found")
        print("Run identify_patches.py first.")
        return {}

    index = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row['date']
            time_str = row['time_utc']

            active = [
                p for p in range(1, N_PATCHES + 1)
                if row.get(f'patch_{p}', '0') == '1'
            ]

            if active:
                index[(date_str, time_str)] = active

    return index


# =============================================================================
# Manifest gate (cross-product timestep filter from intersect_product_coverage)
# =============================================================================

def load_manifest_timesteps(manifest_path):
    """Return the set of (date, 'HH:MM') tuples listed in the manifest.

    `timestep_manifest.csv` (output of `intersect_product_coverage.py`)
    has columns `date,hhmm,<key>_hhmm,...`. We only need the first two -
    the per-product columns are an audit trail consumed by other tooling.
    Returns None when no manifest path is supplied / discovered, so the
    caller can distinguish "no manifest" from "manifest with zero rows".
    """
    if not manifest_path or not os.path.isfile(manifest_path):
        return None
    used: set[tuple[str, str]] = set()
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'date' not in reader.fieldnames \
                or 'hhmm' not in reader.fieldnames:
            return set()
        for row in reader:
            date_str = row['date'].strip()
            hhmm = row['hhmm'].strip()
            if not date_str or len(hhmm) != 4 or not hhmm.isdigit():
                continue
            used.add((date_str, f"{hhmm[:2]}:{hhmm[2:]}"))
    return used


def apply_manifest_gate(index, manifest_set, drops_csv_path):
    """Filter `index` by `manifest_set` and report what was dropped.

    `index` is the dict from `load_patch_index` (one key per active
    (date, time)). `manifest_set` is the set returned by
    `load_manifest_timesteps`, or None when no manifest was supplied.
    `drops_csv_path` is where the per-source audit CSV listing every
    dropped (date, time) is written when there's anything to write.

    Returns the manifest-filtered index. When `manifest_set` is None,
    returns the input unchanged and prints a notice so the no-gate
    behaviour is visible rather than silent.
    """
    if manifest_set is None:
        print()
        print("Manifest gate")
        print("-" * 70)
        print("  No timestep_manifest.csv found - NO cross-product gate "
              "applied.")
        print("  Run `python intersect_product_coverage.py ...` to produce "
              "one, or pass --manifest PATH explicitly.")
        print()
        return index

    before = len(index)
    kept_keys = [k for k in index.keys() if k in manifest_set]
    dropped_keys = sorted(k for k in index.keys() if k not in manifest_set)
    after = len(kept_keys)
    n_dropped = before - after

    print()
    print("Manifest gate")
    print("-" * 70)
    print(f"  Patch index entries        : {before}")
    print(f"  Manifest entries           : {len(manifest_set)}")
    print(f"  Kept (in both)             : {after}")
    print(f"  Dropped (index \\ manifest) : {n_dropped} "
          f"({(n_dropped / before * 100) if before else 0:.1f}%)")

    if n_dropped:
        # Per-date breakdown: which days lost the most timesteps?
        drop_count_by_date = Counter(d for d, _ in dropped_keys)
        top_drops = drop_count_by_date.most_common(10)
        print()
        print(f"  Top {min(10, len(top_drops))} dates by drop count:")
        for date_str, n in top_drops:
            print(f"    {date_str} : {n} timesteps dropped")

        # Audit CSV with every dropped (date, time) so the decision is
        # reproducible from disk, not just from this script's stdout.
        os.makedirs(os.path.dirname(drops_csv_path) or '.', exist_ok=True)
        with open(drops_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['date', 'time_utc'])
            for date_str, time_str in dropped_keys:
                w.writerow([date_str, time_str])
        print()
        print(f"  Audit CSV: {drops_csv_path} ({n_dropped} rows)")
    print()

    return {k: index[k] for k in kept_keys}


# =============================================================================
# Sequence analysis (full 24h scan)
# =============================================================================

def find_all_sequences(index, past_steps, future_steps,
                       step_minutes=None):
    """
    Find all qualifying sequences across all dates and all 24 hours.

    A patch qualifies at reference timestep T if it is active at every step
    from T - past_steps*step_minutes to T + future_steps*step_minutes
    (inclusive).

    Args:
        index: dict from load_patch_index()
        past_steps, future_steps: window size in step units
        step_minutes: spacing between adjacent steps (default: module-level
            STEP_MINUTES from timestep_config.json).
    """
    if step_minutes is None:
        step_minutes = STEP_MINUTES
    step = timedelta(minutes=step_minutes)
    offsets = list(range(-past_steps, future_steps + 1))
    results = []

    dates = sorted(set(d for d, _ in index.keys()))

    for date_str in dates:
        base_dt = datetime.strptime(date_str, '%Y-%m-%d')
        t = base_dt
        t_end = base_dt + timedelta(hours=24)

        while t < t_end:
            ref_time = t.strftime('%H:%M')

            active_at_ref = index.get((date_str, ref_time))
            if not active_at_ref:
                t += step
                continue

            # Collect active patches at every step in the window
            window_active = {}
            for offset in offsets:
                check_dt = t + offset * step
                check_date = check_dt.strftime('%Y-%m-%d')
                check_time = check_dt.strftime('%H:%M')
                window_active[offset] = index.get(
                    (check_date, check_time), []
                )

            # Check each patch for continuity
            qualifying_patches = []
            for patch_num in active_at_ref:
                if all(patch_num in window_active[o] for o in offsets):
                    qualifying_patches.append(patch_num)

            if qualifying_patches:
                indices_per_step = []
                for offset in offsets:
                    step_active = window_active[offset]
                    step_indices = [
                        step_active.index(p) for p in qualifying_patches
                    ]
                    indices_per_step.append(step_indices)

                window_start = t - past_steps * step
                window_end = t + future_steps * step

                results.append({
                    'date': date_str,
                    'reference_utc': ref_time,
                    'start_utc': window_start.strftime('%H:%M'),
                    'end_utc': window_end.strftime('%H:%M'),
                    'patch_numbers': qualifying_patches,
                    'indices_per_step': indices_per_step,
                    'n_qualifying': len(qualifying_patches),
                    'n_total_active': len(active_at_ref),
                    'ref_hour': t.hour,
                })

            t += step

    return results


# =============================================================================
# Czibula temporal block split
# =============================================================================

def split_sequences_czibula(all_sequences, test_frac, val_frac, block_hours):
    """
    Split sequences using the Czibula et al. (2024) temporal block method.

    Each day is divided into equal periods of block_hours length.
    Within each period, the qualifying sequences are ordered chronologically:
        - First test_frac (10%) → test
        - Next val_frac (10%)  → validation
        - Remaining (80%)      → training

    This ensures all three splits sample from the same diurnal distribution.

    Args:
        all_sequences: list of sequence dicts (from find_all_sequences)
        test_frac: fraction for test (default 0.10)
        val_frac: fraction for validation (default 0.10)
        block_hours: size of each temporal block in hours

    Returns:
        tuple: (train_list, val_list, test_list)
    """
    train = []
    val = []
    test = []

    block_boundaries = build_block_boundaries(block_hours)

    # Group sequences by (date, block)
    blocks = defaultdict(list)
    for seq in all_sequences:
        hour = seq['ref_hour']
        for block_start, block_end in block_boundaries:
            if block_start <= hour < block_end:
                blocks[(seq['date'], block_start, block_end)].append(seq)
                break

    # Split each block chronologically
    for block_key in sorted(blocks.keys()):
        block_seqs = blocks[block_key]
        # Already chronological from find_all_sequences
        n = len(block_seqs)

        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0

        # Ensure we don't exceed total
        if n_test + n_val >= n:
            # Too few samples — put everything in training
            train.extend(block_seqs)
            continue

        test.extend(block_seqs[:n_test])
        val.extend(block_seqs[n_test:n_test + n_val])
        train.extend(block_seqs[n_test + n_val:])

    return train, val, test


# =============================================================================
# Output
# =============================================================================

def step_column_name(offset):
    """Index-encoded column name for a given step offset.

    Examples: -2 -> 'idx_t-2', 0 -> 'idx_t0', +1 -> 'idx_t+1'.
    Step *indices* are used (rather than minute offsets) so the CSV format
    remains stable across different timestep_config.json choices.
    """
    if offset < 0:
        return f'idx_t{offset}'
    if offset == 0:
        return 'idx_t0'
    return f'idx_t+{offset}'


def save_sequences(results, output_path, past_steps, future_steps):
    """Save sequence results to CSV with per-timestep npy index columns.

    Column names use step *indices* (idx_t-2, idx_t-1, idx_t0, idx_t+1, ...)
    rather than minute offsets, so the schema is independent of the cadence
    chosen via validate_timestep.py. Convert back to minutes via
    `STEP_MINUTES * offset` when needed.
    """
    if not results:
        print(f"  No qualifying sequences for {output_path}")
        return 0

    step_columns = [step_column_name(o) for o in range(-past_steps, future_steps + 1)]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'reference_utc', 'start_utc', 'end_utc',
            'patch_numbers', *step_columns,
            'n_qualifying', 'n_total_active'
        ])

        for r in results:
            row = [
                r['date'],
                r['reference_utc'],
                r['start_utc'],
                r['end_utc'],
                str(r['patch_numbers']),
            ]
            for step_indices in r['indices_per_step']:
                row.append(str(step_indices))
            row.extend([
                r['n_qualifying'],
                r['n_total_active'],
            ])
            writer.writerow(row)

    print(f"  Saved: {output_path} ({len(results)} rows)")
    return len(results)


def print_summary(train, val, test, past_steps, future_steps, block_hours):
    """Print summary statistics for all three splits."""
    total_window = past_steps + 1 + future_steps

    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"  Window             : {past_steps} past + current + "
          f"{future_steps} future = {total_window} steps "
          f"({total_window * STEP_MINUTES} min)")
    print(f"  Block size         : {block_hours}h "
          f"({24 // block_hours} blocks per day)")

    for name, data in [('Train', train), ('Validation', val), ('Test', test)]:
        if not data:
            print(f"\n  {name:12s}: 0 sequences")
            continue

        dates = sorted(set(r['date'] for r in data))
        total_patches = sum(r['n_qualifying'] for r in data)
        hours = sorted(set(r['ref_hour'] for r in data))

        print(f"\n  {name:12s}: {len(data)} sequences, "
              f"{total_patches} patch samples, "
              f"{len(dates)} dates")
        print(f"  {'':12s}  Hours: {min(hours):02d}–{max(hours):02d} UTC")

    # Diurnal distribution check
    block_boundaries = build_block_boundaries(block_hours)
    print(f"\n  Diurnal distribution (sequences per {block_hours}h block):")
    print(f"  {'Block':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*40}")
    for block_start, block_end in block_boundaries:
        block_label = f"{block_start:02d}-{block_end:02d} UTC"
        n_train = sum(1 for r in train if block_start <= r['ref_hour'] < block_end)
        n_val = sum(1 for r in val if block_start <= r['ref_hour'] < block_end)
        n_test = sum(1 for r in test if block_start <= r['ref_hour'] < block_end)
        print(f"  {block_label:<12} {n_train:>8} {n_val:>8} {n_test:>8}")

    # Patch frequency per split
    for name, data in [('Train', train), ('Validation', val), ('Test', test)]:
        if not data:
            continue
        patch_counts = {}
        for r in data:
            for p in r['patch_numbers']:
                patch_counts[p] = patch_counts.get(p, 0) + 1
        if patch_counts:
            print(f"\n  {name} patch frequency:")
            for p in sorted(patch_counts.keys()):
                print(f"    Patch {p:2d}: {patch_counts[p]:4d} times")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract continuous patch sequences and split into "
                    "train/validation/test using the Czibula temporal "
                    "block method."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--past", type=int, default=DEFAULT_PAST_STEPS,
        help=f"Past steps required ({STEP_MINUTES}-min cadence) "
             f"(default: {DEFAULT_PAST_STEPS})"
    )
    parser.add_argument(
        "--future", type=int, default=DEFAULT_FUTURE_STEPS,
        help=f"Future steps required ({STEP_MINUTES}-min cadence) "
             f"(default: {DEFAULT_FUTURE_STEPS})"
    )
    parser.add_argument(
        "--test_frac", type=float, default=DEFAULT_TEST_FRAC,
        help=f"Fraction for test per 6h block (default: {DEFAULT_TEST_FRAC})"
    )
    parser.add_argument(
        "--val_frac", type=float, default=DEFAULT_VAL_FRAC,
        help=f"Fraction for validation per block (default: {DEFAULT_VAL_FRAC})"
    )
    parser.add_argument(
        "--block_hours", type=int, default=DEFAULT_BLOCK_HOURS,
        help=f"Temporal block size in hours; must divide 24 evenly "
             f"(default: {DEFAULT_BLOCK_HOURS})"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to timestep_manifest.csv from intersect_product_coverage.py "
             "(default: auto-discover at <data_root>/timestep_manifest.csv). "
             "Pass 'none' to disable the manifest gate explicitly; useful for "
             "debugging where you want sequences before the cross-product "
             "intersection is applied."
    )

    args = parser.parse_args()

    # Resolve effective cadence + index path from the chosen source
    _, effective_step_minutes = resolve_index_source(args.data_root, SOURCE)

    train_frac = 1.0 - args.test_frac - args.val_frac
    n_blocks = 24 // args.block_hours if 24 % args.block_hours == 0 else '?'

    print("=" * 70)
    print("COALITION-4 Sequence Extractor (Czibula temporal block split)")
    print("=" * 70)
    print(f"Data root        : {args.data_root}")
    print(f"Activity source  : {SOURCE}")
    print(f"Step interval    : {effective_step_minutes} min")
    print(f"Window           : {args.past} past + current + {args.future} future "
          f"= {args.past + 1 + args.future} steps "
          f"({(args.past + 1 + args.future) * effective_step_minutes} min)")
    print(f"Blocks           : {args.block_hours}h ({n_blocks} blocks/day)")
    print(f"Split            : test {args.test_frac:.0%} / "
          f"val {args.val_frac:.0%} / "
          f"train {train_frac:.0%} per block")

    # Load patch index
    print(f"\nLoading {SOURCE} patch index...")
    index = load_patch_index(args.data_root, source=SOURCE)
    if not index:
        return

    dates = sorted(set(d for d, _ in index.keys()))
    print(f"  {len(index)} active timestamps across {len(dates)} dates")

    # Apply the cross-product manifest gate. This is the final filter -
    # extract_patches.py reads the same manifest, so any (date, time) we
    # surface in train/val/test_data.csv has guaranteed coverage from
    # every product in the intersect set. Dropped entries are printed +
    # written to an audit CSV so the decision is reproducible from disk.
    if args.manifest is None:
        manifest_path = os.path.join(args.data_root, 'timestep_manifest.csv')
    elif args.manifest.lower() in ('none', ''):
        manifest_path = ''
    else:
        manifest_path = args.manifest
    manifest_set = (load_manifest_timesteps(manifest_path)
                    if manifest_path else None)
    drops_csv_path = os.path.join(
        args.data_root, f'extract_patch_seq_drops_{SOURCE}.csv'
    )
    index = apply_manifest_gate(index, manifest_set, drops_csv_path)
    if not index:
        print("No timestamps survive the manifest gate. Nothing to do.")
        return

    # Find all qualifying sequences (full 24h)
    print("Analyzing temporal continuity (all 24h)...")
    all_sequences = find_all_sequences(
        index, args.past, args.future,
        step_minutes=effective_step_minutes,
    )
    print(f"  {len(all_sequences)} qualifying sequences found")

    if not all_sequences:
        print(f"No qualifying sequences. Check the {SOURCE} index CSV.")
        return

    # Split using Czibula method
    print("\nSplitting with Czibula temporal block method...")
    train, val, test = split_sequences_czibula(
        all_sequences, args.test_frac, args.val_frac, args.block_hours
    )
    print(f"  Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

    # Save three CSVs. Outputs are suffixed with `_<source>` so the
    # DBSCAN-driven and lightning-driven tracks can coexist on disk
    # (domain-adaptation pipeline trains both and uses them as separate
    # feature extractors).
    print("\nSaving results...")
    src = SOURCE
    save_sequences(
        train,
        os.path.join(args.data_root, f'train_data_{src}.csv'),
        args.past, args.future
    )
    save_sequences(
        val,
        os.path.join(args.data_root, f'validation_data_{src}.csv'),
        args.past, args.future
    )
    save_sequences(
        test,
        os.path.join(args.data_root, f'test_data_{src}.csv'),
        args.past, args.future
    )

    # Drop a sidecar metadata file so create_datasets.py can recover step
    # minutes and window length without re-parsing column names.
    # `step_minutes` is the *effective* spacing between adjacent steps
    # (i.e. aggregation_minutes when source=lightning), so downstream
    # consumers can treat it uniformly. `source_step_minutes_native` is
    # the configured cadence from timestep_config.json - preserved for
    # traceability. Suffixed by source so the two tracks don't clobber
    # each other's metadata.
    seq_meta = {
        "source": SOURCE,
        "step_minutes": effective_step_minutes,
        "source_step_minutes_native": STEP_MINUTES,
        "past_steps": args.past,
        "future_steps": args.future,
        "step_columns": [step_column_name(o)
                         for o in range(-args.past, args.future + 1)],
    }
    seq_meta_path = os.path.join(
        args.data_root, f'sequence_meta_{src}.json'
    )
    with open(seq_meta_path, 'w') as f:
        json.dump(seq_meta, f, indent=2)
    print(f"  Saved sequence metadata -> {seq_meta_path}")

    # Summary
    print_summary(train, val, test, args.past, args.future, args.block_hours)


if __name__ == "__main__":
    main()
    