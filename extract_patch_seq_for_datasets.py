# """
# COALITION-4 continuous patch sequence extractor.

# Analyzes patch_index.csv to find patches with temporal continuity:
# each qualifying patch must be active for at least 6 consecutive 15-minute
# steps (2 past + current + 3 future) around the reference timestep.

# The output tells the training/testing pipeline exactly which npy array
# indices to use for each valid sample.

# Window structure (15-min steps):
#     T-30  T-15   T   T+15  T+30  T+45
#      ↑     ↑     ↑     ↑     ↑     ↑
#     past  past  curr  fut   fut   fut

# Output columns:
#     date           — YYYY-MM-DD
#     reference_utc  — HH:MM (the "current" timestep T)
#     start_utc      — HH:MM (T - 30 min)
#     end_utc        — HH:MM (T + 45 min)
#     patch_numbers  — which patch IDs qualify (1-indexed)
#     npy_indices    — position of those patches in the npy file at time T

# Usage (run from F:\\nowcasting\\coalition4-rcnn):
#     python extract_sequences.py
#     python extract_sequences.py --hours 0 6
#     python extract_sequences.py --hours 0 24 --past 2 --future 3
#     python extract_sequences.py --output test_sequences.csv
# """

# import csv
# import os
# import argparse
# from datetime import datetime, timedelta


# # =============================================================================
# # Configuration
# # =============================================================================

# DEFAULT_DATA_ROOT = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'our_data'
# )

# N_PATCHES = 18
# STEP_MINUTES = 15

# # Default temporal window
# DEFAULT_PAST_STEPS = 2     # 2 × 15 min = 30 min before T
# DEFAULT_FUTURE_STEPS = 3   # 3 × 15 min = 45 min after T

# # Default hour range for reference timestep (test dataset)
# DEFAULT_HOUR_START = 0
# DEFAULT_HOUR_END = 6


# # =============================================================================
# # Patch index loader
# # =============================================================================

# def load_patch_index(data_root):
#     """
#     Load patch_index.csv into a structured dict.

#     Returns:
#         dict: {(date_str, time_str): [sorted list of active patch numbers]}
#               e.g. {('2025-05-15', '05:15'): [3, 4, 9, 10]}
#     """
#     csv_path = os.path.join(data_root, 'patch_index', 'patch_index.csv')
#     if not os.path.isfile(csv_path):
#         print(f"ERROR: patch_index.csv not found at {csv_path}")
#         print("Run identify_patches.py first.")
#         return {}

#     index = {}
#     with open(csv_path, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             date_str = row['date']
#             time_str = row['time_utc']

#             active = []
#             for p in range(1, N_PATCHES + 1):
#                 if row.get(f'patch_{p}', '0') == '1':
#                     active.append(p)

#             if active:
#                 index[(date_str, time_str)] = active

#     return index


# # =============================================================================
# # Sequence analysis
# # =============================================================================

# def find_continuous_sequences(index, hour_start, hour_end,
#                                past_steps, future_steps):
#     """
#     Find patches with temporal continuity within the specified hour window.

#     For each reference timestep T in [hour_start, hour_end):
#         - A patch qualifies if it is active at every step from
#           T - past_steps*15min  to  T + future_steps*15min (inclusive).
#         - The npy index at each timestep is the patch's position in the
#           active patches list at that specific timestep (0-indexed).

#     The window offsets are: [-past_steps, ..., -1, 0, +1, ..., +future_steps]
#     giving (past_steps + 1 + future_steps) timesteps total.

#     Args:
#         index: dict from load_patch_index()
#         hour_start, hour_end: UTC hour range for reference timestep
#         past_steps: number of 15-min steps required before T
#         future_steps: number of 15-min steps required after T

#     Returns:
#         list[dict]: One entry per valid (date, reference_time) pair.
#             'indices_per_step' is a list of length (past+1+future),
#             where each element is the list of npy indices for that step,
#             positionally matching 'patch_numbers'.
#     """
#     step = timedelta(minutes=STEP_MINUTES)
#     total_window = past_steps + 1 + future_steps
#     # Offsets relative to reference: [-past, ..., -1, 0, +1, ..., +future]
#     offsets = list(range(-past_steps, future_steps + 1))
#     results = []

#     # Collect all unique dates
#     dates = sorted(set(d for d, _ in index.keys()))

#     for date_str in dates:
#         base_dt = datetime.strptime(date_str, '%Y-%m-%d')
#         t = base_dt + timedelta(hours=hour_start)
#         t_end = base_dt + timedelta(hours=hour_end)

#         while t < t_end:
#             ref_time = t.strftime('%H:%M')

#             # Get active patches at reference timestep
#             active_at_ref = index.get((date_str, ref_time))
#             if not active_at_ref:
#                 t += step
#                 continue

#             # Collect active patches at every step in the window
#             window_active = {}  # offset → active list
#             for offset in offsets:
#                 check_dt = t + offset * step
#                 check_date = check_dt.strftime('%Y-%m-%d')
#                 check_time = check_dt.strftime('%H:%M')
#                 window_active[offset] = index.get(
#                     (check_date, check_time), []
#                 )

#             # Check each patch for continuity across all window steps
#             qualifying_patches = []
#             for patch_num in active_at_ref:
#                 if all(patch_num in window_active[o] for o in offsets):
#                     qualifying_patches.append(patch_num)

#             if qualifying_patches:
#                 # For each step in the window, find the npy index of each
#                 # qualifying patch within that step's active list
#                 indices_per_step = []
#                 for offset in offsets:
#                     step_active = window_active[offset]
#                     step_indices = [
#                         step_active.index(p) for p in qualifying_patches
#                     ]
#                     indices_per_step.append(step_indices)

#                 window_start = t - past_steps * step
#                 window_end = t + future_steps * step

#                 results.append({
#                     'date': date_str,
#                     'reference_utc': ref_time,
#                     'start_utc': window_start.strftime('%H:%M'),
#                     'end_utc': window_end.strftime('%H:%M'),
#                     'patch_numbers': qualifying_patches,
#                     'indices_per_step': indices_per_step,
#                     'n_qualifying': len(qualifying_patches),
#                     'n_total_active': len(active_at_ref),
#                 })

#             t += step

#     return results


# # =============================================================================
# # Output
# # =============================================================================

# def save_sequences(results, output_path, past_steps, future_steps):
#     """Save sequence results to CSV with per-timestep npy index columns."""
#     if not results:
#         print("No qualifying sequences found.")
#         return

#     # Build column names for each timestep in the window
#     step_columns = []
#     for offset in range(-past_steps, future_steps + 1):
#         if offset < 0:
#             step_columns.append(f'idx_t{offset * 15}')    # e.g. idx_t-30
#         elif offset == 0:
#             step_columns.append('idx_t0')
#         else:
#             step_columns.append(f'idx_t+{offset * 15}')   # e.g. idx_t+15

#     with open(output_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             'date', 'reference_utc', 'start_utc', 'end_utc',
#             'patch_numbers', *step_columns,
#             'n_qualifying', 'n_total_active'
#         ])

#         for r in results:
#             row = [
#                 r['date'],
#                 r['reference_utc'],
#                 r['start_utc'],
#                 r['end_utc'],
#                 str(r['patch_numbers']),
#             ]
#             # Add per-step index lists
#             for step_indices in r['indices_per_step']:
#                 row.append(str(step_indices))
#             row.extend([
#                 r['n_qualifying'],
#                 r['n_total_active'],
#             ])
#             writer.writerow(row)

#     print(f"  Saved: {output_path} ({len(results)} rows)")


# def print_summary(results, hour_start, hour_end, past_steps, future_steps):
#     """Print summary statistics."""
#     if not results:
#         print("No qualifying sequences found.")
#         return

#     dates = sorted(set(r['date'] for r in results))
#     total_samples = sum(r['n_qualifying'] for r in results)
#     total_rows = len(results)

#     # Patch frequency
#     patch_counts = {}
#     for r in results:
#         for p in r['patch_numbers']:
#             patch_counts[p] = patch_counts.get(p, 0) + 1

#     print(f"\n{'='*70}")
#     print(f"Summary")
#     print(f"{'='*70}")
#     print(f"  Hour range         : {hour_start:02d}:00 — {hour_end:02d}:00 UTC")
#     print(f"  Window             : {past_steps} past + current + "
#           f"{future_steps} future = {past_steps + 1 + future_steps} steps "
#           f"({(past_steps + 1 + future_steps) * STEP_MINUTES} min)")
#     print(f"  Dates              : {len(dates)}")
#     print(f"  Valid timesteps    : {total_rows}")
#     print(f"  Total patch samples: {total_samples}")
#     print(f"  Avg patches/step   : {total_samples / total_rows:.1f}")

#     if patch_counts:
#         print(f"\n  Patch frequency (how many timesteps each patch qualifies):")
#         for p in sorted(patch_counts.keys()):
#             print(f"    Patch {p:2d}: {patch_counts[p]:4d} times")


# # =============================================================================
# # CLI
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Extract continuous patch sequences from patch_index.csv "
#                     "for model training/testing."
#     )
#     parser.add_argument(
#         "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
#         help="Path to our_data directory"
#     )
#     parser.add_argument(
#         "--output", "-o", type=str, default=None,
#         help="Output CSV path (default: our_data/sequences.csv)"
#     )
#     parser.add_argument(
#         "--hours", nargs=2, type=int, default=[DEFAULT_HOUR_START, DEFAULT_HOUR_END],
#         metavar=('START', 'END'),
#         help=f"UTC hour range for reference timestep "
#              f"(default: {DEFAULT_HOUR_START} {DEFAULT_HOUR_END})"
#     )
#     parser.add_argument(
#         "--past", type=int, default=DEFAULT_PAST_STEPS,
#         help=f"Number of past 15-min steps required (default: {DEFAULT_PAST_STEPS})"
#     )
#     parser.add_argument(
#         "--future", type=int, default=DEFAULT_FUTURE_STEPS,
#         help=f"Number of future 15-min steps required (default: {DEFAULT_FUTURE_STEPS})"
#     )

#     args = parser.parse_args()

#     output_path = args.output or os.path.join(
#         args.data_root, 'sequences.csv'
#     )

#     print("=" * 70)
#     print("COALITION-4 Continuous Patch Sequence Extractor")
#     print("=" * 70)
#     print(f"Data root : {args.data_root}")
#     print(f"Hours     : {args.hours[0]:02d}:00 — {args.hours[1]:02d}:00 UTC")
#     print(f"Window    : {args.past} past + current + {args.future} future "
#           f"= {args.past + 1 + args.future} steps "
#           f"({(args.past + 1 + args.future) * STEP_MINUTES} min)")

#     # Load patch index
#     print("\nLoading patch index...")
#     index = load_patch_index(args.data_root)
#     if not index:
#         return

#     dates = sorted(set(d for d, _ in index.keys()))
#     print(f"  {len(index)} active timestamps across {len(dates)} dates")

#     # Find sequences
#     print("\nAnalyzing temporal continuity...")
#     results = find_continuous_sequences(
#         index, args.hours[0], args.hours[1], args.past, args.future
#     )

#     # Save
#     print("\nSaving results...")
#     out_dir = os.path.dirname(output_path)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)
#     save_sequences(results, output_path, args.past, args.future)

#     # Summary
#     print_summary(results, args.hours[0], args.hours[1], args.past, args.future)


# if __name__ == "__main__":
#     main()
    

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
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


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

    - source='radar': reads our_data/patch_index/patch_index.csv with the
      cadence from timestep_config.json (STEP_MINUTES).
    - source='lightning': reads our_data/lightning_periods/lightning_patches.csv
      with the cadence from lightning_periods_config.json (aggregation_minutes).
    """
    if source == 'radar':
        return (
            os.path.join(data_root, 'patch_index', 'patch_index.csv'),
            STEP_MINUTES,
        )
    if source == 'lightning':
        lp_dir = os.path.join(data_root, 'lightning_periods')
        cfg_path = os.path.join(lp_dir, 'lightning_periods_config.json')
        if not os.path.isfile(cfg_path):
            print(
                f"ERROR: {cfg_path} not found.\n"
                f"Run from the project root:\n"
                f"    python identify_lightning_periods.py",
                file=sys.stderr,
            )
            sys.exit(2)
        lp_cfg = json.loads(Path(cfg_path).read_text())
        return (
            os.path.join(lp_dir, 'lightning_patches.csv'),
            int(lp_cfg['aggregation_minutes']),
        )
    raise ValueError(f"Unknown source: {source}")


def load_patch_index(data_root, source='radar'):
    """
    Load the per-(date, time) patch-activity index for the chosen source.

    Returns:
        dict: {(date_str, time_str): [sorted list of active patch numbers]}
    """
    csv_path, _ = resolve_index_source(data_root, source)
    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found")
        if source == 'radar':
            print("Run identify_patches.py first.")
        else:
            print("Run identify_lightning_periods.py first.")
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
            STEP_MINUTES from timestep_config.json; pass aggregation_minutes
            when consuming a lightning_patches.csv index).
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
        "--source", type=str, default='radar',
        choices=['radar', 'lightning'],
        help="Activity source. 'radar' (default) reads patch_index.csv from "
             "identify_patches.py and uses step_minutes from "
             "timestep_config.json. 'lightning' reads lightning_patches.csv "
             "from identify_lightning_periods.py and uses aggregation_minutes "
             "from lightning_periods_config.json as the step interval."
    )

    args = parser.parse_args()

    # Resolve effective cadence + index path from the chosen source
    _, effective_step_minutes = resolve_index_source(args.data_root, args.source)

    train_frac = 1.0 - args.test_frac - args.val_frac
    n_blocks = 24 // args.block_hours if 24 % args.block_hours == 0 else '?'

    print("=" * 70)
    print("COALITION-4 Sequence Extractor (Czibula temporal block split)")
    print("=" * 70)
    print(f"Data root        : {args.data_root}")
    print(f"Activity source  : {args.source}")
    print(f"Step interval    : {effective_step_minutes} min")
    print(f"Window           : {args.past} past + current + {args.future} future "
          f"= {args.past + 1 + args.future} steps "
          f"({(args.past + 1 + args.future) * effective_step_minutes} min)")
    print(f"Blocks           : {args.block_hours}h ({n_blocks} blocks/day)")
    print(f"Split            : test {args.test_frac:.0%} / "
          f"val {args.val_frac:.0%} / "
          f"train {train_frac:.0%} per block")

    # Load patch index
    print(f"\nLoading {args.source} patch index...")
    index = load_patch_index(args.data_root, source=args.source)
    if not index:
        return

    dates = sorted(set(d for d, _ in index.keys()))
    print(f"  {len(index)} active timestamps across {len(dates)} dates")

    # Find all qualifying sequences (full 24h)
    print("\nAnalyzing temporal continuity (all 24h)...")
    all_sequences = find_all_sequences(
        index, args.past, args.future,
        step_minutes=effective_step_minutes,
    )
    print(f"  {len(all_sequences)} qualifying sequences found")

    if not all_sequences:
        print(f"No qualifying sequences. Check the {args.source} index CSV.")
        return

    # Split using Czibula method
    print("\nSplitting with Czibula temporal block method...")
    train, val, test = split_sequences_czibula(
        all_sequences, args.test_frac, args.val_frac, args.block_hours
    )
    print(f"  Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

    # Save three CSVs
    print("\nSaving results...")
    save_sequences(
        train,
        os.path.join(args.data_root, 'train_data.csv'),
        args.past, args.future
    )
    save_sequences(
        val,
        os.path.join(args.data_root, 'validation_data.csv'),
        args.past, args.future
    )
    save_sequences(
        test,
        os.path.join(args.data_root, 'test_data.csv'),
        args.past, args.future
    )

    # Drop a sidecar metadata file so create_datasets.py can recover step
    # minutes and window length without re-parsing column names.
    # `step_minutes` is the *effective* spacing between adjacent steps (i.e.
    # aggregation_minutes when source=lightning), so downstream consumers can
    # treat it uniformly. `source_step_minutes_native` is the configured
    # cadence from timestep_config.json — preserved for traceability.
    seq_meta = {
        "source": args.source,
        "step_minutes": effective_step_minutes,
        "source_step_minutes_native": STEP_MINUTES,
        "past_steps": args.past,
        "future_steps": args.future,
        "step_columns": [step_column_name(o)
                         for o in range(-args.past, args.future + 1)],
    }
    seq_meta_path = os.path.join(args.data_root, 'sequence_meta.json')
    with open(seq_meta_path, 'w') as f:
        json.dump(seq_meta, f, indent=2)
    print(f"  Saved sequence metadata -> {seq_meta_path}")

    # Summary
    print_summary(train, val, test, args.past, args.future, args.block_hours)


if __name__ == "__main__":
    main()
    