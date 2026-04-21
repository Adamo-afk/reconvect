"""
COALITION-4 continuous patch sequence extractor.

Analyzes patch_index.csv to find patches with temporal continuity:
each qualifying patch must be active for at least 6 consecutive 15-minute
steps (2 past + current + 3 future) around the reference timestep.

The output tells the training/testing pipeline exactly which npy array
indices to use for each valid sample.

Window structure (15-min steps):
    T-30  T-15   T   T+15  T+30  T+45
     ↑     ↑     ↑     ↑     ↑     ↑
    past  past  curr  fut   fut   fut

Output columns:
    date           — YYYY-MM-DD
    reference_utc  — HH:MM (the "current" timestep T)
    start_utc      — HH:MM (T - 30 min)
    end_utc        — HH:MM (T + 45 min)
    patch_numbers  — which patch IDs qualify (1-indexed)
    npy_indices    — position of those patches in the npy file at time T

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python extract_sequences.py
    python extract_sequences.py --hours 0 6
    python extract_sequences.py --hours 0 24 --past 2 --future 3
    python extract_sequences.py --output test_sequences.csv
"""

import csv
import os
import argparse
from datetime import datetime, timedelta


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)

N_PATCHES = 18
STEP_MINUTES = 15

# Default temporal window
DEFAULT_PAST_STEPS = 2     # 2 × 15 min = 30 min before T
DEFAULT_FUTURE_STEPS = 3   # 3 × 15 min = 45 min after T

# Default hour range for reference timestep (test dataset)
DEFAULT_HOUR_START = 0
DEFAULT_HOUR_END = 6


# =============================================================================
# Patch index loader
# =============================================================================

def load_patch_index(data_root):
    """
    Load patch_index.csv into a structured dict.

    Returns:
        dict: {(date_str, time_str): [sorted list of active patch numbers]}
              e.g. {('2025-05-15', '05:15'): [3, 4, 9, 10]}
    """
    csv_path = os.path.join(data_root, 'patch_index', 'patch_index.csv')
    if not os.path.isfile(csv_path):
        print(f"ERROR: patch_index.csv not found at {csv_path}")
        print("Run identify_patches.py first.")
        return {}

    index = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row['date']
            time_str = row['time_utc']

            active = []
            for p in range(1, N_PATCHES + 1):
                if row.get(f'patch_{p}', '0') == '1':
                    active.append(p)

            if active:
                index[(date_str, time_str)] = active

    return index


# =============================================================================
# Sequence analysis
# =============================================================================

def find_continuous_sequences(index, hour_start, hour_end,
                               past_steps, future_steps):
    """
    Find patches with temporal continuity within the specified hour window.

    For each reference timestep T in [hour_start, hour_end):
        - A patch qualifies if it is active at every step from
          T - past_steps*15min  to  T + future_steps*15min (inclusive).
        - The npy index at each timestep is the patch's position in the
          active patches list at that specific timestep (0-indexed).

    The window offsets are: [-past_steps, ..., -1, 0, +1, ..., +future_steps]
    giving (past_steps + 1 + future_steps) timesteps total.

    Args:
        index: dict from load_patch_index()
        hour_start, hour_end: UTC hour range for reference timestep
        past_steps: number of 15-min steps required before T
        future_steps: number of 15-min steps required after T

    Returns:
        list[dict]: One entry per valid (date, reference_time) pair.
            'indices_per_step' is a list of length (past+1+future),
            where each element is the list of npy indices for that step,
            positionally matching 'patch_numbers'.
    """
    step = timedelta(minutes=STEP_MINUTES)
    total_window = past_steps + 1 + future_steps
    # Offsets relative to reference: [-past, ..., -1, 0, +1, ..., +future]
    offsets = list(range(-past_steps, future_steps + 1))
    results = []

    # Collect all unique dates
    dates = sorted(set(d for d, _ in index.keys()))

    for date_str in dates:
        base_dt = datetime.strptime(date_str, '%Y-%m-%d')
        t = base_dt + timedelta(hours=hour_start)
        t_end = base_dt + timedelta(hours=hour_end)

        while t < t_end:
            ref_time = t.strftime('%H:%M')

            # Get active patches at reference timestep
            active_at_ref = index.get((date_str, ref_time))
            if not active_at_ref:
                t += step
                continue

            # Collect active patches at every step in the window
            window_active = {}  # offset → active list
            for offset in offsets:
                check_dt = t + offset * step
                check_date = check_dt.strftime('%Y-%m-%d')
                check_time = check_dt.strftime('%H:%M')
                window_active[offset] = index.get(
                    (check_date, check_time), []
                )

            # Check each patch for continuity across all window steps
            qualifying_patches = []
            for patch_num in active_at_ref:
                if all(patch_num in window_active[o] for o in offsets):
                    qualifying_patches.append(patch_num)

            if qualifying_patches:
                # For each step in the window, find the npy index of each
                # qualifying patch within that step's active list
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
                })

            t += step

    return results


# =============================================================================
# Output
# =============================================================================

def save_sequences(results, output_path, past_steps, future_steps):
    """Save sequence results to CSV with per-timestep npy index columns."""
    if not results:
        print("No qualifying sequences found.")
        return

    # Build column names for each timestep in the window
    step_columns = []
    for offset in range(-past_steps, future_steps + 1):
        if offset < 0:
            step_columns.append(f'idx_t{offset * 15}')    # e.g. idx_t-30
        elif offset == 0:
            step_columns.append('idx_t0')
        else:
            step_columns.append(f'idx_t+{offset * 15}')   # e.g. idx_t+15

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
            # Add per-step index lists
            for step_indices in r['indices_per_step']:
                row.append(str(step_indices))
            row.extend([
                r['n_qualifying'],
                r['n_total_active'],
            ])
            writer.writerow(row)

    print(f"  Saved: {output_path} ({len(results)} rows)")


def print_summary(results, hour_start, hour_end, past_steps, future_steps):
    """Print summary statistics."""
    if not results:
        print("No qualifying sequences found.")
        return

    dates = sorted(set(r['date'] for r in results))
    total_samples = sum(r['n_qualifying'] for r in results)
    total_rows = len(results)

    # Patch frequency
    patch_counts = {}
    for r in results:
        for p in r['patch_numbers']:
            patch_counts[p] = patch_counts.get(p, 0) + 1

    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"  Hour range         : {hour_start:02d}:00 — {hour_end:02d}:00 UTC")
    print(f"  Window             : {past_steps} past + current + "
          f"{future_steps} future = {past_steps + 1 + future_steps} steps "
          f"({(past_steps + 1 + future_steps) * STEP_MINUTES} min)")
    print(f"  Dates              : {len(dates)}")
    print(f"  Valid timesteps    : {total_rows}")
    print(f"  Total patch samples: {total_samples}")
    print(f"  Avg patches/step   : {total_samples / total_rows:.1f}")

    if patch_counts:
        print(f"\n  Patch frequency (how many timesteps each patch qualifies):")
        for p in sorted(patch_counts.keys()):
            print(f"    Patch {p:2d}: {patch_counts[p]:4d} times")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract continuous patch sequences from patch_index.csv "
                    "for model training/testing."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output CSV path (default: our_data/sequences.csv)"
    )
    parser.add_argument(
        "--hours", nargs=2, type=int, default=[DEFAULT_HOUR_START, DEFAULT_HOUR_END],
        metavar=('START', 'END'),
        help=f"UTC hour range for reference timestep "
             f"(default: {DEFAULT_HOUR_START} {DEFAULT_HOUR_END})"
    )
    parser.add_argument(
        "--past", type=int, default=DEFAULT_PAST_STEPS,
        help=f"Number of past 15-min steps required (default: {DEFAULT_PAST_STEPS})"
    )
    parser.add_argument(
        "--future", type=int, default=DEFAULT_FUTURE_STEPS,
        help=f"Number of future 15-min steps required (default: {DEFAULT_FUTURE_STEPS})"
    )

    args = parser.parse_args()

    output_path = args.output or os.path.join(
        args.data_root, 'sequences.csv'
    )

    print("=" * 70)
    print("COALITION-4 Continuous Patch Sequence Extractor")
    print("=" * 70)
    print(f"Data root : {args.data_root}")
    print(f"Hours     : {args.hours[0]:02d}:00 — {args.hours[1]:02d}:00 UTC")
    print(f"Window    : {args.past} past + current + {args.future} future "
          f"= {args.past + 1 + args.future} steps "
          f"({(args.past + 1 + args.future) * STEP_MINUTES} min)")

    # Load patch index
    print("\nLoading patch index...")
    index = load_patch_index(args.data_root)
    if not index:
        return

    dates = sorted(set(d for d, _ in index.keys()))
    print(f"  {len(index)} active timestamps across {len(dates)} dates")

    # Find sequences
    print("\nAnalyzing temporal continuity...")
    results = find_continuous_sequences(
        index, args.hours[0], args.hours[1], args.past, args.future
    )

    # Save
    print("\nSaving results...")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_sequences(results, output_path, args.past, args.future)

    # Summary
    print_summary(results, args.hours[0], args.hours[1], args.past, args.future)


if __name__ == "__main__":
    main()
    