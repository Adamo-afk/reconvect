"""
COALITION-4 dataset statistics and visualizations.

Generates 6 diagnostic plots derived from the per-source split CSVs
(train_data_<source>.csv, validation_data_<source>.csv,
test_data_<source>.csv). All three splits are loaded so the
overview plots (1-4) reflect the full dataset; the split-specific
plots (5, 6) only describe the --split passed on the CLI.

    1. Diurnal cycle of convective activity                 (all splits)
    2. Spatial heatmap of patch activation frequency        (all splits)
    3. Daily activity timeline (dates × hours)              (all splits)
    4. Distribution of simultaneously active patches        (all splits)
    5. Valid training samples per date                      (--split)
    6. Per-patch active vs qualifying + overall survival    (all + --split)

All plots are saved to our_data/data_statistics/

Usage:
    # Default: DBSCAN-driven train split
    python data_statistics.py

    # Lightning-driven train split
    python data_statistics.py --source lightning

    # Validation / test for either track
    python data_statistics.py --source dbscan --split validation
    python data_statistics.py --source lightning --split test

    # Explicit override (any CSV with the per-source split schema)
    python data_statistics.py --csv our_data/train_data_dbscan.csv
"""

import numpy as np
import csv
import json
import os
import argparse
from collections import defaultdict, Counter
from datetime import datetime
import ast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from periods import sequence_meta_name, split_csv_name
from pipeline_config import SOURCE


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)

N_PATCHES = 18
N_COLS = 6
N_ROWS = 3


def _load_step_minutes(data_root, source="dbscan", period=None):
    """Read the cadence (`step_minutes`) used to build the per-source
    split CSVs.

    Looks in this order:
      1. `our_data/sequence_meta_<source>.json` (canonical, recorded
         alongside the train/val/test CSVs). For the lightning track
         it carries both the aggregation `step_minutes` and the native
         `source_step_minutes_native`; we prefer the native value
         because the daily-timeline plot is laid out on the
         per-timestep grid before any aggregation.
      2. `our_data/timestep_config.json` (master cadence from
         `validate_timestep.py`).
      3. 15 minutes (historical hardcoded default).
    """
    candidates = [
        os.path.join(data_root, sequence_meta_name(source, period)),
        os.path.join(data_root, "timestep_config.json"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r') as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        val = (cfg.get('source_step_minutes_native')
               or cfg.get('step_minutes'))
        if val:
            return int(val)
    return 15

COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'accent': '#4CAF50',
    'grid': '#e0e0e0',
    'text': '#333333',
}


# =============================================================================
# Data loaders
# =============================================================================

def load_sequences(seq_path):
    """
    Load a per-source split CSV (train_data_<source>.csv /
    validation_data_<source>.csv / test_data_<source>.csv).

    Returns:
        list[dict]: each dict has 'date', 'reference_utc', 'patch_numbers',
                    'n_qualifying', 'n_total_active'.
    """
    if not os.path.isfile(seq_path):
        print(f"  sequence CSV not found at {seq_path}")
        return []

    rows = []
    with open(seq_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patches = ast.literal_eval(row['patch_numbers'])
            rows.append({
                'date': row['date'],
                'reference_utc': row['reference_utc'],
                'patch_numbers': patches,
                'n_qualifying': int(row['n_qualifying']),
                'n_total_active': int(row['n_total_active']),
            })
    return rows


def patch_activity_from_sequences(seq_data):
    """Derive per-(date, time) patch-activity rows from the sequence CSV.

    Each sequence row's `patch_numbers` IS the activity ground truth
    for what the model trains on at that reference timestep, so
    plots 1-4 build directly from this rather than the upstream
    patch_index.csv. Consequence: the
    diagnostics always reflect the split CSV on disk (no risk of a
    stale upstream file showing one date while the training CSV
    spans eighty).
    """
    rows = []
    for r in seq_data:
        time_str = r["reference_utc"].strip()
        active = r["patch_numbers"]
        if not active:
            continue
        parts = time_str.split(":")
        rows.append({
            "date":           r["date"],
            "time":           time_str,
            "hour":           int(parts[0]),
            "minute":         int(parts[1]),
            "active_patches": active,
            "n_active":       len(active),
        })
    return rows


# =============================================================================
# Plot 1: Diurnal cycle
# =============================================================================

def plot_diurnal_cycle(patch_data, out_dir):
    """
    Line plot: average number of active timesteps per hour of day.
    Shows the convective diurnal cycle and whether the test window
    (00-06 UTC) is representative.
    """
    # Count active timesteps per hour per date
    hour_date_counts = defaultdict(lambda: defaultdict(int))
    for row in patch_data:
        hour_date_counts[row['hour']][row['date']] += 1

    hours = list(range(24))
    dates = sorted(set(r['date'] for r in patch_data))
    n_dates = len(dates)

    # Mean and std across dates
    means = []
    stds = []
    for h in hours:
        counts = [hour_date_counts[h].get(d, 0) for d in dates]
        means.append(np.mean(counts))
        stds.append(np.std(counts))

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, means, color=COLORS['primary'], linewidth=2, zorder=3)
    ax.fill_between(hours, means - stds, means + stds,
                    color=COLORS['primary'], alpha=0.15, zorder=2)

    ax.set_xlabel('Hour (UTC)', fontsize=11)
    ax.set_ylabel('Active timesteps per hour (mean ± std)', fontsize=11)
    ax.set_title('Diurnal cycle of convective activity', fontsize=13, fontweight='bold')
    ax.set_xticks(hours)
    ax.set_xlim(0, 23)
    ax.grid(axis='y', alpha=0.3)

    save_path = os.path.join(out_dir, '1_diurnal_cycle.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Plot 2: Spatial heatmap
# =============================================================================

def plot_spatial_heatmap(patch_data, out_dir):
    """
    6×3 grid heatmap: how often each patch is active across all data.
    Reveals geographic bias in convective activity.
    """
    patch_counts = Counter()
    for row in patch_data:
        for p in row['active_patches']:
            patch_counts[p] += 1

    total_timesteps = len(patch_data)
    grid = np.zeros((N_ROWS, N_COLS))
    for p in range(1, N_PATCHES + 1):
        r = (p - 1) // N_COLS
        c = (p - 1) % N_COLS
        grid[r, c] = patch_counts.get(p, 0) / total_timesteps * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = LinearSegmentedColormap.from_list('conv', ['#f5f5f5', '#ffcc80', '#ef5350'])
    im = ax.imshow(grid, cmap=cmap, aspect='equal', vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Active (% of timesteps)', fontsize=10)

    for p in range(1, N_PATCHES + 1):
        r = (p - 1) // N_COLS
        c = (p - 1) % N_COLS
        val = grid[r, c]
        text_color = 'white' if val > grid.max() * 0.6 else '#333333'
        ax.text(c, r, f'{p}\n{val:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color)

    ax.set_xticks(range(N_COLS))
    ax.set_xticklabels([f'C{i+1}' for i in range(N_COLS)])
    ax.set_yticks(range(N_ROWS))
    ax.set_yticklabels([f'R{i+1}' for i in range(N_ROWS)])
    ax.set_title('Patch activation frequency (% of active timesteps)',
                 fontsize=13, fontweight='bold')

    save_path = os.path.join(out_dir, '2_spatial_heatmap.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Plot 3: Daily activity timeline
# =============================================================================

def plot_daily_timeline(patch_data, out_dir, step_minutes=15):
    """
    Heatmap: dates (y) × hours (x), colored by number of active patches.
    Exposes data gaps, missing days, and convective clustering.

    The grid resolution adapts to `step_minutes` so each row of
    `patch_index.csv` maps to exactly one slot — previously the 96-slot
    grid was hardcoded for a 15-min cadence and silently aliased entries
    at finer cadences (e.g. :00 and :10 both wrote to slot 0).
    """
    dates = sorted(set(r['date'] for r in patch_data))
    n_dates = len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    # 1440 min / step_minutes slots per day. For step=15 -> 96 slots; for
    # step=10 -> 144 slots; for step=30 -> 48 slots.
    slots_per_day = 24 * 60 // step_minutes
    grid = np.zeros((n_dates, slots_per_day))
    for row in patch_data:
        di = date_idx[row['date']]
        slot = (row['hour'] * 60 + row['minute']) // step_minutes
        if 0 <= slot < slots_per_day:
            grid[di, slot] = row['n_active']

    fig, ax = plt.subplots(figsize=(14, max(4, n_dates * 0.35)))
    cmap = LinearSegmentedColormap.from_list('act', ['#fafafa', '#bbdefb', '#1565c0'])
    im = ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Active patches', fontsize=10)

    # X-axis: show every 2 hours — slot index per hour scales with cadence.
    slots_per_hour = 60 // step_minutes
    xticks = [h * slots_per_hour for h in range(0, 24, 2)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], fontsize=8)

    ax.set_yticks(range(n_dates))
    ax.set_yticklabels(dates, fontsize=7)
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.set_ylabel('Date', fontsize=11)
    ax.set_title('Daily convective activity timeline',
                 fontsize=13, fontweight='bold')

    save_path = os.path.join(out_dir, '3_daily_timeline.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Plot 4: Simultaneously active patches distribution
# =============================================================================

def plot_active_distribution(patch_data, out_dir):
    """
    Histogram: how many patches are simultaneously active per timestep.
    Shows whether convection is typically localized or widespread.
    """
    counts = [row['n_active'] for row in patch_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(1, max(counts) + 2)
    ax.hist(counts, bins=bins, color=COLORS['primary'], edgecolor='white',
            linewidth=0.5, alpha=0.85, align='left', zorder=3)

    mean_val = np.mean(counts)
    median_val = np.median(counts)
    ax.axvline(mean_val, color=COLORS['secondary'], linestyle='--',
               linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color=COLORS['accent'], linestyle=':',
               linewidth=1.5, label=f'Median: {median_val:.0f}')

    ax.set_xlabel('Number of simultaneously active patches', fontsize=11)
    ax.set_ylabel('Number of timesteps', fontsize=11)
    ax.set_title('Distribution of active patches per timestep',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    save_path = os.path.join(out_dir, '4_active_distribution.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Plot 5: Valid training samples per date
# =============================================================================

def plot_samples_per_date(seq_data, out_dir, prefix='sequences'):
    """
    Bar plot: how many qualifying sequences each date contributes.
    Reveals if a few dates dominate the training set.
    """
    date_counts = Counter(r['date'] for r in seq_data)
    dates = sorted(date_counts.keys())
    counts = [date_counts[d] for d in dates]

    fig, ax = plt.subplots(figsize=(max(8, len(dates) * 0.6), 5))
    x = range(len(dates))
    ax.bar(x, counts, color=COLORS['accent'], edgecolor='white',
           linewidth=0.5, alpha=0.85, zorder=3)

    mean_val = np.mean(counts)
    ax.axhline(mean_val, color=COLORS['secondary'], linestyle='--',
               linewidth=1.2, label=f'Mean: {mean_val:.1f}')

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Valid sequences', fontsize=11)
    ax.set_title(f'Valid samples per date ({prefix})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    save_path = os.path.join(out_dir, f'5_samples_per_date_{prefix}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Plot 6: Patch survival rate
# =============================================================================

def plot_patch_survival(seq_all, seq_split, out_dir,
                        prefix='sequences', split_label='train'):
    """
    Per-patch active-vs-qualifying diagnostic.

    Restored two-bar shape (active blue, qualifying green) without
    going back to the upstream patch_index.csv. The two scopes are:

      - Blue (`Active`): per-patch occurrence count summed over ALL
        per-source split CSVs (train + validation + test combined).
        Tracks how often each patch survived the cross-product filter
        anywhere in the dataset - a sequence-CSV-side stand-in for the
        old "DBSCAN-active timesteps" denominator.
      - Green (`Qualifying`): per-patch occurrence count in JUST the
        --split CSV (the one passed via --split / --sequences).

    The printed percentage is the split's share of total dataset
    qualification for that patch:
        rate = seq_split_count[p] / seq_all_count[p] * 100
    e.g. ~80% on a patch the Czibula 80/10/10 split puts mostly in
    train; ~10% on val/test patches.

    Overall-survival annotation in the title still uses
    sum(n_qualifying) / sum(n_total_active) computed on seq_split so
    the "what fraction of initially-active patches passed the window
    filter" number stays anchored to the chosen split.

    All 18 patches are always listed for layout consistency, including
    those that never qualified (e.g. patches over the Black Sea or
    outside the OPERA footprint) - they show as zero-height bars.
    """
    all_counts = Counter()
    for row in seq_all:
        for p in row['patch_numbers']:
            all_counts[p] += 1

    split_counts = Counter()
    for row in seq_split:
        for p in row['patch_numbers']:
            split_counts[p] += 1

    total_active = sum(r['n_total_active'] for r in seq_split)
    total_qualifying = sum(r['n_qualifying'] for r in seq_split)
    overall_survival = (
        (total_qualifying / total_active * 100.0) if total_active > 0 else 0.0
    )

    patches_present = list(range(1, N_PATCHES + 1))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(patches_present))
    width = 0.35

    all_vals = [all_counts.get(p, 0) for p in patches_present]
    split_vals = [split_counts.get(p, 0) for p in patches_present]

    ax.bar(x - width / 2, all_vals, width,
           label='Active (train + val + test)',
           color=COLORS['primary'], edgecolor='white', linewidth=0.5,
           alpha=0.85, zorder=3)
    ax.bar(x + width / 2, split_vals, width,
           label=f'Qualifying ({split_label} split)',
           color=COLORS['accent'], edgecolor='white', linewidth=0.5,
           alpha=0.85, zorder=3)

    max_h = max(all_vals + split_vals + [1])
    for i, p in enumerate(patches_present):
        a = all_counts.get(p, 0)
        s = split_counts.get(p, 0)
        rate = s / a * 100 if a > 0 else 0
        ax.text(x[i], max(a, s) + max_h * 0.02,
                f'{rate:.0f}%', ha='center', va='bottom',
                fontsize=7, color=COLORS['text'])

    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in patches_present], fontsize=9)
    ax.set_xlabel('Patch number (1..18, row-major over the 6x3 grid)',
                  fontsize=11)
    ax.set_ylabel('Sequence rows where this patch qualified',
                  fontsize=11)
    ax.set_title(
        f'Patch survival: {split_label} share of total qualification '
        f'({prefix})  |  {split_label} window-filter survival = '
        f'{total_qualifying:,} / {total_active:,} patch-timesteps  '
        f'({overall_survival:.1f}%)',
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    save_path = os.path.join(out_dir, f'6_patch_survival_{prefix}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from patch_index.csv "
                    "and the per-source split CSVs written by "
                    "extract_patch_seq_for_datasets.py."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test"],
        help="Which split to plot when --csv is auto-resolved "
             "from --source. Ignored when --csv is given "
             "explicitly. Default: train."
    )
    parser.add_argument(
        "--period", "-p", type=str, default=None, metavar="LABEL",
        help="Describe a period's split instead of the whole archive: "
             "--period w34 reads <split>_data_<source>_w34.csv and the "
             "matching sequence_meta. Ignored when --csv is given "
             "explicitly. Omit for the untagged whole-archive split."
    )
    parser.add_argument(
        "--csv", "-c", type=str, default=None,
        help="Explicit path to a per-source split CSV "
             "(train_data_<source>.csv / validation_data_<source>.csv / "
             "test_data_<source>.csv). Overrides --source / --split. "
             "Default: <data_root>/<split>_data_<source>.csv."
    )

    args = parser.parse_args()
    if args.csv is not None:
        seq_path = args.csv
        # If provided path doesn't exist, try resolving relative to data_root
        if not os.path.isfile(seq_path):
            alt_path = os.path.join(args.data_root, args.csv)
            if os.path.isfile(alt_path):
                seq_path = alt_path
    else:
        seq_path = os.path.join(
            args.data_root,
            split_csv_name(args.split, SOURCE, args.period),
        )
    out_dir = os.path.join(args.data_root, 'data_statistics')
    if args.period:
        out_dir = os.path.join(out_dir, args.period)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("COALITION-4 Dataset Statistics")
    print("=" * 60)
    print(f"Data root : {args.data_root}")
    print(f"Source    : {SOURCE}  (split={args.split})")
    print(f"Sequences : {seq_path}")
    print(f"Output    : {out_dir}")

    # Load the --split CSV first. This is the canonical source for
    # plots 5 + 6 ("samples per date" and "patch survival") - they
    # only describe the split the user picked. Plots 1-4 then union
    # all three split CSVs (train + validation + test) so the
    # diurnal / spatial / timeline / active-count diagnostics
    # describe the full dataset rather than just one split.
    print(f"\nLoading split CSV ({os.path.basename(seq_path)})...")
    seq_split = load_sequences(seq_path)
    if not seq_split:
        print(f"Cannot proceed without {seq_path}")
        return
    split_dates = sorted(set(r['date'] for r in seq_split))
    print(f"  {len(seq_split)} sequences across {len(split_dates)} dates")

    seq_all = list(seq_split)
    if args.csv is None:
        # Auto-resolved seq_path: try to also load the other two splits
        # so the overview plots reflect the full dataset. Missing files
        # are skipped with a note - useful for partial setups (e.g. a
        # dev machine that only has train).
        for other in ("train", "validation", "test"):
            if other == args.split:
                continue
            other_name = split_csv_name(other, SOURCE, args.period)
            other_path = os.path.join(args.data_root, other_name)
            if not os.path.isfile(other_path):
                print(f"  ({other_name} not found - "
                      f"plots 1-4 will skip it)")
                continue
            extra = load_sequences(other_path)
            if extra:
                seq_all.extend(extra)
                print(f"  + {len(extra)} sequences from {other}")

    patch_data = patch_activity_from_sequences(seq_all)
    activity_dates = sorted(set(r['date'] for r in patch_data))
    print(f"  Combined patch-activity: {len(patch_data)} active "
          f"timesteps across {len(activity_dates)} dates")

    # Cadence is read from sequence_meta_<source>.json (preferred) or
    # timestep_config.json (fallback) so the daily-timeline grid sizes
    # itself correctly for any --step_minutes validate_timestep.py was
    # run with.
    step_minutes = _load_step_minutes(args.data_root, SOURCE)
    print(f"  step_minutes: {step_minutes} (from config)")

    # Generate plots
    print("\nGenerating plots...")

    plot_diurnal_cycle(patch_data, out_dir)
    plot_spatial_heatmap(patch_data, out_dir)
    plot_daily_timeline(patch_data, out_dir, step_minutes=step_minutes)
    plot_active_distribution(patch_data, out_dir)

    seq_prefix = os.path.splitext(os.path.basename(seq_path))[0]
    plot_samples_per_date(seq_split, out_dir, seq_prefix)
    plot_patch_survival(seq_all, seq_split, out_dir,
                        prefix=seq_prefix, split_label=args.split)

    print(f"\nDone. All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
    