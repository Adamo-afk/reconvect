"""
COALITION-4 dataset statistics and visualizations.

Generates 6 diagnostic plots from patch_index.csv and sequences.csv:
    1. Diurnal cycle of convective activity
    2. Spatial heatmap of patch activation frequency
    3. Daily activity timeline (dates × hours)
    4. Distribution of simultaneously active patches
    5. Valid training samples per date (from sequences.csv)
    6. Patch survival rate (active vs qualifying for sequences)

All plots are saved to our_data/data_statistics/

Usage:
    python dataset_statistics.py
    python dataset_statistics.py -d F:/nowcasting/coalition4-rcnn/our_data
    python dataset_statistics.py --sequences my_sequences.csv
"""

import numpy as np
import csv
import os
import argparse
from collections import defaultdict, Counter
from datetime import datetime
import ast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)

N_PATCHES = 18
N_COLS = 6
N_ROWS = 3

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

def load_patch_index(data_root):
    """
    Load patch_index.csv.

    Returns:
        list[dict]: Each dict has 'date', 'time', 'hour', 'active_patches'
    """
    csv_path = os.path.join(data_root, 'patch_index', 'patch_index.csv')
    if not os.path.isfile(csv_path):
        print(f"  patch_index.csv not found at {csv_path}")
        return []

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            active = [
                p for p in range(1, N_PATCHES + 1)
                if row.get(f'patch_{p}', '0') == '1'
            ]
            if active:
                time_str = row['time_utc']
                hour = int(time_str.split(':')[0])
                rows.append({
                    'date': row['date'],
                    'time': time_str,
                    'hour': hour,
                    'minute': int(time_str.split(':')[1]),
                    'active_patches': active,
                    'n_active': len(active),
                })
    return rows


def load_sequences(seq_path):
    """
    Load sequences.csv.

    Returns:
        list[dict]: Each dict has 'date', 'reference_utc', 'patch_numbers',
                    'n_qualifying'
    """
    if not os.path.isfile(seq_path):
        print(f"  sequences.csv not found at {seq_path}")
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

def plot_daily_timeline(patch_data, out_dir):
    """
    Heatmap: dates (y) × hours (x), colored by number of active patches.
    Exposes data gaps, missing days, and convective clustering.
    """
    dates = sorted(set(r['date'] for r in patch_data))
    n_dates = len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    # Grid: (n_dates, 24 hours × 4 quarter-hours = 96 slots)
    grid = np.zeros((n_dates, 96))
    for row in patch_data:
        di = date_idx[row['date']]
        slot = row['hour'] * 4 + row['minute'] // 15
        grid[di, slot] = row['n_active']

    fig, ax = plt.subplots(figsize=(14, max(4, n_dates * 0.35)))
    cmap = LinearSegmentedColormap.from_list('act', ['#fafafa', '#bbdefb', '#1565c0'])
    im = ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Active patches', fontsize=10)

    # X-axis: show every 2 hours
    xticks = [h * 4 for h in range(0, 24, 2)]
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

def plot_patch_survival(patch_data, seq_data, out_dir, prefix='sequences'):
    """
    Grouped bar: how often each patch is active (patch_index) vs how often
    it qualifies for sequences (passes continuity check).
    Large gaps indicate transient, non-persistent convection.
    """
    # Count from patch_index
    active_counts = Counter()
    for row in patch_data:
        for p in row['active_patches']:
            active_counts[p] += 1

    # Count from sequences
    seq_counts = Counter()
    for row in seq_data:
        for p in row['patch_numbers']:
            seq_counts[p] += 1

    patches_present = sorted(
        set(active_counts.keys()) | set(seq_counts.keys())
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(patches_present))
    width = 0.35

    active_vals = [active_counts.get(p, 0) for p in patches_present]
    seq_vals = [seq_counts.get(p, 0) for p in patches_present]

    ax.bar(x - width / 2, active_vals, width, label='Active (patch_index)',
           color=COLORS['primary'], edgecolor='white', linewidth=0.5,
           alpha=0.85, zorder=3)
    ax.bar(x + width / 2, seq_vals, width, label='Qualifying (sequences)',
           color=COLORS['accent'], edgecolor='white', linewidth=0.5,
           alpha=0.85, zorder=3)

    # Survival rate labels
    for i, p in enumerate(patches_present):
        a = active_counts.get(p, 0)
        s = seq_counts.get(p, 0)
        rate = s / a * 100 if a > 0 else 0
        ax.text(x[i], max(a, s) + max(active_vals) * 0.02,
                f'{rate:.0f}%', ha='center', va='bottom',
                fontsize=7, color=COLORS['text'])

    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in patches_present], fontsize=9)
    ax.set_xlabel('Patch number', fontsize=11)
    ax.set_ylabel('Number of timesteps', fontsize=11)
    ax.set_title(f'Patch survival rate: active vs qualifying ({prefix})',
                 fontsize=13, fontweight='bold')
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
                    "and sequences.csv."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--sequences", "-s", type=str, default=None,
        help="Path to sequences.csv (default: our_data/sequences.csv)"
    )

    args = parser.parse_args()
    seq_path = args.sequences or os.path.join(args.data_root, 'sequences.csv')
    # If provided path doesn't exist, try resolving relative to data_root
    if args.sequences and not os.path.isfile(seq_path):
        alt_path = os.path.join(args.data_root, args.sequences)
        if os.path.isfile(alt_path):
            seq_path = alt_path
    out_dir = os.path.join(args.data_root, 'data_statistics')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("COALITION-4 Dataset Statistics")
    print("=" * 60)
    print(f"Data root : {args.data_root}")
    print(f"Output    : {out_dir}")

    # Load data
    print("\nLoading patch_index.csv...")
    patch_data = load_patch_index(args.data_root)
    if not patch_data:
        print("Cannot proceed without patch_index.csv")
        return

    dates = sorted(set(r['date'] for r in patch_data))
    print(f"  {len(patch_data)} active timesteps across {len(dates)} dates")

    print("\nLoading sequences.csv...")
    seq_data = load_sequences(seq_path)
    if seq_data:
        seq_dates = sorted(set(r['date'] for r in seq_data))
        print(f"  {len(seq_data)} valid sequences across {len(seq_dates)} dates")
    else:
        print("  Not found — plots 5 and 6 will be skipped")

    # Generate plots
    print("\nGenerating plots...")

    plot_diurnal_cycle(patch_data, out_dir)
    plot_spatial_heatmap(patch_data, out_dir)
    plot_daily_timeline(patch_data, out_dir)
    plot_active_distribution(patch_data, out_dir)

    if seq_data:
        # Derive prefix from sequences filename (e.g. "train_data.csv" → "train_data")
        seq_prefix = os.path.splitext(os.path.basename(seq_path))[0]
        plot_samples_per_date(seq_data, out_dir, seq_prefix)
        plot_patch_survival(patch_data, seq_data, out_dir, seq_prefix)

    print(f"\nDone. All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
    