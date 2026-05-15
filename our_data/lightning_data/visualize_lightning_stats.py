"""
Lightning activity statistics — two bar plots and a CSV.

Plot 1 — Per day:
    For each date, count how many of the 96 time steps have at least one
    non-zero pixel, grouped by product (density, current, occurrence).

Plot 2 — Per time step:
    For each HHMM across all dates, count how many days have activity at
    that time step, grouped by product.

CSV — Active steps:
    One row per (date, time) pair that has activity in at least one product.
    Columns: date, time_utc, density, current, occurrence (1/0 flags).

Input structure (COALITION-4):
    {data_root}/{product}/nc4_{date}-Romania_{product}/
        lightning_{product}_YYYYMMDD_HHMM.npy

Output:
    {output_dir}/lightning_activity_per_day.png
    {output_dir}/lightning_activity_per_timestep.png
    {output_dir}/lightning_active_steps.csv

Usage:
    python lightning_activity_stats.py
    python lightning_activity_stats.py -d F:/nowcasting/.../lightning_data -o F:/figures
"""

import os
import re
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = r"F:\nowcasting\coalition4-rcnn\our_data\lightning_data"

PRODUCTS = {
    "density":    {"prefix": "lightning_density",    "label": "Density"},
    "current":    {"prefix": "lightning_current",    "label": "Current"},
    "occurrence": {"prefix": "lightning_occurrence", "label": "Occurrence"},
}

# Regex for date directories: nc4_YYYY-MM-DD-Romania_{product}
DATE_DIR_PATTERN = re.compile(r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_')

# Regex for filenames: lightning_{product}_YYYYMMDD_HHMM.npy
# (read_kml_version2.py writes .npy directly — see its docstring).
FILE_PATTERN = re.compile(r'^lightning_\w+_(\d{8})_(\d{4})\.npy$')


# =============================================================================
# Scanning
# =============================================================================

def check_file_has_activity(filepath):
    """
    Check if a lightning `.npy` file has at least one non-zero pixel.

    Returns True if any pixel in the array is != 0.
    """
    try:
        data = np.load(filepath)
        return bool(np.any(data != 0))
    except Exception as e:
        print(f"  WARNING: Could not read {filepath}: {e}")
        return False


def scan_product(data_root, product_name):
    """
    Scan all files for one product and determine activity per (date, time).

    Args:
        data_root (str): Root lightning data directory
        product_name (str): Product directory name (density/current/occurrence)

    Returns:
        dict: {date_str: {time_str: bool}} — True if active
    """
    product_dir = os.path.join(data_root, product_name)
    if not os.path.isdir(product_dir):
        print(f"  Product directory not found: {product_dir}")
        return {}

    activity = defaultdict(dict)  # {date: {HHMM: bool}}

    # Iterate over date directories
    for entry in sorted(os.listdir(product_dir)):
        match = DATE_DIR_PATTERN.match(entry)
        if not match:
            continue

        date_str = match.group(1)
        day_dir = os.path.join(product_dir, entry)

        if not os.path.isdir(day_dir):
            continue

        npy_files = sorted(f for f in os.listdir(day_dir) if f.endswith('.npy'))

        for npy_file in npy_files:
            fmatch = FILE_PATTERN.match(npy_file)
            if not fmatch:
                continue

            time_str = fmatch.group(2)  # HHMM
            filepath = os.path.join(day_dir, npy_file)

            active = check_file_has_activity(filepath)
            activity[date_str][time_str] = active

    return dict(activity)


def scan_all_products(data_root):
    """
    Scan all 3 products and return activity dictionaries.

    Returns:
        dict: {product_name: {date_str: {time_str: bool}}}
    """
    all_activity = {}

    for product_name in PRODUCTS:
        print(f"Scanning {product_name}...")
        activity = scan_product(data_root, product_name)
        n_dates = len(activity)
        n_files = sum(len(times) for times in activity.values())
        n_active = sum(
            sum(1 for v in times.values() if v)
            for times in activity.values()
        )
        print(f"  {n_dates} dates, {n_files} files, {n_active} active steps")
        all_activity[product_name] = activity

    return all_activity


# =============================================================================
# Statistics
# =============================================================================

def compute_per_day_stats(all_activity):
    """
    For each date, count active time steps per product.

    Returns:
        dates (sorted list), stats (dict: product -> list of counts)
    """
    all_dates = set()
    for product_activity in all_activity.values():
        all_dates.update(product_activity.keys())
    dates = sorted(all_dates)

    stats = {}
    for product_name in PRODUCTS:
        counts = []
        for date_str in dates:
            day_data = all_activity.get(product_name, {}).get(date_str, {})
            active_count = sum(1 for v in day_data.values() if v)
            counts.append(active_count)
        stats[product_name] = counts

    return dates, stats


def compute_per_timestep_stats(all_activity):
    """
    For each HHMM, count how many days have activity per product.

    Returns:
        times (sorted list of HHMM), stats (dict: product -> list of counts)
    """
    all_times = set()
    for product_activity in all_activity.values():
        for day_data in product_activity.values():
            all_times.update(day_data.keys())
    times = sorted(all_times)

    # Total number of dates per product (for reference)
    n_dates = {}
    for product_name in PRODUCTS:
        n_dates[product_name] = len(all_activity.get(product_name, {}))

    stats = {}
    for product_name in PRODUCTS:
        counts = []
        for time_str in times:
            active_days = 0
            for date_str, day_data in all_activity.get(product_name, {}).items():
                if day_data.get(time_str, False):
                    active_days += 1
            counts.append(active_days)
        stats[product_name] = counts

    return times, stats, n_dates


# =============================================================================
# Plotting
# =============================================================================

COLORS = {
    "density":    "#2196F3",  # blue
    "current":    "#FF9800",  # orange
    "occurrence": "#4CAF50",  # green
}


def plot_per_day(dates, stats, save_path):
    """
    Bar plot: active time steps per day, grouped by product.
    """
    n_dates = len(dates)
    if n_dates == 0:
        print("No data to plot (per day).")
        return

    x = np.arange(n_dates)
    n_products = len(PRODUCTS)
    width = 0.8 / n_products

    fig, ax = plt.subplots(figsize=(max(12, n_dates * 0.6), 6))

    for i, (product_name, info) in enumerate(PRODUCTS.items()):
        counts = stats[product_name]
        offset = (i - n_products / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, counts, width,
            label=info["label"],
            color=COLORS[product_name],
            edgecolor='white', linewidth=0.5,
            alpha=0.85, zorder=3
        )

    # X-axis: dates
    ax.set_xticks(x)
    if n_dates <= 30:
        ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=8)
    else:
        # Show every Nth label to avoid overlap
        step = max(1, n_dates // 25)
        labels = [d if i % step == 0 else '' for i, d in enumerate(dates)]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Active time steps (out of 96)', fontsize=11)
    ax.set_title('Lightning Activity per Day', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(max(c) for c in stats.values()) * 1.12)
    ax.axhline(y=96, color='red', linestyle='--', alpha=0.3, linewidth=0.8,
               label='max (96)')
    ax.grid(axis='y', alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_per_timestep(times, stats, n_dates, save_path):
    """
    Bar plot: active days per time step, grouped by product.
    """
    n_times = len(times)
    if n_times == 0:
        print("No data to plot (per timestep).")
        return

    x = np.arange(n_times)
    n_products = len(PRODUCTS)
    width = 0.8 / n_products

    fig, ax = plt.subplots(figsize=(max(14, n_times * 0.18), 6))

    for i, (product_name, info) in enumerate(PRODUCTS.items()):
        counts = stats[product_name]
        offset = (i - n_products / 2 + 0.5) * width
        ax.bar(
            x + offset, counts, width,
            label=f"{info['label']} ({n_dates.get(product_name, '?')} days)",
            color=COLORS[product_name],
            edgecolor='white', linewidth=0.3,
            alpha=0.85, zorder=3
        )

    # X-axis: format HHMM as HH:MM, show every Nth
    formatted_times = [f"{t[:2]}:{t[2:]}" for t in times]
    ax.set_xticks(x)

    # Show labels at :00 boundaries for readability
    labels = []
    for i, t in enumerate(times):
        if t.endswith("00"):
            labels.append(f"{t[:2]}:{t[2:]}")
        else:
            labels.append('')
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

    ax.set_xlabel('Time step (UTC)', fontsize=11)
    ax.set_ylabel('Number of days with activity', fontsize=11)
    ax.set_title('Lightning Activity per Time Step (across all days)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)

    max_count = max(max(c) for c in stats.values()) if any(stats.values()) else 1
    ax.set_ylim(0, max_count * 1.12)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Add hour separators
    for i, t in enumerate(times):
        if t.endswith("00") and i > 0:
            ax.axvline(x=i - 0.5, color='gray', linestyle=':', alpha=0.2,
                       linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# CSV export
# =============================================================================

def export_active_steps_csv(all_activity, output_path):
    """
    Export a CSV listing every active time step per product.

    Columns: date, time_utc, density, current, occurrence

    Each row is a (date, time) pair that has activity in at least one product.
    The product columns contain 1 (active) or 0 (inactive).

    Args:
        all_activity (dict): {product: {date: {HHMM: bool}}}
        output_path (str): Path to the output CSV file
    """
    import csv

    # Collect all (date, time) pairs across all products
    all_pairs = set()
    for product_activity in all_activity.values():
        for date_str, day_data in product_activity.items():
            for time_str, active in day_data.items():
                if active:
                    all_pairs.add((date_str, time_str))

    # Sort by date, then time
    sorted_pairs = sorted(all_pairs)

    # For each active pair, check which products have activity
    product_names = list(PRODUCTS.keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'time_utc'] + product_names)

        for date_str, time_str in sorted_pairs:
            time_formatted = f"{time_str[:2]}:{time_str[2:]}"
            row = [date_str, time_formatted]
            for product_name in product_names:
                active = all_activity.get(product_name, {}).get(
                    date_str, {}).get(time_str, False)
                row.append(1 if active else 0)
            writer.writerow(row)

    print(f"  CSV saved: {output_path}  ({len(sorted_pairs)} active rows)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate bar plots and CSV of lightning activity statistics."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Root lightning data directory (default: {DEFAULT_DATA_ROOT})"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=".",
        help="Directory to save plots and CSV (default: current directory)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Lightning Activity Statistics")
    print("=" * 70)
    print(f"Data root  : {args.data_root}")
    print(f"Output dir : {args.output_dir}")
    print()

    # Scan all products
    all_activity = scan_all_products(args.data_root)

    if not any(all_activity.values()):
        print("\nNo data found. Check that process_lightning_kml.py has been run.")
        return

    # Compute stats
    print("\nComputing statistics...")
    dates, per_day_stats = compute_per_day_stats(all_activity)
    times, per_ts_stats, n_dates = compute_per_timestep_stats(all_activity)

    print(f"  {len(dates)} dates, {len(times)} unique time steps")

    # Print summary table
    print(f"\n{'Product':<14} {'Total files':>12} {'Active':>8} {'Pct':>7}")
    print("-" * 45)
    for product_name, info in PRODUCTS.items():
        total = sum(len(t) for t in all_activity[product_name].values())
        active = sum(
            sum(1 for v in t.values() if v)
            for t in all_activity[product_name].values()
        )
        pct = (active / total * 100) if total > 0 else 0
        print(f"{info['label']:<14} {total:>12} {active:>8} {pct:>6.1f}%")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Plots
    print("\nGenerating plots...")
    plot_per_day(
        dates, per_day_stats,
        save_path=os.path.join(args.output_dir, "lightning_activity_per_day.png")
    )
    plot_per_timestep(
        times, per_ts_stats, n_dates,
        save_path=os.path.join(args.output_dir, "lightning_activity_per_timestep.png")
    )

    # CSV
    print("\nExporting CSV...")
    export_active_steps_csv(
        all_activity,
        os.path.join(args.output_dir, "lightning_active_steps.csv")
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
    