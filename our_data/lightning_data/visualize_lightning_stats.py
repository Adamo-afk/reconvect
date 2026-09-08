"""
Lightning activity bar plots - reads `lightning_active_steps.csv` and
`lightning_summary.csv` produced by `summarize_lightning_data.py` and
emits three bar charts.

The active CSV exposes three sub-product flags (density, current,
occurrence) per (date, HH:MM), but in practice they always agree
(non-zero strokes in a window light up all three). For visualization
we collapse them into a single "active" boolean per (date, HH:MM) -
a row counts as active iff any flag column is 1. This avoids three
identical bars stacking on top of each other in the output PNGs.

Plot 1 - Per day:
    For each date, count how many (filter-grid) time steps were active.

Plot 2 - Per time step:
    For each HH:MM across all dates, count how many days have activity
    at that time step.

Plot 3 - Per month, active vs non-active:
    Stacked bars of the timesteps that carry a stroke and those that
    exist but do not, against the cadence expectation. This is the
    view of the activity balance; the coverage intersection gates on
    presence and does not favour active timesteps.

The scanning + CSV-writing logic that used to live here now lives in
`summarize_lightning_data.py` so the .npy files are walked exactly once
per pipeline run. This script is read-only over the CSV.

Usage:
    python visualize_lightning_stats.py
    python visualize_lightning_stats.py --csv path/to/lightning_active_steps.csv \\
        --output_dir F:/figures
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The summariser writes beside the product, and so do these plots.
PRODUCT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PRODUCT_DIR / "lightning_active_steps.csv"
DEFAULT_SUMMARY = PRODUCT_DIR / "lightning_summary.csv"

BAR_COLOR = "#1976D2"  # blue
BAR_LABEL = "Active timesteps"
QUIET_COLOR = "#B0BEC5"  # grey: frame present, no stroke
EXPECTED_COLOR = "#455A64"


# =============================================================================
# CSV reader
# =============================================================================

def load_active_steps(csv_path: Path):
    """
    Read `lightning_active_steps.csv` and collapse the per-sub-product
    flags into a single "active" boolean per (date, HH:MM).

    A row counts as active iff any of the flag columns (everything after
    `date` and `time_utc`) is `1`. The sub-product identity is dropped -
    in practice all three columns agree, so the collapse is lossless for
    visualization purposes; the original CSV still carries the per-product
    detail for any caller that wants it.

    Returns: dict[date_str] -> set[HHMM]
    """
    if not csv_path.exists():
        print(
            f"ERROR: {csv_path} not found.\n"
            f"Run from the project root:\n"
            f"    python our_data/lightning_data/summarize_lightning_data.py",
            file=sys.stderr,
        )
        sys.exit(2)

    activity: dict[str, set[str]] = defaultdict(set)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"ERROR: {csv_path}: empty CSV")
        for col in ("date", "time_utc"):
            if col not in reader.fieldnames:
                sys.exit(
                    f"ERROR: {csv_path}: missing required column {col!r}."
                )
        flag_cols = [c for c in reader.fieldnames
                     if c not in ("date", "time_utc")]
        if not flag_cols:
            sys.exit(
                f"ERROR: {csv_path}: no sub-product flag columns (need "
                f"at least one column beyond date,time_utc)."
            )
        for row in reader:
            if not any((row.get(c) or "").strip() == "1" for c in flag_cols):
                continue
            date_str = row["date"].strip()
            time_str = row["time_utc"].strip().replace(":", "")
            if not date_str or not time_str:
                continue
            activity[date_str].add(time_str)

    return dict(activity)


def load_monthly_balance(summary_csv: Path):
    """Per month: (active, present, expected) timestep counts.

    From the summary CSV: `complete_union` is the active count,
    `present_all` the frames on disk (falls back to the smallest
    per-product `_files` count for a summary written before that
    column existed), `expected_grid` the cadence expectation. Reads
    only the CSV, so the chart can be redrawn without rescanning the
    archive.
    """
    if not summary_csv.exists():
        print(
            f"ERROR: {summary_csv} not found.\n"
            f"Run from the project root:\n"
            f"    python our_data/lightning_data/summarize_lightning_data.py",
            file=sys.stderr,
        )
        sys.exit(2)
    months: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["date"][:7]
            active = int(row["complete_union"])
            if row.get("present_all") not in (None, ""):
                present = int(row["present_all"])
            else:
                present = min(int(row[c]) for c in reader.fieldnames
                              if c.endswith("_files"))
            months[m][0] += active
            months[m][1] += present
            months[m][2] += int(row["expected_grid"])
    return dict(months)


# =============================================================================
# Statistics
# =============================================================================

def compute_per_day_stats(activity):
    """For each date, count active time steps."""
    dates = sorted(activity.keys())
    counts = [len(activity[d]) for d in dates]
    return dates, counts


def compute_per_timestep_stats(activity):
    """For each HHMM, count how many days have activity at that step."""
    all_times: set[str] = set()
    for times in activity.values():
        all_times.update(times)
    times = sorted(all_times)
    counts = [
        sum(1 for day_times in activity.values() if t in day_times)
        for t in times
    ]
    return times, counts, len(activity)


# =============================================================================
# Plotting
# =============================================================================

def plot_per_day(dates, counts, save_path):
    if not dates:
        print("No data to plot (per day).")
        return

    x = np.arange(len(dates))

    fig, ax = plt.subplots(figsize=(max(12, len(dates) * 0.5), 6))
    ax.bar(
        x, counts,
        width=0.85,
        label=BAR_LABEL,
        color=BAR_COLOR,
        edgecolor="white", linewidth=0.5,
        alpha=0.9, zorder=3,
    )

    ax.set_xticks(x)
    if len(dates) <= 30:
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    else:
        step = max(1, len(dates) // 25)
        labels = [d if i % step == 0 else "" for i, d in enumerate(dates)]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Active time steps", fontsize=11)
    ax.set_title("Lightning Activity per Day", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    max_count = max(counts, default=1)
    ax.set_ylim(0, max(1, max_count) * 1.12)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_per_timestep(times, counts, n_dates, save_path):
    if not times:
        print("No data to plot (per timestep).")
        return

    x = np.arange(len(times))

    fig, ax = plt.subplots(figsize=(max(14, len(times) * 0.18), 6))
    ax.bar(
        x, counts,
        width=0.85,
        label=f"{BAR_LABEL} ({n_dates} days)",
        color=BAR_COLOR,
        edgecolor="white", linewidth=0.3,
        alpha=0.9, zorder=3,
    )

    ax.set_xticks(x)
    labels = [f"{t[:2]}:{t[2:]}" if t.endswith("00") else "" for t in times]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Time step (UTC)", fontsize=11)
    ax.set_ylabel("Number of days with activity", fontsize=11)
    ax.set_title("Lightning Activity per Time Step (across all days)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)

    max_count = max(counts, default=1)
    ax.set_ylim(0, max(1, max_count) * 1.12)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for i, t in enumerate(times):
        if t.endswith("00") and i > 0:
            ax.axvline(x=i - 0.5, color="gray", linestyle=":",
                       alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_monthly_balance(months, save_path):
    """Active vs non-active timesteps per month, stacked, with the
    cadence expectation as a dashed outline so missing days stay
    visible rather than looking like quiet ones."""
    if not months:
        print("No data to plot (per month).")
        return
    keys = sorted(months)
    active = np.array([months[m][0] for m in keys])
    present = np.array([months[m][1] for m in keys])
    expected = np.array([months[m][2] for m in keys])
    quiet = np.clip(present - active, 0, None)
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.65), 5.2))
    ax.bar(x, expected, color="none", edgecolor=EXPECTED_COLOR,
           linewidth=1.0, linestyle="--", zorder=1, label="expected (cadence)")
    ax.bar(x, quiet, color=QUIET_COLOR, width=0.8, zorder=2,
           label="present, no stroke")
    ax.bar(x, active, bottom=quiet, color=BAR_COLOR, width=0.8, zorder=3,
           label=BAR_LABEL)
    for i in range(len(keys)):
        if present[i]:
            ax.text(x[i], present[i], f"{100 * active[i] / present[i]:.0f}%",
                    ha="center", va="bottom", fontsize=8, color=EXPECTED_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Timesteps", fontsize=11)
    ax.set_title("Lightning timesteps per month: active vs present-but-quiet",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(1, expected.max(), present.max()) * 1.12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate bar plots of lightning activity from "
                    "lightning_active_steps.csv. The per-sub-product flags "
                    "in the CSV are collapsed into a single `active` "
                    "boolean for display."
    )
    parser.add_argument(
        "--csv", "-c", type=str, default=str(DEFAULT_CSV),
        help=f"Path to lightning_active_steps.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--summary", "-s", type=str, default=str(DEFAULT_SUMMARY),
        help=f"Path to lightning_summary.csv, for the per-month "
             f"active/non-active chart (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=str(PRODUCT_DIR),
        help=f"Directory to save plots (default: {PRODUCT_DIR})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Lightning Activity Plots")
    print("=" * 70)
    print(f"CSV        : {args.csv}")
    print(f"Output dir : {args.output_dir}")
    print()

    activity = load_active_steps(Path(args.csv))

    print("\nComputing statistics...")
    dates, per_day_counts = compute_per_day_stats(activity)
    times, per_ts_counts, n_dates = compute_per_timestep_stats(activity)

    print(f"  {len(dates)} dates, {len(times)} unique time steps")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating plots...")
    plot_per_day(
        dates, per_day_counts,
        save_path=os.path.join(args.output_dir, "lightning_activity_per_day.png"),
    )
    plot_per_timestep(
        times, per_ts_counts, n_dates,
        save_path=os.path.join(args.output_dir, "lightning_activity_per_timestep.png"),
    )
    plot_monthly_balance(
        load_monthly_balance(Path(args.summary)),
        save_path=os.path.join(args.output_dir, "lightning_activity_monthly.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
