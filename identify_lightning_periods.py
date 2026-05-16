"""
identify_lightning_periods.py - occurrence-fraction filter for lightning.

Pipeline placement:

    Step 1a  summarize_lightning_data.py     emits lightning_summary.csv +
                                             lightning_active_steps.csv at
                                             project root (applies the
                                             timestep_config.json filter
                                             and records which sub-product
                                             fired at each on-grid HH:MM).

    Step 1b  identify_lightning_periods.py   THIS SCRIPT - reads the active
                                             CSV, computes the
                                             occurrence-map fraction for
                                             each surviving (date, HH:MM)
                                             individually, keeps only
                                             those with fraction >=
                                             ratio * mean (default ratio
                                             0.30 over the active set),
                                             and emits a per-map per-patch
                                             activity CSV at the native
                                             cadence.

    Step 4.1 extract_patch_seq_for_datasets  --source lightning consumes
                                             lightning_patches.csv as-is.

Design contract
---------------
- Active set is taken from `lightning_active_steps.csv`, rows where
  occurrence == 1. The timestep_config.json minute filter has already
  been applied upstream (in summarize_lightning_data.py); we do NOT
  re-apply it here.
- For each (date, HH:MM) in that set, load the occurrence `.npy` map and
  compute `fraction = nonzero_pixels / total_pixels`. Fraction is
  evaluated at the **individual map** level - no temporal aggregation
  into wider bins.
- Threshold = `--fraction_threshold_ratio * mean(fraction)`, where the
  mean is taken over the same active set. Default ratio is 0.30.
- Maps with fraction below the threshold are dropped. Each surviving
  map produces exactly one row in `lightning_patches.csv` at its native
  HH:MM.
- Patch activity uses the WHOLE-MAP gate: if the surviving map has at
  least one non-zero pixel inside the patch's 256x256 region, the patch
  flag is 1. The fraction threshold was already applied at the per-map
  level - the patch step is just a spatial breakdown of the same map.

Outputs (default our_data/lightning_periods/)
---------------------------------------------
    lightning_periods_config.json   # CLI parameters + computed mean /
                                    # threshold / pre-and-post counts
                                    # for reproducibility. The
                                    # `step_minutes` field tells
                                    # extract_patch_seq_for_datasets.py
                                    # the cadence to use when looking
                                    # for continuous sequences.
    lightning_patches.csv           # Per-map per-patch activity:
                                    # date,time_utc,iso_timestamp,
                                    # patch_1,...,patch_18
                                    # (schema matches patch_index.csv)

Usage:
    python validate_timestep.py --step_minutes 15           # one-time
    python our_data/lightning_data/summarize_lightning_data.py
    python identify_lightning_periods.py                    # all defaults
    python identify_lightning_periods.py --fraction_threshold_ratio 0.50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# =============================================================================
# Project paths + grid constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"
DEFAULT_LIGHTNING_DIR = DEFAULT_DATA_ROOT / "lightning_data"
DEFAULT_ACTIVE_CSV = PROJECT_ROOT / "lightning_active_steps.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "lightning_periods"

# Romania grid + 6x3 patch layout (must match identify_patches.py).
GRID_WIDTH = 1536
GRID_HEIGHT = 768
PATCH_SIZE = 256
N_COLS = 6
N_ROWS = 3
N_PATCHES = N_COLS * N_ROWS  # 18
TOTAL_PIXELS = GRID_WIDTH * GRID_HEIGHT  # 1_179_648

OCCURRENCE_PRODUCT = "occurrence"

DEFAULT_FRACTION_THRESHOLD_RATIO = 0.30


# =============================================================================
# Config loaders
# =============================================================================

def load_timestep_config() -> tuple[int, set[int]]:
    """Return (step_minutes, lightning_filter_minutes) from timestep_config.json.

    `step_minutes` defines the master grid every downstream script walks.
    The lightning filter is the subset of minute-of-hour slots whose
    .npy maps actually exist (snap target). Both come from
    validate_timestep.py - they're the single source of truth for the
    pipeline cadence.
    """
    if not TIMESTEP_CONFIG_PATH.exists():
        print(
            f"ERROR: timestep config not found at {TIMESTEP_CONFIG_PATH}.\n"
            f"Run from the project root:\n"
            f"    python validate_timestep.py --step_minutes <N>",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = json.loads(TIMESTEP_CONFIG_PATH.read_text())
    step_minutes = int(cfg["step_minutes"])
    flt = cfg.get("products", {}).get("lightning", {}).get("filter")
    if flt is None:
        sys.exit(
            "ERROR: products.lightning.filter is null in "
            f"{TIMESTEP_CONFIG_PATH}. Re-run validate_timestep.py with a "
            "non-continuous lightning cadence in product_cadences.config."
        )
    return step_minutes, set(int(m) for m in flt)


def snap_hhmm_to_filter(hhmm: str, filter_minutes: set[int]) -> str:
    """Snap master-grid HHMM to the nearest minute in `filter_minutes`.

    Same rule used in intersect_product_coverage.snap_hhmm so the
    master grid and the active CSV land on the same filter HHMM. Ties
    break to the lower minute (matches validate_timestep.compute_minute_filter).
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


# =============================================================================
# Inputs
# =============================================================================

def load_active_occurrence_keys(csv_path: Path) -> list[tuple[str, str]]:
    """Return the sorted list of (date_str, HHMM) where occurrence == 1.

    The active CSV is produced by summarize_lightning_data.py. We only
    use rows where the occurrence flag fired - those are the maps the
    fraction filter is allowed to consider.
    """
    if not csv_path.is_file():
        sys.exit(
            f"ERROR: lightning_active_steps.csv not found at {csv_path}.\n"
            f"Run from the project root:\n"
            f"    python our_data/lightning_data/summarize_lightning_data.py"
        )
    rows: list[tuple[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"ERROR: {csv_path}: empty CSV")
        for col in ("date", "time_utc", "occurrence"):
            if col not in reader.fieldnames:
                sys.exit(
                    f"ERROR: {csv_path}: missing required column {col!r}"
                )
        for row in reader:
            if (row.get("occurrence") or "").strip() != "1":
                continue
            date_str = (row.get("date") or "").strip()
            time_str = (row.get("time_utc") or "").strip()
            if not date_str or not time_str:
                continue
            hhmm = time_str.replace(":", "").zfill(4)
            rows.append((date_str, hhmm))
    return sorted(set(rows))


def occurrence_npy_path(lightning_dir: Path, date_str: str, hhmm: str) -> Path:
    compact = date_str.replace("-", "")
    return (
        lightning_dir
        / OCCURRENCE_PRODUCT
        / f"nc4_{date_str}-Romania_{OCCURRENCE_PRODUCT}"
        / f"lightning_occurrence_{compact}_{hhmm}.npy"
    )


def load_occurrence_map(path: Path) -> np.ndarray | None:
    """Load an occurrence .npy as a 2-D uint8 array (or None on failure).

    The on-disk dtype is int8 (binary 0/1) per read_kml_version2.py.
    """
    try:
        data = np.load(path)
    except Exception as e:
        print(f"  WARNING: cannot read {path}: {e}", file=sys.stderr)
        return None
    if data.ndim == 3:
        data = data[0]
    return (data != 0).astype(np.uint8)


def compute_fraction(arr: np.ndarray) -> float:
    nonzero = int(np.count_nonzero(arr))
    return nonzero / TOTAL_PIXELS if TOTAL_PIXELS else 0.0


# =============================================================================
# Per-patch activity (whole-map gate)
# =============================================================================

def patches_with_activity(map_arr: np.ndarray) -> list[int]:
    """Return 1-indexed patch IDs with at least one non-zero pixel."""
    active: list[int] = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            chunk = map_arr[
                r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                c * PATCH_SIZE:(c + 1) * PATCH_SIZE,
            ]
            if chunk.any():
                active.append(r * N_COLS + c + 1)
    return active


# =============================================================================
# Output writers
# =============================================================================

def write_config(output_dir: Path, payload: dict) -> Path:
    path = output_dir / "lightning_periods_config.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_patches_csv(output_dir: Path,
                      rows_payload: list[dict]) -> tuple[Path, int]:
    """Write lightning_patches.csv.

    Schema: date, time_utc, iso_timestamp, patch_1, ..., patch_18.
    `time_utc` is the native HH:MM of the map (one row per surviving map).
    """
    path = output_dir / "lightning_patches.csv"
    n_rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["date", "time_utc", "iso_timestamp"]
        header += [f"patch_{p}" for p in range(1, N_PATCHES + 1)]
        w.writerow(header)
        for r in rows_payload:
            active = r["active_patches"]
            if not active:
                continue
            time_str = f"{r['hhmm'][:2]}:{r['hhmm'][2:]}"
            iso = f"{r['date']}T{time_str}:00"
            row = [r["date"], time_str, iso]
            row += [1 if p in active else 0 for p in range(1, N_PATCHES + 1)]
            w.writerow(row)
            n_rows += 1
    return path, n_rows


# =============================================================================
# Monitoring
# =============================================================================

def print_fraction_stats(fractions: list[float], threshold: float,
                          ratio: float, n_above: int) -> None:
    """Compact, scannable summary so the threshold decision is auditable."""
    n = len(fractions)
    if n == 0:
        print("  Active on-grid occurrence maps : 0 - nothing to threshold")
        return

    arr = np.asarray(fractions, dtype=np.float64)
    pct_arr = arr * 100.0
    q25, q50, q75 = np.percentile(pct_arr, [25, 50, 75])

    print("Occurrence-fraction threshold (per-map)")
    print("-" * 70)
    print(f"  Active on-grid maps    : {n}")
    print(f"  Fraction (% non-zero pixels):")
    print(f"    min                  : {pct_arr.min():9.4f}%")
    print(f"    p25                  : {q25:9.4f}%")
    print(f"    median               : {q50:9.4f}%")
    print(f"    p75                  : {q75:9.4f}%")
    print(f"    mean                 : {pct_arr.mean():9.4f}%")
    print(f"    max                  : {pct_arr.max():9.4f}%")
    print(f"  Threshold ratio        : {ratio:.3f} "
          f"(--fraction_threshold_ratio)")
    print(f"  Threshold (= ratio*mean): {threshold * 100:9.4f}% "
          f"non-zero pixels")
    pct_pass = (n_above / n * 100) if n else 0.0
    print(f"  Above threshold        : {n_above} / {n}  ({pct_pass:.1f}%)")
    print()


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-map occurrence-fraction filter for the lightning "
                    "training pipeline. No temporal aggregation - one "
                    "input map = one output row."
    )
    parser.add_argument(
        "--data_root", type=str, default=str(DEFAULT_DATA_ROOT),
        help=f"Root data directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--lightning_dir", type=str, default=str(DEFAULT_LIGHTNING_DIR),
        help=f"Lightning .npy root holding the occurrence subdir "
             f"(default: {DEFAULT_LIGHTNING_DIR})",
    )
    parser.add_argument(
        "--active_csv", type=str, default=str(DEFAULT_ACTIVE_CSV),
        help=f"Path to lightning_active_steps.csv "
             f"(default: {DEFAULT_ACTIVE_CSV}).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--fraction_threshold_ratio", type=float,
        default=DEFAULT_FRACTION_THRESHOLD_RATIO,
        help=f"Keep maps whose lightning fraction is >= ratio * mean "
             f"(over the active occurrence set). Default: "
             f"{DEFAULT_FRACTION_THRESHOLD_RATIO} (= 30%% of the average).",
    )
    args = parser.parse_args()

    step_minutes, filter_minutes = load_timestep_config()
    master_slots_per_day = 1440 // step_minutes

    print("=" * 70)
    print("Lightning occurrence-fraction filter (per-map, master-grid keyed)")
    print("=" * 70)
    print(f"step_minutes               : {step_minutes} "
          f"(timestep_config.json - master grid)")
    print(f"lightning filter           : {sorted(filter_minutes)} "
          f"(products.lightning.filter)")
    print(f"master slots / day         : {master_slots_per_day}")
    print(f"fraction_threshold_ratio   : {args.fraction_threshold_ratio}")
    print(f"active CSV                 : {args.active_csv}")
    print(f"lightning_dir              : {args.lightning_dir}")
    print(f"output_dir                 : {args.output_dir}")
    print()

    active_keys_filter = set(
        load_active_occurrence_keys(Path(args.active_csv))
    )
    print(f"Active occurrence timesteps (from CSV, filter HHMM): "
          f"{len(active_keys_filter)}")
    if not active_keys_filter:
        print("Nothing to filter. Exiting.")
        return 0

    # Dates that appear anywhere in the active set; we only walk those.
    dates = sorted({d for d, _ in active_keys_filter})

    # ------------------------------------------------------------------
    # Pass 1: walk the master grid (step_minutes) per date. For each
    # master HHMM, snap to the lightning filter and look up the .npy.
    # The active CSV is keyed by filter HHMM, so we only keep master
    # HHMMs whose snapped filter HHMM is active. The result is keyed by
    # master HHMM throughout so extract_patch_seq_for_datasets.py can
    # iterate the same grid every other product walks.
    # ------------------------------------------------------------------
    lightning_dir = Path(args.lightning_dir)
    fractions: list[float] = []
    fraction_by_master: dict[tuple[str, str], float] = {}
    map_by_master: dict[tuple[str, str], np.ndarray] = {}
    snapped_by_master: dict[tuple[str, str], str] = {}
    missing_files: list[tuple[str, str]] = []

    for date_str in dates:
        for k in range(master_slots_per_day):
            t = k * step_minutes
            master_hhmm = f"{t // 60:02d}{t % 60:02d}"
            filter_hhmm = snap_hhmm_to_filter(master_hhmm, filter_minutes)
            if (date_str, filter_hhmm) not in active_keys_filter:
                continue
            npy = occurrence_npy_path(lightning_dir, date_str, filter_hhmm)
            if not npy.exists():
                missing_files.append((date_str, master_hhmm))
                continue
            arr = load_occurrence_map(npy)
            if arr is None:
                missing_files.append((date_str, master_hhmm))
                continue
            frac = compute_fraction(arr)
            fractions.append(frac)
            fraction_by_master[(date_str, master_hhmm)] = frac
            map_by_master[(date_str, master_hhmm)] = arr
            snapped_by_master[(date_str, master_hhmm)] = filter_hhmm

    if missing_files:
        print(f"  WARNING: {len(missing_files)} master timesteps mapped to "
              f"active filter HHMMs but the underlying .npy is unreadable "
              f"(first: {missing_files[0]}, last: {missing_files[-1]}).")

    # ------------------------------------------------------------------
    # Threshold + per-map gating
    # ------------------------------------------------------------------
    mean_fraction = float(np.mean(fractions)) if fractions else 0.0
    threshold = args.fraction_threshold_ratio * mean_fraction

    kept_keys = sorted(
        k for k, f in fraction_by_master.items() if f >= threshold
    )

    print()
    print_fraction_stats(
        fractions, threshold, args.fraction_threshold_ratio, len(kept_keys)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not kept_keys:
        print("No occurrence maps pass the threshold. "
              "Writing empty lightning_patches.csv.")
        write_patches_csv(output_dir, [])
        write_config(output_dir, {
            "step_minutes":               step_minutes,
            "lightning_filter_minutes":   sorted(filter_minutes),
            "fraction_threshold_ratio":   args.fraction_threshold_ratio,
            "mean_fraction":              mean_fraction,
            "threshold_fraction":         threshold,
            "n_master_slots_active":      len(fraction_by_master),
            "n_above_threshold":          0,
            "n_csv_rows":                 0,
            "active_csv":                 str(Path(args.active_csv).resolve()),
            "lightning_dir":              str(Path(args.lightning_dir).resolve()),
            "created_utc":                datetime.now(timezone.utc)
                                                  .isoformat(timespec="seconds"),
        })
        return 0

    # ------------------------------------------------------------------
    # Per-patch activity for each surviving master HHMM (one row per
    # surviving slot, time_utc on the master grid so the row indexes
    # exactly the same way patch_index.csv does for radar).
    # ------------------------------------------------------------------
    rows_payload: list[dict] = []
    for date_str, master_hhmm in kept_keys:
        arr = map_by_master[(date_str, master_hhmm)]
        active = patches_with_activity(arr)
        rows_payload.append({
            "date": date_str,
            "hhmm": master_hhmm,
            "active_patches": active,
        })

    csv_path, n_csv_rows = write_patches_csv(output_dir, rows_payload)

    write_config(output_dir, {
        "step_minutes":               step_minutes,
        "lightning_filter_minutes":   sorted(filter_minutes),
        "fraction_threshold_ratio":   args.fraction_threshold_ratio,
        "mean_fraction":              mean_fraction,
        "threshold_fraction":         threshold,
        "n_master_slots_active":      len(fraction_by_master),
        "n_above_threshold":          len(kept_keys),
        "n_csv_rows":                 n_csv_rows,
        "active_csv":                 str(Path(args.active_csv).resolve()),
        "lightning_dir":              str(Path(args.lightning_dir).resolve()),
        "created_utc":                datetime.now(timezone.utc)
                                              .isoformat(timespec="seconds"),
    })

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    n_total_patches = sum(len(r["active_patches"]) for r in rows_payload)
    print("Summary")
    print("-" * 70)
    print(f"  Active master slots    : {len(fraction_by_master)}")
    print(f"  Above threshold        : {len(kept_keys)}")
    print(f"  CSV rows (>=1 patch)   : {n_csv_rows}")
    print(f"  Total patch flags      : {n_total_patches}")
    print()
    print(f"Wrote: {output_dir}")
    print(f"  lightning_periods_config.json")
    print(f"  lightning_patches.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
