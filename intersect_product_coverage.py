"""
intersect_product_coverage.py — Compute the per-timestep intersection of
available reprojected data across the chosen product set, and emit a
manifest that extract_patches.py consumes to know exactly which
timesteps to process.

Pipeline placement (Step 4.2):

    Step 4.1  extract_patch_seq_for_datasets.py    train/val/test CSVs
    Step 4.2  intersect_product_coverage.py        THIS SCRIPT - emits
                                                   timestep_manifest.csv +
                                                   a per-date accounting
                                                   plot
    Step 4.3  compute_normalization_stats.py       per-variable mean/std
    Step 5    extract_patches.py                   patch extraction reads
                                                   the manifest

Design contract
---------------
- The set of *active* products is determined by which `--summary` keys
  the user passes. Lightning is included only when
  `--summary lightning=...` is supplied. There is no implicit product
  list - what you pass is what you get.
- Per-timestep availability is read from one of two sources, per product:
    * Default: `<name>_missing_timesteps.json` (auto-discovered next to
      the summary CSV; overridable with `--missing KEY=PATH`). A slot
      survives iff its snapped HHMM is NOT in the missing set.
    * Opt-in via `--active KEY=PATH`: an active-steps CSV with
      `date,time_utc,...` columns where each remaining column is a 1/0
      activity flag. A slot survives iff its snapped HHMM IS in the
      active set (any of the flag columns == 1). Used for lightning
      (`lightning_active_steps.csv` from summarize_lightning_data.py)
      where the activity signal — not just file presence — is what we
      want the manifest to gate on. When `--active` is provided for a
      product, the missing-JSON path is skipped for it entirely.
- Per-product minute filters come from `timestep_config.json` (written
  by validate_timestep.py). For each master-grid HHMM we snap to the
  nearest minute in each product's filter, then check whether it is
  in the available set for that product.
- The `--products` flag belongs to extract_patches.py, not to this
  script - intersect's product set is implicit in `--summary`.
- Train / val / test CSVs are not touched. They keep whatever
  extract_patch_seq_for_datasets.py wrote.

Outputs (only two, by design)
-----------------------------
- `timestep_manifest.csv` (default: our_data/timestep_manifest.csv)
    One row per surviving (date, HHMM) timestep, with the per-product
    snapped HHMM each product loaded so the manifest doubles as an
    audit trail:
        date,hhmm,mtg_hhmm,opera_hhmm,...
- `intersect_summary.png` (default: our_data/intersect_summary.png)
    Per-date stacked bar chart: kept timesteps + drops attributed to
    each product / error log.

Example
-------
    python intersect_product_coverage.py \
        --summary mtg=our_data/satellite_data/mtg_summary.csv \
        --summary opera=our_data/opera_data/opera_summary.csv \
        --summary lightning=our_data/lightning_data/lightning_summary.csv \
        --active lightning=our_data/lightning_data/lightning_active_steps.csv \
        --errors_log our_data/reprojected_data/reproject_satellite_MTG.log
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"
DEFAULT_OUTPUT_CSV = DEFAULT_DATA_ROOT / "timestep_manifest.csv"
DEFAULT_OUTPUT_PLOT = DEFAULT_DATA_ROOT / "intersect_summary.png"
DEFAULT_TIMESTEP_CONFIG = DEFAULT_DATA_ROOT / "timestep_config.json"

# Known product keys and their mapping to:
#   - the `products.<name>` block in timestep_config.json (for the filter)
#   - the conventional missing-JSON file name (alongside the summary CSV)
PRODUCT_LAYOUT: dict[str, dict[str, str]] = {
    "mtg":       {"tsconfig_product": "mtg",
                  "missing_name":     "mtg_missing_timesteps.json"},
    # OPERA ships two independent fields. Name them separately so a
    # manifest can require exactly the ones its modes consume: a
    # rainfall-only model should not lose a sample because reflectivity
    # was absent that timestep, and a model that reads reflectivity must
    # not be handed a timestep lacking it. Both keys read the same
    # summary CSV and the same missing JSON, selecting different blocks.
    "opera_rainfall_rate": {
        "tsconfig_product": "opera_rainfall_rate",
        "missing_name":     "opera_missing_timesteps.json"},
    "opera_reflectivity": {
        "tsconfig_product": "opera_reflectivity",
        "missing_name":     "opera_missing_timesteps.json"},
    # Backwards-compatible alias: `opera` has always meant rainfall_rate,
    # the DBSCAN driver and label source.
    "opera":     {"tsconfig_product": "opera_rainfall_rate",
                  "missing_name":     "opera_missing_timesteps.json"},
    "lightning": {"tsconfig_product": "lightning",
                  "missing_name":     "lightning_missing_timesteps.json"},
}

# Which block inside our_data/opera_data/opera_missing_timesteps.json each key reads.
OPERA_SUBPRODUCT: dict[str, str] = {
    "opera":               "opera_rainfall_rate",
    "opera_rainfall_rate": "opera_rainfall_rate",
    "opera_reflectivity":  "opera_reflectivity",
}


# =============================================================================
# Argument parsing
# =============================================================================

def parse_keyed_arg(raw: str, flag_name: str) -> tuple[str, Path]:
    """Parse `KEY=PATH`. Tolerates a legacy trailing `:COLUMN`."""
    if "=" not in raw:
        sys.exit(f"ERROR: {flag_name} expects KEY=PATH, got {raw!r}")
    key, _, rest = raw.partition("=")
    key = key.strip().lower()
    if ":" in rest and not re.match(r"^[A-Za-z]:[\\/]", rest):
        rest = rest.split(":", 1)[0]   # drop legacy ":column"
    if key not in PRODUCT_LAYOUT:
        sys.exit(
            f"ERROR: {flag_name} key {key!r} is not a known product. "
            f"Choose from: {sorted(PRODUCT_LAYOUT)}"
        )
    return key, Path(rest.strip())


def parse_summary_arg(raw: str) -> tuple[str, Path]:
    return parse_keyed_arg(raw, "--summary")


def parse_missing_arg(raw: str) -> tuple[str, Path]:
    return parse_keyed_arg(raw, "--missing")


def parse_active_arg(raw: str) -> tuple[str, Path]:
    return parse_keyed_arg(raw, "--active")


# =============================================================================
# Summary CSV - we use it for the date list (which days the product was
# scanned for) and as a presence marker for the active product set.
# =============================================================================

def read_summary_dates(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        sys.exit(f"ERROR: summary CSV not found: {csv_path}")
    dates: set[str] = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            sys.exit(f"ERROR: {csv_path}: missing `date` column")
        for row in reader:
            d = row.get("date", "").strip()
            if d:
                dates.add(d)
    return dates


# =============================================================================
# Missing-JSON parsers (one schema per product family)
# =============================================================================

def _hhmm_4d(hhmm_or_hhcolon: str) -> str:
    """Normalise 'HH:MM' or 'HHMM' to 4-digit 'HHMM'."""
    return hhmm_or_hhcolon.replace(":", "").zfill(4)


def load_missing(product: str, json_path: Path) -> set[tuple[str, str]]:
    """Return the set of (date, 'HHMM') tuples missing for `product`.

    The three summarizers emit slightly different JSON shapes; this
    function normalises them into a single set.
    """
    if not json_path.exists():
        print(f"  WARNING: missing-timesteps JSON not found for "
              f"{product!r}: {json_path}. Treating product as fully "
              f"present (probably wrong - re-run summarize_{product}.py "
              f"with default --missing to populate it).")
        return set()

    try:
        data = json.loads(json_path.read_text())
    except (OSError, ValueError) as e:
        sys.exit(f"ERROR: failed to parse {json_path}: {e}")

    out: set[tuple[str, str]] = set()
    dates_block = data.get("dates", {}) or {}

    for date_str, block in dates_block.items():
        if not isinstance(block, dict):
            continue

        if product in OPERA_SUBPRODUCT:
            # our_data/opera_data/opera_missing_timesteps.json nests per sub-product; take the
            # one this key stands for.
            inner = block.get(OPERA_SUBPRODUCT[product], {})
            times = inner.get("missing_times", []) or []
        elif product == "mtg":
            # our_data/satellite_data/mtg_missing_timesteps.json: per-date block with
            # 'missing_times' at the top level (single product).
            times = block.get("missing_times", []) or []
            # MTG also tracks 'incomplete_times' (only 1 of 2 chunks);
            # treat those as missing too because the reproject would
            # have failed on them.
            times = list(times) + list(block.get("incomplete_times", []) or [])
        else:
            # Generic fallback for radar / lightning / future products.
            times = (block.get("missing_times")
                     or block.get("missing")
                     or [])

        for t in times:
            if not isinstance(t, str):
                continue
            out.add((date_str, _hhmm_4d(t)))

    return out


def load_active(product: str, csv_path: Path) -> set[tuple[str, str]]:
    """Return the set of (date, 'HHMM') tuples marked active for `product`.

    The active CSV (e.g. `lightning_active_steps.csv` from
    summarize_lightning_data.py) has columns
        date,time_utc,<flag_1>,<flag_2>,...
    where time_utc is `HH:MM` and the remaining columns are 1/0 flags
    per sub-product. A row counts as "active" iff at least one flag
    column is `1` — the any-of-three semantics requested for lightning.

    When `--active` is supplied for a product, this set replaces the
    missing-JSON gate entirely: a slot survives iff its snapped HHMM
    is IN the returned set.
    """
    if not csv_path.is_file():
        sys.exit(
            f"ERROR: --active {product}: file not found: {csv_path}.\n"
            f"Run summarize_lightning_data.py to regenerate it."
        )

    out: set[tuple[str, str]] = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"ERROR: --active {product}: empty CSV {csv_path}")
        missing_cols = [c for c in ("date", "time_utc")
                        if c not in reader.fieldnames]
        if missing_cols:
            sys.exit(
                f"ERROR: --active {product}: {csv_path} missing required "
                f"columns: {missing_cols}"
            )
        flag_cols = [c for c in reader.fieldnames
                     if c not in ("date", "time_utc")]
        if not flag_cols:
            sys.exit(
                f"ERROR: --active {product}: {csv_path} has no flag columns "
                f"(need at least one column beyond date,time_utc)"
            )
        for row in reader:
            date_str = (row.get("date") or "").strip()
            time_str = (row.get("time_utc") or "").strip()
            if not date_str or not time_str:
                continue
            if not any((row.get(c) or "").strip() == "1" for c in flag_cols):
                continue
            out.add((date_str, _hhmm_4d(time_str)))

    return out


# =============================================================================
# Cadence snap (same rule as extract_patches.py)
# =============================================================================

def load_timestep_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"ERROR: timestep_config.json not found at {path}.\n"
            f"Run `python validate_timestep.py --step_minutes <N>` first."
        )
    return json.loads(path.read_text())


def snap_hhmm(hhmm: str, filter_minutes: set[int]) -> str:
    """Snap to nearest minute in `filter_minutes`; handle hour wrap; tie
    breaks prefer the earlier minute."""
    h, m = int(hhmm[:2]), int(hhmm[2:])
    best = min(filter_minutes, key=lambda fm: (
        min(abs(fm - m), 60 - abs(fm - m)),
        fm,
    ))
    diff = best - m
    if diff > 30:
        diff -= 60
    elif diff < -30:
        diff += 60
    total = (h * 60 + m + diff) % (24 * 60)
    return f"{total // 60:02d}{total % 60:02d}"


# =============================================================================
# Error-log parsing (reused from the previous design)
# =============================================================================

_ERROR_FILENAME_PATTERNS = (
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hhmm>\d{4})\d{2}Z"),
    re.compile(r"_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})T(?P<hhmm>\d{4})\d{2}Z"),
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})-Romania_(?P<hhmm>\d{4})_"),
    re.compile(r"_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_(?P<hhmm>\d{4})"),
)


def parse_error_log(log_path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not log_path.exists():
        print(f"  ERROR log not found, skipping: {log_path}")
        return pairs
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("ERROR "):
                continue
            fname = line[len("ERROR "):].split(":", 1)[0].strip()
            for pat in _ERROR_FILENAME_PATTERNS:
                mm = pat.search(fname)
                if not mm:
                    continue
                groups = mm.groupdict()
                if "date" in groups:
                    date_str = groups["date"]
                else:
                    date_str = f"{groups['y']}-{groups['m']}-{groups['d']}"
                pairs.add((date_str, groups["hhmm"]))
                break
    return pairs


# =============================================================================
# Intersection
# =============================================================================

def build_master_grid(dates: list[str],
                       step_minutes: int) -> list[tuple[str, str]]:
    """Every step_minutes slot on every date."""
    n_slots = 24 * 60 // step_minutes
    out: list[tuple[str, str]] = []
    for date_str in dates:
        for k in range(n_slots):
            total = k * step_minutes
            out.append((date_str, f"{total // 60:02d}{total % 60:02d}"))
    return out


def intersect(master, product_keys, missing_by_product, active_by_product,
              filter_by_product, error_pairs, dates_by_product):
    """For each master slot:
      1. snap HHMM to each product's filter,
      2. presence check, branching per product:
         - if the product has an `active_by_product` entry (set), the slot
           survives iff `(date, snapped) IN active_set` (active-CSV mode),
         - otherwise the slot survives iff `(date, snapped) NOT IN missing`
           (missing-JSON mode, the legacy default),
         and the date must be in the product's scanned-dates set,
      3. check that no error-log entry matches.

    Returns:
        kept: list of (date, master_hhmm, {product: snapped_hhmm})
        dropped_by_reason: dict reason -> list of (date, master_hhmm)
    """
    kept: list[tuple[str, str, dict[str, str]]] = []
    dropped: dict[str, list[tuple[str, str]]] = {p: [] for p in product_keys}
    dropped["error_log"] = []
    dropped["unscanned_date"] = []

    for date_str, hhmm in master:
        per_product_hhmm: dict[str, str] = {}
        drop_reason: str | None = None

        for product in product_keys:
            if date_str not in dates_by_product[product]:
                drop_reason = "unscanned_date"
                break
            flt = filter_by_product[product]
            snapped = snap_hhmm(hhmm, flt) if flt else hhmm
            active_set = active_by_product.get(product)
            if active_set is not None:
                if (date_str, snapped) not in active_set:
                    drop_reason = product
                    break
            else:
                if (date_str, snapped) in missing_by_product[product]:
                    drop_reason = product
                    break
            per_product_hhmm[product] = snapped

        if drop_reason is not None:
            dropped[drop_reason].append((date_str, hhmm))
            continue

        # Error-log filter: any product's snapped HHMM appearing in
        # the global error set kills the master slot.
        in_error = any(
            (date_str, snapped) in error_pairs
            for snapped in per_product_hhmm.values()
        )
        if in_error:
            dropped["error_log"].append((date_str, hhmm))
            continue

        kept.append((date_str, hhmm, per_product_hhmm))

    return kept, dropped


# =============================================================================
# Output
# =============================================================================

def write_manifest(kept, product_keys, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "hhmm"] + [f"{p}_hhmm" for p in product_keys]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for date_str, hhmm, per_product in kept:
            row = {"date": date_str, "hhmm": hhmm}
            for p in product_keys:
                row[f"{p}_hhmm"] = per_product.get(p, "")
            w.writerow(row)
    print(f"Manifest:     {out_path}  ({len(kept)} surviving timesteps)")


CATEGORY_COLOURS = {
    "kept":            "#4caf50",
    "unscanned_date":  "#bdbdbd",
    "mtg":             "#42a5f5",
    "opera":           "#ef5350",
    "lightning":       "#ffd54f",
    "error_log":       "#212121",
}


def per_date_counts(kept, dropped, dates, product_keys):
    """Per-date timestep counts for `kept` and for each drop reason.

    Returns (kept_count, drop_count, drop_reasons).
    """
    date_idx = {d: i for i, d in enumerate(dates)}

    kept_count = np.zeros(len(dates), dtype=int)
    for date_str, _hhmm, _ in kept:
        kept_count[date_idx[date_str]] += 1

    drop_reasons = ["unscanned_date"] + product_keys + ["error_log"]
    drop_count = {r: np.zeros(len(dates), dtype=int) for r in drop_reasons}
    for r in drop_reasons:
        for date_str, _hhmm in dropped.get(r, []):
            di = date_idx.get(date_str)
            if di is not None:
                drop_count[r][di] += 1
    return kept_count, drop_count, drop_reasons


def _calendar(dates: list[str]) -> list[_dt.date]:
    """Every day from the first to the last, including ones with no data."""
    first = _dt.date.fromisoformat(dates[0])
    last = _dt.date.fromisoformat(dates[-1])
    return [first + _dt.timedelta(days=i)
            for i in range((last - first).days + 1)]


def _months(cal: list[_dt.date]) -> list[_dt.date]:
    """First-of-month for every month the calendar touches."""
    out, seen = [], set()
    for day in cal:
        key = (day.year, day.month)
        if key not in seen:
            seen.add(key)
            out.append(_dt.date(day.year, day.month, 1))
    return out


def write_plot(kept, dropped, dates, step_minutes, product_keys,
               out_path: Path) -> None:
    """Monthly line graph: timesteps kept, and timesteps omitted by reason.

    Aggregated to months rather than days on purpose. At 15-minute cadence
    a daily series over a year and a half is ~590 points per line, which
    renders as noise - the month is the smallest unit at which the shape
    of the archive is actually readable, and it matches the monthly
    coverage charts the summarisers produce.

    Every category is its own line of the same quantity, and they sum to
    the grid capacity, so kept and omitted are directly comparable rather
    than being segments of one stack where a small reason is squeezed
    into invisibility beside a large one.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_slots_per_day = 24 * 60 // step_minutes
    kept_count, drop_count, drop_reasons = per_date_counts(
        kept, dropped, dates, product_keys)

    cal = _calendar(dates)
    idx = {d: i for i, d in enumerate(dates)}
    months = _months(cal)
    month_of = {m: k for k, m in enumerate(months)}

    def monthly(counts) -> np.ndarray:
        totals = np.zeros(len(months))
        for day in cal:
            i = idx.get(day.isoformat())
            if i is not None:
                totals[month_of[_dt.date(day.year, day.month, 1)]] += counts[i]
        return totals

    # Grid capacity per month: how many slots the master grid defines for
    # the days this run actually covers. Without it a short month reads as
    # a dip that is really just February.
    capacity = np.zeros(len(months))
    for day in cal:
        capacity[month_of[_dt.date(day.year, day.month, 1)]] += n_slots_per_day

    series = [("kept", kept_count)]
    series += [(r, drop_count[r]) for r in drop_reasons
               if int(drop_count[r].sum()) > 0]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(months, capacity, linestyle="--", linewidth=1.2,
            color="#bdbdbd", zorder=1,
            label=f"Grid capacity ({n_slots_per_day}/day)")

    for name, counts in series:
        label = "Kept" if name == "kept" else f"Omitted: {name}"
        ax.plot(months, monthly(counts), marker="o", markersize=4,
                linewidth=2.2, solid_capstyle="round",
                color=CATEGORY_COLOURS.get(name, "#9e9e9e"),
                label=f"{label} ({int(counts.sum()):,})", zorder=3)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Timesteps per month", fontsize=11)
    ax.set_title("Timestep accounting by month", fontsize=13, pad=14,
                 color="#444")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#666", labelsize=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot:         {out_path}  ({len(months)} month(s))")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the per-timestep intersection of available "
                    "reprojected data across the chosen product set. "
                    "Emits a manifest CSV + a per-date accounting plot."
    )
    parser.add_argument(
        "--summary", action="append", default=[], type=parse_summary_arg,
        metavar="KEY=PATH",
        help="Required at least once. KEY is one of "
             f"{sorted(PRODUCT_LAYOUT)}; PATH is the per-product summary "
             "CSV produced by a `summarize_*.py` script. The KEYs you "
             "pass define the active product set.",
    )
    parser.add_argument(
        "--missing", action="append", default=[], type=parse_missing_arg,
        metavar="KEY=PATH",
        help="Override the auto-discovered missing-timesteps JSON for "
             "a product. By default each `--summary KEY=PATH/summary.csv` "
             "looks for `KEY_missing_timesteps.json` in the same "
             "directory.",
    )
    parser.add_argument(
        "--active", action="append", default=[], type=parse_active_arg,
        metavar="KEY=PATH",
        help="Replace the missing-JSON gate for a product with an "
             "active-steps CSV (`date,time_utc,<flag1>,<flag2>,...`). A "
             "slot survives iff its snapped HHMM is IN this set. "
             "Mutually exclusive with --missing for the same product. "
             "Typical use: `--active lightning=our_data/lightning_data/lightning_active_steps.csv`.",
    )
    parser.add_argument(
        "--errors_log", action="append", default=[], metavar="PATH",
        help="Reproject error log (`reproject_<category>.log`). Repeat for "
             "each category. (date, HHMM) pairs parsed from these logs are "
             "removed from the kept set.",
    )
    parser.add_argument(
        "--timestep_config", type=str, default=str(DEFAULT_TIMESTEP_CONFIG),
        help=f"Path to timestep_config.json (default: "
             f"{DEFAULT_TIMESTEP_CONFIG}).",
    )
    parser.add_argument(
        "--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV),
        help=f"Manifest CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--output_plot", type=str, default=str(DEFAULT_OUTPUT_PLOT),
        help=f"Plot path (default: {DEFAULT_OUTPUT_PLOT}).",
    )

    args = parser.parse_args()

    if not args.summary:
        parser.error("at least one --summary KEY=PATH is required")

    product_keys: list[str] = []
    summary_paths: dict[str, Path] = {}
    for key, path in args.summary:
        if key in summary_paths:
            sys.exit(f"ERROR: --summary {key!r} appears more than once")
        if not path.exists():
            sys.exit(f"ERROR: --summary {key}: file not found: {path}")
        product_keys.append(key)
        summary_paths[key] = path

    missing_overrides: dict[str, Path] = {}
    for key, path in args.missing:
        if key not in summary_paths:
            sys.exit(f"ERROR: --missing {key!r} has no matching --summary")
        missing_overrides[key] = path

    active_sources: dict[str, Path] = {}
    for key, path in args.active:
        if key not in summary_paths:
            sys.exit(f"ERROR: --active {key!r} has no matching --summary")
        if key in missing_overrides:
            sys.exit(
                f"ERROR: --active and --missing both supplied for "
                f"{key!r}; choose one (active replaces missing for that product)."
            )
        active_sources[key] = path

    config = load_timestep_config(Path(args.timestep_config))
    step_minutes = int(config["step_minutes"])

    filter_by_product: dict[str, set[int]] = {}
    for key in product_keys:
        ts_key = PRODUCT_LAYOUT[key]["tsconfig_product"]
        block = config.get("products", {}).get(ts_key, {})
        flt = block.get("filter")
        filter_by_product[key] = set(int(m) for m in flt) if flt else set()

    print("=" * 70)
    print("Cross-product timestep intersection (Step 4.2)")
    print("=" * 70)
    print(f"step_minutes  : {step_minutes}")
    for key in product_keys:
        flt = sorted(filter_by_product[key])
        print(f"  {key:10s} : filter={flt or '(continuous)'} "
              f"<- {summary_paths[key]}")
    print()

    # 1) Dates each product was scanned for (from the summary CSV).
    dates_by_product: dict[str, set[str]] = {}
    for key in product_keys:
        dates_by_product[key] = read_summary_dates(summary_paths[key])
        print(f"  {key:10s} : {len(dates_by_product[key])} dates in "
              f"summary CSV")

    # 2) Presence gate per product: either an active-CSV (--active) or a
    #    missing-JSON (--missing override or auto-discovered).
    missing_by_product: dict[str, set[tuple[str, str]]] = {}
    active_by_product: dict[str, set[tuple[str, str]] | None] = {}
    for key in product_keys:
        if key in active_sources:
            apath = active_sources[key]
            active_by_product[key] = load_active(key, apath)
            missing_by_product[key] = set()
            print(f"  {key:10s} : {len(active_by_product[key])} active "
                  f"(date, HHMM) pairs <- {apath.name}")
        else:
            active_by_product[key] = None
            if key in missing_overrides:
                mpath = missing_overrides[key]
            else:
                mpath = (summary_paths[key].parent
                         / PRODUCT_LAYOUT[key]["missing_name"])
            missing_by_product[key] = load_missing(key, mpath)
            print(f"  {key:10s} : {len(missing_by_product[key])} missing "
                  f"(date, HHMM) pairs <- {mpath.name}")

    # 3) Error logs.
    error_pairs: set[tuple[str, str]] = set()
    for raw in args.errors_log:
        p = Path(raw)
        pairs = parse_error_log(p)
        error_pairs |= pairs
        print(f"  errors_log {p}: {len(pairs)} (date, HHMM) pairs")

    # 4) Master grid = union of dates seen across the active products.
    all_dates = sorted({d for s in dates_by_product.values() for d in s})
    if not all_dates:
        sys.exit("ERROR: no dates listed in any --summary CSV. Re-run "
                 "the summarize_*.py scripts first.")
    master = build_master_grid(all_dates, step_minutes)
    print(f"\nMaster grid   : {len(master)} slots "
          f"({len(all_dates)} dates x {24 * 60 // step_minutes}/day)")

    # 5) Intersect.
    kept, dropped = intersect(
        master, product_keys, missing_by_product, active_by_product,
        filter_by_product, error_pairs, dates_by_product,
    )
    print(f"Kept          : {len(kept)} timesteps "
          f"({len(kept) / max(1, len(master)) * 100:.1f}% of master grid)")
    for r in ["unscanned_date"] + product_keys + ["error_log"]:
        n = len(dropped[r])
        if n:
            print(f"  dropped by {r:14s}: {n}")

    # 6) Outputs.
    write_manifest(kept, product_keys, Path(args.output_csv))
    write_plot(kept, dropped, all_dates, step_minutes, product_keys,
               Path(args.output_plot))

    return 0


if __name__ == "__main__":
    sys.exit(main())
