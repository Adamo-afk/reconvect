"""
intersect_product_coverage.py — Keep train/val/test consistent across products.

Pipeline placement (Step 4.2):

    Step 4.1  extract_patch_seq_for_datasets.py    train/val/test CSVs from
                                                   the patch activity index
    Step 4.2  intersect_product_coverage.py        THIS SCRIPT — reads any
                                                   subset of per-product
                                                   summary CSVs and filters
                                                   train/val/test to the
                                                   dates where every chosen
                                                   product has data
    Step 4.3  compute_normalization_stats.py       per-variable mean/std on
                                                   the filtered training set
    Step 5    create_datasets.py                   build TF datasets

Running this step before normalization is deliberate: the stats script
then computes on a smaller, fully consistent set of timesteps instead of
including samples that will later be discarded for missing inputs.

Inputs
------
Each `--summary` argument names one product, the per-product summary CSV
produced by a `summarize_*.py` script, and the column to read. The
script works with **any** combination — pass only the products you'll
actually feed to the model, skip the rest. With/without NWCSAF, with/
without lightning, with/without OPERA — all valid.

    --summary KEY=PATH[:COLUMN]

Default COLUMN is `coverage_pct` and works for every standard
`summarize_*.py` output (MTG, NWCSAF, OPERA). Use `:kept` for the
lightning summary's boolean flag, or one of OPERA's diagnostic
per-product columns (`:opera_reflectivity_coverage_pct` /
`:opera_rainfall_rate_coverage_pct`) only when you want to gate on a
single OPERA product instead of the intersection.

Per-product thresholds default to `--min_coverage` (default 100.0,
"full coverage required"). Override per product with
`--threshold KEY=VALUE` — e.g. `--threshold lightning=1` for the boolean
`kept` flag, or `--threshold mtg=80` to relax MTG to 80%.

Outputs (under --output_dir, default `our_data/`)
-------------------------------------------------
    consistent_dates.csv
        date, kept,
        <key1>_value, <key1>_ok, <key2>_value, <key2>_ok, ...
        — every date encountered in any CSV, with per-product values
        and the final keep flag.

    train_data_consistent.csv
    validation_data_consistent.csv
    test_data_consistent.csv
        — copies of the original CSVs with rows whose `date` is not in
        the kept set removed.

    intersect_product_coverage.json
        — manifest with all CLI args, per-product source paths,
        thresholds, and per-split row counts (for reproducibility).

Pass `--in_place` to overwrite the originals instead of writing
`_consistent.csv` copies. Pass `--copy_to_canonical` to write the
filtered files as `train_data.csv` / `validation_data.csv` /
`test_data.csv` (still keeping the originals as `*_original.csv`).

Examples
--------
    # MTG + NWCSAF + OPERA (both products) + lightning — full coverage required
    python intersect_product_coverage.py \\
        --summary mtg=mtg_summary.csv \\
        --summary nwcsaf=nwcsaf_summary.csv \\
        --summary opera=opera_summary.csv \\
        --summary lightning=lightning_summary.csv:kept \\
        --threshold lightning=1

    # No NWCSAF, no lightning, just radar + MTG, relaxed to 80%
    python intersect_product_coverage.py \\
        --summary mtg=mtg_summary.csv:coverage_pct \\
        --min_coverage 80
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"

# Default column to read when the user doesn't supply one
DEFAULT_COLUMN = "coverage_pct"

# Regexes for the filename forms that show up in `regrid_<cat>.log`.
# Each one captures (date, hhmm). Order matters — the most specific
# pattern is tried first.
_ERROR_FILENAME_PATTERNS = (
    # OPERA ISO: 2026-03-27T174500Z-rainfall_rate-composite-opera.h5
    re.compile(r'(?P<date>\d{4}-\d{2}-\d{2})T(?P<hhmm>\d{4})\d{2}Z'),
    # NWCSAF compact ISO: S_NWC_CMIC_..._20260314T084000Z.nc
    re.compile(r'_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})T(?P<hhmm>\d{4})\d{2}Z'),
    # Radar / MTG / NWCSAF outputs: nc4_2026-04-14-Romania_1610_vis_06.npy
    re.compile(r'(?P<date>\d{4}-\d{2}-\d{2})-Romania_(?P<hhmm>\d{4})_'),
    # Lightning outputs: lightning_density_20260314_0840.npy
    re.compile(r'_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_(?P<hhmm>\d{4})'),
)

# Matches a Python-like identifier — used to tell a column name apart from
# the tail of a Windows path when splitting `KEY=PATH:COLUMN`.
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


# =============================================================================
# CLI arg parsing
# =============================================================================

def parse_summary_arg(arg: str) -> tuple[str, Path, str]:
    """
    Parse `key=path[:column]`. Robust to Windows drive letters (`C:`)
    because the right-most segment is only treated as a column when it
    looks like an identifier (no slashes/backslashes).
    """
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--summary must look like 'key=path[:column]', got {arg!r}"
        )
    key, rest = arg.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"--summary entry has empty key in {arg!r}"
        )
    rest = rest.strip()

    column = DEFAULT_COLUMN
    if ":" in rest:
        # Split on the rightmost ':'. Treat the tail as a column only if it
        # has no path separators AND looks like a column identifier.
        head, tail = rest.rsplit(":", 1)
        tail = tail.strip()
        if ("/" not in tail and "\\" not in tail
                and _IDENTIFIER_RE.match(tail)):
            rest = head.strip()
            column = tail
    return key, Path(rest), column


def parse_threshold_arg(arg: str) -> tuple[str, float]:
    """Parse `key=value` into (key, float)."""
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--threshold must look like 'key=value', got {arg!r}"
        )
    key, value = arg.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"--threshold has empty key in {arg!r}"
        )
    try:
        return key, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--threshold value must be numeric, got {value!r}"
        )


# =============================================================================
# Summary parsing
# =============================================================================

def load_summary_coverage(csv_path: Path, column: str) -> dict[str, float]:
    """Return {date: numeric_value} read from `column` in `csv_path`."""
    if not csv_path.exists():
        sys.exit(f"ERROR: summary CSV not found: {csv_path}")

    out: dict[str, float] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            sys.exit(f"ERROR: {csv_path}: no `date` column "
                     f"(have: {reader.fieldnames})")
        if column not in reader.fieldnames:
            sys.exit(f"ERROR: {csv_path}: column {column!r} not in "
                     f"{reader.fieldnames}")
        for row in reader:
            date = row.get("date", "").strip()
            if not date:
                continue
            raw = row.get(column, "").strip()
            if raw == "":
                continue
            try:
                out[date] = float(raw)
            except ValueError:
                continue
    return out


# =============================================================================
# Intersection
# =============================================================================

def intersect(coverage_by_product: dict[str, dict[str, float]],
              threshold_by_product: dict[str, float]
              ) -> tuple[set[str], list[dict]]:
    """
    Build the per-date keep decision. A date is kept iff every product in
    `coverage_by_product` has a value >= its threshold. Dates missing from
    a product's CSV count as failing for that product.
    """
    all_dates: set[str] = set()
    for d in coverage_by_product.values():
        all_dates.update(d.keys())

    kept_dates: set[str] = set()
    rows: list[dict] = []
    for date in sorted(all_dates):
        row: dict = {"date": date}
        ok = True
        for key, cov_map in coverage_by_product.items():
            thr = threshold_by_product[key]
            val = cov_map.get(date)
            row[f"{key}_value"] = "" if val is None else f"{val:.4f}"
            present = val is not None and val >= thr
            row[f"{key}_ok"] = 1 if present else 0
            if not present:
                ok = False
        row["kept"] = 1 if ok else 0
        rows.append(row)
        if ok:
            kept_dates.add(date)
    return kept_dates, rows


# =============================================================================
# CSV filtering
# =============================================================================

def parse_error_log(log_path: Path) -> set[tuple[str, str]]:
    """Read a `regrid_<category>.log` file and return the (date, hhmm) pairs.

    Lines that don't begin with `ERROR ` or whose filename can't be parsed
    are silently ignored — defensive against partial / hand-edited logs.
    """
    pairs: set[tuple[str, str]] = set()
    if not log_path.exists():
        print(f"  ERROR log not found, skipping: {log_path}")
        return pairs
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("ERROR "):
                continue
            # `ERROR <filename>: <message>` — pull the filename out
            after = line[len("ERROR "):]
            fname = after.split(":", 1)[0].strip()
            for pat in _ERROR_FILENAME_PATTERNS:
                m = pat.search(fname)
                if not m:
                    continue
                groups = m.groupdict()
                if "date" in groups:
                    date_str = groups["date"]
                else:
                    date_str = f"{groups['y']}-{groups['m']}-{groups['d']}"
                pairs.add((date_str, groups["hhmm"]))
                break
    return pairs


def _row_timesteps(row: dict, idx_cols: list[str]) -> set[tuple[str, str]]:
    """Compute the set of (date, HHMM) timesteps a sequence row covers.

    `idx_cols` are the per-step columns like `idx_t-30`, `idx_t0`,
    `idx_t+45` — the numeric suffix is the minute offset from
    `reference_utc`. Day rollover is handled via a real datetime add.
    """
    date_str = row.get("date", "").strip()
    ref = row.get("reference_utc", "").strip()
    if not date_str or not ref:
        return set()
    try:
        ref_dt = datetime.strptime(f"{date_str}T{ref}", "%Y-%m-%dT%H:%M")
    except ValueError:
        return set()
    out: set[tuple[str, str]] = set()
    for col in idx_cols:
        m = re.match(r"^idx_t(?P<sign>[+-]?)(?P<n>\d+)$", col)
        if not m:
            continue
        offset = int(m.group("n"))
        if m.group("sign") == "-":
            offset = -offset
        step_dt = ref_dt + timedelta(minutes=offset)
        out.add((step_dt.strftime("%Y-%m-%d"),
                 step_dt.strftime("%H%M")))
    return out


def filter_csv(input_csv: Path, output_csv: Path,
               kept_dates: set[str],
               error_pairs: set[tuple[str, str]] | None = None,
               ) -> tuple[int, int, int]:
    """Copy `input_csv` to `output_csv`, dropping rows whose date isn't kept
    OR whose timesteps overlap with `error_pairs` (regrid ERROR set).

    Returns (rows_in, rows_kept, rows_dropped_for_errors).
    """
    if not input_csv.exists():
        print(f"  SKIP {input_csv} (not found)")
        return 0, 0, 0
    rows_in = rows_kept = rows_dropped_for_errors = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    error_pairs = error_pairs or set()
    with open(input_csv, "r", newline="") as f_in, \
         open(output_csv, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            sys.exit(f"ERROR: {input_csv}: missing `date` column")
        idx_cols = [c for c in reader.fieldnames if c.startswith("idx_t")]
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            rows_in += 1
            if row.get("date", "").strip() not in kept_dates:
                continue
            if error_pairs:
                steps = _row_timesteps(row, idx_cols)
                if steps & error_pairs:
                    rows_dropped_for_errors += 1
                    continue
            writer.writerow(row)
            rows_kept += 1
    return rows_in, rows_kept, rows_dropped_for_errors


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intersect per-product coverage CSVs and filter "
                    "train/val/test to the dates where every requested "
                    "product has data. Step 4.2 — runs after the "
                    "per-product summarisers and before "
                    "compute_normalization_stats.py."
    )
    parser.add_argument(
        "--summary", action="append", required=True,
        metavar="KEY=PATH[:COLUMN]",
        help="Per-product summary CSV. Repeat once per product. "
             "Default COLUMN is 'coverage_pct'. Skipping a product is "
             "fine — only the products you list are required to be "
             "present.",
    )
    parser.add_argument(
        "--min_coverage", type=float, default=100.0,
        help="Global threshold applied to every --summary that doesn't have "
             "a matching --threshold override (default: 100.0). For "
             "boolean-flag columns (e.g. lightning's `kept`), set this "
             "via --threshold instead of relying on the default.",
    )
    parser.add_argument(
        "--threshold", action="append", default=[], type=parse_threshold_arg,
        metavar="KEY=VALUE",
        help="Per-product threshold override. Repeat for each product "
             "that needs a different threshold to --min_coverage.",
    )
    parser.add_argument(
        "--train_csv", type=str,
        default=str(DEFAULT_DATA_ROOT / "train_data.csv"),
    )
    parser.add_argument(
        "--val_csv", type=str,
        default=str(DEFAULT_DATA_ROOT / "validation_data.csv"),
    )
    parser.add_argument(
        "--test_csv", type=str,
        default=str(DEFAULT_DATA_ROOT / "test_data.csv"),
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_DATA_ROOT),
        help=f"Where to write consistent_dates.csv and the filtered "
             f"train/val/test CSVs (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--in_place", action="store_true",
        help="Overwrite the original train_data.csv / validation_data.csv "
             "/ test_data.csv files. Originals are renamed to "
             "*_original.csv before being replaced.",
    )
    parser.add_argument(
        "--errors_log", action="append", default=[], metavar="PATH",
        help="Path to a `regrid_<category>.log` file (or `errors.txt`) "
             "produced by regrid.py. Sequence rows whose timesteps "
             "intersect any (date, HHMM) parsed from these logs are "
             "dropped in addition to the per-date coverage filter. "
             "Repeat for each log (radar, MTG, NWCSAF, OPERA, ...).",
    )

    args = parser.parse_args()

    # Collect all (date, HHMM) error pairs from the supplied logs.
    error_pairs: set[tuple[str, str]] = set()
    error_log_summaries: list[dict] = []
    for raw in args.errors_log:
        log_path = Path(raw)
        pairs = parse_error_log(log_path)
        error_pairs |= pairs
        error_log_summaries.append({
            "path":  str(log_path),
            "pairs": len(pairs),
        })
        print(f"  errors_log {log_path}: {len(pairs)} (date, HHMM) pairs")

    # Parse summary triples
    coverage_by_product: dict[str, dict[str, float]] = {}
    source_by_product: dict[str, dict] = {}
    for raw in args.summary:
        key, csv_path, column = parse_summary_arg(raw)
        if key in coverage_by_product:
            sys.exit(f"ERROR: duplicate --summary key {key!r}")
        cov = load_summary_coverage(csv_path, column)
        coverage_by_product[key] = cov
        source_by_product[key] = {
            "csv": str(csv_path),
            "column": column,
            "n_dates": len(cov),
        }

    # Resolve per-product thresholds
    threshold_by_product: dict[str, float] = {
        k: args.min_coverage for k in coverage_by_product
    }
    for k, v in args.threshold:
        if k not in threshold_by_product:
            sys.exit(f"ERROR: --threshold {k}=... has no matching --summary")
        threshold_by_product[k] = v

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cross-product coverage intersection (Step 4.2)")
    print("=" * 70)
    print(f"Min coverage (global) : {args.min_coverage}")
    for key, meta in source_by_product.items():
        thr = threshold_by_product[key]
        print(f"  {key:14s} : {meta['csv']}")
        print(f"      column={meta['column']}, threshold={thr}, "
              f"{meta['n_dates']} dates")
    print()

    kept_dates, rows = intersect(coverage_by_product, threshold_by_product)

    # Decision table
    decisions_path = output_dir / "consistent_dates.csv"
    fieldnames = ["date"]
    for key in coverage_by_product.keys():
        fieldnames.extend([f"{key}_value", f"{key}_ok"])
    fieldnames.append("kept")
    with open(decisions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {decisions_path}  ({len(rows)} dates total, "
          f"{len(kept_dates)} kept)")

    # Filter train/val/test
    splits = [
        ("train",      Path(args.train_csv)),
        ("validation", Path(args.val_csv)),
        ("test",       Path(args.test_csv)),
    ]
    split_summaries: dict[str, dict] = {}
    for split, input_csv in splits:
        if args.in_place:
            backup_csv = input_csv.with_name(input_csv.stem + "_original" + input_csv.suffix)
            tmp_csv = output_dir / f"{split}_data_consistent.tmp"
            rows_in, rows_kept, rows_dropped_err = filter_csv(
                input_csv, tmp_csv, kept_dates, error_pairs,
            )
            if input_csv.exists():
                if not backup_csv.exists():
                    shutil.copy2(input_csv, backup_csv)
                if tmp_csv.exists():
                    shutil.move(str(tmp_csv), str(input_csv))
            out_path = input_csv
        else:
            out_path = output_dir / f"{split}_data_consistent.csv"
            rows_in, rows_kept, rows_dropped_err = filter_csv(
                input_csv, out_path, kept_dates, error_pairs,
            )

        split_summaries[split] = {
            "input":              str(input_csv),
            "output":             str(out_path),
            "rows_in":            rows_in,
            "rows_kept":          rows_kept,
            "rows_dropped_error": rows_dropped_err,
        }
        if rows_in:
            pct = rows_kept / rows_in * 100
        else:
            pct = 0
        extra = (f" ({rows_dropped_err} dropped for regrid errors)"
                 if rows_dropped_err else "")
        print(f"  {split:10s}: {rows_kept}/{rows_in} rows kept "
              f"({pct:.1f}%){extra}  ->  {out_path}")

    # Manifest
    manifest_path = output_dir / "intersect_product_coverage.json"
    manifest = {
        "computed_utc": datetime.now(timezone.utc)
                                .isoformat(timespec="seconds"),
        "min_coverage": args.min_coverage,
        "in_place":     bool(args.in_place),
        "sources":      source_by_product,
        "thresholds":   threshold_by_product,
        "error_logs":   error_log_summaries,
        "n_error_pairs": len(error_pairs),
        "n_dates_seen": len(rows),
        "n_dates_kept": len(kept_dates),
        "splits":       split_summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
