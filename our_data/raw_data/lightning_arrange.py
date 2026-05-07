"""
Arrange lightning KML files into the COALITION-4 date-based directory structure.

Supports two naming conventions:

1. Date-based filenames (original):
    lightning/15_05_2025.kml -> kml_data/2025-05-15/2025-05-15.kml

2. Sequential filenames (--start-date mode):
    lightning.kml        -> kml_data/2026-03-01/2026-03-01.kml  (day 0)
    lightning (1).kml    -> kml_data/2026-03-02/2026-03-02.kml  (day 1)
    lightning (2).kml    -> kml_data/2026-03-03/2026-03-03.kml  (day 2)
    ...

Target structure:
    {target_root}/kml_data/yyyy-mm-dd/yyyy-mm-dd.kml

Usage:
    # Date-based filenames (original behavior)
    python lightning_arrange.py -s D:/lightning -t F:/nowcasting/coalition4-rcnn/our_data/lightning_data

    # Sequential filenames with start/end date
    python lightning_arrange.py -s D:/lightning --start-date 2026-03-01 --end-date 2026-03-31
    python lightning_arrange.py -s D:/lightning --start-date 2026-03-01 --end-date 2026-03-31 --dry-run
"""

import os
import re
import csv
import shutil
import argparse
from datetime import datetime, timedelta
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SOURCE_ROOT = "lightning"
DEFAULT_TARGET_ROOT = r"F:\nowcasting\coalition4-rcnn\our_data\lightning_data"

# Regex: dd_mm_yyyy.kml  (or dd.mm.yyyy.kml)
FILENAME_PATTERN = re.compile(r'^(\d{2})[_.](\d{2})[_.](\d{4})\.kml$')

# Regex: "lightning.kml" or "lightning (N).kml"
SEQUENTIAL_PATTERN = re.compile(r'^lightning(?:\s*\((\d+)\))?\.kml$')


# =============================================================================
# Core logic
# =============================================================================

def parse_date_from_filename(filename):
    """
    Extract date from a lightning KML filename.

    Args:
        filename (str): e.g. '15_05_2025.kml' or '15.05.2025.kml'

    Returns:
        str or None: Date as 'YYYY-MM-DD', or None if parsing fails
    """
    basename = os.path.basename(filename)
    match = FILENAME_PATTERN.match(basename)
    if not match:
        return None

    day, month, year = match.group(1), match.group(2), match.group(3)

    # Validate
    try:
        datetime.strptime(f"{year}{month}{day}", "%Y%m%d")
    except ValueError:
        return None

    return f"{year}-{month}-{day}"


def discover_source_files(source_root):
    """Find all KML files in the source directory."""
    if not os.path.isdir(source_root):
        print(f"  WARNING: Source directory not found: {source_root}")
        return []

    return sorted(
        os.path.join(source_root, f)
        for f in os.listdir(source_root)
        if f.endswith('.kml')
    )


def discover_sequential_files(source_root):
    """
    Find and sort sequential lightning KML files.

    Returns list of (index, filepath) tuples sorted by index.
    'lightning.kml' has index 0, 'lightning (N).kml' has index N.
    """
    if not os.path.isdir(source_root):
        print(f"  WARNING: Source directory not found: {source_root}")
        return []

    results = []
    for f in os.listdir(source_root):
        if not f.endswith('.kml'):
            continue
        match = SEQUENTIAL_PATTERN.match(f)
        if match:
            index = int(match.group(1)) if match.group(1) else 0
            results.append((index, os.path.join(source_root, f)))

    return sorted(results, key=lambda x: x[0])


def arrange_lightning(source_root, target_root, copy_mode=False, dry_run=False):
    """
    Arrange all KML files into date-based directories.

    Target: {target_root}/kml_data/{yyyy-mm-dd}/{yyyy-mm-dd}.kml
    """
    stats = {"arranged": 0, "skipped_existing": 0, "skipped_parse_error": 0}
    action_verb = "COPY" if copy_mode else "MOVE"

    files = discover_source_files(source_root)
    if not files:
        print("No KML files found.")
        return stats

    print(f"Found {len(files)} KML files")

    for src_path in files:
        basename = os.path.basename(src_path)
        date_str = parse_date_from_filename(basename)

        if date_str is None:
            print(f"  SKIP (cannot parse date): {basename}")
            stats["skipped_parse_error"] += 1
            continue

        # Target: kml_data/yyyy-mm-dd/yyyy-mm-dd.kml
        dst_dir = os.path.join(target_root, "kml_data", date_str)
        dst_path = os.path.join(dst_dir, f"{date_str}.kml")

        if os.path.exists(dst_path):
            stats["skipped_existing"] += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] {action_verb}: {basename} -> {dst_path}")
            stats["arranged"] += 1
            continue

        os.makedirs(dst_dir, exist_ok=True)

        if copy_mode:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

        print(f"  {action_verb}: {basename} -> {dst_path}")
        stats["arranged"] += 1

    return stats


def arrange_lightning_sequential(source_root, target_root, start_date, end_date,
                                 copy_mode=False, dry_run=False):
    """
    Arrange sequential KML files (lightning.kml, lightning (1).kml, ...)
    into date-based directories starting from start_date.

    Files are mapped by index: index 0 = start_date, index 1 = start_date + 1 day, etc.
    Only files within [start_date, end_date] range are processed.
    """
    stats = {"arranged": 0, "skipped_existing": 0, "skipped_out_of_range": 0}
    mapping = []  # list of (original_filename, date_str) for CSV export
    action_verb = "COPY" if copy_mode else "MOVE"

    indexed_files = discover_sequential_files(source_root)
    if not indexed_files:
        print("No sequential KML files found (expected: lightning.kml, lightning (1).kml, ...)")
        return stats, mapping

    print(f"Found {len(indexed_files)} sequential KML files")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    total_days = (end_date - start_date).days + 1
    print(f"Expected files for range: {total_days}")

    for index, src_path in indexed_files:
        target_date = start_date + timedelta(days=index)

        if target_date > end_date:
            stats["skipped_out_of_range"] += 1
            continue

        date_str = target_date.strftime("%Y-%m-%d")
        basename = os.path.basename(src_path)
        mapping.append((basename, date_str))

        # Target: kml_data/yyyy-mm-dd/yyyy-mm-dd.kml
        dst_dir = os.path.join(target_root, "kml_data", date_str)
        dst_path = os.path.join(dst_dir, f"{date_str}.kml")

        if os.path.exists(dst_path):
            stats["skipped_existing"] += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] {action_verb}: {basename} -> {dst_path}")
            stats["arranged"] += 1
            continue

        os.makedirs(dst_dir, exist_ok=True)

        if copy_mode:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

        print(f"  {action_verb}: {basename} -> {dst_path}")
        stats["arranged"] += 1

    return stats, mapping


def save_mapping_csv(mapping, output_path):
    """Save the filename-to-date mapping as a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_filename', 'assigned_date'])
        writer.writerows(mapping)
    print(f"  Mapping saved: {output_path} ({len(mapping)} entries)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Arrange lightning KML files into COALITION-4 date directories."
    )
    parser.add_argument(
        "--source_root", "-s", type=str, default=DEFAULT_SOURCE_ROOT,
        help=f"Source directory with KML files (default: {DEFAULT_SOURCE_ROOT})"
    )
    parser.add_argument(
        "--target_root", "-t", type=str, default=DEFAULT_TARGET_ROOT,
        help=f"Target root directory (default: {DEFAULT_TARGET_ROOT})"
    )
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date (YYYY-MM-DD) for sequential mode. "
             "Enables sequential naming: lightning.kml=day0, lightning (1).kml=day1, ..."
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date (YYYY-MM-DD) for sequential mode (inclusive)"
    )

    args = parser.parse_args()

    # Determine mode
    sequential_mode = args.start_date is not None

    if sequential_mode and args.end_date is None:
        parser.error("--end-date is required when using --start-date")

    print("=" * 70)
    print("Lightning KML Data Arrangement -> COALITION-4 Structure")
    print("=" * 70)
    print(f"Source root : {args.source_root}")
    print(f"Target root : {args.target_root}")
    print(f"Mode        : {'COPY' if args.copy else 'MOVE'}")
    if sequential_mode:
        print(f"Naming      : Sequential (lightning.kml, lightning (1).kml, ...)")
        print(f"Date range  : {args.start_date} to {args.end_date}")
    else:
        print(f"Naming      : Date-based (dd_mm_yyyy.kml)")
    if args.dry_run:
        print("*** DRY RUN ***")

    print()

    if sequential_mode:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        stats, mapping = arrange_lightning_sequential(
            args.source_root, args.target_root,
            start_date, end_date,
            copy_mode=args.copy, dry_run=args.dry_run,
        )
        print(f"\n{'=' * 70}")
        print(f"Summary")
        print(f"{'=' * 70}")
        print(f"  Files arranged        : {stats['arranged']}")
        print(f"  Skipped (existing)    : {stats['skipped_existing']}")
        print(f"  Skipped (out of range): {stats['skipped_out_of_range']}")

        # Save mapping CSV
        if mapping:
            csv_path = os.path.join(args.target_root, "lightning_filename_mapping.csv")
            save_mapping_csv(mapping, csv_path)
    else:
        stats = arrange_lightning(
            args.source_root, args.target_root,
            copy_mode=args.copy, dry_run=args.dry_run,
        )
        print(f"\n{'=' * 70}")
        print(f"Summary")
        print(f"{'=' * 70}")
        print(f"  Files arranged       : {stats['arranged']}")
        print(f"  Skipped (existing)   : {stats['skipped_existing']}")
        print(f"  Skipped (parse error): {stats['skipped_parse_error']}")


if __name__ == "__main__":
    main()
    