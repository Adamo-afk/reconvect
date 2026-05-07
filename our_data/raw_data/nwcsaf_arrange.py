"""
Arrange NWCSAF CMIC and CTTH NetCDF files into COALITION-4 date directories.

Reads NWCSAF products from their flat source directories and organizes them
into per-date folders where both CMIC and CTTH timestep files coexist, so
COALITION-4's patch extraction can access all NWCSAF variables for a given
date from a single directory.

Source structure (flat):
    nwcsaf/cmic/S_NWC_CMIC_MSG4_MSG-N-VISIR_20240613T115500Z.nc
    nwcsaf/ctth/S_NWC_CTTH_MSG4_MSG-N-VISIR_20240613T115500Z.nc

Target structure (COALITION-4):
    {target_root}/YYYY-MM-DD-Romania/
        S_NWC_CMIC_MSG4_MSG-N-VISIR_20240613T115500Z.nc
        S_NWC_CTTH_MSG4_MSG-N-VISIR_20240613T115500Z.nc
        ...all timesteps for that date from both products...

Filename parsing:
    S_NWC_{PRODUCT}_{SAT}_{REGION}_{YYYYMMDD}T{HHMMSS}Z.nc
    The date is extracted from the timestamp segment.

Usage:
    # Arrange all NWCSAF data (default paths)
    python arrange_nwcsaf_data.py

    # Custom paths
    python arrange_nwcsaf_data.py -s D:/nwcsaf -t F:/nowcasting/coalition4-rcnn/our_data/nwcsaf_data

    # Specific products only
    python arrange_nwcsaf_data.py --products cmic

    # Copy instead of move
    python arrange_nwcsaf_data.py --copy

    # Dry run
    python arrange_nwcsaf_data.py --dry-run
"""

import os
import re
import shutil
import argparse
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

# Default source root (flat NWCSAF directories)
DEFAULT_SOURCE_ROOT = r"nwcsaf"

# Default target root (COALITION-4 date-based structure)
DEFAULT_TARGET_ROOT = (
    r"F:\nowcasting\coalition4-rcnn\our_data\nwcsaf_data"
)

# NWCSAF product subdirectories to process
NWCSAF_PRODUCTS = ["cmic", "ctth"]

# Regex to extract the timestamp from NWCSAF filenames
# Matches: S_NWC_{PRODUCT}_{SAT}_{REGION}_{YYYYMMDD}T{HHMMSS}Z.nc
FILENAME_PATTERN = re.compile(
    r'^S_NWC_\w+_\w+_[\w-]+_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z\.nc$'
)


# =============================================================================
# Core logic
# =============================================================================

def parse_date_from_filename(filename):
    """
    Extract the date from an NWCSAF filename.

    Args:
        filename (str): e.g. 'S_NWC_CMIC_MSG4_MSG-N-VISIR_20240613T115500Z.nc'

    Returns:
        str or None: Date as 'YYYY-MM-DD', or None if parsing fails
    """
    basename = os.path.basename(filename)
    match = FILENAME_PATTERN.match(basename)
    if not match:
        return None

    year, month, day = match.group(1), match.group(2), match.group(3)
    return f"{year}-{month}-{day}"


def discover_source_files(source_root, product):
    """
    Find all NetCDF files in a flat NWCSAF product directory.

    Args:
        source_root (str): Root source directory
        product (str): Product subdirectory name ('cmic' or 'ctth')

    Returns:
        list: Sorted list of absolute file paths
    """
    source_dir = os.path.join(source_root, product)
    if not os.path.isdir(source_dir):
        print(f"  WARNING: Source directory not found: {source_dir}")
        return []

    all_nc = [f for f in os.listdir(source_dir) if f.endswith('.nc')]
    kept_files = [f for f in all_nc if 'PLAX' not in f]
    skipped = len(all_nc) - len(kept_files)

    if skipped:
        print(f"  Skipped {skipped} PLAX files from {product}/")

    files = sorted(os.path.join(source_dir, f) for f in kept_files)
    return files


def arrange_product(source_root, target_root, product,
                    copy_mode=False, dry_run=False):
    """
    Arrange all files for one NWCSAF product into per-date directories.

    Args:
        source_root (str): Root source directory
        target_root (str): Root target directory
        product (str): Product name ('cmic' or 'ctth')
        copy_mode (bool): Copy (True) or move (False)
        dry_run (bool): Preview only

    Returns:
        dict: Statistics {arranged, skipped_existing, skipped_parse_error}
              plus a set of dates touched
    """
    stats = {"arranged": 0, "skipped_existing": 0, "skipped_parse_error": 0}
    dates_touched = set()
    action_verb = "COPY" if copy_mode else "MOVE"

    files = discover_source_files(source_root, product)
    if not files:
        return stats, dates_touched

    print(f"\n  {product}: {len(files)} source files found")

    for src_path in files:
        basename = os.path.basename(src_path)
        date_str = parse_date_from_filename(basename)

        if date_str is None:
            print(f"    SKIP (cannot parse date): {basename}")
            stats["skipped_parse_error"] += 1
            continue

        dates_touched.add(date_str)

        # Target: {target_root}/YYYY-MM-DD-Romania/{original_filename}
        date_dir = f"{date_str}-Romania"
        dst_path = os.path.join(target_root, date_dir, basename)

        if os.path.exists(dst_path):
            stats["skipped_existing"] += 1
            continue

        if dry_run:
            print(f"    [DRY RUN] {action_verb}: {basename}")
            print(f"      -> {dst_path}")
            stats["arranged"] += 1
            continue

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if copy_mode:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

        stats["arranged"] += 1

    return stats, dates_touched


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Arrange NWCSAF CMIC/CTTH files into per-date COALITION-4 directories."
    )
    parser.add_argument(
        "--source_root", "-s",
        type=str,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root directory of NWCSAF products (default: {DEFAULT_SOURCE_ROOT})"
    )
    parser.add_argument(
        "--target_root", "-t",
        type=str,
        default=DEFAULT_TARGET_ROOT,
        help=f"Root target directory (default: {DEFAULT_TARGET_ROOT})"
    )
    parser.add_argument(
        "--products",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific products to process (default: all). "
             f"Available: {NWCSAF_PRODUCTS}"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without executing them"
    )

    args = parser.parse_args()

    products = args.products if args.products else NWCSAF_PRODUCTS

    # Validate product names
    for p in products:
        if p not in NWCSAF_PRODUCTS:
            print(f"WARNING: Unknown product '{p}'. Available: {NWCSAF_PRODUCTS}")

    products = [p for p in products if p in NWCSAF_PRODUCTS]

    # Header
    print("=" * 70)
    print("NWCSAF Data Arrangement -> COALITION-4 Date Directories")
    print("=" * 70)
    print(f"Source root : {args.source_root}")
    print(f"Target root : {args.target_root}")
    print(f"Products    : {products}")
    print(f"Mode        : {'COPY' if args.copy else 'MOVE'}")
    if args.dry_run:
        print("*** DRY RUN — no files will be written ***")

    # Process each product
    total_stats = defaultdict(int)
    all_dates = set()

    for product in products:
        stats, dates = arrange_product(
            args.source_root, args.target_root, product,
            copy_mode=args.copy, dry_run=args.dry_run,
        )
        for k, v in stats.items():
            total_stats[k] += v
        all_dates.update(dates)

        if stats["arranged"] > 0 or stats["skipped_existing"] > 0:
            print(f"    -> arranged={stats['arranged']}, "
                  f"existing={stats['skipped_existing']}, "
                  f"parse_errors={stats['skipped_parse_error']}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"  Dates covered        : {len(all_dates)}")
    print(f"  Files arranged       : {total_stats['arranged']}")
    print(f"  Skipped (existing)   : {total_stats['skipped_existing']}")
    print(f"  Skipped (parse error): {total_stats['skipped_parse_error']}")

    if all_dates:
        sorted_dates = sorted(all_dates)
        print(f"  Date range           : {sorted_dates[0]} to {sorted_dates[-1]}")


if __name__ == "__main__":
    main()
    