"""
Arrange raw radar NetCDF files into the COALITION-4 directory structure.

Reads radar products from their raw source directories and organizes them
into the same naming convention used by the satellite pipeline, so that
COALITION-4's data loaders can consume both satellite and radar data uniformly.

Only timesteps matching the COALITION-4 cadence are kept. By default this
selects :00, :10, :30, :50 — the native 10-min radar timestamps closest to
the 15-min pipeline grid (:00, :15, :30, :45). This avoids optical flow
interpolation errors that would be introduced by synthesizing :15 and :45.

Source structure (raw):
    {source_root}/{raw_name}/
        e.g. radar/netcdf/rain_rate/2025051512500000dBR_sri.nc

Target structure (COALITION-4):
    {target_root}/{COALITION_NAME}/nc4_YYYY-MM-DD-Romania_{COALITION_NAME}/
        nc4_YYYY-MM-DD-Romania_HHMM_{COALITION_NAME}.nc

Filename parsing:
    Raw filenames begin with 12 digits: YYYYMMDD + HHMM (date + time).
    Everything after is product-specific metadata.

Product mapping (raw_name -> COALITION_NAME):
    acum1h       -> CPCH
    cmax         -> CZC
    rain_rate    -> RZC
    reflectivity -> BZC
    vil          -> LZC
    eht          -> EZC-20

Usage:
    # Arrange all products (default: only :00, :10, :30, :50 timesteps)
    python radar_arrange.py

    # Custom paths
    python radar_arrange.py -s D:/radar/netcdf -t F:/nowcasting/coalition4-rcnn/our_data/radar_data

    # Keep all timesteps (no filtering)
    python radar_arrange.py --timesteps all

    # Custom timestep filter (minutes within each hour)
    python radar_arrange.py --timesteps 00 10 20 30 40 50

    # Specific products only
    python radar_arrange.py --products rain_rate reflectivity

    # Copy instead of move
    python radar_arrange.py --copy

    # Dry run (preview without file operations)
    python radar_arrange.py --dry-run
"""

import os
import re
import shutil
import argparse
from datetime import datetime
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

# Default source root (raw radar NetCDF files)
DEFAULT_SOURCE_ROOT = r"radar\netcdf"

# Default target root (COALITION-4 structure)
DEFAULT_TARGET_ROOT = (
    r"F:\nowcasting\coalition4-rcnn\our_data\radar_data"
)

# Product name mapping: raw directory name -> COALITION-4 product name
PRODUCT_MAP = {
    "acum1h":       "CPCH",
    "cmax":         "CZC",
    "rain_rate":    "RZC",
    "reflectivity": "BZC",
    "vil":          "LZC",
    "eht":          "EZC-20",
}

# Regex to extract date (8 digits) and time (4 digits) from the filename start
FILENAME_PATTERN = re.compile(r'^(\d{8})(\d{4})')

# Default timestep filter: nearest native 10-min timestamps to the 15-min grid.
# :00->:00, :15->:10, :30->:30, :45->:50
DEFAULT_MINUTE_FILTER = {'00', '10', '30', '50'}


# =============================================================================
# Core logic
# =============================================================================

def parse_datetime_from_filename(filename):
    """
    Extract date and time from a radar product filename.
    
    The first 8 digits encode the date (YYYYMMDD) and the next 4 encode
    the time (HHMM). Everything after is product-specific metadata.
    
    Args:
        filename (str): Raw filename, e.g. '2025051512500000dBR_sri.nc'
        
    Returns:
        tuple: (date_str 'YYYY-MM-DD', time_str 'HHMM') or (None, None)
    """
    basename = os.path.basename(filename)
    match = FILENAME_PATTERN.match(basename)
    if not match:
        return None, None

    date_raw = match.group(1)  # e.g. '20250515'
    time_raw = match.group(2)  # e.g. '1250'

    # Validate
    try:
        datetime.strptime(date_raw + time_raw, "%Y%m%d%H%M")
    except ValueError:
        return None, None

    date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
    return date_str, time_raw


def build_target_path(target_root, coalition_name, date_str, time_str):
    """
    Build the COALITION-4 target path for a radar file.
    
    Target layout mirrors the satellite pipeline convention:
        {target_root}/{coalition_name}/nc4_{date}-Romania_{coalition_name}/
            nc4_{date}-Romania_{time}_{coalition_name}.nc
    
    Args:
        target_root (str): Root target directory
        coalition_name (str): COALITION-4 product name (e.g. 'RZC')
        date_str (str): Date in 'YYYY-MM-DD' format
        time_str (str): Time in 'HHMM' format
        
    Returns:
        str: Full target file path
    """
    day_dir = f"nc4_{date_str}-Romania_{coalition_name}"
    filename = f"nc4_{date_str}-Romania_{time_str}_{coalition_name}.nc"
    return os.path.join(target_root, coalition_name, day_dir, filename)


def discover_source_files(source_root, raw_name):
    """
    Find all NetCDF files in a raw product directory.
    
    Args:
        source_root (str): Root source directory
        raw_name (str): Raw product subdirectory name (e.g. 'rain_rate')
        
    Returns:
        list: Sorted list of absolute file paths
    """
    source_dir = os.path.join(source_root, raw_name)
    if not os.path.isdir(source_dir):
        print(f"  WARNING: Source directory not found: {source_dir}")
        return []

    files = sorted(
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith('.nc')
    )
    return files


def arrange_product(source_root, target_root, raw_name, coalition_name,
                    minute_filter=None, copy_mode=False, dry_run=False):
    """
    Arrange all files for one radar product into COALITION-4 structure.

    Args:
        source_root (str): Root source directory
        target_root (str): Root target directory
        raw_name (str): Raw product directory name
        coalition_name (str): COALITION-4 product name
        minute_filter (set or None): Set of allowed minute strings (e.g. {'00','10','30','50'}).
                                      If None, all timesteps are kept.
        copy_mode (bool): If True, copy files; if False, move them
        dry_run (bool): If True, only print actions without executing

    Returns:
        dict: Statistics {arranged, skipped_existing, skipped_parse_error, skipped_timestep}
    """
    stats = {"arranged": 0, "skipped_existing": 0, "skipped_parse_error": 0,
             "skipped_timestep": 0}
    action_verb = "COPY" if copy_mode else "MOVE"

    files = discover_source_files(source_root, raw_name)
    if not files:
        return stats

    print(f"\n  {raw_name} -> {coalition_name}: {len(files)} source files found")

    for src_path in files:
        basename = os.path.basename(src_path)
        date_str, time_str = parse_datetime_from_filename(basename)

        if date_str is None:
            print(f"    SKIP (cannot parse date/time): {basename}")
            stats["skipped_parse_error"] += 1
            continue

        # Filter by minute within the hour
        if minute_filter is not None:
            minute = time_str[2:]  # last 2 digits of HHMM
            if minute not in minute_filter:
                stats["skipped_timestep"] += 1
                continue

        dst_path = build_target_path(target_root, coalition_name,
                                     date_str, time_str)

        if os.path.exists(dst_path):
            stats["skipped_existing"] += 1
            continue

        if dry_run:
            print(f"    [DRY RUN] {action_verb}: {basename}")
            print(f"      -> {dst_path}")
            stats["arranged"] += 1
            continue

        # Create target directory
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Copy or move
        if copy_mode:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

        stats["arranged"] += 1

    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Arrange raw radar NetCDF files into COALITION-4 structure."
    )
    parser.add_argument(
        "--source_root", "-s",
        type=str,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root directory of raw radar products (default: {DEFAULT_SOURCE_ROOT})"
    )
    parser.add_argument(
        "--target_root", "-t",
        type=str,
        default=DEFAULT_TARGET_ROOT,
        help=f"Root target directory for COALITION-4 layout (default: {DEFAULT_TARGET_ROOT})"
    )
    parser.add_argument(
        "--products",
        type=str,
        nargs="+",
        default=None,
        help="Specific raw product names to process (default: all). "
             f"Available: {list(PRODUCT_MAP.keys())}"
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        nargs="+",
        default=None,
        help="Minute values to keep (e.g. 00 10 30 50). "
             "Default: 00 10 30 50 (nearest to 15-min grid). "
             "Use 'all' to keep every timestep."
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

    # Build minute filter
    if args.timesteps is not None and args.timesteps != ['all']:
        minute_filter = set(args.timesteps)
    elif args.timesteps == ['all']:
        minute_filter = None
    else:
        minute_filter = DEFAULT_MINUTE_FILTER

    # Determine which products to process
    if args.products:
        products = {}
        for raw_name in args.products:
            if raw_name not in PRODUCT_MAP:
                print(f"WARNING: Unknown product '{raw_name}'. "
                      f"Available: {list(PRODUCT_MAP.keys())}")
                continue
            products[raw_name] = PRODUCT_MAP[raw_name]
    else:
        products = PRODUCT_MAP

    # Header
    print("=" * 70)
    print("Radar Data Arrangement -> COALITION-4 Structure")
    print("=" * 70)
    print(f"Source root : {args.source_root}")
    print(f"Target root : {args.target_root}")
    print(f"Products    : {len(products)}")
    print(f"Timesteps   : {sorted(minute_filter) if minute_filter else 'all'}")
    print(f"Mode        : {'COPY' if args.copy else 'MOVE'}")
    if args.dry_run:
        print("*** DRY RUN — no files will be written ***")

    # Process each product
    total_stats = defaultdict(int)

    for raw_name, coalition_name in products.items():
        stats = arrange_product(
            args.source_root, args.target_root,
            raw_name, coalition_name,
            minute_filter=minute_filter,
            copy_mode=args.copy,
            dry_run=args.dry_run,
        )
        for k, v in stats.items():
            total_stats[k] += v

        if stats["arranged"] > 0 or stats["skipped_existing"] > 0:
            print(f"    -> arranged={stats['arranged']}, "
                  f"existing={stats['skipped_existing']}, "
                  f"timestep_skipped={stats['skipped_timestep']}, "
                  f"parse_errors={stats['skipped_parse_error']}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"  Files arranged       : {total_stats['arranged']}")
    print(f"  Skipped (existing)   : {total_stats['skipped_existing']}")
    print(f"  Skipped (timestep)   : {total_stats['skipped_timestep']}")
    print(f"  Skipped (parse error): {total_stats['skipped_parse_error']}")


if __name__ == "__main__":
    main()
    