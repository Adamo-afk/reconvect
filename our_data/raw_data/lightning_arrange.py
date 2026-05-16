"""
Arrange lightning KML files into the COALITION-4 date-based directory structure.

Supports three naming conventions:

1. Date-based filenames (original):
    lightning/15_05_2025.kml -> kml_data/2025-05-15/2025-05-15.kml

2. Numbered sequential filenames (--start-date mode):
    lightning.kml        -> kml_data/2026-03-01/2026-03-01.kml  (day 0)
    lightning (1).kml    -> kml_data/2026-03-02/2026-03-02.kml  (day 1)
    lightning (2).kml    -> kml_data/2026-03-03/2026-03-03.kml  (day 2)
    ...
    lightning (100).kml  -> kml_data/2026-06-09/2026-06-09.kml  (day 100)

3. Timestamped sequential filenames (--start-date mode, mixed with above):
    `lightning - YYYY-MM-DDTHHMMSS.fff.kml`
    These come AFTER the highest numbered file and are sorted ascending by
    their embedded timestamp, then assigned consecutive sequence indices:
        first-by-timestamp  -> day 101 (if max numbered index was 100)
        second-by-timestamp -> day 102
        ...
    The embedded timestamp is the download time (used only for ordering);
    the date a file maps to is determined entirely by its sequence
    position, like the numbered files.

Target structure:
    {target_root}/kml_data/yyyy-mm-dd/yyyy-mm-dd.kml

Usage:
    # Date-based filenames (original behavior)
    python lightning_arrange.py -s D:/lightning -t F:/nowcasting/coalition4-rcnn/our_data/lightning_data

    # Sequential filenames (numbered + timestamped, mixed) with start/end date
    python lightning_arrange.py -s D:/lightning --start-date 2026-03-01 --end-date 2026-08-30
    python lightning_arrange.py -s D:/lightning --start-date 2026-03-01 --end-date 2026-08-30 --dry-run
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

# Regex: "lightning - YYYY-MM-DDTHHMMSS.fff.kml"
# The captured group is the full timestamp string for datetime.strptime
# with the format "%Y-%m-%dT%H%M%S.%f".
TIMESTAMPED_PATTERN = re.compile(
    r'^lightning\s*-\s*(\d{4}-\d{2}-\d{2}T\d{6}\.\d+)\.kml$'
)


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
    Find and sort sequential lightning KML files (numbered + timestamped).

    Two filename forms are recognised in --start-date mode:
        1. Numbered:    `lightning.kml` (index 0) and `lightning (N).kml`
           (index N). The number IS the sequence index.
        2. Timestamped: `lightning - YYYY-MM-DDTHHMMSS.fff.kml`. These
           come AFTER the highest numbered index, sorted ascending by
           the embedded timestamp; they receive consecutive indices
           starting at max_numbered_index + 1.

    Returns list of (index, filepath, embedded_ts_or_None) tuples sorted
    by index. `embedded_ts_or_None` is the parsed datetime for
    timestamped files and None for numbered files — useful for the
    dry-run output so the user can audit the ordering.
    """
    if not os.path.isdir(source_root):
        print(f"  WARNING: Source directory not found: {source_root}")
        return []

    numbered: list[tuple[int, str]] = []
    timestamped: list[tuple[datetime, str]] = []
    for f in os.listdir(source_root):
        if not f.endswith('.kml'):
            continue
        m = SEQUENTIAL_PATTERN.match(f)
        if m:
            index = int(m.group(1)) if m.group(1) else 0
            numbered.append((index, os.path.join(source_root, f)))
            continue
        m = TIMESTAMPED_PATTERN.match(f)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S.%f")
            except ValueError:
                # Unparseable embedded timestamp — leave the file behind
                # so the user can spot it instead of mis-ordering it.
                print(f"  WARNING: unparseable embedded timestamp, "
                      f"skipping: {f}")
                continue
            timestamped.append((ts, os.path.join(source_root, f)))

    results: list[tuple[int, str, datetime | None]] = [
        (idx, path, None) for idx, path in sorted(numbered, key=lambda x: x[0])
    ]

    # Timestamped files come after the highest numbered index, sorted
    # ascending by their embedded timestamp (tie-break by filename for
    # determinism if two timestamps happen to coincide).
    if timestamped:
        max_numbered_index = max((idx for idx, _ in numbered), default=-1)
        timestamped.sort(key=lambda x: (x[0], x[1]))
        for offset, (ts, path) in enumerate(timestamped, start=1):
            results.append((max_numbered_index + offset, path, ts))

    return results


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
    Arrange sequential KML files (numbered + timestamped) into date-based
    directories starting from start_date.

    Files are mapped by sequence index: index 0 = start_date,
    index 1 = start_date + 1 day, etc. Indices come from
    discover_sequential_files(): numbered files first, then timestamped
    files (sorted ascending by embedded timestamp) appended after the
    highest numbered index. Only files within [start_date, end_date]
    range are processed.
    """
    stats = {
        "arranged":              0,
        "skipped_existing":      0,
        "skipped_out_of_range":  0,
        "numbered":              0,
        "timestamped":           0,
    }
    mapping = []  # list of (original_filename, date_str, source_kind, embedded_ts)
    action_verb = "COPY" if copy_mode else "MOVE"

    indexed_files = discover_sequential_files(source_root)
    if not indexed_files:
        print("No sequential KML files found "
              "(expected: lightning.kml, lightning (1).kml, "
              "lightning - YYYY-MM-DDTHHMMSS.fff.kml, ...)")
        return stats, mapping

    n_numbered = sum(1 for _, _, ts in indexed_files if ts is None)
    n_timestamped = len(indexed_files) - n_numbered
    print(f"Found {len(indexed_files)} sequential KML files "
          f"({n_numbered} numbered + {n_timestamped} timestamped)")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} "
          f"to {end_date.strftime('%Y-%m-%d')}")

    total_days = (end_date - start_date).days + 1
    print(f"Expected files for range: {total_days}")
    print()

    if dry_run:
        # Header to make per-file lines easy to scan during audit.
        print(f"  {'idx':>4}  {'kind':<11}  {'embedded_ts':<23}  "
              f"{'source':<46}  -> target_date")
        print(f"  {'-' * 4}  {'-' * 11}  {'-' * 23}  {'-' * 46}  -> {'-' * 10}")

    for index, src_path, embedded_ts in indexed_files:
        target_date = start_date + timedelta(days=index)
        is_timestamped = embedded_ts is not None
        kind = "timestamped" if is_timestamped else "numbered"

        if target_date > end_date:
            stats["skipped_out_of_range"] += 1
            if dry_run:
                ts_str = (embedded_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                          if embedded_ts else "")
                print(f"  {index:>4}  {kind:<11}  {ts_str:<23}  "
                      f"{os.path.basename(src_path):<46}  -> "
                      f"SKIP (out of range, would be {target_date.strftime('%Y-%m-%d')})")
            continue

        date_str = target_date.strftime("%Y-%m-%d")
        basename = os.path.basename(src_path)
        mapping.append((
            basename,
            date_str,
            kind,
            embedded_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            if embedded_ts else "",
        ))

        if is_timestamped:
            stats["timestamped"] += 1
        else:
            stats["numbered"] += 1

        # Target: kml_data/yyyy-mm-dd/yyyy-mm-dd.kml
        dst_dir = os.path.join(target_root, "kml_data", date_str)
        dst_path = os.path.join(dst_dir, f"{date_str}.kml")

        if os.path.exists(dst_path):
            stats["skipped_existing"] += 1
            if dry_run:
                ts_str = (embedded_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                          if embedded_ts else "")
                print(f"  {index:>4}  {kind:<11}  {ts_str:<23}  "
                      f"{basename:<46}  -> {date_str} (EXISTS, skip)")
            continue

        if dry_run:
            ts_str = (embedded_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                      if embedded_ts else "")
            print(f"  {index:>4}  {kind:<11}  {ts_str:<23}  "
                  f"{basename:<46}  -> {date_str}")
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
    """Save the filename-to-date mapping as a CSV file.

    Each row: original_filename, assigned_date, source_kind, embedded_ts.
    `source_kind` is `numbered` or `timestamped`; `embedded_ts` is the
    parsed download time for timestamped files (empty otherwise), kept so
    a reviewer can verify ordering decisions after the fact.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_filename', 'assigned_date',
                         'source_kind', 'embedded_ts'])
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
        print(f"    numbered            : {stats['numbered']}")
        print(f"    timestamped         : {stats['timestamped']}")
        print(f"  Skipped (existing)    : {stats['skipped_existing']}")
        print(f"  Skipped (out of range): {stats['skipped_out_of_range']}")

        # Save mapping CSV (also useful as the audit trail for dry-runs,
        # so don't skip emission when --dry-run is set).
        if mapping:
            csv_path = os.path.join(
                args.target_root, "lightning_filename_mapping.csv"
            )
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
    