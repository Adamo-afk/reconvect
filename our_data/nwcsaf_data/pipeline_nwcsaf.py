"""
pipeline_nwcsaf.py — NWCSAF L2 (CMIC + CTTH) SFTP downloader + arranger

End-to-end ingestion of NWCSAF L2 NetCDF files from the SAFNWC archive on
the ANM internal network. Combines the SFTP transfer pattern from
`pipeline_msg_mtg.py` with the filename-based filtering from
`our_data/raw_data/nwcsaf_arrange.py`.

Two-stage workflow (both in this single script):

    1. SFTP download to a flat cache directory
       (default: `our_data/nwcsaf_data/_raw_data/`)
       The cache is kept around for resumability; already-present files are
       skipped without re-transferring.

    2. Arrange the cache into per-date COALITION-4 directories
       (`our_data/nwcsaf_data/{YYYY-MM-DD}-Romania/`) by either copying or
       hard-linking each file. The cache stays intact.

Filters applied **before** transfer (server-side `ls` globs + client-side
checks) so off-grid or non-matching files are never downloaded:

    1. Date range            — sensing timestamp must fall in [start, end]
    2. Minute-of-hour filter — read from `our_data/timestep_config.json`
                               (override with --timesteps NN NN ...)
    3. Product allowlist     — only CMIC and CTTH (override with --products)
    4. PLAX exclusion        — files containing 'PLAX' in the name are skipped

Usage:
    # 1. Pick the cadence first
    python validate_timestep.py --step_minutes 15

    # 2. Download + arrange NWCSAF for a date range
    python our_data/nwcsaf_data/pipeline_nwcsaf.py \\
        --start 2026/02/01-0000 \\
        --end   2026/04/01-0000 \\
        --password_file password.txt

    # Skip the arrange step (download only, leave files in _raw_data/)
    python our_data/nwcsaf_data/pipeline_nwcsaf.py \\
        --start ... --end ... --password_file ... --no_arrange

    # Skip the download step (re-arrange what is already in the cache)
    python our_data/nwcsaf_data/pipeline_nwcsaf.py \\
        --start ... --end ... --password_file ... --skip_download
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    import paramiko
except ImportError:
    print(
        "ERROR: paramiko is required for SFTP transfers.\n"
        "Install with: pip install paramiko",
        file=sys.stderr,
    )
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"

# Remote SAFNWC archive (ANM internal network)
REMOTE_HOST = "192.168.11.212"
REMOTE_USER = "safnwc"
REMOTE_DIRS = {
    "cmic": "/home/safnwc/prod_arch/CMIC/",
    "ctth": "/home/safnwc/prod_arch/CTTH/",
}

# Default local SFTP cache (flat)
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent / "_raw_data"
)
# Default arranged output root — per-date directories live alongside the cache
DEFAULT_ARRANGED_ROOT = Path(__file__).resolve().parent

NWCSAF_PRODUCTS = ["cmic", "ctth"]

# S_NWC_{PRODUCT}_{SAT}_{REGION}_{YYYYMMDD}T{HHMMSS}Z.nc
FILENAME_PATTERN = re.compile(
    r'^S_NWC_(?P<product>\w+)_\w+_[\w-]+_'
    r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    r'T(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z'
    r'(?P<suffix>.*?)\.nc$'
)


# =============================================================================
# Timestep configuration
# =============================================================================

def load_timestep_filter(product_key: str = "nwcsaf"):
    """
    Load the per-product minute filter from timestep_config.json.

    Returns (set of int minutes, step_minutes). Errors out with a clear
    message if the config is missing — the pipeline must not run without an
    explicit cadence decision.
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
    flt = cfg["products"].get(product_key, {}).get("filter")
    if flt is None:
        print(
            f"ERROR: product '{product_key}' has no minute filter "
            f"in {TIMESTEP_CONFIG_PATH}",
            file=sys.stderr,
        )
        sys.exit(2)
    return set(flt), cfg["step_minutes"]


# =============================================================================
# Shared utilities
# =============================================================================

def parse_date_range(start_str, end_str, fmt='%Y/%m/%d-%H%M'):
    """Parse start and end datetime strings."""
    try:
        return (
            datetime.datetime.strptime(start_str, fmt),
            datetime.datetime.strptime(end_str, fmt),
        )
    except ValueError as e:
        print(f"Error parsing dates. Expected '{fmt}'. {e}", file=sys.stderr)
        return None, None


def parse_nwcsaf_filename(filename: str) -> dict | None:
    """
    Extract metadata from a SAFNWC L2 filename.

    Returns a dict with product / sensing_dt / minute / is_plax, or None if
    the filename doesn't match the expected pattern.
    """
    match = FILENAME_PATTERN.match(os.path.basename(filename))
    if not match:
        return None

    try:
        sensing_dt = datetime.datetime(
            int(match['year']), int(match['month']), int(match['day']),
            int(match['hour']), int(match['minute']), int(match['second']),
        )
    except ValueError:
        return None

    return {
        'product':   match['product'].lower(),
        'sensing_dt': sensing_dt,
        'minute':    int(match['minute']),
        'is_plax':   'PLAX' in (match['suffix'] or ''),
    }


def read_password(password_file: str) -> str:
    path = Path(password_file)
    if not path.exists():
        print(f"ERROR: Password file not found: {password_file}",
              file=sys.stderr)
        sys.exit(1)
    return path.read_text().strip()


# =============================================================================
# SFTP download
# =============================================================================

def download_nwcsaf_sftp(
    password_file: str,
    local_dir: Path,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    products: list[str],
    minute_filter: set[int] | None,
    remote_host: str = REMOTE_HOST,
    remote_user: str = REMOTE_USER,
    remote_dirs: dict[str, str] = REMOTE_DIRS,
) -> dict:
    """
    Download NWCSAF files via SFTP, applying date / minute / PLAX filters
    before transfer.

    Returns a stats dict.
    """
    stats = {
        'downloaded': 0,
        'skipped_existing': 0,
        'skipped_plax': 0,
        'skipped_date': 0,
        'skipped_minute': 0,
        'skipped_unparsable': 0,
        'errors': 0,
    }

    password = read_password(password_file)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {remote_user}@{remote_host}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(remote_host, username=remote_user, password=password)
    except Exception as e:
        print(f"ERROR: SSH connection failed: {e}", file=sys.stderr)
        return stats

    try:
        sftp = ssh.open_sftp()
    except Exception as e:
        print(f"ERROR: SFTP session failed: {e}", file=sys.stderr)
        ssh.close()
        return stats

    # Build YYYYMMDD date prefixes for server-side glob filtering
    date_prefixes = []
    d = start_dt.date()
    while d <= end_dt.date():
        date_prefixes.append(d.strftime('%Y%m%d'))
        d += datetime.timedelta(days=1)

    to_download: list[tuple[str, str]] = []  # (remote_path, local_path)

    for product in products:
        remote_dir = remote_dirs.get(product)
        if remote_dir is None:
            print(f"  WARNING: no remote directory configured for '{product}'")
            continue

        print(f"\nListing remote {product.upper()}: {remote_dir}")

        # Server-side filter via `ls` with date-prefix globs
        globs = [
            f"{remote_dir}S_NWC_*_{prefix}T*.nc"
            for prefix in date_prefixes
        ]
        remote_files: set[str] = set()
        batch_size = 50
        for i in range(0, len(globs), batch_size):
            batch = globs[i:i + batch_size]
            cmd = "ls " + " ".join(batch) + " 2>/dev/null"
            _, stdout, _ = ssh.exec_command(cmd)
            for line in stdout:
                line = line.strip()
                if line:
                    remote_files.add(line)

        print(f"  Remote glob matched {len(remote_files)} files")

        # Client-side filtering
        for remote_path in sorted(remote_files):
            filename = os.path.basename(remote_path)
            info = parse_nwcsaf_filename(filename)

            if info is None:
                stats['skipped_unparsable'] += 1
                continue
            if info['is_plax']:
                stats['skipped_plax'] += 1
                continue
            if not (start_dt <= info['sensing_dt'] <= end_dt):
                stats['skipped_date'] += 1
                continue
            if minute_filter is not None and info['minute'] not in minute_filter:
                stats['skipped_minute'] += 1
                continue

            local_path = local_dir / filename
            to_download.append((remote_path, str(local_path)))

    print(f"\nFiltered: {len(to_download)} files queued for download")
    if not to_download:
        sftp.close()
        ssh.close()
        return stats

    total = len(to_download)
    for i, (remote_path, local_path) in enumerate(to_download, start=1):
        filename = os.path.basename(local_path)
        if os.path.exists(local_path):
            stats['skipped_existing'] += 1
            print(f"  [{i}/{total}] Already local: {filename}")
            continue
        print(f"  [{i}/{total}] Downloading {filename}")
        try:
            sftp.get(remote_path, local_path)
            stats['downloaded'] += 1
        except Exception as e:
            stats['errors'] += 1
            print(f"    ERROR {filename}: {e}", file=sys.stderr)

    sftp.close()
    ssh.close()
    return stats


# =============================================================================
# Arrange step — flat cache → per-date COALITION-4 directories
# =============================================================================

def arrange_from_cache(
    cache_dir: Path,
    arranged_root: Path,
    products: list[str],
    start_dt: datetime.datetime | None = None,
    end_dt: datetime.datetime | None = None,
    use_hardlinks: bool = False,
) -> dict:
    """
    Copy (or hard-link) downloaded NWCSAF files from a flat cache into
    per-date COALITION-4 directories.

    Reads `cache_dir` non-destructively. Files are placed at:
        {arranged_root}/{YYYY-MM-DD}-Romania/<original filename>

    PLAX files are skipped here too (defence in depth — the SFTP step
    already filters them, but if files were dropped in manually they'd
    still be filtered out).

    Args:
        cache_dir:   Flat directory containing S_NWC_*.nc files.
        arranged_root: Root directory under which {date}-Romania/ dirs
                     will be created.
        products:    Product names to include (lowercase, e.g. ['cmic','ctth']).
        start_dt, end_dt: Optional date range — when set, only files whose
                     sensing timestamp falls inside [start_dt, end_dt] are
                     arranged.
        use_hardlinks: If True, hard-link the file instead of copying. Saves
                     disk space on the same volume; falls back to copy if
                     hard-link is not supported.
    Returns:
        Stats dict.
    """
    stats = {
        'arranged': 0,
        'skipped_existing': 0,
        'skipped_plax': 0,
        'skipped_product': 0,
        'skipped_date': 0,
        'skipped_unparsable': 0,
        'errors': 0,
        'dates': set(),
    }

    if not cache_dir.is_dir():
        print(f"  WARNING: cache directory not found: {cache_dir}")
        return stats

    products_upper = {p.upper() for p in products}

    nc_files = sorted(
        f for f in os.listdir(cache_dir) if f.endswith('.nc')
    )
    print(f"\nArranging {len(nc_files)} cached file(s) from {cache_dir}")

    for filename in nc_files:
        info = parse_nwcsaf_filename(filename)
        if info is None:
            stats['skipped_unparsable'] += 1
            continue
        if info['is_plax']:
            stats['skipped_plax'] += 1
            continue
        if info['product'].upper() not in products_upper:
            stats['skipped_product'] += 1
            continue
        if start_dt is not None and end_dt is not None:
            if not (start_dt <= info['sensing_dt'] <= end_dt):
                stats['skipped_date'] += 1
                continue

        date_str = info['sensing_dt'].strftime('%Y-%m-%d')
        dst_dir = arranged_root / f"{date_str}-Romania"
        dst_path = dst_dir / filename

        if dst_path.exists():
            stats['skipped_existing'] += 1
            continue

        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            src_path = cache_dir / filename
            if use_hardlinks:
                try:
                    os.link(src_path, dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            stats['arranged'] += 1
            stats['dates'].add(date_str)
        except Exception as e:
            stats['errors'] += 1
            print(f"    ERROR arranging {filename}: {e}", file=sys.stderr)

    return stats


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SFTP download + arrange of NWCSAF L2 (CMIC + CTTH) with "
                    "cadence and PLAX filtering applied before transfer."
    )
    parser.add_argument('--start', '-s', type=str, required=True,
                        help='Start datetime (yyyy/mm/dd-hhmm)')
    parser.add_argument('--end', '-e', type=str, required=True,
                        help='End datetime (yyyy/mm/dd-hhmm), inclusive.')
    parser.add_argument('--password_file', '-pw', type=str, required=True,
                        help='Path to text file containing the SSH password.')
    parser.add_argument('--cache_dir', '-c', type=str,
                        default=str(DEFAULT_CACHE_DIR),
                        help=f'Flat SFTP cache directory '
                             f'(default: {DEFAULT_CACHE_DIR})')
    parser.add_argument('--arranged_root', '-a', type=str,
                        default=str(DEFAULT_ARRANGED_ROOT),
                        help=f'Root for per-date arranged dirs '
                             f'(default: {DEFAULT_ARRANGED_ROOT})')
    parser.add_argument('--products', type=str, nargs='+',
                        default=NWCSAF_PRODUCTS,
                        choices=NWCSAF_PRODUCTS,
                        help='NWCSAF products to ingest '
                             '(default: cmic ctth)')
    parser.add_argument('--timesteps', type=str, nargs='+', default=None,
                        help="Override the minute filter (e.g. 00 10 30 40). "
                             "Default: read from timestep_config.json. "
                             "'all' keeps every native :00–:50.")
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip SFTP and only arrange files already in '
                             'the cache.')
    parser.add_argument('--no_arrange', action='store_true',
                        help='Skip the per-date arrange step (download only).')
    parser.add_argument('--hardlinks', action='store_true',
                        help='Hard-link rather than copy when arranging '
                             '(saves disk space; falls back to copy).')

    args = parser.parse_args()

    start_dt, end_dt = parse_date_range(args.start, args.end)
    if start_dt is None:
        return 1
    if start_dt > end_dt:
        print(f"ERROR: --start ({args.start}) is after --end ({args.end})",
              file=sys.stderr)
        return 1

    products = [p.lower() for p in args.products]

    # Resolve minute filter
    if args.timesteps is not None and args.timesteps != ['all']:
        minute_filter = {int(m) for m in args.timesteps}
        filter_msg = f"{sorted(minute_filter)} (CLI override)"
    elif args.timesteps == ['all']:
        minute_filter = None
        filter_msg = "none (all native 10-min timestamps)"
    else:
        minute_filter, step = load_timestep_filter("nwcsaf")
        filter_msg = (f"{sorted(minute_filter)} "
                      f"(timestep_config.json, step={step} min)")

    cache_dir = Path(args.cache_dir)
    arranged_root = Path(args.arranged_root)

    print("=" * 70)
    print("NWCSAF L2 SFTP Pipeline (download + arrange)")
    print("=" * 70)
    print(f"Remote host    : {REMOTE_USER}@{REMOTE_HOST}")
    print(f"Date range     : {args.start} → {args.end}")
    print(f"Products       : {products}")
    print(f"Minute filter  : {filter_msg}")
    print(f"Cache dir      : {cache_dir}")
    print(f"Arranged root  : {arranged_root}")
    print(f"Skip download  : {args.skip_download}")
    print(f"Skip arrange   : {args.no_arrange}")

    # ---- Stage 1: download ----
    dl_stats = {
        'downloaded': 0, 'skipped_existing': 0, 'skipped_plax': 0,
        'skipped_date': 0, 'skipped_minute': 0,
        'skipped_unparsable': 0, 'errors': 0,
    }
    if args.skip_download:
        print("\nSkipping SFTP download.")
    else:
        dl_stats = download_nwcsaf_sftp(
            password_file=args.password_file,
            local_dir=cache_dir,
            start_dt=start_dt,
            end_dt=end_dt,
            products=products,
            minute_filter=minute_filter,
        )

    # ---- Stage 2: arrange ----
    arr_stats = None
    if not args.no_arrange:
        arr_stats = arrange_from_cache(
            cache_dir=cache_dir,
            arranged_root=arranged_root,
            products=products,
            start_dt=start_dt,
            end_dt=end_dt,
            use_hardlinks=args.hardlinks,
        )

    # ---- Summary ----
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("[Download]")
    print(f"  Downloaded           : {dl_stats['downloaded']}")
    print(f"  Skipped (existing)   : {dl_stats['skipped_existing']}")
    print(f"  Skipped (PLAX)       : {dl_stats['skipped_plax']}")
    print(f"  Skipped (date)       : {dl_stats['skipped_date']}")
    print(f"  Skipped (minute)     : {dl_stats['skipped_minute']}")
    print(f"  Skipped (unparseable): {dl_stats['skipped_unparsable']}")
    print(f"  Errors               : {dl_stats['errors']}")

    if arr_stats is not None:
        print("[Arrange]")
        print(f"  Arranged             : {arr_stats['arranged']}")
        print(f"  Skipped (existing)   : {arr_stats['skipped_existing']}")
        print(f"  Skipped (PLAX)       : {arr_stats['skipped_plax']}")
        print(f"  Skipped (product)    : {arr_stats['skipped_product']}")
        print(f"  Skipped (date)       : {arr_stats['skipped_date']}")
        print(f"  Skipped (unparseable): {arr_stats['skipped_unparsable']}")
        print(f"  Errors               : {arr_stats['errors']}")
        if arr_stats['dates']:
            sorted_dates = sorted(arr_stats['dates'])
            print(f"  Dates covered        : {len(sorted_dates)} "
                  f"({sorted_dates[0]} → {sorted_dates[-1]})")

    rc = 0
    if dl_stats['errors']:
        rc = 3
    if arr_stats is not None and arr_stats['errors']:
        rc = 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
