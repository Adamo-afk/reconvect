"""
pipeline_opera.py — SFTP/SCP download of OPERA radar data from the EWC VM.

Two products:

    opera_reflectivity   Maximum reflectivity (dBZ), 1 km, 5-min cadence
    opera_rainfall_rate  Instantaneous rain rate (mm/h), 2 km, 15-min cadence

Remote layout on `claudiu@64.225.128.186`:

    /eumetsatdata/opera-reflectivity/{YYYY}/{MM}/{DD}/<file>.h5
    /eumetsatdata/opera-rainfall-rate/{YYYY}/{MM}/{DD}/<file>.h5

OPERA distributes its products as HDF5 (`.h5`) files; this script only
transfers files with that extension. Local destination mirrors the remote
year/month/day hierarchy:

    our_data/opera_data/reflectivity/{YYYY}/{MM}/{DD}/<file>.h5
    our_data/opera_data/rainfall_rate/{YYYY}/{MM}/{DD}/<file>.h5

Filtering applied **before** transfer (client-side, after a server-side
listdir of each date directory):

    1. Date range      — sensing timestamp inside [start, end]
    2. Minute filter   — per-product, from our_data/timestep_config.json
                         (override with --timesteps NN NN ...)

The OPERA archive is already per-date on the server, so no separate
"arrange" step is needed.

Usage:
    python validate_timestep.py --step_minutes 15            # one-time
    python our_data/opera_data/pipeline_opera.py \\
        --start 2025/06/15-0000 --end 2025/06/15-2359 \\
        --password_file password.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
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

REMOTE_HOST = "64.225.128.186"
REMOTE_USER = "claudiu"

# Default base directory holding the per-product subdirectories on the
# remote VM. Some EWC images mount the data at /eumetsatdata, others at
# /home/eumetsatdata — override with --remote_base.
DEFAULT_REMOTE_BASE = "/eumetsatdata"

# Local product key -> (remote subdirectory name, local subdir)
PRODUCTS = {
    "opera_reflectivity":  {
        "remote_subdir": "opera-reflectivity",
        "local_dir":     "reflectivity",
    },
    "opera_rainfall_rate": {
        "remote_subdir": "opera-rainfall-rate",
        "local_dir":     "rainfall_rate",
    },
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent

# Filename timestamp parsers — OPERA uses two common conventions:
#   1. ISO with separators (current EWC dump):
#        2026-05-11T120500Z-reflectivity-composite-opera.h5
#   2. Compact (legacy / EUMETSAT):
#        T_PAAH21_C_LFPW_20250615120000.h5
#        odc.20250615120000.h5
#        composite_201801011500.h5
TIMESTAMP_PATTERN_ISO = re.compile(
    r'(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})(\d{2})Z?'
)
TIMESTAMP_PATTERN_COMPACT = re.compile(r'(\d{12,14})')


# =============================================================================
# Timestep configuration
# =============================================================================

def load_timestep_filter(product_key: str):
    """
    Load the per-product minute filter from timestep_config.json.

    Returns (set of int minutes, step_minutes). Errors out with a clear
    message if the config is missing or doesn't contain the requested
    product — the pipeline must not run without an explicit cadence
    decision for each OPERA product.
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
            f"ERROR: product '{product_key}' has no minute filter in "
            f"{TIMESTEP_CONFIG_PATH}.\n"
            f"Make sure '{product_key}' is listed in product_cadences.json, "
            f"then re-run:\n    python validate_timestep.py "
            f"--step_minutes <N>",
            file=sys.stderr,
        )
        sys.exit(2)
    return set(flt), cfg["step_minutes"]


# =============================================================================
# Shared utilities
# =============================================================================

def parse_date_range(start_str: str, end_str: str,
                     fmt: str = '%Y/%m/%d-%H%M'):
    try:
        return (
            datetime.datetime.strptime(start_str, fmt),
            datetime.datetime.strptime(end_str, fmt),
        )
    except ValueError as e:
        print(f"Error parsing dates. Expected '{fmt}'. {e}", file=sys.stderr)
        return None, None


def parse_opera_timestamp(filename: str) -> datetime.datetime | None:
    """
    Extract the sensing timestamp from an OPERA filename.

    Supports two filename conventions:

      ISO:     2026-05-11T000500Z-reflectivity-composite-opera.h5
      Compact: T_PAAH21_C_LFPW_20250615120000.h5

    ISO is tried first because the dashes/`T` would prevent the compact
    pattern from matching anyway. Seconds (if present) are dropped — we
    only need minute resolution for filtering.
    """
    m = TIMESTAMP_PATTERN_ISO.search(filename)
    if m:
        try:
            return datetime.datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            )
        except ValueError:
            pass
    m = TIMESTAMP_PATTERN_COMPACT.search(filename)
    if m:
        ts = m.group(1)[:12]  # YYYYMMDDHHMM
        try:
            return datetime.datetime.strptime(ts, '%Y%m%d%H%M')
        except ValueError:
            pass
    return None


def read_password(password_file: str) -> str:
    path = Path(password_file)
    if not path.exists():
        print(f"ERROR: Password file not found: {password_file}",
              file=sys.stderr)
        sys.exit(1)
    return path.read_text().strip()


def daterange(start_dt: datetime.datetime,
              end_dt: datetime.datetime):
    """Yield each date (datetime.date) in [start, end], inclusive."""
    d = start_dt.date()
    end = end_dt.date()
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


# =============================================================================
# SFTP download
# =============================================================================

def download_opera(
    cache_dir: Path,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    products: list[str],
    minute_filters: dict[str, set[int] | None],
    password_file: str | None = None,
    ssh_key: str | None = None,
    remote_host: str = REMOTE_HOST,
    remote_user: str = REMOTE_USER,
    remote_base: str = DEFAULT_REMOTE_BASE,
) -> dict:
    """
    Download OPERA files via SFTP, applying date / minute filters before
    transfer.

    Authentication: exactly one of `password_file` or `ssh_key` must be set.

    Args:
        cache_dir:        local root (our_data/opera_data/)
        start_dt, end_dt: inclusive sensing-time range
        products:         list of product keys (subset of PRODUCTS)
        minute_filters:   {product_key: set of allowed minutes or None}
                          None = keep every native timestep for that product
        password_file:    path to text file with the SSH password (or None)
        ssh_key:          path to SSH private key (e.g. ~/.ssh/id_ed25519)
                          (or None)
        remote_base:      remote dir holding the per-product subdirectories
                          (typically `/eumetsatdata` or `/home/eumetsatdata`)
    Returns:
        stats dict
    """
    stats: dict = {
        'downloaded':         0,
        'skipped_existing':   0,
        'skipped_minute':     0,
        'skipped_date':       0,
        'skipped_unparsable': 0,
        'errors':             0,
        'remote_dirs_missing': 0,
    }

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {remote_user}@{remote_host}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if ssh_key:
            ssh.connect(
                remote_host, username=remote_user,
                key_filename=os.path.expanduser(ssh_key),
            )
        else:
            password = read_password(password_file)
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

    # Enumerate (product, date) -> list of (remote_path, local_path) to transfer
    to_download: list[tuple[str, str, str]] = []

    for product in products:
        cfg = PRODUCTS[product]
        product_root = f"{remote_base.rstrip('/')}/{cfg['remote_subdir']}"
        local_base = cache_dir / cfg["local_dir"]
        mfilter = minute_filters.get(product)

        print(f"\nListing remote {product} under {product_root} ...")

        for d in daterange(start_dt, end_dt):
            remote_dir = (
                f"{product_root}/{d.strftime('%Y')}/"
                f"{d.strftime('%m')}/{d.strftime('%d')}"
            )
            local_dir = local_base / d.strftime('%Y') / d.strftime('%m') / d.strftime('%d')

            try:
                remote_files = sftp.listdir(remote_dir)
            except (FileNotFoundError, IOError) as e:
                # paramiko reports both "no such file" and "permission denied"
                # as IOError; surface the underlying message so the user can
                # tell why the listdir failed.
                stats['remote_dirs_missing'] += 1
                print(f"  [{d}] cannot list {remote_dir}: {e}",
                      file=sys.stderr)
                continue

            for filename in remote_files:
                if not filename.endswith('.h5'):
                    # Skip anything that isn't an OPERA HDF5 product (README,
                    # index files, etc.) — don't count these against the
                    # "unparsable" bucket since they're expected.
                    continue
                ts = parse_opera_timestamp(filename)
                if ts is None:
                    stats['skipped_unparsable'] += 1
                    continue
                if not (start_dt <= ts <= end_dt):
                    stats['skipped_date'] += 1
                    continue
                if mfilter is not None and ts.minute not in mfilter:
                    stats['skipped_minute'] += 1
                    continue

                remote_path = f"{remote_dir}/{filename}"
                local_path = local_dir / filename
                to_download.append((product, remote_path, str(local_path)))

    print(f"\nFiltered: {len(to_download)} files queued for download")
    if not to_download:
        sftp.close()
        ssh.close()
        return stats

    total = len(to_download)
    for i, (product, remote_path, local_path) in enumerate(to_download, start=1):
        filename = os.path.basename(local_path)
        if os.path.exists(local_path):
            stats['skipped_existing'] += 1
            print(f"  [{i}/{total}] Already local: {filename}")
            continue
        print(f"  [{i}/{total}] Downloading {filename} ({product})")
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            sftp.get(remote_path, local_path)
            stats['downloaded'] += 1
        except Exception as e:
            stats['errors'] += 1
            print(f"    ERROR {filename}: {e}", file=sys.stderr)

    sftp.close()
    ssh.close()
    return stats


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SFTP/SCP download of OPERA radar data (reflectivity + "
                    "rainfall rate) with cadence filtering applied before "
                    "transfer."
    )
    parser.add_argument('--start', '-s', type=str, required=True,
                        help='Start datetime (yyyy/mm/dd-hhmm)')
    parser.add_argument('--end', '-e', type=str, required=True,
                        help='End datetime (yyyy/mm/dd-hhmm), inclusive.')

    auth = parser.add_mutually_exclusive_group(required=True)
    auth.add_argument('--password_file', '-pw', type=str, default=None,
                      help='Path to text file containing the SSH password.')
    auth.add_argument('--ssh_key', '-i', type=str, default=None,
                      help='Path to SSH private key (e.g. ~/.ssh/id_ed25519). '
                           'Mutually exclusive with --password_file.')

    parser.add_argument('--cache_dir', '-c', type=str,
                        default=str(DEFAULT_CACHE_DIR),
                        help=f'Local OPERA root '
                             f'(default: {DEFAULT_CACHE_DIR})')
    parser.add_argument('--products', type=str, nargs='+',
                        default=list(PRODUCTS.keys()),
                        choices=list(PRODUCTS.keys()),
                        help='OPERA products to download '
                             '(default: both reflectivity and rainfall_rate)')
    parser.add_argument('--timesteps', type=str, nargs='+', default=None,
                        help="Override minute filter for ALL chosen products "
                             "(e.g. 00 15 30 45). 'all' keeps every native "
                             "timestep. Default: read per-product filters "
                             "from timestep_config.json.")
    parser.add_argument('--remote_host', type=str, default=REMOTE_HOST,
                        help=f'Remote SSH host (default: {REMOTE_HOST})')
    parser.add_argument('--remote_user', type=str, default=REMOTE_USER,
                        help=f'Remote SSH user (default: {REMOTE_USER})')
    parser.add_argument('--remote_base', type=str,
                        default=DEFAULT_REMOTE_BASE,
                        help=f'Remote directory containing the per-product '
                             f'subdirs (default: {DEFAULT_REMOTE_BASE}). Try '
                             f'/home/eumetsatdata if the default returns '
                             f'"No such file".')

    args = parser.parse_args()

    start_dt, end_dt = parse_date_range(args.start, args.end)
    if start_dt is None:
        return 1
    if start_dt > end_dt:
        print(f"ERROR: --start ({args.start}) is after --end ({args.end})",
              file=sys.stderr)
        return 1

    products = list(args.products)

    # Resolve minute filters per product
    minute_filters: dict[str, set[int] | None] = {}
    filter_msgs: dict[str, str] = {}
    if args.timesteps is not None and args.timesteps != ['all']:
        override = {int(m) for m in args.timesteps}
        for p in products:
            minute_filters[p] = override
            filter_msgs[p] = f"{sorted(override)} (CLI override)"
    elif args.timesteps == ['all']:
        for p in products:
            minute_filters[p] = None
            filter_msgs[p] = "none (all native timestamps)"
    else:
        for p in products:
            flt, step = load_timestep_filter(p)
            minute_filters[p] = flt
            filter_msgs[p] = (f"{sorted(flt)} "
                              f"(timestep_config.json, step={step} min)")

    cache_dir = Path(args.cache_dir)

    print("=" * 70)
    print("OPERA Radar SFTP Pipeline")
    print("=" * 70)
    print(f"Remote host    : {args.remote_user}@{args.remote_host}")
    print(f"Auth           : {'ssh_key (' + args.ssh_key + ')' if args.ssh_key else 'password_file (' + args.password_file + ')'}")
    print(f"Remote base    : {args.remote_base}")
    print(f"Date range     : {args.start} → {args.end}")
    print(f"Products       : {products}")
    for p in products:
        print(f"  {p:22s} : {filter_msgs[p]}")
    print(f"Cache dir      : {cache_dir}")
    print()

    stats = download_opera(
        cache_dir=cache_dir,
        start_dt=start_dt,
        end_dt=end_dt,
        products=products,
        minute_filters=minute_filters,
        password_file=args.password_file,
        ssh_key=args.ssh_key,
        remote_host=args.remote_host,
        remote_user=args.remote_user,
        remote_base=args.remote_base,
    )

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Downloaded            : {stats['downloaded']}")
    print(f"  Skipped (existing)    : {stats['skipped_existing']}")
    print(f"  Skipped (date)        : {stats['skipped_date']}")
    print(f"  Skipped (minute)      : {stats['skipped_minute']}")
    print(f"  Skipped (unparseable) : {stats['skipped_unparsable']}")
    print(f"  Remote dirs missing   : {stats['remote_dirs_missing']}")
    print(f"  Errors                : {stats['errors']}")

    return 0 if stats['errors'] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
