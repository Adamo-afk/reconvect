"""
pipeline_msg_mtg.py — MTG FCI L1C data pipeline (Romania)

Downloads FCI chunk files from a remote machine via SFTP (paramiko),
filters to Romania-covering chunks (35–36), reads them directly with
netCDF4, and saves per-channel NetCDF files with geostationary
projection metadata for downstream reprojection.

Usage:
    python pipeline_msg_mtg.py -s 2025/05/01-0000 -e 2025/05/01-2350 \\
        --password_file password.txt --products_file satellite_products.json

Requirements:
    pip install paramiko hdf5plugin netcdf4 numpy xarray
"""

import os
import re
import sys
import gc
import json
import datetime
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import paramiko
except ImportError:
    print(
        "ERROR: paramiko is required for SFTP transfers.\n"
        "Install with: pip install paramiko"
    )
    sys.exit(1)

try:
    import hdf5plugin  # noqa: F401 — required for CharLS-compressed FCI L1c
    # Export the plugin directory as HDF5_PLUGIN_PATH before netCDF4 (and
    # its bundled libhdf5) get imported anywhere. On Windows/anaconda the
    # libhdf5 that netCDF4 loads doesn't always scan hdf5plugin's own
    # registration, so we fall back to the env var it *does* honour.
    # Without this the FCI CharLS/JPEG-LS chunks fail with
    # "NetCDF: Filter error: undefined filter encountered" on every read.
    # setdefault so a user-supplied HDF5_PLUGIN_PATH is not clobbered.
    os.environ.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGIN_PATH)
except ImportError:
    print(
        "WARNING: hdf5plugin not installed. FCI L1c reading may fail.\n"
        "Install with: pip install hdf5plugin"
    )


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A store may be zstd-compressed in place. Without this the already-extracted
# check below would miss every .npy.zst and re-extract the whole store.
sys.path.insert(0, str(PROJECT_ROOT))
from compress_datasets import array_exists  # noqa: E402
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"

# FCI chunks covering the Romania study area.
# From the Météo-France FCI scan diagram (and verified against the
# geostationary projection), Romania falls in chunks 35–36 of the
# 40 body chunks per repeat cycle.
ROMANIA_CHUNKS = {35, 36}

# Remote machine hosting the FCI data (NMA internal network).
# NOTE: REMOTE_USER below is the literal login account on that host — it
# stays lowercase "anm" regardless of how the institution is named in prose.
REMOTE_HOST = "192.168.11.223"
REMOTE_USER = "anm"
REMOTE_FCI_DIR = "/ShortTermStorage/GEOSTATIONARY/MTG/FCI/"

# Channel resolution groups
CHANNELS_1KM = {
    'vis_04', 'vis_05', 'vis_06', 'vis_08', 'vis_09',
    'nir_13', 'nir_16', 'nir_22',
}
CHANNELS_2KM = {
    'ir_38', 'wv_63', 'wv_73', 'ir_87', 'ir_97',
    'ir_105', 'ir_123', 'ir_133',
}
VALID_MTG_CHANNELS = CHANNELS_1KM | CHANNELS_2KM


# =============================================================================
# Timestep configuration
# =============================================================================

def load_timestep_filter(product_key="mtg"):
    """
    Load the minute filter for a product from timestep_config.json.

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
        start = datetime.datetime.strptime(start_str, fmt)
        end = datetime.datetime.strptime(end_str, fmt)
        return start, end
    except ValueError as e:
        print(f"Error parsing dates. Please use format '{fmt}'. Error: {e}")
        return None, None


def load_variables_from_file(filepath, satellite='mtg'):
    """
    Read channel names from a JSON file for the specified satellite.

    Args:
        filepath (str): Path to JSON file with "msg" and "mtg" keys.
        satellite (str): 'msg' or 'mtg' — determines which key to read.

    Returns:
        list: List of FCI variable names.
    """
    if satellite.lower() not in ('msg', 'mtg'):
        raise ValueError(f"Unknown satellite: {satellite}. Use 'msg' or 'mtg'.")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Product list file not found: {filepath}\n"
            f"Create a JSON file with 'msg' and 'mtg' keys, e.g.:\n"
            f'{{\n  "msg": ["VIS006", "IR_108", "WV_062"],\n'
            f'  "mtg": ["vis_06", "ir_105", "wv_63"]\n}}'
        )

    with open(filepath, 'r') as f:
        config = json.load(f)

    key = satellite.lower()
    if key not in config:
        raise KeyError(
            f"Key '{key}' not found in {filepath}. "
            f"Available keys: {list(config.keys())}"
        )

    channels = config[key]
    if not isinstance(channels, list):
        raise TypeError(
            f"'{key}' must be a list, got {type(channels).__name__}"
        )

    valid_set = VALID_MTG_CHANNELS
    validated = []
    invalid = []
    for ch in channels:
        if ch in valid_set:
            if ch not in validated:
                validated.append(ch)
        else:
            invalid.append(ch)

    if invalid:
        print(f"WARNING: Unrecognized MTG channels: {invalid}")
        print(f"  Valid channels: {sorted(valid_set)}")

    print(f"Loaded {len(validated)} MTG channels from {filepath}:")
    for v in validated:
        print(f"  {v}")

    return validated


# =============================================================================
# FCI filename parsing
# =============================================================================

def parse_fci_filename(filename):
    """
    Parse an FCI L1C filename and extract metadata.

    Example filename (body chunk):
        W_XX-EUMETSAT-..._OPE_20250402024826_20250402024911_N_JLS_O_0017_0036.nc

    Returns:
        dict with keys:
            chunk_number        (int or None)
            repeat_cycle_in_day (int or None)
            sensing_start       (datetime or None)
            sensing_end         (datetime or None)
            is_body             (bool)
            is_trailer          (bool)
    """
    info = {
        'chunk_number': None,
        'repeat_cycle_in_day': None,
        'sensing_start': None,
        'sensing_end': None,
        'is_body': 'CHK-BODY' in filename or 'CHK_BODY' in filename,
        'is_trailer': 'TRAIL' in filename,
    }

    # Last two 4-digit groups before .nc:
    #   _<repeat_cycle_in_day>_<chunk_number>.nc
    match = re.search(r'_(\d{4})_(\d{4})\.nc$', filename)
    if match:
        info['repeat_cycle_in_day'] = int(match.group(1))
        info['chunk_number'] = int(match.group(2))

    # Sensing start and end after the _OPE_ (or _DEV_) facility marker
    time_pattern = r'_(?:OPE|DEV)_(\d{14})_(\d{14})_'
    match = re.search(time_pattern, filename)
    if match:
        try:
            info['sensing_start'] = datetime.datetime.strptime(
                match.group(1), '%Y%m%d%H%M%S'
            )
            info['sensing_end'] = datetime.datetime.strptime(
                match.group(2), '%Y%m%d%H%M%S'
            )
        except ValueError:
            pass

    return info


def nominal_time_from_repeat_cycle(date_ref, repeat_cycle_in_day,
                                    period_min=10):
    """
    Compute the nominal start time of a repeat cycle.

    Args:
        date_ref (datetime): Any datetime on the same day (used for the date).
        repeat_cycle_in_day (int): 1-indexed repeat cycle number.
        period_min (int): Repeat cycle period in minutes (10 for FDSS).

    Returns:
        datetime: Nominal start time (e.g., cycle 17 → 02:40 UTC).
    """
    base = date_ref.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + datetime.timedelta(
        minutes=(repeat_cycle_in_day - 1) * period_min
    )


def nominal_minute_of_hour(repeat_cycle_in_day, period_min=10):
    """
    Get the minute-within-the-hour for a given repeat cycle number.

    Used for timestep filtering before download.
    E.g., cycle 1 → minute 0, cycle 4 → minute 30, cycle 7 → minute 0.
    """
    total_minutes = (repeat_cycle_in_day - 1) * period_min
    return total_minutes % 60


# =============================================================================
# SFTP transfer
# =============================================================================

def read_password(password_file):
    """Read the SSH password from a plain-text file (first line, stripped)."""
    path = Path(password_file)
    if not path.exists():
        print(f"ERROR: Password file not found: {password_file}",
              file=sys.stderr)
        sys.exit(1)
    return path.read_text().strip()


def download_fci_files_sftp(
    password_file,
    local_dir,
    start_dt,
    end_dt,
    chunk_filter=ROMANIA_CHUNKS,
    minute_filter=None,
    remote_host=REMOTE_HOST,
    remote_user=REMOTE_USER,
    remote_dir=REMOTE_FCI_DIR,
):
    """
    Download FCI L1C chunk files from a remote machine via SFTP.

    Applies three filters *before* downloading:
        1. Date range (sensing_start within [start_dt, end_dt])
        2. Chunk number (body chunks in chunk_filter; trailers always kept)
        3. Minute-of-hour filter (if minute_filter is not None)

    Args:
        password_file (str): Path to text file containing SSH password.
        local_dir (str): Local directory to store downloaded chunk files.
        start_dt (datetime): Start of the requested period.
        end_dt (datetime): End of the requested period.
        chunk_filter (set): Allowed body chunk numbers (1-indexed).
        minute_filter (set or None): Allowed minutes within the hour.
        remote_host (str): SSH hostname or IP.
        remote_user (str): SSH username.
        remote_dir (str): Remote directory containing .nc chunk files.

    Returns:
        dict: {group_key: [local_paths]} keyed by 'YYYYMMDD_RRRR'
              (date + repeat_cycle_in_day), ready for processing.
    """
    password = read_password(password_file)
    os.makedirs(local_dir, exist_ok=True)

    # --- Connect ---
    print(f"Connecting to {remote_user}@{remote_host}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(remote_host, username=remote_user, password=password)
    except Exception as e:
        print(f"ERROR: SSH connection failed: {e}", file=sys.stderr)
        return {}

    sftp = ssh.open_sftp()

    # --- List remote files ---
    # For large directories, using SSH 'ls' with a glob is faster than
    # sftp.listdir(). We build a shell glob that pre-filters by date
    # prefix and chunk numbers.
    print(f"Listing remote directory: {remote_dir}")

    # Build date prefixes for the requested range (YYYYMMDD strings)
    date_prefixes = set()
    d = start_dt.date()
    while d <= end_dt.date():
        date_prefixes.add(d.strftime('%Y%m%d'))
        d += datetime.timedelta(days=1)

    # Build ls command with globs for each chunk + trailer
    chunk_globs = []
    for chunk_num in sorted(chunk_filter):
        for prefix in sorted(date_prefixes):
            chunk_globs.append(
                f"{remote_dir}*{prefix}*_{chunk_num:04d}.nc"
            )
    # Also grab trailer files for matching dates
    for prefix in sorted(date_prefixes):
        chunk_globs.append(f"{remote_dir}*{prefix}*TRAIL*.nc")

    # Execute in batches to avoid overly long command lines
    all_remote_files = set()
    batch_size = 50
    for i in range(0, len(chunk_globs), batch_size):
        batch = chunk_globs[i:i + batch_size]
        cmd = "ls " + " ".join(batch) + " 2>/dev/null"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        for line in stdout:
            line = line.strip()
            if line:
                all_remote_files.add(line)

    print(f"  Remote glob matched {len(all_remote_files)} files")

    # --- Parse and filter ---
    to_download = []  # list of (remote_path, filename, info)

    for remote_path in sorted(all_remote_files):
        filename = os.path.basename(remote_path)
        info = parse_fci_filename(filename)

        # Filter by date range (precise, using sensing_start)
        if info['sensing_start']:
            if not (start_dt <= info['sensing_start'] <= end_dt):
                continue
        else:
            # Cannot determine time — skip to be safe
            continue

        # Filter by minute-of-hour (applied to both body and trailer)
        if minute_filter is not None and info['repeat_cycle_in_day']:
            minute = nominal_minute_of_hour(info['repeat_cycle_in_day'])
            if minute not in minute_filter:
                continue

        # Body chunks: must be in chunk_filter
        if info['is_body'] and info['chunk_number'] is not None:
            if info['chunk_number'] not in chunk_filter:
                continue

        to_download.append((remote_path, filename, info))

    print(f"  After filtering: {len(to_download)} files to download")

    if not to_download:
        sftp.close()
        ssh.close()
        return {}

    # --- Download and group by repeat cycle ---
    groups = defaultdict(list)

    for i, (remote_path, filename, info) in enumerate(to_download):
        local_path = os.path.join(local_dir, filename)

        if not os.path.exists(local_path):
            print(
                f"  [{i + 1}/{len(to_download)}] "
                f"Downloading {filename}"
            )
            try:
                sftp.get(remote_path, local_path)
            except Exception as e:
                print(f"    ERROR downloading {filename}: {e}")
                continue
        else:
            print(
                f"  [{i + 1}/{len(to_download)}] "
                f"Already local: {filename}"
            )

        # Group key: YYYYMMDD_RRRR (date + repeat cycle number)
        if info['sensing_start'] and info['repeat_cycle_in_day'] is not None:
            date_str = info['sensing_start'].strftime('%Y%m%d')
            rc = info['repeat_cycle_in_day']
            key = f"{date_str}_{rc:04d}"
        else:
            key = "unknown"

        groups[key].append(local_path)

    sftp.close()
    ssh.close()

    print(f"  Organised into {len(groups)} repeat-cycle groups")
    return dict(groups)


# =============================================================================
# FCI grid constants
# =============================================================================

CONSTANTS_FILENAME = 'mtg_constants.json'


def save_mtg_constants(body_files, variables, output_path):
    """
    Extract FCI grid constants from chunk files and save as JSON.

    Called once at the start of the pipeline. Stores geostationary
    projection parameters and 1-D coordinate arrays for both 1km and
    2km resolutions. All subsequent timestep files are saved as raw
    .npy arrays — the constants file provides the spatial reference.

    Args:
        body_files (list): Paths to body chunk files (any repeat cycle).
        variables (list): Requested channel names (used to determine
            which resolutions are needed).
        output_path (str): Path to save the JSON file.
    """
    from netCDF4 import Dataset as NC4Dataset

    constants = {'projection': {}, '1km': None, '2km': None}

    # Read projection parameters (same for both resolutions)
    ds0 = NC4Dataset(body_files[0], 'r')
    gm_var = ds0['data']['mtg_geos_projection']
    constants['projection'] = {
        'grid_mapping_name': 'geostationary',
        'perspective_point_height': float(
            gm_var.getncattr('perspective_point_height')),
        'semi_major_axis': float(
            gm_var.getncattr('semi_major_axis')),
        'semi_minor_axis': float(
            gm_var.getncattr('semi_minor_axis')),
        'longitude_of_projection_origin': float(
            gm_var.getncattr('longitude_of_projection_origin')),
        'sweep_angle_axis': str(
            gm_var.getncattr('sweep_angle_axis')),
    }

    # Extract coordinate arrays per resolution
    needs_1km = any(v in CHANNELS_1KM for v in variables)
    needs_2km = any(v in CHANNELS_2KM for v in variables)

    for res_key, ref_channels, needed in [
        ('1km', CHANNELS_1KM, needs_1km),
        ('2km', CHANNELS_2KM, needs_2km),
    ]:
        if not needed:
            continue

        # Find a channel at this resolution that exists in the chunk
        ref_channel = None
        available = list(ds0['data'].groups.keys())
        for ch in ref_channels:
            if ch in available:
                ref_channel = ch
                break

        if ref_channel is None:
            continue

        # Read x from first chunk (same across all chunks)
        x_geos = np.array(
            ds0['data'][ref_channel]['measured'].variables['x'][:]
        ).tolist()

        # Read y from all chunks and concatenate
        y_arrays = []
        for chunk_file in sorted(body_files):
            ds = NC4Dataset(chunk_file, 'r')
            if ref_channel in ds['data'].groups:
                y_1d = np.array(
                    ds['data'][ref_channel]['measured'].variables['y'][:]
                )
                y_arrays.append(y_1d)
            ds.close()

        y_geos = np.concatenate(y_arrays).tolist() if y_arrays else []

        # Determine data shape
        ds_tmp = NC4Dataset(body_files[0], 'r')
        sample = ds_tmp['data'][ref_channel]['measured'].variables[
            'effective_radiance']
        chunk_rows = sample.shape[0]
        cols = sample.shape[1]
        ds_tmp.close()
        total_rows = chunk_rows * len(body_files)

        constants[res_key] = {
            'x_geos': x_geos,
            'y_geos': y_geos,
            'shape': [total_rows, cols],
        }

    ds0.close()

    with open(output_path, 'w') as f:
        json.dump(constants, f, indent=2)

    print(f"  Saved grid constants to {output_path}")
    for res in ['1km', '2km']:
        if constants[res]:
            print(f"    {res}: x={len(constants[res]['x_geos'])}, "
                  f"y={len(constants[res]['y_geos'])}, "
                  f"shape={constants[res]['shape']}")


# =============================================================================
# FCI chunk processing (direct netCDF4 reading)
# =============================================================================

def process_repeat_cycle(files, base_dir, variables, group_key=None,
                          overwrite=False):
    """
    Process one repeat cycle of FCI chunk files.

    Reads chunk files directly with netCDF4 (hdf5plugin provides the
    CharLS decompression filter). Only the Romania-covering chunks are
    present, so no full-disk stitching is needed — chunks are simply
    concatenated vertically.

    netCDF4 auto-applies scale_factor and add_offset, so the output
    contains effective radiance as float32.

    Each variable is saved as a raw .npy file. The spatial reference
    (projection parameters, coordinate arrays) is stored once in
    mtg_constants.json by save_mtg_constants().

    Args:
        files (list): Paths to chunk .nc files (body + optional trailer).
        base_dir (str): Root output directory.
        variables (list): FCI channel names to extract.
        group_key (str): Repeat cycle key 'YYYYMMDD_RRRR' for computing
            the nominal time.
        overwrite (bool): Re-extract variables whose .npy already exists.
        provenance_source (str or None): origin recorded for every
            cycle extracted here. None disables recording.
        delete_only (bool): Skip download AND extraction, deleting the
            raw chunks in the window immediately. For when the
            summary has already confirmed coverage, which makes the
            re-extraction pass redundant.
        delete_raw (bool): Remove the raw .nc chunks once every cycle
            has been extracted without error. Irreversible.
            Default False, so a re-run over _raw_chunks/ only fills in
            what is genuinely absent.

    Returns:
        dict with keys: 'key', 'ok' (list), 'skipped' (list),
        'failed' (list of (var, error))
    """
    from netCDF4 import Dataset as NC4Dataset

    result = {'key': group_key, 'ok': [], 'skipped': [], 'failed': []}

    body_files = sorted([f for f in files if 'TRAIL' not in f])
    if not body_files:
        result['failed'].append(('_all', 'no body chunk files'))
        return result

    # Compute nominal time from group key (YYYYMMDD_RRRR)
    if group_key:
        parts = group_key.split('_')
        date_ref = datetime.datetime.strptime(parts[0], '%Y%m%d')
        rc_num = int(parts[1])
        nominal_start = nominal_time_from_repeat_cycle(date_ref, rc_num)
    else:
        info = parse_fci_filename(os.path.basename(body_files[0]))
        if info['sensing_start'] and info['repeat_cycle_in_day']:
            nominal_start = nominal_time_from_repeat_cycle(
                info['sensing_start'], info['repeat_cycle_in_day']
            )
        else:
            nominal_start = info['sensing_start'] or datetime.datetime.now()

    dir_name = f"nc4_{nominal_start.strftime('%Y-%m-%d')}-Romania"
    time_str = nominal_start.strftime('%H%M')

    # --- Skip variables already extracted ---
    # Deliberately checked BEFORE the first netCDF4 open: the output path
    # is fully determined by the nominal time, so an already-processed
    # cycle costs one stat() per variable instead of decompressing both
    # CharLS chunks again. This is what makes a --skip_download pass over
    # a full _raw_chunks/ cheap enough to run after every backfill.
    if not overwrite:
        pending = []
        for variable in variables:
            existing = os.path.join(
                base_dir, variable, f"{dir_name}_{variable}",
                f"{dir_name}_{time_str}_{variable}.npy",
            )
            if array_exists(existing):
                result['skipped'].append(variable)
            else:
                pending.append(variable)
        variables = pending
        if not variables:
            return result

    # --- Process each variable ---
    for variable in variables:
        chunk_arrays = []

        for chunk_file in body_files:
            try:
                ds = NC4Dataset(chunk_file, 'r')
                if variable not in ds['data'].groups:
                    ds.close()
                    continue
                measured = ds['data'][variable]['measured']
                data = measured.variables['effective_radiance'][:]
                data = np.array(data, dtype=np.float32)
                chunk_arrays.append(data)
                ds.close()
            except Exception as e:
                result['failed'].append((variable, str(e)))
                try:
                    ds.close()
                except Exception:
                    pass

        if not chunk_arrays:
            result['failed'].append((variable, 'no data in chunks'))
            continue

        combined = np.concatenate(chunk_arrays, axis=0)

        var_dir = os.path.join(base_dir, variable)
        date_subdir = f"{dir_name}_{variable}"
        full_output_dir = os.path.join(var_dir, date_subdir)
        os.makedirs(full_output_dir, exist_ok=True)

        npy_filename = f"{dir_name}_{time_str}_{variable}.npy"
        output_path = os.path.join(full_output_dir, npy_filename)

        np.save(output_path, combined)
        result['ok'].append(variable)

        del combined

    gc.collect()
    return result


# =============================================================================
# Main pipeline
# =============================================================================

def volume_id(path):
    """Identity of the volume holding `path`, or None.

    `st_dev` rather than the drive letter: the MTG store is usually
    reached through a junction, so `F:\\...\\MTG` and `E:\\` are the same
    disk and must not be treated as two places to spill between.
    """
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return os.stat(str(probe)).st_dev
    except OSError:
        return None


def store_candidates(primary, spills):
    """Stores to rotate between, one per distinct volume, primary first."""
    out, seen = [], set()
    for path in [primary] + list(spills or []):
        vid = volume_id(path)
        if vid is not None and vid in seen:
            continue
        if vid is not None:
            seen.add(vid)
        out.append(path)
    return out


def choose_store(current, candidates, min_free_gb):
    """Where the next window should be written.

    Stay put while the current store has room; otherwise move to whichever
    candidate has the most. Deliberately re-evaluated every window and in
    both directions: `--delete_raw` gives space back to the disk it is
    reading from, so a store that was full earlier in the run is often the
    emptiest later, and a one-way spill would never return to it.
    """
    if free_gb(current) >= min_free_gb:
        return current

    ranked = sorted(candidates, key=free_gb, reverse=True)
    best = ranked[0]
    if free_gb(best) < min_free_gb:
        lines = "".join(f"{chr(10)}    {free_gb(c):8.0f} GB  {c}"
                        for c in ranked)
        raise SystemExit(
            f"No store has {min_free_gb:.0f} GB free; stopping rather than "
            f"filling a disk.{lines}{chr(10)}"
            f"Free space, lower --min_free_gb, or add another "
            f"--spill_dir.")

    print(chr(10) + f"  {current} is down to {free_gb(current):.0f} GB "
          f"(threshold {min_free_gb:.0f} GB)")
    print(f"  -> continuing in {best} ({free_gb(best):.0f} GB free)")
    os.makedirs(best, exist_ok=True)
    _carry_constants(current, best)
    return best


def _carry_constants(src_dir, dst_dir):
    """Copy mtg_constants.json into a newly-opened store.

    Every store needs its own copy: reproject.py builds its KD-tree from
    the constants beside the arrays, and a store without them is
    unreprojectable even though the .npy are perfectly good.
    """
    import shutil
    src = os.path.join(src_dir, CONSTANTS_FILENAME)
    dst = os.path.join(dst_dir, CONSTANTS_FILENAME)
    if os.path.isfile(src) and not os.path.isfile(dst):
        shutil.copy2(src, dst)
        print(f"  Copied {CONSTANTS_FILENAME} -> {dst_dir}")


def free_gb(path) -> float:
    """Free space on the volume holding `path`, in GB."""
    import shutil
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(str(probe)).free / (1024 ** 3)


def dates_in_window(window_start, window_end) -> list[str]:
    """Every YYYY-MM-DD the window covers, for store registration."""
    out, cur = [], window_start.date()
    last = window_end.date()
    while cur <= last:
        out.append(cur.isoformat())
        cur += datetime.timedelta(days=1)
    return out


def month_windows(start, end, n_months):
    """Split [start, end] into consecutive windows of `n_months` months.

    Windows are aligned to calendar months and clipped to the requested
    range, so the first and last may be shorter. Returns a list of
    (window_start, window_end) datetimes, both inclusive.

    Calendar-aligned rather than fixed 30-day steps so a window maps onto
    something a person can reason about - "2025-04 to 2025-06" - and onto
    the monthly coverage chart the summary produces.
    """
    if n_months < 1:
        raise ValueError(f"--batch_months must be >= 1, got {n_months}")
    if end < start:
        raise ValueError("--end precedes --start")

    windows = []
    cur = start
    while cur <= end:
        y, m = cur.year, cur.month
        m0 = (m - 1) + n_months
        ny, nm = y + m0 // 12, m0 % 12 + 1
        nxt = datetime.datetime(ny, nm, 1)
        stop = min(nxt - datetime.timedelta(minutes=1), end)
        windows.append((cur, stop))
        cur = nxt
    return windows


def run_batched(args, windows, run_window):
    """Drive `run_window` over each window, reporting and continuing on error.

    One window failing must not abandon the rest: a transient SFTP drop in
    month three should not cost months four onward. Failures are collected
    and reported at the end, and the exit code reflects them.
    """
    failures = []
    total = len(windows)
    for i, (ws, we) in enumerate(windows, start=1):
        label = f"{ws:%Y-%m-%d} .. {we:%Y-%m-%d}"
        print("\n" + "=" * 70 + "\n"
              f"[window {i}/{total}]  {label}\n"
              + "=" * 70)
        try:
            run_window(ws, we)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code:
                failures.append((label, f"exit {code}"))
                print(f"  WINDOW FAILED: {label}: {failures[-1][1]}", file=sys.stderr)
        except Exception as exc:                          # noqa: BLE001
            failures.append((label, str(exc)))
            print(f"  WINDOW FAILED: {label}: {failures[-1][1]}", file=sys.stderr)
            if args.stop_on_error:
                break

    print("\n" + "=" * 70 + "\n"
          + "Batched run complete\n"
          + "=" * 70)
    print(f"  windows : {total}")
    print(f"  failed  : {len(failures)}")
    for label, why in failures:
        print(f"    {label}  {why}")
    if failures:
        print("\n  Re-run the same command to retry only the "
          "failed windows: completed cycles are skipped.")
    return 1 if failures else 0


def fetch_and_process_mtg(
    start_str,
    end_str,
    base_dir,
    variables,
    password_file,
    chunk_filter=ROMANIA_CHUNKS,
    minute_filter=None,
    skip_download=False,
    workers=10,
    overwrite=False,
    delete_raw=False,
    delete_only=False,
    provenance_source="nma",
    raw_dir=None,
):
    """
    End-to-end MTG FCI pipeline: SFTP download → netCDF4 processing.

    Args:
        start_str / end_str (str): Date range in 'yyyy/mm/dd-hhmm' format.
        base_dir (str): Root output directory.
        variables (list): FCI channel names.
        password_file (str): Path to text file with SSH password.
        chunk_filter (set): Allowed chunk numbers (default: ROMANIA_CHUNKS).
        minute_filter (set or None): Allowed minutes within the hour.
        skip_download (bool): If True, skip SFTP and use files already in
            <base_dir>/_raw_chunks/. This is what --source local sets.
        workers (int): Number of parallel workers for repeat cycle processing.
        overwrite (bool): Re-extract variables whose .npy already exists.
        provenance_source (str or None): Stamped on each cycle as it is
            extracted. None records nothing, which is the honest default
            for a local pass: raw already on disk carries no evidence of
            where it came from.
    """
    start, end = parse_date_range(start_str, end_str)
    if start is None:
        return

    # Raw and output normally share a root, but they need not: the .npy
    # a cycle produces is ~3.8x the raw it replaces, so a drive can hold
    # the chunks and still have no room for the arrays. Reading raw from
    # one disk and writing arrays to another is what makes that archive
    # recoverable.
    local_download_dir = raw_dir or os.path.join(base_dir, '_raw_chunks')

    # ---- Step 1: Download (or skip) ----
    if skip_download:
        print(f"Skipping download — using existing files in {local_download_dir}")
        groups = group_local_files(
            local_download_dir, start, end,
            chunk_filter=chunk_filter,
            minute_filter=minute_filter,
        )
    else:
        groups = download_fci_files_sftp(
            password_file=password_file,
            local_dir=local_download_dir,
            start_dt=start,
            end_dt=end,
            chunk_filter=chunk_filter,
            minute_filter=minute_filter,
        )

    if not groups:
        print("No files to process.")
        return

    # ---- Delete-only: nothing to extract, nothing to record ----------
    # The summary already measured coverage from the .npy output, so a
    # re-extraction pass here would only re-confirm what it established.
    # Grouping still runs: it is what restricts deletion to the requested
    # window instead of emptying _raw_chunks wholesale.
    if delete_only:
        print(f"\nDelete-only: skipping extraction for {len(groups)} "
              f"cycle(s) in this window.")
        print("  Coverage was NOT re-verified here - rely on the summary "
              "you ran beforehand.")
        delete_raw_chunks(groups, base_dir, None)
        return

    # ---- Step 2: Save grid constants (once) ----
    constants_path = os.path.join(base_dir, CONSTANTS_FILENAME)
    if not os.path.exists(constants_path):
        # Use body files from the first group to extract constants
        first_key = sorted(groups.keys())[0]
        first_body = sorted(
            f for f in groups[first_key] if 'TRAIL' not in f
        )
        if first_body:
            print(f"\nExtracting grid constants from first repeat cycle...")
            try:
                save_mtg_constants(first_body, variables, constants_path)
            except Exception as e:
                print(f"  WARNING: Could not save constants: {e}")
    else:
        print(f"\nGrid constants already exist: {constants_path}")

    # ---- Step 3: Extract, recording and reclaiming as it goes --------
    # Provenance is written per wave rather than at the end: the NMA
    # server and the Data Store share filenames, so an interrupted run
    # that lost the ledger would leave its own output unattributable.
    #
    # Deletion is folded into the same loop, download run or not. Doing
    # it after the whole range would need room for every raw chunk at once,
    # which is the situation --delete_raw exists to avoid - and on a local
    # pass over an archive that is already on disk, deferring it frees
    # nothing until the run ends, which is when the space is least useful.
    #
    # Cycles that failed keep their raw either way; --delete_only is the
    # unconditional sweep for whatever is left after the summary has judged
    # it.
    process_groups(
        groups, base_dir, variables, workers=workers, overwrite=overwrite,
        delete_after=delete_raw, provenance_source=provenance_source,
    )
    if provenance_source:
        print(f"Provenance      : recorded as '{provenance_source}' "
              f"(per wave)")


def record_existing_provenance(base_dir, variables, source):
    """Stamp cycles already on disk with a source you know out-of-band.

    The ledger can only be written when a cycle is downloaded. Anything
    that arrived before the ledger existed reports as `unrecorded`, and no
    amount of re-summarising changes that - the summary reads, it does not
    write, and the two sources share filenames so origin is not
    recoverable from the data.

    This is the escape hatch for the one case where the information does
    exist: you know the current archive came from a particular source.
    Recording that is not guesswork, it is entering knowledge the system
    never had.

    Only cycles with NO existing entry are stamped. A cycle already
    recorded as `datastore` is left alone - overwriting it here would
    destroy a fact with an assumption.
    """
    from datastore_fill import load_provenance, provenance_of, record_provenance

    base = Path(base_dir)
    blob = load_provenance(base)

    # Enumerate what the .npy tree actually holds, at cycle granularity.
    found = set()
    for channel in variables:
        ch_root = base / channel
        if not ch_root.is_dir():
            continue
        for day_dir in ch_root.iterdir():
            if not day_dir.is_dir():
                continue
            for filename in os.listdir(day_dir):
                m = re.match(
                    r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_(\d{4})_', filename)
                if m:
                    hhmm = m.group(2)
                    found.add((m.group(1), f"{hhmm[:2]}:{hhmm[2:]}"))

    if not found:
        print("\nNo .npy cycles found - nothing to record.")
        return 0

    already = {(d, t) for d, t in found if provenance_of(blob, d, t)}
    fresh = sorted(found - already)

    print("\nRecording provenance for existing data")
    print(f"  cycles on disk    : {len(found):,}")
    print(f"  already recorded  : {len(already):,} (left untouched)")
    print(f"  to stamp as {source:<9}: {len(fresh):,}")

    if not fresh:
        print("  nothing to do")
        return 0

    n = record_provenance(base, fresh, source)
    print(f"  recorded {n:,} cycle(s)")
    return n


def delete_raw_chunks(groups, base_dir, counts=None):
    """Delete the raw .nc chunks for every cycle in this run.

    Unconditional by design. Deciding what is missing - including cycles
    that failed to extract - is the summary's job, and the summary is run
    BEFORE deletion, recording the gaps in mtg_missing_timesteps.json.
    Re-deriving that judgement here would duplicate it, and a partial
    deletion is worse than none: it leaves the archive in a state that
    neither the raw nor the .npy scan describes.

    Irreversible. Recovering a chunk means downloading it again, and it
    removes the --skip_download reprocessing path - which is why it is
    opt-in via --delete_raw and never happens as a side effect.

    `counts` is accepted and ignored so callers need not know whether a
    guard exists.
    """
    files = [f for paths in groups.values() for f in paths]
    if not files:
        print("\nNo raw chunks to delete.")
        return 0

    total = len(files)
    print(f"\nDeleting {total:,} raw chunk file(s) ...")

    freed = removed = failed = 0
    for path in files:
        try:
            freed += os.path.getsize(path)
            os.remove(path)
            removed += 1
        except OSError as exc:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: could not remove {path}: {exc}",
                      file=sys.stderr)

    print(f"  Removed {removed:,} file(s), freed "
          f"{freed / (1024 ** 3):.1f} GB")
    if failed:
        print(f"  {failed} file(s) could not be removed")
    print("  Coverage is now measured from the .npy output "
          "(summarize_mtg.py, --scan npy is the default).")
    return removed


def cycle_from_key(group_key):
    """'YYYYMMDD_RRRR' -> ('YYYY-MM-DD', 'HH:MM') nominal, or None."""
    try:
        date_part, rc_part = group_key.split("_")
        ref = datetime.datetime.strptime(date_part, "%Y%m%d")
        nominal = nominal_time_from_repeat_cycle(ref, int(rc_part))
        return nominal.strftime("%Y-%m-%d"), nominal.strftime("%H:%M")
    except (ValueError, IndexError):
        return None


def process_groups(groups, base_dir, variables, workers=10,
                   overwrite=False, delete_after=False,
                   provenance_source=None):
    """Extract every variable of every repeat cycle in `groups` to .npy.

    Work is dispatched in waves of `workers` cycles. After each wave the
    cycles that extracted cleanly have their provenance recorded and, with
    `delete_after`, their raw chunks deleted before the next wave starts.

    That bound is the point: peak raw-on-disk is one wave, not the whole
    download. Extracting a full archive and only then deleting would
    require room for every chunk at once, which is exactly the situation
    the deletion exists to avoid.

    Cycles that FAILED keep their raw chunks. Deleting those would destroy
    the only copy before any summary has seen them, and would remove the
    `--skip_download --reprocess` retry path for the one case that needs
    it. This is not the same judgement as the standalone
    `delete_raw_chunks()`: there, the summary has already run and recorded
    the gaps, so there is nothing left to preserve.

    Args:
        delete_after: delete raw chunks of successful cycles, wave by wave.
        provenance_source: 'nma' or 'datastore'; recorded per wave so an
            interrupted run keeps what it already did.

    Returns:
        dict with 'processed', 'skipped', 'errors', 'deleted', 'freed_bytes'.
    """
    sorted_items = sorted(groups.items())
    total = len(sorted_items)
    if not total:
        print("No repeat cycles to process.")
        return {"processed": 0, "skipped": 0, "errors": 0,
                "deleted": 0, "freed_bytes": 0}

    n_workers = min(workers, total)
    print(f"\nProcessing {total} repeat cycles "
              f"({len(variables)} variables, waves of {n_workers})...")
    if delete_after:
        print(f"  Raw chunks are deleted after each wave; peak raw on "
              f"disk is {n_workers} cycle(s).")

    success = skipped_count = error_count = 0
    deleted = freed = 0
    errors = []
    failed_keys = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for wave_start in range(0, total, n_workers):
            wave = sorted_items[wave_start:wave_start + n_workers]
            futures = {
                executor.submit(process_repeat_cycle, files, base_dir,
                                variables, group_key=key,
                                overwrite=overwrite): key
                for key, files in wave
            }

            clean = []
            for future in as_completed(futures):
                key = futures[future]
                done += 1
                try:
                    result = future.result()
                    if result["failed"]:
                        error_count += 1
                        failed_keys.append(key)
                        for var, err in result["failed"]:
                            errors.append(f"  {key} / {var}: {err}")
                    else:
                        if result["ok"]:
                            success += 1
                        elif result["skipped"]:
                            skipped_count += 1
                        clean.append(key)
                except Exception as e:                    # noqa: BLE001
                    error_count += 1
                    failed_keys.append(key)
                    errors.append(f"  {key}: {e}")

            # ---- per-wave: record, then reclaim -------------------------
            if clean and provenance_source:
                pairs = [c for c in (cycle_from_key(k) for k in clean) if c]
                if pairs:
                    try:
                        from datastore_fill import record_provenance
                        record_provenance(base_dir, pairs, provenance_source)
                    except Exception as exc:              # noqa: BLE001
                        print(f"  WARNING: provenance not recorded for this wave: {exc}", file=sys.stderr)

            if clean and delete_after:
                wave_files = [f for k, f in wave if k in set(clean)]
                for paths in wave_files:
                    for path in paths:
                        try:
                            freed += os.path.getsize(path)
                            os.remove(path)
                            deleted += 1
                        except OSError:
                            pass

            pct = done / total * 100
            tail = f"  {freed / (1024 ** 3):.1f} GB freed" if delete_after else ""
            print(f"  [{done}/{total}] ({pct:.0f}%) - {success} OK, "
                  f"{skipped_count} present, {error_count} "
                  f"errors{tail}", flush=True)

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        if delete_after:
            print(f"  Raw chunks kept for {len(failed_keys)} failed cycle(s) - "
                  f"retry with --skip_download --reprocess")

    print(f"\nDone. Newly processed {success}/{total} repeat "
          f"cycles" + (f", {skipped_count} already present" if skipped_count else "") + ".")
    if delete_after:
        print(f"  Deleted {deleted:,} raw chunk file(s), freed "
              f"{freed / (1024 ** 3):.1f} GB")
    return {"processed": success, "skipped": skipped_count,
            "errors": error_count, "deleted": deleted, "freed_bytes": freed}


def group_local_files(local_dir, start_dt, end_dt,
                       chunk_filter=None, minute_filter=None):
    """
    Group already-downloaded .nc files by repeat cycle.

    Used with --skip_download when files are already present locally.
    Applies the same date, chunk, and minute filters as the SFTP path.

    Returns:
        dict: {group_key: [file_paths]}
    """
    if not os.path.isdir(local_dir):
        print(f"ERROR: Local directory does not exist: {local_dir}")
        return {}

    nc_files = [
        f for f in os.listdir(local_dir) if f.endswith('.nc')
    ]
    print(f"Found {len(nc_files)} .nc files in {local_dir}")

    groups = defaultdict(list)

    for filename in sorted(nc_files):
        info = parse_fci_filename(filename)

        # Date range filter
        if info['sensing_start']:
            if not (start_dt <= info['sensing_start'] <= end_dt):
                continue
        else:
            continue

        # Minute filter
        if minute_filter is not None and info['repeat_cycle_in_day']:
            minute = nominal_minute_of_hour(info['repeat_cycle_in_day'])
            if minute not in minute_filter:
                continue

        # Chunk filter (body only; trailers always pass)
        if info['is_body'] and chunk_filter is not None:
            if info['chunk_number'] is not None \
                    and info['chunk_number'] not in chunk_filter:
                continue

        # Group key
        if info['sensing_start'] and info['repeat_cycle_in_day'] is not None:
            date_str = info['sensing_start'].strftime('%Y%m%d')
            rc = info['repeat_cycle_in_day']
            key = f"{date_str}_{rc:04d}"
        else:
            key = "unknown"

        groups[key].append(os.path.join(local_dir, filename))

    print(f"Grouped into {len(groups)} repeat cycles")
    return dict(groups)


def clip_gaps_to_range(gaps, start_dt, end_dt):
    """Drop gap entries outside [start_dt, end_dt].

    The gap list describes the whole archive, because that is what the
    summary measured. --start/--end say which part of it this run is for,
    so they have to be applied here as well: without it, asking for
    October fetches January, which is both surprising and expensive.

    Boundary days are trimmed by time of day, not just by date, so a run
    starting at 12:00 does not pull that morning.
    """
    lo_date = start_dt.strftime("%Y-%m-%d")
    hi_date = end_dt.strftime("%Y-%m-%d")
    lo_time = start_dt.strftime("%H:%M")
    hi_time = end_dt.strftime("%H:%M")

    out = {}
    for date_str in sorted(gaps):
        if date_str < lo_date or date_str > hi_date:
            continue
        times = list(gaps[date_str])
        if date_str == lo_date:
            times = [t for t in times if t >= lo_time]
        if date_str == hi_date:
            times = [t for t in times if t <= hi_time]
        if times:
            out[date_str] = times
    return out


def split_gaps_by_window(gaps, n_months):
    """Split {date: [HH:MM]} into consecutive n-month windows.

    Returns [(label, gaps_subset), ...] ordered in time. With n_months
    None the whole gap list comes back as a single window, which is the
    unbatched behaviour.

    Windows are cut on the gap dates themselves rather than on a
    contiguous calendar range: an archive with a hole from October to
    February should not spend windows on months that have nothing to
    fetch.
    """
    if not gaps:
        return []
    if not n_months:
        days = sorted(gaps)
        return [(f"{days[0]} .. {days[-1]}", gaps)]
    if n_months < 1:
        raise ValueError(f"--batch_months must be >= 1, got {n_months}")

    buckets = {}
    for date_str in sorted(gaps):
        y, m = int(date_str[:4]), int(date_str[5:7])
        idx = (y * 12 + (m - 1)) // n_months
        buckets.setdefault(idx, {})[date_str] = gaps[date_str]

    out = []
    for idx in sorted(buckets):
        sub = buckets[idx]
        days = sorted(sub)
        out.append((f"{days[0]} .. {days[-1]}", sub))
    return out


def datastore_window(gaps_subset, data_dir, variables, chunk_filter,
                     minute_filter, credentials_file, workers,
                     overwrite=False, delete_after=False):
    """Fetch one window's gaps, extract them, and reclaim their raw.

    The whole point of doing this per window: fill_gaps writes raw .nc and
    nothing else, so fetching the entire gap list before extracting any of
    it would need room for every chunk at once. On a wide range that is
    terabytes. Interleaving bounds peak raw to one window.

    Returns (files_downloaded, exit_code).
    """
    from datastore_fill import fill_gaps, print_report

    raw_dir = os.path.join(data_dir, '_raw_chunks')
    report = fill_gaps(gaps_subset, raw_dir=raw_dir,
                       credentials_file=credentials_file, dry_run=False)
    print_report(report)

    if not report['files_downloaded']:
        return 0, 0

    # Extract only the dates this window touched.
    dates = sorted(report['timesteps_filled'])
    if not dates:
        return report['files_downloaded'], 0

    span_start = datetime.datetime.strptime(min(dates), '%Y-%m-%d')
    span_end = (datetime.datetime.strptime(max(dates), '%Y-%m-%d')
                + datetime.timedelta(days=1) - datetime.timedelta(seconds=1))
    groups = group_local_files(raw_dir, span_start, span_end,
                              chunk_filter=chunk_filter,
                              minute_filter=minute_filter)
    day_keys = {d.replace('-', '') for d in dates}
    groups = {k: v for k, v in groups.items() if k.split('_')[0] in day_keys}

    counts = process_groups(groups, data_dir, variables, workers=workers,
                            overwrite=overwrite, delete_after=delete_after,
                            provenance_source='datastore')
    return report['files_downloaded'], (1 if counts['errors'] else 0)


def process_datastore_fill(missing_json_path, base_dir, variables,
                            chunk_filter=ROMANIA_CHUNKS,
                            minute_filter=None, workers=10,
                            overwrite=False, delete_after=False):
    """
    Extract .npy for the cycles a Data Store backfill just recovered.

    datastore_fill.fill_gaps() writes raw .nc into _raw_chunks/ and stops.
    The coverage summary counts those .nc directly, so without this step
    the reported percentage rises while no .npy is produced and
    reproject.py still finds nothing. Called after --source
    close that gap.

    Scoped to the dates the fill touched rather than the whole archive,
    and paired with the skip-existing check in process_repeat_cycle, so
    the already-processed cycles on those dates cost a stat() each.

    Args:
        missing_json_path (Path or str): mtg_missing_timesteps.json, which
            carries the `datastore_fill` report block.
        base_dir (str): Root output directory (holds _raw_chunks/).
        variables (list): FCI channel names to extract.
        chunk_filter (set): Allowed chunk numbers.
        minute_filter (set or None): Allowed minutes within the hour.
        workers (int): Process pool size.
        overwrite (bool): Re-extract even when the .npy is already there.

    Returns:
        int: process exit code — 0 on success, 1 if any cycle failed.
    """
    path = Path(missing_json_path)
    if not path.is_file():
        print(f"\nWARNING: {path} not found — cannot tell which cycles the "
              f"backfill recovered. Run the pipeline again with "
              f"--skip_download to process them.")
        return 1

    with open(path, 'r', encoding='utf-8') as fh:
        report = json.load(fh).get('datastore_fill', {})

    if not report:
        print("\nNo datastore_fill block in the summary — nothing to "
              "process.")
        return 0
    if report.get('dry_run'):
        print("\nBackfill was a dry run — no files to process.")
        return 0
    if not report.get('files_downloaded'):
        print("\nBackfill downloaded nothing — no cycles to process.")
        return 0

    # Which dates gained files. `timesteps_filled` is the authoritative
    # record; fall back to parsing the filenames when the cycle mapping
    # came back empty, so a partial report still gets processed.
    dates = set(report.get('timesteps_filled', {}))
    if not dates:
        for name in report.get('files', []):
            info = parse_fci_filename(name)
            if info['sensing_start']:
                dates.add(info['sensing_start'].strftime('%Y-%m-%d'))
    if not dates:
        print("\nBackfill report lists no recoverable dates — skipping.")
        return 1

    print("\n" + "=" * 70)
    print(f"Processing backfilled cycles on {len(dates)} date(s)")
    print("=" * 70)

    day_keys = {d.replace('-', '') for d in dates}
    span_start = datetime.datetime.strptime(min(dates), '%Y-%m-%d')
    span_end = (datetime.datetime.strptime(max(dates), '%Y-%m-%d')
                + datetime.timedelta(days=1)
                - datetime.timedelta(seconds=1))

    groups = group_local_files(
        os.path.join(base_dir, '_raw_chunks'), span_start, span_end,
        chunk_filter=chunk_filter, minute_filter=minute_filter,
    )
    # group_local_files spans min..max, which may cover untouched dates in
    # between — keep only the days the fill actually wrote to.
    groups = {k: v for k, v in groups.items()
              if k.split('_')[0] in day_keys}

    # 'datastore' per wave, so an interrupted extraction still leaves the
    # cycles it finished correctly attributed.
    counts = process_groups(groups, base_dir, variables,
                            workers=workers, overwrite=overwrite,
                            delete_after=delete_after,
                            provenance_source="datastore")
    return 1 if counts['errors'] else 0


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Download and process MTG FCI L1C satellite data. '
            'Transfers chunks via SFTP from a remote machine and '
            'extracts per-channel radiance data.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--start', '-s', type=str, default=None,
        help='Start datetime (format: yyyy/mm/dd-hhmm). Required except '
             'with --record_existing, which acts on whatever is already '
             'on disk and so has no window.',
    )
    parser.add_argument(
        '--end', '-e', type=str, default=None,
        help='End datetime (format: yyyy/mm/dd-hhmm). See --start.',
    )
    parser.add_argument(
        '--password_file', '-pw', type=str, default=None,
        help='Path to text file containing the SSH password. Required only '
             'when actually reaching the NMA server: not needed with '
             '--skip_download or --source datastore.',
    )
    parser.add_argument(
        '--products_file', '-pf', type=str,
        default='satellite_products.json',
        help='Path to JSON file with channel names '
             '(default: satellite_products.json)',
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default=None,
        help='Output directory (default: <script_dir>/MTG, i.e. '
             'our_data/satellite_data/MTG). Anchored to the script '
             "location, NOT the caller's cwd, so running from any "
             'working directory keeps the on-disk layout the rest of '
             'the pipeline expects.',
    )
    parser.add_argument(
        '--full_disk', action='store_true',
        help='Download all 40 chunks instead of Romania-only',
    )
    parser.add_argument(
        '--source', type=str, default='nma',
        choices=['nma', 'datastore', 'both', 'local'],
        help="Where to pull from. 'nma' (default) is the internal server "
             "over SFTP. 'datastore' fetches ONLY the cycles listed as "
             "missing in mtg_missing_timesteps.json, so run "
             "summarize_mtg.py first - that file is how this knows what "
             "to ask for. 'both' does NMA then fills whatever is still "
             "missing from the Data Store. 'local' downloads nothing at "
             "all: it extracts the raw chunks already in _raw_chunks/ to "
             ".npy and, with --delete_raw, reclaims each wave as it "
             "finishes - for raw left behind by an interrupted run, or an "
             "archive fetched before extraction was wired in. Whichever "
             "is used is recorded per cycle in provenance.json and "
             "reported by the summary.",
    )
    parser.add_argument(
        '--batch_months', type=int, default=None, metavar='N',
        help="Split the range into N-month windows and run them one after "
             "another, each downloading, extracting and (with "
             "--delete_raw) reclaiming before the next begins. Bounds peak "
             "raw-on-disk to one window instead of the whole range. "
             "Calendar-aligned, so windows match the monthly coverage "
             "chart. A failed window is reported and the rest continue "
             "unless --stop_on_error.",
    )
    parser.add_argument(
        '--stop_on_error', action='store_true',
        help="With --batch_months, abort at the first failed window "
             "instead of carrying on.",
    )
    parser.add_argument(
        '--record_existing', type=str, default=None,
        choices=['nma', 'datastore'],
        help="Stamp every cycle already on disk as having come from this "
             "source, then exit. For data downloaded before the ledger "
             "existed, where you know the origin but the files do not "
             "record it. Cycles already recorded are left untouched. Run "
             "this BEFORE pulling from a second source, or the two become "
             "indistinguishable.",
    )
    parser.add_argument(
        '--provenance', type=str, default=None,
        choices=['nma', 'datastore'],
        help="With --source local, stamp the cycles this run extracts as "
             "having come from this source. Omitted, nothing is recorded: "
             "raw sitting in _raw_chunks/ carries no evidence of its "
             "origin, and guessing would make the ledger worthless. "
             "Cycles already recorded are never overwritten.",
    )
    parser.add_argument(
        '--missing_json', type=str, default=None,
        help="Gap list driving --source datastore (default: "
             "mtg_missing_timesteps.json at the project root).",
    )
    parser.add_argument(
        '--fill_dry_run', action='store_true',
        help="With --source datastore, list what would be fetched and "
             "stop. No credentials are spent, nothing is written.",
    )
    parser.add_argument(
        '--no_fill_incomplete', action='store_true',
        help="With --source datastore, recover only fully-absent cycles, "
             "skipping those missing just one of the two Romania chunks.",
    )
    parser.add_argument(
        '--eumdac_credentials', type=str, default=None,
        help='Two-line EUMDAC credentials file (key, then secret) used by '
             '--source datastore. Falls back to EUMDAC_KEY / EUMDAC_SECRET.',
    )
    parser.add_argument(
        '--raw_dir', type=str, default=None, metavar='PATH',
        help="Read raw .nc chunks from here instead of "
             "<output_dir>/_raw_chunks. The .npy a cycle produces is "
             "~3.8x the raw it replaces, so a drive can hold the chunks "
             "and still have no room for the arrays; this lets the "
             "extraction read one disk and write another.",
    )
    parser.add_argument(
        '--spill_dir', type=str, nargs='+', default=None, metavar='PATH',
        help="Further store(s) to rotate through when the active one "
             "drops below --min_free_gb. Re-evaluated before every "
             "window and in both directions, so a disk that --delete_raw "
             "has since emptied is used again rather than abandoned. "
             "Stores on the same volume are collapsed. Requires "
             "--batch_months, since the check happens between windows - "
             "a date is never split across disks.",
    )
    parser.add_argument(
        '--min_free_gb', type=float, default=50.0, metavar='GB',
        help="Free space below which --spill_dir takes over "
             "(default: 50).",
    )
    parser.add_argument(
        '--skip_download', action='store_true',
        help='Skip SFTP download; process files already in '
             '<output_dir>/_raw_chunks/',
    )
    parser.add_argument(
        '--delete_raw', action='store_true',
        help='After extracting every cycle to .npy, delete the raw .nc '
             'chunks. Irreversible, and it removes the --skip_download '
             'reprocessing path, so it is opt-in and never fires on its '
             'own. Skipped entirely if any cycle reported an error. '
             'Coverage is then measured from the .npy output.',
    )
    parser.add_argument(
        '--delete_only', action='store_true',
        help="Delete the raw chunks in the window without re-extracting "
             "first. Implies --skip_download. Use after a summary has "
             "confirmed coverage: the extraction pass would just re-check "
             "the same .npy files, which on a full archive is 100k+ stat "
             "calls. Nothing is verified here, so run the summary first.",
    )
    parser.add_argument(
        '--reprocess', action='store_true',
        help='Re-extract cycles whose .npy already exist. Off by default, '
             'so a --skip_download pass only fills in what is missing.',
    )
    parser.add_argument(
        '--timesteps', type=str, nargs='+', default=None,
        help="Override the minute filter (e.g. 00 10 30 40). "
             "Default: read from timestep_config.json. "
             "Use 'all' to keep every native 10-min timestep.",
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=10,
        help='Number of parallel workers for repeat cycle processing '
             '(default: 10)',
    )

    args = parser.parse_args()

    # The password is only ever read to open the SFTP session, so demand it
    # for download runs only. Enforced here rather than via required=True so
    # a --skip_download pass over _raw_chunks/ needs no credential at all.
    # --record_existing only writes the ledger for data already present:
    # no window to download over, and no server to authenticate to.
    if not args.record_existing and not (args.start and args.end):
        parser.error('--start and --end are required (except with '
                     '--record_existing)')

    if args.delete_only:
        args.skip_download = True

    if args.spill_dir and args.batch_months is None:
        parser.error(
            '--spill_dir needs --batch_months: the free-space check runs '
            'between windows, and without batching there is only one.')

    # 'local' is exactly "the nma path with the download removed", so it is
    # expressed as such rather than as a second copy of the same stage.
    if args.source == 'local':
        args.skip_download = True
    elif args.provenance:
        parser.error('--provenance applies to --source local; the other '
                     'sources already know where their data came from.')

    needs_sftp = (args.source in ('nma', 'both')
                  and not args.skip_download
                  and not args.record_existing)
    if needs_sftp and not args.password_file:
        parser.error(
            '--password_file is required to reach the NMA server. '
            'Not needed with --skip_download, nor with --source datastore, '
            'which authenticates to EUMETSAT via --eumdac_credentials.')

    # Resolve output directory. Anchor to the script's OWN location
    # (our_data/satellite_data/) rather than the caller's cwd, so
    # running `python our_data/satellite_data/pipeline_msg_mtg.py ...`
    # from the project root lands data at
    # our_data/satellite_data/MTG/ (matches the tree diagram in
    # README.md) instead of ./MTG at the project root.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, 'MTG')

    # Creating a store is only ever done deliberately. The default path is
    # usually a junction onto whichever drive holds the archive; if that
    # link is absent - removed, or pointing at a drive that is not mounted
    # - silently creating a fresh empty directory here would send hundreds
    # of gigabytes to the wrong disk without a word. Reading an existing
    # store by default stays free.
    if args.output_dir is None and not os.path.isdir(default_dir):
        parser.error(
            f"no MTG store at {default_dir}, and --output_dir was not "
            f"given.{chr(10)}"
            f"Pass --output_dir explicitly to say which disk the store "
            f"should live on, e.g.{chr(10)}"
            f"    --output_dir G:{os.sep}nowcasting{os.sep}mtg_store"
            f"{chr(10)}"
            f"If the default path is a junction, check that its target "
            f"drive is mounted rather than creating a second store.")

    data_dir = args.output_dir or default_dir
    os.makedirs(data_dir, exist_ok=True)

    # Resolve products file relative to the script dir (anchoring to
    # the caller's cwd would silently drop the default when running
    # from anywhere other than our_data/satellite_data/).
    products_file = args.products_file
    if not os.path.isabs(products_file):
        products_file = os.path.join(script_dir, products_file)

    # Print summary
    print(f"Date range     : {args.start} to {args.end}")
    print(f"Output dir     : {data_dir}")
    print(f"Products file  : {products_file}")
    print(f"Source         : {args.source}")
    print(f"Skip download  : {args.skip_download}")
    print(f"Workers        : {args.workers}")

    # Load variables
    mtg_variables = load_variables_from_file(products_file, satellite='mtg')
    if not mtg_variables:
        print("ERROR: No valid MTG variables found in products file.")
        sys.exit(1)

    # Chunk filter
    chunk_filter = None if args.full_disk else ROMANIA_CHUNKS
    print(
        f"Chunk filter   : "
        f"{sorted(chunk_filter) if chunk_filter else 'full disk (all 40)'}"
    )

    # Minute filter
    if args.timesteps is not None and args.timesteps != ['all']:
        minute_filter = {int(m) for m in args.timesteps}
        print(f"Minute filter  : {sorted(minute_filter)} (CLI override)")
    elif args.timesteps == ['all']:
        minute_filter = None
        print("Minute filter  : none (keeping every native 10-min timestep)")
    else:
        minute_filter, step = load_timestep_filter("mtg")
        print(
            f"Minute filter  : {sorted(minute_filter)} "
            f"(timestep_config.json, step={step} min)"
        )

    # Run
    if args.record_existing:
        raise SystemExit(
            0 if record_existing_provenance(
                data_dir, mtg_variables, args.record_existing) >= 0 else 1)

    # The NMA server is the primary source; --source datastore skips
    # it entirely and pulls only the gaps the summary already found.
    if args.source in ('nma', 'both', 'local'):
        # A local pass has no server to attribute to, so it records only
        # what --provenance was explicitly told.
        prov = args.provenance if args.source == 'local' else 'nma'

        # The active output store, re-chosen before every window.
        stores = store_candidates(data_dir, args.spill_dir)
        store = {"dir": data_dir}
        if len(stores) > 1:
            print(f"Stores         : {len(stores)} volume(s), "
                  f"min free {args.min_free_gb:.0f} GB")
            for c in stores:
                print(f"  {free_gb(c):8.0f} GB free  {c}")

        def _window(ws, we):
            if len(stores) > 1:
                store["dir"] = choose_store(store["dir"], stores,
                                            args.min_free_gb)
            out_dir = store["dir"]
            fetch_and_process_mtg(
                ws.strftime('%Y/%m/%d-%H%M'), we.strftime('%Y/%m/%d-%H%M'),
                out_dir, mtg_variables,
                password_file=args.password_file,
                chunk_filter=chunk_filter,
                minute_filter=minute_filter,
                skip_download=args.skip_download,
                workers=args.workers,
                overwrite=args.reprocess,
                delete_raw=args.delete_raw or args.delete_only,
                delete_only=args.delete_only,
                provenance_source=prov,
                raw_dir=args.raw_dir,
            )

            # Record where this window's dates landed, so reproject.py can
            # find them without stat-ing every store.
            if not args.delete_only:
                try:
                    from store_registry import register
                    register(out_dir, dates_in_window(ws, we))
                except Exception as exc:            # noqa: BLE001
                    print(f"  WARNING: store index not updated: {exc}",
                          file=sys.stderr)


        start_dt, end_dt = parse_date_range(args.start, args.end)
        if start_dt is None:
            sys.exit(2)

        if args.batch_months is not None:
            # One window at a time, so peak raw-on-disk is a window rather
            # than the whole range: each downloads, extracts and (with
            # --delete_raw) reclaims before the next begins.
            try:
                windows = month_windows(start_dt, end_dt, args.batch_months)
            except ValueError as exc:
                parser.error(str(exc))
            print(f"Batching       : {len(windows)} window(s) of "
                  f"{args.batch_months} month(s)")
            rc = run_batched(args, windows, _window)
            if rc:
                raise SystemExit(rc)
        else:
            _window(start_dt, end_dt)

    # ---- Data Store stage -------------------------------------------
    # Lives here rather than in the summariser: downloading is the
    # pipeline's job, and the choice of source belongs to whoever runs it.
    # The summary's role is to say what is missing; this acts on that.
    if args.source in ('datastore', 'both'):
        import json as _json
        from datastore_fill import (collect_gaps, fill_gaps, print_report,
                                    record_provenance)

        missing_json = Path(args.missing_json) if args.missing_json else (
            PROJECT_ROOT / 'mtg_missing_timesteps.json')
        if not missing_json.is_file():
            raise SystemExit(
                f"Gap list not found: {missing_json}" + chr(10) +
                "--source datastore fetches only what the summary reports "
                "as missing, so run it first:" + chr(10) +
                "    python our_data/satellite_data/summarize_mtg.py "
                "--start <YYYY-MM-DD> --end <YYYY-MM-DD>")

        with open(missing_json, encoding='utf-8') as fh:
            gaps = collect_gaps(_json.load(fh),
                                include_incomplete=not args.no_fill_incomplete)

        print(chr(10) + "=" * 70)
        print("EUMETSAT Data Store")
        print("=" * 70)
        print(f"  Gap list : {missing_json}")
        n_all = sum(len(v) for v in gaps.values())
        print(f"  Listed   : {n_all:,} cycle(s) across {len(gaps)} date(s)")

        # The gap list spans the whole archive; --start/--end say which
        # part of it to fetch.
        ds_start, ds_end = parse_date_range(args.start, args.end)
        if ds_start is None:
            sys.exit(2)
        gaps = clip_gaps_to_range(gaps, ds_start, ds_end)
        n_win = sum(len(v) for v in gaps.values())
        print(f"  Window   : {ds_start:%Y-%m-%d %H:%M} .. "
              f"{ds_end:%Y-%m-%d %H:%M}")
        print(f"  Selected : {n_win:,} cycle(s) across {len(gaps)} date(s)"
              + (f"  ({n_all - n_win:,} outside the window)"
                 if n_all != n_win else ""))
        if not gaps:
            raise SystemExit(
                "Nothing to fetch: no gap in the requested window.")

        if args.fill_dry_run:
            # Dry run stays whole-list: it downloads nothing, and seeing
            # the complete set is the point of asking.
            report = fill_gaps(
                gaps, raw_dir=os.path.join(data_dir, '_raw_chunks'),
                credentials_file=args.eumdac_credentials, dry_run=True)
            print_report(report)
            raise SystemExit(0)

        # Download -> extract -> reclaim, one window at a time. Fetching
        # the whole gap list first would need room for every raw chunk
        # simultaneously; on a wide range that is terabytes.
        try:
            windows = split_gaps_by_window(gaps, args.batch_months)
        except ValueError as exc:
            parser.error(str(exc))

        if args.batch_months:
            print(f"  Windows  : {len(windows)} of {args.batch_months} "
                  f"month(s), reclaimed as each completes")

        total_files = 0
        failures = []
        for i, (label, subset) in enumerate(windows, start=1):
            n_cycles = sum(len(v) for v in subset.values())
            print(chr(10) + "=" * 70)
            print(f"[window {i}/{len(windows)}]  {label}  "
                  f"({n_cycles} cycle(s), {len(subset)} date(s))")
            print("=" * 70)
            try:
                got, rc = datastore_window(
                    subset, data_dir, mtg_variables, chunk_filter,
                    minute_filter, args.eumdac_credentials, args.workers,
                    overwrite=args.reprocess,
                    delete_after=args.delete_raw,
                )
                total_files += got
                if rc:
                    failures.append((label, "extraction errors"))
            except Exception as exc:                      # noqa: BLE001
                failures.append((label, str(exc)))
                print(f"  WINDOW FAILED: {exc}", file=sys.stderr)
                if args.stop_on_error:
                    break

        print(chr(10) + "=" * 70)
        print("Data Store run complete")
        print("=" * 70)
        print(f"  windows        : {len(windows)}")
        print(f"  files fetched  : {total_files:,}")
        print(f"  failed windows : {len(failures)}")
        for label, why in failures:
            print(f"    {label}  {why}")
        if failures:
            print(chr(10) + "  Re-run the same command to retry: cycles "
                  "already present are skipped.")
            raise SystemExit(1)
        raise SystemExit(0)
