"""
pipeline_msg_mtg.py — MTG FCI L1C data pipeline (Romania)

Downloads FCI chunk files from a remote machine via SFTP (paramiko),
filters to Romania-covering chunks (35–36), reads them directly with
netCDF4, and saves per-channel NetCDF files with geostationary
projection metadata for downstream regridding.

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
except ImportError:
    print(
        "WARNING: hdf5plugin not installed. FCI L1c reading may fail.\n"
        "Install with: pip install hdf5plugin"
    )


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"

# FCI chunks covering the Romania study area.
# From the Météo-France FCI scan diagram (and verified against the
# geostationary projection), Romania falls in chunks 35–36 of the
# 40 body chunks per repeat cycle.
ROMANIA_CHUNKS = {35, 36}

# Remote machine hosting the FCI data (ANM internal network)
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

def process_repeat_cycle(files, base_dir, variables, group_key=None):
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

    Returns:
        dict with keys: 'key', 'ok' (list), 'failed' (list of (var, error))
    """
    from netCDF4 import Dataset as NC4Dataset

    result = {'key': group_key, 'ok': [], 'failed': []}

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
            <base_dir>/_raw_chunks/.
        workers (int): Number of parallel workers for repeat cycle processing.
    """
    start, end = parse_date_range(start_str, end_str)
    if start is None:
        return

    local_download_dir = os.path.join(base_dir, '_raw_chunks')

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

    # ---- Step 3: Process repeat cycles in parallel ----
    sorted_items = sorted(groups.items())
    total = len(sorted_items)
    n_workers = min(workers, total)
    print(f"\nProcessing {total} repeat cycles "
          f"({len(variables)} variables, {n_workers} workers)...\n")

    success_count = 0
    error_count = 0
    errors = []  # collect errors for summary

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for key, files in sorted_items:
            future = executor.submit(
                process_repeat_cycle,
                files, base_dir, variables, group_key=key
            )
            futures[future] = key

        completed = 0
        for future in as_completed(futures):
            key = futures[future]
            completed += 1

            try:
                result = future.result()
                if result['failed']:
                    error_count += 1
                    for var, err in result['failed']:
                        errors.append(f"  {key} / {var}: {err}")
                if result['ok']:
                    success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(f"  {key}: {e}")

            # Progress update every n_workers completions (or at the end)
            if completed % n_workers == 0 or completed == total:
                pct = completed / total * 100
                print(f"  [{completed}/{total}] ({pct:.0f}%) — "
                      f"{success_count} OK, {error_count} errors")

    # Error summary
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors[:20]:  # cap at 20 to avoid flooding
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    print(
        f"\nDone. Successfully processed {success_count}/{len(groups)} "
        f"repeat cycles."
    )


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
        '--start', '-s', type=str, required=True,
        help='Start datetime (format: yyyy/mm/dd-hhmm)',
    )
    parser.add_argument(
        '--end', '-e', type=str, required=True,
        help='End datetime (format: yyyy/mm/dd-hhmm)',
    )
    parser.add_argument(
        '--password_file', '-pw', type=str, required=True,
        help='Path to text file containing the SSH password',
    )
    parser.add_argument(
        '--products_file', '-pf', type=str,
        default='satellite_products.json',
        help='Path to JSON file with channel names '
             '(default: satellite_products.json)',
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default=None,
        help='Output directory (default: ./MTG in cwd)',
    )
    parser.add_argument(
        '--full_disk', action='store_true',
        help='Download all 40 chunks instead of Romania-only',
    )
    parser.add_argument(
        '--skip_download', action='store_true',
        help='Skip SFTP download; process files already in '
             '<output_dir>/_raw_chunks/',
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

    # Resolve output directory
    base_dir = os.getcwd()
    data_dir = args.output_dir or os.path.join(base_dir, 'MTG')
    os.makedirs(data_dir, exist_ok=True)

    # Resolve products file
    products_file = args.products_file
    if not os.path.isabs(products_file):
        products_file = os.path.join(base_dir, products_file)

    # Print summary
    print(f"Date range     : {args.start} to {args.end}")
    print(f"Output dir     : {data_dir}")
    print(f"Products file  : {products_file}")
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
    fetch_and_process_mtg(
        args.start, args.end,
        data_dir, mtg_variables,
        password_file=args.password_file,
        chunk_filter=chunk_filter,
        minute_filter=minute_filter,
        skip_download=args.skip_download,
        workers=args.workers,
    )


# =============================================================================
# DISABLED — Original Data Store + manual processing code
# =============================================================================
# The code below is the original implementation that used the EUMETSAT Data
# Store (eumdac) for downloading and manual netCDF4-based chunk reading
# for processing. It has been replaced by the SFTP + direct netCDF4 approach
# above. Preserved here for reference.

_DATASTORE_MANUAL_DISABLED = r'''

import eumdac
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from netCDF4 import Dataset as NC4Dataset

COLLECTIONS = {
    'msg_rss':   'EO:EUM:DAT:MSG:MSG15-RSS',
    'fci_fdhsi': 'EO:EUM:DAT:0662',
    'fci_hrfi':  'EO:EUM:DAT:0665',
}

# Old chunk set included buffer chunks 34 and 38 which are not needed
# ROMANIA_CHUNKS = set(range(34, 39))  # {34, 35, 36, 37, 38}

MAX_WORKERS = min(4, os.cpu_count() or 1)


def get_datastore_token():
    """
    Authenticate with the EUMETSAT Data Store.
    """
    consumer_key = os.environ.get(
        'EUMETSAT_CONSUMER_KEY', 'XVibc7bODTpUS7UWHh4169RiX90a'
    )
    consumer_secret = os.environ.get(
        'EUMETSAT_CONSUMER_SECRET', 'RhfFLw3tUbK40QMOoWfXT9H5YK4a'
    )
    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    print(f"This token '{token}' expires {token.expiration}")
    return token


def floor_dt(dt, delta):
    return dt.min + math.floor((dt - dt.min) / delta) * delta


def get_product_sensing_time(product):
    """Extract the sensing start time from a Data Store product."""
    return product.sensing_start


def extract_date_time_fci(product):
    """Extract date and time from an MTG FCI product's sensing_start."""
    dt = product.sensing_start
    dir_name = f"nc4_{dt.strftime('%Y-%m-%d')}-Romania"
    time_str = dt.strftime('%H%M')
    return dir_name, time_str


def extract_date_time_fci_from_filename(filename):
    """Extract date and time from an MTG FCI chunk filename (fallback)."""
    pattern = r'_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})\d{2}_'
    match = re.search(pattern, filename)
    if not match:
        return None, None
    year, month, day, hour, minute = match.groups()
    return f"nc4_{year}-{month}-{day}-Romania", f"{hour}{minute}"


def download_fci_product_entries(product, temp_dir, chunk_filter=None):
    """
    Download NetCDF chunk files from a single FCI product.
    """
    nc_entries = [entry for entry in product.entries if entry.endswith('.nc')]
    if not nc_entries:
        print(f"  No NetCDF entries found in product {product}")
        return []

    if chunk_filter is not None:
        filtered_entries = []
        skipped = 0
        for entry in nc_entries:
            if 'TRAIL' in entry:
                filtered_entries.append(entry)
                continue
            chunk_num = parse_chunk_number_old(entry)
            if chunk_num is not None and chunk_num in chunk_filter:
                filtered_entries.append(entry)
            elif chunk_num is None:
                filtered_entries.append(entry)
            else:
                skipped += 1
        print(f"  Chunk filter: keeping {len(filtered_entries)}/{len(nc_entries)} "
              f"entries (skipped {skipped} outside Romania)")
        nc_entries = filtered_entries

    downloaded_files = []
    print(f"  Downloading {len(nc_entries)} chunk files...")

    for entry in nc_entries:
        temp_file_path = os.path.join(temp_dir, entry)
        try:
            with product.open(entry=entry) as fsrc, \
                    open(temp_file_path, mode='wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
            downloaded_files.append(temp_file_path)
        except Exception as e:
            print(f"  Error downloading {entry}: {e}")

    print(f"  Downloaded {len(downloaded_files)}/{len(nc_entries)} chunks")
    return downloaded_files


def parse_chunk_number_old(filename):
    """
    Extract the chunk number from an FCI L1c filename.
    """
    # _<repeat_cycle_in_day>_<chunk_number>.nc
    match = re.search(r'_(\d{4})_(\d{4})\.nc$', filename)
    if match:
        chunk_num = int(match.group(2))
        return chunk_num
    match = re.search(r'CHK[_-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def fci_scanning_angles_to_latlon(x_rad, y_rad, sub_lon_deg=0.0):
    """
    Convert FCI geostationary scanning angles to latitude/longitude.
    """
    R_eq = 6378.137
    R_pol = 6356.7523
    H = 42164.0
    sub_lon = np.radians(sub_lon_deg)
    xx, yy = np.meshgrid(x_rad, y_rad)
    cos_x = np.cos(xx); cos_y = np.cos(yy)
    sin_x = np.sin(xx); sin_y = np.sin(yy)
    a = sin_x**2 + cos_x**2 * (cos_y**2 + (R_eq / R_pol)**2 * sin_y**2)
    b = -2 * H * cos_x * cos_y
    c = H**2 - R_eq**2
    discriminant = b**2 - 4 * a * c
    valid = discriminant >= 0
    lat = np.full(xx.shape, np.nan, dtype=np.float64)
    lon = np.full(xx.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return lat.astype(np.float32), lon.astype(np.float32)
    a_v = a[valid]; b_v = b[valid]; disc_v = discriminant[valid]
    cos_x_v = cos_x[valid]; cos_y_v = cos_y[valid]
    sin_x_v = sin_x[valid]; sin_y_v = sin_y[valid]
    sn = (-b_v - np.sqrt(disc_v)) / (2 * a_v)
    s1 = H - sn * cos_x_v * cos_y_v
    s2 = -sn * sin_x_v * cos_y_v
    s3 = sn * sin_y_v
    sxy = np.sqrt(s1**2 + s2**2)
    lon[valid] = np.degrees(np.arctan2(s2, s1) + sub_lon)
    lat[valid] = np.degrees(np.arctan((R_eq / R_pol)**2 * s3 / sxy))
    return lat.astype(np.float32), lon.astype(np.float32)


def _process_single_fci_variable(variable, body_files, dir_name, time_str,
                                  base_dir):
    """
    Worker function: process one FCI variable from chunk files (manual).
    """
    try:
        import hdf5plugin
    except ImportError:
        pass
    try:
        if variable in CHANNELS_1KM:
            res_key = '1km'
        elif variable in CHANNELS_2KM:
            res_key = '2km'
        else:
            res_key = '1km'
        chunk_arrays = []
        for chunk_file in body_files:
            try:
                ds = NC4Dataset(chunk_file, 'r')
                measured = ds['data'][variable]['measured']
                data = measured.variables['effective_radiance'][:]
                chunk_arrays.append(np.array(data, dtype=np.float32))
                ds.close()
            except Exception:
                try: ds.close()
                except: pass
        if not chunk_arrays:
            return (variable, False, "no data found in chunks")
        combined = np.concatenate(chunk_arrays, axis=0)
        del chunk_arrays
        var_dir = os.path.join(base_dir, variable)
        date_dir = f"{dir_name}_{variable}"
        full_output_dir = os.path.join(var_dir, date_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        nc_filename = f"{dir_name}_{time_str}_{variable}.nc"
        output_path = os.path.join(full_output_dir, nc_filename)
        dims = ('y', 'x')
        ds_out = xr.Dataset(
            {variable: (dims, combined)},
            coords={
                'y': np.arange(combined.shape[0]),
                'x': np.arange(combined.shape[1]),
            }
        )
        ds_out.to_netcdf(output_path, engine='scipy')
        del combined, ds_out
        return (variable, True, f"saved {output_path}")
    except Exception as e:
        return (variable, False, str(e))


def process_fci_product_all_variables(product, downloaded_files, temp_dir,
                                       base_dir, variables):
    """
    Process ALL variables from downloaded FCI chunk files in parallel (manual).
    """
    if not downloaded_files:
        print(f"  No files to process")
        return
    dir_name, time_str = extract_date_time_fci(product)
    body_files = sorted([f for f in downloaded_files if 'TRAIL' not in f])
    if not body_files:
        print(f"  No body chunk files found")
        return
    print(f"  Processing {len(body_files)} body chunks for {len(variables)} variables "
          f"(max {MAX_WORKERS} parallel workers)")
    coord_dir = os.path.join(base_dir, 'coordinates')
    os.makedirs(coord_dir, exist_ok=True)
    for res_key, ref_channel in [('1km', 'vis_06'), ('2km', 'ir_105')]:
        vars_at_res = [v for v in variables
                       if (v in CHANNELS_1KM and res_key == '1km') or
                          (v in CHANNELS_2KM and res_key == '2km')]
        if not vars_at_res:
            continue
        lat_path = os.path.join(coord_dir, f'lat_{res_key}.npy')
        lon_path = os.path.join(coord_dir, f'lon_{res_key}.npy')
        if os.path.exists(lat_path) and os.path.exists(lon_path):
            lat_2d = np.load(lat_path)
            print(f"    {res_key} coordinates already exist: {lat_2d.shape}")
            del lat_2d
            continue
        try:
            x_arrays = []; y_arrays = []
            for chunk_file in body_files:
                ds = NC4Dataset(chunk_file, 'r')
                measured = ds['data'][ref_channel]['measured']
                x_1d = measured.variables['x'][:]
                y_1d = measured.variables['y'][:]
                y_arrays.append(np.array(y_1d))
                if len(x_arrays) == 0:
                    x_arrays.append(np.array(x_1d))
                ds.close()
            y_combined = np.concatenate(y_arrays)
            x_combined = x_arrays[0]
            lat_2d, lon_2d = fci_scanning_angles_to_latlon(x_combined, y_combined)
            np.save(lat_path, lat_2d)
            np.save(lon_path, lon_2d)
            print(f"    {res_key} grid: {lat_2d.shape}")
            del lat_2d, lon_2d
        except Exception as e:
            print(f"    WARNING: Could not compute {res_key} coordinates: {e}")
    gc.collect()
    n_workers = min(MAX_WORKERS, len(variables))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for variable in variables:
            future = executor.submit(
                _process_single_fci_variable,
                variable, body_files, dir_name, time_str, base_dir
            )
            futures[future] = variable
        success_count = 0
        for future in as_completed(futures):
            variable = futures[future]
            try:
                var_name, success, message = future.result()
                if success:
                    print(f"    {var_name}: {message}")
                    success_count += 1
                else:
                    print(f"    {var_name}: FAILED - {message}")
            except Exception as e:
                print(f"    {variable}: EXCEPTION - {str(e)}")
    print(f"  Completed: {success_count}/{len(variables)} variables")


def cleanup_fci_temp(temp_dir):
    """Remove all temporary FCI chunk files and the temp directory."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"  Cleaned up temp directory: {temp_dir}")


def fetch_and_process_mtg_all_variables(start_str, end_str, base_dir, variables,
                                        collection='fci_fdhsi',
                                        chunk_filter=ROMANIA_CHUNKS,
                                        minute_filter=None):
    """
    Fetch and process MTG FCI data for ALL requested variables (Data Store).
    """
    start, end = parse_date_range(start_str, end_str)
    if start is None:
        return
    token = get_datastore_token()
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(COLLECTIONS[collection])
    all_products = list(selected_collection.search(dtstart=start, dtend=end))
    all_products.sort(key=lambda p: p.sensing_start)
    if minute_filter is not None:
        filtered_products = []
        skipped = 0
        for p in all_products:
            sensing_minute = get_product_sensing_time(p).minute
            if sensing_minute in minute_filter:
                filtered_products.append(p)
            else:
                skipped += 1
        print(f"MTG: {len(all_products)} products at native 10-min cadence")
        print(f"Timestep filter: keeping minutes {sorted(minute_filter)} "
              f"-> {len(filtered_products)} products (skipped {skipped})")
        all_products = filtered_products
    else:
        print(f"MTG: {len(all_products)} products at native 10-min cadence")
    filter_desc = f"chunks {sorted(chunk_filter)}" if chunk_filter else "full disk"
    print(f"Chunk filter: {filter_desc}")
    processed_count = 0
    for product in all_products:
        sensing_time = get_product_sensing_time(product)
        print(f"\nProcessing MTG FCI product at "
              f"{sensing_time.strftime('%Y-%m-%dT%H:%M:%S')} "
              f"[{processed_count + 1}/{len(all_products)}]")
        temp_dir = f'temp_output_mtg_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}'
        os.makedirs(temp_dir, exist_ok=True)
        try:
            downloaded_files = download_fci_product_entries(
                product, temp_dir, chunk_filter=chunk_filter
            )
            if not downloaded_files:
                continue
            process_fci_product_all_variables(
                product, downloaded_files, temp_dir, base_dir, variables
            )
            processed_count += 1
        finally:
            cleanup_fci_temp(temp_dir)
    print(f"\nMTG: processed {processed_count}/{len(all_products)} products")


def discover_chunk_numbers(start_str, end_str, collection='fci_fdhsi',
                           max_products=1):
    """Debug utility: print chunk structure of first product(s)."""
    start, end = parse_date_range(start_str, end_str)
    if start is None:
        return
    token = get_datastore_token()
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(COLLECTIONS[collection])
    products = list(selected_collection.search(dtstart=start, dtend=end))
    print(f"\nFound {len(products)} products. Inspecting first {max_products}:")
    for product in products[:max_products]:
        print(f"\nProduct: {product}")
        print(f"Sensing start: {product.sensing_start}")
        nc_entries = [e for e in product.entries if e.endswith('.nc')]
        print(f"Total .nc entries: {len(nc_entries)}")
        chunk_counts = {}
        for entry in sorted(nc_entries):
            chunk_num = parse_chunk_number_old(entry)
            chunk_counts[chunk_num] = chunk_counts.get(chunk_num, 0) + 1
            if len(chunk_counts) <= 5 or chunk_num in ROMANIA_CHUNKS:
                marker = " <-- ROMANIA" if chunk_num in ROMANIA_CHUNKS else ""
                print(f"  {entry} -> chunk {chunk_num}{marker}")
        print(f"\nChunk number distribution:")
        for num in sorted(chunk_counts.keys(), key=lambda x: (x is None, x)):
            in_romania = " <-- KEEP" if num in ROMANIA_CHUNKS else ""
            print(f"  Chunk {num}: {chunk_counts[num]} entries{in_romania}")
        would_download = sum(v for k, v in chunk_counts.items()
                            if k is not None and k in ROMANIA_CHUNKS)
        would_skip = sum(v for k, v in chunk_counts.items()
                        if k is not None and k not in ROMANIA_CHUNKS)
        unparsed = chunk_counts.get(None, 0)
        print(f"\nWith ROMANIA_CHUNKS filter: download {would_download}, "
              f"skip {would_skip}, unparsed {unparsed}")

'''
# end disabled block


# =============================================================================
# DISABLED — MSG SEVIRI functions
# =============================================================================
# The MSG branch was already disabled in the original pipeline.
# See the original file for the full MSG SEVIRI code if needed.

_MSG_DISABLED = r'''
# MSG SEVIRI RSS functions (snap_to_15min_slot, fetch_and_process_sat, etc.)
# were already disabled. See the original pipeline_msg_mtg.py for reference.
'''