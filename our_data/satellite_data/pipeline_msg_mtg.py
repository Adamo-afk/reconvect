import os
import re
import sys
import gc
import json
import eumdac
import datetime
import shutil
import math
import numpy as np
import xarray as xr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from netCDF4 import Dataset as NC4Dataset
try:
    import hdf5plugin  # Required for reading CharLS-compressed FCI L1c data
except ImportError:
    print("WARNING: hdf5plugin not installed. FCI L1c reading may fail. "
          "Install with: pip install hdf5plugin")
from satpy import Scene, find_files_and_readers


# =============================================================================
# Timestep configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"


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
            f"ERROR: product '{product_key}' has no minute filter in {TIMESTEP_CONFIG_PATH}",
            file=sys.stderr,
        )
        sys.exit(2)
    return set(flt), cfg["step_minutes"]


# =============================================================================
# Shared utilities
# =============================================================================

def get_datastore_token():
    """
    Authenticate with the EUMETSAT Data Store.
    
    Credentials should ideally be loaded from environment variables:
        EUMETSAT_CONSUMER_KEY, EUMETSAT_CONSUMER_SECRET
    Falls back to hardcoded values if env vars are not set.
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


def parse_date_range(start_str, end_str, fmt='%Y/%m/%d-%H%M'):
    """Parse start and end datetime strings."""
    try:
        start = datetime.datetime.strptime(start_str, fmt)
        end = datetime.datetime.strptime(end_str, fmt)
        return start, end
    except ValueError as e:
        print(f"Error parsing dates. Please use format '{fmt}'. Error: {e}")
        return None, None


def get_product_sensing_time(product):
    """Extract the sensing start time from a Data Store product."""
    return product.sensing_start


# =============================================================================
# Channel name mappings (MSG SEVIRI <-> MTG FCI)
# =============================================================================

SEVIRI_TO_FCI = {
    "VIS006": "vis_06",  "VIS008": "vis_08",
    "IR_016": "nir_16",  "IR_039": "ir_38",
    "IR_087": "ir_87",   "IR_097": "ir_97",
    "IR_108": "ir_105",  "IR_120": "ir_123",
    "IR_134": "ir_133",  "WV_062": "wv_63",
    "WV_073": "wv_73",
}

# Valid channel names per satellite (for validation)
VALID_MSG_CHANNELS = {
    "HRV", "VIS006", "VIS008",
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
    "WV_062", "WV_073"
}

VALID_MTG_CHANNELS = {
    "vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
    "nir_13", "nir_16", "nir_22",
    "ir_38", "wv_63", "wv_73", "ir_87", "ir_97", "ir_105", "ir_123", "ir_133"
}


def load_variables_from_file(filepath, satellite='mtg'):
    """
    Read channel names from a JSON file for the specified satellite.
    
    Args:
        filepath (str): Path to JSON file with "msg" and "mtg" keys.
        satellite (str): 'msg' or 'mtg' — determines which key to read.
        
    Returns:
        list: List of SatPy variable names.
        
    Example JSON (satellite_products.json):
        {
            "msg": [
                "VIS006", "VIS008",
                "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
                "WV_062", "WV_073"
            ],
            "mtg": [
                "vis_06", "vis_08",
                "nir_16",
                "ir_38", "ir_87", "ir_97", "ir_105", "ir_123", "ir_133",
                "wv_63", "wv_73"
            ]
        }
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
        raise TypeError(f"'{key}' must be a list, got {type(channels).__name__}")
    
    # Validate channel names
    valid_set = VALID_MSG_CHANNELS if key == 'msg' else VALID_MTG_CHANNELS
    validated = []
    invalid = []
    
    for ch in channels:
        if ch in valid_set:
            if ch not in validated:  # avoid duplicates
                validated.append(ch)
        else:
            invalid.append(ch)
    
    if invalid:
        print(f"WARNING: Unrecognized {satellite.upper()} channels: {invalid}")
        print(f"  Valid channels: {sorted(valid_set)}")
    
    print(f"Loaded {len(validated)} {satellite.upper()} channels from {filepath}:")
    for v in validated:
        print(f"  {v}")
    
    return validated

COLLECTIONS = {
    'msg_rss':   'EO:EUM:DAT:MSG:MSG15-RSS',
    'fci_fdhsi': 'EO:EUM:DAT:0662',
    'fci_hrfi':  'EO:EUM:DAT:0665',
}

# FCI chunk numbers that cover the Romania study area.
# Computed from geostationary projection for bounding box:
#   Lat: 41.23° - 49.25°N, Lon: 15.32° - 33.85°E
# Romania falls in chunks 35-37 (out of 40 total per channel).
# We add ±1 buffer for safety -> chunks 34-38.
# This reduces download volume by ~87% (5/40 chunks instead of 40/40).
ROMANIA_CHUNKS = set(range(34, 39))  # {34, 35, 36, 37, 38}

# Maximum number of parallel workers for per-variable processing.
# Each worker reads chunk files independently and saves one variable.
# 4 workers is a good balance — enough parallelism without overwhelming 
# disk I/O or memory (each worker holds one channel in RAM).
MAX_WORKERS = min(4, os.cpu_count() or 1)

# Channel resolution groups (module-level for use by worker function)
CHANNELS_1KM = {'vis_04', 'vis_05', 'vis_06', 'vis_08', 'vis_09', 
                 'nir_13', 'nir_16', 'nir_22'}
CHANNELS_2KM = {'ir_38', 'wv_63', 'wv_73', 'ir_87', 'ir_97', 
                 'ir_105', 'ir_123', 'ir_133'}


# =============================================================================
# MSG SEVIRI functions — DISABLED
# -----------------------------------------------------------------------------
# The COALITION-4 Romanian adaptation has converged on MTG FCI L1C as the only
# satellite source. The MSG branch is preserved as a docstring block for
# reference but is not exercised by the CLI. Re-enable by uncommenting if you
# need to revive the MSG ingestion path.
# =============================================================================

_MSG_DISABLED = r'''
def extract_date_time(filename):
    """Extract date and time from MSG SEVIRI NAT filename."""
    parts = filename.split('-')
    datetime_part = parts[5].split('.')[0]
    
    date = datetime_part[:8]
    time = datetime_part[8:12]
    
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    
    return f"nc4_{year}-{month}-{day}-Romania", time


def process_nat_file(nat_file, output_base_dir, variable, slot_dt=None):
    """
    Process a single variable from MSG SEVIRI NAT file and save to NetCDF.
    
    Args:
        nat_file (str): Path to the NAT file
        output_base_dir (str): Output directory for this variable
        variable (str): SatPy variable name
        slot_dt (datetime or None): If provided, use this for the output 
                filename/directory instead of the sensing time. This ensures 
                a product at 23:59 gets filed under the 00:00 slot of the 
                next day.
    """
    scn = Scene(filenames=[nat_file], reader='seviri_l1b_native')
    
    if slot_dt is not None:
        # Use slot time for naming (e.g., 23:59 product -> 00:00 slot on next day)
        dir_name = f"nc4_{slot_dt.strftime('%Y-%m-%d')}-Romania"
        time_str = slot_dt.strftime('%H%M')
    else:
        # Fallback: extract from NAT filename
        dir_name, time_str = extract_date_time(os.path.basename(nat_file))
    
    try:
        scn.load([variable])
        
        date_dir = f"{dir_name}_{variable}"
        full_output_dir = os.path.join(output_base_dir, date_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        nc_filename = f"{dir_name}_{time_str}_{variable}.nc"
        output_path = os.path.join(full_output_dir, nc_filename)
        
        scn.save_datasets(writer='cf', filename=output_path)
        print(f"Saved {variable} to {output_path}")
            
    except Exception as e:
        print(f"Error processing {variable} from {nat_file}: {str(e)}")
    
    if os.path.exists(nat_file):
        os.remove(nat_file)
        print(f"Removed processed file: {nat_file}")


def snap_to_15min_slot(dt):
    """
    Snap a datetime to the nearest 15-minute slot.
    
    MSG RSS products are offset by ~4 min from round times, e.g.:
      :04 -> :00 slot, :19 -> :15 slot, :34 -> :30 slot, :49 -> :45 slot
    
    Returns:
        tuple: (slot_datetime, offset_seconds) where offset is the distance
               from the product time to the slot center.
    """
    total_minutes = dt.hour * 60 + dt.minute + dt.second / 60.0
    slot_index = round(total_minutes / 15)
    slot_minutes = slot_index * 15
    
    slot_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + \
              datetime.timedelta(minutes=slot_minutes)
    offset = abs((dt - slot_dt).total_seconds())
    
    return slot_dt, offset


def fetch_and_process_sat(start_str, end_str, base_dir, variable):
    """
    Fetch and process MSG SEVIRI RSS data between two dates for a single variable.
    
    Only downloads the product closest to each 15-min slot (:00, :15, :30, :45).
    MSG RSS has a ~4 min scan offset, so actual timestamps are ~:04, :19, :34, :49.
    A tolerance of ±5 min is used to match products to slots.
    """
    TOLERANCE_SEC = 5 * 60  # ±5 minutes
    
    start, end = parse_date_range(start_str, end_str)
    if start is None:
        return

    token = get_datastore_token()
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(COLLECTIONS['msg_rss'])

    temp_dir = 'temp_output_msg'
    os.makedirs(temp_dir, exist_ok=True)

    # Extend search window 5 min before start to capture the :59 product
    # that snaps to the first :00 slot (RSS offset means :59 is closer 
    # to :00 than :04 is)
    search_start = start - datetime.timedelta(minutes=5)
    
    all_products = list(selected_collection.search(dtstart=search_start, dtend=end))
    all_products.sort(key=lambda p: p.sensing_start)
    
    # Group products by nearest 15-min slot, keep closest per slot.
    # Only keep slots that fall within the user's requested range.
    slots = {}  # slot_dt -> (product, offset_seconds)
    
    for product in all_products:
        sensing_time = get_product_sensing_time(product)
        slot_dt, offset = snap_to_15min_slot(sensing_time)
        
        if offset > TOLERANCE_SEC:
            continue
        
        # Only keep slots within the user's requested time range
        if slot_dt < start or slot_dt > end:
            continue
        
        # Keep the product closest to the slot center
        if slot_dt not in slots or offset < slots[slot_dt][1]:
            slots[slot_dt] = (product, offset)
    
    print(f"MSG: {len(all_products)} total products, "
          f"{len(slots)} matched to 15-min slots")
    
    processed_count = 0
    
    for slot_dt in sorted(slots.keys()):
        product, offset = slots[slot_dt]
        sensing_time = get_product_sensing_time(product)
        
        print(f"Processing MSG product at {sensing_time.strftime('%Y-%m-%dT%H:%M:%S')} "
              f"-> slot {slot_dt.strftime('%Y-%m-%dT%H:%M')} (offset {offset:.0f}s)")
        
        nat_files = [entry for entry in product.entries if '.nat' in entry]
        if not nat_files:
            continue
            
        nat_file = nat_files[0]
        
        temp_file_path = os.path.join(temp_dir, nat_file)
        with product.open(entry=nat_file) as fsrc, \
                open(temp_file_path, mode='wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
            print(f'  Downloaded file {fsrc.name}')
        
        process_nat_file(temp_file_path, base_dir, variable, slot_dt=slot_dt)
        processed_count += 1
    
    skipped = len(all_products) - processed_count
    print(f"\nMSG: processed {processed_count} products "
          f"(skipped {skipped} non-15-min products)")

    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
'''
# end MSG-disabled block


# =============================================================================
# MTG FCI functions — native 10-min cadence, no resampling
# =============================================================================

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
    
    If chunk_filter is provided, only downloads chunks whose number falls 
    within the specified set — dramatically reducing download time and storage.
    
    FCI L1c filenames end with a pattern like '_NNNN_TTTT.nc' where NNNN is 
    the chunk number and TTTT is the total chunk count. The chunk number is 
    extracted and checked against chunk_filter.
    
    Args:
        product: eumdac product object
        temp_dir (str): Temporary directory to store downloaded chunks
        chunk_filter (set or None): Set of allowed chunk numbers (1-indexed).
                                     If None, downloads all chunks.
        
    Returns:
        list: Paths to all downloaded chunk files
    """
    nc_entries = [entry for entry in product.entries if entry.endswith('.nc')]
    
    if not nc_entries:
        print(f"  No NetCDF entries found in product {product}")
        return []
    
    # Apply chunk filter if specified
    if chunk_filter is not None:
        filtered_entries = []
        skipped = 0
        for entry in nc_entries:
            # Always keep trailer chunks (metadata SatPy may need)
            if 'TRAIL' in entry:
                filtered_entries.append(entry)
                continue
            
            chunk_num = parse_chunk_number(entry)
            if chunk_num is not None and chunk_num in chunk_filter:
                filtered_entries.append(entry)
            elif chunk_num is None:
                # If we can't parse the chunk number, include it (safety)
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


def parse_chunk_number(filename):
    """
    Extract the chunk/segment number from an FCI L1c filename.
    
    FCI filenames end with patterns like:
        ..._C_0035_0040.nc  (chunk 35 of 40)
        ..._N__C_0035_0040.nc
        
    Args:
        filename (str): FCI chunk filename (basename)
        
    Returns:
        int: Chunk number (1-indexed), or None if parsing fails
    """
    # Match _OOOO_CCCC.nc at the end: OOOO=orbit number, CCCC=chunk number
    match = re.search(r'_(\d{4})_(\d{4})\.nc$', filename)
    if match:
        chunk_num = int(match.group(2))  # group(2) = chunk, group(1) = orbit
        return chunk_num
    
    # Fallback: try to find any 4-digit chunk pattern
    match = re.search(r'CHK[_-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def fci_scanning_angles_to_latlon(x_rad, y_rad, sub_lon_deg=0.0):
    """
    Convert FCI geostationary scanning angles to latitude/longitude.
    
    Implements the inverse geostationary projection for MTG FCI at 0°E.
    x and y are 1D arrays of scanning angles in radians from the FCI 
    measured group.
    
    Args:
        x_rad (numpy.ndarray): 1D column scanning angles (radians)
        y_rad (numpy.ndarray): 1D row scanning angles (radians)
        sub_lon_deg (float): Sub-satellite longitude in degrees (0.0 for MTG)
        
    Returns:
        tuple: (lat_2d, lon_2d) both as 2D numpy arrays in degrees
    """
    R_eq = 6378.137     # equatorial radius km
    R_pol = 6356.7523   # polar radius km
    H = 42164.0         # orbital radius from Earth center km
    
    sub_lon = np.radians(sub_lon_deg)
    
    # Create 2D meshgrid from 1D scanning angles
    xx, yy = np.meshgrid(x_rad, y_rad)
    
    cos_x = np.cos(xx)
    cos_y = np.cos(yy)
    sin_x = np.sin(xx)
    sin_y = np.sin(yy)
    
    # Ray-ellipsoid intersection: a·sn² + b·sn + c = 0
    a = sin_x**2 + cos_x**2 * (cos_y**2 + (R_eq / R_pol)**2 * sin_y**2)
    b = -2 * H * cos_x * cos_y
    c = H**2 - R_eq**2
    
    discriminant = b**2 - 4 * a * c
    
    # Points outside Earth disk have negative discriminant
    valid = discriminant >= 0
    
    # Initialize output with NaN
    lat = np.full(xx.shape, np.nan, dtype=np.float64)
    lon = np.full(xx.shape, np.nan, dtype=np.float64)
    
    if not np.any(valid):
        return lat.astype(np.float32), lon.astype(np.float32)
    
    # Work only with valid pixels (all flat 1D arrays of same length)
    a_v = a[valid]
    b_v = b[valid]
    disc_v = discriminant[valid]
    cos_x_v = cos_x[valid]
    cos_y_v = cos_y[valid]
    sin_x_v = sin_x[valid]
    sin_y_v = sin_y[valid]
    
    # Distance from satellite to point on Earth surface (near root)
    sn = (-b_v - np.sqrt(disc_v)) / (2 * a_v)
    
    # Cartesian coordinates of point relative to satellite
    # FCI convention: y positive = north, x positive = west (in scanning frame)
    s1 = H - sn * cos_x_v * cos_y_v
    s2 = -sn * sin_x_v * cos_y_v
    s3 = sn * sin_y_v
    
    sxy = np.sqrt(s1**2 + s2**2)
    
    # Geographic coordinates
    lon[valid] = np.degrees(np.arctan2(s2, s1) + sub_lon)
    lat[valid] = np.degrees(np.arctan((R_eq / R_pol)**2 * s3 / sxy))
    
    return lat.astype(np.float32), lon.astype(np.float32)


def _process_single_fci_variable(variable, body_files, dir_name, time_str, 
                                  base_dir):
    """
    Worker function: process one FCI variable from chunk files.
    
    Runs in a separate process. Reads chunk data, builds a NetCDF with 
    radiance data only (no lat/lon — those are saved once separately),
    and saves to the output directory.
    
    Must be defined at module level for ProcessPoolExecutor pickling.
    
    Args:
        variable (str): FCI channel name
        body_files (list): Paths to body chunk files
        dir_name (str): Date directory name (e.g., 'nc4_2025-05-15-Romania')
        time_str (str): Time string (e.g., '1000')
        base_dir (str): Base output directory
        
    Returns:
        tuple: (variable, success_bool, message_string)
    """
    try:
        import hdf5plugin  # needed in each worker for CharLS decompression
    except ImportError:
        pass
    
    try:
        # Determine resolution
        if variable in CHANNELS_1KM:
            res_key = '1km'
        elif variable in CHANNELS_2KM:
            res_key = '2km'
        else:
            res_key = '1km'
        
        # Read and concatenate chunks for this variable
        chunk_arrays = []
        
        for chunk_file in body_files:
            try:
                ds = NC4Dataset(chunk_file, 'r')
                measured = ds['data'][variable]['measured']
                data = measured.variables['effective_radiance'][:]
                chunk_arrays.append(np.array(data, dtype=np.float32))
                ds.close()
            except Exception:
                try:
                    ds.close()
                except:
                    pass
        
        if not chunk_arrays:
            return (variable, False, "no data found in chunks")
        
        combined = np.concatenate(chunk_arrays, axis=0)
        del chunk_arrays
        
        # Build output path
        var_dir = os.path.join(base_dir, variable)
        date_dir = f"{dir_name}_{variable}"
        full_output_dir = os.path.join(var_dir, date_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        nc_filename = f"{dir_name}_{time_str}_{variable}.nc"
        output_path = os.path.join(full_output_dir, nc_filename)
        
        # Build dataset (radiance only, no lat/lon)
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
    Process ALL variables from downloaded FCI chunk files in parallel.
    
    Step 1 (serial, first product only): Reads x/y scanning angles from chunk 
    files, computes lat/lon grids, saves to persistent .npy files in 
    base_dir/coordinates/. Subsequent products skip this step.
    
    Step 2 (parallel): Dispatches up to MAX_WORKERS processes, each reading
    chunk data for one variable and saving a NetCDF file (radiance only).
    
    Args:
        product: eumdac product object
        downloaded_files (list): Paths to downloaded chunk files
        temp_dir (str): Directory containing the chunk files
        base_dir (str/Path): Base output directory
        variables (list): List of FCI variable names to process
    """
    if not downloaded_files:
        print(f"  No files to process")
        return
    
    dir_name, time_str = extract_date_time_fci(product)
    
    # Separate body chunks from trailer
    body_files = sorted([f for f in downloaded_files if 'TRAIL' not in f])
    
    if not body_files:
        print(f"  No body chunk files found")
        return
    
    print(f"  Processing {len(body_files)} body chunks for {len(variables)} variables "
          f"(max {MAX_WORKERS} parallel workers)")
    
    # ---- Step 1: Compute coordinates once and save as .npy ----
    coord_dir = os.path.join(base_dir, 'coordinates')
    os.makedirs(coord_dir, exist_ok=True)
    
    for res_key, ref_channel in [('1km', 'vis_06'), ('2km', 'ir_105')]:
        # Only compute if we have variables at this resolution
        vars_at_res = [v for v in variables 
                       if (v in CHANNELS_1KM and res_key == '1km') or 
                          (v in CHANNELS_2KM and res_key == '2km')]
        if not vars_at_res:
            continue
        
        lat_path = os.path.join(coord_dir, f'lat_{res_key}.npy')
        lon_path = os.path.join(coord_dir, f'lon_{res_key}.npy')
        
        # Skip if already computed from a previous product
        if os.path.exists(lat_path) and os.path.exists(lon_path):
            lat_2d = np.load(lat_path)
            print(f"    {res_key} coordinates already exist: {lat_2d.shape} "
                  f"(lat: {lat_path})")
            del lat_2d
            continue
        
        try:
            x_arrays = []
            y_arrays = []
            
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
            
            print(f"    {res_key} grid: {lat_2d.shape}, "
                  f"lat [{np.nanmin(lat_2d):.2f}, {np.nanmax(lat_2d):.2f}], "
                  f"lon [{np.nanmin(lon_2d):.2f}, {np.nanmax(lon_2d):.2f}]")
            print(f"    Saved: {lat_path}")
            print(f"    Saved: {lon_path}")
            
            del lat_2d, lon_2d
            
        except Exception as e:
            print(f"    WARNING: Could not compute {res_key} coordinates: {e}")
    
    gc.collect()
    
    # ---- Step 2 (parallel): Process each variable ----
    n_workers = min(MAX_WORKERS, len(variables))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for variable in variables:
            future = executor.submit(
                _process_single_fci_variable,
                variable, body_files, dir_name, time_str, base_dir
            )
            futures[future] = variable
        
        # Collect results as they complete
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
    Fetch and process MTG FCI data for ALL requested variables.

    Downloads and processes products at native 10-min cadence, optionally
    filtered to only keep specific minutes within each hour. The filter is
    typically derived from the project-level timestep_config.json by the
    CLI; when invoked programmatically with minute_filter=None, all native
    timesteps are kept.

    When chunk_filter is provided (default: ROMANIA_CHUNKS), only downloads
    the spatial chunks covering the study area — ~87% bandwidth reduction.

    Args:
        start_str (str): Start datetime in format 'yyyy/mm/dd-hhmm'
        end_str (str): End datetime in format 'yyyy/mm/dd-hhmm'
        base_dir (str/Path): Base directory for output
        variables (list): List of FCI variable names
        collection (str): Data Store collection key
        chunk_filter (set or None): Set of chunk numbers to download.
                                     Default: ROMANIA_CHUNKS.
                                     Pass None for full disk.
        minute_filter (set or None): Set of allowed minutes (as int, e.g. {0,10,30,40}).
                                      None = keep all native timesteps.
                                      Pass None to keep all timesteps.
    """
    # minute_filter=None means "keep all native timesteps" (no filtering).
    # The CLI populates it from timestep_config.json when not overridden.

    start, end = parse_date_range(start_str, end_str)
    if start is None:
        return

    token = get_datastore_token()
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(COLLECTIONS[collection])

    all_products = list(selected_collection.search(dtstart=start, dtend=end))

    # Sort by sensing time to ensure chronological processing
    all_products.sort(key=lambda p: p.sensing_start)

    # Filter products by minute
    if minute_filter is not None:
        filtered_products = []
        skipped = 0
        for p in all_products:
            sensing_minute = get_product_sensing_time(p).minute
            if sensing_minute in minute_filter:
                filtered_products.append(p)
            else:
                skipped += 1
        print(f"MTG: {len(all_products)} products found at native 10-min cadence")
        print(f"Timestep filter: keeping minutes {sorted(minute_filter)} "
              f"-> {len(filtered_products)} products (skipped {skipped})")
        all_products = filtered_products
    else:
        print(f"MTG: {len(all_products)} products found at native 10-min cadence")
        print(f"Timestep filter: none (keeping all)")

    filter_desc = f"chunks {sorted(chunk_filter)}" if chunk_filter else "full disk"
    print(f"Chunk filter: {filter_desc}")

    processed_count = 0

    for product in all_products:
        sensing_time = get_product_sensing_time(product)

        print(
            f"\nProcessing MTG FCI product at "
            f"{sensing_time.strftime('%Y-%m-%dT%H:%M:%S')} "
            f"[{processed_count + 1}/{len(all_products)}] "
            f"({len(variables)} variables)"
        )

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
    print(f"Finished processing {len(variables)} variables")

def discover_chunk_numbers(start_str, end_str, collection='fci_fdhsi', max_products=1):
    """
    Debug utility: print all entry filenames and parsed chunk numbers 
    from the first product(s) in the date range.
    
    Use this to verify that parse_chunk_number correctly extracts chunk IDs
    from your specific FCI filenames before running the full pipeline.
    
    Args:
        start_str (str): Start datetime
        end_str (str): End datetime
        collection (str): Data Store collection key
        max_products (int): How many products to inspect
    """
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
            chunk_num = parse_chunk_number(entry)
            chunk_counts[chunk_num] = chunk_counts.get(chunk_num, 0) + 1
            # Print first few entries in detail
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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download and process MTG FCI L1C satellite data'
    )
    parser.add_argument(
        '--satellite', '-sat',
        type=str,
        default='mtg',
        choices=['mtg'],
        help='Satellite to process. Only mtg (FCI FDHSI) is currently supported; '
             'the MSG branch is disabled in this build.'
    )
    parser.add_argument(
        '--start', '-s',
        type=str,
        required=True,
        help='Start datetime in format yyyy/mm/dd-hhmm'
    )
    parser.add_argument(
        '--end', '-e',
        type=str,
        required=True,
        help='End datetime in format yyyy/mm/dd-hhmm'
    )
    parser.add_argument(
        '--products_file', '-pf',
        type=str,
        default='satellite_products.json',
        help='Path to JSON file with channel names (default: satellite_products.json)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: ./MTG in the current working directory)'
    )
    parser.add_argument(
        '--full_disk',
        action='store_true',
        help='MTG only: download full disk instead of Romania chunks'
    )
    parser.add_argument(
        '--discover_chunks',
        action='store_true',
        help='MTG only: inspect chunk structure of first product and exit'
    )
    parser.add_argument(
        '--timesteps',
        type=str,
        nargs='+',
        default=None,
        help="Override the minute filter (e.g. 00 10 30 40). "
             "Default: read from our_data/timestep_config.json (set via "
             "validate_timestep.py). Use 'all' to keep every native timestep."
    )

    args = parser.parse_args()
    
    # Resolve output directory
    base_dir = os.getcwd()
    if args.output_dir:
        data_dir = args.output_dir
    else:
        data_dir = os.path.join(base_dir, args.satellite.upper())
    os.makedirs(data_dir, exist_ok=True)
    
    # Resolve products file path
    products_file = args.products_file
    if not os.path.isabs(products_file):
        products_file = os.path.join(base_dir, products_file)
    
    print(f"Satellite: {args.satellite.upper()}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Output directory: {data_dir}")
    print(f"Products file: {products_file}")
    
    if args.satellite == 'mtg':
        print(f"Romania chunk filter: {sorted(ROMANIA_CHUNKS)}")
        print(f"Max parallel workers: {MAX_WORKERS}")

        # Discovery mode
        if args.discover_chunks:
            discover_chunk_numbers(args.start, args.end)
        else:
            # Load variables from file
            mtg_variables = load_variables_from_file(products_file, satellite='mtg')

            if not mtg_variables:
                print("ERROR: No valid MTG variables found in products file.")
            else:
                chunk_filter = None if args.full_disk else ROMANIA_CHUNKS
                # Build minute filter — explicit CLI override beats config.
                if args.timesteps is not None and args.timesteps != ['all']:
                    mtg_minute_filter = {int(m) for m in args.timesteps}
                    print(f"Minute filter (CLI override): {sorted(mtg_minute_filter)}")
                elif args.timesteps == ['all']:
                    mtg_minute_filter = None
                    print("Minute filter: none (keeping every native timestep)")
                else:
                    mtg_minute_filter, mtg_step = load_timestep_filter("mtg")
                    print(f"Minute filter (timestep_config.json, step={mtg_step} min): "
                          f"{sorted(mtg_minute_filter)}")

                fetch_and_process_mtg_all_variables(
                    args.start, args.end, data_dir, mtg_variables,
                    chunk_filter=chunk_filter,
                    minute_filter=mtg_minute_filter
                )

    # MSG branch is disabled — see _MSG_DISABLED block above.
    # elif args.satellite == 'msg':
    #     msg_variables = load_variables_from_file(products_file, satellite='msg')
    #     if not msg_variables:
    #         print("ERROR: No valid MSG variables found in products file.")
    #     else:
    #         for data_var in msg_variables:
    #             var_dir = os.path.join(data_dir, data_var)
    #             os.makedirs(var_dir, exist_ok=True)
    #             fetch_and_process_sat(args.start, args.end, var_dir, data_var)
                