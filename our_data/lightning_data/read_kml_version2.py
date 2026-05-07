"""
Process LINET lightning KML data into 15-minute NetCDF maps for COALITION-4.

Adapted from read_kml_version2.py with the following changes:
    - 15-minute time steps (was 5-minute for density, 10-minute for occurrence)
    - Fixed UTC time grid: 00:15, 00:30, ..., 23:30, 23:45 (96 steps)
      Each step accumulates lightning in the preceding 15-minute window.
    - Time steps from 23:56 onwards are excluded (last valid step is 23:45)
    - Date extracted from the KML filename (dd_mm_yyyy.kml or dd.mm.yyyy.kml)
    - Batch processing of all dates in the arranged directory structure

Input structure (from arrange_lightning_data.py):
    {data_root}/kml_data/yyyy-mm-dd/yyyy-mm-dd.kml

Output structure (COALITION-4 convention):
    {output_root}/density/nc4_yyyy-mm-dd-Romania_density/lightning_density_yyyymmdd_HHMM.nc
    {output_root}/current/nc4_yyyy-mm-dd-Romania_current/lightning_current_yyyymmdd_HHMM.nc
    {output_root}/occurrence/nc4_yyyy-mm-dd-Romania_occurrence/lightning_occurrence_yyyymmdd_HHMM.nc

Usage:
    # Process all dates
    python process_lightning_kml.py

    # Process a single date
    python process_lightning_kml.py --date 2025-05-15

    # Custom paths
    python process_lightning_kml.py -d F:/nowcasting/.../lightning_data -o ./output/lightning
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import time
import re
import netCDF4 as nc
from pathlib import Path
import argparse
import os


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = r"F:\nowcasting\coalition4-rcnn\our_data\lightning_data"
DEFAULT_OUTPUT_ROOT = r"F:\nowcasting\coalition4-rcnn\our_data\lightning_data"

# Fixed 15-minute grid: 96 steps from 00:15 to 23:45
TIME_STEP_MINUTES = 15

# Number of time steps per day (00:15 through 23:45)
STEPS_PER_DAY = 96


# =============================================================================
# KML reading
# =============================================================================

def read_kml_extract_coordinates(kml_file):
    """
    Read a LINET KML file and extract lightning stroke data.

    Parses the Description field of each Placemark to extract:
        - Position (lon, lat)
        - Current (kA)
        - Timestamp (UTC)

    Args:
        kml_file (str): Path to the KML file

    Returns:
        pd.DataFrame: Columns [lat, lon, current, timestamp]
    """
    gdf = gpd.read_file(kml_file)
    print(f"  Raw placemarks: {len(gdf)}")

    latitudes = []
    longitudes = []
    currents = []
    timestamps = []

    position_pattern = r"Position: (\d+\.\d+), (\d+\.\d+)"
    current_pattern = r"Current: ([-+]?\d*\.?\d+) kA"
    time_pattern = r"Time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{3})"

    for desc in gdf['Description']:
        pos_match = re.search(position_pattern, desc)
        current_match = re.search(current_pattern, desc)
        time_match = re.search(time_pattern, desc)

        if pos_match and current_match and time_match:
            lon = float(pos_match.group(1))
            lat = float(pos_match.group(2))
            current = float(current_match.group(1))
            datetime_str = time_match.group(1)
            timestamp = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

            latitudes.append(lat)
            longitudes.append(lon)
            currents.append(current)
            timestamps.append(timestamp)

    lightning_df = pd.DataFrame({
        'lat': latitudes,
        'lon': longitudes,
        'current': currents,
        'timestamp': timestamps
    })

    return lightning_df


# =============================================================================
# Grid accumulation
# =============================================================================

def grid_accumulate(i, j, grid=None, weights=None, weighted_grid=None):
    """Accumulate values on a grid (occurrence or weighted)."""
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)

    if grid is not None:
        valid_mask = (i >= 0) & (j >= 0) & (i < grid.shape[0]) & (j < grid.shape[1])
        grid[i[valid_mask], j[valid_mask]] = 1

    elif weights is not None and weighted_grid is not None:
        valid_mask = (i >= 0) & (j >= 0) & (i < weighted_grid.shape[0]) & (j < weighted_grid.shape[1])
        np.add.at(weighted_grid, (i[valid_mask], j[valid_mask]), weights[valid_mask])


def lightning_for_time(lightning_df, time, interval=timedelta(minutes=15)):
    """Extract lightning data for the interval [time - interval, time)."""
    start_time = time - interval
    end_time = time
    mask = (lightning_df['timestamp'] >= start_time) & (lightning_df['timestamp'] < end_time)
    return lightning_df[mask]


# =============================================================================
# Map generation — 15-minute time steps
# =============================================================================

def generate_fixed_timesteps(date_str):
    """
    Generate the fixed 15-minute UTC time grid for one day.

    Returns 96 timestamps: 00:00, 00:15, 00:30, ..., 23:30, 23:45.
    Each timestamp represents the END of its 15-minute accumulation window.
    For example, 00:00 accumulates [23:45 prev day, 00:00) — typically empty
    for single-day files. The last step 23:45 accumulates [23:30, 23:45).
    Anything from 23:46 onwards is discarded.

    Args:
        date_str (str): Date in 'YYYY-MM-DD' format

    Returns:
        list[datetime]: 96 timestamps
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    step = timedelta(minutes=TIME_STEP_MINUTES)

    timestamps = []
    current = base  # first step is 00:00
    for _ in range(STEPS_PER_DAY):
        timestamps.append(current)
        current += step

    return timestamps


def create_density_maps(lightning_df, grid_projection, timestamps):
    """
    Generate lightning density and current-weighted density maps at
    15-minute intervals on a fixed time grid.

    Args:
        lightning_df (pd.DataFrame): Lightning data
        grid_projection: GridProjection object
        timestamps (list[datetime]): Fixed 15-minute time grid

    Returns:
        tuple: (normalized_density_maps, normalized_weighted_maps, timestamps)
    """
    interval = timedelta(minutes=TIME_STEP_MINUTES)
    density_maps = []
    weighted_maps = []

    for current_time in timestamps:
        lightning = lightning_for_time(lightning_df, current_time, interval=interval)

        density_map = np.zeros(
            (grid_projection.area.height, grid_projection.area.width), dtype=float
        )
        weighted_map = np.zeros_like(density_map)

        if len(lightning) > 0:
            lon = lightning["lon"].values
            lat = lightning["lat"].values
            current = abs(lightning["current"].values)

            (i, j) = grid_projection(lon, lat)

            grid_accumulate(i, j, weights=np.ones(len(i)), weighted_grid=density_map)
            grid_accumulate(i, j, weights=current, weighted_grid=weighted_map)

        density_maps.append(density_map)
        weighted_maps.append(weighted_map)

    # Normalize
    density_array = np.array(density_maps)
    weighted_array = np.array(weighted_maps)

    density_mean = (
        np.mean(density_array[density_array > 0])
        if np.any(density_array > 0) else 1
    )
    weighted_mean = (
        np.mean(weighted_array[weighted_array > 0])
        if np.any(weighted_array > 0) else 1
    )

    normalized_density = [
        d / density_mean if density_mean > 0 else d for d in density_maps
    ]
    normalized_weighted = [
        w / weighted_mean if weighted_mean > 0 else w for w in weighted_maps
    ]

    return normalized_density, normalized_weighted, timestamps


def generate_lightning_occurrence_maps(lightning_df, grid_projection, timestamps):
    """
    Generate binary lightning occurrence maps with 4-neighbor expansion
    at 15-minute intervals.

    Args:
        lightning_df (pd.DataFrame): Lightning data
        grid_projection: GridProjection object
        timestamps (list[datetime]): Fixed 15-minute time grid

    Returns:
        tuple: (occurrence_maps, timestamps)
    """
    interval = timedelta(minutes=TIME_STEP_MINUTES)
    occurrence_maps = []

    for current_time in timestamps:
        lightning = lightning_for_time(lightning_df, current_time, interval=interval)

        lightning_map = np.zeros(
            (grid_projection.area.height, grid_projection.area.width), dtype=int
        )

        if len(lightning) > 0:
            lon = lightning["lon"].values
            lat = lightning["lat"].values
            (i, j) = grid_projection(lon, lat)
            grid_accumulate(i, j, grid=lightning_map)

        # 4-neighbor expansion
        neighbor_map = np.zeros_like(lightning_map)
        strike_locs = np.argwhere(lightning_map == 1)
        for (r, c) in strike_locs:
            neighbor_map[r, c] = 1
            if r > 0:
                neighbor_map[r - 1, c] = 1
            if r < lightning_map.shape[0] - 1:
                neighbor_map[r + 1, c] = 1
            if c > 0:
                neighbor_map[r, c - 1] = 1
            if c < lightning_map.shape[1] - 1:
                neighbor_map[r, c + 1] = 1

        occurrence_maps.append(neighbor_map)

    return occurrence_maps, timestamps


# =============================================================================
# NetCDF output
# =============================================================================

def write_single_nc_file(output_path, data_slice, lat_data, lon_data,
                         timestamp, var_name, var_units, var_long_name,
                         time_interval):
    """Write a single 15-minute lightning map to NetCDF."""
    with nc.Dataset(output_path, 'w', format='NETCDF4') as dst:
        dst.createDimension('lat', lat_data.shape[0])
        dst.createDimension('lon', lat_data.shape[1])
        dst.createDimension('time', 1)

        times = dst.createVariable('time', 'i4', ('time',))
        lats = dst.createVariable('latitude', 'f4', ('lat', 'lon'))
        lons = dst.createVariable('longitude', 'f4', ('lat', 'lon'))
        variable = dst.createVariable(
            var_name, 'f4' if var_name != 'datamap' else 'i1',
            ('time', 'lat', 'lon')
        )

        times.units = 'seconds since 1970-01-01 00:00:00'
        times.calendar = 'standard'
        lats.units = 'degrees_north'
        lons.units = 'degrees_east'
        variable.units = var_units
        variable.long_name = var_long_name

        dst.description = f'{var_long_name} map'
        dst.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        dst.source = f'Lightning data processed with {time_interval}-minute intervals'

        times[:] = [int(timestamp.timestamp())]
        lats[:] = lat_data
        lons[:] = lon_data
        variable[0, :, :] = data_slice

    return output_path


def save_lightning_maps_parallel(density_maps, weighted_maps, occurrence_maps,
                                timestamps, grid_projection, output_root,
                                date_str, batch_size=10):
    """
    Save all lightning maps to individual NetCDF files using parallelization.

    Output structure (COALITION-4 convention):
        {output_root}/density/nc4_{date}-Romania_density/lightning_density_YYYYMMDD_HHMM.nc
        {output_root}/current/nc4_{date}-Romania_current/lightning_current_YYYYMMDD_HHMM.nc
        {output_root}/occurrence/nc4_{date}-Romania_occurrence/lightning_occurrence_YYYYMMDD_HHMM.nc
    """
    output_root = Path(output_root)

    y_coords, x_coords = np.mgrid[
        :grid_projection.area.height, :grid_projection.area.width
    ]
    lons, lats = grid_projection.inverse(y_coords, x_coords)

    # COALITION-4 structure: {product}/nc4_{date}-Romania_{product}/
    products = {
        'density':    ('density',    'lightning_density',    'normalized_density',
                       'Lightning Density'),
        'current':    ('current',    'lightning_current',    'normalized_current_weighted_density',
                       'Lightning Current-Weighted Density'),
        'occurrence': ('occurrence', 'lightning_occurrence', 'binary',
                       'Lightning Occurrence within 15-minute window'),
    }

    folders = {}
    for prod_key, (prod_name, _, _, _) in products.items():
        day_dir = f"nc4_{date_str}-Romania_{prod_name}"
        folder = output_root / prod_name / day_dir
        folder.mkdir(parents=True, exist_ok=True)
        folders[prod_key] = folder

    all_tasks = []
    # Windows WaitForMultipleObjects limit is 63 handles; cap well below that
    num_cores = min(multiprocessing.cpu_count(), 16)
    print(f"  Using {num_cores} CPU cores for parallel writing")
    start_time = time.time()

    for i, timestamp in enumerate(timestamps):
        time_str = timestamp.strftime("%Y%m%d_%H%M")

        for prod_key, (prod_name, file_prefix, var_units, var_long_name) in products.items():
            if prod_key == 'density':
                data = density_maps[i]
            elif prod_key == 'current':
                data = weighted_maps[i]
            else:
                data = occurrence_maps[i]

            all_tasks.append((
                str(folders[prod_key] / f"{file_prefix}_{time_str}.nc"),
                data, lats, lons, timestamp,
                'datamap', var_units, var_long_name, '15'
            ))

    total_tasks = len(all_tasks)
    processed = 0

    # Create pool once — recreating per batch leaks handles on Windows
    with multiprocessing.Pool(processes=num_cores) as pool:
        for i in range(0, total_tasks, batch_size):
            batch_tasks = all_tasks[i:i + batch_size]
            pool.starmap(write_single_nc_file, batch_tasks)

            processed += len(batch_tasks)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_tasks - processed) / rate if rate > 0 else 0

            print(f"  Progress: {processed}/{total_tasks} "
                  f"({processed / total_tasks * 100:.1f}%) - "
                  f"Rate: {rate:.2f} files/sec - ETA: {eta:.1f} sec")

    elapsed = time.time() - start_time
    print(f"  Saved {len(timestamps)} × 3 = {total_tasks} files "
          f"in {elapsed:.2f} sec")


# =============================================================================
# Single-date processing
# =============================================================================

def process_single_date(kml_path, date_str, output_root):
    """
    Process one KML file into 15-minute lightning NetCDF maps.

    Args:
        kml_path (str): Path to the KML file
        date_str (str): Date in 'YYYY-MM-DD' format
        output_root (str): Root output directory (product dirs are created inside)
    """
    from c4dl.projection import GridProjection, romania_grid_area

    print(f"\n{'=' * 70}")
    print(f"Processing: {date_str}")
    print(f"{'=' * 70}")
    print(f"  KML file: {kml_path}")

    # Read lightning data
    lightning_df = read_kml_extract_coordinates(kml_path)
    print(f"  Extracted {len(lightning_df)} lightning records")

    if len(lightning_df) == 0:
        print(f"  WARNING: No records found, skipping")
        return

    print(f"  Time range: {lightning_df['timestamp'].min()} to "
          f"{lightning_df['timestamp'].max()}")

    # Filter to the target date only (discard any spillover)
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    lightning_df = lightning_df[
        lightning_df['timestamp'].dt.date == target_date
    ].copy()
    print(f"  After date filter: {len(lightning_df)} records")

    if len(lightning_df) == 0:
        print(f"  WARNING: No records for {date_str}, skipping")
        return

    # Create grid projection
    grid_projection = GridProjection(romania_grid_area)

    # Generate fixed 15-minute time grid
    timestamps = generate_fixed_timesteps(date_str)
    print(f"  Time steps: {len(timestamps)} "
          f"({timestamps[0].strftime('%H:%M')} to "
          f"{timestamps[-1].strftime('%H:%M')})")

    # Generate density + current maps
    print(f"  Generating density and current-weighted maps...")
    density_maps, weighted_maps, _ = create_density_maps(
        lightning_df, grid_projection, timestamps
    )

    # Generate occurrence maps
    print(f"  Generating occurrence maps...")
    occurrence_maps, _ = generate_lightning_occurrence_maps(
        lightning_df, grid_projection, timestamps
    )

    # Save
    print(f"  Saving to: {output_root}")
    save_lightning_maps_parallel(
        density_maps, weighted_maps, occurrence_maps,
        timestamps, grid_projection, output_root, date_str
    )

    # Summary stats
    non_empty_density = sum(1 for m in density_maps if np.any(np.array(m) > 0))
    non_empty_occ = sum(1 for m in occurrence_maps if np.any(np.array(m) > 0))
    print(f"  Non-empty density steps: {non_empty_density}/{len(timestamps)}")
    print(f"  Non-empty occurrence steps: {non_empty_occ}/{len(timestamps)}")


# =============================================================================
# Batch processing
# =============================================================================

def discover_dates(data_root):
    """
    Discover all available dates in the arranged lightning directory.

    Scans {data_root}/kml_data/ for subdirectories matching YYYY-MM-DD
    that contain a matching KML file.

    Returns:
        list[tuple]: Sorted list of (date_str, kml_path) pairs
    """
    kml_dir = os.path.join(data_root, "kml_data")
    if not os.path.isdir(kml_dir):
        print(f"KML data directory not found: {kml_dir}")
        return []

    results = []
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')

    for entry in sorted(os.listdir(kml_dir)):
        if not date_pattern.match(entry):
            continue

        entry_path = os.path.join(kml_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        kml_path = os.path.join(entry_path, f"{entry}.kml")
        if os.path.isfile(kml_path):
            results.append((entry, kml_path))

    return results


def run_all(data_root, output_root, date_filter=None):
    """
    Process all (or one) dates.

    Args:
        data_root (str): Root lightning data directory
        output_root (str): Root output directory
        date_filter (str or None): Process only this date (YYYY-MM-DD)
    """
    print("=" * 70)
    print("Lightning KML -> 15-minute NetCDF Maps (COALITION-4)")
    print("=" * 70)
    print(f"Data root   : {data_root}")
    print(f"Output root : {output_root}")
    print(f"Time step   : {TIME_STEP_MINUTES} minutes")
    print(f"Steps/day   : {STEPS_PER_DAY}")

    if date_filter:
        kml_path = os.path.join(data_root, "kml_data", date_filter,
                                f"{date_filter}.kml")
        if not os.path.isfile(kml_path):
            print(f"ERROR: KML file not found: {kml_path}")
            return
        dates = [(date_filter, kml_path)]
        print(f"Processing single date: {date_filter}")
    else:
        dates = discover_dates(data_root)
        print(f"Discovered {len(dates)} dates")

    if not dates:
        print("\nNo dates found. Run arrange_lightning_data.py first.")
        return

    for date_str, kml_path in dates:
        process_single_date(kml_path, date_str, output_root)

    print(f"\n{'=' * 70}")
    print(f"Complete: processed {len(dates)} date(s)")
    print(f"{'=' * 70}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process LINET lightning KML data into 15-minute "
                    "NetCDF maps for COALITION-4."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Root lightning data directory (default: {DEFAULT_DATA_ROOT})"
    )
    parser.add_argument(
        "--output_root", "-o", type=str, default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT})"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Process a single date (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    run_all(
        data_root=args.data_root,
        output_root=args.output_root,
        date_filter=args.date,
    )
    