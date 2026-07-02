"""
Process LINET lightning KML data into per-cadence `.npy` maps for COALITION-4.

The bin width is taken from `our_data/timestep_config.json`:
    - Prefer `products.lightning.cadence_minutes` (the LINET native cadence,
      typically 10 min) so the maps are generated at source resolution.
    - Fall back to the global `step_minutes` only when lightning is declared
      continuous / event-based (cadence is null).

For a 10-minute cadence the day is binned into 144 fixed UTC slots
(00:10, 00:20, ..., 23:50, 00:00 of the next day). Each slot accumulates
the LINET strokes in its preceding `cadence` window. Date is extracted
from the KML filename (dd_mm_yyyy.kml or dd.mm.yyyy.kml) and the whole
arranged directory is walked unless `--date` is given.

Input structure (from arrange_lightning_data.py):
    {data_root}/kml_data/yyyy-mm-dd/yyyy-mm-dd.kml

Output structure (COALITION-4 convention):
    {output_root}/density/nc4_yyyy-mm-dd-Romania_density/lightning_density_yyyymmdd_HHMM.npy
    {output_root}/current/nc4_yyyy-mm-dd-Romania_current/lightning_current_yyyymmdd_HHMM.npy
    {output_root}/occurrence/nc4_yyyy-mm-dd-Romania_occurrence/lightning_occurrence_yyyymmdd_HHMM.npy

Output is `.npy` (just the binned grid, no lat/lon metadata). Lightning
is already on the Romania grid by virtue of being binned via
`GridProjection`, so the rest of the pipeline (extract_patches,
compute_normalization_stats, ...) reads these `.npy` files directly.
To regenerate a viewable `.nc` for GIS inspection, use
`our_data/lightning_data/inspect_lightning.py`.

Usage:
    # Process all dates
    python process_lightning_kml.py

    # Process a single date
    python process_lightning_kml.py --date 2025-05-15

    # Custom paths
    python process_lightning_kml.py -d F:/nowcasting/.../lightning_data -o ./output/lightning
"""

import os

# On Windows/conda envs pyproj and pyogrio (used by geopandas) can
# ship separate copies of PROJ whose proj.db databases disagree.
# The failure signature is:
#   CRSError: Invalid projection: EPSG:4326:
#     (Internal Proj Error: proj_create: no database context specified)
# Fix: point PROJ at pyproj's bundled database via PROJ_DATA (the
# modern env var) and PROJ_LIB (the legacy fallback). Do it BEFORE
# importing geopandas so pyogrio initialises with the right path.
try:
    import pyproj as _pyproj
    _proj_data_dir = _pyproj.datadir.get_data_dir()
    os.environ.setdefault("PROJ_DATA", _proj_data_dir)
    os.environ.setdefault("PROJ_LIB", _proj_data_dir)
except Exception:
    # pyproj not installed / broken import - let the downstream
    # geopandas import surface the real error instead of swallowing
    # it here.
    pass

import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import time
import re
import sys
import json
from pathlib import Path
import argparse


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = r"F:\nowcasting\coalition4-rcnn\our_data\lightning_data"
DEFAULT_OUTPUT_ROOT = r"F:\nowcasting\coalition4-rcnn\our_data\lightning_data"

# Time step is read from the project-level timestep_config.json. We prefer
# the lightning product's *native cadence* (products.lightning.cadence_minutes)
# so the maps are generated at the source resolution; the per-product
# minute filter inside the same JSON is what downstream tooling
# (summarize_lightning_data.py, intersect_product_coverage.py) uses to
# subsample. Falling back to the global step_minutes only when no native
# cadence is declared keeps the script working for the legacy
# "continuous / event-based" lightning configuration.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"


def _load_lightning_cadence():
    """Return (cadence_in_minutes, source_label) for the lightning maps.

    Prefers `products.lightning.cadence_minutes` from timestep_config.json
    (the native LINET cadence). Falls back to the global `step_minutes`
    when the lightning product is configured as continuous (cadence is
    null / missing). The label is included so the stdout banner makes
    the source of the chosen value explicit.
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
    lightning_cfg = cfg.get("products", {}).get("lightning", {})
    cadence = lightning_cfg.get("cadence_minutes")
    if cadence is not None:
        return int(cadence), "products.lightning.cadence_minutes"
    return int(cfg["step_minutes"]), "step_minutes (lightning cadence is null)"


def _load_lightning_filter():
    """Return the sorted list of filter minutes for lightning, or None.

    `products.lightning.filter` enumerates the minute-of-hour slots that
    downstream tooling (summarize_lightning_data.py /
    intersect_product_coverage.py) expects. When present, we only
    generate maps at those slots and the accumulation window for each
    map is the gap to the previous filter slot (variable width), so
    every stroke in the day lands in exactly one bin.
    """
    cfg = json.loads(TIMESTEP_CONFIG_PATH.read_text())
    flt = cfg.get("products", {}).get("lightning", {}).get("filter")
    if flt is None:
        return None
    return sorted(int(m) for m in flt)


TIME_STEP_MINUTES, _TIME_STEP_SOURCE = _load_lightning_cadence()
LIGHTNING_FILTER = _load_lightning_filter()


# =============================================================================
# KML reading
# =============================================================================

def _empty_lightning_df():
    """Return the empty schema used when a KML has no usable rows."""
    return pd.DataFrame({
        'lat':       pd.Series(dtype=float),
        'lon':       pd.Series(dtype=float),
        'current':   pd.Series(dtype=float),
        'timestamp': pd.Series(dtype='datetime64[ns]'),
    })


def read_kml_extract_coordinates(kml_file):
    """
    Read a LINET KML file and extract lightning stroke data.

    Parses the Description field of each Placemark to extract:
        - Position (lon, lat)
        - Current (kA)
        - Timestamp (UTC)

    LINET emits a stylesheet-only KML (no `<Placemark>` elements) for
    quiet days with no detected strokes. pyogrio doesn't treat that as
    "zero rows" - it raises `IndexError` from `get_default_layer`
    because there are no readable layers. We catch that (and any other
    read failure) and return an empty DataFrame so the surrounding
    pipeline's existing "empty df -> skip" path takes over instead of
    aborting the whole batch.

    Args:
        kml_file (str): Path to the KML file

    Returns:
        pd.DataFrame: Columns [lat, lon, current, timestamp]
    """
    try:
        gdf = gpd.read_file(kml_file)
    except (IndexError, ValueError, RuntimeError, OSError) as exc:
        # pyogrio surfaces "no layers" as IndexError; gdal-style "failed
        # to open" errors come through as RuntimeError/OSError. Any of
        # them is reported once and treated as a no-stroke day.
        print(f"  WARNING: cannot read {kml_file} ({type(exc).__name__}: "
              f"{exc}) - treating as no-stroke day, skipping.")
        return _empty_lightning_df()
    print(f"  Raw placemarks: {len(gdf)}")
    if len(gdf) == 0 or 'Description' not in gdf.columns:
        # Valid KML structure but zero placemarks (or no Description
        # column to parse) - same treatment as the read failure.
        return _empty_lightning_df()

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


def lightning_for_window(lightning_df, start_time, end_time):
    """Extract lightning rows whose timestamp falls in [start_time, end_time)."""
    mask = (
        (lightning_df['timestamp'] >= start_time)
        & (lightning_df['timestamp'] < end_time)
    )
    return lightning_df[mask]


# =============================================================================
# Map generation - filter-aligned time steps with variable windows
# =============================================================================

def generate_timesteps_and_windows(date_str):
    """Return (timestamps, start_times) lists for one calendar day.

    Two modes, selected by `products.lightning.filter` in
    timestep_config.json:

      * filter is set (typical) -> only filter minutes are emitted, one
        bin per (hour, filter_minute) pair. Each bin's accumulation
        window goes from the **previous** filter slot to this slot,
        looping across midnight to the last filter minute of the
        previous calendar day. Variable window widths ensure every
        minute of the day belongs to exactly one bin (no strokes lost
        in the gaps between filter steps).

      * filter is None (lightning declared continuous) -> fall back to
        a fixed-cadence grid of `TIME_STEP_MINUTES` width.

    Each `timestamps[i]` is the END of bin i (the value used in the
    .npy filename HH:MM). `start_times[i]` is the START of bin i.
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")

    if LIGHTNING_FILTER is not None:
        flt = sorted(LIGHTNING_FILTER)
        timestamps = [
            base.replace(hour=h, minute=m)
            for h in range(24) for m in flt
        ]
        # Previous step before the day's first filter minute is the LAST
        # filter minute of the previous calendar day. The current KML is
        # single-day, so any window crossing midnight just returns no
        # rows - the resulting map is empty, which mirrors the legacy
        # behaviour for the 00:00 slot.
        prev = (base - timedelta(days=1)).replace(hour=23, minute=flt[-1])
        start_times = []
        for ts in timestamps:
            start_times.append(prev)
            prev = ts
        return timestamps, start_times

    # Fixed-cadence fallback.
    steps_per_day = 1440 // TIME_STEP_MINUTES
    step = timedelta(minutes=TIME_STEP_MINUTES)
    timestamps = [base + i * step for i in range(steps_per_day)]
    start_times = [ts - step for ts in timestamps]
    return timestamps, start_times


def create_density_maps(lightning_df, grid_projection, timestamps, start_times):
    """
    Generate lightning density and current-weighted density maps on the
    filter-aligned time grid (with variable window widths from
    generate_timesteps_and_windows).

    Args:
        lightning_df (pd.DataFrame): Lightning data
        grid_projection: GridProjection object
        timestamps (list[datetime]): END of each bin (filter-aligned)
        start_times (list[datetime]): START of each bin (parallel to timestamps)

    Returns:
        tuple: (normalized_density_maps, normalized_weighted_maps, timestamps)
    """
    density_maps = []
    weighted_maps = []

    for current_time, start_time in zip(timestamps, start_times):
        lightning = lightning_for_window(lightning_df, start_time, current_time)

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


def generate_lightning_occurrence_maps(lightning_df, grid_projection,
                                        timestamps, start_times):
    """
    Generate binary lightning occurrence maps with 4-neighbor expansion
    on the filter-aligned time grid.

    Args:
        lightning_df (pd.DataFrame): Lightning data
        grid_projection: GridProjection object
        timestamps (list[datetime]): END of each bin (filter-aligned)
        start_times (list[datetime]): START of each bin (parallel to timestamps)

    Returns:
        tuple: (occurrence_maps, timestamps)
    """
    occurrence_maps = []

    for current_time, start_time in zip(timestamps, start_times):
        lightning = lightning_for_window(lightning_df, start_time, current_time)

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

def write_single_npy_file(output_path, data_slice, lat_data, lon_data,
                           timestamp, var_name, var_units, var_long_name,
                           time_interval):
    """Save a single per-cadence lightning map as a `.npy` array.

    Output is the bare 2-D grid; the shared `romania_grid_{lats,lons}.npy`
    in `reprojected_data/` carries the coordinates for all products, so
    we don't duplicate them per file. The `lat_data` / `lon_data` /
    `timestamp` / `var_units` / `var_long_name` / `time_interval`
    arguments are kept in the signature for call-site compatibility with
    the previous NetCDF writer but are not used here. To rebuild a
    CF-compliant `.nc` for QGIS inspection, see
    `our_data/lightning_data/inspect_lightning.py`.
    """
    # Mirror the dtype convention the old NetCDF path used: int8 for the
    # binary occurrence map, float32 for density / current.
    if var_name == 'datamap':
        # `datamap` was the generic name; here we infer the actual dtype
        # from the data range — occurrence is 0/1, others continuous.
        unique = np.unique(np.asarray(data_slice))
        if unique.size <= 2 and set(unique.tolist()) <= {0, 1}:
            arr = np.asarray(data_slice, dtype=np.int8)
        else:
            arr = np.asarray(data_slice, dtype=np.float32)
    else:
        arr = np.asarray(data_slice, dtype=np.float32)
    np.save(output_path, arr)
    return output_path


def save_lightning_maps_parallel(density_maps, weighted_maps, occurrence_maps,
                                timestamps, grid_projection, output_root,
                                date_str, batch_size=10):
    """
    Save all lightning maps as `.npy` arrays using a process pool.

    Output structure (COALITION-4 convention):
        {output_root}/density/nc4_{date}-Romania_density/lightning_density_YYYYMMDD_HHMM.npy
        {output_root}/current/nc4_{date}-Romania_current/lightning_current_YYYYMMDD_HHMM.npy
        {output_root}/occurrence/nc4_{date}-Romania_occurrence/lightning_occurrence_YYYYMMDD_HHMM.npy
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
                       f'Lightning Occurrence within {TIME_STEP_MINUTES}-minute window'),
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
                str(folders[prod_key] / f"{file_prefix}_{time_str}.npy"),
                data, lats, lons, timestamp,
                'datamap', var_units, var_long_name, str(TIME_STEP_MINUTES)
            ))

    total_tasks = len(all_tasks)
    processed = 0

    # Create pool once — recreating per batch leaks handles on Windows
    with multiprocessing.Pool(processes=num_cores) as pool:
        for i in range(0, total_tasks, batch_size):
            batch_tasks = all_tasks[i:i + batch_size]
            pool.starmap(write_single_npy_file, batch_tasks)

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

def expected_steps_per_day():
    """Number of bins this run will produce per date.

    Mirrors generate_timesteps_and_windows: with a filter, it's
    `len(filter) * 24`; otherwise it falls back to the fixed-cadence
    `1440 / TIME_STEP_MINUTES`. Used by the completeness check below.
    """
    if LIGHTNING_FILTER is not None:
        return len(LIGHTNING_FILTER) * 24
    return 1440 // TIME_STEP_MINUTES


def is_date_complete(output_root, date_str, expected_count):
    """True iff every sub-product subdir for `date_str` already has
    `expected_count` `.npy` files matching the lightning naming
    pattern. Used for resume support: a date that's already fully
    processed should be skipped so an interrupted run can be re-launched
    cheaply.

    The check is strict (all three sub-products at the expected count).
    Partial output -> not complete -> reprocess. A run that uses a
    different `--lightning_dir` for a different cadence would have
    different `expected_count` and would correctly not consider the
    old day "done".
    """
    output_root = Path(output_root)
    for prod_name, file_prefix in (
        ('density',    'lightning_density'),
        ('current',    'lightning_current'),
        ('occurrence', 'lightning_occurrence'),
    ):
        day_dir = output_root / prod_name / f"nc4_{date_str}-Romania_{prod_name}"
        if not day_dir.is_dir():
            return False
        n_found = sum(
            1 for f in os.listdir(day_dir)
            if f.startswith(f"{file_prefix}_") and f.endswith(".npy")
        )
        if n_found != expected_count:
            return False
    return True


def process_single_date(kml_path, date_str, output_root, force=False):
    """
    Process one KML file into per-cadence lightning `.npy` maps.

    Args:
        kml_path (str): Path to the KML file
        date_str (str): Date in 'YYYY-MM-DD' format
        output_root (str): Root output directory (product dirs are created inside)
        force (bool): If False (default), skip the date when all three
            sub-product subdirs already contain the expected number of
            `.npy` files. If True, always reprocess and overwrite.
    """
    from c4dl.projection import GridProjection, romania_grid_area

    print(f"\n{'=' * 70}")
    print(f"Processing: {date_str}")
    print(f"{'=' * 70}")
    print(f"  KML file: {kml_path}")

    expected = expected_steps_per_day()
    if not force and is_date_complete(output_root, date_str, expected):
        print(f"  Already complete ({expected} files per sub-product); "
              f"skipping. Pass --force to reprocess.")
        return

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

    # Generate filter-aligned time grid (or fixed-cadence fallback when
    # products.lightning.filter is null). Each timestamp comes paired
    # with its accumulation window start so variable window widths can
    # cover the gaps between filter slots.
    timestamps, start_times = generate_timesteps_and_windows(date_str)
    print(f"  Time steps: {len(timestamps)} "
          f"({timestamps[0].strftime('%H:%M')} to "
          f"{timestamps[-1].strftime('%H:%M')})")

    # Generate density + current maps
    print(f"  Generating density and current-weighted maps...")
    density_maps, weighted_maps, _ = create_density_maps(
        lightning_df, grid_projection, timestamps, start_times
    )

    # Generate occurrence maps
    print(f"  Generating occurrence maps...")
    occurrence_maps, _ = generate_lightning_occurrence_maps(
        lightning_df, grid_projection, timestamps, start_times
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


def run_all(data_root, output_root, date_filter=None, force=False):
    """
    Process all (or one) dates.

    Args:
        data_root (str): Root lightning data directory
        output_root (str): Root output directory
        date_filter (str or None): Process only this date (YYYY-MM-DD)
        force (bool): If False (default), skip dates whose three
            sub-product subdirs already contain the expected number of
            `.npy` files. If True, reprocess everything.
    """
    print("=" * 70)
    print("Lightning KML -> per-cadence .npy Maps (COALITION-4)")
    print("=" * 70)
    print(f"Data root   : {data_root}")
    print(f"Output root : {output_root}")
    print(f"Time step   : {TIME_STEP_MINUTES} minutes "
          f"({_TIME_STEP_SOURCE})")
    if LIGHTNING_FILTER is not None:
        print(f"Filter      : {LIGHTNING_FILTER}  "
              f"(products.lightning.filter; "
              f"{len(LIGHTNING_FILTER) * 24} maps/day, variable windows)")
    else:
        print(f"Filter      : (none) - {1440 // TIME_STEP_MINUTES} "
              f"maps/day at fixed {TIME_STEP_MINUTES}-min cadence")
    print(f"Resume      : {'OFF (--force)' if force else 'ON (skip complete dates)'}")

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
        process_single_date(kml_path, date_str, output_root, force=force)

    print(f"\n{'=' * 70}")
    print(f"Complete: processed {len(dates)} date(s)")
    print(f"{'=' * 70}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process LINET lightning KML data into per-cadence "
                    "`.npy` maps for COALITION-4. Cadence comes from "
                    "products.lightning.cadence_minutes (falling back to "
                    "step_minutes) in our_data/timestep_config.json."
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
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess and overwrite even when a date's three sub-product "
             "subdirs already contain the expected number of .npy files. "
             "Default behaviour is to skip already-complete dates so an "
             "interrupted batch run can be resumed cheaply."
    )

    args = parser.parse_args()

    run_all(
        data_root=args.data_root,
        output_root=args.output_root,
        date_filter=args.date,
        force=args.force,
    )
    