"""
COALITION-4 data reprojection pipeline (optimized with precomputed mappings).

Regrids all products to the Romania 1536x768 EPSG:31700 grid and caches
the results so the expensive reprojection runs only once per source file.

Optimization: the KD-tree is built ONCE per source geometry (not per file).
All files sharing the same source grid reuse the precomputed index mapping,
reducing reprojection to a fast numpy array lookup.

Products handled — every family writes `.npy`. The Romania grid coordinates
(`romania_grid_lats.npy`, `romania_grid_lons.npy`) and per-source projection
constants (`{mtg,opera}_constants.json`) are written **once** as sidecars
so the reprojected arrays remain self-recoverable for inspection.

    - MTG:       vis_06, ir_38, ir_105, wv_63, wv_73    → .npy
    - Lightning: density, current, occurrence (already on grid) → .npy
    - OPERA:     reflectivity, rainfall_rate            → .npy

Input paths:
    our_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
    our_data/satellite_data/MTG/mtg_constants.json
    our_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/opera_data/{reflectivity|rainfall_rate}/{YYYY}/{MM}/{DD}/*.h5

Output paths:
    our_data/reprojected_data/romania_grid_{lats,lons}.npy             (shared)
    our_data/reprojected_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
    our_data/reprojected_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/reprojected_data/opera_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/reprojected_data/opera_data/opera_constants.json

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python reproject.py --satellite MTG
    python reproject.py --lightning
    python reproject.py --opera
    python reproject.py --all
    python reproject.py --opera --date 2024-06-13
"""

import json
import numpy as np
import os
import re
import argparse
import threading
import time as timer_module
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from netCDF4 import Dataset
from pyresample import geometry, kd_tree
import pyproj

try:
    import h5py
except ImportError:
    h5py = None  # only required by the OPERA path; checked at call time

from c4dl.projection import GridProjection, romania_grid_area

# Global lock for netCDF4/HDF5 reads (C library is not thread-safe)
_nc_lock = threading.Lock()
_h5_lock = threading.Lock()  # h5py builds aren't always thread-safe


# =============================================================================
# OPERA HDF5 helpers (used by reproject_opera and shared with inspect tools)
# =============================================================================

# Filename timestamp parsers — see pipeline_opera.py for the conventions:
#   ISO     (current EWC dump): 2026-05-11T000500Z-reflectivity-composite-opera.h5
#   Compact (legacy / EUMETSAT): T_PAAH21_C_LFPW_20250615120000.h5
_OPERA_TIMESTAMP_PATTERN_ISO = re.compile(
    r'(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})(\d{2})Z?'
)
_OPERA_TIMESTAMP_PATTERN_COMPACT = re.compile(r'(\d{12,14})')


def _parse_opera_filename(name: str):
    """Return (date_str 'YYYY-MM-DD', hhmm 'HHMM') from an OPERA filename."""
    m = _OPERA_TIMESTAMP_PATTERN_ISO.search(name)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                          int(m.group(4)), int(m.group(5)), int(m.group(6)))
            return dt.strftime('%Y-%m-%d'), dt.strftime('%H%M')
        except ValueError:
            pass
    m = _OPERA_TIMESTAMP_PATTERN_COMPACT.search(name)
    if m:
        ts = m.group(1)[:12]
        try:
            dt = datetime.strptime(ts, '%Y%m%d%H%M')
            return dt.strftime('%Y-%m-%d'), dt.strftime('%H%M')
        except ValueError:
            pass
    return None, None


def _decode_h5_attr(value):
    """Decode an HDF5 attribute that may be bytes or a 0-d array."""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return value


def _read_opera_source_grid(h5_path):
    """
    Build a 2-D source lat/lon grid from an OPERA HDF5 file's /where metadata.
    Returns (lats, lons, projection_metadata_dict).
    """
    if h5py is None:
        raise RuntimeError("h5py is required for OPERA reprojection; "
                           "install via `pip install h5py`.")
    with h5py.File(h5_path, 'r') as f:
        where = f['/where'].attrs
        projdef = _decode_h5_attr(where['projdef'])
        xsize = int(_decode_h5_attr(where['xsize']))
        ysize = int(_decode_h5_attr(where['ysize']))
        xscale = float(_decode_h5_attr(where['xscale']))
        yscale = float(_decode_h5_attr(where['yscale']))
        ll_lon = float(_decode_h5_attr(where['LL_lon']))
        ll_lat = float(_decode_h5_attr(where['LL_lat']))
        ur_lon = float(_decode_h5_attr(where['UR_lon']))
        ur_lat = float(_decode_h5_attr(where['UR_lat']))

    src_proj = pyproj.Proj(projdef)
    geographic = pyproj.Proj('epsg:4326')
    to_xy = pyproj.Transformer.from_proj(geographic, src_proj, always_xy=True)
    ll_x, _ = to_xy.transform(ll_lon, ll_lat)   # x of lower-left corner (m)
    _, ur_y = to_xy.transform(ur_lon, ur_lat)   # y of upper-right corner (m)

    x_centres = ll_x + (np.arange(xsize) + 0.5) * xscale
    y_centres = ur_y - (np.arange(ysize) + 0.5) * yscale
    xx, yy = np.meshgrid(x_centres, y_centres)

    to_geo = pyproj.Transformer.from_proj(src_proj, geographic, always_xy=True)
    lons, lats = to_geo.transform(xx, yy)
    invalid = np.isinf(lons) | np.isinf(lats) | np.isnan(lons) | np.isnan(lats)
    lons = np.where(invalid, 0.0, lons).astype(np.float64)
    lats = np.where(invalid, 0.0, lats).astype(np.float64)

    metadata = {
        "projdef":  projdef,
        "xsize":    xsize,
        "ysize":    ysize,
        "xscale":   xscale,
        "yscale":   yscale,
        "LL_lon":   ll_lon,
        "LL_lat":   ll_lat,
        "UR_lon":   ur_lon,
        "UR_lat":   ur_lat,
    }
    return lats, lons, metadata


def _read_opera_data(h5_path):
    """
    Read the raw data and apply gain/offset + nodata/undetect masks.
    Returns a float32 2-D array (NaN where nodata, 0 where undetect).
    """
    if h5py is None:
        raise RuntimeError("h5py is required for OPERA reprojection; "
                           "install via `pip install h5py`.")
    with _h5_lock, h5py.File(h5_path, 'r') as f:
        ds = f['/dataset1/data1']
        what = ds['what'].attrs if 'what' in ds else None
        gain = float(_decode_h5_attr(what['gain'])) if what is not None else 1.0
        offset = float(_decode_h5_attr(what['offset'])) if what is not None else 0.0
        nodata = (float(_decode_h5_attr(what['nodata']))
                  if what is not None and 'nodata' in what else None)
        undetect = (float(_decode_h5_attr(what['undetect']))
                    if what is not None and 'undetect' in what else None)
        raw = np.asarray(ds['data'], dtype=np.float32)

    physical = gain * raw + offset
    if undetect is not None:
        physical = np.where(raw == undetect, 0.0, physical)
    if nodata is not None:
        physical = np.where(raw == nodata, np.nan, physical)
    return physical


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)

MTG_CHANNELS = [
    'vis_06', 'ir_38', 'ir_105', 'wv_63', 'wv_73'
]

MTG_1KM_CHANNELS = {'vis_06'}
MTG_2KM_CHANNELS = {'ir_38', 'ir_105', 'wv_63', 'wv_73'}

LIGHTNING_PRODUCTS = ['density', 'current', 'occurrence']

# Maximum parallel workers for day-folder processing
MAX_WORKERS = 6


# =============================================================================
# Precomputed mapping class
# =============================================================================

class PrecomputedMapping:
    """
    Precomputes and caches the nearest-neighbor index mapping between
    a source geometry and the target Romania grid.

    The expensive KD-tree build happens once in __init__. After that,
    apply() is a fast numpy array operation.
    """

    def __init__(self, source_lats, source_lons, target_lats, target_lons,
                 radius=5000, fill_value=0.0):
        self.target_shape = target_lats.shape
        self.fill_value = fill_value

        source_geo = geometry.GridDefinition(
            lons=source_lons, lats=source_lats
        )
        target_geo = geometry.GridDefinition(
            lons=target_lons, lats=target_lats
        )

        t0 = timer_module.time()
        (self.valid_input_index,
         self.valid_output_index,
         self.index_array,
         self.distance_array) = kd_tree.get_neighbour_info(
            source_geo, target_geo,
            radius_of_influence=radius,
            neighbours=1
        )
        elapsed = timer_module.time() - t0
        print(f"    KD-tree built in {elapsed:.2f}s "
              f"(source: {source_lats.shape}, target: {self.target_shape})")

    def apply(self, source_data, fill_value=None):
        """Apply the precomputed mapping to reproject source data (fast)."""
        fv = fill_value if fill_value is not None else self.fill_value

        reprojected = kd_tree.get_sample_from_neighbour_info(
            'nn',
            self.target_shape,
            source_data,
            self.valid_input_index,
            self.valid_output_index,
            self.index_array,
            fill_value=fv
        )
        return reprojected


# =============================================================================
# Shared utilities
# =============================================================================

def init_romania_grid():
    """Initialize Romania grid projection and return target coordinate grids."""
    print("Initializing Romania grid projection...")
    grid_projection = GridProjection(romania_grid_area)
    y, x = np.mgrid[:grid_projection.area.height, :grid_projection.area.width]
    target_lons, target_lats = grid_projection.inverse(y, x)
    print(f"  Target grid shape: {target_lats.shape}")
    return target_lats, target_lons


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def output_exists(path):
    return os.path.isfile(path)


def find_data_variable(ds, channel_name):
    """Find the data variable name in a NetCDF dataset."""
    for candidate in [channel_name, channel_name.upper(),
                      channel_name.lower(), 'data', 'datamap']:
        if candidate in ds.variables:
            return candidate
    coord_vars = {'latitude', 'longitude', 'lat', 'lon',
                  'x', 'y', 'time', 'nx', 'ny'}
    for v in ds.variables:
        if v not in coord_vars and 'pal' not in v:
            return v
    return None


# =============================================================================
# Radar
# =============================================================================







# =============================================================================
# Module-level workers for ProcessPoolExecutor
# =============================================================================
#
# Each `reproject_*` function below runs its day folders through a fresh
# `ProcessPoolExecutor`. The workers MUST be defined at module level so
# they can be pickled to child processes (nested functions can't be).
#
# The PrecomputedMapping is large (a few MB of KD-tree index arrays). To
# avoid re-pickling it on every job, we pass it once to each worker via
# the pool's `initializer=_init_worker` and stash it in `_WORKER_STATE`.
# Workers fetch their constants from that dict.

_WORKER_STATE: dict = {}


def _rebuild_aggregated_errors(reprojected_root):
    """(Re)build `<reprojected_root>/errors.txt` from every per-category log.

    Walks `reprojected_root` for `reproject_<category>.log` files,
    concatenates their contents into a single `errors.txt`, prefixing
    each non-empty block with a `# category: <name>` comment so a
    reader can see which reproject pass each line came from. The
    `#`-prefixed lines are ignored by intersect_product_coverage.py's
    `parse_error_log` (which only consumes lines starting with `ERROR `),
    so the aggregated file is a drop-in for any single `--errors_log`
    flag downstream.

    Always overwrites — `errors.txt` reflects the state of the *current*
    set of per-category logs on disk, not a cumulative history. Empty
    when no logs have any content.
    """
    agg_path = os.path.join(reprojected_root, 'errors.txt')
    log_files = sorted(
        f for f in os.listdir(reprojected_root)
        if f.startswith('reproject_') and f.endswith('.log')
    )
    total_lines = 0
    with open(agg_path, 'w', encoding='utf-8') as out:
        for log_name in log_files:
            log_path = os.path.join(reprojected_root, log_name)
            category = log_name[len('reproject_'):-len('.log')]
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = [line.rstrip('\n') for line in f
                             if line.strip().startswith('ERROR ')]
            except OSError:
                continue
            if not lines:
                continue
            out.write(f"# category: {category}\n")
            for line in lines:
                out.write(line + '\n')
                total_lines += 1
    return agg_path, total_lines


def _write_reproject_log(reprojected_root, category, all_errors):
    """Write a single-line-per-error log named after the reproject category.

    Format:
        ERROR <filename>: <message>

    The log is written under `reprojected_root` (typically
    `our_data/reprojected_data/`) so it sits next to the reprojected
    outputs. Always overwrites — the log reflects the state of the most
    recent run, not a cumulative history. If no errors occurred, the log
    is still written but empty so downstream consumers can `open()` it
    unconditionally.

    After writing the per-category log, `errors.txt` is re-aggregated
    from every `reproject_*.log` currently on disk so the single-file
    form stays in lockstep with the per-category logs. That lets
    `intersect_product_coverage.py` consume either form interchangeably
    via `--errors_log`.
    """
    ensure_dir(reprojected_root)
    log_path = os.path.join(reprojected_root, f"reproject_{category}.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        for fname, msg in all_errors:
            f.write(f"ERROR {fname}: {msg}\n")
    n = len(all_errors)
    if n > 0:
        print(f"    Wrote {n} error line(s) to {log_path}")

    agg_path, agg_lines = _rebuild_aggregated_errors(reprojected_root)
    if agg_lines > 0:
        print(f"    Aggregated {agg_lines} error line(s) into {agg_path}")
    return log_path


def _init_worker(state):
    """Per-process initializer — populates `_WORKER_STATE` with the
    mapping plus any other per-batch constants (channel, product, base
    output directory, ...) shared across all jobs in a single pool. The
    state dict is passed as a single positional argument because
    `ProcessPoolExecutor.initargs` is positional-only."""
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)






def _mtg_day_worker(job):
    mapping = _WORKER_STATE['mapping']
    day_folder, day_path, out_dir, npy_files = job
    new, skipped = 0, 0
    errors: list[tuple[str, str]] = []
    for npy_file in npy_files:
        out_path = os.path.join(out_dir, npy_file)
        if output_exists(out_path):
            skipped += 1
            continue
        try:
            filepath = os.path.join(day_path, npy_file)
            sat_data = np.load(filepath)
            if isinstance(sat_data, np.ma.MaskedArray):
                sat_data = sat_data.filled(np.nan)
            sat_data = np.asarray(sat_data, dtype=np.float32)
            if sat_data.ndim == 3:
                sat_data = np.squeeze(sat_data, axis=0)
            reprojected = mapping.apply(sat_data, fill_value=np.nan)
            ensure_dir(out_dir)
            np.save(out_path, reprojected)
            new += 1
        except Exception as e:
            errors.append((npy_file, str(e)))
    return day_folder, new, skipped, errors


def _lightning_day_worker(job):
    """Lightning is already on the Romania grid (binned by `GridProjection`
    inside `read_kml_version2.py`), so 'reprojection' is just a one-step
    pass-through: load the source `.npy`, optionally normalise dtype /
    squeeze a stray time axis, save under `reprojected_data/`. The
    occurrence map keeps its int8 dtype; density / current become
    float32. Source and destination both use the
    `lightning_<product>_YYYYMMDD_HHMM.npy` naming convention so file
    names stay stable across the move.
    """
    day_folder, day_path, out_dir, npy_files = job
    new, skipped = 0, 0
    errors: list[tuple[str, str]] = []
    for npy_file in npy_files:
        out_path = os.path.join(out_dir, npy_file)
        if output_exists(out_path):
            skipped += 1
            continue
        try:
            filepath = os.path.join(day_path, npy_file)
            datamap = np.load(filepath)
            if isinstance(datamap, np.ma.MaskedArray):
                datamap = datamap.filled(0.0)
            if datamap.ndim == 3:
                datamap = np.squeeze(datamap, axis=0)
            ensure_dir(out_dir)
            # Preserve the on-disk dtype of the occurrence binary map
            # (int8 from read_kml_version2.write_single_npy_file) so
            # downstream consumers can treat it as a categorical flag.
            np.save(
                out_path,
                datamap if datamap.dtype == np.int8
                else datamap.astype(np.float32),
            )
            new += 1
        except Exception as e:
            errors.append((npy_file, str(e)))
    return day_folder, new, skipped, errors




def _opera_day_worker(job):
    mapping = _WORKER_STATE['mapping']
    product = _WORKER_STATE['product']
    reprojected_base = _WORKER_STATE['reprojected_base']
    date_str, day_path, h5_files = job
    new, skipped = 0, 0
    errors: list[tuple[str, str]] = []
    out_dir = os.path.join(
        reprojected_base, product, f"nc4_{date_str}-Romania_{product}"
    )
    for h5_file in h5_files:
        _, hhmm = _parse_opera_filename(h5_file)
        if hhmm is None:
            errors.append((h5_file, "filename did not match the expected pattern"))
            continue
        out_name = f"nc4_{date_str}-Romania_{hhmm}_{product}.npy"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            skipped += 1
            continue
        try:
            physical = _read_opera_data(os.path.join(day_path, h5_file))
            reprojected = mapping.apply(physical, fill_value=np.nan)
            ensure_dir(out_dir)
            np.save(out_path, reprojected.astype(np.float32))
            new += 1
        except Exception as e:
            errors.append((h5_file, str(e)))
    return date_str, new, skipped, errors







# =============================================================================
# MTG satellite
# =============================================================================

def _read_mtg_source_grid_from_constants(constants_path, resolution):
    """
    Reconstruct source lat/lon grids from the mtg_constants.json file.

    Reads the geostationary projection parameters and 1-D scanning
    angles, then applies the pyproj inverse geostationary projection
    to produce 2-D lat/lon arrays.

    Args:
        constants_path (str): Path to mtg_constants.json.
        resolution (str): '1km' or '2km'.

    Returns:
        tuple: (lat_2d, lon_2d) as float64 numpy arrays.
    """
    import json as _json

    with open(constants_path, 'r') as f:
        constants = _json.load(f)

    proj_params = constants['projection']
    res_data = constants[resolution]

    if res_data is None:
        raise ValueError(f"No {resolution} data in {constants_path}")

    h = float(proj_params['perspective_point_height'])
    a = float(proj_params['semi_major_axis'])
    b = float(proj_params['semi_minor_axis'])
    lon_0 = float(proj_params['longitude_of_projection_origin'])
    sweep = str(proj_params.get('sweep_angle_axis', 'y'))

    x_rad = np.array(res_data['x_geos'], dtype=np.float64)
    y_rad = np.array(res_data['y_geos'], dtype=np.float64)

    # Convert scanning angles (radians) to projection coordinates (meters)
    x_m = x_rad * h
    y_m = y_rad * h
    xx, yy = np.meshgrid(x_m, y_m)

    # Inverse geostationary projection → lon, lat
    proj = pyproj.Proj(
        proj='geos', h=h, a=a, b=b, lon_0=lon_0, sweep=sweep
    )
    lon_2d, lat_2d = proj(xx, yy, inverse=True)

    # Off-disk pixels come back as inf; replace with NaN
    invalid = np.isinf(lat_2d) | np.isinf(lon_2d)
    lat_2d[invalid] = np.nan
    lon_2d[invalid] = np.nan

    return lat_2d.astype(np.float64), lon_2d.astype(np.float64)


def reproject_satellite_mtg(data_root, target_lats, target_lons, date_filter=None):
    """
    Reproject MTG channels from pipeline-produced .npy files.

    Source grid: reconstructed from mtg_constants.json (projection
    parameters + 1-D scanning angle arrays).

    KD-tree built once per resolution (1km/2km), reused across all
    channels sharing that resolution.

    Output: .npy files containing the reprojected 2-D array on the
    768×1536 EPSG:31700 grid. The shared Romania-grid lat/lon arrays
    are written once by run() at reprojected_data/romania_grid_{lats,lons}.npy
    (not per-product). Use inspect_lightning.py as a template to
    reconstruct full .nc files
    for GIS viewing.
    """
    mtg_dir = os.path.join(data_root, 'satellite_data', 'MTG')
    constants_path = os.path.join(mtg_dir, 'mtg_constants.json')
    reprojected_base = os.path.join(
        data_root, 'reprojected_data', 'satellite_data', 'MTG'
    )

    if not os.path.isdir(mtg_dir):
        print(f"  MTG directory not found: {mtg_dir}")
        return

    if not os.path.isfile(constants_path):
        print(f"  ERROR: mtg_constants.json not found at {constants_path}")
        print(f"  Run pipeline_msg_mtg.py first to generate it.")
        return

    # Build KD-tree mappings per resolution from the constants file
    mapping_cache = {}
    all_errors: list[tuple[str, str]] = []

    for res in ['1km', '2km']:
        # Check if any requested channel needs this resolution
        channels_at_res = (
            MTG_1KM_CHANNELS if res == '1km' else MTG_2KM_CHANNELS
        )
        has_channels = any(
            os.path.isdir(os.path.join(mtg_dir, ch))
            for ch in channels_at_res
        )
        if not has_channels:
            continue

        print(f"  Extracting MTG {res} source grid from constants...")
        try:
            src_lats, src_lons = _read_mtg_source_grid_from_constants(
                constants_path, res
            )
        except Exception as e:
            print(f"  ERROR reading {res} source grid: {e}")
            continue

        # Clean NaN values (off-disk pixels)
        nan_mask = np.isnan(src_lats) | np.isnan(src_lons)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            print(f"  MTG {res}: {n_nan} off-disk pixels "
                  f"({n_nan / nan_mask.size * 100:.1f}%) → zeroed for KD-tree")
            src_lats[nan_mask] = 0.0
            src_lons[nan_mask] = 0.0

        src_lats = np.clip(src_lats, -90.0, 90.0)
        src_lons = np.clip(src_lons, -180.0, 180.0)

        print(f"  Building MTG {res} KD-tree "
              f"(source shape: {src_lats.shape})...")
        mapping_cache[res] = PrecomputedMapping(
            src_lats, src_lons, target_lats, target_lons
        )

    # Process each channel
    for channel in MTG_CHANNELS:
        channel_dir = os.path.join(mtg_dir, channel)
        if not os.path.isdir(channel_dir):
            print(f"\n  Channel: {channel} — NOT FOUND at {channel_dir}")
            continue

        res = '1km' if channel in MTG_1KM_CHANNELS else '2km'
        if res not in mapping_cache:
            print(f"  Skipping {channel}: no {res} mapping available")
            continue

        mapping = mapping_cache[res]
        print(f"\n  Channel: {channel} ({res}, mapping reused)")

        # Collect day folders
        day_jobs = []
        for day_folder in sorted(os.listdir(channel_dir)):
            if date_filter and date_filter not in day_folder:
                continue
            day_path = os.path.join(channel_dir, day_folder)
            if not os.path.isdir(day_path):
                continue
            out_dir = os.path.join(reprojected_base, channel, day_folder)
            npy_files = sorted(
                f for f in os.listdir(day_path) if f.endswith('.npy')
            )
            if npy_files:
                day_jobs.append((day_folder, day_path, out_dir, npy_files))

        if not day_jobs:
            print(f"    No day folders found for {channel}")
            continue

        # The cadence filter is enforced upstream by pipeline_msg_mtg.py
        # (via timestep_config.json), so no minute filtering is needed here.
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=({'mapping': mapping},),
        ) as pool:
            futures = {
                pool.submit(_mtg_day_worker, job): job[0]
                for job in day_jobs
            }
            for future in as_completed(futures):
                day_folder, new, skipped, errs = future.result()
                total = new + skipped
                if total > 0 or errs:
                    print(f"    {day_folder}: {new} new, {skipped} cached, "
                          f"{total} used, {len(errs)} errors")
                all_errors.extend(errs)

    _write_reproject_log(
        os.path.join(data_root, 'reprojected_data'),
        'satellite_MTG', all_errors,
    )


# =============================================================================
# Lightning (no reprojection)
# =============================================================================

def reproject_lightning(data_root, date_filter=None):
    """Copy lightning `.npy` files from `lightning_data/` to
    `reprojected_data/lightning_data/`, day folders in parallel.

    Lightning is already on the Romania grid by virtue of being binned
    with the `GridProjection` inside `read_kml_version2.py`, which now
    writes `.npy` directly. This step exists to keep lightning's
    on-disk layout consistent with the other products (raw source dir
    -> reprojected_data mirror), not to do any actual reprojection.
    """
    lightning_dir = os.path.join(data_root, 'lightning_data')
    reprojected_base = os.path.join(
        data_root, 'reprojected_data', 'lightning_data'
    )

    if not os.path.isdir(lightning_dir):
        print(f"  Lightning directory not found: {lightning_dir}")
        return

    all_errors: list[tuple[str, str]] = []

    for product in LIGHTNING_PRODUCTS:
        product_dir = os.path.join(lightning_dir, product)
        if not os.path.isdir(product_dir):
            print(f"\n  Product: {product} — NOT FOUND at {product_dir}")
            continue

        print(f"\n  Product: {product}")

        # Collect day folders
        day_jobs = []
        for day_folder in sorted(os.listdir(product_dir)):
            if date_filter and date_filter not in day_folder:
                continue
            day_path = os.path.join(product_dir, day_folder)
            if not os.path.isdir(day_path):
                continue
            out_dir = os.path.join(reprojected_base, product, day_folder)
            npy_files = sorted(
                f for f in os.listdir(day_path) if f.endswith('.npy')
            )
            if npy_files:
                day_jobs.append((day_folder, day_path, out_dir, npy_files))

        if not day_jobs:
            print(f"    No day folders found for {product}")
            continue

        # Lightning has no reproject step (already on the Romania grid) —
        # the worker just reads NetCDF and writes `.npy`. No mapping
        # to share, but we still hand `_init_worker` an empty state so
        # `_WORKER_STATE` is in a known shape inside the pool.
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=({},),
        ) as pool:
            futures = {
                pool.submit(_lightning_day_worker, job): job[0]
                for job in day_jobs
            }
            for future in as_completed(futures):
                day_folder, new, skipped, errs = future.result()
                total = new + skipped
                if total > 0 or errs:
                    print(f"    {day_folder}: {new} new, {skipped} cached, "
                          f"{total} total, {len(errs)} errors")
                all_errors.extend(errs)

    _write_reproject_log(
        os.path.join(data_root, 'reprojected_data'),
        'lightning', all_errors,
    )










# =============================================================================
# OPERA radar (per-variable .npy, day-folder parallelism)
# =============================================================================

OPERA_PRODUCTS = {
    "reflectivity": {
        "remote_subdir": "reflectivity",
        "long_name":     "Maximum reflectivity",
        "units":         "dBZ",
    },
    "rainfall_rate": {
        "remote_subdir": "rainfall_rate",
        "long_name":     "Instantaneous rainfall rate",
        "units":         "mm h-1",
    },
}


def reproject_opera(data_root, target_lats, target_lons, date_filter=None):
    """
    Reproject OPERA HDF5 files to per-product .npy on the Romania grid.

    Source layout: our_data/opera_data/{product}/{YYYY}/{MM}/{DD}/*.h5
    Output layout: our_data/reprojected_data/opera_data/{product}/
                       nc4_{date}-Romania_{product}/
                           nc4_{date}-Romania_{HHMM}_{product}.npy

    One KD-tree mapping is built per product from that product's first
    available `.h5` file (both reflectivity and rainfall_rate are on the
    2 km grid).
    Day folders for each product are processed in parallel.
    """
    if h5py is None:
        print("  h5py not installed; skipping OPERA reprojection "
              "(pip install h5py).")
        return

    opera_dir = os.path.join(data_root, 'opera_data')
    reprojected_base = os.path.join(data_root, 'reprojected_data', 'opera_data')

    if not os.path.isdir(opera_dir):
        print(f"  OPERA directory not found: {opera_dir}")
        return

    ensure_dir(reprojected_base)
    constants = {}
    all_errors: list[tuple[str, str]] = []

    for product, cfg in OPERA_PRODUCTS.items():
        product_dir = os.path.join(opera_dir, cfg["remote_subdir"])
        if not os.path.isdir(product_dir):
            print(f"  [{product}] no source dir at {product_dir}; skipping")
            continue

        # Walk {YYYY}/{MM}/{DD}/ and collect per-day batches
        day_jobs = []
        for year in sorted(os.listdir(product_dir)):
            year_path = os.path.join(product_dir, year)
            if not os.path.isdir(year_path) or not year.isdigit():
                continue
            for month in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path) or not month.isdigit():
                    continue
                for day in sorted(os.listdir(month_path)):
                    day_path = os.path.join(month_path, day)
                    if not os.path.isdir(day_path) or not day.isdigit():
                        continue
                    date_str = f"{year}-{month}-{day}"
                    if date_filter and date_filter != date_str:
                        continue
                    h5_files = sorted(
                        f for f in os.listdir(day_path) if f.endswith('.h5')
                    )
                    if h5_files:
                        day_jobs.append((date_str, day_path, h5_files))

        if not day_jobs:
            print(f"  [{product}] no day folders found")
            continue

        # Build the KD-tree once for this product
        first_file = os.path.join(day_jobs[0][1], day_jobs[0][2][0])
        print(f"  [{product}] building mapping from {day_jobs[0][2][0]}...")
        try:
            src_lats, src_lons, meta = _read_opera_source_grid(first_file)
        except Exception as e:
            print(f"  [{product}] ERROR reading source grid: {e}")
            all_errors.append((day_jobs[0][2][0],
                               f"source-grid read failed: {e}"))
            continue
        constants[product] = meta
        nan_mask = np.isnan(src_lats) | np.isnan(src_lons)
        if nan_mask.any():
            src_lats[nan_mask] = 0.0
            src_lons[nan_mask] = 0.0
        print(f"  [{product}] source grid: {src_lats.shape}")
        mapping = PrecomputedMapping(
            src_lats, src_lons, target_lats, target_lons,
        )

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=({'mapping': mapping,
                       'product': product,
                       'reprojected_base': reprojected_base},),
        ) as pool:
            futures = {pool.submit(_opera_day_worker, j): j[0] for j in day_jobs}
            for future in as_completed(futures):
                date_str, new, skipped, errs = future.result()
                total = new + skipped
                if total > 0 or errs:
                    print(f"    [{product}] {date_str}: {new} new, "
                          f"{skipped} cached, {len(errs)} errors")
                all_errors.extend(errs)

    if constants:
        constants_path = os.path.join(reprojected_base, 'opera_constants.json')
        with open(constants_path, 'w') as f:
            json.dump(constants, f, indent=2)
        print(f"  Wrote {constants_path}")

    _write_reproject_log(
        os.path.join(data_root, 'reprojected_data'),
        'opera', all_errors,
    )


# =============================================================================
# Main pipeline
# =============================================================================

def run(data_root, mode, instrument=None, date_filter=None):
    print("=" * 70)
    print("COALITION-4 Data Reprojection Pipeline (precomputed mappings)")
    print("=" * 70)
    print(f"Data root : {data_root}")
    print(f"Mode      : {mode}" + (f" ({instrument})" if instrument else ""))
    print(f"Workers   : {MAX_WORKERS}")
    if date_filter:
        print(f"Date      : {date_filter}")

    t_start = timer_module.time()

    needs_grid = mode in ('satellite', 'opera', 'all')
    if needs_grid:
        target_lats, target_lons = init_romania_grid()
        # Write the shared Romania-grid coordinate arrays once at the root
        # of reprojected_data/ so every product can consume them as a sidecar.
        reprojected_root = os.path.join(data_root, 'reprojected_data')
        ensure_dir(reprojected_root)
        grid_lats_path = os.path.join(reprojected_root, 'romania_grid_lats.npy')
        grid_lons_path = os.path.join(reprojected_root, 'romania_grid_lons.npy')
        if not os.path.isfile(grid_lats_path):
            np.save(grid_lats_path, target_lats)
            np.save(grid_lons_path, target_lons)
            print(f"  Wrote Romania grid coords -> {reprojected_root}")
    else:
        target_lats = target_lons = None

    if mode in ('satellite', 'all'):
        print(f"\n{'='*70}")
        print("MTG satellite channels")
        print(f"{'='*70}")
        if target_lats is None:
            target_lats, target_lons = init_romania_grid()
        reproject_satellite_mtg(data_root, target_lats, target_lons, date_filter)

    if mode in ('lightning', 'all'):
        print(f"\n{'='*70}")
        print("Lightning products")
        print(f"{'='*70}")
        reproject_lightning(data_root, date_filter)

    if mode in ('opera', 'all'):
        print(f"\n{'='*70}")
        print("OPERA radar products")
        print(f"{'='*70}")
        reproject_opera(data_root, target_lats, target_lons, date_filter)

    elapsed = timer_module.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.1f}s.")
    print(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COALITION-4 data reprojection pipeline. "
                    "Uses precomputed KD-tree mappings for speed."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Process a single date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=MAX_WORKERS,
        help=f"Max parallel workers for day-folder processing (default: {MAX_WORKERS})"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--satellite", type=str, choices=['MTG'],
                       metavar='INSTRUMENT', help="Reproject satellite channels")
    group.add_argument("--lightning", action="store_true",
                       help="Cache lightning data as .npy")
    group.add_argument("--opera", action="store_true",
                       help="Reproject OPERA radar products (HDF5 -> .npy)")
    group.add_argument("--all", action="store_true",
                       help="Reproject all products")

    args = parser.parse_args()

    # Override worker count from CLI
    MAX_WORKERS = args.workers

    if args.satellite:
        mode, instrument = 'satellite', args.satellite
    elif args.lightning:
        mode, instrument = 'lightning', None
    elif args.opera:
        mode, instrument = 'opera', None
    elif args.all:
        mode, instrument = 'all', None

    run(
        data_root=args.data_root,
        mode=mode,
        instrument=instrument,
        date_filter=args.date,
    )
