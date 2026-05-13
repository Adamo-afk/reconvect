"""
COALITION-4 patch identification pipeline.

Full pipeline for identifying which 256×256 patches from the standardized
6×3 grid contain convective activity, based on DBSCAN clustering of RZC
(rain rate) radar data.

Steps:
    1. Read raw RZC radar NetCDF files
    2. Regrid to the Romania 1536×768 EPSG:31700 grid
    3. Run DBSCAN to identify convective clusters (threshold >10, eps=5)
    4. Create binary mask: all cluster pixels = 1, rest = 0
    5. Overlay fixed 6×3 grid (18 patches of 256×256)
    6. Mark patches with ≥1 non-zero pixel as active
    7. Save patch index (CSV + JSON)

Grid layout (1536×768, numbered left-to-right, top-to-bottom):
    R1:  1   2   3   4   5   6
    R2:  7   8   9  10  11  12
    R3: 13  14  15  16  17  18

Input structure:
    {data_root}/radar_data/RZC/nc4_YYYY-MM-DD-Romania_RZC/*.nc

Output:
    {output_dir}/patch_index.csv
    {output_dir}/patch_index.json

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python identify_patches.py
    python identify_patches.py --date 2024-06-13
    python identify_patches.py --data_root ./our_data --output_dir ./patch_index
"""

import numpy as np
import os
import re
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from sklearn.cluster import DBSCAN
from netCDF4 import Dataset
from pyresample import geometry, kd_tree

from c4dl.projection import GridProjection, romania_grid_area


# Path to the timestep configuration file produced by validate_timestep.py
PROJECT_ROOT = Path(__file__).resolve().parent
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / "our_data" / "timestep_config.json"


def load_valid_minutes(source='radar'):
    """
    Read the per-source minute filter from timestep_config.json.

    For `source='radar'` returns the `products.radar.filter` set (e.g.
    `{'00','10','30','40'}` for step=15 with 10-min radar). For
    `source='opera'` returns `products.opera_rainfall_rate.filter`
    (the DBSCAN driver) — typically `{'00','15','30','45'}` for the
    15-min cadence.

    Errors out if the config or the requested product is missing.
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
    product_key = (
        'opera_rainfall_rate' if source == 'opera' else 'radar'
    )
    flt = cfg["products"].get(product_key, {}).get("filter")
    if flt is None:
        print(
            f"ERROR: {product_key!r} filter missing from "
            f"{TIMESTEP_CONFIG_PATH}",
            file=sys.stderr,
        )
        sys.exit(2)
    return {f"{m:02d}" for m in flt}

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_ROOT, 'patch_index')

# Grid dimensions
GRID_WIDTH = 1536   # 6 × 256
GRID_HEIGHT = 768   # 3 × 256
PATCH_SIZE = 256
N_COLS = 6          # patches per row
N_ROWS = 3          # rows of patches
N_PATCHES = N_COLS * N_ROWS  # 18

# DBSCAN parameters (same as current test_nc.py)
DBSCAN_THRESHOLD = 10   # mm/h — pixels below this are ignored
DBSCAN_EPS = 5           # neighborhood radius in pixels
DBSCAN_MIN_SAMPLES = 20  # minimum cluster size


# =============================================================================
# Radar I/O
# =============================================================================

def get_preferred_radar_group(ds):
    """Determine the correct radar group name in the NetCDF hierarchy."""
    data_group = ds.groups['data']
    if 'radarpicture_0' in data_group.groups:
        return 'radarpicture_0'
    elif 'radarpicture' in data_group.groups:
        return 'radarpicture'
    else:
        raise ValueError(f"No radarpicture group found. Available: {list(data_group.groups.keys())}")


def read_radar_netcdf(filepath):
    """
    Read a single radar NetCDF file and return the datamap plus
    source coordinate grids.

    Returns:
        tuple: (datamap, lat_grid, lon_grid) — all 2D numpy arrays
    """
    with Dataset(filepath, 'r') as ds:
        radarpicture = get_preferred_radar_group(ds)
        proj = ds.groups['data'].groups[radarpicture].groups['projection']
        datamap = ds.groups['data'].groups[radarpicture].groups['datamap'].variables['datamap'][:]

        lats = np.linspace(
            float(proj.getncattr('lat_ul')),
            float(proj.getncattr('lat_lr')),
            int(proj.getncattr('size_y'))
        )
        lons = np.linspace(
            float(proj.getncattr('lon_ul')),
            float(proj.getncattr('lon_lr')),
            int(proj.getncattr('size_x'))
        )

        lon_grid, lat_grid = np.meshgrid(lons, lats)

    return datamap, lat_grid, lon_grid


def parse_radar_filename(filename):
    """
    Parse datetime from radar filename.

    Handles nc4 format: nc4_2025-05-15-Romania_0110_RZC.nc
        → parts[1][:10] = date, parts[2] = HHMM

    Returns:
        tuple: (date_str 'YYYY-MM-DD', time_str 'HH:MM', iso_str)
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')

    # nc4 format: nc4_{date-Romania}_{HHMM}_{product}
    if parts[0] == 'nc4' and len(parts) >= 3:
        date_str = parts[1][:10]  # '2025-05-15' from '2025-05-15-Romania'
        time_str_raw = parts[2]   # '0110'
        dt = datetime.strptime(f"{date_str} {time_str_raw}", '%Y-%m-%d %H%M')
    else:
        # Fallback: first 12 chars = YYYYMMDDHHMM (old raw radar naming)
        dt = datetime.strptime(basename[:12], '%Y%m%d%H%M')

    return (
        dt.strftime('%Y-%m-%d'),
        dt.strftime('%H:%M'),
        dt.strftime('%Y-%m-%dT%H:%M:%S.000000000')
    )


# =============================================================================
# Regridding
# =============================================================================

def regrid_to_romania(datamap, source_lats, source_lons, target_lats, target_lons):
    """
    Regrid radar data to the Romania 1536×768 grid using nearest-neighbor.

    Args:
        datamap: 2D source data
        source_lats, source_lons: 2D source coordinate grids
        target_lats, target_lons: 2D target coordinate grids

    Returns:
        np.ndarray: Regridded data (768×1536), flipped vertically
    """
    source_geo = geometry.GridDefinition(lons=source_lons, lats=source_lats)
    target_geo = geometry.GridDefinition(lons=target_lons, lats=target_lats)

    regridded = kd_tree.resample_nearest(
        source_geo, datamap, target_geo,
        radius_of_influence=5000,
        fill_value=0.0
    )

    # Flip vertically to match Romania grid orientation (same as test_nc.py)
    regridded = np.flipud(regridded)

    return regridded


# =============================================================================
# DBSCAN → binary mask
# =============================================================================

def dbscan_binary_mask(datamap, threshold=None, eps=None, min_samples=None):
    """
    Run DBSCAN on the regridded radar data and produce a binary mask.

    All pixels belonging to any cluster (label != -1) are set to 1.
    Everything else (noise + below threshold) is 0.

    Args:
        datamap: 2D array (768×1536) of regridded RZC values
        threshold: minimum RZC value to consider. Defaults to the
            current module-level `DBSCAN_THRESHOLD` so CLI overrides
            (which reassign that global) actually take effect.
        eps: DBSCAN neighborhood radius. Defaults to `DBSCAN_EPS`.
        min_samples: DBSCAN minimum cluster size. Defaults to
            `DBSCAN_MIN_SAMPLES`.

    Returns:
        np.ndarray: Binary mask (768×1536), dtype uint8
    """
    # Resolve defaults at call time, not at module-import time. Using
    # `threshold=DBSCAN_THRESHOLD` in the signature would freeze the
    # default to whatever `DBSCAN_THRESHOLD` was when the function was
    # defined (Python evaluates defaults once), making `--threshold`
    # silently ineffective even after the CLI rebinds the global.
    if threshold is None:
        threshold = DBSCAN_THRESHOLD
    if eps is None:
        eps = DBSCAN_EPS
    if min_samples is None:
        min_samples = DBSCAN_MIN_SAMPLES
    # Find pixels above threshold
    points = np.where(datamap > threshold)

    binary_mask = np.zeros(datamap.shape, dtype=np.uint8)

    if len(points[0]) == 0:
        return binary_mask

    # Cluster the above-threshold pixel coordinates
    coords = np.column_stack((points[0], points[1]))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    # Set all cluster pixels (label != -1) to 1
    cluster_mask = labels != -1
    binary_mask[coords[cluster_mask, 0], coords[cluster_mask, 1]] = 1

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_cluster_pixels = int(cluster_mask.sum())

    return binary_mask


# =============================================================================
# Fixed grid patch identification
# =============================================================================

def get_patch_bounds(patch_number):
    """
    Get (row_start, row_end, col_start, col_end) for a 1-indexed patch number.

    Patches are numbered left-to-right, top-to-bottom:
        1  2  3  4  5  6
        7  8  9 10 11 12
       13 14 15 16 17 18
    """
    idx = patch_number - 1
    row = idx // N_COLS
    col = idx % N_COLS
    r0 = row * PATCH_SIZE
    c0 = col * PATCH_SIZE
    return r0, r0 + PATCH_SIZE, c0, c0 + PATCH_SIZE


def identify_active_patches(binary_mask):
    """
    Check which of the 18 fixed patches contain at least one non-zero pixel.

    Args:
        binary_mask: 2D array (768×1536), dtype uint8

    Returns:
        list[int]: Sorted list of active patch numbers (1-indexed)
    """
    active = []
    for p in range(1, N_PATCHES + 1):
        r0, r1, c0, c1 = get_patch_bounds(p)
        patch = binary_mask[r0:r1, c0:c1]
        if np.any(patch != 0):
            active.append(p)
    return active


# =============================================================================
# Visualization
# =============================================================================

def plot_patch_grid(regridded, binary_mask, active_patches, date_str, time_str,
                    output_dir):
    """
    Plot the regridded RZC data with the 6×3 patch grid overlay.

    Active patches (containing DBSCAN cluster pixels) are highlighted.
    Each patch is numbered at its upper-left corner (1–18).

    Args:
        regridded: 2D array (768×1536) of regridded RZC values
        binary_mask: 2D array (768×1536) of DBSCAN binary mask
        active_patches: list of active patch numbers (1-indexed)
        date_str: 'YYYY-MM-DD'
        time_str: 'HH:MM'
        output_dir: directory to save the PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.5),
                             constrained_layout=True)

    active_set = set(active_patches)

    for ax_idx, (ax, data, title_suffix) in enumerate(zip(
        axes,
        [regridded, binary_mask],
        ['RZC (regridded)', 'DBSCAN binary mask']
    )):
        if ax_idx == 0:
            # RZC with a continuous colormap
            im = ax.imshow(data, cmap='viridis', aspect='equal',
                           interpolation='nearest')
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='RZC')
        else:
            # Binary mask: 0=gray, 1=red
            cmap = ListedColormap(['#e0e0e0', '#d32f2f'])
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='equal',
                           interpolation='nearest')

        # Draw grid lines and patch labels
        for p in range(1, N_PATCHES + 1):
            r0, r1, c0, c1 = get_patch_bounds(p)

            is_active = p in active_set

            # Highlight active patches
            if is_active:
                rect = Rectangle(
                    (c0, r0), PATCH_SIZE, PATCH_SIZE,
                    linewidth=2.0, edgecolor='#f44336', facecolor='#f4433620',
                    linestyle='-', zorder=3
                )
            else:
                rect = Rectangle(
                    (c0, r0), PATCH_SIZE, PATCH_SIZE,
                    linewidth=0.8, edgecolor='white', facecolor='none',
                    linestyle='--', zorder=2
                )
            ax.add_patch(rect)

            # Patch number at upper-left corner
            label_color = '#f44336' if is_active else 'white'
            fontweight = 'bold' if is_active else 'normal'
            ax.text(
                c0 + 8, r0 + 22, str(p),
                color=label_color, fontsize=9, fontweight=fontweight,
                ha='left', va='top', zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.55, edgecolor='none')
            )

        ax.set_title(f'{title_suffix}', fontsize=11)
        ax.set_xlabel('Column (px)')
        ax.set_ylabel('Row (px)')

    patches_str = ', '.join(str(p) for p in active_patches) if active_patches else 'none'
    fig.suptitle(
        f'{date_str}  {time_str} UTC — Active patches: [{patches_str}]',
        fontsize=13, fontweight='bold'
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    safe_time = time_str.replace(':', '')
    filename = f"patches_{date_str}_{safe_time}.png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path


# =============================================================================
# File discovery
# =============================================================================

# Valid minutes per source are loaded lazily (and cached) from
# timestep_config.json so identify_patches.py works at any cadence
# selected by validate_timestep.py. Keyed by source ('radar' or 'opera').
_VALID_MINUTES_CACHE: dict[str, set[str]] = {}


def _valid_minutes_for(source):
    cached = _VALID_MINUTES_CACHE.get(source)
    if cached is None:
        cached = load_valid_minutes(source)
        _VALID_MINUTES_CACHE[source] = cached
    return cached


def is_on_grid(filename, source='radar'):
    """
    Check if a filename's timestamp falls on the configured cadence grid
    for the given source (radar uses `products.radar.filter`, OPERA uses
    `products.opera_rainfall_rate.filter`).

    Handles:
        nc4_2025-05-15-Romania_0110_RZC.nc  → parts[2] = '0110' → '10'
        202406131215something.nc             → basename[10:12] = '15'
    """
    valid_minutes = _valid_minutes_for(source)
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')

    # nc4 format: nc4_{date}_{HHMM}_{product}
    if parts[0] == 'nc4' and len(parts) >= 3:
        time_str = parts[2]
        if len(time_str) == 4 and time_str.isdigit():
            return time_str[2:4] in valid_minutes

    # Old raw format: first 12 chars = YYYYMMDDHHMM
    if len(basename) >= 12 and basename[:12].isdigit():
        return basename[10:12] in valid_minutes

    return True  # include if pattern not recognized


def discover_rzc_files(data_root):
    """
    Discover RZC NetCDF files on the configured cadence grid.

    The valid minute set is read from our_data/timestep_config.json, so
    a step=15 / cadence=10 setup keeps minutes {:00, :10, :30, :40} while
    step=10 keeps every native minute. Files at off-grid timestamps are skipped.

    Scans: {data_root}/radar_data/RZC/nc4_*-Romania_RZC/*.nc

    Returns:
        list[tuple]: Sorted list of (date_str, filepath) pairs
    """
    rzc_dir = os.path.join(data_root, 'radar_data', 'RZC')
    if not os.path.isdir(rzc_dir):
        print(f"RZC directory not found: {rzc_dir}")
        return []

    results = []
    filtered = 0
    date_dir_pattern = re.compile(r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_RZC$')

    for entry in sorted(os.listdir(rzc_dir)):
        match = date_dir_pattern.match(entry)
        if not match:
            continue

        date_str = match.group(1)
        day_dir = os.path.join(rzc_dir, entry)

        if not os.path.isdir(day_dir):
            continue

        nc_files = sorted(f for f in os.listdir(day_dir) if f.endswith('.nc'))
        for nc_file in nc_files:
            if not is_on_grid(nc_file):
                filtered += 1
                continue
            filepath = os.path.join(day_dir, nc_file)
            results.append((date_str, filepath))

    if filtered > 0:
        print(f"Filtered {filtered} non-quarter-hour files")

    return results


def discover_opera_files(data_root):
    """
    Discover the regridded OPERA rainfall-rate `.npy` files.

    Unlike the RZC path (which reads raw source NetCDFs and regrids
    on the fly inside this script), the OPERA branch consumes the
    output of `regrid.py --opera`. The arrays are already on the
    Romania 1536×768 grid, so the DBSCAN step skips the per-file
    regridding done for RZC.

    Scans: {data_root}/regridded_data/opera_data/rainfall_rate/
                nc4_{date}-Romania_rainfall_rate/*.npy

    Returns:
        list[tuple]: Sorted list of (date_str, filepath) pairs.
    """
    opera_dir = os.path.join(
        data_root, 'regridded_data', 'opera_data', 'rainfall_rate'
    )
    if not os.path.isdir(opera_dir):
        print(f"OPERA rainfall_rate dir not found: {opera_dir}")
        print("  Run `python regrid.py --opera` first.")
        return []

    results = []
    filtered = 0
    date_dir_pattern = re.compile(
        r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_rainfall_rate$'
    )

    for entry in sorted(os.listdir(opera_dir)):
        match = date_dir_pattern.match(entry)
        if not match:
            continue
        date_str = match.group(1)
        day_dir = os.path.join(opera_dir, entry)
        if not os.path.isdir(day_dir):
            continue
        for f in sorted(os.listdir(day_dir)):
            if not f.endswith('.npy'):
                continue
            if not is_on_grid(f, source='opera'):
                filtered += 1
                continue
            results.append((date_str, os.path.join(day_dir, f)))

    if filtered > 0:
        print(f"Filtered {filtered} off-grid OPERA files")
    return results


def parse_opera_filename(filepath):
    """Parse `nc4_{YYYY-MM-DD}-Romania_{HHMM}_rainfall_rate.npy`."""
    name = os.path.basename(filepath)
    m = re.match(
        r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_(\d{4})_rainfall_rate\.npy$', name
    )
    if not m:
        return None, None, None
    date_str = m.group(1)
    hhmm = m.group(2)
    time_str = f"{hhmm[:2]}:{hhmm[2:]}"
    iso_str = f"{date_str}T{time_str}:00.000000000"
    return date_str, time_str, iso_str


# =============================================================================
# Pipeline
# =============================================================================

def process_single_file(filepath, target_lats, target_lons):
    """
    Full pipeline for a single RZC source file:
        read → regrid → DBSCAN → binary mask → identify patches.
    """
    date_str, time_str, iso_str = parse_radar_filename(filepath)
    datamap, src_lats, src_lons = read_radar_netcdf(filepath)
    regridded = regrid_to_romania(datamap, src_lats, src_lons,
                                  target_lats, target_lons)
    binary_mask = dbscan_binary_mask(regridded)
    active_patches = identify_active_patches(binary_mask)
    return date_str, time_str, iso_str, active_patches, regridded, binary_mask


def process_single_opera_file(filepath):
    """
    Pipeline for one pre-regridded OPERA rainfall-rate `.npy`:
        load → DBSCAN → binary mask → identify patches.

    The array is already on the Romania 1536×768 grid (output of
    `regrid.py --opera`), so no per-file regridding is needed.
    """
    date_str, time_str, iso_str = parse_opera_filename(filepath)
    if date_str is None:
        return None
    regridded = np.load(filepath)
    # NaN may appear for off-grid pixels; DBSCAN expects finite values.
    regridded = np.nan_to_num(regridded, nan=0.0)
    binary_mask = dbscan_binary_mask(regridded)
    active_patches = identify_active_patches(binary_mask)
    return date_str, time_str, iso_str, active_patches, regridded, binary_mask


def run_pipeline(data_root, output_dir, date_filter=None, save_plots=False,
                 source='radar', start_date=None, end_date=None):
    """
    Run the full patch identification pipeline.

    Args:
        data_root: Path to our_data directory
        output_dir: Where to save CSV + JSON
        date_filter: Optional YYYY-MM-DD to process a single date
        save_plots: If True, save a PNG for each active timestamp
        source: 'radar' (RZC source NetCDFs, regrid in-script) or
                'opera' (pre-regridded OPERA rainfall_rate .npy files).
        start_date: Optional inclusive lower bound YYYY-MM-DD
        end_date:   Optional inclusive upper bound YYYY-MM-DD
    """
    print("=" * 70)
    print("COALITION-4 Patch Identification Pipeline")
    print("=" * 70)
    print(f"Source     : {source}")
    print(f"Data root  : {data_root}")
    print(f"Output dir : {output_dir}")
    print(f"Grid       : {GRID_WIDTH}×{GRID_HEIGHT} → {N_COLS}×{N_ROWS} patches of {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"DBSCAN     : threshold={DBSCAN_THRESHOLD}, eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")
    if save_plots:
        print(f"Plots      : enabled")

    # Discover files
    if source == 'opera':
        all_files = discover_opera_files(data_root)
        source_label = "OPERA rainfall_rate"
    else:
        all_files = discover_rzc_files(data_root)
        source_label = "RZC"

    if date_filter:
        all_files = [(d, f) for d, f in all_files if d == date_filter]
        print(f"Filtering to date: {date_filter}")

    # YYYY-MM-DD strings are lexicographically orderable, so a simple
    # string compare implements the inclusive range filter correctly.
    if start_date:
        all_files = [(d, f) for d, f in all_files if d >= start_date]
        print(f"Start date : {start_date}")
    if end_date:
        all_files = [(d, f) for d, f in all_files if d <= end_date]
        print(f"End date   : {end_date}")

    if not all_files:
        print(f"\nNo {source_label} files found.")
        return

    dates = sorted(set(d for d, _ in all_files))
    print(f"Found {len(all_files)} {source_label} files across {len(dates)} dates")

    # The RZC path needs target lat/lon for its per-file regridding.
    # The OPERA path consumes pre-regridded data, so target_lats/lons are unused.
    target_lats = target_lons = None
    if source == 'radar':
        print("\nInitializing Romania grid projection...")
        grid_projection = GridProjection(romania_grid_area)
        y, x = np.mgrid[:grid_projection.area.height, :grid_projection.area.width]
        target_lons, target_lats = grid_projection.inverse(y, x)
        print(f"Target grid shape: {target_lats.shape}")

    # Plot output directory
    plot_dir = os.path.join(output_dir, 'plots') if save_plots else None

    # Process all files
    results = []  # list of (date, time, iso, [active_patches])
    total = len(all_files)

    for i, (date_str, filepath) in enumerate(all_files):
        try:
            if source == 'opera':
                out = process_single_opera_file(filepath)
            else:
                out = process_single_file(filepath, target_lats, target_lons)
            if out is None:
                continue
            d, t, iso, active, regridded, binary_mask = out
            results.append((d, t, iso, active))

            if active:
                patches_str = ','.join(str(p) for p in active)
                print(f"  [{i+1}/{total}] {d} {t} → patches: [{patches_str}]")

                # Save plot for active timestamps
                if save_plots:
                    plot_patch_grid(
                        regridded, binary_mask, active, d, t, plot_dir
                    )
            else:
                print(f"  [{i+1}/{total}] {d} {t} → no active patches")

        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR processing {filepath}: {e}")
            continue

    if not results:
        print("\nNo results produced.")
        return

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    save_csv(results, output_dir)
    save_json(results, output_dir)

    # Summary
    total_timesteps = len(results)
    active_timesteps = sum(1 for _, _, _, a in results if a)
    all_active = set()
    for _, _, _, a in results:
        all_active.update(a)

    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"  Total time steps   : {total_timesteps}")
    print(f"  Active time steps  : {active_timesteps} ({active_timesteps/total_timesteps*100:.1f}%)")
    print(f"  Unique patches hit : {sorted(all_active) if all_active else 'none'}")
    print(f"  Output             : {output_dir}")
    if save_plots:
        print(f"  Plots saved        : {active_timesteps} PNGs in {plot_dir}")


# =============================================================================
# Output
# =============================================================================

def save_csv(results, output_dir):
    """
    Save patch index as CSV.

    Columns: date, time_utc, iso_timestamp, patch_1, patch_2, ..., patch_18
    Each patch column is 1 (active) or 0 (inactive).
    Only rows where at least one patch is active are included.
    """
    csv_path = os.path.join(output_dir, 'patch_index.csv')

    header = ['date', 'time_utc', 'iso_timestamp']
    header += [f'patch_{p}' for p in range(1, N_PATCHES + 1)]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for date_str, time_str, iso_str, active in results:
            if not active:
                continue
            row = [date_str, time_str, iso_str]
            row += [1 if p in active else 0 for p in range(1, N_PATCHES + 1)]
            writer.writerow(row)

    n_rows = sum(1 for _, _, _, a in results if a)
    print(f"  CSV saved: {csv_path} ({n_rows} active rows)")


def save_json(results, output_dir):
    """
    Save patch index as JSON.

    Structure:
    {
        "metadata": { ... },
        "index": {
            "2024-06-13T12:15:00.000000000": [3, 4, 8, 9, 10],
            ...
        }
    }
    """
    json_path = os.path.join(output_dir, 'patch_index.json')

    index = {}
    for date_str, time_str, iso_str, active in results:
        if active:
            index[iso_str] = active

    output = {
        "metadata": {
            "grid_width": GRID_WIDTH,
            "grid_height": GRID_HEIGHT,
            "patch_size": PATCH_SIZE,
            "n_cols": N_COLS,
            "n_rows": N_ROWS,
            "n_patches": N_PATCHES,
            "dbscan_threshold": DBSCAN_THRESHOLD,
            "dbscan_eps": DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
            "projection": "EPSG:31700",
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "patch_numbering": "left-to-right, top-to-bottom, 1-indexed"
        },
        "index": index
    }

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  JSON saved: {json_path} ({len(index)} active timestamps)")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COALITION-4 patch identification pipeline. "
                    "Identifies which 256×256 patches from the fixed 6×3 grid "
                    "contain convective activity based on DBSCAN clustering of RZC data."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Path to our_data directory (default: {DEFAULT_DATA_ROOT})"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV + JSON (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Process a single date (YYYY-MM-DD). Mutually exclusive with "
             "--start / --end."
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Inclusive lower bound date filter (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Inclusive upper bound date filter (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--threshold", type=float, default=DBSCAN_THRESHOLD,
        help=f"RZC threshold for DBSCAN (default: {DBSCAN_THRESHOLD})"
    )
    parser.add_argument(
        "--eps", type=float, default=DBSCAN_EPS,
        help=f"DBSCAN eps parameter (default: {DBSCAN_EPS})"
    )
    parser.add_argument(
        "--min_samples", type=int, default=DBSCAN_MIN_SAMPLES,
        help=f"DBSCAN min_samples parameter (default: {DBSCAN_MIN_SAMPLES})"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a PNG for each active timestamp (requires --date)"
    )
    parser.add_argument(
        "--source", type=str, default='radar', choices=['radar', 'opera'],
        help="Activity driver: 'radar' (legacy RZC source NetCDFs, "
             "regridded in-script) or 'opera' (pre-regridded OPERA "
             "rainfall_rate .npy from regrid.py --opera). Same DBSCAN "
             "threshold semantics — both interpret values as rain rate "
             "in mm/h. Default: radar."
    )

    args = parser.parse_args()

    # Validate: --plot requires --date
    if args.plot and args.date is None:
        parser.error("--plot requires --date to avoid generating thousands of PNGs")

    # --date is the single-date shortcut; combining it with --start/--end is
    # ambiguous, so disallow it.
    if args.date and (args.start or args.end):
        parser.error("--date is mutually exclusive with --start / --end")

    # Override globals if CLI args provided
    DBSCAN_THRESHOLD = args.threshold
    DBSCAN_EPS = args.eps
    DBSCAN_MIN_SAMPLES = args.min_samples

    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        date_filter=args.date,
        save_plots=args.plot,
        source=args.source,
        start_date=args.start,
        end_date=args.end,
    )
    