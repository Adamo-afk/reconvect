"""
COALITION-4 patch identification pipeline.

Full pipeline for identifying which 256×256 patches from the standardized
6×3 grid contain convective activity, based on DBSCAN clustering of RZC
(rain rate) radar data.

Steps:
    1. Read raw RZC radar NetCDF files
    2. Reproject to the Romania 1536×768 EPSG:31700 grid
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
# Country-border overlay (cartopy Natural Earth preferred, pyproj fallback)
# =============================================================================
# Mirrors the helpers in visualize_gt_vs_pred.py so the
# patch-selection diagnostic plot reads as a consistent panel of the same
# Romania visualisation - just with the GT/Pred rows swapped for the
# RZC field + DBSCAN mask. Coords go lat/lon -> EPSG:31700 -> pixel via
# the canvas's known UTM extent (no GeoAxes needed).

# UTM zone 35N (EPSG:31700) extent of the Romania grid, from c4dl/projection.py.
# Order: (lower_left_x, upper_right_x, lower_left_y, upper_right_y).
ROMANIA_EXTENT_UTM = (-177324.0, 1331353.0, 77148.0, 723370.0)

# Pixels of breathing room beyond the farthest data-canvas edge from
# Romania's centroid. Same value the prediction plotter uses (~250 km in
# UTM 35N at the 1 km/pixel grid resolution).
VIEW_EXTRA_PAD = 80

# Natural Earth admin_0_countries that we want to draw alongside Romania.
# Aliases included because the attribute name varies across NE versions.
NEIGHBOUR_NAMES = {
    "Hungary", "Serbia", "Bulgaria", "Moldova", "Republic of Moldova",
    "Ukraine", "Slovakia", "Austria", "Czech Republic", "Czechia",
    "Poland", "Belarus", "Russia", "Croatia", "Bosnia and Herz.",
    "Bosnia and Herzegovina", "Greece", "North Macedonia", "Macedonia",
    "Albania", "Italy", "Slovenia", "Turkey", "Montenegro", "Kosovo",
    "Republic of Serbia",
}

# Coarse fallback used only when cartopy is unavailable.
_ROMANIA_OUTLINE_LONLAT_FALLBACK = [
    (22.69, 47.99), (23.14, 48.10), (24.30, 47.91), (25.41, 47.93),
    (26.40, 48.22), (27.05, 47.99), (27.55, 47.40), (28.10, 46.81),
    (28.21, 45.97), (28.83, 45.30), (29.65, 45.18), (29.69, 44.81),
    (28.84, 44.05), (28.05, 43.81), (27.00, 44.13), (25.65, 43.69),
    (24.50, 43.68), (23.27, 43.83), (22.65, 44.22), (22.42, 44.71),
    (21.56, 44.77), (21.36, 45.04), (20.79, 45.46), (20.25, 46.10),
    (20.79, 46.30), (21.06, 46.83), (22.13, 47.59), (22.69, 47.99),
]

_BORDERS_PIXEL_CACHE = None    # populated lazily on first plot
_BORDERS_SOURCE = None         # "natural_earth_10m" or "hardcoded_coarse"
_VIEW_EXTENT = None            # (col_lo, col_hi, row_lo, row_hi)


def _latlon_to_pixel(lon, lat):
    """Convert lon/lat arrays to (col, row) on the 768x1536 grid via
    pyproj (lat/lon -> EPSG:31700 -> pixel through the UTM extent)."""
    import pyproj
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:31700", always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xmin, xmax, ymin, ymax = ROMANIA_EXTENT_UTM
    col = (x - xmin) / (xmax - xmin) * GRID_WIDTH
    row = (ymax - y) / (ymax - ymin) * GRID_HEIGHT
    return col, row


def _load_country_borders():
    """Resolve Romania + neighbour-country borders to (col, row, name)
    tuples. Cartopy Natural Earth 10m if available, hardcoded fallback
    otherwise. Returns (rings, source_label)."""
    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth(
            resolution="10m", category="cultural",
            name="admin_0_countries",
        )
        wanted = NEIGHBOUR_NAMES | {"Romania"}
        rings = []
        for country in shpreader.Reader(shp).records():
            name = country.attributes.get("NAME") \
                or country.attributes.get("ADMIN") \
                or country.attributes.get("name")
            if name not in wanted:
                continue
            geom = country.geometry
            geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" \
                else [geom]
            for poly in geoms:
                lon = np.asarray(poly.exterior.coords.xy[0], dtype=np.float64)
                lat = np.asarray(poly.exterior.coords.xy[1], dtype=np.float64)
                col, row = _latlon_to_pixel(lon, lat)
                rings.append((col, row, name))
                for interior in poly.interiors:
                    lon_i = np.asarray(interior.coords.xy[0], dtype=np.float64)
                    lat_i = np.asarray(interior.coords.xy[1], dtype=np.float64)
                    c_i, r_i = _latlon_to_pixel(lon_i, lat_i)
                    rings.append((c_i, r_i, name))
        if rings:
            return rings, "natural_earth_10m"
    except Exception:
        pass

    lonlat = np.asarray(_ROMANIA_OUTLINE_LONLAT_FALLBACK, dtype=np.float64)
    col, row = _latlon_to_pixel(lonlat[:, 0], lonlat[:, 1])
    return [(col, row, "Romania")], "hardcoded_coarse"


def _ensure_borders_cached():
    global _BORDERS_PIXEL_CACHE, _BORDERS_SOURCE, _VIEW_EXTENT
    if _BORDERS_PIXEL_CACHE is None:
        _BORDERS_PIXEL_CACHE, _BORDERS_SOURCE = _load_country_borders()
    if _VIEW_EXTENT is None:
        ro_cols, ro_rows = [], []
        for col, row, name in _BORDERS_PIXEL_CACHE:
            if name == "Romania":
                ro_cols.append(col)
                ro_rows.append(row)
        if ro_cols:
            all_col = np.concatenate(ro_cols)
            all_row = np.concatenate(ro_rows)
            c_x = float((all_col.min() + all_col.max()) / 2)
            c_y = float((all_row.min() + all_row.max()) / 2)
        else:
            c_x, c_y = GRID_WIDTH / 2.0, GRID_HEIGHT / 2.0
        half_w = max(c_x - 0.0, GRID_WIDTH - c_x) + VIEW_EXTRA_PAD
        half_h = max(c_y - 0.0, GRID_HEIGHT - c_y) + VIEW_EXTRA_PAD
        _VIEW_EXTENT = (c_x - half_w, c_x + half_w,
                        c_y - half_h, c_y + half_h)


def _overlay_borders(ax, *, color="black", linewidth=1.3):
    """Solid country borders, Romania drawn last so it stays prominent."""
    _ensure_borders_cached()
    for col, row, name in _BORDERS_PIXEL_CACHE:
        if name == "Romania":
            continue
        ax.plot(col, row, color=color, linewidth=linewidth, zorder=5)
    for col, row, name in _BORDERS_PIXEL_CACHE:
        if name != "Romania":
            continue
        ax.plot(col, row, color=color, linewidth=linewidth, zorder=6)


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
# Reprojection
# =============================================================================

def reproject_to_romania(datamap, source_lats, source_lons, target_lats, target_lons):
    """
    Reproject radar data to the Romania 1536×768 grid using nearest-neighbor.

    Args:
        datamap: 2D source data
        source_lats, source_lons: 2D source coordinate grids
        target_lats, target_lons: 2D target coordinate grids

    Returns:
        np.ndarray: Reprojected data (768×1536), flipped vertically
    """
    source_geo = geometry.GridDefinition(lons=source_lons, lats=source_lats)
    target_geo = geometry.GridDefinition(lons=target_lons, lats=target_lats)

    reprojected = kd_tree.resample_nearest(
        source_geo, datamap, target_geo,
        radius_of_influence=5000,
        fill_value=0.0
    )

    # Flip vertically to match Romania grid orientation (same as test_nc.py)
    reprojected = np.flipud(reprojected)

    return reprojected


# =============================================================================
# DBSCAN → binary mask
# =============================================================================

def dbscan_binary_mask(datamap, threshold=None, eps=None, min_samples=None):
    """
    Run DBSCAN on the reprojected radar data and produce a binary mask.

    All pixels belonging to any cluster (label != -1) are set to 1.
    Everything else (noise + below threshold) is 0.

    Args:
        datamap: 2D array (768×1536) of reprojected RZC values
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

def plot_patch_grid(reprojected, binary_mask, active_patches, date_str, time_str,
                    output_dir, source='radar'):
    """
    Plot the EPSG:31700-reprojected RZC / OPERA data with the 6×3 patch grid overlay.

    Styled to match `visualize_gt_vs_pred.py`: Romania
    sits centred in the figure, neighbour-country borders frame the
    canvas, every patch slot is outlined with a dashed grid in black,
    and active patches get a red highlight on top.

    Args:
        reprojected: 2D array (768×1536) of reprojected rain-rate values
            (RZC mm/h when source='radar', OPERA instantaneous rain rate
            mm/h when source='opera').
        binary_mask: 2D array (768×1536) of DBSCAN binary mask
        active_patches: list of active patch numbers (1-indexed)
        date_str: 'YYYY-MM-DD'
        time_str: 'HH:MM'
        output_dir: directory to save the PNG
    """
    _ensure_borders_cached()
    c_lo, c_hi, r_lo, r_hi = _VIEW_EXTENT

    # Source-aware labels for the rain-rate panel. Both products are
    # KD-tree reprojected into the Romania EPSG:31700 grid in mm/h;
    # only the title + colorbar label change so the same plot reads
    # correctly for either track.
    if source == 'opera':
        field_title = 'OPERA instantaneous rain rate (reprojected)'
        field_cbar  = 'OPERA rain rate (mm/h)'
    else:
        field_title = 'RZC (reprojected)'
        field_cbar  = 'RZC (mm/h)'

    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             constrained_layout=True)

    active_set = set(active_patches)

    for ax_idx, (ax, data, title_suffix) in enumerate(zip(
        axes,
        [reprojected, binary_mask],
        [field_title, 'DBSCAN binary mask']
    )):
        if ax_idx == 0:
            im = ax.imshow(data, cmap='viridis', aspect='equal',
                           interpolation='nearest')
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=field_cbar)
        else:
            cmap = ListedColormap(['#e0e0e0', '#d32f2f'])
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='equal',
                           interpolation='nearest')

        # 1. Dashed grid in black over every patch slot (matches the
        #    prediction plotter's _plot_patch_grid).
        for p in range(1, N_PATCHES + 1):
            r0, _, c0, _ = get_patch_bounds(p)
            ax.add_patch(Rectangle(
                (c0, r0), PATCH_SIZE, PATCH_SIZE,
                linewidth=0.7, edgecolor='black',
                linestyle=(0, (1, 3)), facecolor='none',
                zorder=3,
            ))

        # 2. Active-patch highlight: red outline + light red fill on top.
        for p in active_patches:
            r0, _, c0, _ = get_patch_bounds(p)
            ax.add_patch(Rectangle(
                (c0, r0), PATCH_SIZE, PATCH_SIZE,
                linewidth=2.0, edgecolor='#f44336',
                facecolor='#f4433620', linestyle='-', zorder=4,
            ))

        # 3. Patch number labels (red+bold for active, white otherwise).
        for p in range(1, N_PATCHES + 1):
            r0, _, c0, _ = get_patch_bounds(p)
            is_active = p in active_set
            ax.text(
                c0 + 8, r0 + 22, str(p),
                color='#f44336' if is_active else 'white',
                fontsize=9, fontweight='bold' if is_active else 'normal',
                ha='left', va='top', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.55, edgecolor='none')
            )

        # 4. Country borders on top.
        try:
            _overlay_borders(ax)
        except Exception:
            pass

        ax.set_title(f'{title_suffix}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(c_lo, c_hi)
        ax.set_ylim(r_hi, r_lo)  # image y is flipped
        ax.set_aspect('equal')

    patches_str = ', '.join(str(p) for p in active_patches) if active_patches else 'none'
    fig.suptitle(
        f'{date_str}  {time_str} UTC — Active patches: [{patches_str}]',
        fontsize=13, fontweight='bold'
    )

    os.makedirs(output_dir, exist_ok=True)
    safe_time = time_str.replace(':', '')
    filename = f"patches_{date_str}_{safe_time}.png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path


def plot_binary_mask_selection(binary_mask, active_patches,
                               date_str, time_str, output_dir):
    """Single-panel view of the DBSCAN binary mask + the 18-patch
    "polygon" of candidate 256x256 tiles that feed the model.

    Separate from `plot_patch_grid` on purpose: that function is the
    2-panel operational figure (OPERA rain rate + binary mask, red-
    highlighted selection) and stays untouched. This one focuses on
    the SELECTION step:

      - Binary mask underneath (gray / red).
      - Orange dashed rectangles outlining EVERY one of the 18
        256x256 candidate patches — together they form the full
        6x3 polygon of positions the pipeline can pick from.
      - Each patch labelled with its 1..18 number (upper-left = 1,
        row-major) coloured green when the patch was selected by
        DBSCAN (`in active_patches`) and red when it was rejected.
        Matches the green/red numbering convention already used by
        the full-domain training/inference plots so a viewer switching
        between the two sets of figures reads them the same way.

    Args:
        binary_mask: (768, 1536) uint8 — the DBSCAN binary mask.
        active_patches: list of 1-indexed patch numbers selected by
            `identify_active_patches`.
        date_str: 'YYYY-MM-DD'
        time_str: 'HH:MM'
        output_dir: directory to save the PNG.

    Returns:
        Absolute path to the written PNG.
    """
    _ensure_borders_cached()
    c_lo, c_hi, r_lo, r_hi = _VIEW_EXTENT

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)

    # Binary mask: dry = light gray so the orange grid + red mask pixels
    # pop; DBSCAN cluster pixels stay in the operational red.
    mask_cmap = ListedColormap(['#e6e6e6', '#d32f2f'])
    ax.imshow(binary_mask, cmap=mask_cmap, vmin=0, vmax=1,
              aspect='equal', interpolation='nearest')

    active_set = set(active_patches)

    # Orange dashed grid — one rectangle per candidate 256x256 patch.
    # The 18 rectangles together outline the "selection polygon" the
    # pipeline picks from.
    for p in range(1, N_PATCHES + 1):
        r0, _, c0, _ = get_patch_bounds(p)
        ax.add_patch(Rectangle(
            (c0, r0), PATCH_SIZE, PATCH_SIZE,
            linewidth=1.4, edgecolor='#ff8c00',
            linestyle=(0, (4, 3)), facecolor='none',
            zorder=4,
        ))

    # Green (active) / red (inactive) numbering — same colour codes as
    # visualize_gt_vs_pred._plot_patch_grid so the two figure families
    # read identically.
    for p in range(1, N_PATCHES + 1):
        r0, _, c0, _ = get_patch_bounds(p)
        is_active = p in active_set
        ax.text(
            c0 + 6, r0 + 6, str(p),
            color='#1b7a1b' if is_active else '#c11515',
            fontsize=10, fontweight='bold',
            ha='left', va='top', zorder=6,
            bbox=dict(boxstyle='round,pad=0.18',
                      facecolor='white', alpha=0.85,
                      edgecolor='#888888', linewidth=0.4),
        )

    try:
        _overlay_borders(ax)
    except Exception:
        pass

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_hi, r_lo)  # image y is flipped
    ax.set_aspect('equal')
    ax.set_title('DBSCAN binary mask + 18-patch candidate polygon',
                 fontsize=11)

    n_active = len(active_patches)
    patches_str = ', '.join(str(p) for p in active_patches) if active_patches else 'none'
    fig.suptitle(
        f'Patch selection  |  {date_str}  {time_str} UTC  |  '
        f'active: {n_active}/{N_PATCHES}  → [{patches_str}]',
        fontsize=12, fontweight='bold',
    )

    os.makedirs(output_dir, exist_ok=True)
    safe_time = time_str.replace(':', '')
    filename = f"patches_{date_str}_{safe_time}_selection.png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path


def write_diagnostic_nc(reprojected, binary_mask, active_patches,
                        date_str, time_str, output_dir, source,
                        data_root):
    """
    Write a CF-compliant NetCDF mirroring `plot_patch_grid()`.

    Mirrors the `inspect_mtg.py --reprojected` pattern: load the shared
    Romania-grid lat/lon arrays from
    `<data_root>/reprojected_data/romania_grid_{lats,lons}.npy` and
    package the reprojected field + DBSCAN binary mask + per-patch
    active flag into a single dataset that opens cleanly in QGIS / any
    CF-aware GIS.

    Output layout: `<output_dir>/nc/patches_<date>_<HHMM>.nc`, kept in a
    sibling folder of the PNGs so the two artifacts pair up by filename.

    Args:
        reprojected: 2D array (768×1536) — the same array `plot_patch_grid`
            renders on the left.
        binary_mask: 2D array (768×1536) — the same DBSCAN mask the right
            subplot draws.
        active_patches: list of active patch numbers (1-indexed).
        date_str: 'YYYY-MM-DD'.
        time_str: 'HH:MM'.
        output_dir: matches the PNG `output_dir`; the `.nc` lands under
            `<output_dir>/nc/`.
        source: 'radar' or 'opera'; controls the variable name and
            long_name so the file is self-describing in GIS.
        data_root: project `our_data/` root, used to locate
            `reprojected_data/romania_grid_{lats,lons}.npy`.

    Returns the path to the written `.nc` file, or None if xarray is not
    installed or the grid coordinate files are missing.
    """
    try:
        import xarray as xr
    except ImportError:
        print("    [nc] xarray not installed — skipping .nc output")
        return None

    grid_dir = os.path.join(data_root, "reprojected_data")
    lats_path = os.path.join(grid_dir, "romania_grid_lats.npy")
    lons_path = os.path.join(grid_dir, "romania_grid_lons.npy")
    if not (os.path.isfile(lats_path) and os.path.isfile(lons_path)):
        print(f"    [nc] romania_grid_{{lats,lons}}.npy not found under "
              f"{grid_dir} — skipping .nc output")
        return None

    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # Per-pixel patch ID (0..18 — 0 = outside any patch, never happens
    # for this 1536x768 layout, but kept for safety). Useful in QGIS to
    # symbolise patches by ID directly without re-deriving the grid.
    patch_id_grid = np.zeros(reprojected.shape, dtype=np.int16)
    for p in range(1, N_PATCHES + 1):
        r0, r1, c0, c1 = get_patch_bounds(p)
        patch_id_grid[r0:r1, c0:c1] = p

    active_set = set(active_patches)
    is_active_grid = np.isin(patch_id_grid, list(active_set)).astype(np.int8)

    if source == "opera":
        data_var_name = "opera_rainfall_rate"
        long_name = "OPERA instantaneous rainfall rate (DBSCAN driver)"
        units = "mm/h"
    else:
        data_var_name = "RZC"
        long_name = "Reprojected RZC rain rate (DBSCAN driver)"
        units = "mm/h"

    ds = xr.Dataset(
        {
            data_var_name: (["y", "x"],
                            np.asarray(reprojected, dtype=np.float32)),
            "dbscan_mask": (["y", "x"],
                            np.asarray(binary_mask, dtype=np.int8)),
            "patch_id":    (["y", "x"], patch_id_grid),
            "active_patch": (["y", "x"], is_active_grid),
        },
        coords={
            "latitude":  (["y", "x"], lats),
            "longitude": (["y", "x"], lons),
        },
    )
    ds[data_var_name].attrs = {
        "long_name":   long_name,
        "units":       units,
        "coordinates": "latitude longitude",
        "grid_mapping": "crs",
    }
    ds["dbscan_mask"].attrs = {
        "long_name":   "DBSCAN cluster membership (1 = in any cluster)",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "background cluster",
        "coordinates": "latitude longitude",
        "grid_mapping": "crs",
    }
    ds["patch_id"].attrs = {
        "long_name":   "1-indexed patch number on the 6x3 grid (1..18)",
        "coordinates": "latitude longitude",
        "grid_mapping": "crs",
    }
    ds["active_patch"].attrs = {
        "long_name":   "1 if the pixel's patch is in `active_patches`",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "inactive_patch active_patch",
        "coordinates": "latitude longitude",
        "grid_mapping": "crs",
    }

    # CF grid_mapping. EPSG:31700 (Stereo70). QGIS reads this and lays
    # the raster out at the correct geographic location.
    ds["crs"] = xr.DataArray(np.int32(0))
    ds["crs"].attrs = {
        "grid_mapping_name": "oblique_stereographic",
        "EPSG":              31700,
        "comment":           "Romania Stereo70 / Dealul Piscului 1970",
    }

    ds.attrs = {
        "title":         f"COALITION-4 patch index diagnostic — "
                         f"{date_str} {time_str} UTC",
        "active_patches": ",".join(str(p) for p in active_patches)
                          if active_patches else "",
        "source":        f"identify_patches.py (--source {source})",
        "Conventions":   "CF-1.8",
    }

    nc_dir = os.path.join(output_dir, "nc")
    os.makedirs(nc_dir, exist_ok=True)
    safe_time = time_str.replace(":", "")
    nc_path = os.path.join(nc_dir,
                           f"patches_{date_str}_{safe_time}.nc")
    ds.to_netcdf(nc_path)
    ds.close()
    return nc_path


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
    Discover the reprojected OPERA rainfall-rate `.npy` files.

    Unlike the RZC path (which reads raw source NetCDFs and reprojects
    on the fly inside this script), the OPERA branch consumes the
    output of `reproject.py --opera`. The arrays are already on the
    Romania 1536×768 grid, so the DBSCAN step skips the per-file
    reprojection done for RZC.

    Scans: {data_root}/reprojected_data/opera_data/rainfall_rate/
                nc4_{date}-Romania_rainfall_rate/*.npy

    Returns:
        list[tuple]: Sorted list of (date_str, filepath) pairs.
    """
    opera_dir = os.path.join(
        data_root, 'reprojected_data', 'opera_data', 'rainfall_rate'
    )
    if not os.path.isdir(opera_dir):
        print(f"OPERA rainfall_rate dir not found: {opera_dir}")
        print("  Run `python reproject.py --opera` first.")
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
        read → reproject → DBSCAN → binary mask → identify patches.
    """
    date_str, time_str, iso_str = parse_radar_filename(filepath)
    datamap, src_lats, src_lons = read_radar_netcdf(filepath)
    reprojected = reproject_to_romania(datamap, src_lats, src_lons,
                                  target_lats, target_lons)
    binary_mask = dbscan_binary_mask(reprojected)
    active_patches = identify_active_patches(binary_mask)
    return date_str, time_str, iso_str, active_patches, reprojected, binary_mask


def process_single_opera_file(filepath):
    """
    Pipeline for one pre-reprojected OPERA rainfall-rate `.npy`:
        load → DBSCAN → binary mask → identify patches.

    The array is already on the Romania 1536×768 grid (output of
    `reproject.py --opera`), so no per-file reprojection is needed.
    """
    date_str, time_str, iso_str = parse_opera_filename(filepath)
    if date_str is None:
        return None
    reprojected = np.load(filepath)
    # NaN may appear for off-grid pixels; DBSCAN expects finite values.
    reprojected = np.nan_to_num(reprojected, nan=0.0)
    binary_mask = dbscan_binary_mask(reprojected)
    active_patches = identify_active_patches(binary_mask)
    return date_str, time_str, iso_str, active_patches, reprojected, binary_mask


def run_pipeline(data_root, output_dir, date_filter=None, save_plots=False,
                 source='radar', start_date=None, end_date=None):
    """
    Run the full patch identification pipeline.

    Args:
        data_root: Path to our_data directory
        output_dir: Where to save CSV + JSON
        date_filter: Optional YYYY-MM-DD to process a single date
        save_plots: If True, save a PNG for each active timestamp
        source: 'radar' (RZC source NetCDFs, reproject in-script) or
                'opera' (pre-reprojected OPERA rainfall_rate .npy files).
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

    # The RZC path needs target lat/lon for its per-file reprojection.
    # The OPERA path consumes pre-reprojected data, so target_lats/lons are unused.
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
            d, t, iso, active, reprojected, binary_mask = out
            results.append((d, t, iso, active))

            if active:
                patches_str = ','.join(str(p) for p in active)
                print(f"  [{i+1}/{total}] {d} {t} → patches: [{patches_str}]")

                # Save plot + companion .nc for active timestamps. The
                # .nc carries the same arrays the plot renders plus the
                # Romania-grid lat/lon coords + EPSG:31700 metadata, so
                # the same outputs can be inspected in GIS software
                # (open the .nc in QGIS to overlay on satellite basemap).
                if save_plots:
                    plot_patch_grid(
                        reprojected, binary_mask, active, d, t, plot_dir,
                        source=source,
                    )
                    # Sibling `_selection.png` file — binary mask +
                    # orange 6x3 candidate polygon + green/red patch
                    # numbering. Companion to plot_patch_grid, not a
                    # replacement (the 2-panel figure stays untouched).
                    plot_binary_mask_selection(
                        binary_mask, active, d, t, plot_dir,
                    )
                    write_diagnostic_nc(
                        reprojected, binary_mask, active,
                        d, t, plot_dir, source=source,
                        data_root=data_root,
                    )
            else:
                print(f"  [{i+1}/{total}] {d} {t} → no active patches")

        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR processing {filepath}: {e}")
            continue

    if not results:
        print("\nNo results produced.")
        return

    # Save outputs. When --date is set the run only processed one day,
    # so the master patch_index.csv would be overwritten with a single
    # day's worth of rows - clobbering the upstream activity record the
    # downstream pipeline (extract_patch_seq, data_statistics, ...)
    # depends on. Skip the master write in that case; the per-date PNGs
    # (and a date-suffixed CSV/JSON if save_plots is on) still land in
    # the per-date plot directory below.
    os.makedirs(output_dir, exist_ok=True)
    if date_filter is None:
        save_csv(results, output_dir)
        save_json(results, output_dir)
    else:
        print(f"\nSingle-date run (--date {date_filter}) — NOT overwriting "
              f"the master patch_index.csv / patch_index.json. Use a "
              f"--start/--end range (or drop --date) to refresh those.")

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
        print(f"  NetCDF saved       : {active_timesteps} .nc files in "
              f"{os.path.join(plot_dir, 'nc')}")


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
             "reprojected in-script) or 'opera' (pre-reprojected OPERA "
             "rainfall_rate .npy from reproject.py --opera). Same DBSCAN "
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
    