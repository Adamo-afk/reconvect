"""
inspect_mtg.py — Reconstruct .nc files and plot MTG data.

Combines a .npy data file with the grid constants to produce a
CF-compliant NetCDF (viewable in Panoply/QGIS) and/or a matplotlib
plot.

Works with both pipeline output (geostationary grid) and regridded
output (EPSG:31700 Romania grid).

Usage:
    # Plot raw pipeline output (geostationary grid)
    python inspect_mtg.py --raw \\
        --npy MTG/vis_06/nc4_2026-02-13-Romania_vis_06/nc4_2026-02-13-Romania_0930_vis_06.npy \\
        --constants MTG/mtg_constants.json

    # Plot regridded output (Romania EPSG:31700 grid)
    python inspect_mtg.py --regridded \\
        --npy regridded_data/satellite_data/MTG/vis_06/.../nc4_..._0930_vis_06.npy

    # Save .nc without plotting
    python inspect_mtg.py --raw --npy <path> --constants <path> --save_nc --no_plot

    # Both
    python inspect_mtg.py --regridded --npy <path> --save_nc
"""

import os
import re
import sys
import json
import argparse
import numpy as np

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray required. Install: pip install xarray")
    sys.exit(1)


# Channel resolution lookup
CHANNELS_1KM = {
    'vis_04', 'vis_05', 'vis_06', 'vis_08', 'vis_09',
    'nir_13', 'nir_16', 'nir_22',
}
CHANNELS_2KM = {
    'ir_38', 'wv_63', 'wv_73', 'ir_87', 'ir_97',
    'ir_105', 'ir_123', 'ir_133',
}


def guess_channel_from_filename(filename):
    """Extract the channel name from a pipeline-produced filename."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Pattern: nc4_YYYY-MM-DD-Romania_HHMM_<channel>
    parts = basename.split('_')
    if len(parts) >= 4:
        return parts[-1]
    # Fallback: check if any known channel is in the filename
    for ch in sorted(CHANNELS_1KM | CHANNELS_2KM):
        if ch in basename:
            return ch
    return None


def get_resolution(channel):
    """Return '1km' or '2km' for a given channel name."""
    if channel in CHANNELS_1KM:
        return '1km'
    elif channel in CHANNELS_2KM:
        return '2km'
    return '2km'  # default


# =============================================================================
# Raw pipeline output (geostationary grid)
# =============================================================================

def build_raw_nc(npy_path, constants_path, channel=None):
    """
    Reconstruct a CF-compliant NetCDF from a raw pipeline .npy file.

    Reads the geostationary grid constants from mtg_constants.json and
    combines them with the data array to produce a complete NetCDF with
    x_geos/y_geos coordinates and projection metadata.

    Args:
        npy_path (str): Path to the .npy data file.
        constants_path (str): Path to mtg_constants.json.
        channel (str): Channel name. Auto-detected from filename if None.

    Returns:
        xr.Dataset: Complete dataset ready for saving or plotting.
    """
    if channel is None:
        channel = guess_channel_from_filename(npy_path)
        if channel is None:
            raise ValueError(
                f"Cannot determine channel from filename: {npy_path}. "
                f"Pass --channel explicitly."
            )

    data = np.load(npy_path)
    res = get_resolution(channel)

    with open(constants_path, 'r') as f:
        constants = json.load(f)

    res_data = constants.get(res)
    if res_data is None:
        raise ValueError(
            f"No {res} entry in {constants_path}. "
            f"Available: {[k for k in constants if k != 'projection']}"
        )

    x_geos = np.array(res_data['x_geos'], dtype=np.float64)
    y_geos = np.array(res_data['y_geos'], dtype=np.float64)
    proj = constants['projection']

    ds = xr.Dataset(
        {channel: (['y', 'x'], data)},
        coords={
            'y': np.arange(data.shape[0]),
            'x': np.arange(data.shape[1]),
        },
    )
    ds['x_geos'] = ('x', x_geos)
    ds['x_geos'].attrs = {'units': 'rad', 'long_name': 'scanning angle (E-W)'}
    ds['y_geos'] = ('y', y_geos)
    ds['y_geos'].attrs = {'units': 'rad', 'long_name': 'scanning angle (N-S)'}

    ds['mtg_geos_projection'] = xr.DataArray(np.int32(0))
    ds['mtg_geos_projection'].attrs = proj

    return ds, channel


# =============================================================================
# Regridded output (EPSG:31700 Romania grid)
# =============================================================================

def build_regridded_nc(npy_path, regridded_base=None, channel=None):
    """
    Reconstruct a CF-compliant NetCDF from a regridded .npy file.

    Reads the Romania grid lat/lon arrays from romania_grid_lats.npy
    and romania_grid_lons.npy, and combines them with the data.

    Args:
        npy_path (str): Path to the regridded .npy data file.
        regridded_base (str): Directory containing romania_grid_*.npy.
            Auto-detected from npy_path if None.
        channel (str): Channel name. Auto-detected from filename if None.

    Returns:
        xr.Dataset: Complete dataset with lat/lon coordinates.
    """
    if channel is None:
        channel = guess_channel_from_filename(npy_path)
        if channel is None:
            raise ValueError(
                f"Cannot determine channel from filename: {npy_path}. "
                f"Pass --channel explicitly."
            )

    data = np.load(npy_path)

    # Find the grid coordinate files
    if regridded_base is None:
        # Walk up from the .npy path to find romania_grid_lats.npy
        search_dir = os.path.dirname(npy_path)
        for _ in range(5):
            lats_path = os.path.join(search_dir, 'romania_grid_lats.npy')
            if os.path.isfile(lats_path):
                regridded_base = search_dir
                break
            search_dir = os.path.dirname(search_dir)

    if regridded_base is None:
        raise FileNotFoundError(
            "Cannot find romania_grid_lats.npy. "
            "Pass --grid_dir explicitly."
        )

    lats = np.load(os.path.join(regridded_base, 'romania_grid_lats.npy'))
    lons = np.load(os.path.join(regridded_base, 'romania_grid_lons.npy'))

    ds = xr.Dataset(
        {channel: (['y', 'x'], np.asarray(data, dtype=np.float32))},
        coords={
            'latitude': (['y', 'x'], lats),
            'longitude': (['y', 'x'], lons),
        },
    )
    ds[channel].attrs['coordinates'] = 'latitude longitude'

    ds['crs'] = xr.DataArray(np.int32(0))
    ds['crs'].attrs = {
        'grid_mapping_name': 'oblique_stereographic',
        'EPSG': 31700,
        'comment': 'Romania Stereo70 / Dealul Piscului 1970',
    }

    return ds, channel


# =============================================================================
# Plotting
# =============================================================================

def plot_data(ds, channel, title=None, output_image=None):
    """
    Plot the data array with matplotlib.

    For regridded data (has latitude/longitude), plots on a map.
    For raw data (integer indices), plots as a 2D image.

    Args:
        ds (xr.Dataset): The dataset to plot.
        channel (str): Variable name to plot.
        title (str): Plot title. Auto-generated if None.
        output_image (str): Path to save the plot. Shows interactively
            if None.
    """
    import matplotlib.pyplot as plt

    data = ds[channel].values

    if title is None:
        basename = channel
        title = f"MTG FCI — {basename}"

    has_latlon = 'latitude' in ds.coords

    if has_latlon:
        # Regridded data — plot on lat/lon axes
        lats = ds['latitude'].values
        lons = ds['longitude'].values

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Mask fill values for better color scaling
        plot_data = np.where(
            (data == 0) | (data >= 65000) | np.isnan(data),
            np.nan, data
        )

        im = ax.pcolormesh(
            lons, lats, plot_data,
            shading='auto', cmap='viridis'
        )
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title + ' (EPSG:31700)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label=f'{channel} (effective radiance)')

    else:
        # Raw geostationary data — plot as 2D image
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        # Mask fill values
        plot_data = np.where(
            (data >= 65000) | np.isnan(data),
            np.nan, data
        )

        im = ax.imshow(
            plot_data, aspect='auto', cmap='viridis',
            origin='upper', interpolation='nearest'
        )
        ax.set_xlabel('x (pixel)')
        ax.set_ylabel('y (pixel)')
        ax.set_title(title + ' (geostationary)')
        plt.colorbar(im, ax=ax, label=f'{channel} (effective radiance)')

    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_image}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Inspect MTG FCI data: reconstruct .nc from .npy + constants '
            'and/or plot the data.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--raw', action='store_true',
        help='Input is raw pipeline output (geostationary grid)',
    )
    mode.add_argument(
        '--regridded', action='store_true',
        help='Input is regridded output (EPSG:31700 Romania grid)',
    )

    # Required
    parser.add_argument(
        '--npy', type=str, required=True,
        help='Path to the .npy data file',
    )

    # Optional
    parser.add_argument(
        '--constants', type=str, default=None,
        help='Path to mtg_constants.json (required for --raw)',
    )
    parser.add_argument(
        '--grid_dir', type=str, default=None,
        help='Directory containing romania_grid_lats/lons.npy '
             '(auto-detected for --regridded)',
    )
    parser.add_argument(
        '--channel', type=str, default=None,
        help='Channel name (auto-detected from filename)',
    )
    parser.add_argument(
        '--save_nc', action='store_true',
        help='Save a .nc file alongside the .npy',
    )
    parser.add_argument(
        '--save_png', type=str, default=None,
        help='Save plot to this PNG path (default: show interactively)',
    )
    parser.add_argument(
        '--no_plot', action='store_true',
        help='Skip plotting (only save .nc if --save_nc)',
    )

    args = parser.parse_args()

    # Validate
    if args.raw and args.constants is None:
        # Try to find constants in parent directories
        search = os.path.dirname(os.path.abspath(args.npy))
        for _ in range(5):
            candidate = os.path.join(search, 'mtg_constants.json')
            if os.path.isfile(candidate):
                args.constants = candidate
                break
            search = os.path.dirname(search)

        if args.constants is None:
            parser.error("--constants is required for --raw mode")

    # Build the dataset
    if args.raw:
        ds, channel = build_raw_nc(
            args.npy, args.constants, channel=args.channel
        )
    else:
        ds, channel = build_regridded_nc(
            args.npy, regridded_base=args.grid_dir, channel=args.channel
        )

    print(f"Channel  : {channel}")
    print(f"Shape    : {ds[channel].shape}")
    print(f"Data range: {float(np.nanmin(ds[channel])):.4f} — "
          f"{float(np.nanmax(ds[channel])):.4f}")

    # Save .nc
    if args.save_nc:
        nc_path = args.npy.replace('.npy', '.nc')
        ds.to_netcdf(nc_path)
        print(f"Saved .nc: {nc_path}")

    # Plot
    if not args.no_plot:
        # Build a nice title from the filename
        basename = os.path.splitext(os.path.basename(args.npy))[0]
        title = f"MTG FCI — {basename}"

        plot_data(ds, channel, title=title, output_image=args.save_png)

    ds.close()
    