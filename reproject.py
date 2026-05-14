# """
# COALITION-4 data reprojection pipeline (optimized with precomputed mappings).

# Regrids all products to the Romania 1536x768 EPSG:31700 grid and caches
# the results so the expensive reprojection runs only once per source file.

# Optimization: the KD-tree is built ONCE per source geometry (not per file).
# All files sharing the same source grid reuse the precomputed index mapping,
# reducing reprojection to a fast numpy array lookup.

# Products handled:
#     - Radar:     RZC, BZC, CZC, EZC-20, LZC, CPCH       → .npy
#     - MSG:       VIS006, IR_039, IR_108, WV_062, WV_073   → .npy
#     - MTG:       vis_06, ir_38, ir_105, wv_63, wv_73      → .npy
#     - Lightning: density, current, occurrence (already on grid)   → .npy
#     - NWCSAF:    ctth_alti, ctth_tempe, cmic_phase, cmic_cot     → .nc

# Input paths:
#     our_data/radar_data/{product}/nc4_{date}-Romania_{product}/*.nc
#     our_data/satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.nc
#     our_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.nc
#     our_data/satellite_data/MTG/coordinates/{lat,lon}_{1km,2km}.npy
#     our_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.nc
#     our_data/nwcsaf_data/{date}-Romania/*.nc

# Output paths:
#     our_data/reprojected_data/radar_data/{product}/nc4_{date}-Romania_{product}/*.npy
#     our_data/reprojected_data/satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.npy
#     our_data/reprojected_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
#     our_data/reprojected_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
#     our_data/reprojected_data/nwcsaf_data/{date}-Romania/*.nc

# Usage (run from F:\\nowcasting\\coalition4-rcnn):
#     python reproject_data.py --radar
#     python reproject_data.py --satellite MSG
#     python reproject_data.py --satellite MTG
#     python reproject_data.py --lightning
#     python reproject_data.py --nwcsaf
#     python reproject_data.py --all
#     python reproject_data.py --radar --date 2024-06-13
# """

# import numpy as np
# import xarray as xr
# import os
# import re
# import argparse
# import time as timer_module
# from pathlib import Path
# from datetime import datetime
# from netCDF4 import Dataset
# from pyresample import geometry, kd_tree

# from c4dl.projection import GridProjection, romania_grid_area


# # =============================================================================
# # Configuration
# # =============================================================================

# DEFAULT_DATA_ROOT = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'our_data'
# )

# RADAR_PRODUCTS = ['RZC', 'BZC', 'CZC', 'EZC-20', 'LZC', 'CPCH']

# MSG_CHANNELS = [
#     'VIS006', 'IR_039', 'IR_108', 'WV_062', 'WV_073'
# ]

# MTG_CHANNELS = [
#     'vis_06', 'ir_38', 'ir_105', 'wv_63', 'wv_73'
# ]

# MTG_1KM_CHANNELS = {'vis_06'}
# MTG_2KM_CHANNELS = {'ir_38', 'ir_105', 'wv_63', 'wv_73'}

# LIGHTNING_PRODUCTS = ['density', 'current', 'occurrence']

# QUARTER_HOUR_MINUTES = {'00', '15', '30', '45'}


# # =============================================================================
# # Precomputed mapping class
# # =============================================================================

# class PrecomputedMapping:
#     """
#     Precomputes and caches the nearest-neighbor index mapping between
#     a source geometry and the target Romania grid.

#     The expensive KD-tree build happens once in __init__. After that,
#     apply() is a fast numpy array operation.
#     """

#     def __init__(self, source_lats, source_lons, target_lats, target_lons,
#                  radius=5000, fill_value=0.0):
#         self.target_shape = target_lats.shape
#         self.fill_value = fill_value

#         source_geo = geometry.GridDefinition(
#             lons=source_lons, lats=source_lats
#         )
#         target_geo = geometry.GridDefinition(
#             lons=target_lons, lats=target_lats
#         )

#         t0 = timer_module.time()
#         (self.valid_input_index,
#          self.valid_output_index,
#          self.index_array,
#          self.distance_array) = kd_tree.get_neighbour_info(
#             source_geo, target_geo,
#             radius_of_influence=radius,
#             neighbours=1
#         )
#         elapsed = timer_module.time() - t0
#         print(f"    KD-tree built in {elapsed:.2f}s "
#               f"(source: {source_lats.shape}, target: {self.target_shape})")

#     def apply(self, source_data, fill_value=None):
#         """Apply the precomputed mapping to reproject source data (fast)."""
#         fv = fill_value if fill_value is not None else self.fill_value

#         reprojected = kd_tree.get_sample_from_neighbour_info(
#             'nn',
#             self.target_shape,
#             source_data,
#             self.valid_input_index,
#             self.valid_output_index,
#             self.index_array,
#             fill_value=fv
#         )
#         return reprojected


# # =============================================================================
# # Shared utilities
# # =============================================================================

# def init_romania_grid():
#     """Initialize Romania grid projection and return target coordinate grids."""
#     print("Initializing Romania grid projection...")
#     grid_projection = GridProjection(romania_grid_area)
#     y, x = np.mgrid[:grid_projection.area.height, :grid_projection.area.width]
#     target_lons, target_lats = grid_projection.inverse(y, x)
#     print(f"  Target grid shape: {target_lats.shape}")
#     return target_lats, target_lons


# def is_quarter_hour(filename):
#     """Check if a filename's timestamp is on the 15-minute grid."""
#     basename = os.path.splitext(os.path.basename(filename))[0]
#     parts = basename.split('_')
#     if len(parts) >= 3 and parts[0] == 'nc4':
#         time_str = parts[2]
#         if len(time_str) == 4 and time_str.isdigit():
#             return time_str[2:4] in QUARTER_HOUR_MINUTES
#     if len(basename) >= 12 and basename[:12].isdigit():
#         return basename[10:12] in QUARTER_HOUR_MINUTES
#     match = re.search(r'_(\d{8})_(\d{4})', basename)
#     if match:
#         return match.group(2)[2:4] in QUARTER_HOUR_MINUTES
#     return True


# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)


# def output_exists(path):
#     return os.path.isfile(path)


# def find_data_variable(ds, channel_name):
#     """Find the data variable name in a NetCDF dataset."""
#     for candidate in [channel_name, channel_name.upper(),
#                       channel_name.lower(), 'data', 'datamap']:
#         if candidate in ds.variables:
#             return candidate
#     coord_vars = {'latitude', 'longitude', 'lat', 'lon',
#                   'x', 'y', 'time', 'nx', 'ny'}
#     for v in ds.variables:
#         if v not in coord_vars and 'pal' not in v:
#             return v
#     return None


# # =============================================================================
# # Radar
# # =============================================================================

# def get_preferred_radar_group(ds):
#     data_group = ds.groups['data']
#     if 'radarpicture_0' in data_group.groups:
#         return 'radarpicture_0'
#     elif 'radarpicture' in data_group.groups:
#         return 'radarpicture'
#     else:
#         raise ValueError(
#             f"No radarpicture group found. "
#             f"Available: {list(data_group.groups.keys())}"
#         )


# def _read_radar_source_grid(filepath):
#     """Read source coordinate grids from a radar NetCDF file."""
#     with Dataset(filepath, 'r') as ds:
#         rp = get_preferred_radar_group(ds)
#         proj = ds.groups['data'].groups[rp].groups['projection']
#         lats = np.linspace(
#             float(proj.getncattr('lat_ul')),
#             float(proj.getncattr('lat_lr')),
#             int(proj.getncattr('size_y'))
#         )
#         lons = np.linspace(
#             float(proj.getncattr('lon_ul')),
#             float(proj.getncattr('lon_lr')),
#             int(proj.getncattr('size_x'))
#         )
#         lon_grid, lat_grid = np.meshgrid(lons, lats)
#     return lat_grid, lon_grid


# def _read_radar_data(filepath):
#     """Read the datamap from a radar NetCDF file."""
#     with Dataset(filepath, 'r') as ds:
#         rp = get_preferred_radar_group(ds)
#         datamap = ds.groups['data'].groups[rp].groups['datamap'].variables['datamap'][:]
#     return datamap


# def reproject_radar(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Reproject all radar products. KD-tree built once, reused for all files.
#     """
#     radar_dir = os.path.join(data_root, 'radar_data')
#     reprojected_base = os.path.join(data_root, 'reprojected_data', 'radar_data')
#     mapping = None

#     for product in RADAR_PRODUCTS:
#         product_dir = os.path.join(radar_dir, product)
#         if not os.path.isdir(product_dir):
#             continue

#         print(f"\n  Product: {product}")

#         for day_folder in sorted(os.listdir(product_dir)):
#             if date_filter and date_filter not in day_folder:
#                 continue
#             day_path = os.path.join(product_dir, day_folder)
#             if not os.path.isdir(day_path):
#                 continue

#             out_dir = os.path.join(reprojected_base, product, day_folder)
#             nc_files = sorted(
#                 f for f in os.listdir(day_path) if f.endswith('.nc')
#             )

#             new, skipped = 0, 0
#             for nc_file in nc_files:
#                 npy_file = nc_file.replace('.nc', '.npy')
#                 out_path = os.path.join(out_dir, npy_file)
#                 if output_exists(out_path):
#                     skipped += 1
#                     continue

#                 filepath = os.path.join(day_path, nc_file)
#                 try:
#                     if mapping is None:
#                         print(f"    Building radar mapping from {nc_file}...")
#                         src_lats, src_lons = _read_radar_source_grid(filepath)
#                         mapping = PrecomputedMapping(
#                             src_lats, src_lons, target_lats, target_lons
#                         )

#                     datamap = _read_radar_data(filepath)
#                     reprojected = mapping.apply(datamap)
#                     reprojected = np.flipud(reprojected)

#                     ensure_dir(out_dir)
#                     np.save(out_path, reprojected)
#                     new += 1
#                 except Exception as e:
#                     print(f"    ERROR {nc_file}: {e}")

#             total = new + skipped
#             if total > 0:
#                 print(f"    {day_folder}: {new} new, {skipped} cached, "
#                       f"{total} total")


# # =============================================================================
# # MSG satellite
# # =============================================================================

# def reproject_satellite_msg(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Reproject MSG channels. KD-tree built once per channel, reused for all files.
#     """
#     msg_dir = os.path.join(data_root, 'satellite_data', 'MSG')
#     reprojected_base = os.path.join(
#         data_root, 'reprojected_data', 'satellite_data', 'MSG'
#     )

#     for channel in MSG_CHANNELS:
#         channel_dir = os.path.join(msg_dir, channel)
#         if not os.path.isdir(channel_dir):
#             continue

#         print(f"\n  Channel: {channel}")
#         mapping = None

#         for day_folder in sorted(os.listdir(channel_dir)):
#             if date_filter and date_filter not in day_folder:
#                 continue
#             day_path = os.path.join(channel_dir, day_folder)
#             if not os.path.isdir(day_path):
#                 continue

#             out_dir = os.path.join(reprojected_base, channel, day_folder)
#             nc_files = sorted(
#                 f for f in os.listdir(day_path) if f.endswith('.nc')
#             )

#             new, skipped = 0, 0
#             for nc_file in nc_files:
#                 npy_file = nc_file.replace('.nc', '.npy')
#                 out_path = os.path.join(out_dir, npy_file)
#                 if output_exists(out_path):
#                     skipped += 1
#                     continue

#                 filepath = os.path.join(day_path, nc_file)
#                 try:
#                     with Dataset(filepath, 'r') as ds:
#                         if mapping is None:
#                             print(f"    Building MSG {channel} mapping "
#                                   f"from {nc_file}...")
#                             lat_grid = ds.variables['latitude'][:]
#                             lon_grid = ds.variables['longitude'][:]
#                             mapping = PrecomputedMapping(
#                                 lat_grid, lon_grid,
#                                 target_lats, target_lons
#                             )

#                         var_name = find_data_variable(ds, channel)
#                         if var_name is None:
#                             print(f"    WARNING: no data var in {nc_file}")
#                             continue
#                         sat_data = ds.variables[var_name][:]

#                     reprojected = mapping.apply(sat_data)
#                     ensure_dir(out_dir)
#                     np.save(out_path, reprojected)
#                     new += 1
#                 except Exception as e:
#                     print(f"    ERROR {nc_file}: {e}")

#             total = new + skipped
#             if total > 0:
#                 print(f"    {day_folder}: {new} new, {skipped} cached, "
#                       f"{total} total")


# # =============================================================================
# # MTG satellite
# # =============================================================================

# def reproject_satellite_mtg(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Reproject MTG channels. KD-tree built once per resolution (1km/2km),
#     reused for all channels sharing that resolution.
#     """
#     mtg_dir = os.path.join(data_root, 'satellite_data', 'MTG')
#     coord_dir = os.path.join(mtg_dir, 'coordinates')
#     reprojected_base = os.path.join(
#         data_root, 'reprojected_data', 'satellite_data', 'MTG'
#     )

#     # Precompute mappings per resolution
#     mapping_cache = {}
#     for res in ['1km', '2km']:
#         lat_path = os.path.join(coord_dir, f'lat_{res}.npy')
#         lon_path = os.path.join(coord_dir, f'lon_{res}.npy')
#         if os.path.isfile(lat_path) and os.path.isfile(lon_path):
#             src_lats = np.load(lat_path)
#             src_lons = np.load(lon_path)
#             print(f"  Building MTG {res} mapping...")
#             mapping_cache[res] = PrecomputedMapping(
#                 src_lats, src_lons, target_lats, target_lons
#             )
#         else:
#             print(f"  WARNING: MTG {res} coordinates not found at {coord_dir}")

#     for channel in MTG_CHANNELS:
#         channel_dir = os.path.join(mtg_dir, channel)
#         if not os.path.isdir(channel_dir):
#             continue

#         res = '1km' if channel in MTG_1KM_CHANNELS else '2km'
#         if res not in mapping_cache:
#             print(f"  Skipping {channel}: no {res} mapping")
#             continue

#         mapping = mapping_cache[res]
#         print(f"\n  Channel: {channel} ({res}, mapping reused)")

#         for day_folder in sorted(os.listdir(channel_dir)):
#             if date_filter and date_filter not in day_folder:
#                 continue
#             day_path = os.path.join(channel_dir, day_folder)
#             if not os.path.isdir(day_path):
#                 continue

#             out_dir = os.path.join(reprojected_base, channel, day_folder)
#             nc_files = sorted(
#                 f for f in os.listdir(day_path) if f.endswith('.nc')
#             )

#             new, skipped, filtered = 0, 0, 0
#             for nc_file in nc_files:
#                 if not is_quarter_hour(nc_file):
#                     filtered += 1
#                     continue

#                 npy_file = nc_file.replace('.nc', '.npy')
#                 out_path = os.path.join(out_dir, npy_file)
#                 if output_exists(out_path):
#                     skipped += 1
#                     continue

#                 filepath = os.path.join(day_path, nc_file)
#                 try:
#                     with Dataset(filepath, 'r') as ds:
#                         var_name = find_data_variable(ds, channel)
#                         if var_name is None:
#                             print(f"    WARNING: no data var in {nc_file}")
#                             continue
#                         sat_data = ds.variables[var_name][:]

#                     reprojected = mapping.apply(sat_data)
#                     ensure_dir(out_dir)
#                     np.save(out_path, reprojected)
#                     new += 1
#                 except Exception as e:
#                     print(f"    ERROR {nc_file}: {e}")

#             total = new + skipped
#             if total > 0 or filtered > 0:
#                 print(f"    {day_folder}: {new} new, {skipped} cached, "
#                       f"{filtered} filtered, {total} used")


# # =============================================================================
# # Lightning (no reprojection)
# # =============================================================================

# def reproject_lightning(data_root, date_filter=None):
#     """Cache lightning NetCDF data as .npy (already on grid)."""
#     lightning_dir = os.path.join(data_root, 'lightning_data')
#     reprojected_base = os.path.join(
#         data_root, 'reprojected_data', 'lightning_data'
#     )

#     for product in LIGHTNING_PRODUCTS:
#         product_dir = os.path.join(lightning_dir, product)
#         if not os.path.isdir(product_dir):
#             continue

#         print(f"\n  Product: {product}")

#         for day_folder in sorted(os.listdir(product_dir)):
#             if date_filter and date_filter not in day_folder:
#                 continue
#             day_path = os.path.join(product_dir, day_folder)
#             if not os.path.isdir(day_path):
#                 continue

#             out_dir = os.path.join(reprojected_base, product, day_folder)
#             nc_files = sorted(
#                 f for f in os.listdir(day_path) if f.endswith('.nc')
#             )

#             new, skipped = 0, 0
#             for nc_file in nc_files:
#                 npy_file = nc_file.replace('.nc', '.npy')
#                 out_path = os.path.join(out_dir, npy_file)
#                 if output_exists(out_path):
#                     skipped += 1
#                     continue

#                 try:
#                     filepath = os.path.join(day_path, nc_file)
#                     with Dataset(filepath, 'r') as ds:
#                         datamap = ds.variables['datamap'][:]

#                     if isinstance(datamap, np.ma.MaskedArray):
#                         datamap = datamap.filled(0.0)
#                     if datamap.ndim == 3:
#                         datamap = np.squeeze(datamap, axis=0)

#                     ensure_dir(out_dir)
#                     np.save(out_path, datamap.astype(np.float32))
#                     new += 1
#                 except Exception as e:
#                     print(f"    ERROR {nc_file}: {e}")

#             total = new + skipped
#             if total > 0:
#                 print(f"    {day_folder}: {new} new, {skipped} cached, "
#                       f"{total} total")


# # =============================================================================
# # NWCSAF (saves as .nc)
# # =============================================================================

# def reproject_nwcsaf(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Reproject NWCSAF files. KD-tree built once from first file, reused for all.
#     """
#     nwcsaf_dir = os.path.join(data_root, 'nwcsaf_data')
#     reprojected_base = os.path.join(data_root, 'reprojected_data', 'nwcsaf_data')

#     if not os.path.isdir(nwcsaf_dir):
#         print("  NWCSAF directory not found")
#         return

#     mapping = None

#     for day_folder in sorted(os.listdir(nwcsaf_dir)):
#         if date_filter and date_filter not in day_folder:
#             continue
#         day_path = os.path.join(nwcsaf_dir, day_folder)
#         if not os.path.isdir(day_path):
#             continue

#         out_dir = os.path.join(reprojected_base, day_folder)
#         nc_files = sorted(
#             f for f in os.listdir(day_path) if f.endswith('.nc')
#         )

#         print(f"\n  Folder: {day_folder} ({len(nc_files)} files)")

#         new, skipped = 0, 0
#         for nc_file in nc_files:
#             out_path = os.path.join(out_dir, nc_file)
#             if output_exists(out_path):
#                 skipped += 1
#                 continue

#             filepath = os.path.join(day_path, nc_file)
#             try:
#                 with Dataset(filepath, 'r') as ds:
#                     if mapping is None:
#                         print(f"    Building NWCSAF mapping from {nc_file}...")
#                         lat_grid = np.asarray(
#                             ds.variables['lat'][:], dtype=np.float64
#                         )
#                         lon_grid = np.asarray(
#                             ds.variables['lon'][:], dtype=np.float64
#                         )
#                         if isinstance(lat_grid, np.ma.MaskedArray):
#                             lat_grid = lat_grid.filled(0.0)
#                         if isinstance(lon_grid, np.ma.MaskedArray):
#                             lon_grid = lon_grid.filled(0.0)
#                         lat_grid = np.nan_to_num(lat_grid, nan=0.0)
#                         lon_grid = np.nan_to_num(lon_grid, nan=0.0)
#                         lat_grid = np.clip(lat_grid, -90.0, 90.0)
#                         lon_grid = np.clip(lon_grid, -180.0, 180.0)

#                         mapping = PrecomputedMapping(
#                             lat_grid, lon_grid,
#                             target_lats, target_lons
#                         )

#                     # Identify data variables
#                     exclude = {'lat', 'lon', 'nx', 'ny', 'time'}
#                     exclude |= {v for v in ds.variables if 'pal' in v}
#                     var_names = [v for v in ds.variables if v not in exclude]

#                     reprojected_dict = {}
#                     for var_name in var_names:
#                         data = ds.variables[var_name][:]

#                         if data.ndim == 3:
#                             data = np.squeeze(data, axis=0)
#                         elif data.ndim == 1:
#                             if isinstance(data, np.ma.MaskedArray):
#                                 data = data.filled(0.0)
#                             reprojected_dict[var_name] = np.asarray(
#                                 data, dtype=np.float32
#                             )
#                             continue
#                         elif data.ndim != 2:
#                             continue

#                         if isinstance(data, np.ma.MaskedArray):
#                             fill_val = getattr(
#                                 data, 'fill_value',
#                                 np.float32(-3.4028235e+38)
#                             )
#                             data = data.filled(fill_val)

#                         data = np.asarray(data, dtype=np.float32)
#                         data = np.nan_to_num(
#                             data, nan=0.0, posinf=1e6, neginf=-1e6
#                         )

#                         reprojected = mapping.apply(data)
#                         reprojected_dict[var_name] = reprojected

#                 if reprojected_dict:
#                     _save_nwcsaf_nc(
#                         reprojected_dict, target_lats, target_lons,
#                         out_dir, nc_file
#                     )
#                     new += 1

#             except Exception as e:
#                 print(f"    ERROR {nc_file}: {e}")

#         total = new + skipped
#         if total > 0:
#             print(f"    {new} new, {skipped} cached, {total} total")


# def _save_nwcsaf_nc(data_dict, lats, lons, out_dir, filename):
#     """Save reprojected NWCSAF data as a single NetCDF file."""
#     ensure_dir(out_dir)
#     out_path = os.path.join(out_dir, filename)

#     data_vars = {}
#     for var_name, data in data_dict.items():
#         if data.ndim == 2:
#             data_vars[var_name] = (['y', 'x'], data)
#         elif data.ndim == 1:
#             data_vars[var_name] = ([f'{var_name}_dim'], data)

#     ds = xr.Dataset(
#         data_vars,
#         coords={
#             'latitude': (['y', 'x'], lats),
#             'longitude': (['y', 'x'], lons)
#         }
#     )
#     ds.to_netcdf(out_path)


# # =============================================================================
# # Main pipeline
# # =============================================================================

# def run(data_root, mode, instrument=None, date_filter=None):
#     print("=" * 70)
#     print("COALITION-4 Data Reprojection Pipeline (precomputed mappings)")
#     print("=" * 70)
#     print(f"Data root : {data_root}")
#     print(f"Mode      : {mode}" + (f" ({instrument})" if instrument else ""))
#     if date_filter:
#         print(f"Date      : {date_filter}")

#     t_start = timer_module.time()

#     needs_grid = mode in ('radar', 'satellite', 'nwcsaf', 'all')
#     if needs_grid:
#         target_lats, target_lons = init_romania_grid()
#     else:
#         target_lats = target_lons = None

#     if mode in ('radar', 'all'):
#         print(f"\n{'='*70}")
#         print("Radar products")
#         print(f"{'='*70}")
#         reproject_radar(data_root, target_lats, target_lons, date_filter)

#     if mode == 'satellite' and instrument == 'MSG' or mode == 'all':
#         print(f"\n{'='*70}")
#         print("MSG satellite channels")
#         print(f"{'='*70}")
#         reproject_satellite_msg(data_root, target_lats, target_lons, date_filter)

#     if mode == 'satellite' and instrument == 'MTG' or mode == 'all':
#         print(f"\n{'='*70}")
#         print("MTG satellite channels")
#         print(f"{'='*70}")
#         if target_lats is None:
#             target_lats, target_lons = init_romania_grid()
#         reproject_satellite_mtg(data_root, target_lats, target_lons, date_filter)

#     if mode in ('lightning', 'all'):
#         print(f"\n{'='*70}")
#         print("Lightning products")
#         print(f"{'='*70}")
#         reproject_lightning(data_root, date_filter)

#     if mode in ('nwcsaf', 'all'):
#         print(f"\n{'='*70}")
#         print("NWCSAF products")
#         print(f"{'='*70}")
#         reproject_nwcsaf(data_root, target_lats, target_lons, date_filter)

#     elapsed = timer_module.time() - t_start
#     print(f"\n{'='*70}")
#     print(f"Done in {elapsed:.1f}s.")
#     print(f"{'='*70}")


# # =============================================================================
# # CLI
# # =============================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="COALITION-4 data reprojection pipeline. "
#                     "Uses precomputed KD-tree mappings for speed."
#     )
#     parser.add_argument(
#         "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
#         help="Path to our_data directory"
#     )
#     parser.add_argument(
#         "--date", type=str, default=None,
#         help="Process a single date (YYYY-MM-DD)"
#     )

#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument("--radar", action="store_true",
#                        help="Reproject radar products")
#     group.add_argument("--satellite", type=str, choices=['MSG', 'MTG'],
#                        metavar='INSTRUMENT', help="Reproject satellite channels")
#     group.add_argument("--lightning", action="store_true",
#                        help="Cache lightning data as .npy")
#     group.add_argument("--nwcsaf", action="store_true",
#                        help="Reproject NWCSAF products")
#     group.add_argument("--all", action="store_true",
#                        help="Reproject all products")

#     args = parser.parse_args()

#     if args.radar:
#         mode, instrument = 'radar', None
#     elif args.satellite:
#         mode, instrument = 'satellite', args.satellite
#     elif args.lightning:
#         mode, instrument = 'lightning', None
#     elif args.nwcsaf:
#         mode, instrument = 'nwcsaf', None
#     elif args.all:
#         mode, instrument = 'all', None

#     run(
#         data_root=args.data_root,
#         mode=mode,
#         instrument=instrument,
#         date_filter=args.date,
#     )

"""
COALITION-4 data reprojection pipeline (optimized with precomputed mappings).

Regrids all products to the Romania 1536x768 EPSG:31700 grid and caches
the results so the expensive reprojection runs only once per source file.

Optimization: the KD-tree is built ONCE per source geometry (not per file).
All files sharing the same source grid reuse the precomputed index mapping,
reducing reprojection to a fast numpy array lookup.

Products handled — every family writes `.npy`. The Romania grid coordinates
(`romania_grid_lats.npy`, `romania_grid_lons.npy`) and per-source projection
constants (`{mtg,nwcsaf,opera}_constants.json`) are written **once** as
sidecars so the reprojected arrays remain self-recoverable for inspection.

    - Radar:     RZC, BZC, CZC, EZC-20, LZC, CPCH       → .npy
    - MSG:       VIS006, IR_039, IR_108, WV_062, WV_073 → .npy   (disabled)
    - MTG:       vis_06, ir_38, ir_105, wv_63, wv_73    → .npy
    - Lightning: density, current, occurrence (already on grid) → .npy
    - NWCSAF:    ctth_alti, ctth_tempe, cmic_phase (int8), cmic_cot → .npy
    - OPERA:     reflectivity, rainfall_rate            → .npy

Input paths:
    our_data/radar_data/{product}/nc4_{date}-Romania_{product}/*.nc
    our_data/satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.nc
    our_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
    our_data/satellite_data/MTG/mtg_constants.json
    our_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.nc
    our_data/nwcsaf_data/{date}-Romania/*.nc
    our_data/opera_data/{reflectivity|rainfall_rate}/{YYYY}/{MM}/{DD}/*.h5

Output paths:
    our_data/reprojected_data/romania_grid_{lats,lons}.npy             (shared)
    our_data/reprojected_data/radar_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/reprojected_data/satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.npy
    our_data/reprojected_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
    our_data/reprojected_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/reprojected_data/nwcsaf_data/{var}/nc4_{date}-Romania_{var}/*.npy
    our_data/reprojected_data/nwcsaf_data/nwcsaf_constants.json
    our_data/reprojected_data/opera_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/reprojected_data/opera_data/opera_constants.json

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python reproject_data.py --radar
    python reproject_data.py --satellite MSG
    python reproject_data.py --satellite MTG
    python reproject_data.py --lightning
    python reproject_data.py --nwcsaf
    python reproject_data.py --all
    python reproject_data.py --radar --date 2024-06-13
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

RADAR_PRODUCTS = ['RZC', 'BZC', 'CZC', 'EZC-20', 'LZC', 'CPCH']

MSG_CHANNELS = [
    'VIS006', 'IR_039', 'IR_108', 'WV_062', 'WV_073'
]

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

def get_preferred_radar_group(ds):
    data_group = ds.groups['data']
    if 'radarpicture_0' in data_group.groups:
        return 'radarpicture_0'
    elif 'radarpicture' in data_group.groups:
        return 'radarpicture'
    else:
        raise ValueError(
            f"No radarpicture group found. "
            f"Available: {list(data_group.groups.keys())}"
        )


def _read_radar_source_grid(filepath):
    """Read source coordinate grids from a radar NetCDF file."""
    with _nc_lock:
        with Dataset(filepath, 'r') as ds:
            rp = get_preferred_radar_group(ds)
            proj = ds.groups['data'].groups[rp].groups['projection']
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
    return lat_grid, lon_grid


def _read_radar_data(filepath):
    """Read the datamap from a radar NetCDF file."""
    with _nc_lock:
        with Dataset(filepath, 'r') as ds:
            rp = get_preferred_radar_group(ds)
            datamap = ds.groups['data'].groups[rp].groups['datamap'].variables['datamap'][:]
    return datamap


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


def _write_reproject_log(reprojected_root, category, all_errors):
    """Write a single-line-per-error log named after the reproject category.

    Format matches `errors.txt`:
        ERROR <filename>: <message>

    The log is written under `reprojected_root` (typically
    `our_data/reprojected_data/`) so it sits next to the reprojected outputs.
    Always overwrites — the log reflects the state of the most recent
    run, not a cumulative history. If no errors occurred, the log is
    still written but empty so downstream consumers can `open()` it
    unconditionally.
    """
    ensure_dir(reprojected_root)
    log_path = os.path.join(reprojected_root, f"reproject_{category}.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        for fname, msg in all_errors:
            f.write(f"ERROR {fname}: {msg}\n")
    n = len(all_errors)
    if n > 0:
        print(f"    Wrote {n} error line(s) to {log_path}")
    return log_path


def _init_worker(state):
    """Per-process initializer — populates `_WORKER_STATE` with the
    mapping plus any other per-batch constants (channel, product, base
    output directory, ...) shared across all jobs in a single pool. The
    state dict is passed as a single positional argument because
    `ProcessPoolExecutor.initargs` is positional-only."""
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _radar_day_worker(job):
    mapping = _WORKER_STATE['mapping']
    day_folder, day_path, out_dir, nc_files = job
    new, skipped = 0, 0
    errors: list[tuple[str, str]] = []
    for nc_file in nc_files:
        npy_file = nc_file.replace('.nc', '.npy')
        out_path = os.path.join(out_dir, npy_file)
        if output_exists(out_path):
            skipped += 1
            continue
        try:
            filepath = os.path.join(day_path, nc_file)
            datamap = _read_radar_data(filepath)
            reprojected = mapping.apply(datamap)
            reprojected = np.flipud(reprojected)
            ensure_dir(out_dir)
            np.save(out_path, reprojected)
            new += 1
        except Exception as e:
            errors.append((nc_file, str(e)))
    return day_folder, new, skipped, errors


def _msg_day_worker(job):
    mapping = _WORKER_STATE['mapping']
    channel = _WORKER_STATE['channel']
    day_folder, day_path, out_dir, nc_files = job
    new, skipped = 0, 0
    errors: list[tuple[str, str]] = []
    for nc_file in nc_files:
        npy_file = nc_file.replace('.nc', '.npy')
        out_path = os.path.join(out_dir, npy_file)
        if output_exists(out_path):
            skipped += 1
            continue
        try:
            filepath = os.path.join(day_path, nc_file)
            with Dataset(filepath, 'r') as ds:
                var_name = find_data_variable(ds, channel)
                if var_name is None:
                    continue
                sat_data = ds.variables[var_name][:]
            reprojected = mapping.apply(sat_data)
            ensure_dir(out_dir)
            np.save(out_path, reprojected)
            new += 1
        except Exception as e:
            errors.append((nc_file, str(e)))
    return day_folder, new, skipped, errors


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
    day_folder, day_path, out_dir, nc_files = job
    new, skipped = 0, 0
    errors: list[tuple[str, str]] = []
    for nc_file in nc_files:
        npy_file = nc_file.replace('.nc', '.npy')
        out_path = os.path.join(out_dir, npy_file)
        if output_exists(out_path):
            skipped += 1
            continue
        try:
            filepath = os.path.join(day_path, nc_file)
            with Dataset(filepath, 'r') as ds:
                datamap = ds.variables['datamap'][:]
            if isinstance(datamap, np.ma.MaskedArray):
                datamap = datamap.filled(0.0)
            if datamap.ndim == 3:
                datamap = np.squeeze(datamap, axis=0)
            ensure_dir(out_dir)
            np.save(out_path, datamap.astype(np.float32))
            new += 1
        except Exception as e:
            errors.append((nc_file, str(e)))
    return day_folder, new, skipped, errors


def _nwcsaf_day_worker(job):
    mapping = _WORKER_STATE['mapping']
    reprojected_base = _WORKER_STATE['reprojected_base']
    day_folder, day_path, nc_files = job
    new, skipped = 0, 0
    errors: list[tuple[str, str]] = []
    for nc_file in nc_files:
        date_str, hhmm, product = _parse_nwcsaf_filename(nc_file)
        if product is None:
            continue
        vars_in_file = [
            v for v, spec in NWCSAF_VAR_SPEC.items()
            if spec["product"] == product
        ]
        if not vars_in_file:
            continue

        all_outputs = []
        for var in vars_in_file:
            out_dir = os.path.join(
                reprojected_base, var, f"nc4_{date_str}-Romania_{var}"
            )
            out_name = f"nc4_{date_str}-Romania_{hhmm}_{var}.npy"
            all_outputs.append((var, out_dir, os.path.join(out_dir, out_name)))
        if all(os.path.exists(p) for _, _, p in all_outputs):
            skipped += 1
            continue

        try:
            filepath = os.path.join(day_path, nc_file)
            with Dataset(filepath, 'r') as ds:
                raw_data = {
                    var: ds.variables[var][:] for var in vars_in_file
                    if var in ds.variables
                }

            for var, out_dir, out_path in all_outputs:
                if var not in raw_data:
                    continue
                if os.path.exists(out_path):
                    continue
                data = raw_data[var]
                # Cast to float32 BEFORE filling the mask. NWCSAF stores
                # cmic_phase as a uint8 MaskedArray; calling .filled(np.nan)
                # on the uint8 view raises "Cannot convert fill_value nan
                # to dtype uint8". Casting first promotes both the data
                # and the mask's fill-value to a NaN-capable dtype.
                if isinstance(data, np.ma.MaskedArray):
                    data = data.astype(np.float32).filled(np.nan)
                else:
                    data = np.asarray(data, dtype=np.float32)
                if data.ndim == 3:
                    data = np.squeeze(data, axis=0)
                if data.ndim != 2:
                    continue
                dtype = NWCSAF_VAR_SPEC[var]["dtype"]
                if dtype == np.int8:
                    data = np.nan_to_num(data, nan=0.0)
                    reprojected = mapping.apply(data, fill_value=0)
                    reprojected = np.round(reprojected).astype(np.int8)
                else:
                    data = np.nan_to_num(
                        data, nan=0.0, posinf=1e6, neginf=-1e6,
                    )
                    reprojected = mapping.apply(data, fill_value=np.nan)
                    reprojected = reprojected.astype(np.float32)
                ensure_dir(out_dir)
                np.save(out_path, reprojected)
            new += 1
        except Exception as e:
            errors.append((nc_file, str(e)))
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


def reproject_radar(data_root, target_lats, target_lons, date_filter=None):
    """
    Reproject all radar products. KD-tree built once, day folders in parallel.
    """
    radar_dir = os.path.join(data_root, 'radar_data')
    reprojected_base = os.path.join(data_root, 'reprojected_data', 'radar_data')

    if not os.path.isdir(radar_dir):
        print(f"  Radar directory not found: {radar_dir}")
        return

    # Cache mappings by source grid shape to avoid rebuilding
    # when multiple products share the same grid
    mapping_cache = {}

    # Collect errors across all products so we can write one
    # `reproject_radar.log` at the end.
    all_errors: list[tuple[str, str]] = []

    for product in RADAR_PRODUCTS:
        product_dir = os.path.join(radar_dir, product)
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
            nc_files = sorted(
                f for f in os.listdir(day_path) if f.endswith('.nc')
            )
            if nc_files:
                day_jobs.append((day_folder, day_path, out_dir, nc_files))

        if not day_jobs:
            print(f"    No day folders found for {product}")
            continue

        # Build mapping for this product (or reuse if same grid shape)
        first_day_path = day_jobs[0][1]
        first_file = day_jobs[0][3][0]
        filepath = os.path.join(first_day_path, first_file)
        src_lats, src_lons = _read_radar_source_grid(filepath)
        grid_key = src_lats.shape

        if grid_key in mapping_cache:
            mapping = mapping_cache[grid_key]
            print(f"    Reusing mapping for grid {grid_key}")
        else:
            print(f"    Building radar mapping from {first_file} "
                  f"(grid: {grid_key})...")
            mapping = PrecomputedMapping(
                src_lats, src_lons, target_lats, target_lons
            )
            mapping_cache[grid_key] = mapping

        # Run day folders in parallel — the worker is module-level
        # (`_radar_day_worker`) and reads the shared mapping from
        # `_WORKER_STATE`, populated once per worker by `_init_worker`.
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=({'mapping': mapping},),
        ) as pool:
            futures = {
                pool.submit(_radar_day_worker, job): job[0]
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
        'radar', all_errors,
    )


# =============================================================================
# MSG satellite
# =============================================================================

def reproject_satellite_msg(data_root, target_lats, target_lons, date_filter=None):
    """
    Reproject MSG channels. KD-tree built once per channel, day folders in parallel.
    """
    msg_dir = os.path.join(data_root, 'satellite_data', 'MSG')
    reprojected_base = os.path.join(
        data_root, 'reprojected_data', 'satellite_data', 'MSG'
    )

    if not os.path.isdir(msg_dir):
        print(f"  MSG directory not found: {msg_dir}")
        return

    all_errors: list[tuple[str, str]] = []

    for channel in MSG_CHANNELS:
        channel_dir = os.path.join(msg_dir, channel)
        if not os.path.isdir(channel_dir):
            print(f"\n  Channel: {channel} — NOT FOUND at {channel_dir}")
            continue

        print(f"\n  Channel: {channel}")
        mapping = None

        # Collect day folders
        day_jobs = []
        for day_folder in sorted(os.listdir(channel_dir)):
            if date_filter and date_filter not in day_folder:
                continue
            day_path = os.path.join(channel_dir, day_folder)
            if not os.path.isdir(day_path):
                continue
            out_dir = os.path.join(reprojected_base, channel, day_folder)
            nc_files = sorted(
                f for f in os.listdir(day_path) if f.endswith('.nc')
            )
            if nc_files:
                day_jobs.append((day_folder, day_path, out_dir, nc_files))

        if not day_jobs:
            print(f"    No day folders found for {channel}")
            continue

        # Build mapping from first file (must be sequential)
        first_filepath = os.path.join(day_jobs[0][1], day_jobs[0][3][0])
        print(f"    Building MSG {channel} mapping from "
              f"{day_jobs[0][3][0]}...")
        with _nc_lock:
            with Dataset(first_filepath, 'r') as ds:
                lat_grid = np.asarray(ds.variables['latitude'][:],
                                      dtype=np.float64)
                lon_grid = np.asarray(ds.variables['longitude'][:],
                                      dtype=np.float64)

        # Clean NaN values (geostationary off-disk pixels)
        nan_mask = np.isnan(lat_grid) | np.isnan(lon_grid)
        if nan_mask.any():
            lat_grid[nan_mask] = 0.0
            lon_grid[nan_mask] = 0.0
        lat_grid = np.clip(lat_grid, -90.0, 90.0)
        lon_grid = np.clip(lon_grid, -180.0, 180.0)

        mapping = PrecomputedMapping(
            lat_grid, lon_grid, target_lats, target_lons
        )

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=({'mapping': mapping, 'channel': channel},),
        ) as pool:
            futures = {
                pool.submit(_msg_day_worker, job): job[0]
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
        'satellite_MSG', all_errors,
    )


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
    (not per-product). Use inspect_mtg.py to reconstruct full .nc files
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
    """Cache lightning NetCDF data as .npy, day folders in parallel."""
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
            nc_files = sorted(
                f for f in os.listdir(day_path) if f.endswith('.nc')
            )
            if nc_files:
                day_jobs.append((day_folder, day_path, out_dir, nc_files))

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
# NWCSAF (per-variable .npy, mirroring the radar / MTG layout)
# =============================================================================

# Which NWCSAF product file holds each variable, and the variable's dtype.
NWCSAF_VAR_SPEC = {
    "ctth_alti":  {"product": "CTTH", "dtype": np.float32},
    "ctth_tempe": {"product": "CTTH", "dtype": np.float32},
    "cmic_phase": {"product": "CMIC", "dtype": np.int8},     # categorical
    "cmic_cot":   {"product": "CMIC", "dtype": np.float32},
}

# S_NWC_{PRODUCT}_{SAT}_{REGION}_{YYYYMMDD}T{HHMMSS}Z[suffix].nc
_NWCSAF_NAME_PATTERN = re.compile(
    r'^S_NWC_(?P<product>CMIC|CTTH)_[^_]+_[^_]+_'
    r'(?P<date>\d{8})T(?P<hhmm>\d{4})\d{2}Z'
)


def _parse_nwcsaf_filename(name):
    """Return (date_str 'YYYY-MM-DD', hhmm 'HHMM', product) or (None, None, None)."""
    m = _NWCSAF_NAME_PATTERN.match(name)
    if not m:
        return None, None, None
    d = m.group("date")
    return (f"{d[:4]}-{d[4:6]}-{d[6:8]}", m.group("hhmm"), m.group("product"))


def reproject_nwcsaf(data_root, target_lats, target_lons, date_filter=None):
    """
    Reproject NWCSAF CMIC + CTTH files to per-variable .npy on the Romania grid.

    KD-tree built once from the first file; day folders processed in parallel.
    `cmic_phase` is saved as int8 (NaN replaced with 0 = "no cloud / missing")
    so the categorical codes survive intact on disk. All other variables are
    saved as float32.
    """
    nwcsaf_dir = os.path.join(data_root, 'nwcsaf_data')
    reprojected_base = os.path.join(data_root, 'reprojected_data', 'nwcsaf_data')

    if not os.path.isdir(nwcsaf_dir):
        print("  NWCSAF directory not found")
        return

    # Collect day folders (input layout: {date}-Romania/S_NWC_*.nc)
    day_jobs = []
    for day_folder in sorted(os.listdir(nwcsaf_dir)):
        if date_filter and date_filter not in day_folder:
            continue
        day_path = os.path.join(nwcsaf_dir, day_folder)
        if not os.path.isdir(day_path):
            continue
        nc_files = sorted(
            f for f in os.listdir(day_path) if f.endswith('.nc')
        )
        if nc_files:
            day_jobs.append((day_folder, day_path, nc_files))

    if not day_jobs:
        print("    No NWCSAF day folders found")
        return

    # Build KD-tree once from the first file
    first_filepath = os.path.join(day_jobs[0][1], day_jobs[0][2][0])
    print(f"    Building NWCSAF mapping from {day_jobs[0][2][0]}...")
    with _nc_lock:
        with Dataset(first_filepath, 'r') as ds:
            nx = np.asarray(ds.variables['nx'][:], dtype=np.float64)
            ny = np.asarray(ds.variables['ny'][:], dtype=np.float64)
            gdal_proj = ds.getncattr('gdal_projection')

    nx_2d, ny_2d = np.meshgrid(nx, ny)
    print(f"    NWCSAF grid: {ny_2d.shape} ({gdal_proj[:40]}...)")

    # NWCSAF stores the geostationary projection with ellipsoid radii
    # normalised by satellite distance (+a / +b ~ 0.178), which PROJ 9.x
    # rejects as a "non-Earth body" mixed with the EPSG:4326 Earth CRS.
    # Setting PROJ_IGNORE_CELESTIAL_BODY=YES is the documented PROJ escape
    # hatch — see the error message PROJ itself emits when this trips.
    os.environ.setdefault('PROJ_IGNORE_CELESTIAL_BODY', 'YES')

    geos_proj = pyproj.Proj(gdal_proj)
    transformer = pyproj.Transformer.from_proj(
        geos_proj, pyproj.Proj('epsg:4326'), always_xy=True
    )
    lon_grid, lat_grid = transformer.transform(nx_2d, ny_2d)
    invalid = ~(np.isfinite(lat_grid) & np.isfinite(lon_grid))
    if invalid.any():
        n_invalid = int(invalid.sum())
        print(f"    Cleaned {n_invalid} off-disk pixels "
              f"({n_invalid / invalid.size * 100:.1f}%)")
        lat_grid[invalid] = 0.0
        lon_grid[invalid] = 0.0
    lat_grid = np.clip(lat_grid, -90.0, 90.0)
    lon_grid = np.clip(lon_grid, -180.0, 180.0)

    mapping = PrecomputedMapping(
        lat_grid, lon_grid, target_lats, target_lons
    )

    # Write the per-source projection constants once so consumers can
    # rebuild the source grid later without re-opening a CMIC/CTTH file.
    ensure_dir(reprojected_base)
    constants_path = os.path.join(reprojected_base, 'nwcsaf_constants.json')
    if not os.path.isfile(constants_path):
        with open(constants_path, 'w') as f:
            json.dump({
                "gdal_projection": gdal_proj,
                "source_grid_shape": list(ny_2d.shape),
                "note": "lat/lon arrays for the Romania target grid live at "
                        "reprojected_data/romania_grid_{lats,lons}.npy",
            }, f, indent=2)
        print(f"    Wrote {constants_path}")

    all_errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=({'mapping': mapping,
                   'reprojected_base': reprojected_base},),
    ) as pool:
        futures = {pool.submit(_nwcsaf_day_worker, j): j[0] for j in day_jobs}
        for future in as_completed(futures):
            day_folder, new, skipped, errs = future.result()
            total = new + skipped
            if total > 0 or errs:
                print(f"    {day_folder}: {new} new, {skipped} cached, "
                      f"{len(errs)} errors")
            all_errors.extend(errs)

    _write_reproject_log(
        os.path.join(data_root, 'reprojected_data'),
        'nwcsaf', all_errors,
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

    needs_grid = mode in ('radar', 'satellite', 'nwcsaf', 'opera', 'all')
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

    if mode in ('radar', 'all'):
        print(f"\n{'='*70}")
        print("Radar products")
        print(f"{'='*70}")
        reproject_radar(data_root, target_lats, target_lons, date_filter)

    if mode == 'satellite' and instrument == 'MSG' or mode == 'all':
        print(f"\n{'='*70}")
        print("MSG satellite channels")
        print(f"{'='*70}")
        reproject_satellite_msg(data_root, target_lats, target_lons, date_filter)

    if mode == 'satellite' and instrument == 'MTG' or mode == 'all':
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

    if mode in ('nwcsaf', 'all'):
        print(f"\n{'='*70}")
        print("NWCSAF products")
        print(f"{'='*70}")
        reproject_nwcsaf(data_root, target_lats, target_lons, date_filter)

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
    group.add_argument("--radar", action="store_true",
                       help="Reproject radar products")
    group.add_argument("--satellite", type=str, choices=['MSG', 'MTG'],
                       metavar='INSTRUMENT', help="Reproject satellite channels")
    group.add_argument("--lightning", action="store_true",
                       help="Cache lightning data as .npy")
    group.add_argument("--nwcsaf", action="store_true",
                       help="Reproject NWCSAF products")
    group.add_argument("--opera", action="store_true",
                       help="Reproject OPERA radar products (HDF5 -> .npy)")
    group.add_argument("--all", action="store_true",
                       help="Reproject all products")

    args = parser.parse_args()

    # Override worker count from CLI
    MAX_WORKERS = args.workers

    if args.radar:
        mode, instrument = 'radar', None
    elif args.satellite:
        mode, instrument = 'satellite', args.satellite
    elif args.lightning:
        mode, instrument = 'lightning', None
    elif args.nwcsaf:
        mode, instrument = 'nwcsaf', None
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
