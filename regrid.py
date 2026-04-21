# """
# COALITION-4 data regridding pipeline (optimized with precomputed mappings).

# Regrids all products to the Romania 1536x768 EPSG:31700 grid and caches
# the results so the expensive reprojection runs only once per source file.

# Optimization: the KD-tree is built ONCE per source geometry (not per file).
# All files sharing the same source grid reuse the precomputed index mapping,
# reducing regridding to a fast numpy array lookup.

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
#     our_data/regridded_data/radar_data/{product}/nc4_{date}-Romania_{product}/*.npy
#     our_data/regridded_data/satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.npy
#     our_data/regridded_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
#     our_data/regridded_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
#     our_data/regridded_data/nwcsaf_data/{date}-Romania/*.nc

# Usage (run from F:\\nowcasting\\coalition4-rcnn):
#     python regrid_data.py --radar
#     python regrid_data.py --satellite MSG
#     python regrid_data.py --satellite MTG
#     python regrid_data.py --lightning
#     python regrid_data.py --nwcsaf
#     python regrid_data.py --all
#     python regrid_data.py --radar --date 2024-06-13
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
#         """Apply the precomputed mapping to regrid source data (fast)."""
#         fv = fill_value if fill_value is not None else self.fill_value

#         regridded = kd_tree.get_sample_from_neighbour_info(
#             'nn',
#             self.target_shape,
#             source_data,
#             self.valid_input_index,
#             self.valid_output_index,
#             self.index_array,
#             fill_value=fv
#         )
#         return regridded


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


# def regrid_radar(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Regrid all radar products. KD-tree built once, reused for all files.
#     """
#     radar_dir = os.path.join(data_root, 'radar_data')
#     regridded_base = os.path.join(data_root, 'regridded_data', 'radar_data')
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

#             out_dir = os.path.join(regridded_base, product, day_folder)
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
#                     regridded = mapping.apply(datamap)
#                     regridded = np.flipud(regridded)

#                     ensure_dir(out_dir)
#                     np.save(out_path, regridded)
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

# def regrid_satellite_msg(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Regrid MSG channels. KD-tree built once per channel, reused for all files.
#     """
#     msg_dir = os.path.join(data_root, 'satellite_data', 'MSG')
#     regridded_base = os.path.join(
#         data_root, 'regridded_data', 'satellite_data', 'MSG'
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

#             out_dir = os.path.join(regridded_base, channel, day_folder)
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

#                     regridded = mapping.apply(sat_data)
#                     ensure_dir(out_dir)
#                     np.save(out_path, regridded)
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

# def regrid_satellite_mtg(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Regrid MTG channels. KD-tree built once per resolution (1km/2km),
#     reused for all channels sharing that resolution.
#     """
#     mtg_dir = os.path.join(data_root, 'satellite_data', 'MTG')
#     coord_dir = os.path.join(mtg_dir, 'coordinates')
#     regridded_base = os.path.join(
#         data_root, 'regridded_data', 'satellite_data', 'MTG'
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

#             out_dir = os.path.join(regridded_base, channel, day_folder)
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

#                     regridded = mapping.apply(sat_data)
#                     ensure_dir(out_dir)
#                     np.save(out_path, regridded)
#                     new += 1
#                 except Exception as e:
#                     print(f"    ERROR {nc_file}: {e}")

#             total = new + skipped
#             if total > 0 or filtered > 0:
#                 print(f"    {day_folder}: {new} new, {skipped} cached, "
#                       f"{filtered} filtered, {total} used")


# # =============================================================================
# # Lightning (no regridding)
# # =============================================================================

# def regrid_lightning(data_root, date_filter=None):
#     """Cache lightning NetCDF data as .npy (already on grid)."""
#     lightning_dir = os.path.join(data_root, 'lightning_data')
#     regridded_base = os.path.join(
#         data_root, 'regridded_data', 'lightning_data'
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

#             out_dir = os.path.join(regridded_base, product, day_folder)
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

# def regrid_nwcsaf(data_root, target_lats, target_lons, date_filter=None):
#     """
#     Regrid NWCSAF files. KD-tree built once from first file, reused for all.
#     """
#     nwcsaf_dir = os.path.join(data_root, 'nwcsaf_data')
#     regridded_base = os.path.join(data_root, 'regridded_data', 'nwcsaf_data')

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

#         out_dir = os.path.join(regridded_base, day_folder)
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

#                     regridded_dict = {}
#                     for var_name in var_names:
#                         data = ds.variables[var_name][:]

#                         if data.ndim == 3:
#                             data = np.squeeze(data, axis=0)
#                         elif data.ndim == 1:
#                             if isinstance(data, np.ma.MaskedArray):
#                                 data = data.filled(0.0)
#                             regridded_dict[var_name] = np.asarray(
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

#                         regridded = mapping.apply(data)
#                         regridded_dict[var_name] = regridded

#                 if regridded_dict:
#                     _save_nwcsaf_nc(
#                         regridded_dict, target_lats, target_lons,
#                         out_dir, nc_file
#                     )
#                     new += 1

#             except Exception as e:
#                 print(f"    ERROR {nc_file}: {e}")

#         total = new + skipped
#         if total > 0:
#             print(f"    {new} new, {skipped} cached, {total} total")


# def _save_nwcsaf_nc(data_dict, lats, lons, out_dir, filename):
#     """Save regridded NWCSAF data as a single NetCDF file."""
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
#     print("COALITION-4 Data Regridding Pipeline (precomputed mappings)")
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
#         regrid_radar(data_root, target_lats, target_lons, date_filter)

#     if mode == 'satellite' and instrument == 'MSG' or mode == 'all':
#         print(f"\n{'='*70}")
#         print("MSG satellite channels")
#         print(f"{'='*70}")
#         regrid_satellite_msg(data_root, target_lats, target_lons, date_filter)

#     if mode == 'satellite' and instrument == 'MTG' or mode == 'all':
#         print(f"\n{'='*70}")
#         print("MTG satellite channels")
#         print(f"{'='*70}")
#         if target_lats is None:
#             target_lats, target_lons = init_romania_grid()
#         regrid_satellite_mtg(data_root, target_lats, target_lons, date_filter)

#     if mode in ('lightning', 'all'):
#         print(f"\n{'='*70}")
#         print("Lightning products")
#         print(f"{'='*70}")
#         regrid_lightning(data_root, date_filter)

#     if mode in ('nwcsaf', 'all'):
#         print(f"\n{'='*70}")
#         print("NWCSAF products")
#         print(f"{'='*70}")
#         regrid_nwcsaf(data_root, target_lats, target_lons, date_filter)

#     elapsed = timer_module.time() - t_start
#     print(f"\n{'='*70}")
#     print(f"Done in {elapsed:.1f}s.")
#     print(f"{'='*70}")


# # =============================================================================
# # CLI
# # =============================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="COALITION-4 data regridding pipeline. "
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
#                        help="Regrid radar products")
#     group.add_argument("--satellite", type=str, choices=['MSG', 'MTG'],
#                        metavar='INSTRUMENT', help="Regrid satellite channels")
#     group.add_argument("--lightning", action="store_true",
#                        help="Cache lightning data as .npy")
#     group.add_argument("--nwcsaf", action="store_true",
#                        help="Regrid NWCSAF products")
#     group.add_argument("--all", action="store_true",
#                        help="Regrid all products")

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
COALITION-4 data regridding pipeline (optimized with precomputed mappings).

Regrids all products to the Romania 1536x768 EPSG:31700 grid and caches
the results so the expensive reprojection runs only once per source file.

Optimization: the KD-tree is built ONCE per source geometry (not per file).
All files sharing the same source grid reuse the precomputed index mapping,
reducing regridding to a fast numpy array lookup.

Products handled:
    - Radar:     RZC, BZC, CZC, EZC-20, LZC, CPCH       → .npy
    - MSG:       VIS006, IR_039, IR_108, WV_062, WV_073   → .npy
    - MTG:       vis_06, ir_38, ir_105, wv_63, wv_73      → .npy
    - Lightning: density, current, occurrence (already on grid)   → .npy
    - NWCSAF:    ctth_alti, ctth_tempe, cmic_phase, cmic_cot     → .nc

Input paths:
    our_data/radar_data/{product}/nc4_{date}-Romania_{product}/*.nc
    our_data/satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.nc
    our_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.nc
    our_data/satellite_data/MTG/coordinates/{lat,lon}_{1km,2km}.npy
    our_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.nc
    our_data/nwcsaf_data/{date}-Romania/*.nc

Output paths:
    our_data/regridded_data/radar_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/regridded_data/satellite_data/MSG/{channel}/nc4_{date}-Romania_{channel}/*.npy
    our_data/regridded_data/satellite_data/MTG/{channel}/nc4_{date}-Romania_{channel}/*.npy
    our_data/regridded_data/lightning_data/{product}/nc4_{date}-Romania_{product}/*.npy
    our_data/regridded_data/nwcsaf_data/{date}-Romania/*.nc

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python regrid_data.py --radar
    python regrid_data.py --satellite MSG
    python regrid_data.py --satellite MTG
    python regrid_data.py --lightning
    python regrid_data.py --nwcsaf
    python regrid_data.py --all
    python regrid_data.py --radar --date 2024-06-13
"""

import numpy as np
import xarray as xr
import os
import re
import argparse
import threading
import time as timer_module
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from netCDF4 import Dataset
from pyresample import geometry, kd_tree
import pyproj

from c4dl.projection import GridProjection, romania_grid_area

# Global lock for netCDF4/HDF5 reads (C library is not thread-safe)
_nc_lock = threading.Lock()


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

QUARTER_HOUR_MINUTES = {'00', '15', '30', '45'}

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
        """Apply the precomputed mapping to regrid source data (fast)."""
        fv = fill_value if fill_value is not None else self.fill_value

        regridded = kd_tree.get_sample_from_neighbour_info(
            'nn',
            self.target_shape,
            source_data,
            self.valid_input_index,
            self.valid_output_index,
            self.index_array,
            fill_value=fv
        )
        return regridded


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


def is_quarter_hour(filename):
    """Check if a filename's timestamp is on the 15-minute grid."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')
    if len(parts) >= 3 and parts[0] == 'nc4':
        time_str = parts[2]
        if len(time_str) == 4 and time_str.isdigit():
            return time_str[2:4] in QUARTER_HOUR_MINUTES
    if len(basename) >= 12 and basename[:12].isdigit():
        return basename[10:12] in QUARTER_HOUR_MINUTES
    match = re.search(r'_(\d{8})_(\d{4})', basename)
    if match:
        return match.group(2)[2:4] in QUARTER_HOUR_MINUTES
    return True


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


def regrid_radar(data_root, target_lats, target_lons, date_filter=None):
    """
    Regrid all radar products. KD-tree built once, day folders in parallel.
    """
    radar_dir = os.path.join(data_root, 'radar_data')
    regridded_base = os.path.join(data_root, 'regridded_data', 'radar_data')

    if not os.path.isdir(radar_dir):
        print(f"  Radar directory not found: {radar_dir}")
        return

    # Cache mappings by source grid shape to avoid rebuilding
    # when multiple products share the same grid
    mapping_cache = {}

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
            out_dir = os.path.join(regridded_base, product, day_folder)
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

        # Worker function for one day folder
        def process_radar_day(job, _mapping=mapping):
            day_folder, day_path, out_dir, nc_files = job
            new, skipped = 0, 0
            for nc_file in nc_files:
                npy_file = nc_file.replace('.nc', '.npy')
                out_path = os.path.join(out_dir, npy_file)
                if output_exists(out_path):
                    skipped += 1
                    continue
                try:
                    filepath = os.path.join(day_path, nc_file)
                    datamap = _read_radar_data(filepath)
                    regridded = _mapping.apply(datamap)
                    regridded = np.flipud(regridded)
                    ensure_dir(out_dir)
                    np.save(out_path, regridded)
                    new += 1
                except Exception as e:
                    print(f"    ERROR {nc_file}: {e}")
            return day_folder, new, skipped

        # Run day folders in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_radar_day, job): job[0]
                for job in day_jobs
            }
            for future in as_completed(futures):
                day_folder, new, skipped = future.result()
                total = new + skipped
                if total > 0:
                    print(f"    {day_folder}: {new} new, {skipped} cached, "
                          f"{total} total")


# =============================================================================
# MSG satellite
# =============================================================================

def regrid_satellite_msg(data_root, target_lats, target_lons, date_filter=None):
    """
    Regrid MSG channels. KD-tree built once per channel, day folders in parallel.
    """
    msg_dir = os.path.join(data_root, 'satellite_data', 'MSG')
    regridded_base = os.path.join(
        data_root, 'regridded_data', 'satellite_data', 'MSG'
    )

    if not os.path.isdir(msg_dir):
        print(f"  MSG directory not found: {msg_dir}")
        return

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
            out_dir = os.path.join(regridded_base, channel, day_folder)
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

        # Worker function
        def process_msg_day(job, _mapping=mapping, _channel=channel):
            day_folder, day_path, out_dir, nc_files = job
            new, skipped = 0, 0
            for nc_file in nc_files:
                npy_file = nc_file.replace('.nc', '.npy')
                out_path = os.path.join(out_dir, npy_file)
                if output_exists(out_path):
                    skipped += 1
                    continue
                try:
                    filepath = os.path.join(day_path, nc_file)
                    with _nc_lock:
                        with Dataset(filepath, 'r') as ds:
                            var_name = find_data_variable(ds, _channel)
                            if var_name is None:
                                continue
                            sat_data = ds.variables[var_name][:]
                    regridded = _mapping.apply(sat_data)
                    ensure_dir(out_dir)
                    np.save(out_path, regridded)
                    new += 1
                except Exception as e:
                    print(f"    ERROR {nc_file}: {e}")
            return day_folder, new, skipped

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_msg_day, job): job[0]
                for job in day_jobs
            }
            for future in as_completed(futures):
                day_folder, new, skipped = future.result()
                total = new + skipped
                if total > 0:
                    print(f"    {day_folder}: {new} new, {skipped} cached, "
                          f"{total} total")


# =============================================================================
# MTG satellite
# =============================================================================

def regrid_satellite_mtg(data_root, target_lats, target_lons, date_filter=None):
    """
    Regrid MTG channels. KD-tree built once per resolution (1km/2km),
    day folders in parallel.
    """
    mtg_dir = os.path.join(data_root, 'satellite_data', 'MTG')
    coord_dir = os.path.join(mtg_dir, 'coordinates')
    regridded_base = os.path.join(
        data_root, 'regridded_data', 'satellite_data', 'MTG'
    )

    if not os.path.isdir(mtg_dir):
        print(f"  MTG directory not found: {mtg_dir}")
        return

    # Precompute mappings per resolution
    mapping_cache = {}
    for res in ['1km', '2km']:
        lat_path = os.path.join(coord_dir, f'lat_{res}.npy')
        lon_path = os.path.join(coord_dir, f'lon_{res}.npy')
        if os.path.isfile(lat_path) and os.path.isfile(lon_path):
            src_lats = np.load(lat_path).astype(np.float64)
            src_lons = np.load(lon_path).astype(np.float64)

            # Clean NaN values (geostationary off-disk pixels)
            nan_mask = np.isnan(src_lats) | np.isnan(src_lons)
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                print(f"  MTG {res}: cleaning {n_nan} NaN pixels "
                      f"({n_nan / nan_mask.size * 100:.1f}%)")
                src_lats[nan_mask] = 0.0
                src_lons[nan_mask] = 0.0

            src_lats = np.clip(src_lats, -90.0, 90.0)
            src_lons = np.clip(src_lons, -180.0, 180.0)

            print(f"  Building MTG {res} mapping "
                  f"(shape: {src_lats.shape})...")
            mapping_cache[res] = PrecomputedMapping(
                src_lats, src_lons, target_lats, target_lons
            )
        else:
            print(f"  WARNING: MTG {res} coordinates not found at {coord_dir}")

    for channel in MTG_CHANNELS:
        channel_dir = os.path.join(mtg_dir, channel)
        if not os.path.isdir(channel_dir):
            print(f"\n  Channel: {channel} — NOT FOUND at {channel_dir}")
            continue

        res = '1km' if channel in MTG_1KM_CHANNELS else '2km'
        if res not in mapping_cache:
            print(f"  Skipping {channel}: no {res} mapping")
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
            out_dir = os.path.join(regridded_base, channel, day_folder)
            nc_files = sorted(
                f for f in os.listdir(day_path) if f.endswith('.nc')
            )
            if nc_files:
                day_jobs.append((day_folder, day_path, out_dir, nc_files))

        if not day_jobs:
            print(f"    No day folders found for {channel}")
            continue

        # Worker function
        def process_mtg_day(job, _mapping=mapping, _channel=channel):
            day_folder, day_path, out_dir, nc_files = job
            new, skipped, filtered = 0, 0, 0
            for nc_file in nc_files:
                if not is_quarter_hour(nc_file):
                    filtered += 1
                    continue
                npy_file = nc_file.replace('.nc', '.npy')
                out_path = os.path.join(out_dir, npy_file)
                if output_exists(out_path):
                    skipped += 1
                    continue
                try:
                    filepath = os.path.join(day_path, nc_file)
                    with _nc_lock:
                        with Dataset(filepath, 'r') as ds:
                            var_name = find_data_variable(ds, _channel)
                            if var_name is None:
                                continue
                            sat_data = ds.variables[var_name][:]
                    regridded = _mapping.apply(sat_data)
                    ensure_dir(out_dir)
                    np.save(out_path, regridded)
                    new += 1
                except Exception as e:
                    print(f"    ERROR {nc_file}: {e}")
            return day_folder, new, skipped, filtered

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_mtg_day, job): job[0]
                for job in day_jobs
            }
            for future in as_completed(futures):
                day_folder, new, skipped, filtered = future.result()
                total = new + skipped
                if total > 0 or filtered > 0:
                    print(f"    {day_folder}: {new} new, {skipped} cached, "
                          f"{filtered} filtered, {total} used")


# =============================================================================
# Lightning (no regridding)
# =============================================================================

def regrid_lightning(data_root, date_filter=None):
    """Cache lightning NetCDF data as .npy, day folders in parallel."""
    lightning_dir = os.path.join(data_root, 'lightning_data')
    regridded_base = os.path.join(
        data_root, 'regridded_data', 'lightning_data'
    )

    if not os.path.isdir(lightning_dir):
        print(f"  Lightning directory not found: {lightning_dir}")
        return

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
            out_dir = os.path.join(regridded_base, product, day_folder)
            nc_files = sorted(
                f for f in os.listdir(day_path) if f.endswith('.nc')
            )
            if nc_files:
                day_jobs.append((day_folder, day_path, out_dir, nc_files))

        if not day_jobs:
            print(f"    No day folders found for {product}")
            continue

        # Worker function
        def process_lightning_day(job):
            day_folder, day_path, out_dir, nc_files = job
            new, skipped = 0, 0
            for nc_file in nc_files:
                npy_file = nc_file.replace('.nc', '.npy')
                out_path = os.path.join(out_dir, npy_file)
                if output_exists(out_path):
                    skipped += 1
                    continue
                try:
                    filepath = os.path.join(day_path, nc_file)
                    with _nc_lock:
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
                    print(f"    ERROR {nc_file}: {e}")
            return day_folder, new, skipped

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_lightning_day, job): job[0]
                for job in day_jobs
            }
            for future in as_completed(futures):
                day_folder, new, skipped = future.result()
                total = new + skipped
                if total > 0:
                    print(f"    {day_folder}: {new} new, {skipped} cached, "
                          f"{total} total")


# =============================================================================
# NWCSAF (saves as .nc)
# =============================================================================

def regrid_nwcsaf(data_root, target_lats, target_lons, date_filter=None):
    """
    Regrid NWCSAF files. KD-tree built once from first file,
    day folders in parallel.
    """
    nwcsaf_dir = os.path.join(data_root, 'nwcsaf_data')
    regridded_base = os.path.join(data_root, 'regridded_data', 'nwcsaf_data')

    if not os.path.isdir(nwcsaf_dir):
        print("  NWCSAF directory not found")
        return

    # Collect day folders
    day_jobs = []
    for day_folder in sorted(os.listdir(nwcsaf_dir)):
        if date_filter and date_filter not in day_folder:
            continue
        day_path = os.path.join(nwcsaf_dir, day_folder)
        if not os.path.isdir(day_path):
            continue
        out_dir = os.path.join(regridded_base, day_folder)
        nc_files = sorted(
            f for f in os.listdir(day_path) if f.endswith('.nc')
        )
        if nc_files:
            day_jobs.append((day_folder, day_path, out_dir, nc_files))

    if not day_jobs:
        print("    No NWCSAF day folders found")
        return

    # Build mapping from first file in first day folder (sequential)
    # NWCSAF files store coordinates as geostationary projection (nx/ny in meters),
    # not lat/lon. We compute lat/lon using pyproj.
    first_filepath = os.path.join(day_jobs[0][1], day_jobs[0][3][0])
    print(f"    Building NWCSAF mapping from {day_jobs[0][3][0]}...")
    with _nc_lock:
        with Dataset(first_filepath, 'r') as ds:
            nx = np.asarray(ds.variables['nx'][:], dtype=np.float64)
            ny = np.asarray(ds.variables['ny'][:], dtype=np.float64)
            gdal_proj = ds.getncattr('gdal_projection')

    # Build 2D grids from 1D coordinate arrays
    nx_2d, ny_2d = np.meshgrid(nx, ny)
    print(f"    NWCSAF grid: {ny_2d.shape} ({gdal_proj[:40]}...)")

    # Convert geostationary projection → lat/lon
    geos_proj = pyproj.Proj(gdal_proj)
    transformer = pyproj.Transformer.from_proj(
        geos_proj, pyproj.Proj('epsg:4326'), always_xy=True
    )
    lon_grid, lat_grid = transformer.transform(nx_2d, ny_2d)

    # Clean inf/NaN (off-disk pixels in geostationary projection)
    invalid = ~(np.isfinite(lat_grid) & np.isfinite(lon_grid))
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        print(f"    Cleaned {n_invalid} off-disk pixels "
              f"({n_invalid / invalid.size * 100:.1f}%)")
        lat_grid[invalid] = 0.0
        lon_grid[invalid] = 0.0

    lat_grid = np.clip(lat_grid, -90.0, 90.0)
    lon_grid = np.clip(lon_grid, -180.0, 180.0)

    mapping = PrecomputedMapping(
        lat_grid, lon_grid, target_lats, target_lons
    )

    # Worker function
    def process_nwcsaf_day(job, _mapping=mapping,
                           _target_lats=target_lats,
                           _target_lons=target_lons):
        day_folder, day_path, out_dir, nc_files = job
        new, skipped = 0, 0
        for nc_file in nc_files:
            out_path = os.path.join(out_dir, nc_file)
            if output_exists(out_path):
                skipped += 1
                continue
            try:
                filepath = os.path.join(day_path, nc_file)

                # Read all data under lock (HDF5 not thread-safe)
                with _nc_lock:
                    with Dataset(filepath, 'r') as ds:
                        exclude = {'nx', 'ny', 'time'}
                        exclude |= {v for v in ds.variables if 'pal' in v}
                        var_names = [
                            v for v in ds.variables if v not in exclude
                        ]

                        raw_data = {}
                        for var_name in var_names:
                            raw_data[var_name] = ds.variables[var_name][:]

                # Process outside lock (numpy is thread-safe)
                regridded_dict = {}
                for var_name, data in raw_data.items():
                    if data.ndim == 3:
                        data = np.squeeze(data, axis=0)
                    elif data.ndim == 1:
                        if isinstance(data, np.ma.MaskedArray):
                            data = data.filled(0.0)
                        regridded_dict[var_name] = np.asarray(
                            data, dtype=np.float32
                        )
                        continue
                    elif data.ndim != 2:
                        continue

                    if isinstance(data, np.ma.MaskedArray):
                        fill_val = getattr(
                            data, 'fill_value',
                            np.float32(-3.4028235e+38)
                        )
                        data = data.filled(fill_val)

                    data = np.asarray(data, dtype=np.float32)
                    data = np.nan_to_num(
                        data, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    regridded = _mapping.apply(data)
                    regridded_dict[var_name] = regridded

                if regridded_dict:
                    _save_nwcsaf_nc(
                        regridded_dict, _target_lats, _target_lons,
                        out_dir, nc_file
                    )
                    new += 1

            except Exception as e:
                print(f"    ERROR {nc_file}: {e}")
        return day_folder, new, skipped

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_nwcsaf_day, job): job[0]
            for job in day_jobs
        }
        for future in as_completed(futures):
            day_folder, new, skipped = future.result()
            total = new + skipped
            if total > 0:
                print(f"    {day_folder}: {new} new, {skipped} cached, "
                      f"{total} total")


def _save_nwcsaf_nc(data_dict, lats, lons, out_dir, filename):
    """Save regridded NWCSAF data as a single NetCDF file."""
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)

    data_vars = {}
    for var_name, data in data_dict.items():
        if data.ndim == 2:
            data_vars[var_name] = (['y', 'x'], data)
        elif data.ndim == 1:
            data_vars[var_name] = ([f'{var_name}_dim'], data)

    ds = xr.Dataset(
        data_vars,
        coords={
            'latitude': (['y', 'x'], lats),
            'longitude': (['y', 'x'], lons)
        }
    )
    ds.to_netcdf(out_path)


# =============================================================================
# Main pipeline
# =============================================================================

def run(data_root, mode, instrument=None, date_filter=None):
    print("=" * 70)
    print("COALITION-4 Data Regridding Pipeline (precomputed mappings)")
    print("=" * 70)
    print(f"Data root : {data_root}")
    print(f"Mode      : {mode}" + (f" ({instrument})" if instrument else ""))
    print(f"Workers   : {MAX_WORKERS}")
    if date_filter:
        print(f"Date      : {date_filter}")

    t_start = timer_module.time()

    needs_grid = mode in ('radar', 'satellite', 'nwcsaf', 'all')
    if needs_grid:
        target_lats, target_lons = init_romania_grid()
    else:
        target_lats = target_lons = None

    if mode in ('radar', 'all'):
        print(f"\n{'='*70}")
        print("Radar products")
        print(f"{'='*70}")
        regrid_radar(data_root, target_lats, target_lons, date_filter)

    if mode == 'satellite' and instrument == 'MSG' or mode == 'all':
        print(f"\n{'='*70}")
        print("MSG satellite channels")
        print(f"{'='*70}")
        regrid_satellite_msg(data_root, target_lats, target_lons, date_filter)

    if mode == 'satellite' and instrument == 'MTG' or mode == 'all':
        print(f"\n{'='*70}")
        print("MTG satellite channels")
        print(f"{'='*70}")
        if target_lats is None:
            target_lats, target_lons = init_romania_grid()
        regrid_satellite_mtg(data_root, target_lats, target_lons, date_filter)

    if mode in ('lightning', 'all'):
        print(f"\n{'='*70}")
        print("Lightning products")
        print(f"{'='*70}")
        regrid_lightning(data_root, date_filter)

    if mode in ('nwcsaf', 'all'):
        print(f"\n{'='*70}")
        print("NWCSAF products")
        print(f"{'='*70}")
        regrid_nwcsaf(data_root, target_lats, target_lons, date_filter)

    elapsed = timer_module.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.1f}s.")
    print(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COALITION-4 data regridding pipeline. "
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
                       help="Regrid radar products")
    group.add_argument("--satellite", type=str, choices=['MSG', 'MTG'],
                       metavar='INSTRUMENT', help="Regrid satellite channels")
    group.add_argument("--lightning", action="store_true",
                       help="Cache lightning data as .npy")
    group.add_argument("--nwcsaf", action="store_true",
                       help="Regrid NWCSAF products")
    group.add_argument("--all", action="store_true",
                       help="Regrid all products")

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
    elif args.all:
        mode, instrument = 'all', None

    run(
        data_root=args.data_root,
        mode=mode,
        instrument=instrument,
        date_filter=args.date,
    )


