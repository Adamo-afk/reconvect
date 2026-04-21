from c4dl.features.regions import (
    save_patches_radar, save_patches_msg, save_patches_solar,
    save_patches_dem, save_patches_lightning, save_patches_cosmo,
    save_patches_nwcsaf
)

import numpy as np
# import cupy as cp
import xarray as xr
import pickle
from pyresample import geometry, kd_tree, bilinear, ewa
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta, date
from netCDF4 import Dataset
from scipy.ndimage import maximum_filter
from scipy.ndimage import label, generate_binary_structure # for finding connected components
from skimage.measure import regionprops, label
from skimage.morphology import disk, opening, closing
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from functools import reduce
import argparse
import os
import re
from pathlib import Path
from inspect import signature
from time import sleep

from our_data.satellite_data.pipeline import fetch_and_process_sat
from our_data.dem_data.processing import tiff_to_netcdf
# from our_data.nwcsaf_data.process_nwcsaf import extract_latlon_coordinates
from our_data.lightning_data.read_kml_version2 import kml_to_netcdf
from concatenate_nc_files import consolidate_weather_patches
from vizualisation import plot_variable_patches_comparison
from c4dl.projection import GridProjection, romania_grid_area
from c4dl.datasets.swissdem import SwissDEMReader
from c4dl.datasets.msgccs4 import MSGRadianceCCS4Reader
from c4dl.datasets.mchlightning import MCHLightningReader
from c4dl.datasets.msgccs4 import NWCSAFCCS4Reader
from c4dl.datasets.mchradar import MCHRadarReader
from c4dl.datasets.cosmonwp import COSMOCCS4Reader

# Keywords to search for
# TODO
# TO FIX:

# Define arguments
parser = argparse.ArgumentParser(description='Extract patches from radar data')
parser.add_argument(
    '--satellite',
    '-sat',
    action='store_true', 
    help='Flag to indicate satelite data to be processed (MSG)'
)
parser.add_argument(
    '--solar', 
    '-sol', 
    action='store_true', 
    help='Flag to indicate solar data to be processed'
)
parser.add_argument(
    '--digital_elevation', 
    '-dem', 
    action='store_true', 
    help='Flag to indicate digital elevation data to be processed'
)
parser.add_argument(
    '--lightning',
    '-light',
    action='store_true',
    help='Flag to indicate lightning data to be processed'
)
parser.add_argument(
    '--numerical_weather_prediction',
    '-nwp',
    action='store_true',
    help='Flag to indicate numerical weather prediction data to be processed'
)
parser.add_argument(
    '--nwcsaf',
    '-saf',
    action='store_true',
    help='Flag to indicate NWCSAF data to be processed'
)
parser.add_argument(
    '--get_satellite_data_store', 
    '-satds',
    nargs=2, 
    metavar=('START_TIME', 'END_TIME'), 
    help='Start and end times in format yyyy/mm/dd-hhmm'
)
parser.add_argument(
    '--process_lightning',
    '-pl',
    action='store_true',
    help='Flag to indicate lightning data to be processed'
)
parser.add_argument(
    '--timestamp',
    '-t',
    type=str,
    default=None,
    required=False,
    help='Timestamp in HHMM format (e.g., 0000, 0015, 1230)'
)
parser.add_argument(
    '--var1',
    '-v1',
    type=str,
    default=None,
    required=False,
    help='Variable 1 to be processed (e.g., IR, VIS, WV, etc.) and patches plotted'
)
parser.add_argument(
    '--var2',
    '-v2',
    type=str,
    default=None,
    required=False,
    help='Variable 2 to be processed (e.g., IR, VIS, WV, etc.) and patches plotted'
)
parser.add_argument(
    '--check_projection',
    '-cp',
    type=str,
    default=None,
    required=False,
    help='Check projection for a specific data source'
)
parser.add_argument(
    '--save_sample',
    '-ss',
    action='store_true',
    help='Flag to indicate whether to save a sample of the processed data'
)
parser.add_argument(
    '--precipitation_only',
    '-pp',
    action='store_true',
    help='Flag to indicate whether to process only precipitation data (RZC)'
)
parser.add_argument(
    '--radar',
    '-rad',
    action='store_true',
    help='Flag to indicate whether to process only radar data'
)
parser.add_argument(
    '--sample_size',
    '-nsamples',
    type=int,
    default=None,
    required=False,
    help='Number of samples to save for radar patches for'
)
args = parser.parse_args()

# Get the path to the repository root (c4dl directory)
# go up from features to root (from coalition4-rcnn\c4dl\features to coalition4-rcnn)
repo_root = Path(__file__).parent.parent.parent  
base_dir = repo_root / 'our_data'

# Define the path for saving and loading processed radar patches
save_path_radar_pickle = os.path.join(base_dir, 'radar_32x32_patches.pkl')

# Create regridded directory if it does not exist
regridded_dir = base_dir / 'regridded_data'
if not os.path.exists(regridded_dir):
    os.makedirs(regridded_dir)

# Create validation directory if it does not exist
validation_dir = base_dir / 'validated_data'
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

    
# Process different data sources
# Satellite data
if args.satellite and args.get_satellite_data_store is not None:
    sat_data_dir = base_dir / 'satellite_data'
    # Get list of variables to process
    variables = save_patches_msg(patches=None, archive_path=None, out_dir=None, suffix='')
    # Process each variable separately
    for data_var in variables:
        var_dir = sat_data_dir / data_var
        if not os.path.exists(var_dir):
            os.makedirs(var_dir)
        # Fetch and process satellite data
        fetch_and_process_sat(
            args.get_satellite_data_store[0], args.get_satellite_data_store[1], var_dir, data_var
        )

    # Get satellite paths
    satellite_paths = [
        var / timestamp 
        for var in sat_data_dir.iterdir() 
            if os.path.isdir(sat_data_dir / var) 
                for timestamp in os.listdir(var)
    ]
    sat_output_dir = base_dir / 'satellite_patches'
elif args.satellite:
    sat_data_dir = base_dir / 'satellite_data'
    # Get satellite paths
    satellite_paths = [
        var / timestamp 
        for var in sat_data_dir.iterdir() 
            if os.path.isdir(sat_data_dir / var) 
                for timestamp in os.listdir(var)
    ]
    sat_output_dir = base_dir / 'satellite_patches'
# DEM data
elif args.digital_elevation:
    dem_data_dir = base_dir / 'dem_data'
    # variables = ['Altitude', 'NS_deriv', 'EW_deriv']
    variables = SwissDEMReader.fields
    # Process each variable separately
    concat_vars = str(reduce(lambda x, y: x + '_' + y, variables))
    # Get DEM TIFF file for Romania
    romania_dem_tiff_file = [
        tiff_file 
        for tiff_file in os.listdir(dem_data_dir / "DEM_1km") 
            if tiff_file.endswith('.tif') and "RO" in tiff_file
    ]
    # Process DEM data
    tiff_to_netcdf(dem_data_dir / "DEM_1km" / romania_dem_tiff_file[0], dem_data_dir, concat_vars)
# Lightning data
elif args.lightning:
    lightning_data_dir = base_dir / 'lightning_data'

    if args.process_lightning:
        # Process lightning data
        kml_data = lightning_data_dir / 'kml_data'
        for timestamp in os.listdir(kml_data):
            kml_folder = kml_data / timestamp
            for kml_file in os.listdir(kml_folder):
                kml_to_netcdf(kml_folder / kml_file) if kml_file.endswith('.kml') else None
    
    # Get lightning paths
    lightning_paths = [
        var / timestamp 
        for var in lightning_data_dir.iterdir() 
            if os.path.isdir(lightning_data_dir / var) 
                for timestamp in os.listdir(var)
    ]
    lightning_output_dir = base_dir / 'lightning_patches'
# Numerical Weather Prediction data
elif args.numerical_weather_prediction:
    nwp_data_dir = base_dir / 'nwp_data'
    nwp_output_dir = base_dir / 'nwp_patches'    

# NWCSAF data
elif args.nwcsaf:
    nwcsaf_data_dir = base_dir / 'nwcsaf_data'
    nwcsaf_output_dir = base_dir / 'nwcsaf_patches'


# Define rainfall rate parameters for region extraction
MIN_POINTS = 10 # num of pixels in a contiguous area
MIN_PRECIPITATION_THRESHOLD = 10 # mm/h

# Define clustering parameters
MULTIPLIER = 2
MIN_CLUSTER_SIZE = MIN_POINTS * MULTIPLIER
MAX_CLUSTERS = 12

# Define patch and data extraction parameters
PATCH_STRIDE = 32
PATCH_SIZE = 256


# ONLY FOR RADAR (different naming in netcdf files used for extracting data and projection)
def get_preferred_radar_group(ds):
    data_group = ds.groups['data']
    
    # Check if radarpicture_0 exists (preferred)
    if 'radarpicture_0' in data_group.groups:
        return 'radarpicture_0'
    
    # Fall back to 'radarpicture' if it exists
    elif 'radarpicture' in data_group.groups:
        return 'radarpicture'


def check_projection(data_dir, all_default_vars, lat, lon, method='nearest'):

    # Check projection for each weather product
    # Get all dir paths and the first file in each directory for all variables

    # Loop through each data directory
    if data_dir == 'radar_data':
        # For radar data, we need to check each variable directory
        for var in all_default_vars:
            var_path = base_dir / data_dir / var
            if os.path.isdir(var_path):
                # Get the first folder and file in the variable directory
                first_folder = next(iter(os.listdir(var_path)), None)
                folder_path = var_path / first_folder
                first_file = next(iter(os.listdir(folder_path)), None)
                file_path = folder_path / first_file
                # Read the NetCDF file
                with Dataset(file_path, 'r') as ds:
                    radarpicture = get_preferred_radar_group(ds)
                    radar_proj = ds.groups['data'].groups[radarpicture].groups['projection']
                    # Extract radar data
                    radar_data = ds.groups['data'].groups[radarpicture].groups['datamap'].variables['datamap'][:]
                    print(f"Radar data projection for {var}: {radar_proj}")

                    # Create 1D arrays
                    lats = np.linspace(
                        float(radar_proj.getncattr('lat_ul')),
                        float(radar_proj.getncattr('lat_lr')),
                        int(radar_proj.getncattr('size_y'))
                    )  # From north to south
                    lons = np.linspace(
                        float(radar_proj.getncattr('lon_ul')), 
                        float(radar_proj.getncattr('lon_lr')),
                        int(radar_proj.getncattr('size_x'))
                    )  # From west to east

                    # Create 2D matrices
                    lon_grid, lat_grid = np.meshgrid(lons, lats)

                    print(f"Latitude grid shape: {lat_grid.shape}")
                    print(f"Longitude grid shape: {lon_grid.shape}")
                    print(f"Lat range: {lat_grid.min():.6f} to {lat_grid.max():.6f}")
                    print(f"Lon range: {lon_grid.min():.6f} to {lon_grid.max():.6f}")

                    # Regrid the radar data to the grid projection
                    regridded = regrid_data(
                        source_lats=lat_grid,
                        source_lons=lon_grid,
                        source_data=radar_data,
                        target_lats=lat,
                        target_lons=lon,
                        method=method
                    )

                    print(f"Regridded radar data shape: {regridded.shape}")
                    print(f"Regridded radar data min: {np.min(regridded)}, max: {np.max(regridded)}")

                    save_simple_netcdf(
                        regridded_data=regridded,
                        target_lats=lat,
                        target_lons=lon,
                        original_filename=first_file,
                        variable_name=var,
                        output_dir=validation_dir
                    )

    elif data_dir == 'satellite_data':
        # For satellite data, we need to check each variable directory
        for var in all_default_vars:
            var_path = base_dir / data_dir / var
            if os.path.isdir(var_path):
                # Get the first folder and file in the variable directory
                first_folder = next(iter(os.listdir(var_path)), None)
                folder_path = var_path / first_folder
                first_file = next(iter(os.listdir(folder_path)), None)
                file_path = folder_path / first_file
                # Read the NetCDF file
                with Dataset(file_path, 'r') as ds:
                    # Extract latitude, longitude, and data
                    lat_grid = ds.variables['latitude'][:]
                    lon_grid = ds.variables['longitude'][:]
                    sat_data = ds.variables[var][:]

                    # Regrid the radar data to the grid projection
                    regridded = regrid_data(
                        source_lats=lat_grid,
                        source_lons=lon_grid,
                        source_data=sat_data,
                        target_lats=lat,
                        target_lons=lon,
                        method=method
                    )

                    print(f"Regridded radar data shape: {regridded.shape}")
                    print(f"Regridded radar data min: {np.min(regridded)}, max: {np.max(regridded)}")

                    save_simple_netcdf(
                        regridded_data=regridded,
                        target_lats=lat,
                        target_lons=lon,
                        original_filename=first_file,
                        variable_name=var,
                        output_dir=validation_dir
                    )

    elif data_dir == 'dem_data':

        main_dir_name = str(reduce(lambda x, y: x + '_' + y, all_default_vars))
        dem_path = base_dir / data_dir / main_dir_name / str('nc4-Romania_' + main_dir_name) 
        first_file = next(iter(os.listdir(dem_path)), None)
        file_path = dem_path / first_file

        for var in all_default_vars:
            # Read the NetCDF file
            with Dataset(file_path, 'r') as ds:
                # Extract latitude, longitude, and data
                lat_grid = ds.variables['latitude'][:]
                lon_grid = ds.variables['longitude'][:]
                dem_data = ds.variables[var][:]

                # Regrid the radar data to the grid projection
                regridded = regrid_data(
                    source_lats=lat_grid,
                    source_lons=lon_grid,
                    source_data=dem_data,
                    target_lats=lat,
                    target_lons=lon,
                    method=method
                )

                print(f"Regridded radar data shape: {regridded.shape}")
                print(f"Regridded radar data min: {np.min(regridded)}, max: {np.max(regridded)}")

                save_simple_netcdf(
                    regridded_data=regridded,
                    target_lats=lat,
                    target_lons=lon,
                    original_filename='nc4-Romania_' + var,
                    variable_name=var,
                    output_dir=validation_dir
                )

    elif data_dir == 'lightning_data':
        # For lightning data, we need to check each variable directory
        for var in all_default_vars:
            var_path = base_dir / data_dir / var
            if os.path.isdir(var_path):
                # Get the first folder and file in the variable directory
                first_folder = next(iter(os.listdir(var_path)), None)
                folder_path = var_path / first_folder
                first_file = next(iter(os.listdir(folder_path)), None)
                file_path = folder_path / first_file
                # Read the NetCDF file
                with Dataset(file_path, 'r') as ds:
                    # Extract latitude, longitude, and data
                    lat_grid = ds.variables['latitude'][:]
                    lon_grid = ds.variables['longitude'][:]
                    lightning_data = ds.variables['datamap'][:]

                    # Regrid the radar data to the grid projection
                    regridded = regrid_data(
                        source_lats=lat_grid,
                        source_lons=lon_grid,
                        source_data=lightning_data,
                        target_lats=lat,
                        target_lons=lon,
                        method=method
                    )

                    print(f"Regridded radar data shape: {regridded.shape}")
                    print(f"Regridded radar data min: {np.min(regridded)}, max: {np.max(regridded)}")

                    save_simple_netcdf(
                        regridded_data=regridded,
                        target_lats=lat,
                        target_lons=lon,
                        original_filename=first_file,
                        variable_name=var,
                        output_dir=validation_dir
                    )

    elif data_dir == 'nwp_data':
        # Get the main directory 
        var_path = base_dir / data_dir

        # For NWP data, we need to check each variable directory
        for var in all_default_vars:
            
            # Get the first folder and file in the variable directory
            first_folder = next(iter(os.listdir(var_path)), None)
            folder_path = var_path / first_folder
            first_file = next(iter(os.listdir(folder_path)), None)
            file_path = folder_path / first_file
            # Read the NetCDF file
            with Dataset(file_path, 'r') as ds:
                # Extract latitude, longitude, and data
                lat_grid = ds.variables['lat'][:]
                lon_grid = ds.variables['lon'][:]
                nwp_data = ds.variables[var.lower()][:]

                # Create 2D matrices
                lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

                if len(nwp_data.shape) == 3:  
                    # Squeeze to remove singleton dimensions
                    nwp_data = np.squeeze(nwp_data, axis=0)
                elif len(nwp_data.shape) == 4:
                    # First, squeeze to remove singleton dimensions
                    nwp_data = np.squeeze(nwp_data, axis=0)  # Remove first
                    # Second, transpose to move the time dimension to the front
                    nwp_data = np.transpose(nwp_data, (1, 2, 0))  # Move time to first dimension
                    # Third, mean over the first dimension if it has more than one level
                    nwp_data = np.mean(nwp_data, axis=-1)

                # Regrid the radar data to the grid projection
                regridded = regrid_data(
                    source_lats=lat_grid,
                    source_lons=lon_grid,
                    source_data=nwp_data,
                    target_lats=lat,
                    target_lons=lon,
                    method=method
                )

                print(f"Regridded radar data shape: {regridded.shape}")
                print(f"Regridded radar data min: {np.min(regridded)}, max: {np.max(regridded)}")

                save_simple_netcdf(
                    regridded_data=regridded,
                    target_lats=lat,
                    target_lons=lon,
                    original_filename=first_file,
                    variable_name=var,
                    output_dir=validation_dir
                )

    elif data_dir == 'nwcsaf_data':
        # Get the main directory 
        var_path = base_dir / data_dir
        
        # For NWCSAF data, we need to check each variable directory
        for var in all_default_vars:
            
            # Get the first folder and file in the variable directory
            first_folder = next(iter(os.listdir(var_path)), None)
            folder_path = var_path / first_folder
            
            # Get all .nc files and find CMIC files (first half based on NWCSAF structure)
            all_nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
            
            # Take first CMIC file for testing
            first_file = all_nc_files[0]  # This should be a CMIC file
            file_path = folder_path / first_file
            
            # Read the NetCDF file
            with Dataset(file_path, 'r') as ds:
                # Extract latitude, longitude from NWCSAF file structure
                lat_grid = ds.variables['lat'][:]  # 2D array for NWCSAF
                lon_grid = ds.variables['lon'][:]  # 2D array for NWCSAF
                
                # Get the variable data (convert to lowercase to match NetCDF variable names)
                var_name_lower = var.lower()
                if var_name_lower in ds.variables:
                    nwcsaf_data = ds.variables[var_name_lower][:]
                else:
                    # Try with 'cmic_' prefix for cloud variables
                    cmic_var_name = f'cmic_{var_name_lower}'
                    if cmic_var_name in ds.variables:
                        nwcsaf_data = ds.variables[cmic_var_name][:]
                    else:
                        print(f"Variable {var} not found in NWCSAF file")
                        continue

                # NWCSAF coordinates are already 2D, no need for meshgrid
                # Handle NaN values that might exist in NWCSAF data
                lat_grid = np.nan_to_num(lat_grid, nan=0.0)
                lon_grid = np.nan_to_num(lon_grid, nan=0.0)

                # Regrid the NWCSAF data to the target grid projection
                regridded = regrid_data(
                    source_lats=lat_grid,
                    source_lons=lon_grid,
                    source_data=nwcsaf_data,
                    target_lats=lat,
                    target_lons=lon,
                    method=method
                )

                print(f"Regridded NWCSAF data shape: {regridded.shape}")
                print(f"Regridded NWCSAF data min: {np.min(regridded)}, max: {np.max(regridded)}")

                save_simple_netcdf(
                    regridded_data=regridded,
                    target_lats=lat,
                    target_lons=lon,
                    original_filename=first_file,
                    variable_name=var,
                    output_dir=validation_dir
                )


def regrid_data(source_lats, source_lons, source_data, target_lats, target_lons, method='nearest'):
    """
    Regrid data to match Romania's grid projection.
    """
    # Define source geometry (data to be regridded)
    source_geo = geometry.GridDefinition(lons=source_lons, lats=source_lats)
    
    # Define target geometry (where data will be regridded to - Romania projection)
    target_geo = geometry.GridDefinition(lons=target_lons, lats=target_lats)
    
    # Resample data
    if method == 'nearest':
        regridded = kd_tree.resample_nearest(
            source_geo, source_data, target_geo,
            radius_of_influence=5000,  # search radius (meters)
            fill_value=0.0
        )

    elif method == 'gauss':
        regridded = kd_tree.resample_gauss(
            source_geo, source_data, target_geo,
            radius_of_influence=5000,
            sigmas=2500,  # smoothing parameter (meters)
            fill_value=0.0
        )

    elif method == 'bilinear':
        # Create bilinear resampler
        regridded = bilinear.XArrayBilinearResampler(
            source_geo, target_geo,
            radius=5000
        )

    elif method == 'ewa':
        # Elliptical Weighted Averaging using the two-step ll2cr + fornav process
        
        # Step 1: ll2cr - Convert lat/lon to column/row coordinates
        # This step maps source coordinates to target grid coordinates
        cols, rows = ewa.ewa.ll2cr(
            source_lons, source_lats,  # Input coordinates
            target_geo                  # Target grid definition
        )
        
        # Step 2: fornav - Forward navigation EWA algorithm
        # This performs the actual elliptical weighted averaging
        regridded = ewa.ewa.fornav(
            cols, rows,                 # Column/row coordinates from ll2cr
            source_data,                # Input data to be regridded
            target_geo.shape,           # Output grid shape
            rows_per_scan=None,         # Number of rows per scan (for swath data)
            weight_delta_max=10.0,      # Maximum weight delta
            weight_distance_max=1.0,    # Maximum weight distance
            maximum_weight_mode=False   # Use cumulative weights
        )
    
    return regridded


def save_simple_netcdf(regridded_data, target_lats, target_lons, original_filename, 
                      variable_name, output_dir):
    """
    Simple NetCDF saving function
    """
    
    # Create output path
    output_filename = f"regridded_{original_filename}"
    output_path = output_dir / variable_name / output_filename
    if not os.path.exists(output_path.parent):
        os.makedirs(output_path.parent)
    
    # Create simple dataset
    ds = xr.Dataset({
        'data': (['y', 'x'], regridded_data),
        'lat': (['y', 'x'], target_lats),
        'lon': (['y', 'x'], target_lons)
    })
    
    # Add basic attributes
    ds.attrs['title'] = f'Regridded {variable_name} data'
    ds.attrs['created'] = str(datetime.now())
    ds.attrs['source'] = original_filename
    
    # Save
    ds.to_netcdf(output_path)
    print(f"Saved to: {output_path}")


def extract_path_components(file_path):
    """
    Extract folder and file components from the full file path
    
    Returns:
    --------
    dict: Contains data_type, variable_name, folder_name, file_name
    """
    path_parts = Path(file_path).parts
    
    # Find the data type (radar_data, dem_data, etc.)
    data_type = None
    for part in path_parts:
        if part.endswith('_data') and part.startswith(('radar', 'dem', 'lightning', 'satellite', 'nwp', 'nwcsaf')):
            data_type = part
            print(f"Detected data type: {data_type}")
            break
    
    if not data_type:
        raise ValueError(f"Cannot determine data type from path: {file_path}")
    
    # Extract components based on data type
    if data_type == 'radar_data': # this works
        # Path: .../radar_data/radar_var/nc4_2024-06-13-Romania_radar_var/radar_data.nc
        var_folder = path_parts[path_parts.index(data_type) + 1]
        nc_folder = path_parts[path_parts.index(data_type) + 2]
        file_name = Path(file_path).name
        
        # Extract variable name from var_folder
        variable_name = var_folder
        
        return {
            'data_type': data_type,
            'variable_name': variable_name,
            'folder_name': nc_folder,
            'file_name': file_name
        }
    
    elif data_type == 'dem_data':
        # Path: .../dem_data/EW_deriv_NS_deriv_Altitude/nc4-Romania_EW_deriv_NS_deriv_Altitude/dem_data.nc
        return {
            'data_type': data_type
        }
    
    elif data_type == 'lightning_data': # this works
        # Path: .../lightning_data/lightning_var/nc4_2024-06-13-Romania_lightning_var/lightning_data.nc
        var_folder = path_parts[path_parts.index(data_type) + 1]
        nc_folder = path_parts[path_parts.index(data_type) + 2]
        file_name = Path(file_path).name
        
        variable_name = var_folder
        
        return {
            'data_type': data_type,
            'variable_name': variable_name,
            'folder_name': nc_folder,
            'file_name': file_name
        }
    
    elif data_type == 'satellite_data': # this works
        # Path: .../satellite_data/satellite_var/nc4_2024-06-12-Romania_satellite_var/satellite_data.nc
        var_folder = path_parts[path_parts.index(data_type) + 1]
        nc_folder = path_parts[path_parts.index(data_type) + 2]
        file_name = Path(file_path).name
        
        variable_name = var_folder
        
        return {
            'data_type': data_type,
            'variable_name': variable_name,
            'folder_name': nc_folder,
            'file_name': file_name
        }
    
    elif data_type == 'nwp_data':
        # Path: .../nwp_data/2024-06-13-Romania/nwp_data.nc
        return {
            'data_type': data_type
        }
    
    elif data_type == 'nwcsaf_data':
        # Path: .../nwcsaf_data/2024-06-13-Romania/nwcsaf_data.nc
        return {
            'data_type': data_type
        }
    
    else:
        print(f"Path does not provide a valid data type: {file_path}")
        raise ValueError(f"Unknown data type in path: {file_path}")


def save_regridded_npy(regridded_data, path_components, base_output_dir, variable_name=None, original_nc_filename=None):
    """
    Save regridded data as .npy file with proper folder structure
    
    Parameters:
    -----------
    regridded_data : numpy.ndarray
        The regridded data to save
    path_components : dict
        Dictionary containing path components from extract_path_components
    base_output_dir : str or Path
        Base directory for saving (regridded_data folder)
    variable_name : str, optional
        Specific variable name (for cases with multiple variables)
    original_nc_filename : str, optional
        Original NetCDF filename to preserve
    """
    # Find the our_data directory from base_output_dir and create correct path
    base_output_dir = Path(base_output_dir)
    
    # Navigate up to find our_data directory
    current_path = base_output_dir
    while current_path.name != 'our_data' and current_path.parent != current_path:
        current_path = current_path.parent
    
    if current_path.name != 'our_data':
        raise ValueError("Could not find 'our_data' directory")
    
    # Create regridded_data path at our_data level
    regridded_data_dir = current_path / 'regridded_data'
    
    # Create folder structure: our_data/regridded_data/data_type/variable_name/folder_name/
    data_type = path_components['data_type']
    
    if data_type in ['radar_data', 'satellite_data', 'lightning_data']:
        var_name = variable_name or path_components['variable_name']
        folder_name = path_components['folder_name']
        output_dir = regridded_data_dir / data_type / var_name / folder_name
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove .nc extension and add .npy
    npy_filename = original_nc_filename.replace('.nc', '.npy')
    
    output_path = output_dir / npy_filename
    
    # Save the data
    np.save(output_path, regridded_data)
    
    print(f"Saved regridded data to: {output_path}")
    print(f"Data shape: {regridded_data.shape}")
    print(f"Data range: {np.min(regridded_data):.3f} to {np.max(regridded_data):.3f}")


def save_regridded_nc(regridded_data_dict, lat_coords, lon_coords, file_path, base_output_dir, original_nc_filename=None):
    """
    Save regridded data as .nc file with only lat, lon, and data values
    
    Parameters:
    -----------
    regridded_data_dict : dict
        Dictionary with variable names as keys and regridded numpy arrays as values
    lat_coords : numpy.ndarray
        Target latitude coordinates (1D or 2D)
    lon_coords : numpy.ndarray
        Target longitude coordinates (1D or 2D)
    file_path : str or Path
        Original file path to extract path components from
    base_output_dir : str or Path
        Base directory for saving (regridded_data folder)
    original_nc_filename : str, optional
        Original NetCDF filename to preserve
    """
    # Extract path components from file_path
    path = Path(file_path)
    parts = path.parts
    
    # Find our_data index
    our_data_idx = None
    for i, part in enumerate(parts):
        if part == 'our_data':
            our_data_idx = i
            break
    
    if our_data_idx is None:
        raise ValueError("Could not find 'our_data' directory")
    
    data_type = parts[our_data_idx + 1]  # dem_data, nwp_data, or nwcsaf_data
    
    # Find the our_data directory from base_output_dir and create correct path
    base_output_dir = Path(base_output_dir)
    
    # Navigate up to find our_data directory
    current_path = base_output_dir
    while current_path.name != 'our_data' and current_path.parent != current_path:
        current_path = current_path.parent
    
    if current_path.name != 'our_data':
        raise ValueError("Could not find 'our_data' directory")
    
    # Create regridded_data path at our_data level
    regridded_data_dir = current_path / 'regridded_data'
    
    # Create folder structure based on data type
    if data_type == 'dem_data':
        # Structure: regridded_data/dem_data/variable_folder/nc4_folder/
        variable_folder = parts[our_data_idx + 2]  # EW_deriv_NS_deriv_Altitude
        nc4_folder = parts[our_data_idx + 3]       # nc4-Romania_EW_deriv_NS_deriv_Altitude
        output_dir = regridded_data_dir / data_type / variable_folder / nc4_folder
        
    elif data_type in ['nwp_data', 'nwcsaf_data']:
        # Structure: regridded_data/nwp_data/date_folder/ or regridded_data/nwcsaf_data/date_folder/
        date_folder = parts[our_data_idx + 2]      # 2024-06-13-Romania
        output_dir = regridded_data_dir / data_type / date_folder
        
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle coordinate arrays - check if they are 1D or 2D
    if lat_coords.ndim == 1 and lon_coords.ndim == 1:
        # 1D coordinates - standard case
        coords = {
            'latitude': lat_coords,
            'longitude': lon_coords
        }
        
        # Create data variables
        data_vars = {}
        for var_name, data in regridded_data_dict.items():
            if var_name in ['nx', 'ny']:
                # Add nx,ny as coordinate variables (this is what the function expects)
                coords[var_name] = data
            elif data.ndim == 2:
                data_vars[var_name] = (['latitude', 'longitude'], data)
            elif data.ndim == 3:
                data_vars[var_name] = (['level', 'latitude', 'longitude'], data)
                coords['level'] = range(data.shape[0])
            elif data.ndim == 1:
                data_vars[var_name] = ([var_name + '_dim'], data)
        
    elif lat_coords.ndim == 2 and lon_coords.ndim == 2:
        # 2D coordinates - curvilinear grid
        # Use generic dimension names for 2D coordinates
        coords = {
            'latitude': (['y', 'x'], lat_coords),
            'longitude': (['y', 'x'], lon_coords)
        }
        
        # Create data variables
        data_vars = {}
        for var_name, data in regridded_data_dict.items():
            if var_name in ['nx', 'ny']:
                # Add nx,ny as coordinate variables
                coords[var_name] = data
            elif data.ndim == 2:
                data_vars[var_name] = (['y', 'x'], data)
            elif data.ndim == 3:
                data_vars[var_name] = (['level', 'y', 'x'], data)
                coords['level'] = range(data.shape[0])
            elif data.ndim == 1:
                data_vars[var_name] = ([var_name + '_dim'], data)
    else:
        raise ValueError("Latitude and longitude coordinates must both be 1D or both be 2D")
    
    # Create dataset with no attributes
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Keep original filename but ensure .nc extension
    nc_filename = original_nc_filename if original_nc_filename else 'regridded_data.nc'
    if not nc_filename.endswith('.nc'):
        nc_filename += '.nc'
    
    output_path = output_dir / nc_filename
    
    # Save the dataset
    ds.to_netcdf(output_path)
    
    print(f"Saved regridded data to: {output_path}")
    print(f"Variables: {list(data_vars.keys())}")
    for var_name, data in regridded_data_dict.items():
        print(f"{var_name} shape: {data.shape}, range: {np.min(data):.3f} to {np.max(data):.3f}")
    

def process_netcdf_file(file_path, lat, lon, method='nearest', base_output_dir=None):
    """
    Process a single NetCDF file and save regridded data as .npy files
    
    Parameters:
    -----------
    file_path : str
        Full path to the NetCDF file
    lat : numpy.ndarray
        Target latitude grid
    lon : numpy.ndarray
        Target longitude grid
    method : str
        Regridding method ('nearest', 'bilinear', etc.)
    base_output_dir : str or Path
        Base directory for saving (defaults to regridded_data)
    """
    file_path = Path(file_path)
    
    if base_output_dir is None:
        base_output_dir = file_path.parent.parent.parent / 'regridded_data'
    
    # Extract path components
    path_components = extract_path_components(file_path)
    data_type = path_components['data_type']
    
    # print(f"Processing {data_type} file: {file_path.name}")
    # print(f"Path components: {path_components}")
    
    # Process based on data type
    if data_type == 'radar_data':
        with Dataset(file_path, 'r') as ds:
            radarpicture = get_preferred_radar_group(ds)
            radar_proj = ds.groups['data'].groups[radarpicture].groups['projection']
            radar_data = ds.groups['data'].groups[radarpicture].groups['datamap'].variables['datamap'][:]
            
            # Create coordinate grids
            lats = np.linspace(
                float(radar_proj.getncattr('lat_ul')),
                float(radar_proj.getncattr('lat_lr')),
                int(radar_proj.getncattr('size_y'))
            )
            lons = np.linspace(
                float(radar_proj.getncattr('lon_ul')), 
                float(radar_proj.getncattr('lon_lr')),
                int(radar_proj.getncattr('size_x'))
            )
            
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # Regrid the data
            regridded = regrid_data(
                source_lats=lat_grid,
                source_lons=lon_grid,
                source_data=radar_data,
                target_lats=lat,
                target_lons=lon,
                method=method
            )
            # print(f"Regridded radar data shape: {regridded.shape}")
            
            # Save as .npy
            save_regridded_npy(
                regridded, 
                path_components, 
                base_output_dir,
                original_nc_filename=os.path.split(file_path)[1]
            )
    
    elif data_type == 'satellite_data':
        with Dataset(file_path, 'r') as ds:
            lat_grid = ds.variables['latitude'][:]
            lon_grid = ds.variables['longitude'][:]
            
            # Get variable name and data
            var_name = path_components['variable_name']
            sat_data = ds.variables[var_name][:]
            
            # Regrid the data
            regridded = regrid_data(
                source_lats=lat_grid,
                source_lons=lon_grid,
                source_data=sat_data,
                target_lats=lat,
                target_lons=lon,
                method=method
            )
            
            # Save as .npy
            save_regridded_npy(
                regridded, 
                path_components, 
                base_output_dir,
                original_nc_filename=os.path.split(file_path)[1]
            )
    
    elif data_type == 'dem_data':
        with Dataset(file_path, 'r') as ds:
            lat_grid = ds.variables['latitude'][:]
            lon_grid = ds.variables['longitude'][:]
            
            # Extract variable names directly from the NetCDF file
            # Exclude coordinate variables (latitude, longitude, x, y)
            coordinate_vars = {'latitude', 'longitude', 'x', 'y'}
            variable_names = [var for var in ds.variables.keys() if var not in coordinate_vars]
            
            # Update path_components with actual variable names from file
            path_components['variable_names'] = variable_names
            
            # print(f"Found variables in file: {variable_names}")
            
            # Process each variable in the DEM file and collect regridded data
            regridded_data_dict = {}
            
            for var_name in variable_names:
                if var_name in ds.variables:
                    dem_data = ds.variables[var_name][:]
                    
                    # Regrid the data
                    regridded = regrid_data(
                        source_lats=lat_grid,
                        source_lons=lon_grid,
                        source_data=dem_data,
                        target_lats=lat,
                        target_lons=lon,
                        method=method
                    )
                    
                    regridded_data_dict[var_name] = regridded
            
            # Save all variables as single .nc file
            save_regridded_nc(
                regridded_data_dict=regridded_data_dict,
                lat_coords=lat,
                lon_coords=lon,
                file_path=file_path,
                base_output_dir=base_output_dir,
                original_nc_filename=os.path.split(file_path)[1]
            )
    
    elif data_type == 'lightning_data':
        with Dataset(file_path, 'r') as ds:
            lat_grid = ds.variables['latitude'][:]
            lon_grid = ds.variables['longitude'][:]
            lightning_data = ds.variables['datamap'][:]
            
            # Regrid the data
            regridded = regrid_data(
                source_lats=lat_grid,
                source_lons=lon_grid,
                source_data=lightning_data,
                target_lats=lat,
                target_lons=lon,
                method=method
            )
            
            # Save as .npy
            save_regridded_npy(
                regridded, 
                path_components, 
                base_output_dir,
                original_nc_filename=os.path.split(file_path)[1]
            )
    
    elif data_type == 'nwp_data':
        with Dataset(file_path, 'r') as ds:
            lat_grid = ds.variables['lat'][:]
            lon_grid = ds.variables['lon'][:]
            
            # Create 2D grids
            lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

            # Extract variable names (exclude coordinates and metadata)
            coordinate_vars = {'lat', 'lon', 'time', 'height', 'depth', 'height_2', 'height_bnds'}
            variable_names = [var for var in ds.variables.keys() if var not in coordinate_vars]

            # Collect regridded data
            regridded_data_dict = {}
            
            # Process each variable in the NWP file
            for var_name in variable_names:
                if var_name in ds.variables:
                    nwp_data = ds.variables[var_name][:]
                    
                    if len(nwp_data.shape) == 3:  
                        # Squeeze to remove singleton dimensions
                        nwp_data = np.squeeze(nwp_data, axis=0)
                    elif len(nwp_data.shape) == 4:
                        # First, squeeze to remove singleton dimensions
                        nwp_data = np.squeeze(nwp_data, axis=0)  # Remove first
                        # Second, transpose to move the time dimension to the front
                        nwp_data = np.transpose(nwp_data, (1, 2, 0))  # Move time to first dimension
                        # Third, mean over the first dimension if it has more than one level
                        nwp_data = np.mean(nwp_data, axis=-1)
                    
                    # Regrid the data
                    regridded = regrid_data(
                        source_lats=lat_grid,
                        source_lons=lon_grid,
                        source_data=nwp_data,
                        target_lats=lat,
                        target_lons=lon,
                        method=method
                    )

                    regridded_data_dict[var_name] = regridded

            # Save all variables as single .nc file
            save_regridded_nc(
                regridded_data_dict=regridded_data_dict,
                lat_coords=lat,
                lon_coords=lon,
                file_path=file_path,
                base_output_dir=base_output_dir,
                original_nc_filename=os.path.split(file_path)[1]
            )

    elif data_type == 'nwcsaf_data':
        with Dataset(file_path, 'r') as ds:
            lat_grid = ds.variables['lat'][:]  # Already 2D for NWCSAF
            lon_grid = ds.variables['lon'][:]  # Already 2D for NWCSAF
            
            # Convert masked arrays to regular arrays for coordinates
            if isinstance(lat_grid, np.ma.MaskedArray):
                lat_grid = lat_grid.filled(fill_value=0.0)
            if isinstance(lon_grid, np.ma.MaskedArray):
                lon_grid = lon_grid.filled(fill_value=0.0)
            
            # Handle NaN values and ensure coordinates are finite
            lat_grid = np.nan_to_num(lat_grid, nan=0.0, posinf=0.0, neginf=0.0)
            lon_grid = np.nan_to_num(lon_grid, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure coordinates are regular numpy arrays (not masked)
            lat_grid = np.asarray(lat_grid, dtype=np.float64)
            lon_grid = np.asarray(lon_grid, dtype=np.float64)
            
            # Validate coordinate ranges
            lat_grid = np.clip(lat_grid, -90.0, 90.0)
            lon_grid = np.clip(lon_grid, -180.0, 180.0)
            
            # Extract and process coordinate variables first
            coordinate_data = {}
            
            # Handle nx,ny coordinates (keep original names for compatibility)
            if 'nx' in ds.variables:
                nx_data = ds.variables['nx'][:]
                if isinstance(nx_data, np.ma.MaskedArray):
                    nx_data = nx_data.filled(fill_value=0.0)
                coordinate_data['nx'] = np.asarray(nx_data, dtype=np.float64)
            
            if 'ny' in ds.variables:
                ny_data = ds.variables['ny'][:]
                if isinstance(ny_data, np.ma.MaskedArray):
                    ny_data = ny_data.filled(fill_value=0.0)
                coordinate_data['ny'] = np.asarray(ny_data, dtype=np.float64)
            
            # If nx,ny don't exist, create them from lat,lon dimensions
            if 'nx' not in coordinate_data:
                coordinate_data['nx'] = np.arange(lat_grid.shape[1], dtype=np.float64)
            if 'ny' not in coordinate_data:
                coordinate_data['ny'] = np.arange(lat_grid.shape[0], dtype=np.float64)
            
            # Extract variable names (exclude coordinates and palette variables)
            coordinate_vars = {'lat', 'lon', 'nx', 'ny', 'time'}
            palette_vars = {var for var in ds.variables.keys() if 'pal' in var}
            exclude_vars = coordinate_vars | palette_vars
            
            variable_names = [var for var in ds.variables.keys() if var not in exclude_vars]
            
            # print(f"Found NWCSAF variables: {variable_names}")
            # print(f"Coordinate data keys: {list(coordinate_data.keys())}")

            # Collect regridded data
            regridded_data_dict = {}
            
            # Add coordinate data first
            regridded_data_dict.update(coordinate_data)
            
            # Process each variable in the NWCSAF file
            for var_name in variable_names:
                if var_name in ds.variables:
                    nwcsaf_data = ds.variables[var_name][:]
                    
                    # Handle different dimensionalities for NWCSAF data
                    if len(nwcsaf_data.shape) == 2:
                        # Most NWCSAF variables are 2D (lat, lon)
                        pass  # Keep as is
                    elif len(nwcsaf_data.shape) == 3:
                        # Some variables might have time dimension, take the first/only time step
                        nwcsaf_data = np.squeeze(nwcsaf_data, axis=0)
                    elif len(nwcsaf_data.shape) == 1:  
                        # 1D variables - keep as is
                        if isinstance(nwcsaf_data, np.ma.MaskedArray):
                            nwcsaf_data = nwcsaf_data.filled(fill_value=0.0)
                        regridded_data_dict[var_name] = np.asarray(nwcsaf_data, dtype=np.float32)
                        continue
                    
                    # Handle masked arrays completely (convert to regular arrays)
                    if isinstance(nwcsaf_data, np.ma.MaskedArray):
                        fill_value = getattr(nwcsaf_data, 'fill_value', np.float32(-3.4028235e+38))
                        nwcsaf_data = nwcsaf_data.filled(fill_value)
                    
                    # Ensure data is a regular numpy array
                    nwcsaf_data = np.asarray(nwcsaf_data, dtype=np.float32)
                    
                    # Handle NaN and infinite values in data
                    nwcsaf_data = np.nan_to_num(nwcsaf_data, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    # Additional check: ensure 2D data shape matches coordinate shape
                    if len(nwcsaf_data.shape) == 2 and nwcsaf_data.shape != lat_grid.shape:
                        print(f"Warning: Data shape {nwcsaf_data.shape} doesn't match coordinate shape {lat_grid.shape}")
                        continue
                    
                    # Regrid the 2D data
                    regridded = regrid_data(
                        source_lats=lat_grid,
                        source_lons=lon_grid,
                        source_data=nwcsaf_data,
                        target_lats=lat,
                        target_lons=lon,
                        method=method
                    )

                    regridded_data_dict[var_name] = regridded

            # Save all variables as single .nc file
            if regridded_data_dict:  # Only save if we have valid data
                save_regridded_nc(
                    regridded_data_dict=regridded_data_dict,
                    lat_coords=lat,
                    lon_coords=lon,
                    file_path=file_path,
                    base_output_dir=base_output_dir,
                    original_nc_filename=os.path.split(file_path)[1]
                )
            else:
                print(f"No valid data to save for file: {file_path}")


def find_rain_regions(datamap, threshold=10, min_size=10):
    # Create binary map where 1 represents values > threshold
    binary_map = (datamap > threshold).astype(int)
    
    # Find connected components
    # Use 8-connectivity (includes diagonals)
    structure = generate_binary_structure(np.ndim(datamap), np.ndim(datamap))
    labeled_array, num_features = label(binary_map, structure=structure)
    
    # Get list of regions
    regions = []
    
    # For each labeled region
    for reg in range(1, num_features + 1):
        # Get coordinates where the label appears
        points = np.where(labeled_array == reg)
        points = list(zip(points[0], points[1]))  # Convert to list of (x,y) coordinates
        
        # If region has more than min_size points
        if len(points) >= min_size:
            regions.append(points)
    
    return regions


def extract_meteorological_insights(datamap, pixel_size_km=1.0, min_intensity=0.1, 
                                   dbscan_eps=5, min_cluster_points=10):
    """
    Extract comprehensive meteorological insights from a 2D weather data array.
    
    Parameters:
    -----------
    datamap : numpy.ndarray
        2D array containing meteorological data (e.g., precipitation, reflectivity)
    pixel_size_km : float
        Spatial resolution in kilometers per pixel
    min_intensity : float
        Minimum threshold for considering significant weather activity
    dbscan_eps : float
        DBSCAN epsilon parameter for clustering
    min_cluster_points : int
        Minimum points required for DBSCAN clustering
    
    Returns:
    --------
    dict : Dictionary containing various meteorological insights
    """
    
    insights = {}
    
    # Basic statistics
    insights['basic_stats'] = {
        'mean_intensity': np.mean(datamap),
        'max_intensity': np.max(datamap),
        'min_intensity': np.min(datamap),
        'std_intensity': np.std(datamap),
        'total_area_km2': datamap.size * (pixel_size_km ** 2)
    }
    
    # Precipitation/weather coverage analysis
    active_mask = datamap > min_intensity
    insights['coverage'] = {
        'active_pixels': np.sum(active_mask),
        'active_area_km2': np.sum(active_mask) * (pixel_size_km ** 2),
        'coverage_percentage': (np.sum(active_mask) / datamap.size) * 100,
        'inactive_area_km2': np.sum(~active_mask) * (pixel_size_km ** 2)
    }
    
    # Intensity distribution analysis
    if np.sum(active_mask) > 0:
        active_data = datamap[active_mask]
        insights['intensity_distribution'] = {
            'light_threshold': np.percentile(active_data, 33),
            'moderate_threshold': np.percentile(active_data, 66),
            'heavy_threshold': np.percentile(active_data, 90),
            'extreme_threshold': np.percentile(active_data, 99),
            'light_area_km2': 0,
            'moderate_area_km2': 0,
            'heavy_area_km2': 0,
            'extreme_area_km2': 0
        }
        
        # Calculate areas for different intensity categories
        light_mask = (datamap > min_intensity) & (datamap <= insights['intensity_distribution']['light_threshold'])
        moderate_mask = (datamap > insights['intensity_distribution']['light_threshold']) & (datamap <= insights['intensity_distribution']['moderate_threshold'])
        heavy_mask = (datamap > insights['intensity_distribution']['moderate_threshold']) & (datamap <= insights['intensity_distribution']['heavy_threshold'])
        extreme_mask = datamap > insights['intensity_distribution']['heavy_threshold']
        
        insights['intensity_distribution']['light_area_km2'] = np.sum(light_mask) * (pixel_size_km ** 2)
        insights['intensity_distribution']['moderate_area_km2'] = np.sum(moderate_mask) * (pixel_size_km ** 2)
        insights['intensity_distribution']['heavy_area_km2'] = np.sum(heavy_mask) * (pixel_size_km ** 2)
        insights['intensity_distribution']['extreme_area_km2'] = np.sum(extreme_mask) * (pixel_size_km ** 2)
    
    # Spatial patterns and structure analysis
    if np.sum(active_mask) > 0:
        # Connected component analysis
        labeled_regions = label(active_mask)
        regions = regionprops(labeled_regions, intensity_image=datamap)
        
        insights['spatial_structure'] = {
            'num_weather_systems': len(regions),
            'largest_system_area_km2': 0,
            'average_system_area_km2': 0,
            'system_areas_km2': [],
            'system_intensities': [],
            'system_centroids': []
        }
        
        if regions:
            areas = [region.area * (pixel_size_km ** 2) for region in regions]
            intensities = [region.mean_intensity for region in regions]
            centroids = [(region.centroid[1] * pixel_size_km, region.centroid[0] * pixel_size_km) for region in regions]
            
            insights['spatial_structure']['largest_system_area_km2'] = max(areas)
            insights['spatial_structure']['average_system_area_km2'] = np.mean(areas)
            insights['spatial_structure']['system_areas_km2'] = areas
            insights['spatial_structure']['system_intensities'] = intensities
            insights['spatial_structure']['system_centroids'] = centroids
    
    # Gradient analysis (useful for fronts, convergence zones)
    gradient_y, gradient_x = np.gradient(datamap)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    insights['gradients'] = {
        'max_gradient': np.max(gradient_magnitude),
        'mean_gradient': np.mean(gradient_magnitude),
        'high_gradient_area_km2': np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 95)) * (pixel_size_km ** 2),
        'gradient_direction_std': np.std(np.arctan2(gradient_y, gradient_x))
    }
    
    # Clustering analysis for weather centers/cores
    if np.sum(active_mask) >= min_cluster_points:
        # Get coordinates of active pixels
        coords = np.column_stack(np.where(active_mask))
        
        try:
            clustering = DBSCAN(eps=dbscan_eps, min_samples=min_cluster_points).fit(coords)
            cluster_labels = clustering.labels_
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            insights['clustering'] = {
                'num_clusters': n_clusters,
                'num_noise_points': np.sum(cluster_labels == -1),
                'cluster_centers': [],
                'cluster_sizes': [],
                'cluster_max_intensities': []
            }
            
            # Analyze each cluster
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:  # Ignore noise points
                    cluster_mask = cluster_labels == cluster_id
                    cluster_coords = coords[cluster_mask]
                    
                    # Calculate cluster center
                    center_y, center_x = np.mean(cluster_coords, axis=0)
                    center_km = (center_x * pixel_size_km, center_y * pixel_size_km)
                    
                    # Get cluster properties
                    cluster_size = len(cluster_coords)
                    cluster_intensities = [datamap[coord[0], coord[1]] for coord in cluster_coords]
                    max_intensity = max(cluster_intensities)
                    
                    insights['clustering']['cluster_centers'].append(center_km)
                    insights['clustering']['cluster_sizes'].append(cluster_size)
                    insights['clustering']['cluster_max_intensities'].append(max_intensity)
        
        except Exception as e:
            insights['clustering'] = {'error': f"Clustering failed: {str(e)}"}
    else:
        insights['clustering'] = {'error': f"Insufficient points for clustering (need >= {min_cluster_points})"}
    
    # Texture and smoothness analysis
    # Apply Gaussian filter to get smoothed version
    smoothed = gaussian(datamap, sigma=2)
    texture = np.abs(datamap - smoothed)
    
    insights['texture'] = {
        'mean_texture': np.mean(texture),
        'max_texture': np.max(texture),
        'high_texture_area_km2': np.sum(texture > np.percentile(texture, 90)) * (pixel_size_km ** 2),
        'smoothness_index': 1 - (np.std(texture) / (np.std(datamap) + 1e-10))
    }
    
    # Hotspot analysis (local maxima)
    # Find local maxima using maximum filter
    local_maxima = maximum_filter(datamap, size=5) == datamap
    significant_maxima = local_maxima & (datamap > np.percentile(datamap[active_mask], 80) if np.sum(active_mask) > 0 else False)
    
    if np.sum(significant_maxima) > 0:
        maxima_coords = np.column_stack(np.where(significant_maxima))
        maxima_intensities = [datamap[coord[0], coord[1]] for coord in maxima_coords]
        
        insights['hotspots'] = {
            'num_hotspots': len(maxima_coords),
            'hotspot_locations_km': [(coord[1] * pixel_size_km, coord[0] * pixel_size_km) for coord in maxima_coords],
            'hotspot_intensities': maxima_intensities,
            'strongest_hotspot_intensity': max(maxima_intensities) if maxima_intensities else 0
        }
    else:
        insights['hotspots'] = {'num_hotspots': 0}
    
    # Morphological analysis (useful for storm structure)
    if np.sum(active_mask) > 0:
        # Apply morphological operations
        structuring_element = disk(3)
        opened = opening(active_mask, structuring_element)
        closed = closing(active_mask, structuring_element)
        
        insights['morphology'] = {
            'compactness': np.sum(opened) / (np.sum(active_mask) + 1e-10),
            'connectivity': np.sum(closed) / (np.sum(active_mask) + 1e-10),
            'fragmentation_index': 1 - (np.sum(opened) / (np.sum(closed) + 1e-10))
        }
    else:
        insights['morphology'] = {'compactness': 0, 'connectivity': 0, 'fragmentation_index': 0}
    
    return insights


def print_meteorological_summary(insights):
    """
    Print a human-readable summary of meteorological insights.
    """
    print("=== METEOROLOGICAL DATA ANALYSIS SUMMARY ===\n")
    
    print("📊 BASIC STATISTICS:")
    stats = insights['basic_stats']
    print(f"  • Mean intensity: {stats['mean_intensity']:.3f}")
    print(f"  • Max intensity: {stats['max_intensity']:.3f}")
    print(f"  • Standard deviation: {stats['std_intensity']:.3f}")
    print(f"  • Total area: {stats['total_area_km2']:.1f} km²\n")
    
    print("🌧️ WEATHER COVERAGE:")
    coverage = insights['coverage']
    print(f"  • Active weather area: {coverage['active_area_km2']:.1f} km²")
    print(f"  • Coverage percentage: {coverage['coverage_percentage']:.1f}%")
    print(f"  • Active pixels: {coverage['active_pixels']:,}\n")
    
    if 'intensity_distribution' in insights:
        print("⚡ INTENSITY DISTRIBUTION:")
        intensity = insights['intensity_distribution']
        print(f"  • Light precipitation: {intensity['light_area_km2']:.1f} km²")
        print(f"  • Moderate precipitation: {intensity['moderate_area_km2']:.1f} km²")
        print(f"  • Heavy precipitation: {intensity['heavy_area_km2']:.1f} km²")
        print(f"  • Extreme precipitation: {intensity['extreme_area_km2']:.1f} km²\n")
    
    if 'spatial_structure' in insights:
        print("🏗️ SPATIAL STRUCTURE:")
        spatial = insights['spatial_structure']
        print(f"  • Number of weather systems: {spatial['num_weather_systems']}")
        if spatial['num_weather_systems'] > 0:
            print(f"  • Largest system: {spatial['largest_system_area_km2']:.1f} km²")
            print(f"  • Average system size: {spatial['average_system_area_km2']:.1f} km²\n")
    
    if 'clustering' in insights and 'num_clusters' in insights['clustering']:
        print("🎯 CLUSTERING ANALYSIS:")
        clustering = insights['clustering']
        print(f"  • Weather clusters found: {clustering['num_clusters']}")
        print(f"  • Noise points: {clustering['num_noise_points']}\n")
    
    if 'hotspots' in insights:
        print("🔥 HOTSPOT ANALYSIS:")
        hotspots = insights['hotspots']
        print(f"  • Number of intensity hotspots: {hotspots['num_hotspots']}")
        if hotspots['num_hotspots'] > 0:
            print(f"  • Strongest hotspot: {hotspots['strongest_hotspot_intensity']:.3f}\n")


def find_cluster_centers(datamap, max_clusters=5, min_points=10):
    
    # Get coordinates of points where rainfall > 10
    points = np.where(datamap > 10)

    if len(points[0]) > 0:
        # print(f"Found points with rainfall: {len(points[0])}")
        coords = np.column_stack((points[0], points[1]))
        # print(f"Number of points with rainfall:\n {points}")
        # print(f"Coordinates:\n {coords}")
        # print(f"Shape of coordinates:\n {coords.shape}")
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=5, min_samples=min_points).fit(coords)
        labels = clustering.labels_
        
        # Find all clusters and their sizes
        unique_labels = list(set(labels))
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:  # Ignore noise points
                cluster_size = np.sum(labels == label)
                cluster_sizes.append((label, cluster_size))

        # Sort clusters by size and take top max_clusters
        # print(f"Cluster sizes before filtering: {cluster_sizes}")
        cluster_sizes = sorted(dict(cluster_sizes).items(), key=lambda x: x[1], reverse=True)
        
        if max_clusters < len(cluster_sizes):
            top_clusters = cluster_sizes[:max_clusters]
        else:
            top_clusters = cluster_sizes
        
        # Find center points for top clusters
        centers = []
        for label, _ in top_clusters:
            cluster_points = coords[labels == label]
            # Calculate centroid
            center = np.mean(cluster_points, axis=0).astype(int)
            centers.append(center)
        
        return centers
    else:
        print("No points found")
        return []


def extract_patches(datamap, centers, patch_size=50, stride=32):
    half_size = patch_size // 2
    patch_info = []  # Will store tuples of (patch, row_indices, col_indices)
    valid_centers = []

    if centers:
        for center in centers:
            # Check if we can extract a full patch around this center
            if (center[0] >= half_size and center[0] < datamap.shape[0] - half_size and
                center[1] >= half_size and center[1] < datamap.shape[1] - half_size):
                
                # Calculate row and column indices for this patch
                row_start = center[0] - half_size
                row_end = center[0] + half_size
                col_start = center[1] - half_size
                col_end = center[1] + half_size
                
                patch = datamap[row_start:row_end, col_start:col_end]
                
                # Calculate corner coordinates ((upper_left, lower_right) coordonate pairs)
                upper_left = (center[0] - half_size, center[1] - half_size)

                # Get upper-left coordinates for each 32x32 patch
                ul_small_patch_coords = []
                lr_small_patch_coords = []

                for i in range(upper_left[0], upper_left[0] + 224, stride):
                    for j in range(upper_left[1], upper_left[1] + 224, stride):
                        ul_small_patch_coords.append((i, j))
                        lr_small_patch_coords.append((i+stride-1, j+stride-1))

                # # Create array of (row_indices, col_indices) pairs
                # row_indices = np.arange(row_start, row_end)
                # col_indices = np.arange(col_start, col_end)
                # (row_indices, col_indices) = np.meshgrid(row_indices, col_indices)
                # indexes_list = np.column_stack((row_indices.ravel(), col_indices.ravel()))
                
                patch_info.append((patch, ul_small_patch_coords, lr_small_patch_coords))
                valid_centers.append(center)
        
        return patch_info, valid_centers
    else:
        print("No valid centers found for patch extraction.")
        return [], []


def plot_clusters_and_patches_with_flip_control(datamap, centers, patch_size=256, 
                                               flip_horizontal=False, flip_vertical=False):
    """
    Enhanced version with explicit flip control
    """
    
    centers = np.array(centers)
    half_size = patch_size // 2

    # Method 1: Threshold-based binary conversion
    binary_datamap = (datamap > 10).astype(int)
    
    # Apply flips to the data if needed
    processed_datamap = binary_datamap.copy()
    processed_centers = centers.copy()
    
    if flip_horizontal:
        processed_datamap = np.fliplr(processed_datamap)
        processed_centers[:, 1] = binary_datamap.shape[1] - 1 - processed_centers[:, 1]
    
    if flip_vertical:
        processed_datamap = np.flipud(processed_datamap)
        processed_centers[:, 0] = binary_datamap.shape[0] - 1 - processed_centers[:, 0]
    
    # Create the figure
    fig = go.Figure()
    
    # Add the processed datamap
    fig.add_trace(go.Heatmap(
        z=processed_datamap,
        colorscale='gray',
        showscale=False,
        name='Rainfall Data'
    ))
    
    # Add cluster centers
    fig.add_trace(go.Scatter(
        x=processed_centers[:, 1],
        y=processed_centers[:, 0],
        mode='markers',
        marker=dict(
            symbol='x',
            size=15,
            color='red',
            line=dict(width=3)
        ),
        name='Cluster Centers'
    ))
    
    # Add patch boundaries
    shapes = []
    for center in processed_centers:
        x0 = center[1] - half_size
        y0 = center[0] - half_size
        x1 = center[1] + half_size
        y1 = center[0] + half_size
        
        shapes.append(dict(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="blue", width=2),
            fillcolor="rgba(0,0,0,0)"
        ))
    
    # Update layout
    fig.update_layout(
        title='Rainfall Regions with Cluster Centers and Patch Boundaries',
        width=800,
        height=800,
        shapes=shapes,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(autorange=True),
        showlegend=True
    )
    
    fig.show()


def convert_time_string(time_str):
    # Just extract the date part (first 10 characters) from the string
    date_part = time_str.split('T')[0]
    
    # Format to the desired output
    return f"nc4_{date_part}-Romania"


def parse_filename_datetime(filename):
    # Extract the datetime portion (first 12 characters for YYYYMMDDHHMM)
    datetime_str = filename[:12]
    
    # Parse the datetime string using strptime
    # Format: YYYYMMDDHHMM (without seconds)
    dt = datetime.strptime(datetime_str, '%Y%m%d%H%M')
    
    # Convert to timestamp (seconds since epoch)
    timestamp = dt.timestamp()
    
    return {
        'datetime': dt,
        'timestamp': timestamp,
        'formatted_date': dt.strftime('%d/%m/%Y'),
        'formatted_time': dt.strftime('%H:%M:%S'),
        'iso_format': dt.strftime('%Y-%m-%dT%H:%M:%S.000000000')
    }


def parse_nc4_filename_datetime(filename):
    """
    Parse datetime from nc4 filename format: nc4_YYYY-MM-DD-Country_HHMM_product_number
    Example: nc4_2024-06-12-Romania_2359_IR_016
    """
    # Split by underscores
    parts = filename.split('_')
    
    if len(parts) < 4:
        raise ValueError(f"Invalid filename format: {filename}")
    
    # Extract date from the second part (which includes country)
    date_country_part = parts[1]  # "2024-06-12-Romania"
    date_str = date_country_part[:10]  # Extract first 10 chars for "2024-06-12"
    
    # Time is the third part (index 2)
    time_str = parts[2]  # "2359"
    
    # Validate time format (should be 4 digits)
    if not time_str.isdigit() or len(time_str) != 4:
        raise ValueError(f"Invalid time format: {time_str}")
    
    # Parse the datetime
    dt = datetime.strptime(f"{date_str} {time_str[:2]}:{time_str[2:4]}", '%Y-%m-%d %H:%M')
    
    return {
        'datetime': dt,
        'timestamp': dt.timestamp(),
        'formatted_date': dt.strftime('%d/%m/%Y'),
        'formatted_time': dt.strftime('%H:%M:%S'),
        'iso_format': dt.strftime('%Y-%m-%dT%H:%M:%S')
    }


def parse_lightning_filename(filename):
    """
    Parse lightning filename and return timestamp in ISO format.
    
    Parameters:
    filename (str): Filename like "lightning_current_20240613_0400.nc"
    
    Returns:
    str: Timestamp in format "2024-06-13T04:00:00.000000000"
    """
    # Remove path and extension if present
    basename = filename.split('/')[-1].split('\\')[-1]  # Handle both / and \ path separators
    basename = basename.replace('.nc', '')
    
    # Regular expression to extract date and time
    # Matches patterns like: lightning_current_20240613_0400, lightning_density_20240613_0400, etc.
    pattern = r'lightning_\w+_(\d{8})_(\d{4})'
    
    match = re.search(pattern, basename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern 'lightning_<type>_YYYYMMDD_HHMM.nc'")
    
    date_str = match.group(1)  # YYYYMMDD
    time_str = match.group(2)  # HHMM
    
    # Parse date and time
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    hour = time_str[:2]
    minute = time_str[2:4]
    
    # Create datetime object
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
    
    # Return in ISO format with nanoseconds
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000")


def parse_nwp_date_and_time_filenames_to_iso(date_string, time_filename):
    """
    Parse meteorological filename components to ISO format timestamp.
    
    Args:
        date_string (str): Date string in format "YYYY-MM-DD-Location" (e.g., "2024-06-13-Romania")
        time_filename (str): Filename containing time in last 8 characters where middle 4 are time
                           (e.g., "ICON-LAM_RO2p8_00_ML_00000000.nc")
    
    Returns:
        str: ISO format timestamp (YYYY-MM-DDTHH:MM:SS)
    
    Examples:
        >>> parse_filename_to_iso("2024-06-13-Romania", "ICON-LAM_RO2p8_00_ML_00000000.nc")
        '2024-06-13T00:00:00'
        >>> parse_filename_to_iso("2024-06-13-Romania", "ICON-LAM_RO2p8_00_ML_12345678.nc")
        '2024-06-13T23:45:00'
    """
    # Extract date from date_string (remove location part)
    date_part = date_string.split('-')[0:3]  # Get YYYY, MM, DD
    year, month, day = date_part
    
    # Extract time from filename - last 8 characters, middle 4 for time
    last_8_chars = time_filename[-11:-3]  # Remove .nc extension, get last 8 chars
    time_part = last_8_chars[2:6]  # Middle 4 characters
    
    # Parse time - assuming HHMM format
    hours = time_part[:2]
    minutes = time_part[2:4]
    
    # Create datetime object
    dt = datetime(int(year), int(month), int(day), int(hours), int(minutes))
    
    # Return ISO format
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000")


def parse_nwcsaf_filename_to_iso(filename):
    """
    Parse NWCSAF filename to ISO format timestamp.
    
    Args:
        filename (str): NWCSAF filename in format:
                       S_NWC_{PRODUCT}_{SATELLITE}_{REGION}-{SENSOR}_{YYYYMMDDTHHMMSS}Z.nc
                       Examples: 
                       - S_NWC_CMIC_MSG3_Europe-VISIR_20240613T000000Z.nc (old format)
                       - S_NWC_CMIC_MSG4_MSG-N-VISIR_20240613T000000Z.nc (new format)
    
    Returns:
        dict: Dictionary containing parsed components:
            - 'iso_timestamp': ISO format timestamp (YYYY-MM-DDTHH:MM:SS.000000000)
            - 'datetime': Python datetime object
            - 'product': Product name (e.g., 'CMIC', 'CTTH')
            - 'satellite': Satellite name (e.g., 'MSG3', 'MSG4')
            - 'region': Region name (e.g., 'Europe', 'MSG-N')
            - 'sensor': Sensor name (e.g., 'VISIR')
            - 'filename': Original filename
    
    Examples:
        >>> parse_nwcsaf_filename_to_iso("S_NWC_CMIC_MSG4_MSG-N-VISIR_20240613T000000Z.nc")
        {
            'iso_timestamp': '2024-06-13T00:00:00.000000000',
            'datetime': datetime(2024, 6, 13, 0, 0),
            'product': 'CMIC',
            'satellite': 'MSG4',
            'region': 'MSG-N',
            'sensor': 'VISIR',
            'filename': 'S_NWC_CMIC_MSG4_MSG-N-VISIR_20240613T000000Z.nc'
        }
    """
    
    # Remove path if full path is provided, keep only filename
    filename = Path(filename).name
    
    # Updated NWCSAF filename pattern to handle both old and new formats
    # Old: S_NWC_{PRODUCT}_{SATELLITE}_{REGION}-{SENSOR}_{YYYYMMDDTHHMMSS}Z.nc
    # New: S_NWC_{PRODUCT}_{SATELLITE}_{REGION-PART1}-{REGION-PART2}-{SENSOR}_{YYYYMMDDTHHMMSS}Z.nc
    
    # More flexible pattern that captures everything between satellite and timestamp
    pattern = r'S_NWC_([A-Z]+)_([A-Z0-9]+)_(.+)_(\d{8}T\d{6})Z\.nc'
    
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Filename '{filename}' does not match NWCSAF pattern")
    
    # Extract basic components
    product = match.group(1)      # CMIC, CTTH, etc.
    satellite = match.group(2)    # MSG3, MSG4, etc.
    datetime_str = match.group(4) # 20240613T000000
    
    # Parse datetime string
    # Format: YYYYMMDDTHHMMSS
    year = int(datetime_str[0:4])
    month = int(datetime_str[4:6])
    day = int(datetime_str[6:8])
    # T separator at position 8
    hour = int(datetime_str[9:11])
    minute = int(datetime_str[11:13])
    second = int(datetime_str[13:15])
    
    # Create datetime object
    dt = datetime(year, month, day, hour, minute, second)
    
    # Create ISO format timestamp with nanosecond precision
    iso_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.000000000")
    
    return {
        'iso_timestamp': iso_timestamp,
        'datetime': dt,
        'product': product,
        'satellite': satellite,
        'filename': filename
    }


def parse_nwcsaf_filename_simple(filename):
    """
    Simplified version that returns only the ISO timestamp string.
    
    Args:
        filename (str): NWCSAF filename
    
    Returns:
        str: ISO format timestamp (YYYY-MM-DDTHH:MM:SS.000000000)
    
    Example:
        >>> parse_nwcsaf_filename_simple("S_NWC_CMIC_MSG3_Europe-VISIR_20240613T000000Z.nc")
        '2024-06-13T00:00:00.000000000'
    """
    result = parse_nwcsaf_filename_to_iso(filename)
    return result['iso_timestamp']


# def find_closest_indices_gpu(matrix, coords):
#     """
#     Find indices of closest points to upper left and lower right coordinates using GPU.
    
#     Parameters:
#     matrix: np.array of shape (H, W, 2) where [..., 0] is lat and [..., 1] is lon
#     coords: tuple of (ul_lat, ul_lon, lr_lat, lr_lon)
    
#     Returns:
#     tuple: ((ul_i, ul_j), (lr_i, lr_j)) - indices for upper left and lower right points
#     """
#     # Transfer matrix to GPU if not already there
#     if not isinstance(matrix, cp.ndarray):
#         matrix_gpu = cp.asarray(matrix)
#     else:
#         matrix_gpu = matrix
    
#     # print(f"Matrix on GPU: {matrix_gpu}")
#     ul_lat, ul_lon, lr_lat, lr_lon = coords
    
#     # print(f"Upper left coordinates: ({ul_lat}, {ul_lon})")
#     # print(f"Lower right coordinates: ({lr_lat}, {lr_lon})")

#     # Calculate Euclidean distance for both points on GPU
#     # For upper left point
#     ul_distances = cp.sqrt(
#         (matrix_gpu[..., 0] - ul_lat)**2 + 
#         (matrix_gpu[..., 1] - ul_lon)**2
#     )
#     # print(f"Upper left distances:\n {ul_distances}")
#     # print(f"Upper left distances shape: {ul_distances.shape}")
    
#     # For lower right point
#     lr_distances = cp.sqrt(
#         (matrix_gpu[..., 0] - lr_lat)**2 + 
#         (matrix_gpu[..., 1] - lr_lon)**2
#     )
#     # print(f"Lower right distances:\n {lr_distances}")
#     # print(f"Lower right distances shape: {lr_distances.shape}")
    
#     # Get indices of minimum distances
#     ul_idx_flat = cp.argmin(ul_distances)
#     lr_idx_flat = cp.argmin(lr_distances)
#     # print(f"Upper left index (flat): {ul_idx_flat}")
#     # print(f"Lower right index (flat): {lr_idx_flat}")

#     # print(f"Minimum upper left distance: {ul_distances.flatten()[ul_idx_flat]}")
#     # print(f"Minimum lower right distance: {lr_distances.flatten()[lr_idx_flat]}")
    
#     # Convert flat indices to multidimensional indices
#     ul_idx = cp.unravel_index(ul_idx_flat, ul_distances.shape)
#     lr_idx = cp.unravel_index(lr_idx_flat, lr_distances.shape)
#     # print(f"Upper left index: {ul_idx}")
#     # print(f"Lower right index: {lr_idx}")
    
#     # print(f"Original coordinate values: {matrix_gpu[ul_idx]}, {matrix_gpu[lr_idx]}")
    
#     # Convert CuPy arrays to Python tuples for return
#     ul_idx = tuple(idx.get().item() for idx in ul_idx)
#     lr_idx = tuple(idx.get().item() for idx in lr_idx)
    
#     # Return with inversed 1st and 3rd indices (as in the original function)
#     return restructure_patch_corners((ul_idx[0], ul_idx[1], lr_idx[0], lr_idx[1])) 


def restructure_patch_corners(values):
    """
    Restructure a 4-value tuple by finding the two closest pairs and ordering them
    
    Returns:
    --------
    tuple : (upper_left_smaller_pair, upper_left_bigger_pair, 
             lower_right_smaller_pair, lower_right_bigger_pair)
    
    Example: (950, 1450, 954, 1454)
    """
    if len(values) != 4:
        raise ValueError("Input must have exactly 4 values")
    
    a, b, c, d = values
    
    # All possible ways to pair 4 values
    possible_pairings = [
        (((a, 0), (b, 1)), ((c, 2), (d, 3))),  # (a,b) and (c,d)
        (((a, 0), (c, 2)), ((b, 1), (d, 3))),  # (a,c) and (b,d)
        (((a, 0), (d, 3)), ((b, 1), (c, 2)))   # (a,d) and (b,c)
    ]
    
    # Find the pairing with minimum total within-pair distances
    best_pairing = None
    min_total_distance = float('inf')
    
    for pair1, pair2 in possible_pairings:
        # Calculate distance within each pair
        dist1 = abs(pair1[0][0] - pair1[1][0])  # distance in first pair
        dist2 = abs(pair2[0][0] - pair2[1][0])  # distance in second pair
        total_distance = dist1 + dist2
        
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_pairing = (pair1, pair2)
    
    pair1, pair2 = best_pairing
    
    # Order each pair: smaller value first (upper_left), larger second (lower_right)
    def order_pair(pair):
        if pair[0][0] <= pair[1][0]:
            return pair[0], pair[1]  # (upper_left, lower_right)
        else:
            return pair[1], pair[0]  # (upper_left, lower_right)
    
    ordered_pair1 = order_pair(pair1)
    ordered_pair2 = order_pair(pair2)
    
    # Determine which pair has smaller values overall (compare averages)
    avg1 = (ordered_pair1[0][0] + ordered_pair1[1][0]) / 2
    avg2 = (ordered_pair2[0][0] + ordered_pair2[1][0]) / 2
    
    if avg1 <= avg2:
        smaller_pair = ordered_pair1
        bigger_pair = ordered_pair2
    else:
        smaller_pair = ordered_pair2
        bigger_pair = ordered_pair1
    
    # Return in the format: (upper_left_smaller, upper_left_bigger, lower_right_smaller, lower_right_bigger)
    restructured_values = (
        smaller_pair[0][0],  # upper_left of smaller pair
        bigger_pair[0][0],   # upper_left of bigger pair
        smaller_pair[1][0],  # lower_right of smaller pair
        bigger_pair[1][0]    # lower_right of bigger pair
    )
    
    return restructured_values


def combine_coordinates(upper_left_list, lower_right_list):
    """
    Combines corresponding tuples from upper left and lower right coordinates
    into tuples of 4 elements (ul_x, ul_y, lr_x, lr_y)
    """
    combined = []
    for ul, lr in zip(upper_left_list, lower_right_list):
        # Combine the two tuples into one
        combined.append((ul[0], ul[1], lr[0], lr[1]))
    return combined


def extract_coordinate_pairs(matrix, indices_lists):
    """
    Extract coordinate pairs from a matrix of shape (height, width, 2)
    using tuples of (x1, y1, x2, y2)
    
    Args:
        matrix: numpy array of shape (height, width, 2) where last dimension
               represents [lat, lon] coordinates
        indices_lists: List of lists containing tuples of (x1, y1, x2, y2)
    
    Returns:
        List of lists containing tuples of (lat1, lon1, lat2, lon2)
    """
    extracted_values = []
    
    if len(matrix.shape) != 3 or matrix.shape[2] != 2:
        raise ValueError(f"Matrix must have shape (height, width, 2), got {matrix.shape}")
    
    height, width = matrix.shape[:2]
    
    for sublist in indices_lists:
        values_sublist = []
        for coords in sublist:
            try:
                if len(coords) != 4:
                    continue
                    
                x1, y1, x2, y2 = coords
                
                # Clip coordinates to valid bounds
                x1 = max(0, min(int(x1), height - 1))
                y1 = max(0, min(int(y1), width - 1))
                x2 = max(0, min(int(x2), height - 1))
                y2 = max(0, min(int(y2), width - 1))
                
                lat1, lon1 = matrix[x1, y1]
                lat2, lon2 = matrix[x2, y2]
                values_sublist.append((lat1, lon1, lat2, lon2))
                
            except Exception as e:
                print(f"Error processing {coords}: {e}")
                continue
                
        extracted_values.append(values_sublist)
    
    return extracted_values


def variable_patch_comparison(all_default_vars):
    """
    Compare two variables from the list of default radar variables.
    This function checks if the provided variables are in the list of default variables,
    and returns the corresponding truth values.
    Args:
        all_default_vars (list): List of default radar variables.
    """
    if args.var1 is not None and args.var2 is not None:
        all_truth_values_var1 = list(map(lambda x: args.var1 in x, all_default_vars))
        all_truth_values_var2 = list(map(lambda x: args.var2 in x, all_default_vars))
        resulted_truth_values = np.logical_or(all_truth_values_var1, all_truth_values_var2)
        resulted_truth_values = resulted_truth_values.tolist()
        print(f"Resulted truth values: {resulted_truth_values}")
        count_true_values = np.sum(resulted_truth_values)
        print(f"Count of true values: {count_true_values}")
        
        if count_true_values == 2:
            # Compare two variables
            print(f"Comparing variables: {args.var1} and {args.var2}")
            results = plot_variable_patches_comparison(
                variable_name_1=args.var1,
                variable_name_2=args.var2, 
                timestamp=args.timestamp
            )
            for k, v in results.items():
                print(f"{k}: {v}")
    else:
        print("No comparison performed. Check variable names provided.")


# Double check: Verify uniqueness and 4-value constraint
def validate_unique_4tuples(data_list):
    """
    Double check that the list contains only unique tuples of 4 values
    """
    # Check 1: All items are tuples of length 4
    all_4tuples = all(isinstance(item, tuple) and len(item) == 4 for item in data_list)
    
    # Check 2: All tuples are unique (no duplicates)
    unique_count = len(set(data_list))
    total_count = len(data_list)
    all_unique = unique_count == total_count
    
    # Check 3: All tuple elements are of expected type (optional - you can remove if not needed)
    all_valid_types = all(
        all(isinstance(val, (int, float)) for val in item) 
        for item in data_list
    )
    
    print(f"Validation Results:")
    print(f"  - All items are 4-element tuples: {all_4tuples}")
    print(f"  - All tuples are unique: {all_unique} (unique: {unique_count}, total: {total_count})")
    print(f"  - All values are numeric: {all_valid_types}")
    print(f"  - Total tuples: {len(data_list)}")
    
    # Return overall validation result
    is_valid = all_4tuples and all_unique and all_valid_types
    print(f"  - Overall validation: {'PASSED' if is_valid else 'FAILED'}")
    
    if not is_valid:
        # Show problematic items for debugging
        if not all_4tuples:
            non_4tuples = [item for item in data_list if not (isinstance(item, tuple) and len(item) == 4)]
            print(f"  - Non-4-tuples found: {non_4tuples}")
        
        if not all_unique:
            from collections import Counter
            counts = Counter(data_list)
            duplicates = [item for item, count in counts.items() if count > 1]
            print(f"  - Duplicate tuples found: {duplicates}")
    
    return is_valid


def is_valid_romania_dir(directory_name):
    """
    Check if directory name matches the pattern: YYYY-MM-DD-Romania
    
    Args:
        directory_name (str): The directory name to check
    
    Returns:
        bool: True if the pattern matches, False otherwise
    """
    # Pattern: 4 digits (year) - 2 digits (month) - 2 digits (day) - Romania
    pattern = r'^\d{4}-\d{2}-\d{2}-Romania$'
    return bool(re.match(pattern, directory_name))


# Process radar data
data_dir = base_dir / 'radar_data'
# Get radar paths
radar_paths = [var / timestamp for var in data_dir.iterdir() for timestamp in os.listdir(var)]
output_dir = base_dir / 'radar_patches'
    

def main():
  
    # Get the number of files in the first directory of radar data
    if args.sample_size is None:
        SAMPLE_SIZE = len(os.listdir(radar_paths[0]))
    elif args.sample_size > len(os.listdir(radar_paths[0])):
        raise ValueError(
            f"Sample size {args.sample_size} exceeds the number of files in the radar directory: "
            f"{len(os.listdir(radar_paths[0]))}"
        )
    else:
        SAMPLE_SIZE = args.sample_size
    
    print(f"Sample size for radar data: {SAMPLE_SIZE}")

    # Define precipitation variable
    pp_var = 'RZC' # Default precipitation variable for radar data

    # Get radar product names (default variables)
    all_default_vars = []
    default_radar_variables = signature(MCHRadarReader.__init__).parameters['variables'].default
    all_default_vars.extend(default_radar_variables)
    print(f"Default radar variables: {default_radar_variables}")

    # Get current year
    current_date = date.today()
    current_year = current_date.year

    # Get projection for Romania
    grid_projection = GridProjection(romania_grid_area)
    (y,x) = np.mgrid[:grid_projection.area.height, :grid_projection.area.width]
    (lons, lats) = grid_projection.inverse(y,x)
    # lat_lon = np.stack([lats, lons], axis=-1)
    print(f"Latitudes shape: {lats.shape}, Lons shape: {lons.shape}")
  
    if args.precipitation_only:
        all_data_dict = {}
        
        # Get path for precipitation data only
        precipitation_paths = [pp_path for pp_path in radar_paths if pp_var in str(pp_path)]
        print(f"Precipitation paths: {precipitation_paths}")

        for netcdf_dir in precipitation_paths:
            data_dict = {}
            filenames = []
            for nc_file in os.listdir(netcdf_dir)[:SAMPLE_SIZE]:
                # Get filename and parse datetime
                filename = os.path.join(netcdf_dir, nc_file)
                result = parse_filename_datetime(nc_file)
                iso_time_format = result['iso_format']

                ################################ OLD CODE ################################
                # # Read the NetCDF file with netCDF4
                # ds = Dataset(filename, 'r')

                # # Access the datamap data - navigate through the groups
                # datamap = ds.groups['data'].groups['radarpicture'].groups['datamap'].variables['datamap'][:]

                # Read the datamap from netcdf file
                process_netcdf_file(
                    file_path=filename,
                    lat=lats,
                    lon=lons,
                    method=args.check_projection
                )

                # Load the datamap from the processed file
                filename_npy = filename.replace('.nc', '.npy')
                radar_regridded_path = os.path.split(data_dir)[1]
                filename_npy = filename_npy.replace(
                    radar_regridded_path,
                    os.path.join('regridded_data', radar_regridded_path)
                )
                datamap = np.load(filename_npy)
                # print(f"Loaded data shape: {datamap.shape}")

                # insights = extract_meteorological_insights(
                #     datamap, 
                #     pixel_size_km=1.0,  # Adjust based on your data resolution
                #     min_intensity=10,   # Adjust threshold for your data type
                #     dbscan_eps=5,        # Clustering distance parameter
                #     min_cluster_points=10
                # )

                # print_meteorological_summary(insights)

                ################################ OLD CODE ################################
                # # To get values for a specific region
                # regions = find_rain_regions(
                #     datamap,
                #     threshold=MIN_PRECIPITATION_THRESHOLD, 
                #     min_size=MIN_POINTS
                # )
                # print(f"Number of regions found: {len(regions)}")

                # region_values = [datamap[x,y] for x,y in regions[0]]  # for first region
                # print(f"Number of regions with at least {MIN_POINTS} connected points: {len(region_values)}")

                # # Transform precipitation data using formula 10^(datamap/10) for each point
                # transformed_data = np.power(10, datamap/10)

                # insights = extract_meteorological_insights(
                #     transformed_data, 
                #     pixel_size_km=1.0,  # Adjust based on your data resolution
                #     min_intensity=10,   # Adjust threshold for your data type
                #     dbscan_eps=5,        # Clustering distance parameter
                #     min_cluster_points=10
                # )

                # print_meteorological_summary(insights)
                
                # # Get projection attributes (min lat and lon, max lat and lon)
                # projection = ds.groups['data'].groups['radarpicture'].groups['projection']
                # # print(projection)
                # max_lat = float(projection.lat_ul)
                # min_lat = float(projection.lat_lr)
                # max_lon = float(projection.lon_lr)
                # min_lon = float(projection.lon_ul)

                # # Map each pixel from datamap to its lat and lon
                # lat_range = np.linspace(min_lat, max_lat, transformed_data.shape[0])
                # lon_range = np.linspace(min_lon, max_lon, transformed_data.shape[1])
                # lats, lons = np.meshgrid(lat_range, lon_range)
                # lat_lon = np.stack([lats, lons], axis=-1)
                    
                # Get a list of kernels from the datamap
                cluster_centers = find_cluster_centers(
                    datamap, 
                    max_clusters=MAX_CLUSTERS, 
                    min_points=MIN_CLUSTER_SIZE
                )
                print(f"Found {len(cluster_centers)} cluster centers in the datamap")
                
                patches, valid_centers = extract_patches(
                    datamap, 
                    cluster_centers, 
                    stride=PATCH_STRIDE,
                    patch_size=PATCH_SIZE
                )
                print(f"Found {len(valid_centers)} valid centers in the datamap")
                
                # # Plot clusters and patches with flip control
                # plot_clusters_and_patches_with_flip_control(
                #     datamap,
                #     valid_centers,
                #     patch_size=256, 
                #     flip_horizontal=False, 
                #     flip_vertical=False
                # )

                if patches:
                    # Save upper-left indexes for each 32x32 patch
                    # We standardize coordinates for patches to generate data for each variable regardless of the datamap size
                    # Get upper left and lower right coordinates for each 256x256 patch
                    ul_all_small_patch_coords = [ul_idxs for _, ul_idxs, _ in patches]
                    lr_all_small_patch_coords = [lr_idxs for _, _, lr_idxs in patches]
                    
                    combined_coords = []
                    for group_ul, group_lr in zip(ul_all_small_patch_coords, lr_all_small_patch_coords):
                        combined_coords.append(combine_coordinates(group_ul, group_lr))

                    """ 
                    update radar data based on timestamp for all variables
                    data_dict: {timestamp: [(patch indices for a timestamp)] for all variables (one at a time)}
                    all_data_dict: {timestamp: [(patch indices for a timestamp)] for all variables (all at once)}
                    """
                    all_data_dict.update({iso_time_format: combined_coords})
                    data_dict.update({iso_time_format: combined_coords}) # these are indices for each 32x32 patch
                    filenames.append(filename_npy)
                else:
                    print(f"No patches found in {filename}. Skipping this file.")
                    continue

            # Create directory if it doesn't exist
            if not os.path.exists(output_dir / os.path.split(netcdf_dir)[1]):
                os.makedirs(output_dir / os.path.split(netcdf_dir)[1])

            print(f"Number of timestamps: {len(data_dict)}")

            var_names = save_patches_radar(
                # need to convert patches to tuple because list as a dictionary key is unhashable (lists are mutable)
                patches=data_dict,
                archive_path=filenames,
                out_dir=output_dir / os.path.split(netcdf_dir)[1],
                netcdf_dirs=netcdf_dir, # send the current netcdf directory
                suffix=current_year,
                timestamp=args.timestamp
            )

        # Save all_data_dict as pickel file
        # Convert and save
        with open(save_path_radar_pickle, 'wb') as f:
            pickle.dump(all_data_dict, f)

        # TO FIX:
        # consolidate_weather_patches(output_dir, var_names, fallback_strategy='use_list')

        # Get the list of all timestamps of a day (same for all days)
        # data_keys = list(all_data_dict.keys()) # list of timestamps

        ################################ OLD CODE ################################
        # print(f"Number of timestamps: {len(list(data_dict.values()))}")
        # print(
        #     "Total number of lists of 32x32 patches from 256x256 patches for all vars: "\
        #     f"{sum([len(list(all_data_dict.values())[i]) for i in range(len(data_keys))])}"
        # )

        # # Get coordinate pairs (actual lat and lon values) for each patch
        # # TIMESTAMPS ARE ALL UNIQUE
        # """
        # N timestamps
        # M 256x256 patches
        # K 32x32 patches per each 256x256 patch
        # """
        # #########################################################################################################
        # radar_lat_lon_patch_coords = [
        #     extract_coordinate_pairs(lat_lon, list(all_data_dict.values())[i]) for i in range(len(data_keys))
        # ] 
        
        # # Zip data keys and radar lat and lon patch coordinates 
        # radar_lat_lon_patch_coords = dict(zip(data_keys, radar_lat_lon_patch_coords))
        # print(f"Radar lat and lon patch coordinates:\n {radar_lat_lon_patch_coords}")
        # #########################################################################################################

        # # Based on argument get indexes from lat and lon values for any variable (with lat and lon approximation)
        # # Radar data will always have the same lat and lon values (at this point of development)
    
    ############### Radar data processing (except RZC) ################
    if args.radar:
        
        # Regridding coordinates
        if args.check_projection is not None and args.save_sample:
            print(f"Saving radar data sample")
            check_projection(
                data_dir=os.path.split(data_dir)[1], 
                all_default_vars=default_radar_variables,
                lat=lats,
                lon=lons,
                method=args.check_projection
            )

        # Get path for radar data (all products, except precipitation)
        rest_of_radar_paths = [pp_path for pp_path in radar_paths if pp_var not in str(pp_path)]
        print(f"Radar paths: {rest_of_radar_paths}")

        try:
            # Extract data_keys from all_data_dict
            # Load and convert to dict radar data from pickle file
            with open(save_path_radar_pickle, 'rb') as f:
                all_data_dict = pickle.load(f)
        except FileNotFoundError:
            print(f"Radar data pickle file not found at {save_path_radar_pickle}.")
            print("Please run the radar data processing first to generate the pickle file.")
            return

        for netcdf_dir in rest_of_radar_paths:
            rad_filenames = []
            iso_times = []
            for nc_file in os.listdir(netcdf_dir)[:SAMPLE_SIZE]:
                # Get filename and parse datetime
                filename = os.path.join(netcdf_dir, nc_file)
                result = parse_filename_datetime(nc_file)
                iso_time_format = result['iso_format']
                iso_times.append(iso_time_format)

                # Read the datamap from netcdf file
                process_netcdf_file(
                    file_path=filename,
                    lat=lats,
                    lon=lons,
                    method=args.check_projection
                )

                # Load the datamap from the processed file
                filename_npy = filename.replace('.nc', '.npy')
                radar_regridded_path = os.path.split(data_dir)[1]
                filename_npy = filename_npy.replace(
                    radar_regridded_path,
                    os.path.join('regridded_data', radar_regridded_path)
                )
                datamap = np.load(filename_npy)
                rad_filenames.append(filename_npy)
                # print(f"Loaded data shape: {datamap.shape}")

            # Create directory if it doesn't exist
            if not os.path.exists(output_dir / os.path.split(netcdf_dir)[1]):
                os.makedirs(output_dir / os.path.split(netcdf_dir)[1])

            print(f"Number of timestamps: {len(all_data_dict)}")

            var_names = save_patches_radar(
                # need to convert patches to tuple because list as a dictionary key is unhashable (lists are mutable)
                patches=all_data_dict,
                archive_path=rad_filenames,
                out_dir=output_dir / os.path.split(netcdf_dir)[1],
                netcdf_dirs=netcdf_dir, # send the current netcdf directory
                suffix=current_year,
                timestamp=args.timestamp
            )
    
    ############### Satellite data processing ################
    elif args.satellite:
        
        # Get default variables using signature
        default_variables = signature(MSGRadianceCCS4Reader.__init__).parameters['variables'].default
        all_default_vars.extend(default_variables)
        netcdf_dirs = [
            dir for dir in satellite_paths for var in default_variables if var in os.path.split(dir)[1]
        ]

        # Regridding coordinates
        if args.check_projection is not None and args.save_sample:
            check_projection(
                data_dir=os.path.split(sat_data_dir)[1], 
                all_default_vars=default_variables,
                lat=lats,
                lon=lons,
                method=args.check_projection
            )
        
        try:
            # Extract data_keys from all_data_dict
            # Load and convert to dict radar data from pickle file
            with open(save_path_radar_pickle, 'rb') as f:
                all_data_dict = pickle.load(f)
        except FileNotFoundError:
            print(f"Radar data pickle file not found at {save_path_radar_pickle}.")
            print("Please run the radar data processing first to generate the pickle file.")
            return

        for netcdf_dir in netcdf_dirs:
            # sat_data_dict = {}
            sat_filenames = []
            iso_times = []
            # current_var = os.path.split(netcdf_dir)[1]
            
            for nc_file in os.listdir(netcdf_dir):
                # Get filename and parse datetime
                filename = os.path.join(netcdf_dir, nc_file)

                process_netcdf_file(
                    file_path=filename,
                    lat=lats,
                    lon=lons,
                    method=args.check_projection
                )

                # Load the datamap from the processed file
                filename_npy = filename.replace('.nc', '.npy')
                sat_regridded_path = os.path.split(sat_data_dir)[1]
                filename_npy = filename_npy.replace(
                    sat_regridded_path, 
                    os.path.join('regridded_data', sat_regridded_path)
                )

                result = parse_nc4_filename_datetime(nc_file)
                iso_time_format = (
                    datetime.fromisoformat(result['iso_format']) + timedelta(minutes=1)
                ).isoformat()
                iso_time_format = iso_time_format + '.000000000'
                iso_times.append(iso_time_format)

                ################################ OLD CODE ################################
                # # Read the NetCDF file with netCDF4
                # ds = Dataset(filename, 'r')

                # # Access lat and lon  matrices and zip them
                # lat = ds.variables['latitude'][:]
                # lon = ds.variables['longitude'][:]
                # lat_lon = np.stack([lat, lon], axis=-1)

                # # With the NEW CODE, we now know that the radar indices are the same for all variables
                # indices_list = []
                # for larger_patch in radar_lat_lon_patch_coords[iso_time_format]: 
                #     for smaller_patch in larger_patch:
                #         # Find closest indices for each patch
                #         indices_list.append(find_closest_indices_gpu(lat_lon, smaller_patch)) # now MORE EFFICIENT
                # sat_data_dict.update({iso_time_format: [indices_list]})
                sat_filenames.append(filename_npy)
            
            saving_dir = '_'.join(os.path.split(netcdf_dir)[1].split('_')[:2]) # remove last part of the name
            print(f"Saving directory: {saving_dir}")
            
            # Create directory if it doesn't exist
            if not os.path.exists(sat_output_dir / saving_dir):
                os.makedirs(sat_output_dir / saving_dir)

            sat_data_dict = {ts: all_data_dict[ts] for ts in iso_times if ts in all_data_dict}
            
            sat_var_names = save_patches_msg(
                # need to convert patches to tuple because list as a dictionary key is unhashable (lists are mutable)
                patches=sat_data_dict,
                archive_path=sat_filenames,
                netcdf_dirs=netcdf_dir, # send the current netcdf directory
                out_dir=sat_output_dir / saving_dir,
                suffix=current_year,
                timestamp=args.timestamp
            )
        # TO FIX:
        # consolidate_weather_patches(sat_output_dir, sat_var_names, fallback_strategy='use_list')

        ############### Variable patches comparison ################
        variable_patch_comparison(all_default_vars)

    ############### Solar data processing ################
    elif args.solar:

        # Create output directory path
        sol_output_dir = base_dir / 'solar_patches'
        
        ################################ OLD CODE ################################
        # Get grid projection data
        # grid_projection = GridProjection(romania_grid_area)
        # solar_reader = SolarReader(grid_projection=grid_projection)
        solar_data_dict = {}

        # # Create lat and lon matrix from solar lat and lon data
        # lat_lon = np.stack([solar_reader.lat, solar_reader.lon], axis=-1)

        try:
            # Extract data_keys from all_data_dict
            # Load and convert to dict radar data from pickle file
            with open(save_path_radar_pickle, 'rb') as f:
                all_data_dict = pickle.load(f)
                data_keys = list(all_data_dict.keys())  # list of timestamps
        except FileNotFoundError:
            print(f"Radar data pickle file not found at {save_path_radar_pickle}.")
            print("Please run the radar data processing first to generate the pickle file.")
            return
        
        # # Solar data is processed for a full day (one timestamp) 
        for timestamp in data_keys:
        #     for larger_patch in radar_lat_lon_patch_coords[timestamp]:
        #         indices_list = []
        #         for smaller_patch in larger_patch:
        #             # Find closest indices for each patch
        #             indices_list.append(find_closest_indices_gpu(lat_lon, smaller_patch)) # now MORE EFFICIENT
        #             # Update solar dictionary with the indices
        #             # solar_data_dict.update({timestamp: (ul_indices, lr_indices)})
        #         solar_data_dict.update({timestamp: [indices_list]})
            solar_data_dict.update({timestamp: all_data_dict[timestamp]})

        # Create directory if it doesn't exist
        if not os.path.exists(sol_output_dir):
            os.makedirs(sol_output_dir)

        sol_var_names = save_patches_solar(
            patches=solar_data_dict,
            out_dir=sol_output_dir,
            suffix=current_year,
            timestamp=args.timestamp
        )
        # TO FIX:
        # consolidate_weather_patches(sol_output_dir, sol_var_names, fallback_strategy='use_list')
    
    ############### Digital Elevation Model (DEM) data processing ################   
    elif args.digital_elevation:
        
        # Get default variables using signature
        default_variables = SwissDEMReader.fields
        all_default_vars.extend(default_variables)

        # Regridding and extracting lat and lon coordinates
        if args.check_projection is not None and args.save_sample:
            check_projection(
                data_dir=os.path.split(dem_data_dir)[1], 
                all_default_vars=default_variables,
                lat=lats,
                lon=lons,
                method=args.check_projection
            )
        
        # Get DEM data
        filename = list(os.walk(dem_data_dir / concat_vars))
        filename = os.path.join(filename[1][0], filename[1][2][0])
        dem_data_dict = {}

        # We have only one DEM file with all 3 variables
        # process the file and save as .nc instead of .npy
        process_netcdf_file(
            file_path=filename,
            lat=lats,
            lon=lons,
            method=args.check_projection
        )

        # Load the datamap from the processed file
        dem_regridded_path = os.path.split(dem_data_dir)[1]
        filename_nc = filename.replace(
            dem_regridded_path, 
            os.path.join('regridded_data', dem_regridded_path)
        )

        # Create output directory path
        dem_output_dir = base_dir / 'dem_patches'

        ################################ OLD CODE ################################
        # # Read the NetCDF file with netCDF4
        # ds = Dataset(filename, 'r')

        # # Access lat and lon  matrices and stack them
        # lat = ds.variables['latitude'][:]
        # lon = ds.variables['longitude'][:]
        # lat_lon = np.stack([lat, lon], axis=-1)

        try:
            # Extract data_keys from all_data_dict
            # Load and convert to dict radar data from pickle file
            with open(save_path_radar_pickle, 'rb') as f:
                all_data_dict = pickle.load(f)
                data_keys = list(all_data_dict.keys())  # list of timestamps
        except FileNotFoundError:
            print(f"Radar data pickle file not found at {save_path_radar_pickle}.")
            print("Please run the radar data processing first to generate the pickle file.")
            return

        # # DEM data processed
        for timestamp in data_keys:
        #     for larger_patch in radar_lat_lon_patch_coords[timestamp]:
        #         indices_list = []
        #         for smaller_patch in larger_patch:
        #             # Find closest indices for each patch
        #             indices_list.append(find_closest_indices_gpu(lat_lon, smaller_patch)) # now MORE EFFICIENT
        #             # Update solar dictionary with the indices
        #             # solar_data_dict.update({timestamp: (ul_indices, lr_indices)})
        #         dem_data_dict.update({timestamp: [indices_list]}) # IT DOES NOT OVERWRITE THE PREVIOUS DATA

            dem_data_dict.update({timestamp: all_data_dict[timestamp]})

        # Create directory if it doesn't exist
        if not os.path.exists(dem_output_dir):
            os.makedirs(dem_output_dir)
        # Create dictionary with a single timestamp (first one)
        # Dictionary data - list of lists of tuples with indices
        # EXTRACT AND SAVE UNIQUE INDICES
        
        first_key = next(iter(dem_data_dict))
        
        # Flatten and filter unique tuples
        unique_tuples = set()
        list_of_patches = list()
        for ts in data_keys:
            if ts in dem_data_dict:
                for sublist in dem_data_dict[ts]:
                    for tuple_item in sublist:
                        list_of_patches.append(tuple_item)
                        unique_tuples.add(tuple_item)

        # Get avg number of patches per timestamp
        avg_patches_per_timestamp = len(list_of_patches) / len(data_keys)

        # Convert back to list of tuples
        flattened_unique_list = list(unique_tuples)
        print(f"Number of unique 4-tuples: {len(flattened_unique_list)}")
        print(f"Total patches: {len(list_of_patches)}")
        print(f"Average patches per timestamp: {round(avg_patches_per_timestamp)}")

        # # Perform validation
        # validation_passed = validate_unique_4tuples(flattened_unique_list)

        dem_var_names = save_patches_dem(
            patches={first_key: [flattened_unique_list]},
            dem_path=filename_nc,
            out_dir=dem_output_dir,
            suffix=current_year,
            timestamp=args.timestamp
        )
        # TO FIX:
        # consolidate_weather_patches(dem_output_dir, dem_var_names, fallback_strategy='use_list')

        ############### Variable patches comparison ################
        variable_patch_comparison(all_default_vars)

    ############### Lightning data processing ################
    # time intervals from kml file differ from generated data (kml in Google Earth - 3h = generated data)
    elif args.lightning:
        
        # Get default variables using signature
        default_variables = signature(MCHLightningReader.__init__).parameters['variables'].default
        all_default_vars.extend(default_variables)
        default_variables = list(map(lambda x: x.replace('-10', ''), default_variables))

        # Regridding and extracting lat and lon coordinates
        if args.check_projection is not None and args.save_sample:
            check_projection(
                data_dir=os.path.split(lightning_data_dir)[1], 
                all_default_vars=default_variables,
                lat=lats,
                lon=lons,
                method=args.check_projection
            )

        netcdf_dirs = [
            dir for dir in lightning_paths for var in default_variables if var in os.path.split(dir)[1]
        ]

        try:
            # Extract data_keys from all_data_dict
            # Load and convert to dict radar data from pickle file
            with open(save_path_radar_pickle, 'rb') as f:
                all_data_dict = pickle.load(f)
                data_keys = list(all_data_dict.keys())  # list of timestamps
        except FileNotFoundError:
            print(f"Radar data pickle file not found at {save_path_radar_pickle}.")
            print("Please run the radar data processing first to generate the pickle file.")
            return
            
        for netcdf_dir in netcdf_dirs:
            # light_data_dict = {}
            light_filenames = []
            iso_times = []
            for nc_file in os.listdir(netcdf_dir):
                # Get filename and parse datetime
                filename = os.path.join(netcdf_dir, nc_file)

                process_netcdf_file(
                    file_path=filename,
                    lat=lats,
                    lon=lons,
                    method=args.check_projection
                )

                # Load the datamap from the processed file
                filename_npy = filename.replace('.nc', '.npy')
                light_regridded_path = os.path.split(lightning_data_dir)[1]
                filename_npy = filename_npy.replace(
                    light_regridded_path,
                    os.path.join('regridded_data', light_regridded_path)
                )

                iso_time_format = parse_lightning_filename(nc_file)
                iso_times.append(iso_time_format)

                if iso_time_format not in data_keys:
                    continue
                else:
                    ################################ OLD CODE ################################
                    # Read the NetCDF file with netCDF4
                    # ds = Dataset(filename, 'r')

                    # # Access lat and lon  matrices and zip them
                    # lat = ds.variables['latitude'][:]
                    # lon = ds.variables['longitude'][:]
                    # lat_lon = np.stack([lat, lon], axis=-1)
                    
                    # indices_list = []
                    # for larger_patch in radar_lat_lon_patch_coords[iso_time_format]: 
                    #     for smaller_patch in larger_patch:
                    #         # Find closest indices for each patch
                    #         indices_list.append(find_closest_indices_gpu(lat_lon, smaller_patch)) # now MORE EFFICIENT
                    # light_data_dict.update({iso_time_format: [indices_list]})
                    light_filenames.append(filename_npy)            
            
            # if not light_data_dict:
            #     raise ValueError(
            #         f"Lightning data dictionary is empty for {netcdf_dir}. Check the data and the timestamp."
            #     )
            saving_dir = '_'.join(os.path.split(netcdf_dir)[1].split('_')[:2]) # remove last part of the name
            print(f"Saving directory: {saving_dir}")

            light_data_dict = {ts: all_data_dict[ts] for ts in iso_times if ts in all_data_dict}

            # Create directory if it doesn't exist
            if not os.path.exists(lightning_output_dir / saving_dir):
                os.makedirs(lightning_output_dir / saving_dir)
            
            light_var_names = save_patches_lightning(
                # need to convert patches to tuple because list as a dictionary key is unhashable (lists are mutable)
                patches=light_data_dict,
                archive_path=light_filenames,
                netcdf_dirs=netcdf_dir, # send the current netcdf directory
                out_dir=lightning_output_dir / saving_dir,
                suffix=current_year,
                timestamp=args.timestamp
            )
        # TO FIX:
        # consolidate_weather_patches(lightning_output_dir, light_var_names, fallback_strategy='use_list')
        
        ############### Variable patches comparison ################
        variable_patch_comparison(all_default_vars)
        
    ############### Numerical Weather Prediction data processing ################
    elif args.numerical_weather_prediction:
        # nwp dirs (paths until timestamp - actual .nc files)
        netcdf_dirs = [nwp_data_dir / timestamp for timestamp in os.listdir(nwp_data_dir)]

        # Get default variables using signature
        default_variables = signature(COSMOCCS4Reader.__init__).parameters['variables'].default
        all_default_vars.extend(default_variables)

        # Regridding and extracting lat and lon coordinates
        if args.check_projection is not None and args.save_sample:
            check_projection(
                data_dir=os.path.split(nwp_data_dir)[1], 
                all_default_vars=default_variables,
                lat=lats,
                lon=lons,
                method=args.check_projection
            )

        try:
            # Extract data_keys from all_data_dict
            # Load and convert to dict radar data from pickle file
            with open(save_path_radar_pickle, 'rb') as f:
                all_data_dict = pickle.load(f)
        except FileNotFoundError:
            print(f"Radar data pickle file not found at {save_path_radar_pickle}.")
            print("Please run the radar data processing first to generate the pickle file.")
            return

        for netcdf_dir in netcdf_dirs:
            # nwp_data_dict = {}
            nwp_filenames = []
            iso_times = []
            for nc_file in os.listdir(netcdf_dir):

                # Get filename and parse datetime
                filename = os.path.join(netcdf_dir, nc_file)

                process_netcdf_file(
                    file_path=filename,
                    lat=lats,
                    lon=lons,
                    method=args.check_projection
                )

                # Load the datamap from the processed file
                nwp_regridded_path = os.path.split(nwp_data_dir)[1]
                filename_nc = filename.replace(
                    nwp_regridded_path,
                    os.path.join('regridded_data', nwp_regridded_path)
                )

                # Get the ISO time format from directory name (date) and NetCDF filename (time)
                iso_time_format = parse_nwp_date_and_time_filenames_to_iso(
                    os.path.split(netcdf_dir)[1], nc_file
                )
                iso_times.append(iso_time_format)

                ################################ OLD CODE ################################
                # Read the NetCDF file with netCDF4
                # ds = Dataset(filename, 'r')

                # # Access lat and lon  matrices and zip them
                # lat = ds.variables['lat'][:] # 1D array
                # lon = ds.variables['lon'][:] # 1D array
                # # Create a 2D array of lat and lon coordinates
                # # GENERAL OBSERVATION: if patch indices are all the same value
                # # then swap lat and lon values in lat_lon matrix creation function
                # lat_2d, lon_2d = np.meshgrid(lat, lon)
                # lat_lon = np.stack([lat_2d, lon_2d], axis=-1) 
                
                # indices_list = []
                # for larger_patch in radar_lat_lon_patch_coords[iso_time_format]:
                #     for smaller_patch in larger_patch: 
                #         # Find closest indices for each patch
                #         indices_list.append(find_closest_indices_gpu(lat_lon, smaller_patch)) # now MORE EFFICIENT
                
                # nwp_data_dict.update({iso_time_format: [indices_list]})
                nwp_filenames.append(filename_nc)

            saving_dir = 'nc4_' + os.path.split(netcdf_dir)[1]
            print(f"Saving directory: {saving_dir}")

            nwp_data_dict = {ts: all_data_dict[ts] for ts in iso_times if ts in all_data_dict}

            # Create directory if it doesn't exist
            if not os.path.exists(nwp_output_dir / saving_dir):
                os.makedirs(nwp_output_dir / saving_dir)

            nwp_var_names = save_patches_cosmo(
                # need to convert patches to tuple because list as a dictionary key is unhashable (lists are mutable)
                patches=nwp_data_dict,
                archive_path=nwp_filenames,
                netcdf_dirs=netcdf_dir, # send the current netcdf directory
                out_dir=nwp_output_dir / saving_dir,
                suffix=current_year,
                timestamp=args.timestamp
            )
        # TO FIX:
        # consolidate_weather_patches(nwp_output_dir, nwp_var_names, fallback_strategy='use_list')

        ############### Variable patches comparison ################
        variable_patch_comparison(all_default_vars)

    ################ NWCSAF data processing ################
    elif args.nwcsaf:
        # nwcsaf dirs (paths until timestamp - actual .nc files)
        netcdf_dirs = [nwcsaf_data_dir / timestamp for timestamp in os.listdir(nwcsaf_data_dir)]

        # Get default variables using signature
        default_variables = signature(NWCSAFCCS4Reader.__init__).parameters['variables'].default
        all_default_vars.extend(default_variables)

        # Regridding and extracting lat and lon coordinates
        if args.check_projection is not None and args.save_sample:
            check_projection(
                data_dir=os.path.split(nwcsaf_data_dir)[1], 
                all_default_vars=default_variables,
                lat=lats,
                lon=lons,
                method=args.check_projection
            )
        
        try:
            # Extract data_keys from all_data_dict
            # Load and convert to dict radar data from pickle file
            with open(save_path_radar_pickle, 'rb') as f:
                all_data_dict = pickle.load(f)
        except FileNotFoundError:
            print(f"Radar data pickle file not found at {save_path_radar_pickle}.")
            print("Please run the radar data processing first to generate the pickle file.")
            return

        for netcdf_dir in netcdf_dirs:
            # nwcsaf_data_dict = {}
            nwcsaf_filenames = []
            iso_times = []
            # print(f"All .nc files: {os.listdir(netcdf_dir)}")
            # print(f"All data keys (timestamps - date and time values): {data_keys}")

            # Knowing that the .nc files in a directory are ordered in the following way:
            # 1. CMIC files for all timestamps
            # 2. CTTH files for all timestamps

            # We count the total number of files in a directory and we do ground division to get half of the interval
            # Then we iterate until we reach that half of the interval 
            # After that we take one CMIC file from the first half and one CTTH file from the second half
            # This way we can get the CMIC and CTTH files for each timestamp 

            # Check if it is a directory
            if os.path.isdir(netcdf_dir):
                total_files = len(os.listdir(netcdf_dir))
                half_interval = total_files // 2
                all_datetime_nc_files_from_dir = [file for file in os.listdir(netcdf_dir) if file.endswith('.nc')]
                # print(f"Total files in directory: {total_files}, half interval: {half_interval}")

                for i in range(half_interval-1):
                    # Get filename and parse datetime
                    filenames = (
                        os.path.join(netcdf_dir, all_datetime_nc_files_from_dir[i]),
                        os.path.join(netcdf_dir, all_datetime_nc_files_from_dir[i+half_interval-1])
                    )
                    
                    # Process each NWCSAF file pair (CMIC and CTTH)
                    for filename in filenames:
                        process_netcdf_file(
                            file_path=filename,
                            lat=lats,
                            lon=lons,
                            method=args.check_projection
                        )

                    # Load the datamap from the processed files
                    nwcsaf_regridded_path = os.path.split(nwcsaf_data_dir)[1]
                    processed_filenames = []
                    for filename in filenames:
                        filename_nc = filename.replace(
                            nwcsaf_regridded_path,
                            os.path.join('regridded_data', nwcsaf_regridded_path)
                        )
                        processed_filenames.append(filename_nc)

                    # Get the ISO time format from directory name (date) and NetCDF filename (time)
                    iso_time_format = parse_nwcsaf_filename_simple(os.path.split(filenames[0])[1])
                    iso_times.append(iso_time_format)

                    ################################ OLD CODE ################################
                    # # Read the first NetCDF file and get latitude and longitude matrix
                    # lat_lon = extract_latlon_coordinates(filenames[0])
                    # lat_lon = np.nan_to_num(lat_lon, nan=0.0)  # Replace NaN values with 0.0

                    # indices_list = []
                    # for larger_patch in radar_lat_lon_patch_coords[iso_time_format]:
                    #     for smaller_patch in larger_patch: 
                    #         # Find closest indices for each patch
                    #         indices_list.append(find_closest_indices_gpu(lat_lon, smaller_patch)) # now MORE EFFICIENT
                    
                    # nwcsaf_data_dict.update({iso_time_format: [indices_list]})
                    nwcsaf_filenames.append(tuple(processed_filenames))
                
                if is_valid_romania_dir(os.path.split(netcdf_dir)[1]):
                    saving_dir = 'nc4_' + os.path.split(netcdf_dir)[1]
                    print(f"Saving directory: {saving_dir}")
                else:
                    print(
                        f"Invalid NWCSAF directory name: {os.path.split(netcdf_dir)[1]}. "
                        "Please check the directory name and ensure it follows the expected format."
                    )
                    break

                nwcsaf_data_dict = {ts: all_data_dict[ts] for ts in iso_times if ts in all_data_dict}

                # Create directory if it doesn't exist
                if not os.path.exists(nwcsaf_output_dir / saving_dir):
                    os.makedirs(nwcsaf_output_dir / saving_dir)

                nwcsaf_var_names = save_patches_nwcsaf(
                    # need to convert patches to tuple because list as a dictionary key is unhashable (lists are mutable)
                    patches=nwcsaf_data_dict,
                    archive_path=nwcsaf_filenames,
                    netcdf_dirs=netcdf_dir, # send the current netcdf directory
                    out_dir=nwcsaf_output_dir / saving_dir,
                    suffix=current_year,
                    timestamp=args.timestamp
                )
        # TO FIX:
        # consolidate_weather_patches(nwcsaf_output_dir, nwcsaf_var_names, fallback_strategy='use_list')

        ############### Variable patches comparison ################
        variable_patch_comparison(all_default_vars)


if __name__ == "__main__":
    main()
