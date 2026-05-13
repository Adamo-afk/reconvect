####################################################################################################
# Transformer for extracting latitude and longitude coordinates directly from NWCSAF NetCDF files
####################################################################################################
import xarray as xr
import numpy as np
from pyproj import Transformer
import os
from time import sleep


def extract_latlon_coordinates(input_nc_file):
    """
    Extract latitude and longitude coordinates directly from NWCSAF NetCDF file.
    
    Args:
        input_nc_file (str): Path to NWCSAF NetCDF file
    
    Returns:
        numpy.ndarray: Coordinate array of shape (H, W, 2) where:
                      [:, :, 0] = latitude values  
                      [:, :, 1] = longitude values
    
    Example:
        >>> coords = extract_latlon_coordinates_direct("S_NWC_CTTH_MSG3_Europe-VISIR_20240613T000000Z.nc")
        >>> print(f"Coordinate shape: {coords.shape}")  # (1019, 2200, 2)
        >>> lat_grid = coords[:, :, 0]  # Latitude grid
        >>> lon_grid = coords[:, :, 1]  # Longitude grid
    """
    
    print(f"Extracting coordinates directly from NetCDF file: {os.path.basename(input_nc_file)}")
    
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(input_nc_file, decode_cf=False)
        
        # Get coordinate information from the NetCDF file
        # NWCSAF files typically have x,y coordinates in the native projection
        if 'nx' in ds.coords and 'ny' in ds.coords:
            x_coords = ds['nx'].values
            y_coords = ds['ny'].values
            # print(f"Found nx,ny coordinates: nx shape={x_coords.shape}, ny shape={y_coords.shape}")
        elif 'X' in ds.coords and 'Y' in ds.coords:
            x_coords = ds['X'].values
            y_coords = ds['Y'].values
            # print(f"Found X,Y coordinates: X shape={x_coords.shape}, Y shape={y_coords.shape}")
        else:
            print(f"Available coordinates: {list(ds.coords.keys())}")
            raise ValueError("Could not find x,y coordinates in NetCDF file")
        
        # Get data dimensions
        if len(x_coords.shape) == 1 and len(y_coords.shape) == 1:
            # 1D coordinate arrays - create 2D grids
            # print(f"Creating 2D coordinate grids from 1D arrays")
            # print(f"X range: {x_coords.min():.0f} to {x_coords.max():.0f}")
            # print(f"Y range: {y_coords.min():.0f} to {y_coords.max():.0f}")
            
            x_2d, y_2d = np.meshgrid(x_coords, y_coords)
        else:
            # Already 2D coordinate arrays
            # print(f"Using existing 2D coordinate arrays")
            x_2d, y_2d = x_coords, y_coords
        
        ds.close()
        
        # print(f"Native projection coordinate grid shape: {x_2d.shape}")
        
        # Transform from GEOS projection to WGS84 lat/lon
        # print("Transforming coordinates from GEOS projection to WGS84...")
        
        # GEOS projection parameters for MSG
        geos_proj = "+proj=geos +a=6378137.000000 +b=6356752.300000 +lon_0=0.000000 +h=35785863.000000 +sweep=y"
        wgs84_proj = "+proj=longlat +datum=WGS84"
        
        # Create transformer
        transformer = Transformer.from_crs(geos_proj, wgs84_proj, always_xy=True)
        
        # Transform coordinates (x_2d, y_2d are in meters, need to convert to lon, lat in degrees)
        # print("Performing coordinate transformation...")
        lon_2d, lat_2d = transformer.transform(x_2d, y_2d)
        
        # Handle invalid transformations (set to NaN)
        valid_mask = np.isfinite(lon_2d) & np.isfinite(lat_2d)
        lon_2d = np.where(valid_mask, lon_2d, np.nan)
        lat_2d = np.where(valid_mask, lat_2d, np.nan)
        
        # Stack coordinates into (H, W, 2) format: [lat, lon]
        coords_stacked = np.stack([lat_2d, lon_2d], axis=-1)
        
        # print(f"Final coordinate array shape: {coords_stacked.shape}")
        # print(f"Latitude range: {np.nanmin(lat_2d):.2f} to {np.nanmax(lat_2d):.2f}°")
        # print(f"Longitude range: {np.nanmin(lon_2d):.2f} to {np.nanmax(lon_2d):.2f}°")
        # print(f"Valid coordinate pixels: {np.sum(valid_mask):,} / {valid_mask.size:,}")
        # sleep(10)
        
        return coords_stacked
        
    except Exception as e:
        raise RuntimeError(f"Error extracting coordinates: {e}")


def read_and_scale_nwcsaf_variable(nc_filename, variable_name):
    """
    Read and scale a specific variable directly from NWCSAF NetCDF file.
    
    Args:
        nc_filename (str): Path to NWCSAF NetCDF file
        variable_name (str): Name of variable to read (e.g., 'ctth_alti', 'ctth_tempe', 'cmic_phase', 'cmic_cot')
    
    Returns:
        dict: Dictionary containing:
            - 'data': numpy array with scaled physical values (NaN for invalid data)
            - 'longitude': 2D longitude grid in degrees
            - 'latitude': 2D latitude grid in degrees
            - 'metadata': Dictionary with scaling parameters and variable info
            - 'units': Physical units of the data
    
    Examples:
        >>> # Read cloud top height
        >>> result = read_and_scale_nwcsaf_variable_direct("S_NWC_CTTH_MSG3_Europe-VISIR_20240613T000000Z.nc", "ctth_alti")
        >>> altitude_data = result['data']  # Physical values in meters
        >>> longitude = result['longitude']
        >>> latitude = result['latitude']
        >>> print(f"Altitude range: {np.nanmin(altitude_data):.1f} to {np.nanmax(altitude_data):.1f} {result['units']}")
        
        >>> # Read cloud phase (categorical data)
        >>> result = read_and_scale_nwcsaf_variable_direct("S_NWC_CMIC_MSG3_Europe-VISIR_20240613T000000Z.nc", "cmic_phase")
        >>> phase_data = result['data']
    """
    
    print(f"Reading variable '{variable_name}' directly from {os.path.basename(nc_filename)}")
    
    try:
        # Step 1: Open NetCDF file and extract variable data
        # print(f"Step 1: Opening NetCDF file and reading variable data...")
        
        ds = xr.open_dataset(nc_filename, decode_cf=True)
        
        # Check if variable exists
        if variable_name not in ds.data_vars:
            available_vars = list(ds.data_vars.keys())
            ds.close()
            raise ValueError(f"Variable '{variable_name}' not found. Available variables: {available_vars}")
        
        # Get the variable
        var = ds[variable_name]
        raw_data = var.values
        raw_data = np.nan_to_num(raw_data, nan=0.0)  # Convert NaNs to 0 for processing
        
        # print(f"  Raw data shape: {raw_data.shape}")
        # print(f"  Raw data type: {raw_data.dtype}")
        # print(f"  Raw data range: {np.min(raw_data):.2f} to {np.max(raw_data):.2f}")
        # sleep(5)
        
        # Step 2: Extract scaling parameters
        # print(f"Step 2: Extracting scaling parameters...")
        scaling_params = _get_nwcsaf_scaling_parameters(var, variable_name)
        
        # Step 3: Get coordinate information
        # print(f"Step 3: Extracting coordinate information...")
        
        # Get coordinate arrays
        if 'nx' in ds.coords and 'ny' in ds.coords:
            x_coords = ds['nx'].values
            y_coords = ds['ny'].values
        elif 'X' in ds.coords and 'Y' in ds.coords:
            x_coords = ds['X'].values
            y_coords = ds['Y'].values
        else:
            ds.close()
            raise ValueError("Could not find x,y coordinates in NetCDF file")
        
        ds.close()
        
        # Create 2D coordinate grids if needed
        if len(x_coords.shape) == 1 and len(y_coords.shape) == 1:
            x_2d, y_2d = np.meshgrid(x_coords, y_coords)
        else:
            x_2d, y_2d = x_coords, y_coords
        
        # Transform coordinates to lat/lon
        # print("  Transforming coordinates to WGS84...")
        geos_proj = "+proj=geos +a=6378137.000000 +b=6356752.300000 +lon_0=0.000000 +h=35785863.000000 +sweep=y"
        wgs84_proj = "+proj=longlat +datum=WGS84"
        
        transformer = Transformer.from_crs(geos_proj, wgs84_proj, always_xy=True)
        lon_2d, lat_2d = transformer.transform(x_2d, y_2d)
        
        # Handle invalid coordinates
        valid_mask = np.isfinite(lon_2d) & np.isfinite(lat_2d)
        lon_2d = np.where(valid_mask, lon_2d, np.nan)
        lat_2d = np.where(valid_mask, lat_2d, np.nan)
        
        # print(f"  Coordinate transformation complete")
        # print(f"  Latitude range: {np.nanmin(lat_2d):.2f} to {np.nanmax(lat_2d):.2f}°")
        # print(f"  Longitude range: {np.nanmin(lon_2d):.2f} to {np.nanmax(lon_2d):.2f}°")
        # sleep(10)
        
        # Step 4: Apply scaling to data
        # print(f"Step 4: Applying NWCSAF scaling...")
        scaled_data = _apply_nwcsaf_scaling(raw_data, scaling_params, variable_name)
        
        # Step 5: Prepare results
        # print(f"Step 5: Preparing results...")
        
        valid_data = scaled_data[~np.isnan(scaled_data)]
        if len(valid_data) > 0:
            data_min, data_max = np.min(valid_data), np.max(valid_data)
            # print(f"✓ Successfully processed {variable_name}")
            # print(f"  Final data range: {data_min:.2f} to {data_max:.2f} {scaling_params.get('units', '')}")
            # print(f"  Valid pixels: {len(valid_data):,} / {scaled_data.size:,}")
            # print(f"  Data coverage: {len(valid_data)/scaled_data.size*100:.1f}%")
        else:
            raise ValueError("⚠ Warning: No valid data after scaling")
        
        # Create metadata dictionary
        metadata = {
            'variable_name': variable_name,
            'filename': os.path.basename(nc_filename),
            'scaling_parameters': scaling_params,
            'coordinate_system': 'WGS84 (EPSG:4326)',
            'processed_shape': scaled_data.shape,
            'processing_method': 'direct_netcdf'
        }
        
        return {
            'data': scaled_data,
            'longitude': lon_2d,
            'latitude': lat_2d,
            'metadata': metadata,
            'units': scaling_params.get('units', 'unknown')
        }
        
    except Exception as e:
        raise RuntimeError(f"Error processing variable '{variable_name}': {e}")


def _get_nwcsaf_scaling_parameters(var, variable_name):
    """
    Extract scaling parameters directly from xarray variable.
    """
    scaling_params = {
        'scale_factor': var.attrs.get('scale_factor', 1.0),
        'add_offset': var.attrs.get('add_offset', 0.0),
        'fill_value': var.attrs.get('_FillValue', None),
        'valid_range': var.attrs.get('valid_range', None),
        'valid_min': var.attrs.get('valid_min', None),
        'valid_max': var.attrs.get('valid_max', None),
        'units': var.attrs.get('units', 'unknown'),
        'long_name': var.attrs.get('long_name', variable_name),
        'standard_name': var.attrs.get('standard_name', '')
    }
    
    # print(f"  Scaling parameters for '{variable_name}':")
    # for key, value in scaling_params.items():
        # if value is not None and value != 'unknown' and value != '':
        # print(f"    {key}: {value}")
        # sleep(2)
    
    return scaling_params


def _apply_nwcsaf_scaling(raw_data, scaling_params, variable_name):
    """
    Apply NWCSAF scaling directly to raw data.
    """
    # print(f"  Converting raw values to physical values for {variable_name}...")
    
    # Convert to float64 for precision
    scaled_data = raw_data.astype(np.float64)
    
    # print(f"  Raw data range: {np.nanmin(scaled_data):.2f} to {np.nanmax(scaled_data):.2f}")
    
    # Apply scaling formula: physical_value = (raw_value * scale_factor) + add_offset
    scale_factor = scaling_params.get('scale_factor', 1.0)
    add_offset = scaling_params.get('add_offset', 0.0)
    
    if scale_factor != 1.0 or add_offset != 0.0:
        print(f"    Applying scaling: data * {scale_factor} + {add_offset}")
        scaled_data = (scaled_data * scale_factor) + add_offset
    else:
        print(f"    No scaling needed (scale_factor=1.0, add_offset=0.0)")
    
    print(f"  Scaled data range: {np.min(scaled_data):.2f} to {np.max(scaled_data):.2f}")
    # sleep(5)
    
    return scaled_data

