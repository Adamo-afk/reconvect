from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
import os
import dask
import netCDF4
# from numba import njit, prange
import numpy as np
import pandas as pd
from scipy.signal import convolve
from skimage.measure import label, regionprops
from skimage.morphology import closing
from scipy.interpolate import interp1d

from .utils import average_pool, mode_pool, fill_holes
from .utils import (
    log_scale_with_zero, log_quantize_with_zero,
    rzc_transform, czc_transform, lzc_transform, ezc_hzc_transform, bzc_transform, cpch_transform,
    cloud_top_temperature_transform, cloud_top_height_transform, cloud_optical_thickness_transform
)

from time import sleep


def area_threshold(threshold=10, rad=10, missing=255):
    (i,j) = np.mgrid[-rad:rad+1,-rad:rad+1]
    kernel = ((i**2+j**2) <= rad**2).astype(np.uint16)

    def func(x):        
        above_threshold = (x >= threshold) & (x != missing)
        if not above_threshold.any():
            return above_threshold.astype(np.uint16)
        else:
            return convolve(above_threshold.astype(np.uint16), 
                kernel, mode='same').round().astype(np.uint16)
    
    return func


def save_patches_radar(
    patches, archive_path, netcdf_dirs, out_dir, suffix="2020", timestamp=None
):
    from c4dl.datasets import mchradar

    variables = ["RZC", "CZC", "BZC", "EZC-20", "EZC-45", "HZC", "LZC", "CPCH"]
    
    # source_vars = {
    #     "AREA57": "CZC"
    # }
    # reader_vars = list((set(variables) - set(source_vars.keys())) | set(source_vars.values()))
    
    if patches == None and archive_path == None and out_dir == None:
        return variables
    else:
        mchradar_reader = mchradar.MCHRadarReader(
            archive_path=archive_path,
            variables=variables,
            phys_values=False
        )

        ezc_nonzero_count_func = lambda x: np.count_nonzero((x >= 1) & (x<251))
        
        nonzero_count_func = {
            "RZC": lambda x: np.count_nonzero(x > 1),
            "CPCH": lambda x: np.count_nonzero(x > 1),
            "CZC": lambda x: np.count_nonzero((x >= 10) & (x<251)),
            "BZC": lambda x: np.count_nonzero((x >= 1) & (x<=100)),
            "EZC-20": ezc_nonzero_count_func,
            "EZC-45": ezc_nonzero_count_func,
            "HZC": ezc_nonzero_count_func,
            "LZC": lambda x: np.count_nonzero((x > 1) & (x<251)),
            # "AREA57": lambda x: np.count_nonzero(x > 0)
        }

        scale = {
            "RZC": rzc_transform((0.01, 100.0)),  # Rain rate range
            "CPCH": cpch_transform((0.1, 20000.0)),  # Exponential CPCH range
            "BZC": bzc_transform((0, 100.0)),     # Linear BZC range
            "CZC": czc_transform((-5, 70.0)),     # Reflectivity range
            "LZC": lzc_transform((0.5, 10.0)),    # Liquid water content range
            "EZC-20": ezc_hzc_transform((0, 20.0)),  # Echo top height range
            "EZC-45": ezc_hzc_transform((0, 20.0)),  # Echo top height range
            "HZC": ezc_hzc_transform((0, 20.0))  # Echo top height range
        }

        # postproc = {
        #     # 165 is the equivalent of 57 dBZ
        #     "AREA57": area_threshold(threshold=165, rad=10)
        # }

        zero_value = {v: 0 for v in variables}    
        zero_value["RZC"] = 1
        zero_value["CZC"] = 1
        zero_value["CPCH"] = 1

        print("Saving patches")
        save_patches_all(
            mchradar_reader, patches, variables,
            nonzero_count_func, zero_value, out_dir, suffix,
            archive_path=archive_path, timestamp=timestamp,
            netcdf_dirs=netcdf_dirs, scale=scale
            # postproc=postproc
        ) # here archive_path is passed
        return variables


def save_patches_lightning(
    patches, archive_path, netcdf_dirs, out_dir, discharge_type="CG", suffix="2020", timestamp=None
):
    from c4dl import projection
    from c4dl.datasets import mchlightning

    variables = ["occurrence-8-10", "density", "current"]

    if patches == None and archive_path == None and out_dir == None:
        return variables
    else:
        grid_projection = projection.GridProjection(
            projection.romania_grid_area) # grid projection for Romania
        
        mchlightning_reader = mchlightning.MCHLightningReader(
            grid_projection, archive_path=archive_path,
            variables=variables, discharge_type=discharge_type)

        # postproc = {
        #     "occurrence-8-10": lambda x: np.round(x).astype(np.uint8),
        #     "density": lambda x: log_quantize_with_zero(x, (0.001, 50.0))[0], 
        #     "current": lambda x: log_quantize_with_zero(x, (0.001, 500.0))[0], 
        # }
        # variables = list(postproc.keys())
        nonzero_count_func = {
            var_name: lambda x: np.count_nonzero(x >= 0) for var_name in variables
        }
        
        zero_value = {var_name: 0 for var_name in variables}
        
        scale = {
            "occurrence-8-10": np.array([0, 1], dtype=np.float32),
            "density": log_scale_with_zero((0.001, 50.0)),
            "current": log_scale_with_zero((0.001, 500.0))
        }

        save_patches_all(
            mchlightning_reader, patches, variables,
            nonzero_count_func, zero_value, out_dir, suffix,
            scale=scale, archive_path=archive_path,
            netcdf_dirs=netcdf_dirs, timestamp=timestamp
            # postproc=postproc
        ) # here archive_path is passed
        return variables


def save_patches_msg(
    patches, archive_path, netcdf_dirs, out_dir, suffix="2020", parallel=False, timestamp=None
):
    # from c4dl import projection
    from c4dl.datasets import msgccs4
    
    variables = [
        "HRV", 
        "VIS006", "VIS008",
        "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
        "WV_062", "WV_073"
    ]

    if patches == None and archive_path == None and out_dir == None:
        return variables
    else:
        msg_reader = msgccs4.MSGRadianceCCS4Reader(archive_path=archive_path)
        
        nonzero_count_func = {
            v: lambda x: np.count_nonzero(x) for v in variables
        }
        
        zero_value = {
            v: 0 for v in variables
        }
        
        pool = {
            v: lambda x: average_pool(x, factor=4, missing=0) for v in variables
        }
        
        pool["HRV"] = lambda x: x

        save_patches_all(
            msg_reader, patches, variables,
            nonzero_count_func, zero_value, out_dir, suffix,
            pool=pool, parallel=parallel, archive_path=archive_path,
            netcdf_dirs=netcdf_dirs, timestamp=timestamp
        ) # here archive_path is passed
        return variables


def save_patches_nwcsaf(
    patches, archive_path, netcdf_dirs, out_dir, suffix="2020", timestamp=None
):
    from c4dl import projection
    from c4dl.datasets import msgccs4

    grid_projection = projection.GridProjection(
        projection.romania_grid_area)
    nwcsaf_reader = msgccs4.NWCSAFCCS4Reader(
        grid_projection, archive_path=archive_path)
    
    variables = ["ctth_alti", "ctth_tempe", "cmic_phase", "cmic_cot"]

    if patches == None and archive_path == None and out_dir == None:
        return variables
    else:
        nonzero_count_func = {
            "ctth_alti": lambda x: np.count_nonzero(x!=65535),
            "ctth_tempe": lambda x: np.count_nonzero(x!=65535),
            "cmic_phase": lambda x: np.count_nonzero(x!=4),
            "cmic_cot": lambda x: np.count_nonzero(x!=65535),
        }
        
        zero_value = {
            "ctth_alti": 65535,
            "ctth_tempe": 65535,
            "cmic_phase": 4,
            "cmic_cot": 65535,
        }
        
        pool = {
            "ctth_alti": lambda x: average_pool(x, factor=4, missing=65535),
            "ctth_tempe": lambda x: average_pool(x, factor=4, missing=65535),
            "cmic_phase": lambda x: mode_pool(x, factor=4),
            "cmic_cot": lambda x: average_pool(x, factor=4, missing=65535),
        }
        
        postproc = {
            "cmic_cot": fill_holes(missing=65535)
        }

        scale = {
            "ctth_alti": cloud_top_height_transform((-1000, 18000)), # Height range 
            "ctth_tempe": cloud_top_temperature_transform((200, 330)), # Temperature range
            "cmic_phase": None,  # No scaling needed
            "cmic_cot": cloud_optical_thickness_transform((0.1, 100.0)) # Thickness range
        }

        save_patches_all(
            nwcsaf_reader, patches, variables,
            nonzero_count_func, zero_value, out_dir, suffix,
            pool=pool, postproc=postproc, scale=scale,
            archive_path=archive_path, netcdf_dirs=netcdf_dirs, timestamp=timestamp
        )
        return variables


def save_patches_cosmo(
    patches, archive_path, netcdf_dirs, out_dir, suffix="2020", timestamp=None
):
    from c4dl.datasets import cosmonwp

    # # we only get data for every hour, so modify patches
    # cosmo_patches = {}
    # for (dt,pset) in patches.items():
    #     dt0 = datetime(dt.year, dt.month, dt.day, dt.hour)
    #     dt1 = dt0 + timedelta(hours=1)
    #     if dt0 not in cosmo_patches:
    #         cosmo_patches[dt0] = set()
    #     if dt1 not in cosmo_patches:
    #         cosmo_patches[dt1] = set()
    #     cosmo_patches[dt0].update(pset)
    #     cosmo_patches[dt1].update(pset)

    variables = [
        "CAPE_MU", "CIN_MU", "SLI", 
        "HZEROCL", "LCL_ML", "MCONV", "OMEGA",
        "T_2M", "T_SO", "SOILTYP"
    ]

    if patches == None and archive_path == None and out_dir == None:
        return variables
    else:
        cosmonwp_reader = cosmonwp.COSMOCCS4Reader(
            archive_path=archive_path, cache_size=6000
        )

        count_positive = lambda x: np.count_nonzero(x>0)
        all_nonzero = lambda x: np.prod(x.shape)
        nonzero_count_func = {
            "CAPE_MU": count_positive,
            "CIN_MU": count_positive,
            "SLI": all_nonzero,        
            "HZEROCL": count_positive,
            "LCL_ML": count_positive,
            "MCONV": all_nonzero,
            "OMEGA": all_nonzero,
            "T_2M": all_nonzero,
            "T_SO": all_nonzero,
            "SOILTYP": lambda x: np.count_nonzero(x!=5)
        }

        zero_value = {v: 0 for v in variables}
        zero_value["SOILTYP"] = 5
        avg_pool = lambda x: average_pool(x, factor=4, missing=np.float32(-3.4028235e+38))
        pool = {v: avg_pool for v in variables}
        pool["SOILTYP"] = None

        save_patches_all(
            cosmonwp_reader, patches, variables,
            nonzero_count_func, zero_value, out_dir, suffix, pool=pool,
            archive_path=archive_path, netcdf_dirs=netcdf_dirs, timestamp=timestamp
        )

        return variables


def save_patches_dem(
    patches, dem_path, out_dir, suffix="2020", timestamp=None
):
    from c4dl.datasets import swissdem

    variables = ["Altitude", "EW_deriv", "NS_deriv"]
    swissdem_reader = swissdem.SwissDEMReader(dem_file=dem_path, variables=variables)

    # data is static
    min_time = min(patches.keys())
    print(min_time)
    print(type(min_time))
    # dem_patches = {min_time: set()}
    dem_patches = {}
    for (dt, pset) in patches.items():        
        dem_patches[min_time] = pset # keep only unique patches (TODO)
    
    count_positive = lambda x: np.count_nonzero(x>0)
    all_nonzero = lambda x: np.prod(x.shape)
    nonzero_count_func = {v: lambda x: np.prod(x.shape) for v in variables}
    zero_value = {v: 0 for v in variables}

    save_patches_all(
        swissdem_reader, dem_patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix, timestamp=timestamp
    )
    return variables


def save_patches_solar(
    patches, out_dir, suffix="2020", timestamp=None
):
    from c4dl import projection
    from c4dl.datasets import solar

    grid_projection = projection.GridProjection(
        projection.romania_grid_area)
    solar_reader = solar.SolarReader(grid_projection=grid_projection)

    variables = ["sun_z"]
    pool = {"sun_z": lambda x: average_pool(x, factor=4)}
    nonzero_count_func = nonzero_count_func = {
        v: lambda x: np.count_nonzero(x) for v in variables
    }
    zero_value = {v: 0 for v in variables}
    postproc = {"sun_z": lambda x: (x*127).round().astype(np.int8)}

    save_patches_all(
        solar_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        pool=pool, postproc=postproc, timestamp=timestamp
    )
    return variables


# Consider using satpy library for more advanced image processing (interpolation)
def resize_patch(patch, target_shape=(32, 32)):
    """Vectorized version using interp1d for better performance"""
    h, w = patch.shape
    target_h, target_w = target_shape
    
    # Create coordinate arrays
    x_orig = np.arange(w)
    y_orig = np.arange(h)
    x_new = np.linspace(0, w-1, target_w)
    y_new = np.linspace(0, h-1, target_h)
    
    # Step 1: Interpolate along width (axis=1)
    width_interp = interp1d(x_orig, patch, axis=1, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    intermediate = width_interp(x_new)
    
    # Step 2: Interpolate along height (axis=0)  
    height_interp = interp1d(y_orig, intermediate, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    resized_patch = height_interp(y_new)
    
    return resized_patch


def extract_and_save_patches(
        datamap, patch_indices_dict, timestamp_hhmm, variable_name, base_output_dir="."
):
    """
    Save patches flattened.
    
    Args:
        datamap (numpy.ndarray): Full data array covering Romania area (H, W) or (H, W, C)
        patch_indices_dict (dict): Dictionary with ISO timestamps as keys and nested patch indices as values
                                  Format: {'2024-06-13T00:00:00.000000000': [[(y1,x1,y2,x2), ...], [(y1,x1,y2,x2), ...]], ...}
        timestamp_hhmm (str): Target timestamp in HHMM format (e.g., '0000', '0015', '1230')
        variable_name (str): Name of the variable (e.g., 'ctth_alti', 'ctth_tempe')
        base_output_dir (str, optional): Base directory for output. Defaults to current directory.
    
    Returns:
        dict: Information about the extraction process
    
    Examples:
        >>> # Flatten all patches into one file
        >>> result = extract_and_save_patches(datamap, patch_dict, '0000', 'ctth_alti')
    """
    
    print(f"Extracting grouped patches for variable '{variable_name}' at timestamp '{timestamp_hhmm}'")
    
    # Make a copy of the datamap to avoid modifying the original
    copy_datamap = datamap.copy()

    # Validate inputs (same as before)
    if not isinstance(copy_datamap, np.ndarray):
        raise TypeError("datamap must be a numpy array")
    
    if not isinstance(patch_indices_dict, dict):
        raise TypeError("patch_indices_dict must be a dictionary")
    
    if not isinstance(timestamp_hhmm, str) or len(timestamp_hhmm) != 4:
        raise ValueError("timestamp_hhmm must be a 4-character string (HHMM format)")
    
    if not timestamp_hhmm.isdigit():
        raise ValueError("timestamp_hhmm must contain only digits")
    
    # Find matching timestamp in dictionary
    target_time_pattern = f"T{timestamp_hhmm[:2]}:{timestamp_hhmm[2:]}:00.000000000"
    matched_iso_timestamp = None
    
    print(f"Looking for timestamp pattern: '*{target_time_pattern}'")
    
    for iso_timestamp in patch_indices_dict.keys():
        if target_time_pattern in iso_timestamp:
            matched_iso_timestamp = iso_timestamp
            break
    
    if matched_iso_timestamp is None:
        available_times = [ts.split('T')[1][:5].replace(':', '') for ts in patch_indices_dict.keys()]
        raise ValueError(f"Timestamp '{timestamp_hhmm}' not found. Available times: {available_times}")
    
    print(f"Found matching timestamp: {matched_iso_timestamp}")
    
    # Get nested patch indices for the matched timestamp
    patch_indices_nested = patch_indices_dict[matched_iso_timestamp]
    
    if not patch_indices_nested or len(patch_indices_nested) == 0:
        raise ValueError(f"No patch indices found for timestamp {matched_iso_timestamp}")
    
    print(f"Found {len(patch_indices_nested)} patch groups")
    
    # Create output directory
    folder_name = f"{variable_name}_extracted_patches"
    output_dir = Path(base_output_dir) / folder_name
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing directory: {output_dir}")
        
    # If there are multiple patch groups, flatten them
    if len(patch_indices_nested) > 1:
        flattened_patches = []
        for patch_group in patch_indices_nested:
            for patch in patch_group:
                flattened_patches.append(patch)
        
        print(f"Processing {len(flattened_patches)} patches for variable '{variable_name}'")
    
        group_patches = []
        valid_patches = 0
        
        for y1, x1, y2, x2 in flattened_patches:
            # Validate patch boundaries
            if len(copy_datamap.shape) == 3:
                copy_datamap = np.squeeze(copy_datamap, axis=0)
            elif len(copy_datamap.shape) == 4:
                copy_datamap = np.squeeze(copy_datamap, axis=0)
                copy_datamap = copy_datamap.transpose((1, 2, 0))  # (height, width, channels)
                # Global average on the last axis (channels)
                copy_datamap = np.mean(copy_datamap, axis=-1)

            if (y1 >= 0 and y2 <= copy_datamap.shape[0] and 
                x1 >= 0 and x2 <= copy_datamap.shape[1] and
                y2 > y1 and x2 > x1):
                
                # Extract patch
                patch = copy_datamap[y1:y2, x1:x2]
                patch = resize_patch(patch, (32, 32))
                
                group_patches.append(patch)
                valid_patches += 1
            else:
                print(f"Warning: Invalid patch boundaries [{y1}:{y2}, {x1}:{x2}] for datamap shape {copy_datamap.shape}")
    else:
        group_patches = []
        valid_patches = 0

        for y1, x1, y2, x2 in patch_indices_nested[0]:
            # Validate patch boundaries
            if len(copy_datamap.shape) == 3:
                copy_datamap = np.squeeze(copy_datamap, axis=0)
            elif len(copy_datamap.shape) == 4:
                copy_datamap = np.squeeze(copy_datamap, axis=0)
                copy_datamap = copy_datamap.transpose((1, 2, 0))  # (height, width, channels)
                # Global average on the last axis (channels)
                copy_datamap = np.mean(copy_datamap, axis=-1)

            if (y1 >= 0 and y2 <= copy_datamap.shape[0] and 
                x1 >= 0 and x2 <= copy_datamap.shape[1] and
                y2 > y1 and x2 > x1):
                
                # Extract patch
                patch = copy_datamap[y1:y2, x1:x2]
                patch = resize_patch(patch, (32, 32))
                
                group_patches.append(patch)
                valid_patches += 1
            else:
                print(f"Warning: Invalid patch boundaries [{y1}:{y2}, {x1}:{x2}] for datamap shape {copy_datamap.shape}")
    
    print(f"Valid patches {valid_patches} ")

    if valid_patches > 0:
        # Convert to numpy array
        patches_array = np.array(group_patches)
        print(f"Patches array: {patches_array}")
        print(f"Extracted {valid_patches} valid patches with shape: {patches_array.shape}")

        # Create output filename for this group
        output_filename = f"{timestamp_hhmm}_{variable_name}_extracted_patches.npy"
        output_path = output_dir / output_filename
        
        # Save patches to .npy file
        np.save(output_path, patches_array)
        
        print(f"✓ Saved ({valid_patches} patches) to: {output_path}")


# Here starts archive_path=None
def save_patches_all(
    reader, patches, variables, nonzero_count_func, zero_value,
    out_dir, suffix, epoch=datetime(1970,1,1), postproc={}, scale=None,
    pool={}, source_vars={}, parallel=False, archive_path=None, netcdf_dirs=None, timestamp=None
):

    def save_var(var_name):
        src_name = source_vars.get(var_name, var_name)
        
        print("Getting patches for", var_name)
        (patch_data, patch_coords, patch_times, zero_patch_coords, zero_patch_times) = get_patches(
                reader, src_name, patches, 
                nonzero_count_func=nonzero_count_func[var_name],
                postproc=postproc.get(var_name),
                pool=pool.get(var_name),
                archive_path=archive_path,
                timestamp=timestamp
            )
        try:
            time = epoch + timedelta(seconds=int(patch_times[0]))
            var_scale = reader.get_scale(time, var_name)
        except (AttributeError, KeyError):
            var_scale = None if (scale is None) else scale[var_name]
            pass
        
        out_fn = "patches_{}_{}.nc".format(var_name.replace("_", "-"), suffix)
        out_fn = os.path.join(out_dir, out_fn)
        # exit(0) # stop here to check the output (saves the radar patches all the time)
        
        save_patches(
            patch_data, patch_coords, patch_times,
            zero_patch_coords, zero_patch_times, out_fn,
            zero_value=zero_value[var_name], scale=var_scale
        )

    if netcdf_dirs is not None:
        crt_var = os.path.split(netcdf_dirs)[0].split("\\")[-1]
        var_idx = [i for i, v in enumerate(variables) if crt_var in v]
        print(f"Current variable: {crt_var}, index: {var_idx}")
    else:
        print("No netcdf_dirs provided")

    if not archive_path:
        print("No archive path provided")
        print("Saving patch variables")
        print("Weather products for this case: DEM, Solar")
        jobs = [save_var(v) for v in variables]
        if parallel:
            dask.compute(jobs, scheduler='threads')
    else:
        if parallel:
            print("Parallel computation")
            save_var = dask.delayed(save_var)

        if isinstance(archive_path[0], str):
            file_path = archive_path[0]
        elif isinstance(archive_path[0], tuple):
            file_path = archive_path[0][0]

        if "ICON" not in os.path.split(file_path)[1] and 'CMIC' not in os.path.split(file_path)[1]:
            
            print("Saving patch variables")
            print("Weather products for this case: Radar, Lightning, Satellite")
            print(f"Current variable: {crt_var}")
            print(f"Variable index: {var_idx}")
            jobs = save_var(variables[var_idx[0]])
            if parallel:
                dask.compute(jobs, scheduler='threads')
        else:
            print("Saving patch variables")
            print("Weather products for this case: NWP, NWCSAF")
            jobs = [save_var(v) for v in variables]
            if parallel:
                dask.compute(jobs, scheduler='threads')


# Define archive_path=None here too
def get_patches(
    reader, variable, patches,
    patch_shape=(32,32), nonzero_count_func=None,
    epoch=datetime(1970,1,1), postproc=None,
    pool=None, archive_path=None, timestamp=None
):
    
    patch_data = []
    patch_coords = []  # Still stores (pi, pj) for compatibility
    patch_times = []
    zero_patch_coords = []  # Still stores (pi, pj) for compatibility
    zero_patch_times = []
    
    if hasattr(reader, "phys_values"):
        phys_values = reader.phys_values
    
    try:
        if hasattr(reader, "phys_values"):
            reader.phys_values = False


        for idx in range(len(list(patches.items()))):
            if archive_path is not None:
                data = reader.variable_for_time(list(patches.items())[idx][0], variable, archive_path[idx])
            else:
                data = reader.variable_for_time(list(patches.items())[idx][0], variable)
            
            if timestamp is not None:
                print(f"Extracting data for variable: {variable}")
                extract_and_save_patches(
                    datamap=data,
                    patch_indices_dict=patches,
                    timestamp_hhmm=timestamp,      
                    variable_name=variable
                )

            if postproc is not None:
                data = postproc(data)
            
            time_sec = np.int64(
                (pd.Timestamp(list(patches.items())[idx][0]).to_pydatetime()-epoch).total_seconds()
            )
            
            for (pi, pj, pi_end, pj_end) in list(patches.items())[idx][1][0]:
                if pi == pi_end or pj == pj_end:
                    continue

                # Extract patch using the actual coordinates from DBSCAN
                if len(data.shape) == 3:
                    patch_box = np.squeeze(data, axis=0)[
                        int(pi):int(pi_end),
                        int(pj):int(pj_end)
                    ].copy()
                elif len(data.shape) == 4:
                    patch_box = np.squeeze(data, axis=0)
                    patch_box = patch_box.transpose((1, 2, 0))
                    patch_box = np.mean(patch_box, axis=-1)[
                        int(pi):int(pi_end),
                        int(pj):int(pj_end)
                    ].copy()
                else:
                    patch_box = data[
                        int(pi):int(pi_end),
                        int(pj):int(pj_end)
                    ].copy()

                # Resize to standard 32x32 for consistency
                patch_box = resize_patch(patch_box, (32, 32))

                if (nonzero_count_func is not None) and (nonzero_count_func(patch_box) == 0):
                    # Store only top-left coordinates (pi, pj) for compatibility
                    zero_patch_coords.append((pi, pj))
                    zero_patch_times.append(time_sec)
                else:
                    if pool is not None:
                        patch_box = pool(patch_box)

                    patch_box = np.nan_to_num(patch_box, nan=0.0)
                    patch_data.append(patch_box)
                    # Store only top-left coordinates (pi, pj) for compatibility
                    patch_coords.append((pi, pj))
                    patch_times.append(time_sec)
                
    finally:
        if hasattr(reader, "phys_values"):
            reader.phys_values = phys_values

    # Keep existing 2-coordinate format
    if zero_patch_coords:
        zero_patch_coords = np.stack(zero_patch_coords, axis=0).astype(np.uint16)
        zero_patch_times = np.stack(zero_patch_times, axis=0)
    else:
        zero_patch_coords = np.zeros((0,2), dtype=np.uint16)
        zero_patch_times = np.zeros((0,), dtype=np.int64)
    
    print(f"Patch data length: {len(patch_data)}")
    if len(patch_data) > 0:
        patch_data = np.stack(patch_data, axis=0)
        patch_coords = np.stack(patch_coords, axis=0).astype(np.uint16)
        patch_times = np.stack(patch_times, axis=0)
    else:
        patch_data = np.zeros((0, 32, 32), dtype=np.float32)
        patch_coords = np.zeros((0,2), dtype=np.uint16)
        patch_times = np.zeros((0,), dtype=np.int64)
    
    print(f"Patch data shape: {patch_data.shape}")
    print(f"Patch coords shape: {patch_coords.shape}")

    return (patch_data, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times)


def save_patches(patch_data, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times, out_fn, zero_value=0, scale=None):
    print(f"Saving patches to {out_fn}")

    with netCDF4.Dataset(out_fn, 'w') as ds:
        dim_patch = ds.createDimension("dim_patch", patch_data.shape[0])
        dim_zero_patch = ds.createDimension("dim_zero_patch", zero_patch_coords.shape[0])
        dim_coord = ds.createDimension("dim_coord", 2)  # Keep as 2 for compatibility
        dim_height = ds.createDimension("dim_height", patch_data.shape[1])
        dim_width = ds.createDimension("dim_width", patch_data.shape[2])

        var_args = {"zlib": True, "complevel": 1}

        chunksizes = (min(2**10, patch_data.shape[0]), patch_data.shape[1], patch_data.shape[2])
        var_patch = ds.createVariable("patches", patch_data.dtype,
            ("dim_patch","dim_height","dim_width"), chunksizes=chunksizes, **var_args)
        var_patch[:] = patch_data
        
        var_patch_coord = ds.createVariable("patch_coords", patch_coords.dtype,
            ("dim_patch","dim_coord"), **var_args)
        var_patch_coord[:] = patch_coords

        var_patch_time = ds.createVariable("patch_times", patch_times.dtype,
            ("dim_patch",), **var_args)
        var_patch_time[:] = patch_times

        var_zero_patch_coord = ds.createVariable("zero_patch_coords", zero_patch_coords.dtype,
            ("dim_zero_patch","dim_coord"), **var_args)
        var_zero_patch_coord[:] = zero_patch_coords

        var_zero_patch_time = ds.createVariable("zero_patch_times", zero_patch_times.dtype,
            ("dim_zero_patch",), **var_args)
        var_zero_patch_time[:] = zero_patch_times

        ds.zero_value = zero_value

        if scale is not None:
            dim_scale = ds.createDimension("dim_scale", len(scale))
            var_scale = ds.createVariable("scale", scale.dtype, ("dim_scale",), **var_args)
            var_scale[:] = scale


# def box_locations(patch_coords, patch_times, 
#     zero_patch_coords, zero_patch_times,
#     interval=timedelta(minutes=5),
#     box_size=(24,8,8)
# ):
#     """
#     Modified box_locations to work with 2-coordinate format from DBSCAN patches.
#     Since patches are now 32x32 standardized, we can treat coordinates as discrete locations.
#     """
    
#     if len(patch_times) == 0 and len(zero_patch_times) == 0:
#         print("No patch data available!")
#         return ({}, 0)
    
#     t0 = min(patch_times.min(), zero_patch_times.min()) if len(patch_times) > 0 and len(zero_patch_times) > 0 else \
#          patch_times.min() if len(patch_times) > 0 else zero_patch_times.min()
#     t1 = max(patch_times.max(), zero_patch_times.max()) if len(patch_times) > 0 and len(zero_patch_times) > 0 else \
#          patch_times.max() if len(patch_times) > 0 else zero_patch_times.max()
         
#     dt = int(interval.total_seconds())
#     t_size = (t1-t0) // dt + 1    

#     # Find the maximum coordinates - these represent patch positions
#     all_coords = []
#     if len(patch_coords) > 0:
#         all_coords.extend(patch_coords)
#     if len(zero_patch_coords) > 0:
#         all_coords.extend(zero_patch_coords)
    
#     if len(all_coords) == 0:
#         print("No coordinate data available!")
#         return ({}, t0)
    
#     all_coords = np.array(all_coords)
#     i_max = all_coords[:, 0].max()
#     j_max = all_coords[:, 1].max()
#     i_min = all_coords[:, 0].min() 
#     j_min = all_coords[:, 1].min()
    
#     print(f"Coordinate ranges: i=({i_min}, {i_max}), j=({j_min}, {j_max})")
    
#     # Use a simpler approach - don't normalize by 32, use the coordinates as grid positions
#     # but scale them down to reasonable size
#     scale_factor = max(1, max(i_max, j_max) // 100)  # Scale down if coordinates are too large
#     print(f"Using scale factor: {scale_factor}")
    
#     grid_i1 = int(i_max // scale_factor) + 1
#     grid_j1 = int(j_max // scale_factor) + 1

#     print(f"Box location array size: ({t_size}, {grid_i1}, {grid_j1})")
#     loc = np.zeros((t_size, grid_i1, grid_j1), dtype=bool)

#     # Mark locations where patches exist
#     all_time_coords = []
#     if len(patch_coords) > 0 and len(patch_times) > 0:
#         all_time_coords.extend(zip(patch_times, patch_coords))
#     if len(zero_patch_coords) > 0 and len(zero_patch_times) > 0:
#         all_time_coords.extend(zip(zero_patch_times, zero_patch_coords))

#     patches_marked = 0
#     for (t, coords) in all_time_coords:
#         tc = (t-t0) // dt
#         if tc < 0 or tc >= loc.shape[0]:
#             continue
            
#         pi, pj = coords
#         # Scale coordinates to grid
#         grid_i = int(pi // scale_factor)
#         grid_j = int(pj // scale_factor)
        
#         if grid_i < loc.shape[1] and grid_j < loc.shape[2]:
#             loc[tc, grid_i, grid_j] = True
#             patches_marked += 1

#     print(f"Marked {patches_marked} patch locations in grid")
    
#     # Check if we have any data
#     if patches_marked == 0:
#         print("No patches were successfully marked in the grid!")
#         return ({}, t0)

#     # Use a more lenient box detection approach
#     (dtc, di, dj) = box_size
    
#     # If the required box size is too large for our data, reduce it
#     max_spatial_size = min(loc.shape[1], loc.shape[2])
#     if di > max_spatial_size or dj > max_spatial_size:
#         di = min(di, max_spatial_size)
#         dj = min(dj, max_spatial_size)
#         print(f"Reduced spatial box size to ({di}, {dj}) to fit available data")
    
#     box_locs = {}
    
#     # Use a simpler approach - look for any regions with reasonable coverage
#     for tc0 in range(loc.shape[0] - dtc + 1):
#         tc1 = tc0 + dtc
        
#         # Check temporal box
#         temporal_box = loc[tc0:tc1, :, :]
        
#         # Look for spatial regions with some coverage
#         spatial_coverage = np.sum(temporal_box, axis=0)
#         threshold = max(1, dtc // 4)  # At least 25% temporal coverage
        
#         # Find locations with sufficient coverage
#         valid_locations = np.where(spatial_coverage >= threshold)
        
#         if len(valid_locations[0]) > 0:
#             if tc0 not in box_locs:
#                 box_locs[tc0] = []
            
#             # Take up to 10 best locations to avoid too many
#             for idx in range(min(10, len(valid_locations[0]))):
#                 i0 = valid_locations[0][idx]
#                 j0 = valid_locations[1][idx]
                
#                 # Make sure we don't go out of bounds
#                 i0 = max(0, min(i0, loc.shape[1] - di))
#                 j0 = max(0, min(j0, loc.shape[2] - dj))
                
#                 box_locs[tc0].append((i0, j0))

#     print(f"Box locations: {len(box_locs)} time steps with valid locations")
#     if len(box_locs) > 0:
#         total_locations = sum(len(locations) for locations in box_locs.values())
#         print(f"Total spatial locations: {total_locations}")
    
#     return (box_locs, t0)

# def box_locations(
#     patch_coords, patch_times, 
#     zero_patch_coords, zero_patch_times,
#     interval=timedelta(minutes=5),
#     box_size=(24,8,8),  # (time, patches_i, patches_j)
#     patch_size=32
# ):
#     """
#     Find box locations using pixel coordinates from DBSCAN patches.
    
#     Args:
#         patch_coords: Pixel coordinates (N, 2) - upper-left corners
#         patch_times: Unix timestamps for each patch
#         zero_patch_coords: Pixel coordinates for zero patches
#         zero_patch_times: Timestamps for zero patches
#         box_size: (time_steps, n_patches_i, n_patches_j)
#         patch_size: Size of each patch in pixels (default 32)
    
#     Returns:
#         (box_locs, t0): Dictionary of {time_idx: [(pixel_i, pixel_j), ...]}
#     """
#     t0 = min(patch_times.min(), zero_patch_times.min())
#     t1 = max(patch_times.max(), zero_patch_times.max())
#     dt = int(interval.total_seconds())
#     t_size = (t1-t0) // dt + 1
    
#     # Find pixel extent
#     all_coords = np.vstack([patch_coords, zero_patch_coords])
#     pixel_i_max = all_coords[:, 0].max()
#     pixel_j_max = all_coords[:, 1].max()
    
#     # Convert to grid coordinates (divide by patch_size)
#     grid_i_max = (pixel_i_max // patch_size) + 1
#     grid_j_max = (pixel_j_max // patch_size) + 1
    
#     print(f"Box location array size: ({t_size}, {grid_i_max}, {grid_j_max})")
    
#     # Create boolean array indicating patch presence
#     loc = np.zeros((t_size, grid_i_max, grid_j_max), dtype=bool)
    
#     # Mark patch locations in grid
#     times_coords = chain(
#         zip(patch_times, patch_coords),
#         zip(zero_patch_times, zero_patch_coords)
#     )
    
#     for (t, (pi, pj)) in times_coords:
#         tc = (t-t0) // dt
#         # Convert pixel to grid coordinates
#         gi = pi // patch_size
#         gj = pj // patch_size
#         if 0 <= gi < grid_i_max and 0 <= gj < grid_j_max:
#             loc[tc, gi, gj] = True
    
#     # Find boxes where all patches are present
#     (dtc, di, dj) = box_size
#     kernel = np.ones((di, dj), dtype=np.uint32)
#     kernel_size = di * dj
    
#     box_locs = {}
#     (gi, gj) = np.mgrid[:loc.shape[1]-di+1, :loc.shape[2]-dj+1]
    
#     for (tc0, lt) in enumerate(loc):
#         tc1 = tc0 + dtc
#         if tc1 >= loc.shape[0]:
#             continue
        
#         # Find candidate box positions
#         f = convolve(lt, kernel, mode='valid', method='direct')
#         candidates = (f == kernel_size)
        
#         gi_list = gi[candidates]
#         if len(gi_list) == 0:
#             continue
        
#         gj_list = gj[candidates]
        
#         for (gi0, gj0) in zip(gi_list, gj_list):
#             gi1 = gi0 + di
#             gj1 = gj0 + dj
            
#             box = loc[tc0:tc1, gi0:gi1, gj0:gj1]
#             assert(box.shape[1] == di)
#             assert(box.shape[2] == dj)
            
#             if box.all():  # All patches present in spatiotemporal box
#                 if tc0 not in box_locs:
#                     box_locs[tc0] = []
#                 # Convert grid coordinates back to pixel coordinates
#                 pixel_i = gi0 * patch_size
#                 pixel_j = gj0 * patch_size
#                 box_locs[tc0].append((pixel_i, pixel_j))
    
#     return (box_locs, t0)

def box_locations(primary_patch_data, interval_seconds=300):
    """Just return all patches grouped by time - no rectangle requirement"""
    box_locs = {}
    box_locs_misaligned = {}
    t0 = primary_patch_data["patch_times"].min()
    
    for (t, coord) in zip(primary_patch_data["patch_times"], 
                           primary_patch_data["patch_coords"]):
        tc = int((t - t0) // interval_seconds)
        coord_misalligned = tuple(coord.astype(np.int32))
        coord_aligned = tuple(((coord // 32) * 32).astype(np.int32))
        
        if tc not in box_locs:
            box_locs[tc] = []
        if tc not in box_locs_misaligned:
            box_locs_misaligned[tc] = []
        if coord_aligned not in box_locs[tc]:
            box_locs[tc].append(coord_aligned)
        if coord_misalligned not in box_locs_misaligned[tc]:
            box_locs_misaligned[tc].append(coord_misalligned)

    print(f"Box locations (aligned):\n {box_locs[1]}")
    print(f"Box locations (misaligned):\n {box_locs_misaligned[1]}")
    # exit(0)
    
    return (box_locs_misaligned, int(t0))


def load_patches(fn, in_memory=True):
    if in_memory:
        with open(fn, 'rb') as f:
            ds_raw = f.read()
        fn = None
    else:
        ds_raw = None

    with netCDF4.Dataset(fn, 'r', memory=ds_raw) as ds:
        patch_data = {       
            "patches": np.array(ds["patches"]),
            "patch_coords": np.array(ds["patch_coords"]),
            "patch_times": np.array(ds["patch_times"]),
            "zero_patch_coords": np.array(ds["zero_patch_coords"]),
            "zero_patch_times": np.array(ds["zero_patch_times"]),
            "zero_value": ds.zero_value
        }
        if "scale" in ds.variables:
            patch_data["scale"] = np.array(ds["scale"])

    return patch_data


def locate_patches(
    regions,
    mask,
    patch_shape=(32,32),
    interval=timedelta(minutes=5),
    time_range=timedelta(hours=2),
    patch_distance=4,
):
    patch_mask = np.zeros(
        (mask.shape[0]//patch_shape[0], mask.shape[1]//patch_shape[1]),
        dtype=bool
    )
    for pi in range(patch_mask.shape[0]):
        for pj in range(patch_mask.shape[1]):
            mask_patch = mask[
                pi*patch_shape[0]:(pi+1)*patch_shape[0],
                pj*patch_shape[1]:(pj+1)*patch_shape[1], 
            ]
            patch_mask[pi,pj] = mask_patch.any()
    
    patches = {}

    for (time, (centroids, areas)) in regions.items():
        t0 = time - time_range
        t1 = time + time_range
        t = t0
        while t <= t1:
            if not t in patches:
                patches[t] = set()

            for (i,j) in centroids:
                pi = int(round(i)) // patch_shape[0]
                pj = int(round(j)) // patch_shape[1]
                
                box = (
                    (pi-patch_distance, pi+patch_distance+1),
                    (pj-patch_distance, pj+patch_distance+1),
                )
                box = adjust_box(box, patch_mask,
                    step_size=1, max_steps=patch_distance)
                if box is None:
                    continue

                ((pi0, pi1), (pj0, pj1)) = box
                for pi in range(pi0, pi1):
                    for pj in range(pj0, pj1):
                        patches[t].add((pi, pj))

            t += interval

    return patches


def unpack_patches(patch_data):
    return (
        patch_data["patches"],
        patch_data["patch_coords"],
        patch_data["patch_times"],
        patch_data["zero_patch_coords"],
        patch_data["zero_patch_times"]
    )


def image_regions(binary_img, min_area=0):
    binary_img = closing(binary_img)
    label_img = label(binary_img)
    regions = regionprops(label_img)
    regions = [r for r in regions if r.area>=min_area]
    areas = np.array([r.area for r in regions])
    centroids = np.array([r.centroid for r in regions])
    return (centroids, areas)


def all_centroids(
    reader,
    variable, 
    time_range,
    threshold,
    min_area=0,
    interval=timedelta(minutes=5)
):
    
    regions = {}
    missing_times = []
    t = time_range[0]
    while t < time_range[1]:
        if t.hour==0 and t.minute==0:
            print(t)
        try:
            img = reader.variable_for_time(t, variable)
            binary_img = (img > threshold)
            (centroids, areas) = image_regions(binary_img, min_area=min_area)
            if len(areas) > 0:
                regions[t] = (centroids, areas)
        except ValueError:
            missing_times.append(t)
            continue
        finally:
            t += interval

    return (regions, missing_times)


def adjust_box(box, mask, step_size=1, max_steps=1):
    ((i0,i1),(j0,j1)) = box
    i_size = i1-i0
    j_size = j1-j0

    # constrain box to bounds
    i0 = min(max(i0,0),mask.shape[0]-i_size)
    i1 = i0+i_size
    j0 = min(max(j0,0),mask.shape[1]-j_size)
    j1 = j0+j_size

    num_mask = np.count_nonzero(mask[i0:i1,j0:j1])
    if num_mask == 0:
        return ((i0,i1),(j0,j1))

    k = 0
    min_mask = (0,0,num_mask)

    while (min_mask[-1] > 0) and (k < max_steps):

        for di in [-step_size, 0, step_size]:
            it0 = i0+di
            it1 = it0+i_size
            if (it0 < 0) or (it1 >= mask.shape[0]):
                continue

            for dj in [-step_size, 0, step_size]:
                if di == dj == 0:
                    continue                    
                jt0 = j0+dj
                jt1 = jt0+j_size
                if (jt0 < 0) or (jt1 >= mask.shape[1]):
                    continue

                num_mask = np.count_nonzero(mask[it0:it1,jt0:jt1])
                if (num_mask < min_mask[-1]):
                    min_mask = (di,dj,num_mask) 

        (di,dj,num_mask) = min_mask
        i0 += di
        i1 = i0+i_size
        j0 += dj
        j1 = j0+j_size
        k += 1


    if num_mask == 0:
        return ((i0,i1),(j0,j1))
    else:
        return None



# if __name__ != '__main__':
#     # Only import when this module is imported, not when run directly
#     from c4dl.features.test_nc import args

