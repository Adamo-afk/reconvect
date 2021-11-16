from bisect import bisect_left
from datetime import datetime, timedelta
from itertools import chain
import os

import dask
import netCDF4
from numba import njit, prange
import numpy as np
from scipy.signal import convolve
from skimage.measure import label, regionprops
from skimage.morphology import closing

from .utils import average_pool, mode_pool, fill_holes
from .utils import log_scale_with_zero, log_quantize_with_zero


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


def save_patches_radar(patches, archive_path, out_dir, suffix="2020"):
    from ..datasets import mchradar

    variables = ["RZC", "CZC", "BZC", "EZC-20", "EZC-45", "HZC", "LZC", "CPCH", "AREA57"]
    source_vars = {
        "AREA57": "CZC"
    }
    reader_vars = list((set(variables) - set(source_vars.keys())) | set(source_vars.values()))
    mchradar_reader = mchradar.MCHRadarReader(
        archive_path=archive_path,
        variables=reader_vars,
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
        "AREA57": lambda x: np.count_nonzero(x > 0)
    }
    postproc = {
        # 165 is the equivalent of 57 dBZ
        "AREA57": area_threshold(threshold=165, rad=10)
    }
    zero_value = {v: 0 for v in variables}    
    zero_value["RZC"] = 1
    zero_value["CZC"] = 1
    zero_value["CPCH"] = 1

    save_patches_all(mchradar_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        source_vars=source_vars, postproc=postproc)


def save_patches_lightning(patches, archive_path, out_dir, suffix="2020"):
    from .. import projection
    from ..datasets import mchlightning

    grid_projection = projection.GridProjection(
        projection.ccs4_swiss_grid_area)
    mchlightning_reader = mchlightning.MCHLightningReader(
        grid_projection, archive_path=archive_path,
        variables=["density", "current", "occurrence-8-10"])

    postproc = {
        "occurrence-8-10": lambda x: x.astype(np.uint8),
        "density": lambda x: log_quantize_with_zero(x, (0.001, 50.0))[0],
        "current": lambda x: log_quantize_with_zero(x, (0.001, 500.0))[0],
    }
    variables = list(postproc.keys())
    nonzero_count_func = {
        var_name: np.count_nonzero for var_name in variables
    }
    zero_value = {var_name: 0 for var_name in variables}
    scale = {
        "occurrence-8-10": np.array([0, 1], dtype=np.float32),
        "density": log_scale_with_zero((0.001, 50.0)),
        "current": log_scale_with_zero((0.001, 500.0))
    }

    save_patches_all(mchlightning_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        postproc=postproc, scale=scale)


def save_patches_msg(patches, archive_path, out_dir, suffix="2020",
    parallel=False):

    from .. import projection
    from ..datasets import msgccs4

    msg_reader = msgccs4.MSGRadianceCCS4Reader(archive_path=archive_path)

    variables = [
        "HRV", "VIS006", 
        "VIS008", "IR_016", "IR_039",
        "IR_087", "IR_097", "IR_108",
        "IR_120", "IR_134", "WV_062", "WV_073"
    ]
    nonzero_count_func = {
        v: lambda x: np.count_nonzero(x) for v in variables
    }
    zero_value = {v: 0 for v in variables
    }
    pool = {
        v: lambda x: average_pool(x, factor=4, missing=0) for v in variables
    }
    pool["HRV"] = lambda x: x

    save_patches_all(msg_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        pool=pool, parallel=parallel)


def save_patches_nwcsaf(patches, archive_path, out_dir, suffix="2020"):
    from .. import projection
    from ..datasets import msgccs4

    grid_projection = projection.GridProjection(
        projection.ccs4_swiss_grid_area)
    nwcsaf_reader = msgccs4.NWCSAFCCS4Reader(
        grid_projection, archive_path=archive_path)

    #variables = ["ctth_alti", "ctth_tempe", "cmic_phase", "cmic_cot"]
    variables = ["cmic_cot"]
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

    save_patches_all(nwcsaf_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        pool=pool, postproc=postproc)


def save_patches_cosmo(patches, archive_path, out_dir, suffix="2020"):
    from ..datasets import cosmonwp

    cosmonwp_reader = cosmonwp.COSMOCCS4Reader(
        archive_path=archive_path, cache_size=6000)

    # we only get data for every hour, so modify patches
    cosmo_patches = {}
    for (dt,pset) in patches.items():
        dt0 = datetime(dt.year, dt.month, dt.day, dt.hour)
        dt1 = dt0 + timedelta(hours=1)
        if dt0 not in cosmo_patches:
            cosmo_patches[dt0] = set()
        if dt1 not in cosmo_patches:
            cosmo_patches[dt1] = set()
        cosmo_patches[dt0].update(pset)
        cosmo_patches[dt1].update(pset)

    variables = [
        "CAPE_MU", "CIN_MU", "SLI", 
        "HZEROCL", "LCL_ML", "MCONV", "OMEGA",
        "T_2M", "T_SO", "SOILTYP"
    ]
    count_positive = lambda x: np.count_nonzero(x>0)
    all_nonzero = lambda x: np.prod(x.shape)
    nonzero_count_func = {
        "CAPE_MU": lambda x: count_positive,
        "CIN_MU": lambda x: count_positive,
        "SLI": all_nonzero,        
        "HZEROCL": lambda x: count_positive,
        "LCL_ML": count_positive,
        "MCONV": all_nonzero,
        "OMEGA": all_nonzero,
        "T_2M": all_nonzero,
        "T_SO": all_nonzero,
        "SOILTYP": lambda x: np.count_nonzero(x!=5)
    }
    zero_value = {v: 0 for v in variables}
    zero_value["SOILTYP"] = 5
    avg_pool = lambda x: average_pool(x, factor=4,
        missing=np.float32(-3.4028235e+38))
    pool = {v: avg_pool for v in variables}
    pool["SOILTYP"] = None

    save_patches_all(cosmonwp_reader, cosmo_patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix, pool=pool)


def save_patches_dem(patches, dem_path, out_dir, suffix="2020"):
    from ..datasets import swissdem

    variables = ["Altitude", "EW_deriv", "NS_deriv"]
    swissdem_reader = swissdem.SwissDEMReader(dem_file=dem_path, variables=variables)

    # data is static
    min_time = min(patches.keys())
    dem_patches = {min_time: set()}
    for (dt,pset) in patches.items():        
        dem_patches[min_time].update(pset)
    
    count_positive = lambda x: np.count_nonzero(x>0)
    all_nonzero = lambda x: np.prod(x.shape)
    nonzero_count_func = {v: lambda x: np.prod(x.shape) for v in variables}
    zero_value = {v: 0 for v in variables}

    save_patches_all(swissdem_reader, dem_patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix)


def save_patches_solar(patches, out_dir, suffix="2020"):
    from .. import projection
    from ..datasets import solar

    grid_projection = projection.GridProjection(
        projection.ccs4_swiss_grid_area)
    solar_reader = solar.SolarReader(grid_projection=grid_projection)

    variables = ["sun_z"]
    pool = {"sun_z": lambda x: average_pool(x, factor=4)}
    nonzero_count_func = nonzero_count_func = {
        v: lambda x: np.count_nonzero(x) for v in variables
    }
    zero_value = {v: 0 for v in variables}
    postproc = {"sun_z": lambda x: (x*127).round().astype(np.int8)}

    save_patches_all(solar_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        pool=pool, postproc=postproc)


def save_patches_all(
    reader, patches, variables, nonzero_count_func, zero_value,
    out_dir, suffix, epoch=datetime(1970,1,1), postproc={}, scale=None,
    pool={}, source_vars={}, parallel=False
):

    def save_var(var_name):
        src_name = source_vars.get(var_name, var_name)

        (patch_data, patch_coords, patch_times, 
            zero_patch_coords, zero_patch_times) = get_patches(
                reader, src_name, patches, 
                nonzero_count_func=nonzero_count_func[var_name],
                postproc=postproc.get(var_name),
                pool=pool.get(var_name)
            )
        try:
            time = epoch + timedelta(seconds=int(patch_times[0]))
            var_scale = reader.get_scale(time, var_name)
        except (AttributeError, KeyError):
            var_scale = None if (scale is None) else scale[var_name]
            pass
        
        out_fn = "patches_{}_{}.nc".format(var_name.replace("_", "-"), suffix)
        out_fn = os.path.join(out_dir, out_fn)
        
        save_patches(
            patch_data, patch_coords, patch_times,
            zero_patch_coords, zero_patch_times, out_fn,
            zero_value=zero_value[var_name], scale=var_scale
        )

    if parallel:
        save_var = dask.delayed(save_var)

    jobs = [save_var(v) for v in variables]
    if parallel:
        dask.compute(jobs, scheduler='threads')


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


def get_patches(reader, variable, patches,
    patch_shape=(32,32), nonzero_count_func=None,
    epoch=datetime(1970,1,1), postproc=None,
    pool=None
    ):
    num_patches = sum(len(patches[t]) for t in patches)
    patch_data = []
    patch_coords = []
    patch_times = []
    zero_patch_coords = []
    zero_patch_times = []
    
    if hasattr(reader, "phys_values"):
        phys_values = reader.phys_values
    
    k = 0
    try:
        if hasattr(reader, "phys_values"):
            reader.phys_values = False
        for (t, p_coord) in patches.items():
            try:
                data = reader.variable_for_time(t, variable)
            except (ValueError, FileNotFoundError, KeyError, OSError):
                continue

            if postproc is not None:
                data = postproc(data)

            time_sec = np.int64((t-epoch).total_seconds())
            for (pi, pj) in p_coord:
                if k % 100000 == 0:
                    print("{}: {}/{}".format(t, k, num_patches))
                patch_box = data[
                    pi*patch_shape[0]:(pi+1)*patch_shape[0],
                    pj*patch_shape[1]:(pj+1)*patch_shape[1],
                ].copy()
                if (nonzero_count_func is not None) and (nonzero_count_func(patch_box) == 0):
                    zero_patch_coords.append((pi,pj))
                    zero_patch_times.append(time_sec)
                else:
                    if pool is not None:
                        patch_box = pool(patch_box)
                    patch_data.append(patch_box)
                    patch_coords.append((pi,pj))
                    patch_times.append(time_sec)
                k += 1
                
    finally:
        if hasattr(reader, "phys_values"):
            reader.phys_values = phys_values

    if zero_patch_coords:
        zero_patch_coords = np.stack(zero_patch_coords, axis=0).astype(np.uint16)
        zero_patch_times = np.stack(zero_patch_times, axis=0)
    else:
        zero_patch_coords = np.zeros((0,2), dtype=np.uint16)
        zero_patch_times = np.zeros((0,), dtype=np.int64)
    patch_data = np.stack(patch_data, axis=0)
    patch_coords = np.stack(patch_coords, axis=0).astype(np.uint16)
    patch_times = np.stack(patch_times, axis=0)

    return (patch_data, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times)


def save_patches(patch_data, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times, out_fn, zero_value=0, scale=None):

    with netCDF4.Dataset(out_fn, 'w') as ds:
        dim_patch = ds.createDimension("dim_patch", patch_data.shape[0])
        dim_zero_patch = ds.createDimension("dim_zero_patch", zero_patch_coords.shape[0])
        dim_coord = ds.createDimension("dim_coord", 2)
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


def unpack_patches(patch_data):
    return (
        patch_data["patches"],
        patch_data["patch_coords"],
        patch_data["patch_times"],
        patch_data["zero_patch_coords"],
        patch_data["zero_patch_times"]
    )

def box_locations(patch_coords, patch_times, 
    zero_patch_coords, zero_patch_times,
    interval=timedelta(minutes=5),
    box_size=(24,8,8)
    ):
    t0 = min(patch_times.min(), zero_patch_times.min())
    t1 = max(patch_times.max(), zero_patch_times.max())
    dt = int(interval.total_seconds())
    t_size = (t1-t0) // dt + 1    

    (i1,j1) = patch_coords.max(axis=0)
    (zi1,zj1) = zero_patch_coords.max(axis=0)
    i1 = max(i1,zi1)+1
    j1 = max(j1,zj1)+1

    loc = np.zeros((t_size,i1,j1), dtype=bool)

    times_coords = chain(
        zip(patch_times, patch_coords),
        zip(zero_patch_times, zero_patch_coords)
    )
    for (t,(i,j)) in times_coords:
        tc = (t-t0) // dt
        loc[tc,i,j] = True

    (dtc,di,dj) = box_size
    kernel = np.ones((di,dj), dtype=np.uint32)
    kernel_size = di*dj
    box_locs = {}
    (gi,gj) = np.mgrid[:loc.shape[1]-di+1,:loc.shape[2]-dj+1]
    for (tc0,lt) in enumerate(loc):
        tc1 = tc0 + dtc
        if tc1 >= loc.shape[0]:
            continue

        f = convolve(lt, kernel, mode='valid', method='direct')
        candidates = (f == kernel_size)
        
        i_list = gi[candidates]
        if len(i_list) == 0:
            continue
        j_list = gj[candidates]

        for (i0,j0) in zip(i_list,j_list):
            i1 = i0 + di
            j1 = j0 + dj
            box = loc[tc0:tc1,i0:i1,j0:j1]
            assert(box.shape[1]==di)
            assert(box.shape[2]==dj)
            if box.all():
                if tc0 not in box_locs:
                    box_locs[tc0] = []
                box_locs[tc0].append((i0,j0))

    return (box_locs, t0)

