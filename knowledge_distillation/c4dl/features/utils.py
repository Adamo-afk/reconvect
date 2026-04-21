from numba import njit, prange
import numpy as np
from scipy.signal import convolve
from time import sleep


def log_quantize_with_zero(x, range, n=65536, dtype=np.uint16):
    scale = log_scale_with_zero(range, n=n, dtype=x.dtype)
    y = np.empty_like(x, dtype=dtype)
    
    # y = log_quant_with_zeros(x, y, np.log10(scale[1:]))
    log_quant_with_zeros(x, y, np.log10(scale[1:]))
    return (y, scale)


def log_scale_with_zero(range, n=65536, dtype=np.float32):
    # Ensure range[0] > 0 for log10
    if range[0] <= 0:
        range = (0.001, range[1])  # Use small positive value instead of 0
    scale = np.linspace(np.log10(range[0]), np.log10(range[1]), n-1)
    scale = np.hstack((0, 10**scale)).astype(dtype)
    return scale

# RADAR TRANSFORMATION FUNCTIONS

def rzc_transform(range, n=256, dtype=np.float32):
    """
    RZC (Rain rate) transformation: [log10(x) + 0.051]/0.528
    Fill value: 0.01 mm h^-1
    """
    if range[0] <= 0:
        range = (0.01, range[1])  # Use fill value for log10
    
    # Create linear scale in log space
    log_scale = np.linspace(np.log10(range[0]), np.log10(range[1]), n-1)
    # Apply transformation: [log10(x) + 0.051]/0.528
    transformed_scale = (log_scale + 0.051) / 0.528
    # Include zero at the beginning
    scale = np.hstack((0, transformed_scale)).astype(dtype)
    return scale


def czc_transform(range, n=256, dtype=np.float32):
    """
    CZC (Composite reflectivity) transformation: (x - 21.3 dBZ)/8.71 dBZ
    Fill value: -5 dBZ
    """
    # Create linear scale
    linear_scale = np.linspace(range[0], range[1], n-1)
    # Apply transformation: (x - 21.3)/8.71
    transformed_scale = (linear_scale - 21.3) / 8.71
    # Include fill value transformation at the beginning
    fill_transform = (-5 - 21.3) / 8.71
    scale = np.hstack((fill_transform, transformed_scale)).astype(dtype)
    return scale


def lzc_transform(range, n=256, dtype=np.float32):
    """
    LZC (Liquid water content) transformation: [log10(x) + 0.274]/0.135
    Fill value: 0.5 g m^-3
    """
    if range[0] <= 0:
        range = (0.5, range[1])  # Use fill value for log10
    
    # Create linear scale in log space
    log_scale = np.linspace(np.log10(range[0]), np.log10(range[1]), n-1)
    # Apply transformation: [log10(x) + 0.274]/0.135
    transformed_scale = (log_scale + 0.274) / 0.135
    # Include zero at the beginning
    scale = np.hstack((0, transformed_scale)).astype(dtype)
    return scale


def ezc_hzc_transform(range, n=256, dtype=np.float32):
    """
    EZC-20, EZC-45, and HZC transformation: x/1.97 km
    Fill value: 0
    """
    # Create linear scale
    linear_scale = np.linspace(range[0], range[1], n-1)
    # Apply transformation: x/1.97
    transformed_scale = linear_scale / 1.97
    # Include zero at the beginning
    scale = np.hstack((0, transformed_scale)).astype(dtype)
    return scale


# INFERRED TRANSFORMATION FUNCTIONS (based on observed data patterns)

def bzc_transform(range, n=256, dtype=np.float32):
    """
    BZC transformation: Linear scaling from 0 to max_value
    Inferred from observed linear pattern: 0 to 100
    """
    # Simple linear scaling
    linear_scale = np.linspace(range[0], range[1], n)
    return linear_scale.astype(dtype)


def cpch_transform(range, n=256, dtype=np.float32):
    """
    CPCH transformation: Exponential scaling
    Inferred from observed exponential pattern with ~8% growth rate
    Pattern: Many zeros followed by exponential growth to ~19210
    """
    # Create the scale array
    scale = np.zeros(n, dtype=dtype)
    
    # First ~1/3 are zeros (matching observed pattern)
    zero_count = n // 3
    
    # Exponential growth for the remainder
    exp_count = n - zero_count
    
    if exp_count > 0:
        # Use exponential spacing that matches the observed pattern
        # The growth pattern shows ~8% increase per step
        exp_values = np.logspace(np.log10(0.1), np.log10(range[1]), exp_count, base=10)
        scale[zero_count:] = exp_values
    
    return scale

# NWCSAF TRANSFORMATION FUNCTIONS

def cloud_top_temperature_transform(range, n=65536, dtype=np.float32):
    """
    Cloud-top temperature transformation: (x - 260 K)/19.1 K
    Fill value: 330 K
    """
    # Create linear scale
    linear_scale = np.linspace(range[0], range[1], n-1)
    # Apply transformation: (x - 260)/19.1
    transformed_scale = (linear_scale - 260) / 19.1
    # Include fill value transformation at the beginning
    fill_transform = (330 - 260) / 19.1
    scale = np.hstack((fill_transform, transformed_scale)).astype(dtype)
    return scale


def cloud_top_height_transform(range, n=65536, dtype=np.float32):
    """
    Cloud-top height transformation: (x - 5260 m)/2810 m
    Fill value: -1000 m
    """
    # Create linear scale
    linear_scale = np.linspace(range[0], range[1], n-1)
    # Apply transformation: (x - 5260)/2810
    transformed_scale = (linear_scale - 5260) / 2810
    # Include fill value transformation at the beginning
    fill_transform = (-1000 - 5260) / 2810
    scale = np.hstack((fill_transform, transformed_scale)).astype(dtype)
    return scale


def cloud_optical_thickness_transform(range, n=65536, dtype=np.float32):
    """
    Cloud optical thickness transformation: [log10(x) - 0.94]/0.588
    Fill value: 0.1
    """
    if range[0] <= 0:
        range = (0.1, range[1])  # Use fill value for log10
    
    # Create linear scale in log space
    log_scale = np.linspace(np.log10(range[0]), np.log10(range[1]), n-1)
    # Apply transformation: [log10(x) - 0.94]/0.588
    transformed_scale = (log_scale - 0.94) / 0.588
    # Include zero at the beginning
    scale = np.hstack((0, transformed_scale)).astype(dtype)
    return scale


# optimized helper function for the above
# @njit(parallel=True)
def log_quant_with_zeros(x, y, scale):
    x = x.ravel()
    y = y.ravel()
    min_val = 10**scale[0]
    
    for i in prange(x.shape[0]):
        # map small values to 0
        if x[i] < min_val:
            y[i] = 0
            continue
        
        lx = np.log10(x[i])
        if lx >= scale[-1]:
            # map too big values to max of scale
            y[i] = len(scale)
        else:
            # binary search for the rest
            k0 = 0
            k1 = len(scale)
            while k1-k0 > 1:
                km = k0 + (k1-k0)//2
                if lx < scale[km]:
                    k1 = km
                else:
                    k0 = km
            
            if k0 == len(scale)-1:
                q = k0
            elif k0 == 0:
                q = 0
            else:
                d0 = abs(lx-scale[k0])
                d1 = abs(lx-scale[k1])
                if d0 < d1:
                    q = k0
                else:
                    q = k1

            y[i] = q+1 # add 1 to leave space for zero
            # print(f"q value {q+1}")

    # return y


# @njit(parallel=True)
def average_pool(x, factor=2, missing=65535):
    y = np.empty((x.shape[0]//factor, x.shape[1]//factor), dtype=x.dtype)
    N = factor**2
    N_thresh = N//2

    for iy in prange(y.shape[0]):
        ix0 = iy * factor
        ix1 = ix0 + factor
        for jy in range(y.shape[1]):            
            jx0 = jy * factor
            jx1 = jx0 + factor
            v = float(0.0)
            num_valid = 0

            for ix in range(ix0, ix1):
                for jx in range(jx0, jx1):
                    if x[ix,jx] != missing:
                        v += x[ix,jx]
                        num_valid += 1
            
            if num_valid >= N_thresh:
                y[iy,jy] = v/num_valid
            else:
                y[iy,jy] = missing
        
    return y


# @njit(parallel=True)
def mode_pool(x, num_values=256, factor=2):
    y = np.empty((x.shape[0]//factor, x.shape[1]//factor), dtype=x.dtype)
    
    for iy in prange(y.shape[0]):
        v = np.empty(num_values, dtype=np.int64)
        ix0 = iy * factor
        ix1 = ix0 + factor
        for jy in range(y.shape[1]):            
            jx0 = jy * factor
            jx1 = jx0 + factor
            v[:] = 0

            for ix in range(ix0, ix1):
                for jx in range(jx0, jx1):
                    # print(f"ix: {ix}, jx: {jx}, x[ix,jx]: {x[ix,jx]}")
                    # sleep(2)
                    if np.isnan(x[ix,jx]):
                        v[0] += 1
                        print("Value is NaN")
                        # sleep(1)
                    else:
                        v[int(x[ix,jx])] += 1
            
            y[iy,jy] = v.argmax()
        
    return y


def fill_holes(missing=65535, rad=1):
    def fill(x):
        # identify mask of points to fill
        o = np.ones((2*rad+1,2*rad+1), dtype=np.uint16)
        n = np.prod(o.shape)
        valid = (x != missing)
        num_valid_neighbors = convolve(valid, o, mode='same', method='direct')
        mask = ~valid & (num_valid_neighbors > 0)

        # compute mean of valid points around each fillable point
        fx = x.copy()
        fx[~valid] = 0
        mx = convolve(fx, o.astype(np.float64), mode='same', method='direct')        
        mx = mx[mask] / num_valid_neighbors[mask]
        if np.issubdtype(x.dtype, np.integer):
            mx = mx.round().astype(x.dtype)        
        
        # fill holes with mean
        fx = x.copy()
        fx[mask] = mx
        return fx

    return fill

