from numba import njit, prange
import numpy as np


@njit(parallel=True)
def scale_array(in_arr, out_arr, scale):
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = scale[in_arr[i]]

# NumPy version
#def scale_array(in_arr, out_arr, scale):
#    out_arr[:] = scale[in_arr]

def normalize(mean=0.0, std=1.0):
    scaled = None

    def transform(raw):
        nonlocal scaled
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)        
        normalize_array(raw, scaled, mean, std)

        return scaled

    return transform


def normalize_threshold(mean=0.0, std=1.0, threshold=0.0, fill_value=0.0):
    scaled = None

    def transform(raw):
        nonlocal scaled
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)        
        normalize_threshold_array(raw, scaled, mean, std, threshold, fill_value)

        return scaled

    return transform


def scale_log_norm(scale, threshold=None, fill_value=0, mean=0.0, std=1.0):    
    log_scale = np.log10(scale).astype(np.float32)
    if threshold is not None:
        log_scale[log_scale < np.log10(threshold)] = np.log10(fill_value)
    log_scale[~np.isfinite(log_scale)] = np.log10(fill_value)
    log_scale -= mean
    log_scale /= std
    scaled = None

    def transform(raw):
        nonlocal scaled
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
        scale_array(raw, scaled, log_scale)

        return scaled

    return transform


def scale_norm(scale, threshold=None, missing_value=None,
    fill_value=0, mean=0.0, std=1.0):

    scale = scale.astype(np.float32).copy()
    scale[np.isnan(scale)] = fill_value
    if threshold is not None:
        scale[scale < threshold] = fill_value
    if missing_value is not None:
        missing_value = np.atleast_1d(missing_value)
        for m in missing_value:
            scale[m] = fill_value
    scale -= mean
    scale /= std
    scaled = None

    def transform(raw):
        nonlocal scaled
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
        scale_array(raw, scaled, scale)

        return scaled

    return transform


@njit(parallel=True)
def threshold_array(in_arr, out_arr, threshold):
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = np.float32(in_arr[i] >= threshold)


def one_hot(values):    
    translation = np.zeros(max(values)+1, dtype=int)
    num_categories = len(values)
    for (i,v) in enumerate(values):
        translation[v] = i
    onehot = None

    def transform(raw):
        nonlocal onehot
        if (onehot is None) or (onehot.shape[:-1] != raw.shape):
            onehot = np.empty(raw.shape+(num_categories,),
                dtype=np.float32)
        onehot_transform(raw, onehot, translation)
        
        return onehot

    return transform
            
    
@njit(parallel=True)
def onehot_transform(in_arr, out_arr, translation):
    for k in prange(in_arr.shape[0]):
        out_arr[k,...] = 0.0
        for t in range(in_arr.shape[1]):
            for i in range(in_arr.shape[2]):
                for j in range(in_arr.shape[3]):
                    ind = np.uint64(in_arr[k,t,i,j])
                    c = translation[ind]
                    out_arr[k,t,i,j,c] = 1.0


@njit(parallel=True)
def normalize_array(in_arr, out_arr, mean, std):
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = (in_arr[i]-mean)*inv_std


@njit(parallel=True)
def normalize_threshold_array(in_arr, out_arr, mean, std, threshold, fill_value):
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    threshold = np.float32(threshold)
    fill_value = np.float32(fill_value)
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        x = in_arr[i]
        if x < threshold:
            x = fill_value
        out_arr[i] = (x-mean)*inv_std


# NumPy version
#def threshold_array(in_arr, out_arr, threshold):
#    out_arr[:] = (in_arr >= threshold).astype(np.float32)


def R_threshold(scale, threshold):    
    thresholded = None
    scale_treshold = np.nanargmax(scale > threshold)

    def transform(rzc_raw):
        nonlocal thresholded
        if (thresholded is None) or (thresholded.shape != rzc_raw.shape):
            thresholded = np.empty_like(rzc_raw, dtype=np.float32)
        threshold_array(rzc_raw, thresholded, scale_treshold)

        return thresholded

    return transform
