from datetime import datetime, timedelta
import gc
import os

import numpy as np

from c4dl.features import batch, regions, transform


def setup_batch_gen(file_dir, file_suffix="2020", primary="RZC", target="R10", batch_size=64,
    epoch=datetime(1970,1,1)):

    files = os.listdir(file_dir)
    files = [
        fn for fn in files if 
        fn.startswith("patches") and fn.endswith(file_suffix+".nc")
    ]
    files = {
        fn.split("_")[1]: os.path.join(file_dir,fn) for fn in files
    } # map variable name to file

    # raw data
    raw = {
        var_name: regions.load_patches(fn)
        for (var_name, fn) in files.items()
    }

    raw_interp = ["CAPE-MU", "CIN-MU", "HZEROCL", "MCONV",
        "OMEGA", "SLI", "SOILTYP", "T-2M", "T-SO"]
    for var in raw_interp:
        if var in raw:
            raw[var]["interpolation"] = "nearest"
            raw[var]["stride"] = 12

    static_vars = ["Altitude", "EW-deriv", "NS-deriv"]
    for var in static_vars:
        if var in raw:
            raw[var]["static"] = True

    transform_CAPE = lambda: transform.normalize(std=200.0)
    transform_CIN = lambda: transform.normalize(std=21.0)
    transform_HZEROCL = lambda: transform.normalize_threshold(std=3300, 
        threshold=0.0, fill_value=0.0)
    transform_LCL = lambda: transform.normalize(std=1000.0)
    transform_MCONV = lambda: transform.normalize_threshold(std=3.8e-6, 
        threshold=-1.0, fill_value=0.0)
    transform_OMEGA = lambda: transform.normalize(std=4.2)
    transform_SLI = lambda: transform.normalize(std=3.5)
    transform_SOILTYP = lambda: transform.one_hot([1,3,4,5,6,7,9])
    transform_T = lambda: transform.normalize_threshold(
        mean=290.0, std=7.2, threshold=200, fill_value=290.0)
    transform_Altitude = lambda: transform.normalize(std=820.0)
    transform_deriv = lambda: transform.normalize(std=200.0)

    # features and targets are defined by transforming the raw data
    transforms = {
        "RZC": {
            "source_vars": ["RZC"],
            "transform": transform.scale_log_norm(raw["RZC"]["scale"],
                threshold=0.1, fill_value=0.01, mean=-0.051, std=0.528)
        },
        "CZC": {
            "source_vars": ["CZC"],
            "transform": transform.scale_norm(raw["CZC"]["scale"],
                threshold=5.0, fill_value=-5.0, mean=21.3, std=8.71)
        },
        "EZC-20": {
            "source_vars": ["EZC-20"],
            "transform": transform.scale_norm(raw["EZC-20"]["scale"],
                std=1.97)
        },
        "EZC-45": {
            "source_vars": ["EZC-45"],
            "transform": transform.scale_norm(raw["EZC-45"]["scale"],
                std=1.97)
        },
        "HZC": {
            "source_vars": ["HZC"],
            "transform": transform.scale_norm(raw["HZC"]["scale"],
                std=1.97)
        },
        "LZC": {
            "source_vars": ["LZC"],
            "transform": transform.scale_log_norm(raw["LZC"]["scale"],
                threshold=0.75, fill_value=0.5, mean=-0.274, std=0.135)
        },
        "BZC": {
            "source_vars": ["BZC"],
            "transform": transform.scale_norm(raw["BZC"]["scale"],
                std=29.2)
        },
        "occurrence-10-target": {
            "source_vars": ["occurrence-10"],
            "transform": lambda x: x.astype(np.float32),
        },
        "occurrence-10": {
            "source_vars": ["occurrence-10"],
            "transform": lambda x: x.astype(np.float32),
        },
        "density": {
            "source_vars": ["density"],
            "transform": transform.scale_log_norm(raw["density"]["scale"],
               threshold=1e-3, fill_value=1e-4, mean=-0.593, std=0.640)
        },
        "current": {
            "source_vars": ["current"],
            "transform": transform.scale_log_norm(raw["current"]["scale"],
                threshold=1e-7, fill_value=1e-8, mean=0.0718, std=0.731)
        },
        "ctth-tempe": {
            "source_vars": ["ctth-tempe"],
            "transform": transform.scale_norm(raw["ctth-tempe"]["scale"],
                missing_value=65535, fill_value=360.0, mean=260.0, std=19.1)
        },
        "ctth-alti": {
            "source_vars": ["ctth-alti"],
            "transform": transform.scale_norm(raw["ctth-alti"]["scale"],
                missing_value=65535, fill_value=-1000, mean=5260.0, std=2810.0)
        },
        "cmic-phase": {
            "source_vars": ["cmic-phase"],
            "transform": transform.one_hot(values=[1,2,3,4,255])
        },
        "CAPE-MU": {
            "source_vars": ["CAPE-MU"],
            "transform": transform_CAPE(),
        },
        "CAPE-MU-future": {
            "source_vars": ["CAPE-MU"],
            "transform": transform_CAPE(),
            "timeframe": "future"
        },
        "CIN-MU": {
            "source_vars": ["CIN-MU"],
            "transform": transform_CIN()
        },
        "CIN-MU-future": {
            "source_vars": ["CIN-MU"],
            "transform": transform_CIN(),
            "timeframe": "future"
        },
        "HZEROCL": {
            "source_vars": ["HZEROCL"],
            "transform": transform_HZEROCL()
        },
        "HZEROCL-future": {
            "source_vars": ["HZEROCL"],
            "transform": transform_HZEROCL(),
            "timeframe": "future"
        },
        "LCL": {
            "source_vars": ["LCL"],
            "transform": transform_LCL()
        },
        "LCL-future": {
            "source_vars": ["LCL"],
            "transform": transform_LCL(),
            "timeframe": "future"
        },
        "MCONV": {
            "source_vars": ["MCONV"],
            "transform": transform_MCONV()
        },
        "MCONV-future": {
            "source_vars": ["MCONV"],
            "transform": transform_MCONV(),
            "timeframe": "future"
        },
        "OMEGA": {
            "source_vars": ["OMEGA"],
            "transform": transform_OMEGA()
        },
        "OMEGA-future": {
            "source_vars": ["OMEGA"],
            "transform": transform_OMEGA(),
            "timeframe": "future"
        },
        "SLI": {
            "source_vars": ["SLI"],
            "transform": transform_SLI()
        },
        "SLI-future": {
            "source_vars": ["SLI"],
            "transform": transform_SLI(),
            "timeframe": "future"
        },
        "SOILTYP": {
            "source_vars": ["SOILTYP"],
            "transform": transform_SOILTYP(),
            "timeframe": "static"
        },
        "T-2M": {
            "source_vars": ["T-2M"],
            "transform": transform_T()
        },
        "T-2M-future": {
            "source_vars": ["T-2M"],
            "transform": transform_T(),
            "timeframe": "future"
        },
        "T-SO": {
            "source_vars": ["T-SO"],
            "transform": transform_T()
        },
        "T-SO-future": {
            "source_vars": ["T-SO"],
            "transform": transform_T(),
            "timeframe": "future"
        },
        "Altitude": {
            "source_vars": ["Altitude"],
            "transform": transform_Altitude(),
            "timeframe": "static"
        },
        "EW-deriv": {
            "source_vars": ["EW-deriv"],
            "transform": transform_deriv(),
            "timeframe": "static"
        },
        "NS-deriv": {
            "source_vars": ["NS-deriv"],
            "transform": transform_deriv(),
            "timeframe": "static"
        },
        "R10-target": {
            "source_vars": ["RZC"],
            "transform": transform.R_threshold(raw["RZC"]["scale"], 10.0)
        },
        "R10": {
            "source_vars": ["RZC"],
            "transform": transform.R_threshold(raw["RZC"]["scale"], 10.0)
        }
    }

    # predictors
    pred_names = [
        "RZC", "CZC",
        "EZC-20", "EZC-45",
        "HZC", "LZC",
        "density", "current",
        "ctth-tempe", "ctth-alti", "cmic-phase",
        "CAPE-MU-future",
        "CIN-MU-future",
        "HZEROCL-future",
        "MCONV-future",
        "OMEGA-future",
        "SLI-future",
        "SOILTYP",
        "T-2M-future",
        "T-SO-future",
        "Altitude", "EW-deriv", "NS-deriv"
    ]
    pred_names.append(target)

    predictors = {
        var_name: transforms[var_name]
        for var_name in pred_names
    }

    # targets
    target_names = [target+"-target"]
    targets = {var_name: transforms[var_name] for var_name in target_names}

    # we need one "primary" raw data variable
    # that determines the location of the data for all variables
    primary_patch_data = raw[primary]    
    (box_locs, t0) = regions.box_locations(
        primary_patch_data["patch_coords"],
        primary_patch_data["patch_times"],
        primary_patch_data["zero_patch_coords"],
        primary_patch_data["zero_patch_times"]
    )

    batch_gen = batch.BatchGenerator(predictors, targets, raw, box_locs,
        primary, valid_frac=0.1, test_frac=0.1, batch_size=batch_size,
        timesteps=(6,12), random_seed=1234)

    gc.collect()

    return batch_gen
