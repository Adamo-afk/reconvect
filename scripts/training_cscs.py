import argparse
from datetime import datetime, timedelta
from time import sleep
import gc
import os
import sys
import dask
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import load_model

from c4dl.ml.models.blocks import ConvBlock, ResBlock
from c4dl.ml.models.optimizers import AdaBeliefOptimizer
from c4dl.ml.models.rnn import ResGRU
from c4dl.ml.models.models import CastLayer, RepeatLayer
from c4dl.features import batch, regions, transform # only batch and regions have been modified
from knowledge_distillation.c4dl.features import batch as distill_batch
from knowledge_distillation.c4dl.features import regions as distill_regions

from c4dl.ml.models.models import (
    iou_metric, dice_metric, true_pos, true_neg, false_pos, false_neg,
    make_rain_loss_hist, prob_binary_crossentropy,
    WeightedBinaryCrossentropy, WeightedFocalLoss
)
from plot_training import plot_training_history, plot_confusion_matrix_metrics
from plot_predictions import (
    visualize_lightning_nowcasting_complete, plot_encoder_decoder_sample, 
    plot_prediction_analysis, plot_prediction_distribution,
    plot_input_distributions, plot_meteorological_regridded_data,
    save_all_regridded_data_plots, plot_distillation_curves,
    visualize_continuous_nowcasting_complete, correct_distribution_and_plot
)
from plot_test_swiss_data import plot_confusion_matrix_median, plot_calibration_bins
from xi_grad_cam import (
    run_complete_analysis, create_interactive_heatmap, 
    create_clustered_bar_chart, create_box_plots
)
from patch_stitching import StitchingDiagnostics, diagnose_constant_values
# from generate_shaply_maps import run_shap_analysis

# Hyperparameters
BATCH_SIZE = 8
FINETUNE_EPOCHS = 10
DISTILL_EPOCHS = 10
DISTILL_SAMPLES = 24

# Custom objects for loading models
custom_objects = {
    'CastLayer': CastLayer,
    'RepeatLayer': RepeatLayer,
    'ResGRU': ResGRU,
    'ConvBlock': ConvBlock,
    'ResBlock': ResBlock,
    'AdaBeliefOptimizer': AdaBeliefOptimizer,
    'iou_metric': iou_metric,
    'dice_metric': dice_metric,
    'true_pos': true_pos,
    'true_neg': true_neg,
    'false_pos': false_pos,
    'false_neg': false_neg,
    'make_rain_loss_hist': make_rain_loss_hist,
    'prob_binary_crossentropy': prob_binary_crossentropy,
    'WeightedFocalLoss': WeightedFocalLoss,
    'WeightedBinaryCrossentropy': WeightedBinaryCrossentropy
}

# Add the path to the directory where your other Python file is located
file_paths = [
    os.path.join(os.path.split(os.getcwd())[0], "c4dl\\analysis"),
    os.path.join(os.path.split(os.getcwd())[0], "c4dl\\features"),
    os.path.join(os.path.split(os.getcwd())[0], "c4dl\\ml\\models")
]

for file in file_paths:
    sys.path.append(file)

import calibration, evaluation, models


def run_stitching_diagnostics(batch_gen, args):
    """
    Run comprehensive diagnostics on patch stitching
    """
    print("\n" + "="*80)
    print("RUNNING PATCH STITCHING DIAGNOSTICS")
    print("="*80)
    
    print(f"✓ Diagnostics saved to stitching_diagnostics/")
    
    # Run deep diagnostics on problematic variables
    problem_vars = ['density', 'current', 'cmic-phase', 'cmic-cot']
    
    for var in problem_vars:
        if var in batch_gen.raw_batch_index:
            diagnose_constant_values(batch_gen, var, dataset='test', sample_idx=0)
    
    # Create visualizations if hour specified
    diagnostics = StitchingDiagnostics()
    
    if args.visualize_hour is not None:
        print(f"\nCreating visualizations for hour {args.visualize_hour}...")
        
        for var in problem_vars:
            if var in batch_gen.raw_batch_index:
                try:
                    diagnostics.visualize_stitching(
                        var, 
                        target_hour=args.visualize_hour, 
                        timeframe='past'
                    )
                except Exception as e:
                    print(f"  Could not visualize {var}: {e}")


def setup_batch_gen(
    file_dir, file_suffix, primary="RZC",
    target="R10", batch_size=8,
    sources=("rad", "lig", "sat", "nwp", "dem"),
    upscale_mode=None
):

    files = os.listdir(file_dir)
    # print(f"Files 1: {files}")
    files = [
        fn for fn in files if 
        fn.startswith("patches") and fn.endswith(file_suffix+".nc")
    ]
    # print(f"Files 2: {files}")
    files = {
        fn.split("_")[1]: os.path.join(file_dir,fn) for fn in files
    } # map variable name to file
    # print(f"Files 3: {files}")

    # raw data
    raw = {
        var_name: dask.delayed(regions.load_patches)(fn)
        for (var_name, fn) in files.items()
    }

    # raw = dask.compute(raw, scheduler="processes")[0]
    print("Initializing raw data")
    raw = dask.compute(raw, scheduler="synchronous")[0]
    print("Computing data")

    if not sources:
        raw["zeros"] = {
            "patches": np.zeros(
                (0,)+raw[primary]["patches"].shape[1:],
                dtype=np.float32
            ),
            "patch_coords": np.empty((0,2), dtype=np.uint16),
            "patch_times": np.empty(0, dtype=np.int64),
            "zero_patch_coords": np.vstack((
                raw[primary]["patch_coords"],
                raw[primary]["zero_patch_coords"]
            )).T,
            "zero_patch_times": np.hstack((
                raw[primary]["patch_times"],
                raw[primary]["zero_patch_times"]
            )),
            "zero_value": np.float32(0.0)
        }

    # raw_interp = ["CAPE-MU", "CIN-MU", "HZEROCL", "MCONV",
    #     "LCL-ML", "OMEGA", "SLI", "SOILTYP", "T-2M", "T-SO"] # NWP vars
    # for var in raw_interp:
    #     if var in raw:
    #         raw[var]["interpolation"] = "linear"
    #         raw[var]["stride"] = 12

    # static_vars = ["Altitude", "EW-deriv", "NS-deriv"] # static vars
    # for var in static_vars:
    #     if var in raw:
    #         raw[var]["static"] = True

    # configure missing values (to use when data is missing)
    missing_values = {
        # "CAPE-MU": 200.0,
        # "CIN-MU": 21.0,
        # "HZEROCL": 3300.0,
        # "LCL-ML": 1000.0,
        # "MCONV": 0.0,
        # "OMEGA": 0.0,
        # "SLI": 2.0,
        # "SOILTYP": 5,
        # "T-2M": 289.18,
        # "T-SO": 289.63,
        "HRV": 38.9,
        "VIS006": 37.1,
        "VIS008": 57.0,
        "IR-016": 41.8,
        "IR-039": 274.2,
        "WV-062": 232.8,
        "WV-073": 247.3,
        "IR-087": 266.1,
        "IR-097": 247.5,
        "IR-108": 267.5,
        "IR-120": 266.1,
        "IR-134": 250.6,
        "ctth-tempe": 260.0,
        "ctth-alti": 5260.0,
        "cmic-phase": 4
    }

    # CHANGE SUFFIX FROM 2020 TO 2025 OR VICE VERSA
    # print(raw)

    for var in missing_values:
        raw[var]["missing_value"] = np.float32(missing_values[var])

    # transform_CAPE = lambda: transform.normalize(std=200.0)
    # transform_CIN = lambda: transform.normalize(std=21.0)
    # transform_HZEROCL = lambda: transform.normalize_threshold(std=3300, 
    #     threshold=0.0, fill_value=0.0)
    # transform_LCL = lambda: transform.normalize(std=1000.0)
    # transform_MCONV = lambda: transform.normalize_threshold(std=3.8e-6, 
    #     threshold=-1.0, fill_value=0.0)
    # transform_OMEGA = lambda: transform.normalize(std=4.2)
    # transform_SLI = lambda: transform.normalize(std=3.5)
    # transform_SOILTYP = lambda: transform.one_hot([1,3,4,5,6,7,9])
    # transform_T = lambda: transform.normalize_threshold(
    #     mean=290.0, std=7.2, threshold=200, fill_value=290.0)
    # transform_Altitude = lambda: transform.normalize(std=820.0)
    # transform_deriv = lambda: transform.normalize(std=200.0)
    transform_HRV = lambda: transform.normalize(std=100.0, dtype=np.float16)
    transform_radiance = lambda: transform.normalize(std=100.0)
    transform_TB = lambda: transform.normalize(mean=250.0, std=10.0)

    # features and targets are defined by transforming the raw data
    transforms = {
        "RZC": {
            "source_vars": ["RZC"],
            "transform": transform.scale_log_norm(raw["RZC"]["scale"],
                threshold=0.1, fill_value=0.01, mean=-0.051, std=0.528,
                dtype=np.float16)
        },
        "CZC": {
            "source_vars": ["CZC"],
            "transform": transform.scale_norm(raw["CZC"]["scale"],
                threshold=5.0, fill_value=-5.0, mean=21.3, std=8.71,
                dtype=np.float16)
        },
        "EZC-20": {
            "source_vars": ["EZC-20"],
            "transform": transform.scale_norm(raw["EZC-20"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "EZC-45": {
            "source_vars": ["EZC-45"],
            "transform": transform.scale_norm(raw["EZC-45"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "HZC": {
            "source_vars": ["HZC"],
            "transform": transform.scale_norm(raw["HZC"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "LZC": {
            "source_vars": ["LZC"],
            "transform": transform.scale_log_norm(raw["LZC"]["scale"],
                threshold=0.75, fill_value=0.5, mean=-0.274, std=0.135,
                dtype=np.float16)
        },
        "BZC-target": {
            "source_vars": ["BZC"],
            "transform": transform.scale_norm(raw["BZC"]["scale"],
                std=100.0, dtype=np.float16)
        },
        "BZC": {
            "source_vars": ["BZC"],
            "transform": transform.scale_norm(raw["BZC"]["scale"],
                std=100.0, dtype=np.float16)
        },        
        "AREA57": {
            "source_vars": ["AREA57"],
            "transform": transform.normalize(std=14.0, dtype=np.float16)
        },
        "occurrence-8-10-target": {
            "source_vars": ["occurrence-8-10"],
            "transform": transform.cast(np.uint8)
        },
        "occurrence-8-10": {
            "source_vars": ["occurrence-8-10"],
            "transform": transform.cast(np.uint8)
        },
        "density": {
            "source_vars": ["density"],
            "transform": transform.scale_log_norm(raw["density"]["scale"],
               threshold=1e-3, fill_value=1e-4, mean=-0.593, std=0.640,
               dtype=np.float16)
        },
        "current": {
            "source_vars": ["current"],
            "transform": transform.scale_log_norm(raw["current"]["scale"],
                threshold=1e-7, fill_value=1e-8, mean=0.0718, std=0.731,
                dtype=np.float16)
        },
        "ctth-tempe": {
            "source_vars": ["ctth-tempe"],
            "transform": transform.scale_norm(raw["ctth-tempe"]["scale"],
                missing_value=65535, fill_value=330.0, mean=260.0, std=19.1)
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
        "cmic-cot": {
            "source_vars": ["cmic-cot"],
            "transform": transform.scale_log_norm(raw["cmic-cot"]["scale"],
                missing_value=65535, fill_value=0.1, mean=0.94, std=0.588)
        },
        "HRV": {
            "source_vars": ["HRV"],
            "transform": transform_HRV()
        },
        "VIS006": {
            "source_vars": ["VIS006"],
            "transform": transform_radiance()
        },
        "VIS008": {
            "source_vars": ["VIS008"],
            "transform": transform_radiance()
        },
        "IR-016": {
            "source_vars": ["IR-016"],
            "transform": transform_radiance()
        },
        "IR-039": {
            "source_vars": ["IR-039"],
            "transform": transform.normalize(mean=274, std=17.5)
        },
        "WV-062": {
            "source_vars": ["WV-062"],
            "transform": transform_TB()
        },
        "WV-073": {
            "source_vars": ["WV-073"],
            "transform": transform_TB()
        },
        "IR-087": {
            "source_vars": ["IR-087"],
            "transform": transform_TB()
        },
        "IR-097": {
            "source_vars": ["IR-097"],
            "transform": transform_TB()
        },
        "IR-108": {
            "source_vars": ["IR-108"],
            "transform": transform_TB()
        },
        "IR-120": {
            "source_vars": ["IR-120"],
            "transform": transform_TB()
        },
        "IR-134": {
            "source_vars": ["IR-134"],
            "transform": transform_TB()
        },
        # "CAPE-MU": {
        #     "source_vars": ["CAPE-MU"],
        #     "transform": transform_CAPE(),
        # },
        # "CAPE-MU-future": {
        #     "source_vars": ["CAPE-MU"],
        #     "transform": transform_CAPE(),
        #     "timeframe": "future"
        # },
        # "CIN-MU": {
        #     "source_vars": ["CIN-MU"],
        #     "transform": transform_CIN()
        # },
        # "CIN-MU-future": {
        #     "source_vars": ["CIN-MU"],
        #     "transform": transform_CIN(),
        #     "timeframe": "future"
        # },
        # "HZEROCL": {
        #     "source_vars": ["HZEROCL"],
        #     "transform": transform_HZEROCL()
        # },
        # "HZEROCL-future": {
        #     "source_vars": ["HZEROCL"],
        #     "transform": transform_HZEROCL(),
        #     "timeframe": "future"
        # },
        # "LCL-ML": {
        #     "source_vars": ["LCL-ML"],
        #     "transform": transform_LCL()
        # },
        # "LCL-ML-future": {
        #     "source_vars": ["LCL-ML"],
        #     "transform": transform_LCL(),
        #     "timeframe": "future"
        # },
        # "MCONV": {
        #     "source_vars": ["MCONV"],
        #     "transform": transform_MCONV()
        # },
        # "MCONV-future": {
        #     "source_vars": ["MCONV"],
        #     "transform": transform_MCONV(),
        #     "timeframe": "future"
        # },
        # "OMEGA": {
        #     "source_vars": ["OMEGA"],
        #     "transform": transform_OMEGA()
        # },
        # "OMEGA-future": {
        #     "source_vars": ["OMEGA"],
        #     "transform": transform_OMEGA(),
        #     "timeframe": "future"
        # },
        # "SLI": {
        #     "source_vars": ["SLI"],
        #     "transform": transform_SLI()
        # },
        # "SLI-future": {
        #     "source_vars": ["SLI"],
        #     "transform": transform_SLI(),
        #     "timeframe": "future"
        # },
        # "SOILTYP": {
        #     "source_vars": ["SOILTYP"],
        #     "transform": transform_SOILTYP(),
        #     "timeframe": "static"
        # },
        # "T-2M": {
        #     "source_vars": ["T-2M"],
        #     "transform": transform_T()
        # },
        # "T-2M-future": {
        #     "source_vars": ["T-2M"],
        #     "transform": transform_T(),
        #     "timeframe": "future"
        # },
        # "T-SO": {
        #     "source_vars": ["T-SO"],
        #     "transform": transform_T()
        # },
        # "T-SO-future": {
        #     "source_vars": ["T-SO"],
        #     "transform": transform_T(),
        #     "timeframe": "future"
        # },
        # "Altitude": {
        #     "source_vars": ["Altitude"],
        #     "transform": transform_Altitude(),
        #     "timeframe": "static"
        # },
        # "EW-deriv": {
        #     "source_vars": ["EW-deriv"],
        #     "transform": transform_deriv(),
        #     "timeframe": "static"
        # },
        # "NS-deriv": {
        #     "source_vars": ["NS-deriv"],
        #     "transform": transform_deriv(),
        #     "timeframe": "static"
        # },
        "sun-z": {
            "source_vars": ["sun-z"],
            "transform": transform.normalize(std=127.0)
        },
        "R10-target": {
            "source_vars": ["CPCH"],
            "transform": transform.R_threshold(raw["CPCH"]["scale"], 10.0)
        },
        "R10": {
            "source_vars": ["CPCH"],
            "transform": transform.R_threshold(raw["CPCH"]["scale"], 10.0)
        },
        "CPCH": {
            "source_vars": ["CPCH"],
            "transform": transform.scale_log_norm(raw["CPCH"]["scale"],
                threshold=0.1, fill_value=0.01, mean=0.0, std=1.0,
                dtype=np.float16)
        },
        "CPCH-target": {
            "source_vars": ["CPCH"],
            "transform": transform.scale_log_norm(raw["CPCH"]["scale"],
                threshold=0.1, fill_value=0.01, mean=0.0, std=1.0,
                dtype=np.float16)
        },
        "zeros": {
            "source_vars": ["zeros"],
            "transform": lambda x: x
        }
    }

    # predictors
    pred_names = [
        # Radar
        "RZC", "CZC", "EZC-20", "EZC-45", "HZC", "LZC",
        # Lightning
        "density", "current",
        # NWCSAF
        "ctth-tempe", "ctth-alti", "cmic-phase", "cmic-cot",
        # Satellite
        "HRV", "VIS006", "VIS008", "IR-016", "IR-016", "IR-039", "WV-062", "WV-073", "IR-087", "IR-097", "IR-108", "IR-120", "IR-134",
        # NWP
        # "CAPE-MU-future", "CIN-MU-future", "HZEROCL-future", "LCL-ML-future", "MCONV-future", "OMEGA-future", "SLI-future", "SOILTYP", "T-2M-future", "T-SO-future",
        # DEM
        # "Altitude", "EW-deriv", "NS-deriv",
        # Sun azimuth
        "sun-z"
    ]

    if not ("CPCH" in transforms[target]["source_vars"]):
        pred_names.append(target)

    pred_names = select_sources(pred_names, sources)
    if not pred_names:
        pred_names = ["zeros"] # prediction with no input data

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

    if file_suffix == "2020": 
        (box_locs, _) = distill_regions.box_locations(
            primary_patch_data["patch_coords"],
            primary_patch_data["patch_times"],
            primary_patch_data["zero_patch_coords"],
            primary_patch_data["zero_patch_times"]
        )
    elif file_suffix == "2025": 
        # (box_locs, _) = regions.box_locations(
        #     primary_patch_data["patch_coords"],
        #     primary_patch_data["patch_times"],
        #     primary_patch_data["zero_patch_coords"],
        #     primary_patch_data["zero_patch_times"]
        # )

        (box_locs, _) = regions.box_locations(
            primary_patch_data
        )

    # ALL DATA NECESSARY IS IN MEMORY AND STORED IN raw VARIABLE
    print(f"\nPrimary patch data:\n\n {primary_patch_data}")
    print(f"\nPrimary patch data keys:\n\n {list(primary_patch_data.keys())}")
    print(f"\nPredictors:\n\n {predictors}")
    print(f"\nTargets:\n\n {targets}")
    print(f"\nRaw data:\n\n {list(raw.keys())}")
    print(f"\nBox locations:\n\n {len(box_locs)} locations")
    print(f"\nRaw IR-087 data sample:\n\n {raw['IR-087']}")
    print(f"\nRaw IR-087 patches shape:\n\n {raw['IR-087']['patches'].shape}")
    for (i, var) in enumerate(raw):
        print(f"\n{i+1}) Raw {var} patches shape:\n\n {raw[var]['patches'].shape}")
        print(f"\tpatch_coords shape: {raw[var]['patch_coords']}")
    # exit(0)

    # Analyze raw data dict
    analyze_patch_coordinates(raw)
    
    if file_suffix == "2020":
        batch_gen = distill_batch.BatchGenerator(predictors, targets, raw, box_locs,
            primary, valid_frac=0.1, test_frac=0.1, batch_size=batch_size,
            timesteps=(6,12), random_seed=1234, upscale_mode=upscale_mode)
    
    elif file_suffix == "2025":
        batch_gen = batch.BatchGenerator(predictors, targets, raw, box_locs,
            primary, valid_frac=0.1, test_frac=0.1, batch_size=batch_size,
            timesteps=(6,12), random_seed=1234, upscale_mode=upscale_mode)

    gc.collect()

    return batch_gen, predictors

########################### DIAGNOZE PATCH STITCHING IN batch_gen ###########################
"""
Diagnostic script to identify patch stitching issues
"""
import numpy as np
import matplotlib.pyplot as plt

def diagnose_patch_stitching(batch_gen, time_index=0):
    """
    Diagnose what's happening with patch stitching
    """
    print("=" * 80)
    print("PATCH STITCHING DIAGNOSTICS")
    print("=" * 80)
    
    # 1. Check coords_by_time structure
    print("\n1. COORDINATES BY TIME")
    print(f"Number of timesteps with coordinates: {len(batch_gen.coords_by_time)}")
    
    sample_times = list(batch_gen.coords_by_time.keys())[:5]
    for t in sample_times:
        coords = batch_gen.coords_by_time[t]
        print(f"  Time {t}: {len(coords)} coordinates")
        if len(coords) > 0:
            print(f"    Range: i=[{coords[:,0].min()}, {coords[:,0].max()}], "
                  f"j=[{coords[:,1].min()}, {coords[:,1].max()}]")
            print(f"    Sample coords: {coords[:3].tolist()}")
    
    # 2. Check patch_index dimensions
    print("\n2. PATCH INDEX STRUCTURE")
    primary_var = batch_gen.primary_raw_var
    patch_idx = batch_gen.raw_batch_index[primary_var]
    print(f"  Patch index shape: {patch_idx.patch_index.shape}")
    print(f"  (time_steps, grid_i, grid_j)")
    print(f"  Scale factor: {patch_idx.scale_factor}")
    print(f"  Box size (from patch_idx): {patch_idx.box_size}")
    print(f"  Sample shape: {patch_idx.sample_shape}")
    print(f"  Timesteps: {batch_gen.timesteps}")
    
    # Critical check: compare coordinate systems
    print(f"\n  🔍 COORDINATE SYSTEM CHECK:")
    print(f"  Index limits (t0, t1, i1, j1): {patch_idx.index_limits}")
    
    # Check raw patch_coords vs box_locs coords
    # Get sample from raw data
    from itertools import chain
    all_coords_from_box_locs = list(chain(*batch_gen.coords_by_time.values()))
    if all_coords_from_box_locs:
        box_coords_sample = np.array(all_coords_from_box_locs[:10])
        print(f"  Sample box_locs coordinates: {box_coords_sample.tolist()}")
        print(f"  Box_locs coordinate range: i=[{box_coords_sample[:,0].min()}, {box_coords_sample[:,0].max()}], "
              f"j=[{box_coords_sample[:,1].min()}, {box_coords_sample[:,1].max()}]")
    
    # This reveals the issue: if patch_coords were in a different coordinate system,
    # they would have been scaled during init_patch_index
    print(f"\n  ⚠️  POTENTIAL ISSUE DETECTED:")
    print(f"  - Scale factor of {patch_idx.scale_factor} suggests original coords were in range [0, {patch_idx.index_limits[2]*10}]")
    print(f"  - But box_locs coordinates are in range [15-60], suggesting they're ALREADY scaled")
    print(f"  - This means we may be double-scaling when looking up patches!")
    
    # 3. Check how many patches are actually indexed
    print("\n3. PATCH COVERAGE")
    patch_count = (patch_idx.patch_index >= 0).sum()
    zero_count = (patch_idx.patch_index == batch.PatchIndex.IDX_ZERO).sum()
    missing_count = (patch_idx.patch_index == batch.PatchIndex.IDX_MISSING).sum()
    total = patch_idx.patch_index.size
    
    print(f"  Valid patches: {patch_count} ({100*patch_count/total:.2f}%)")
    print(f"  Zero patches: {zero_count} ({100*zero_count/total:.2f}%)")
    print(f"  Missing patches: {missing_count} ({100*missing_count/total:.2f}%)")
    
    # 4. Test coordinate framing
    print("\n4. COORDINATE FRAMING TEST")
    test_time = batch_gen.time_coords["train"][time_index:time_index+1]
    i_batch, j_batch = batch_gen.frame_spatial_coordinates(test_time, "train")
    box_size = patch_idx.box_size
    print(f"  Test time: {test_time[0]}")
    print(f"  Selected coordinate: ({i_batch[0]}, {j_batch[0]})")
    print(f"  Box size: {box_size}")
    print(f"  This means batch will cover:")
    print(f"    i: [{i_batch[0]}, {i_batch[0] + box_size[1]})")
    print(f"    j: [{j_batch[0]}, {j_batch[0] + box_size[2]})")
    
    # Check if coordinates at this time fall within the selected box
    coords_at_time = batch_gen.coords_by_time.get(test_time[0], np.array([]).reshape(0, 2))
    if len(coords_at_time) > 0:
        in_box = (
            (coords_at_time[:, 0] >= i_batch[0]) & 
            (coords_at_time[:, 0] < i_batch[0] + box_size[1]) &
            (coords_at_time[:, 1] >= j_batch[0]) & 
            (coords_at_time[:, 1] < j_batch[0] + box_size[2])
        )
        print(f"    Coordinates in box: {in_box.sum()} / {len(coords_at_time)}")
        if in_box.sum() < len(coords_at_time):
            print(f"    ⚠️  WARNING: {len(coords_at_time) - in_box.sum()} coordinates outside box!")
            print(f"    Coords outside box: {coords_at_time[~in_box].tolist()}")
        
        # Check grid coverage
        print(f"\n  Grid-space analysis (with scale_factor={patch_idx.scale_factor}):")
        grid_coords = coords_at_time // patch_idx.scale_factor
        grid_i_range = [grid_coords[:, 0].min(), grid_coords[:, 0].max()]
        grid_j_range = [grid_coords[:, 1].min(), grid_coords[:, 1].max()]
        print(f"    Grid coordinates range: i={grid_i_range}, j={grid_j_range}")
        print(f"    Grid span: {grid_i_range[1] - grid_i_range[0] + 1}x{grid_j_range[1] - grid_j_range[0] + 1}")
        
        selected_grid_i = i_batch[0] // patch_idx.scale_factor
        selected_grid_j = j_batch[0] // patch_idx.scale_factor
        print(f"    Selected grid position: ({selected_grid_i}, {selected_grid_j})")
        print(f"    Box covers grid positions: i=[{selected_grid_i}, {selected_grid_i + box_size[1]}), "
              f"j=[{selected_grid_j}, {selected_grid_j + box_size[2]})")
    
    # 5. Visualize patch coverage
    print("\n5. PATCH COVERAGE VISUALIZATION")
    visualize_patch_coverage(patch_idx, test_time[0], i_batch[0], j_batch[0], 
                             box_size, coords_at_time)
    
    # 6. Test actual batch generation
    print("\n6. BATCH GENERATION TEST")
    try:
        pred_batch, target_batch = batch_gen.batch(time_index, "train")
        print(f"  Successfully generated batch")
        print(f"  Predictor shapes: {[p.shape for p in pred_batch]}")
        print(f"  Target shapes: {[t.shape for t in target_batch]}")
        
        # Check for missing/zero values
        for i, pred in enumerate(pred_batch):
            zero_pct = (pred == 0).sum() / pred.size * 100
            print(f"    Predictor {i}: {zero_pct:.1f}% zeros")
    except Exception as e:
        print(f"  ERROR generating batch: {e}")
    
    print("\n" + "=" * 80)


def visualize_patch_coverage(patch_idx, time, i0, j0, box_size, coords):
    """
    Create a visualization of patch coverage
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Patch index at this time
    t_idx = int((time - patch_idx.t0) // patch_idx.dt)
    if 0 <= t_idx < patch_idx.patch_index.shape[0]:
        patch_slice = patch_idx.patch_index[t_idx, :, :]
        
        ax = axes[0]
        im = ax.imshow(patch_slice.T, cmap='viridis', 
                      vmin=-2, vmax=10)
        ax.set_title(f'Patch Index at Time {time}')
        ax.set_xlabel('Grid i')
        ax.set_ylabel('Grid j')
        plt.colorbar(im, ax=ax, label='Patch ID (-2=missing, -1=zero, >=0=patch)')
        
        # Mark the selected box
        i0_scaled = i0 // patch_idx.scale_factor
        j0_scaled = j0 // patch_idx.scale_factor
        i1_scaled = (i0 + box_size[1]) // patch_idx.scale_factor
        j1_scaled = (j0 + box_size[2]) // patch_idx.scale_factor
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((i0_scaled, j0_scaled), 
                         i1_scaled - i0_scaled, 
                         j1_scaled - j0_scaled,
                         linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    # Plot 2: Actual coordinates
    ax = axes[1]
    if len(coords) > 0:
        ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, alpha=0.6, 
                  label='Available coords')
        
        # Mark selected box
        box_i = np.array([i0, i0+box_size[1], i0+box_size[1], i0, i0])
        box_j = np.array([j0, j0, j0+box_size[2], j0+box_size[2], j0])
        ax.plot(box_i, box_j, 'r-', linewidth=2, label='Selected box')
        ax.scatter([i0], [j0], c='red', s=200, marker='x', 
                  label='Box origin', linewidths=3)
    
    ax.set_xlabel('Coordinate i')
    ax.set_ylabel('Coordinate j')
    ax.set_title(f'Coordinate Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('patch_stitching_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to 'patch_stitching_diagnosis.png'")
    plt.close()

###########################################################################################

def select_sources(pred_names, sources=()):
    pred_names_flt = []

    if sources:
        source_list = {
            "rad": [
                "RZC", "CZC", "EZC-20", "EZC-45", "HZC", "LZC",
                "R10", "CPCH", "BZC", "AREA57"
            ],
            "lig": [
                "density", "current", "occurrence-8-10",
            ],
            "sat": [
                "ctth-tempe", "ctth-alti", "cmic-phase", "cmic-cot",
                "sun-z", "HRV", "VIS006", "VIS008", "IR-016",
                "IR-016", "IR-039", "WV-062", "WV-073",
                "IR-087", "IR-097", "IR-108", "IR-120", "IR-134"
            ],
            "nwp": [
                "CAPE-MU", "CIN-MU", "HZEROCL", "LCL-ML",
                "MCONV", "OMEGA", "SLI", "SOILTYP",
                "T-2M", "T-SO"                
            ],
            "dem": [                
                "Altitude", "EW-deriv", "NS-deriv"
            ]
        }
        var_list = []
        for source in sources:
            var_list.extend(source_list[source])

        for pred in pred_names:
            for source_var in var_list:
                if (pred == source_var) or pred.startswith(source_var+"-"):
                    pred_names_flt.append(pred)
                    break

    return pred_names_flt


def build_ensemble_model(batch_gen, dropout=True):
    def create_model(init_strategy=False):
        return models.init_model(batch_gen, 
            init_strategy=init_strategy,
            compile=False
        )
        
    (model1, strategy) = create_model(init_strategy=True)
    with strategy.scope():
        (model2, _) = create_model()
        (model3, _) = create_model()

    ind_models = [model1, model2, model3]
    if dropout:
        weight_files = [
            "../models/lightning-study/lightning_dropout_weightdecay_noclassweight.h5",
            "../models/lightning-study/lightning_dropout_weightdecay_noclassweight2.h5",
            "../models/lightning-study/lightning_dropout_weightdecay_noclassweight3.h5",
        ]
    else:
        weight_files = [
            "../models/lightning-study/lightning_noclassweight1.h5",
            "../models/lightning-study/lightning_noclassweight2.h5",
            "../models/lightning-study/lightning_noclassweight3.h5",
        ]

    for (m,w) in zip(ind_models, weight_files):
        m.load_weights(w)
    
    with strategy.scope():
        ens_model = models.ensemble_model(ind_models)
        models.compile_model(ens_model, event_occurrence=0.5, optimizer='sgd')

    return (ens_model, strategy)


def build_persistence_model(batch_gen):
    return models.init_model(batch_gen, 
        model_func=models.persistence_model)


def inspect_dataset_times(batch_gen, dataset="valid"):
    """
    Show all timestamps in a dataset
    
    Args:
        batch_gen: Your BatchGenerator instance
        dataset: "train", "valid", or "test"
    """
    time_indices = batch_gen.time_coords[dataset]
    
    # Get t0 (reference time) from primary variable
    primary = batch_gen.primary_raw_var
    t0 = batch_gen.raw_batch_index[primary].t0
    dt = batch_gen.raw_batch_index[primary].dt  # Interval in seconds
    
    # Convert indices to actual timestamps
    timestamps = t0 + time_indices * dt
    
    # Convert to datetime objects
    datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    print(f"\n{dataset.upper()} DATASET TIMESTAMPS")
    print("=" * 80)
    print(f"Total samples: {len(datetimes)}")
    print(f"Time range: {min(datetimes)} to {max(datetimes)}")
    print(f"\nFirst 10 timestamps:")
    for i, dt in enumerate(datetimes[:10]):
        print(f"  Index {i}: {dt.strftime('%Y-%m-%d %H:%M:%S')} (hour {dt.hour})")
    
    return time_indices, timestamps, datetimes


def find_and_plot_hour_timesteps(batch_gen, model=None, dataset="valid", 
                                  target_hour="17", target_minute="0", max_plots=3, 
                                  time_position="first"):
    """
    Find timestamps at specific hour and minute, then plot first 2 or last 2 timesteps.
    
    Args:
        batch_gen: BatchGenerator instance
        model: Trained model (optional, for predictions)
        dataset: "train", "valid", or "test"
        target_hour: Hour to search for as string ("0"-"23")
        target_minute: Minute to search for as string ("0"-"59")
        max_plots: Maximum number of samples to plot
        time_position: "first" (show t-6, t-5 / t+1, t+2) or 
                      "last" (show t-2, t-1 / t+11, t+12)
    """
    # Convert string parameters to integers
    hour_int = int(target_hour)
    minute_int = int(target_minute)
    
    # Format for display (zero-padded)
    hour_str = str(hour_int).zfill(2)
    minute_str = str(minute_int).zfill(2)
    
    # Find timestamps
    time_indices = batch_gen.time_coords[dataset]
    primary = batch_gen.primary_raw_var
    t0 = batch_gen.raw_batch_index[primary].t0
    dt = batch_gen.raw_batch_index[primary].dt
    
    timestamps = t0 + time_indices * dt
    datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Find matches at target hour and minute
    matches = []
    for i, dt_obj in enumerate(datetimes):
        if dt_obj.hour == hour_int and dt_obj.minute == minute_int:
            matches.append((i, dt_obj, time_indices[i]))
    
    print(f"\n{'='*80}")
    print(f"TIMESTAMPS AT {hour_str}:{minute_str} in {dataset.upper()}")
    print(f"Showing {time_position.upper()} 2 timesteps")
    print(f"{'='*80}")
    print(f"Found {len(matches)} samples\n")
    
    if len(matches) == 0:
        print(f"❌ No timestamps found at {hour_str}:{minute_str}")
        return None
    
    # Limit number of plots
    matches_to_plot = matches[:max_plots]
    
    # Process each match
    results = []
    for match_num, (batch_idx, dt_obj, time_idx) in enumerate(matches_to_plot):
        print(f"\n{'─'*80}")
        print(f"📅 SAMPLE {match_num + 1}: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*80}")
        
        # Calculate which batch contains this sample
        batch_num = batch_idx // batch_gen.batch_size
        sample_idx = batch_idx % batch_gen.batch_size
        
        print(f"   Batch number: {batch_num}, Sample index in batch: {sample_idx}")
        
        # Generate the batch
        pred_batch, target_batch = batch_gen.batch(batch_num, dataset=dataset)
        
        # Extract this specific sample
        sample_predictors = [p[sample_idx:sample_idx+1] for p in pred_batch]
        sample_targets = [t[sample_idx:sample_idx+1] for t in target_batch]
        
        # Get predictor names
        pred_names = (batch_gen.pred_names_past + 
                     batch_gen.pred_names_future + 
                     batch_gen.pred_names_static)
        target_names = batch_gen.target_names
        
        # Make prediction if model provided
        prediction = None
        if model is not None:
            print(f"\n   🔮 Making prediction...")
            prediction = model.predict(sample_predictors, verbose=0)
            print(f"      Prediction shape: {prediction[0].shape}")
        
        # Plot
        print(f"\n   📈 Creating visualization...")
        plot_timesteps_only(
            sample_predictors, sample_targets, prediction,
            pred_names, target_names, dt_obj, match_num, time_position
        )
        
        results.append({
            'datetime': dt_obj,
            'batch_idx': batch_idx,
            'time_idx': time_idx,
            'predictors': sample_predictors,
            'targets': sample_targets,
            'prediction': prediction,
            'pred_names': pred_names,
            'target_names': target_names
        })
    
    print(f"\n{'='*80}")
    print(f"✅ Processed {len(results)} samples")
    print(f"{'='*80}\n")
    
    return results


def plot_timesteps_only(predictors, targets, prediction, pred_names, 
                        target_names, dt_obj, sample_num, time_position="first"):
    """
    Plot only first 2 or last 2 timesteps for past and future products.
    
    Args:
        predictors: List of predictor arrays
        targets: List of target arrays  
        prediction: Model prediction (optional)
        pred_names: List of predictor names
        target_names: List of target names
        dt_obj: Datetime object for the sample (t=0)
        sample_num: Sample number for filename
        time_position: "first" or "last"
    """
    # Determine which timesteps to show
    if time_position.lower() == "first":
        past_timesteps = [0, 1]  # t-6, t-5
        future_timesteps = [0, 1]  # t+1, t+2
        position_label = "First 2"
    elif time_position.lower() == "last":
        past_timesteps = [4, 5]  # t-2, t-1
        future_timesteps = [10, 11]  # t+11, t+12
        position_label = "Last 2"
    else:
        raise ValueError(f"time_position must be 'first' or 'last', got '{time_position}'")
    
    # Find predictors with 6 timesteps (past)
    past_predictors = []
    past_pred_names = []
    for i, (pred, name) in enumerate(zip(predictors, pred_names)):
        if len(pred.shape) == 5 and pred.shape[1] == 6:
            past_predictors.append(pred)
            past_pred_names.append(name)
    
    # Find predictors with 12 timesteps (future) 
    future_predictors = []
    future_pred_names = []
    for i, (pred, name) in enumerate(zip(predictors, pred_names)):
        if len(pred.shape) == 5 and pred.shape[1] == 12:
            future_predictors.append(pred)
            future_pred_names.append(name)
    
    n_past = len(past_predictors)
    n_future = len(future_predictors)
    n_targets = len(targets)
    n_pred = 1 if prediction is not None else 0
    
    # Create figure with 2 columns (2 timesteps)
    total_rows = n_past + n_future + n_targets + n_pred
    fig = plt.figure(figsize=(10, 3.5 * total_rows))
    gs = GridSpec(total_rows, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    row = 0
    
    # Plot past predictors (2 timesteps)
    print(f"      Plotting {n_past} past predictor variables ({position_label} timesteps)...")
    for var_idx in range(n_past):
        pred_data = past_predictors[var_idx][0]  # Shape: (6, H, W, C)
        var_name = past_pred_names[var_idx]
        
        for col_idx, t in enumerate(past_timesteps):
            ax = fig.add_subplot(gs[row, col_idx])
            
            # Get data for this timestep
            img = pred_data[t, :, :, 0]  # First channel
            
            # Plot
            im = ax.imshow(img, cmap='viridis', aspect='auto')
            
            # Time label with color coding (orange for past)
            time_offset = 6 - t
            # time_offset = (pred_data.shape[0] - 1) - t
            time_label = dt_obj - timedelta(minutes=time_offset * 5)
            title = f't-{time_offset}: {time_label.strftime("%Y-%m-%d %H:%M")}'
            ax.set_title(title, fontsize=11, fontweight='bold', color='orange')
            
            # Variable name on leftmost plot
            if col_idx == 0:
                ax.set_ylabel(f'{var_name}\n(Past)', fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        row += 1
    
    # Plot future predictors (2 timesteps)
    if n_future > 0:
        print(f"      Plotting {n_future} future predictor variables ({position_label} timesteps)...")
        for var_idx in range(n_future):
            pred_data = future_predictors[var_idx][0]  # Shape: (12, H, W, C)
            var_name = future_pred_names[var_idx]
            
            for col_idx, t in enumerate(future_timesteps):
                ax = fig.add_subplot(gs[row, col_idx])
                
                img = pred_data[t, :, :, 0]
                im = ax.imshow(img, cmap='viridis', aspect='auto')
                
                # Time label with color coding (dark blue for future)
                time_offset = t + 1
                time_label = dt_obj + timedelta(minutes=time_offset * 5)
                title = f't+{time_offset}: {time_label.strftime("%Y-%m-%d %H:%M")}'
                ax.set_title(title, fontsize=11, fontweight='bold', color='darkblue')
                
                if col_idx == 0:
                    ax.set_ylabel(f'{var_name}\n(Future)', fontsize=10, fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            row += 1
    
    # Plot targets (2 timesteps)
    print(f"      Plotting {n_targets} target variables ({position_label} timesteps)...")
    for var_idx in range(n_targets):
        target_data = targets[var_idx][0]  # Shape: (12, H, W, C)
        var_name = target_names[var_idx]
        
        for col_idx, t in enumerate(future_timesteps):
            ax = fig.add_subplot(gs[row, col_idx])
            
            img = target_data[t, :, :, 0]
            im = ax.imshow(img, cmap='RdYlBu_r', aspect='auto')
            
            # Time label with color coding (red for targets)
            time_offset = t + 1
            time_label = dt_obj + timedelta(minutes=time_offset * 5)
            title = f't+{time_offset}: {time_label.strftime("%Y-%m-%d %H:%M")}'
            ax.set_title(title, fontsize=11, fontweight='bold', color='red')
            
            if col_idx == 0:
                ax.set_ylabel(f'{var_name}\n(Target)', fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        row += 1
    
    # Plot prediction if available (2 timesteps)
    if prediction is not None:
        print(f"      Plotting prediction ({position_label} timesteps)...")
        pred_data = prediction[0][0]  # Shape: (12, H, W, C)
        
        for col_idx, t in enumerate(future_timesteps):
            ax = fig.add_subplot(gs[row, col_idx])
            
            img = pred_data[t, :, :, 0]
            im = ax.imshow(img, cmap='RdYlBu_r', aspect='auto')
            
            # Time label with color coding (dark red for predictions)
            time_offset = t + 1
            time_label = dt_obj + timedelta(minutes=time_offset * 5)
            title = f't+{time_offset}: {time_label.strftime("%Y-%m-%d %H:%M")}'
            ax.set_title(title, fontsize=11, fontweight='bold', color='darkred')
            
            if col_idx == 0:
                ax.set_ylabel(f'Prediction', fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Overall title
    main_title = f'{position_label} Timesteps - Sample at {dt_obj.strftime("%Y-%m-%d %H:%M:%S")}'
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    filename = f'sample_{time_position}_{sample_num}_{dt_obj.strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"      ✅ Saved: {filename}")
    plt.close()


def print_datetime_range(datetimes, dataset):
    """Sort datetimes and print the range in a user-friendly format."""
    earliest = min(datetimes)
    latest = max(datetimes)
    
    # If same day, just show times
    if earliest.date() == latest.date():
        print(f"{dataset.upper()}\n\t Range: {earliest.strftime('%B %d, %Y')} from {earliest.strftime('%I:%M %p')} to {latest.strftime('%I:%M %p')}")
    else:
        print(f"{dataset.upper()}\n\t Range: {earliest.strftime('%B %d, %Y at %I:%M %p')} to {latest.strftime('%B %d, %Y at %I:%M %p')}")
    
    return earliest, latest


# Add diagnostic function OUTSIDE @njit (regular Python)
def diagnose_patch_stitching_issue(batch_gen, dataset="train", sample_idx=0):
    """
    Diagnose why patches aren't being stitched correctly
    """
    print("\n" + "="*80)
    print("PATCH STITCHING DIAGNOSTIC")
    print("="*80)
    
    # Get a batch
    t_pred = batch_gen.time_coords[dataset][sample_idx:sample_idx+1]
    pixel_i0, pixel_j0 = batch_gen.frame_spatial_coordinates(t_pred, dataset)
    
    print(f"\nSample time index: {t_pred[0]}")
    print(f"Selected pixel origin: ({pixel_i0[0]}, {pixel_j0[0]})")
    
    # Get patch index info
    primary = batch_gen.primary_raw_var
    patch_idx = batch_gen.raw_batch_index[primary]
    
    print(f"\nPatch data available:")
    print(f"  Total patches: {len(patch_idx.patch_pixels_i)}")
    print(f"  At time {t_pred[0]}: ", end="")
    
    # Count patches at this time
    patches_at_time = (patch_idx.patch_times_idx == t_pred[0]).sum()
    print(f"{patches_at_time} patches")
    
    # Get patches at this time
    time_mask = patch_idx.patch_times_idx == t_pred[0]
    time_coords_i = patch_idx.patch_pixels_i[time_mask]
    time_coords_j = patch_idx.patch_pixels_j[time_mask]
    time_indices = patch_idx.patch_indices[time_mask]
    
    if len(time_coords_i) > 0:
        print(f"  Pixel coordinates at this time:")
        for i in range(min(10, len(time_coords_i))):
            print(f"    ({time_coords_i[i]}, {time_coords_j[i]}) -> patch {time_indices[i]}")
        
        print(f"\n  Coordinate range:")
        print(f"    i: [{time_coords_i.min()}, {time_coords_i.max()}]")
        print(f"    j: [{time_coords_j.min()}, {time_coords_j.max()}]")
    
    # Check what we're looking for
    box_size = patch_idx.box_size
    print(f"\nSearching for patches in 8×8 grid:")
    print(f"  Box size: {box_size}")
    print(f"  Origin: ({pixel_i0[0]}, {pixel_j0[0]})")
    print(f"  Target positions:")
    
    matches = 0
    for pi_idx in range(min(8, box_size[1])):
        for pj_idx in range(min(8, box_size[2])):
            target_pi = pixel_i0[0] + pi_idx * 32
            target_pj = pixel_j0[0] + pj_idx * 32
            
            # Check if this position exists
            found = False
            for i in range(len(time_coords_i)):
                if time_coords_i[i] == target_pi and time_coords_j[i] == target_pj:
                    found = True
                    matches += 1
                    if matches <= 10:  # Print first 10 matches
                        print(f"    ✓ ({target_pi}, {target_pj}) - FOUND")
                    break
            
            if not found and pi_idx < 2 and pj_idx < 2:  # Print first few misses
                print(f"    ✗ ({target_pi}, {target_pj}) - MISSING")
    
    print(f"\n  Total matches: {matches} / 64")
    print(f"  Match rate: {100*matches/64:.1f}%")
    
    if matches < 10:
        print(f"\n❌ PROBLEM: Only {matches} patches found out of 64!")
        print(f"   This explains why the image looks repeated.")
        print(f"\n   Possible causes:")
        print(f"   1. pixel_i0/j0 doesn't align with where patches actually are")
        print(f"   2. Patches are at different positions than expected")
        print(f"   3. Not enough patches at this time")
    else:
        print(f"\n✓ Found {matches} patches - stitching should work")
    
    print("="*80 + "\n")
    
    return matches


def analyze_patch_coordinates(raw_data_dict):
    """
    Analyzes patch coordinates to understand the coordinate system
    Run this on your raw data before creating BatchGenerator
    """
    print("="*80)
    print("PATCH COORDINATE ANALYSIS")
    print("="*80)
    
    for var_name, var_data in raw_data_dict.items():
        if 'patches' not in var_data or len(var_data['patches']) == 0:
            continue
            
        patches = var_data['patches']
        coords = var_data['patch_coords']
        times = var_data['patch_times']
        
        print(f"\n{var_name}:")
        print(f"  Patch shape: {patches[0].shape}")
        print(f"  Number of patches: {len(patches)}")
        
        # Analyze coordinate distribution
        unique_i = np.unique(coords[:, 0])
        unique_j = np.unique(coords[:, 1])
        
        print(f"  Unique i coordinates: {len(unique_i)}")
        print(f"  Unique j coordinates: {len(unique_j)}")
        
        if len(unique_i) > 1:
            i_diffs = np.diff(np.sort(unique_i))
            i_spacings = np.unique(i_diffs)
            print(f"  i-coordinate spacings: {i_spacings[:10]}")  # Show first 10
            print(f"  Most common i spacing: {np.median(i_diffs):.1f}")
            
            # NEW: Show DOMINANT spacing (mode of large spacings)
            large_spacings = i_diffs[i_diffs >= 10]
            if len(large_spacings) > 0:
                mode_spacing = stats.mode(large_spacings, keepdims=True).mode[0]
                print(f"  DOMINANT i spacing (mode, ≥10px): {mode_spacing}")
        
        if len(unique_j) > 1:
            j_diffs = np.diff(np.sort(unique_j))
            j_spacings = np.unique(j_diffs)
            print(f"  j-coordinate spacings: {j_spacings[:10]}")  # Show first 10
            print(f"  Most common j spacing: {np.median(j_diffs):.1f}")
            
            # NEW: Show DOMINANT spacing (mode of large spacings)
            large_spacings = j_diffs[j_diffs >= 10]
            if len(large_spacings) > 0:
                mode_spacing = stats.mode(large_spacings, keepdims=True).mode[0]
                print(f"  DOMINANT j spacing (mode, ≥10px): {mode_spacing}")
        
        # Sample some coordinates
        print(f"  Sample coordinates (first 10):")
        for k in range(min(10, len(coords))):
            print(f"    [{k}] ({coords[k,0]}, {coords[k,1]}) at time {times[k]}")
        
        # Check if coordinates are on a regular grid
        i_min, j_min = coords.min(axis=0)
        i_max, j_max = coords.max(axis=0)
        print(f"  Bounding box: i=[{i_min}, {i_max}], j=[{j_min}, {j_max}]")
        
        # For one specific time, show all coordinates
        if len(times) > 0:
            sample_time = times[0]
            time_mask = times == sample_time
            time_coords = coords[time_mask]
            print(f"  At time {sample_time}, {len(time_coords)} patches:")
            print(f"    i range: [{time_coords[:,0].min()}, {time_coords[:,0].max()}]")
            print(f"    j range: [{time_coords[:,1].min()}, {time_coords[:,1].max()}]")
            # Show grid structure
            if len(time_coords) > 1:
                print(f"    First 5 coords: {time_coords[:5].tolist()}")


def model_sources(
    sources_str, 
    file_suffix, 
    file_dir, 
    upscale_mode,
    args,
    target="occurrence-8-10", 
    undefined_shape=False
):
    all_sources = ("rad", "lig", "sat", "nwp", "dem")
    # Extract first letter from sources_str 
    # (e.g. "rl" for radar and lightning)
    sources = [s for s in all_sources if s[0] in sources_str] 
    sources_str = "".join(s[0] for s in sources)

    batch_gen, predictors = setup_batch_gen(file_dir=file_dir, target=target,
        batch_size=BATCH_SIZE, sources=sources, file_suffix=file_suffix, upscale_mode=upscale_mode)

    kwargs = {}
    compile_kwargs = {
        "opt_kwargs": {"weight_decay": 1e-4},
        "event_occurrence": 0.5
    }
    if target == "BZC": # hail
        compile_kwargs["loss"] = "prob_binary_crossentropy"
    
    if target == "CPCH": # rain rate  
        bins = np.array(
            [10, 30, 50],
            dtype=np.float32
        )
        compile_kwargs["loss"] = models.make_rain_loss_hist(bins)        
        compile_kwargs["metrics"] = []
        kwargs["last_only"] = True
        kwargs["num_outputs"] = len(bins)+1
        kwargs["final_activation"] = "softmax"

    print("Initializing model")
    (model, strategy) = models.init_model(
        batch_gen,
        dropout=0.1, 
        compile_kwargs=compile_kwargs,
        undefined_shape=undefined_shape,
        **kwargs
    )

    ######################## TEST ROMANIAN DATASET ########################
    # Define target and data for predictions
    # diagnose_patch_stitching(batch_gen, time_index=0)
    # exit(0)

    # Diagnose batch stitching at SPECIFIC HOURS
    # Run diagnostics if requested
    if args.diagnose_stitching:
        run_stitching_diagnostics(batch_gen, args)

    batch_seq = batch.BatchSequence(batch_gen, dataset="train")
    train_indices, train_timestamps, train_datetimes = inspect_dataset_times(batch_gen, "train")
    test_indices, test_timestamps, test_datetimes = inspect_dataset_times(batch_gen, "test")
    val_indices, val_timestamps, val_datetimes = inspect_dataset_times(batch_gen, "valid")

    _, _ = print_datetime_range(train_datetimes, dataset="train")
    _, _ = print_datetime_range(val_datetimes, dataset="valid")
    _, _ = print_datetime_range(test_datetimes, dataset="test")
    
    print(f"Test indices: {test_indices} \nTest timestamps: {test_timestamps}")
    print(f"Validation indices: {val_indices} \nValidation timestamps: {val_timestamps}")
    print(f"Train indices: {train_indices} \nTrain timestamps: {train_timestamps}")
    
    # WITHOUT model (just visualize data)
    _ = find_and_plot_hour_timesteps(
        batch_gen, 
        model=None,
        dataset="test", 
        target_hour=args.visualize_hour,
        target_minute=args.visualize_minutes,
        max_plots=3,
        time_position=args.plot_timestamps
    )
    print("Finished visualizing dataset samples")
    # exit(0)
    
    batch_idx = 0
    sample_idx = 3
    data, label = batch_seq.__getitem__(batch_idx)
    print(f"Data inputs: {len(data)}, Target output shape: {label[0].shape}") # target is a list of length 1
    for (i,d) in enumerate(data):
        print(f"Data input {i+1} shape: {d.shape}")        

    # Get all model input layer names
    input_layer_names = [layer.name for layer in model.inputs]

    # # Diagnose batch_gen
    # _ = diagnose_patch_stitching_issue(batch_gen, dataset="train", sample_idx=sample_idx)

    # Plot sample input and target data
    plot_encoder_decoder_sample(
        data, 
        label[0], 
        sample_idx=sample_idx,
        input_names=input_layer_names,
        cmap='RdBu_r',  # Good for meteorological data
        vmin=-2,
        vmax=2,
        output_path='sample_visualization.gif',
        duration_per_frame=1  # seconds per frame
    )

    plot_input_distributions(
        data, 
        sample_idx=sample_idx,
        input_names=input_layer_names,
        n_cols=8
    )
    plt.show()
    # exit(0)
    #######################################################################

    return (sources_str, batch_gen, model, strategy, predictors)

# CODE TO BE MODIFED TO FIT NEW TRAINING PROCESS
#===========================================================================================================================
# def training_sources(sources_str, file_suffix, file_dir="../data/2020/", target="occurrence-8-10", fn_prefix="lightning"):
#     if sources_str in ("", "null"):
#         sources_str = ""
#         sources_suffix = "null"
#     else:
#         sources_suffix = sources_str
#     (sources_str, batch_gen, model, strategy, _) = model_sources(
#         sources_str, target=target, file_dir=file_dir
#     )

#     print("Training model")
#     models.train_model(model, strategy, batch_gen, epochs=FINETUNE_EPOCHS,
#         weight_fn=f"../models/{fn_prefix}-{sources_suffix}.h5")
#     print("Finished training")


# def export_model(sources_str, file_suffix, model_dir, file_dir="../data/2020/", target="occurrence-8-10"):
#     (sources_str, _, model, _, _) = model_sources(
#         sources_str, target=target, undefined_shape=True, file_dir=file_dir
#     )
#     pred = {
#         "occurrence-8-10": "lightning",
#         "BZC": "hail",
#         "CPCH": "rain"
#     }[target]
#     model.load_weights(f"../models/{pred}/{pred}-{sources_str}.h5")
#     if pred == "lightning":
#         occurrence = np.load(
#             f"../results/lightning/test/calibration-lightning-{sources_str}.npy"
#         )
#         p = np.linspace(0,1,len(occurrence)+1)
#         p = 0.5 * (p[:-1] + p[1:])
#         model = calibration.calibrated_model(model, p, occurrence)
        
#     model.save(model_dir, include_optimizer=False)
#============================================================================================================================


def generate_sources(sources_str, file_suffix, upscale_mode, args, file_dir="../data/2020/", target="occurrence-8-10"):

    print("Generating models and batches")

    if sources_str in ("", "null"):
        sources_str = ""
        sources_suffix = "null"
    else:
        sources_suffix = sources_str

    if file_suffix == "2020":
        file_dir = f"../knowledge_distillation/data/{file_suffix}/"

        (_, swiss_batch_gen, teacher_model, _, _) = model_sources(
            sources_str, args=args, target=target, file_suffix=file_suffix, file_dir=file_dir, upscale_mode=upscale_mode)
        
        weight_fn = os.path.join("../models/", f"{sources_suffix}.h5")
        print("Loading weights...")
        teacher_model.load_weights(weight_fn)
        print("Weights loaded")

        return swiss_batch_gen, teacher_model
    
    elif file_suffix == "2025":
        file_dir = f"../data/{file_suffix}/"

        (_, ro_batch_gen, student_model, _, _) = model_sources(
            sources_str, args=args, target=target, file_suffix=file_suffix, file_dir=file_dir, upscale_mode=upscale_mode)

        return ro_batch_gen, student_model
    

def eval_sources(sources_str, model, batch_gen, args, gt, idx=0, fn_prefix="lightning", dataset="test"):

    print("Evaluating sources")
    batch_sample_idx = 3

    # if sources_str in ("", "null"):
    #     sources_str = ""
    #     sources_suffix = "null"
    # else:
    #     sources_suffix = sources_str

    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)
    result_dir = os.path.join("../results/", dataset)
    # weight_fn = os.path.join("../models/", f"{sources_suffix}.h5")

    if args.create_conf_matrix:
        evaluation.conf_matrix_models(model, batch_seq, args, result_dir)
        calibration.calibration_curve_models(model, batch_seq, args, result_dir)
        return 

    if args.generate_predictions:
        # Define target and data for predictions
        data, target = batch_seq.__getitem__(idx)
        print(f"Data inputs: {len(data)}, Target output shape: {target[0].shape}") # target is a list of length 1
        for (i,d) in enumerate(data):
            print(f"Data input {i+1} shape: {d.shape}")        

        # # Get all model input layer names
        # input_layer_names = [layer.name for layer in model.inputs]

        # Make predictions
        predictions = model.predict(data)
        print(f"Predictions shape: {predictions.shape}")

        if gt == "occurrence-8-10":
            # Plot predictions
            fig, _ = plot_prediction_analysis(
                predictions=predictions,
                sample_idx=batch_sample_idx,
                normalize=False,  # set False if already normalized to [0, 1] interval
                threshold=0.9,
                cmap='YlOrRd'
            )
            # Save the plot
            plt.savefig(os.path.join(result_dir, f"predictions_analysis-{sources_str}.png"))
            plt.close(fig)
            print(f"Predictions analysis plot saved to {result_dir}")
        
        elif gt == "BZC":
            # results = correct_distribution_and_plot(
            #         predictions, 
            #         target[0], 
            #         methods=['quantile_mapping'],
            #         sample_idx=batch_sample_idx,
            #         timestamps_to_plot=12
            #     )
            # predictions = results['quantile_mapping']

            fig, _ = plot_prediction_distribution(
                predictions * 100, # denormalize
                sample_idx=batch_sample_idx,
                cmap='YlGnBu'  # Good for precipitation/meteorological data
            )
            # Save the plot
            plt.savefig(os.path.join(result_dir, f"predictions_analysis-{sources_str}.png"))
            plt.close(fig)
            print(f"Predictions analysis plot saved to {result_dir}")

        elif gt == "R10":
            results = correct_distribution_and_plot(
                    predictions, 
                    target[0], 
                    methods=['quantile_mapping'],
                    sample_idx=batch_sample_idx,
                    timestamps_to_plot=12
                )
            predictions = results['quantile_mapping']

            fig, _ = plot_prediction_distribution(
                predictions * 10, # denormalize
                sample_idx=batch_sample_idx,
                cmap='YlGnBu'  # Good for precipitation/meteorological data
            )
            # Save the plot
            plt.savefig(os.path.join(result_dir, f"predictions_analysis-{sources_str}.png"))
            plt.close(fig)
            print(f"Predictions analysis plot saved to {result_dir}")

        return predictions, target[0]

    if args.evaluate_model:
        # Evaluate model
        eval_result = model.evaluate(batch_seq)

        print(f"Eval result: {eval_result}")
        gc.collect()
        eval_fn = os.path.join(result_dir,
            f"eval-{fn_prefix}-{sources_str}.csv")
        if np.ndim(eval_result) == 0:
            eval_result = [eval_result]

        # Create result_dir if it does not exist
        os.makedirs(result_dir, exist_ok=True)
        np.savetxt(eval_fn, eval_result, delimiter=',', fmt='%.6e', )

        print(f"Eval result saved to {eval_fn}")
        return 

    # if sources_str in ("", "null"):
    #     sources_str = ""
    #     sources_suffix = "null"
    # else:
    #     sources_suffix = sources_str

    # if file_suffix == "2020":
    #     file_dir = f"../knowledge_distillation/data/{file_suffix}/"

    #     (_, swiss_batch_gen, teacher_model, _, _) = model_sources(
    #         sources_str, target=target, file_suffix=file_suffix, file_dir=file_dir)
        
    #     # # weight_fn = os.path.join("../models/", fn_prefix, f"{fn_prefix}-{sources_suffix}.h5")
    #     weight_fn = os.path.join("../models/", f"{sources_suffix}.h5")
    #     print("Loading weights...")
    #     teacher_model.load_weights(weight_fn)
    #     print("Weights loaded")

    #     return swiss_batch_gen, teacher_model
    
    # elif file_suffix == "2025":
    #     file_dir = f"../data/{file_suffix}/"

    #     (_, ro_batch_gen, student_model, _, _) = model_sources(
    #         sources_str, target=target, file_suffix=file_suffix, file_dir=file_dir)

    #     # result_dir = os.path.join("../results/", fn_prefix, dataset)
    #     # result_dir = os.path.join("../results/", dataset)
    #     # batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)

    #     return ro_batch_gen, student_model

    # data_item = batch_seq.__getitem__(0)
    # # print(f"Data item: {data_item}")
    # print(f"Data item length: {len(data_item)}")
    # print(f"Data item 1 length: {len(data_item[1])}")
    # print(f"Data item 1 type: {type(data_item[1])}")
    # print(f"Data item 1 shape: {np.array(data_item[1]).shape}")
    # print(f"Data item 0 length: {len(data_item[0])}")
    # print(f"Data item 0 type: {type(data_item[0])}")
    # # data_len = batch_seq.__len__()

    # if not separate_leadtimes:

    #     print("Evaluating model")

    #     # import visualkeras
    #     # # More customization options
    #     # visualkeras.layered_view(
    #     #     model,
    #     #     to_file='model_arch_post-modifications.png',
    #     #     legend=True,
    #     #     show_dimension=True
    #     # )

    #     if len(batch_seq) == 0:
    #         raise ValueError("Warning: No evaluation batches available")
    #     else:
    #         print(f"Number of batches: {len(batch_seq)}")
    #         eval_result = model.evaluate(batch_seq)
        
    #     print(f"Eval result: {eval_result}")
    #     gc.collect()
    #     eval_fn = os.path.join(result_dir,
    #         f"eval-{fn_prefix}-{sources_str}.csv")
    #     if np.ndim(eval_result) == 0:
    #         eval_result = [eval_result]
    #     np.savetxt(eval_fn, eval_result, delimiter=',', fmt='%.6e')

    #     print(f"Eval result saved to {eval_fn}")

    #     calibration.calibration_curve_models(model, batch_gen, [weight_fn],
    #        result_dir, dataset=dataset)
        
    #     evaluation.conf_matrix_models(model, batch_gen, [weight_fn],
    #        result_dir, dataset=dataset)
    #     return
    # else:
    #     def loss_timestep(loss, timestep):
    #         def l(y_true, y_pred):
    #             y_true = y_true[:,timestep:timestep+1,...]
    #             y_pred = y_pred[:,timestep:timestep+1,...]
    #             return loss(y_true, y_pred)
    #         l.__name__ = f"loss_{timestep}"
    #         return l        
    #     metrics = [loss_timestep(model.loss, i) for i in range(12)]
    #     with strategy.scope():
    #         model.compile(loss=model.loss, metrics=metrics, optimizer='sgd')
    
    #     eval_result = model.evaluate(batch_seq)
    #     eval_fn = os.path.join(result_dir,
    #         f"eval_leadtime-{fn_prefix}-{sources_str}.csv")
    #     np.savetxt(eval_fn, eval_result, delimiter=',', fmt='%.6e')
    #     return 

# def create_float16_model(original_model, custom_objects={}):
#     """Create a float16 copy without modifying the original"""
#     # Get model config and weights
#     config = original_model.get_config()
#     weights = original_model.get_weights()
    
#     # Cast weights to float16
#     weights_fp16 = [tf.cast(w, tf.float16).numpy() for w in weights]
    
#     # Recreate model
#     model_fp16 = tf.keras.Model.from_config(config, custom_objects=custom_objects)
#     model_fp16.set_weights(weights_fp16)
    
#     return model_fp16


def knowledge_distillation_with_batch_gen(
    student_batch_gen, teacher_batch_gen,
    student_model_path, teacher_model_path,
    epochs=20, n_test_samples=50, save_path=None
):
    """
    Knowledge distillation using batch_gen objects with limited test samples
    
    Args:
        student_batch_gen: Batch generator for student model (Romanian format)
        teacher_batch_gen: Batch generator for teacher model (Swiss format)
        student_model_path: Path to student model
        teacher_model_path: Path to teacher model
        epochs: Number of distillation epochs
        temperature: Temperature for knowledge distillation
        alpha: Weight between teacher guidance and ground truth
        n_test_samples: Number of test samples to use (default: 50)
        save_path: Path to save distilled model
    """ 
    # Clear any previous GPU memory
    print("Clearing all GPU memory...")
    tf.keras.backend.clear_session()

    # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")

    print(f"Starting knowledge distillation with {n_test_samples} test samples...")
    
    # Load models
    teacher_model = load_model(teacher_model_path, compile=False, custom_objects=custom_objects)
    student_model = load_model(student_model_path, compile=False, custom_objects=custom_objects)

    # # Checking if computation is done in float32
    # if tf.config.experimental.tensor_float_32_execution_enabled():
    #     print("Tensorflow float-32 is enabled. Setting mixed precision policy.")
    #     tf.keras.mixed_precision.set_global_policy('mixed_float16')
    #     sleep(3)  # Pause to ensure the setting takes effect
    # else:
    #     print("Tensorflow float-32 is disabled.")

    # # Convert to float16 models
    # teacher_model = create_float16_model(teacher_model, custom_objects=custom_objects)
    # student_model = create_float16_model(student_model, custom_objects=custom_objects)

    # Freeze teacher model
    teacher_model.trainable = False
    
    print(f"Teacher model output shape: {teacher_model.output_shape}")
    print(f"Student model output shape: {student_model.output_shape}")
    
    # Limit test dataset to first n_test_samples
    print(f"\nOriginal test samples - Teacher: {len(teacher_batch_gen.time_coords['test'])}")
    print(f"Original test samples - Student: {len(student_batch_gen.time_coords['test'])}")
    
    # Create limited test coordinates
    student_test_coords_limited = student_batch_gen.time_coords["test"][:n_test_samples]
    teacher_test_coords_limited = teacher_batch_gen.time_coords["test"][:n_test_samples]
    
    # Temporarily modify the batch_gen test coordinates
    student_batch_gen.time_coords["test"] = student_test_coords_limited
    teacher_batch_gen.time_coords["test"] = teacher_test_coords_limited

    print(f"Limited test samples - Teacher: {len(teacher_batch_gen.time_coords['test'])}")
    print(f"Limited test samples - Student: {len(student_batch_gen.time_coords['test'])}")
    
    # Create BatchSequence objects for test dataset (following train_model pattern)
    teacher_batch_seq_test = batch.BatchSequence(teacher_batch_gen, dataset='test')
    student_batch_seq_test = batch.BatchSequence(student_batch_gen, dataset='test')

    print(f"Teacher BatchSequence length: {teacher_batch_seq_test.__len__()}")
    print(f"Student BatchSequence length: {student_batch_seq_test.__len__()}")
    
    # # Debug first batch structure (like in train_model)
    # print("\nDebugging first batch structure...")
    # try:
    #     teacher_sample = teacher_batch_seq_test.__getitem__(0)
    #     student_sample = student_batch_seq_test.__getitem__(0)
        
    #     teacher_data, teacher_target = teacher_sample
    #     student_data, student_target = student_sample

    #     print(f"Teacher data: {teacher_data[0].shape}, teacher targets: {teacher_target[0].shape}")
    #     print(f"Student data: {student_data[0].shape}, student targets: {student_target[0].shape}")
    #     sleep(3)

    # except Exception as e:
    #     print(f"Could not debug batch structure: {e}")
    
    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # ========== Helper functions for feature extraction ==========

    def extract_teacher_features(model, inputs):
        """Extract intermediate features from teacher model."""
        
        # Create a feature extractor model
        feature_layers = [
            layer.name if len(layer.name.split('_')) == 3 else layer.name + '_0' for layer in teacher_model.layers if 'res_block' in layer.name
        ][:4] # extract only first 4 ResBlocks
        print(feature_layers)
        exit(0)
        
        features = []
        for layer_name in feature_layers:
            try:
                layer = model.get_layer(layer_name)
                feature_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
                feature = feature_model(inputs, training=False)
                features.append(feature)
            except:
                print(f"Warning: Layer {layer_name} not found in teacher model")
        
        return features


    def extract_student_features(model, inputs):
        """Extract intermediate features from student model."""

        feature_layers = [
            layer.name if len(layer.name.split('_')) == 3 else layer.name + '_0' for layer in teacher_model.layers if 'res_block' in layer.name
        ][:4] # extract only first 4 ResBlocks
        print(feature_layers)
        exit(0)
        
        features = []
        for layer_name in feature_layers:
            try:
                layer = model.get_layer(layer_name)
                feature_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
                feature = feature_model(inputs, training=True)
                features.append(feature)
            except:
                print(f"Warning: Layer {layer_name} not found in student model")
        
        return features


    def compute_feature_loss(teacher_features, student_features):
        """Compute MSE loss between teacher and student features."""
        if len(teacher_features) != len(student_features):
            raise ValueError("Teacher and student must have same number of feature layers")
        
        total_feature_loss = 0.0
        
        for t_feat, s_feat in zip(teacher_features, student_features):
            # Handle dimension mismatch with projection if needed
            if t_feat.shape[-1] != s_feat.shape[-1]:
                
                # Simple adaptive pooling (quick fix)
                s_feat = tf.keras.layers.Dense(t_feat.shape[-1], use_bias=False)(s_feat)
            
            # Compute MSE between features
            feature_loss = tf.reduce_mean(tf.square(t_feat - s_feat))
            total_feature_loss += feature_loss
        
        # Average across all layers
        return total_feature_loss / len(teacher_features)
    

    # Knowledge distillation training step
    # @tf.function(jit_compile=True)
    @tf.function
    def distillation_train_step(teacher_X, student_X, student_y, 
                           alpha=0.3, beta=0.3, gamma=0.4, temperature=3.0):
        """
        Implements: L_total = α × L_hard + β × L_soft + γ × L_feature
        
        Args:
            alpha: Weight for hard loss (ground truth)
            beta: Weight for soft loss (output distillation)
            gamma: Weight for feature loss (intermediate layer matching)
            temperature: Temperature for soft targets
        """
        with tf.GradientTape() as tape:
            # ========== Teacher predictions (frozen) ==========
            teacher_outputs = teacher_model(teacher_X, training=False)
            
            # ========== Student predictions (trainable) ==========
            student_outputs = student_model(student_X, training=True)
            
            # Handle potential shape mismatches
            if teacher_outputs.shape != student_outputs.shape:
                print(f"Shape mismatch - Teacher: {teacher_outputs.shape}, Student: {student_outputs.shape}")
            
            # ========== 1. L_hard: Ground truth supervision ==========
            hard_loss = tf.keras.losses.binary_crossentropy(
                tf.squeeze(student_y, axis=0), 
                tf.nn.sigmoid(student_outputs)
            )
            hard_loss = tf.reduce_mean(hard_loss)
            
            # ========== 2. L_soft: Knowledge distillation ==========
            # Apply temperature scaling for soft targets
            teacher_soft = tf.nn.sigmoid(teacher_outputs / temperature)
            student_soft = tf.nn.sigmoid(student_outputs / temperature)
            
            # soft_loss = tf.keras.losses.binary_crossentropy(teacher_soft, student_soft)
            # soft_loss = tf.reduce_mean(soft_loss) * (temperature ** 2)
            soft_loss = tf.keras.losses.KLDivergence()(teacher_soft, student_soft) * (temperature ** 2)
           
            # ========== 3. L_feature: Feature matching ==========
            # Extract intermediate features from teacher and student
            teacher_features = extract_teacher_features(teacher_model, teacher_X)
            student_features = extract_student_features(student_model, student_X)
            
            # Compute feature matching loss
            feature_loss = compute_feature_loss(teacher_features, student_features)
            
            # ========== Combined loss: α × L_hard + β × L_soft + γ × L_feature ==========
            total_loss = alpha * hard_loss + beta * soft_loss + gamma * feature_loss
            scaled_total_loss = optimizer.get_scaled_loss(total_loss)
        
        # Compute gradients only for student model
        scaled_gradients = tape.gradient(scaled_total_loss, student_model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        
        return total_loss, hard_loss, soft_loss, feature_loss
    

    # Training loop
    min_batches = min(teacher_batch_seq_test.__len__(), student_batch_seq_test.__len__())
    print(f"\nUsing {min_batches} batches for knowledge distillation")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Shuffle test data at epoch end (following BatchSequence pattern)
        if hasattr(teacher_batch_seq_test, 'on_epoch_end'):
            teacher_batch_seq_test.on_epoch_end()
        if hasattr(student_batch_seq_test, 'on_epoch_end'):
            student_batch_seq_test.on_epoch_end()
        
        epoch_losses = []
        epoch_soft_losses = []
        epoch_hard_losses = []
        epoch_feature_losses = []
        
        # Process batches
        for batch_idx in range(min_batches):
            try:
                # Get batches (following train_model pattern)
                teacher_batch_data = teacher_batch_seq_test.__getitem__(batch_idx)
                student_batch_data = student_batch_seq_test.__getitem__(batch_idx)
                
                teacher_X, _ = teacher_batch_data
                student_X, student_y = student_batch_data
                
                # Convert to tensors (following the list handling in train_model)
                if isinstance(teacher_X, list):
                    teacher_X = [tf.convert_to_tensor(x, dtype=tf.float32) for x in teacher_X]
                else:
                    teacher_X = tf.convert_to_tensor(teacher_X, dtype=tf.float32)
                
                if isinstance(student_X, list):
                    student_X = [tf.convert_to_tensor(x, dtype=tf.float32) for x in student_X]
                else:
                    student_X = tf.convert_to_tensor(student_X, dtype=tf.float32)
                
                student_y = tf.convert_to_tensor(student_y, dtype=tf.float32)
                
                # Knowledge distillation training step
                total_loss, soft_loss, hard_loss, feature_loss = distillation_train_step(
                    teacher_X, student_X, student_y
                )
                
                epoch_losses.append(float(total_loss))
                epoch_soft_losses.append(float(soft_loss))
                epoch_hard_losses.append(float(hard_loss))
                epoch_feature_losses.append(float(feature_loss))
                
                # # Progress reporting (every 5 batches since we have limited data)
                # if batch_idx % 5 == 0:
                #     print(f"  Batch {batch_idx}/{min_batches}: "
                #           f"Total={total_loss:.4f}, Soft={soft_loss:.4f}, Hard={hard_loss:.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
        if epoch_losses:
            avg_total = sum(epoch_losses) / len(epoch_losses)
            avg_soft = sum(epoch_soft_losses) / len(epoch_soft_losses)
            avg_hard = sum(epoch_hard_losses) / len(epoch_hard_losses)
            avg_feature = sum(epoch_feature_losses) / len(epoch_feature_losses)
            
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Average Total Loss: {avg_total:.4f}")
            print(f"  Average Soft Loss: {avg_soft:.4f} (Teacher guidance)")
            print(f"  Average Hard Loss: {avg_hard:.4f} (Ground truth)")
            print(f"  Average Feature Loss: {avg_feature:.4f}")
        
        # Save checkpoint every 5 epochs
        if save_path and (epoch + 1) % 5 == 0:
            checkpoint_path = save_path.replace('.h5', f'_epoch_{epoch + 1}.h5')
            student_model.save(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final distilled model
    if save_path:
        student_model.save(save_path)
        print(f"Final distilled model saved to: {save_path}")

    # Plot distillation learning curves
    plot_distillation_curves(epochs, total_loss, soft_loss, hard_loss)
    
    print("Knowledge distillation completed!")
    return student_model

# PROBLEM: student_batch_gen valid is training set, this should not be the case
def fine_tune_distilled_model_with_batch_gen(
    student_model, student_batch_gen, 
    epochs=15, learning_rate=1e-5, 
    use_train_set=True, save_path=None
):
    """
    Fine-tune the distilled model using batch_gen (following train_model pattern)
    
    Args:
        student_model: Distilled student model
        student_batch_gen: Batch generator for student data
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        use_train_set: Whether to use train set (True) or continue with test set (False)
        save_path: Path to save fine-tuned model
    """
    print("Fine-tuning distilled model...")
    
    # Recompile model for fine-tuning
    models.compile_model(
        student_model,
        optimizer='adam',
        loss='binary_crossentropy',
        opt_kwargs={'learning_rate': learning_rate}
    )
    
    # Create strategy (following train_model pattern)
    if len(tf.config.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    
    with strategy.scope():
        # Calculate steps (following train_model pattern)
        dataset_name = "valid" if use_train_set else "test"
        steps_per_epoch = len(student_batch_gen.time_coords[dataset_name]) // BATCH_SIZE
        
        # Create BatchSequence for training
        ####################################################
        # INVERSED train and valid until finding solution
        ####################################################
        if use_train_set:
            batch_seq_train = batch.BatchSequence(student_batch_gen, dataset='valid')
            validation_data = batch.BatchSequence(student_batch_gen, dataset='train') if 'train' in student_batch_gen.time_coords else None
            validation_steps = len(student_batch_gen.time_coords['train']) // BATCH_SIZE if validation_data else None
        else:
            batch_seq_train = batch.BatchSequence(student_batch_gen, dataset='test')
            validation_data = None
            validation_steps = None
        
        print(f"Fine-tuning steps per epoch: {steps_per_epoch}")
        print(f"Using dataset: {dataset_name}")
        
        # Setup callbacks (following train_model pattern)
        callbacks = []
        
        if save_path:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                save_path, save_weights_only=False, save_best_only=True,
                monitor='loss', mode='min', verbose=1
            )
            callbacks.append(checkpoint)
        
        callbacks.extend([
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=3, mode="min", factor=0.2, monitor="loss",
                verbose=1, min_delta=0.0
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=6, mode="min", restore_best_weights=True,
                monitor="loss"
            )
        ])
        
        # Fine-tune the model
        history = student_model.fit(
            batch_seq_train,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
    
    print("Fine-tuning completed!")
    return student_model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'task', 
        type=str
    )
    parser.add_argument(
        '--sources', 
        type=str
    )
    parser.add_argument(
        '--target', 
        type=str, 
        default="occurrence-8-10"
    )
    parser.add_argument(
        '--prefix', 
        type=str, 
        default="lightning"
    )
    parser.add_argument(
        '--overwrite', 
        type=bool, 
        default=False
    )
    parser.add_argument(
        '--kd_ft_years', 
        type=str, 
        nargs=2, 
        metavar=('DISTILL_YEAR', 'FINE_TUNE_YEAR'), 
        help='Years for knowledge distillation and fine-tuning'
    )
    parser.add_argument(
        '--development',
        '-dev',
        action='store_true',
        help='Perform knowledge distillation/fine-tuning'
    )
    parser.add_argument(
        '--distillation',
        '-distill',
        action='store_true',
        help='Perform knowledge distillation'
    )
    parser.add_argument(
        '--finetuning',
        '-tune',
        action='store_true',
        help='Perform fine-tuning'
    )
    parser.add_argument(
        '--romanian_evaluation',
        '-ro_eval',
        action='store_true',
        help='Perform evaluation on Romanian data'
    )
    parser.add_argument(
        '--swiss_evaluation',
        '-swiss_eval',
        action='store_true',
        help='Perform evaluation on Swiss data'
    )
    parser.add_argument(
        '--romanian_generation',
        '-ro_gen',
        action='store_true',
        help='Perform generation on Romanian data'
    )
    parser.add_argument(
        '--swiss_generation',
        '-swiss_gen',
        action='store_true',
        help='Perform generation on Swiss data'
    )
    parser.add_argument(
        '--create_conf_matrix',
        '-conf_mat',
        action='store_true',
        help='Create confusion matrix'
    )
    parser.add_argument(
        '--evaluate_model',
        '-eval_model',
        action='store_true',
        help='Evaluate model'
    )
    parser.add_argument(
        '--generate_predictions',
        '-gen_pred',
        action='store_true',
        help='Generate predictions'
    )
    parser.add_argument(
        '--check_products',
        '-cp',
        action='store_true',
        help='Check which products are beneficial to model predictions'
    )
    parser.add_argument(
        '--upscale_patches',
        '-upscale',
        type=str,
        choices=['save', 'load', 'none'],
        default='none',
        help='How to handle 8×8 patch upscaling: "save" to upscale and save to disk, '
            '"load" to load pre-saved upscaled patches, "none" to upscale in memory (default)'
    )
    parser.add_argument(
        '--diagnose_stitching',
        '-stitch',
        action='store_true',
        help='Save diagnostic information during batch generation'
    )
    parser.add_argument(
        '--visualize_hour',
        '-visual_h',
        type=str,
        default=None,
        help='Hour to visualize (00-23) for diagnostic plots'
    )
    parser.add_argument(
        '--visualize_minutes',
        '-visual_m',
        type=str,
        default=None,
        help='Minutes to visualize (00-59) for diagnostic plots'
    )
    parser.add_argument(
        '--plot_timestamps',
        '-plot_ts',
        type=str,
        default=None,
        required=False,
        help='String that determines to plot the first 2 timestamps or the last 2 timestamps. '
            'Options are "first" or "last". Default is None (no plotting).'
    )
    parser.add_argument(
        '--plot_regridded',
        '-plot_regrid',
        action='store_true',
        help='Plot regridded meteorological data for a specific timestamp'
    )
    parser.add_argument(
        '--date',
        type=str,
        default='2024-06-13',
        help='Date for regridded data plotting in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--products',
        type=str,
        nargs='+',
        default=['RZC', 'HRV', 'occurrence'],
        help='List of meteorological products to plot for regridded data'
    )
    parser.add_argument(
        '--transformed',
        action='store_true',
        help='Indicates whether to plot transformed regridded data'
    )
    args = parser.parse_args()
    task = args.task

    # if task == "train_sources":
    #     sources_str = args.sources
    #     target = args.target
    #     fn_prefix = args.prefix
    #     overwrite = args.overwrite
    #     (distill_year, _) = args.kd_ft_years
    #     model_exists = os.path.isfile(
    #         f"../models/{fn_prefix}-{sources_str}.h5"
    #     )
    #     if model_exists and not overwrite:
    #         return
    #     training_sources(sources_str, file_suffix=distill_year, target=target, fn_prefix=fn_prefix)
    
    # elif task == "eval_sources":
    #     sources_str = args.sources
    #     target = args.target
    #     fn_prefix = args.prefix
    #     (distill_year, _) = args.kd_ft_years

    #     eval_sources(sources_str, file_suffix=distill_year, target=target, fn_prefix=fn_prefix)
    
    # elif task == "eval_sources_leadtime":
    #     sources_str = args.sources
    #     target = args.target
    #     fn_prefix = args.prefix
    #     (distill_year, _) = args.kd_ft_years

    #     eval_sources(sources_str, file_suffix=distill_year, target=target, fn_prefix=fn_prefix,
    #         separate_leadtimes=True)
        
    if task == "kd_ft":
        if args.romanian_generation or args.swiss_generation or args.development:
            sources_str = args.sources
            target = args.target
            (distill_year, fine_tune_year) = args.kd_ft_years
            print(f"Distill year: {distill_year}, fine-tune year: {fine_tune_year}")

            # Get current working directory and define paths robustly
            current_dir = os.getcwd()
            print(f"Current working directory: {current_dir}")

            # Navigate to project root (go up one level from scripts directory)
            project_root = os.path.dirname(current_dir)

            # Define target directories using os.path.join for cross-platform compatibility
            models_dir = os.path.join(project_root, "knowledge_distillation", "models")
            data_dir = os.path.join(project_root, "knowledge_distillation", "distill_batches")

            # Create directories if they don't exist
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)

            print(f"Models path: {models_dir}")
            print(f"Data path: {data_dir}")

            # Student path
            student_model_filename = f"student_model_{fine_tune_year}_{sources_str}_{target}.h5"
            student_model_path = os.path.join(models_dir, student_model_filename)

            # Student distilled model save path
            distilled_model_filename = f"student_distilled_model_{distill_year}_to_{fine_tune_year}_{sources_str}_{target}.h5"
            student_distilled_model_path = os.path.join(models_dir, distilled_model_filename)

            # Fine-tuned model save path
            fine_tuned_model_filename = f"student_fine_tuned_model_{fine_tune_year}_{sources_str}_{target}.h5"
            student_fine_tuned_model_path = os.path.join(models_dir, fine_tuned_model_filename)

            # Teacher path
            teacher_model_filename = f"teacher_model_{distill_year}_{sources_str}_{target}.h5"
            teacher_model_path = os.path.join(models_dir, teacher_model_filename)

            # Romanian batch_gen path
            romanian_data_filename = f"romanian_data_batches_{fine_tune_year}_{sources_str}_{target}.pkl"
            romanian_data_path = os.path.join(data_dir, romanian_data_filename)

            # Swiss batch_gen path
            swiss_data_filename = f"swiss_data_batches_{distill_year}_{sources_str}_{target}.pkl"
            swiss_data_path = os.path.join(data_dir, swiss_data_filename)

            # Convert 'none' to None
            upscale_patches = None if args.upscale_patches == 'none' else args.upscale_patches

        else:
            # Checking regridded data
            if args.plot_regridded and args.visualize_hour is not None and args.visualize_minutes is not None:
                plot_meteorological_regridded_data(
                    args.products,
                    args.date + ' ' + args.visualize_hour + ':' + args.visualize_minutes,
                    args
                )
            elif args.plot_regridded and args.visualize_hour is None and args.visualize_minutes is None:
                save_all_regridded_data_plots(args.products, args)

        if args.romanian_generation:

            # Return the student model that will be fine-tuned on Romanian data
            romanian_data_batches, student_model = generate_sources(
                sources_str, 
                args=args,
                file_suffix=fine_tune_year, 
                target=target,
                upscale_mode=upscale_patches
            )

            # Summary of the models (save as text file)
            # Save model summary directly to file
            with open('student_model_summary.txt', 'w') as f:
                student_model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.close()
            print("Student model summary saved!")

            # Save student model in H5 format
            student_model.save(student_model_path)
            print(f"Student model saved to: {student_model_path}")

            # Save Romanian data batches
            with open(romanian_data_path, 'wb') as f:
                dill.dump(romanian_data_batches, f)
            print(f"Romanian data batches saved to: {romanian_data_path}")

        elif args.swiss_generation:
            
            # Return the teacher model that was trained on Swiss data
            swiss_data_batches, teacher_model = generate_sources(
                sources_str, 
                file_suffix=distill_year, 
                target=target
            )

            with open('teacher_model_summary.txt', 'w') as f:
                teacher_model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.close()
            print("Teacher model summary saved!")

            # Save teacher model in H5 format
            teacher_model.save(teacher_model_path)
            print(f"Teacher model saved to: {teacher_model_path}")

            # Save Swiss data batches
            with open(swiss_data_path, 'wb') as f:
                dill.dump(swiss_data_batches, f)
            print(f"Swiss data batches saved to: {swiss_data_path}")

        elif args.development:
            # You would load your batch_gen objects here
            # Load romanian batch_gen
            with open(romanian_data_path, 'rb') as f:
                romanian_data_batches = dill.load(f)
            print(f"Romanian data batches loaded from: {romanian_data_path}")

            # Load swiss batch_gen
            with open(swiss_data_path, 'rb') as f:
                swiss_data_batches = dill.load(f)
            print(f"Swiss data batches loaded from: {swiss_data_path}")

            # Perform knowledge distillation only
            if args.distillation:
                # Phase 1: Knowledge Distillation on limited test set
                print("\nPhase 1: Knowledge Distillation")
                distilled_model = knowledge_distillation_with_batch_gen(
                    student_batch_gen=romanian_data_batches,
                    teacher_batch_gen=swiss_data_batches,
                    student_model_path=student_model_path,
                    teacher_model_path=teacher_model_path,
                    epochs=DISTILL_EPOCHS,
                    n_test_samples=DISTILL_SAMPLES
                )
                # Save distilled model
                distilled_model.save(student_distilled_model_path)

            # Perform fine-tuning only
            elif args.finetuning:
                # Load distilled model
                distilled_model = load_model(
                    student_distilled_model_path, 
                    compile=False, 
                    custom_objects=custom_objects
                )

                # Phase 2: Fine-tuning on full training set
                print("\nPhase 2: Fine-tuning on training set")
                fine_tuned_model, history = fine_tune_distilled_model_with_batch_gen(
                    student_model=distilled_model,
                    student_batch_gen=romanian_data_batches,
                    epochs=FINETUNE_EPOCHS,
                    use_train_set=True  # Use full training set for fine-tuning
                )

                # Save fine-tuned model
                fine_tuned_model.save(student_fine_tuned_model_path)

                # Plot and save all metrics
                plot_training_history(history, save_dir='train_history_plots')

                # # Also create confusion matrix specific plots
                # plot_confusion_matrix_metrics(history, save_dir='train_history_plots')

            # Perform evaluation on Romanian data
            elif args.romanian_evaluation:
                # Load fine-tuned model
                fine_tuned_model = load_model(
                    student_fine_tuned_model_path, 
                    custom_objects=custom_objects
                )

                # Evaluate the fine-tuned model
                print("\nEvaluating fine-tuned model")
                # Evaluation and predictions on train dataset because it was used for validation
                # Validation dataset was used for training
                preds, targets = eval_sources(
                    sources_str, 
                    fine_tuned_model, 
                    romanian_data_batches,
                    gt=target,
                    args=args, 
                    idx=2, # batch number that returns batch of shape (8, 12, 256, 256, 1)
                    dataset='valid'
                )

                # # Create visualizations
                # visualize_lightning_nowcasting_complete(
                #     preds, 
                #     targets, 
                #     save_dir='fine-tuned_model_results'
                # )

                # For BZC data
                visualize_continuous_nowcasting_complete(
                    preds, 
                    targets, 
                    save_dir='fine-tuned_model_results',
                    cmap='viridis',
                    var_name='BZC'
                )

            # Perform evaluation on Swiss data
            elif args.swiss_evaluation:
                # Load fine-tuned model
                teacher_model = load_model(
                    teacher_model_path, 
                    custom_objects=custom_objects
                )

                # Check what inputs to exclude using Xi + Grad-CAM
                # Using the first batch for analysis
                batch_seq = batch.BatchSequence(swiss_data_batches, dataset='test')
                batch_data = batch_seq.__getitem__(0)
                data, _ = batch_data

                # Evaluate the fine-tuned model
                print("\nEvaluating teacher model")
                # Evaluation and predictions on train dataset because it was used for validation
                # Validation dataset was used for training
                preds, targets = eval_sources(
                    sources_str, 
                    teacher_model, 
                    swiss_data_batches, 
                    gt=target,
                    args=args,
                    idx=2, 
                    dataset='test'
                )

                # # Create visualizations
                # visualize_lightning_nowcasting_complete(
                #     preds, 
                #     targets, 
                #     save_dir='teacher_model_results'
                # )
                
                if args.check_products:
                    print("\nAnalyzing teacher model with Xi + Grad-CAM")
                    # ============================================================================
                    target_resblocks = [
                        layer.name if len(layer.name.split('_')) == 3 else layer.name + '_0' for layer in teacher_model.layers if 'res_block' in layer.name
                    ][:4] # extract only first 4 ResBlocks

                    concatenate_layers = [
                        layer.name if len(layer.name.split('_')) == 2 else layer.name + '_0' for layer in teacher_model.layers if 'concatenate' in layer.name
                    ][:4] # extract only first 4 Concatenate layers
                    
                    print(f"Target ResBlocks: {target_resblocks}")
                    print(f"Concatenate layers: {concatenate_layers}")

                    # ============================================================================
                    # RUN ANALYSIS - ONE RESBLOCK AT A TIME FOR MEMORY EFFICIENCY
                    # ============================================================================

                    print("\n" + "="*70)
                    print("STARTING GRAD-CAM + XI CORRELATION ANALYSIS")
                    print("="*70)

                    # MEMORY-EFFICIENT APPROACH: Analyze ResBlocks one by one
                    # Process high-resolution blocks separately from low-resolution ones

                    # Group 1: Past timeframe blocks
                    past_blocks = sorted(target_resblocks, key=lambda x: int(x.split('_')[-1]))[:2]
                    past_concat = sorted(concatenate_layers, key=lambda x: int(x.split('_')[-1]))[:2]

                    # Extract '_0' suffix if present
                    past_blocks = [c.replace('_0', '') if i==0 else c for i, c in enumerate(past_blocks)]
                    past_concat = [c.replace('_0', '') if i==0 else c for i, c in enumerate(past_concat)]
                    print(past_blocks)
                    print(past_concat)

                    # Group 2: Future timeframe blocks
                    future_blocks = sorted(target_resblocks, key=lambda x: int(x.split('_')[-1]))[2:]
                    future_concat = sorted(concatenate_layers, key=lambda x: int(x.split('_')[-1]))[2:]

                    # Extract '_0' suffix if present
                    future_blocks = [s.replace('_0', '') if i == 0 else s for i, s in enumerate(future_blocks)]
                    future_concat = [s.replace('_0', '') if i == 0 else s for i, s in enumerate(future_concat)]
                    print(future_blocks)
                    print(future_concat)


                    # Define input names for each concatenation layer
                    input_names_by_concat = {
                        past_concat[0]: [
                            'RZC', 'CZC', 'EZC-20', 'EZC-45', 'HZC', 'LZC',
                            'density', 'current', 'HRV', 'occurrence-8-10',
                            'SOILTYP', 'Altitude', 'EW-deriv', 'NS-deriv'
                        ],
                        past_concat[1]: [
                            'ctth-tempe', 'ctth-alti', 'cmic-phase', 'cmic-cot',
                            'VIS006', 'VIS008', 'IR-016', 'IR-039', 'WV-062', 'WV-073',
                            'IR-087', 'IR-097', 'IR-108', 'IR-120', 'IR-134', 'sun-z'
                        ],
                        future_concat[1]: [
                            'SOILTYP', 'Altitude', 'EW-deriv', 'NS-deriv'
                        ],
                        future_concat[0]: [
                            'CAPE-MU', 'CIN-MU', 'HZEROCL', 'LCL-ML', 
                            'MCONV', 'OMEGA', 'SLI', 'T-2M', 'T-SO'
                        ]
                    }

                    # ============================================================================
                    # RUN GRAD-CAM + XI ANALYSIS - MEMORY EFFICIENT
                    # ============================================================================

                    print("\n" + "="*70)
                    print("STARTING GRAD-CAM + XI CORRELATION ANALYSIS")
                    print("="*70)

                    all_results = {}
                    all_dataframes_avg = []
                    all_dataframes_time = []

                    # Process PAST timeframe blocks
                    print("\n" + "="*70)
                    print("PROCESSING PAST TIMEFRAME BLOCKS")
                    print("="*70)

                    for resblock, concat in zip(past_blocks, past_concat):
                        input_names = input_names_by_concat[concat]
                        
                        print(f"\n--- Analyzing {resblock} (concatenation: {concat}) ---")
                        print(f"Number of inputs: {len(input_names)}")
                        
                        # Clear memory before processing
                        tf.keras.backend.clear_session()
                        gc.collect()
                        
                        # Run analysis for this specific ResBlock
                        try:
                            results, df_avg, df_time = run_complete_analysis(
                                model=teacher_model,
                                input_data_batch=data,
                                concat_layer_name=concat,
                                resblock_name=resblock,
                                input_names=input_names,
                                output_dir=f'results_{resblock}'
                            )
                            
                            all_results[resblock] = results
                            all_dataframes_avg.append(df_avg.assign(ResBlock=resblock))
                            all_dataframes_time.append(df_time.assign(ResBlock=resblock))
                            
                            print(f"\nCompleted {resblock}")
                            print(f"Top 5 inputs by Xi:")
                            print(df_avg.head(5))
                            
                        except Exception as e:
                            print(f"ERROR processing {resblock}: {str(e)}")
                            continue
                        
                        # Clear memory after processing
                        tf.keras.backend.clear_session()
                        gc.collect()

                    # Process FUTURE timeframe blocks
                    print("\n" + "="*70)
                    print("PROCESSING FUTURE TIMEFRAME BLOCKS")
                    print("="*70)

                    for resblock, concat in zip(future_blocks, future_concat):
                        input_names = input_names_by_concat[concat]
                        
                        print(f"\n--- Analyzing {resblock} (concatenation: {concat}) ---")
                        print(f"Number of inputs: {len(input_names)}")
                        
                        # Clear memory before processing
                        tf.keras.backend.clear_session()
                        gc.collect()
                        
                        # Run analysis
                        try:
                            results, df_avg, df_time = run_complete_analysis(
                                model=teacher_model,
                                input_data_batch=data,
                                concat_layer_name=concat,
                                resblock_name=resblock,
                                input_names=input_names,
                                output_dir=f'results_{resblock}'
                            )
                            
                            all_results[resblock] = results
                            all_dataframes_avg.append(df_avg.assign(ResBlock=resblock))
                            all_dataframes_time.append(df_time.assign(ResBlock=resblock))
                            
                            print(f"\nCompleted {resblock}")
                            print(f"Top 5 inputs by Xi:")
                            print(df_avg.head(5))
                            
                        except Exception as e:
                            print(f"ERROR processing {resblock}: {str(e)}")
                            continue
                        
                        # Clear memory
                        tf.keras.backend.clear_session()
                        gc.collect()

                    # ============================================================================
                    # COMBINE RESULTS ACROSS ALL RESBLOCKS
                    # ============================================================================

                    os.makedirs('results_combined', exist_ok=True)

                    if len(all_dataframes_avg) > 0:
                        # Combine all averaged correlations
                        df_all_averaged = pd.concat(all_dataframes_avg, ignore_index=True)
                        df_all_averaged.to_csv('results_combined/all_averaged_correlations.csv', index=False)
                        
                        # Combine all timestep correlations
                        df_all_timesteps = pd.concat(all_dataframes_time, ignore_index=True)
                        df_all_timesteps.to_csv('results_combined/all_timestep_correlations.csv', index=False)
                        
                        # Aggregate: Average Xi across all ResBlocks for each unique input
                        df_global_importance = (df_all_averaged.groupby('Input')
                                                .agg({'Xi': ['mean', 'std', 'min', 'max', 'count']})
                                                .reset_index())
                        df_global_importance.columns = ['Input', 'Xi_mean', 'Xi_std', 'Xi_min', 'Xi_max', 'Xi_count']
                        df_global_importance = df_global_importance.sort_values('Xi_mean', ascending=False)
                        
                        print("\n" + "="*70)
                        print("GLOBAL INPUT IMPORTANCE RANKING")
                        print("="*70)
                        print(df_global_importance.to_string())
                        
                        df_global_importance.to_csv('results_combined/global_input_importance.csv', index=False)
                        
                        # ============================================================================
                        # IDENTIFY LOW-IMPORTANCE INPUTS FOR REMOVAL
                        # ============================================================================
                        
                        # Define threshold
                        xi_threshold = 0.3  # Adjust based on your results
                        
                        low_importance = df_global_importance[df_global_importance['Xi_mean'] < xi_threshold]
                        high_importance = df_global_importance[df_global_importance['Xi_mean'] >= xi_threshold]
                        
                        print("\n" + "="*70)
                        print(f"FEATURE SELECTION BASED ON Xi THRESHOLD: {xi_threshold}")
                        print("="*70)
                        
                        print(f"\nHIGH IMPORTANCE INPUTS (Xi >= {xi_threshold}):")
                        print(f"Count: {len(high_importance)}")
                        for _, row in high_importance.iterrows():
                            print(f"  {row['Input']:<20} Xi: {row['Xi_mean']:.4f} ± {row['Xi_std']:.4f}")
                        
                        print(f"\nLOW IMPORTANCE INPUTS (Xi < {xi_threshold}) - CANDIDATES FOR REMOVAL:")
                        print(f"Count: {len(low_importance)}")
                        for _, row in low_importance.iterrows():
                            print(f"  {row['Input']:<20} Xi: {row['Xi_mean']:.4f} ± {row['Xi_std']:.4f}")
                        
                        print(f"\nPotential input reduction: {len(low_importance)}/{len(df_global_importance)} ")
                        print(f"({len(low_importance) / len(df_global_importance) * 100:.1f}%)")
                        
                        # Save lists
                        with open('results_combined/inputs_to_keep.txt', 'w') as f:
                            f.write("# High importance inputs to KEEP\n")
                            for inp in high_importance['Input'].tolist():
                                f.write(f"{inp}\n")
                        
                        with open('results_combined/inputs_to_remove.txt', 'w') as f:
                            f.write("# Low importance inputs - candidates for REMOVAL\n")
                            for inp in low_importance['Input'].tolist():
                                f.write(f"{inp}\n")
                        
                        # ============================================================================
                        # CREATE GLOBAL VISUALIZATIONS
                        # ============================================================================
                        
                        # Global bar chart
                        fig_global = go.Figure(data=[
                            go.Bar(
                                x=df_global_importance['Input'], 
                                y=df_global_importance['Xi_mean'],
                                error_y=dict(type='data', array=df_global_importance['Xi_std']),
                                marker_color=df_global_importance['Xi_mean'],
                                marker_colorscale='RdYlBu',
                                marker_cmin=0,
                                marker_cmax=1,
                                text=np.round(df_global_importance['Xi_mean'].values, 3),
                                textposition='auto'
                            )
                        ])
                        
                        fig_global.add_hline(y=xi_threshold, line_dash="dash", line_color="red",
                                            annotation_text=f"Threshold: {xi_threshold}")
                        
                        fig_global.update_layout(
                            title='Global Input Importance - Averaged Across All ResBlocks',
                            xaxis_title='Input Variable',
                            yaxis_title='Mean Xi Correlation',
                            height=600,
                            width=1600,
                            showlegend=False
                        )
                        fig_global.update_xaxes(tickangle=45)
                        fig_global.write_html('results_combined/global_importance.html')
                        print("\nSaved: results_combined/global_importance.html")
                        
                        # Heatmap: ResBlock vs Input
                        pivot_resblock = df_all_averaged.pivot(index='Input', columns='ResBlock', values='Xi')
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=pivot_resblock.values,
                            x=pivot_resblock.columns,
                            y=pivot_resblock.index,
                            colorscale='RdYlBu',
                            zmid=0.5,
                            zmin=0,
                            zmax=1,
                            text=np.round(pivot_resblock.values, 3),
                            texttemplate='%{text}',
                            textfont={"size": 9},
                            colorbar=dict(title='Xi Correlation')
                        ))
                        
                        fig_heatmap.update_layout(
                            title='Xi Correlation Heatmap: Inputs vs ResBlocks',
                            xaxis_title='ResBlock',
                            yaxis_title='Input Variable',
                            height=1000,
                            width=900
                        )
                        fig_heatmap.write_html('results_combined/resblock_comparison.html')
                        print("Saved: results_combined/resblock_comparison.html")
                        
                        # Timestep analysis across all inputs
                        if len(all_dataframes_time) > 0:
                            
                            fig_time_heatmap = create_interactive_heatmap(df_all_timesteps)
                            fig_time_heatmap.write_html('results_combined/timestep_heatmap_all.html')
                            print("Saved: results_combined/timestep_heatmap_all.html")
                            
                            fig_bar = create_clustered_bar_chart(df_all_timesteps, top_n=15)
                            fig_bar.write_html('results_combined/timestep_barchart_top15.html')
                            print("Saved: results_combined/timestep_barchart_top15.html")
                            
                            fig_box = create_box_plots(df_all_timesteps)
                            fig_box.write_html('results_combined/timestep_boxplots.html')
                            print("Saved: results_combined/timestep_boxplots.html")

                    else:
                        print("\nWARNING: No results were successfully processed.")

                # Plot model calibration and confusion matrix from npy files
                result_dir = os.path.join("../results/", 'test')

                # Load calibration and confusion matrix for lightning target
                calibration_path = os.path.join(result_dir, f"calibration-{sources_str}.npy")
                conf_matrix_path = os.path.join(result_dir, f"conf_matrix-{sources_str}.npy")
                if (os.path.isfile(calibration_path) or os.path.isfile(conf_matrix_path)) and target == "occurrence-8-10":
                    print("\nTeacher model calibration and confusion matrix")
                    calibration = np.load(calibration_path)
                    conf_matrix = np.load(conf_matrix_path)
                    
                    # For your confusion matrix (shape: 2, 2, 1001)
                    plot_confusion_matrix_median(conf_matrix, 
                                                class_names=['No Lightning', 'Lightning'],
                                                title='Weather Forecast - Confusion Matrix')

                    # For your calibration data (shape: 100,)
                    plot_calibration_bins(calibration,
                                        title='Weather Forecast - Calibration Plot')
                
                


if __name__ == "__main__":
    main()
