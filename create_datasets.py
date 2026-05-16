"""
create_datasets.py — COALITION-4 Romanian Adaptation
=====================================================
Creates train, validation, and test TF datasets from pre-extracted .npy patches.

Usage:
    python create_datasets.py --mode mtg_lightning --data_root ./our_data
    python create_datasets.py --mode mtg_radar --data_root ./our_data
    python create_datasets.py --mode mtg_radar_continuous --data_root ./our_data

OPERA Shapley study (4-model coalition, "is NWCSAF useful?"):
    python create_datasets.py --mode mtg_opera_radar_only --data_root ./our_data
    python create_datasets.py --mode mtg_opera_mtgmr      --data_root ./our_data
    python create_datasets.py --mode mtg_opera_nwcsaf     --data_root ./our_data
    python create_datasets.py --mode mtg_opera_full       --data_root ./our_data

The training cadence is read from our_data/timestep_config.json (set via
validate_timestep.py) and the per-sample window from our_data/sequence_meta.json
(written by extract_patch_seq_for_datasets.py). MSG modes are disabled in this
build — see comments in get_mode_config().

Inputs:
    - train_data.csv, validation_data.csv, test_data.csv in data_root/
    - sequence_meta.json in data_root/ (step_minutes, past_steps, future_steps)
    - .npy patch files in data_root/patches/{date}/{variable}_{HHMM}_{HR|LR}.npy

Outputs:
    - Saved tf.data.Dataset in data_root/datasets/{mode}/train/
    - Saved tf.data.Dataset in data_root/datasets/{mode}/validation/
    - Saved tf.data.Dataset in data_root/datasets/{mode}/test/
    - metadata.json per split (input_shapes, label_type, step_minutes, ...)
"""

import argparse
import ast
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================================
# Per-variable transforms (data-driven via normalization_stats.json)
# ============================================================================
#
# Variables that use linear z-score `(x - mean) / std` or log-then-z-score
# `(log10(clip(x)) - mean) / std` look their `mean` / `std` up from
# `our_data/normalization_stats.json` (produced by
# `compute_normalization_stats.py`). The JSON is loaded lazily on the first
# call and cached. There is NO fallback to the Leinonen Swiss constants:
# if the JSON is missing, or if a required variable is missing from it,
# the call raises a clear error pointing the user at the stats script.
#
# Variables that use simple physical scaling (BZC, VIS, EZC = x/k) or are
# categorical/binary (occurrence, cmic_phase) are NOT driven by the JSON.
# Their transforms are hardcoded because the scaling factor is a property
# of the physical units, not the training distribution.

_NORMALIZATION_STATS_CACHE: dict | None = None
_NORMALIZATION_PATH: Path | None = None
_NORMALIZATION_WARNED: set[str] = set()


def _load_normalization_stats(force: bool = False) -> dict:
    """Load normalization_stats.json (lazy, cached). Errors if absent."""
    global _NORMALIZATION_STATS_CACHE, _NORMALIZATION_PATH
    if _NORMALIZATION_STATS_CACHE is not None and not force:
        return _NORMALIZATION_STATS_CACHE
    # _NORMALIZATION_PATH defaults to our_data/normalization_stats.json next
    # to this script; override via set_normalization_stats_path() at runtime.
    if _NORMALIZATION_PATH is None:
        _NORMALIZATION_PATH = (
            Path(__file__).resolve().parent / "our_data"
            / "normalization_stats.json"
        )
    if not _NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"normalization_stats.json not found at "
            f"{_NORMALIZATION_PATH}. Build it once before running "
            f"create_datasets.py:\n"
            f"    python compute_normalization_stats.py"
        )
    _NORMALIZATION_STATS_CACHE = json.loads(_NORMALIZATION_PATH.read_text())
    return _NORMALIZATION_STATS_CACHE


def set_normalization_stats_path(path: str | Path) -> None:
    """Override the location of normalization_stats.json before any transform
    runs. Useful for testing or for running create_datasets.py against a
    non-default data_root."""
    global _NORMALIZATION_PATH, _NORMALIZATION_STATS_CACHE
    _NORMALIZATION_PATH = Path(path)
    _NORMALIZATION_STATS_CACHE = None


def _norm(var: str) -> dict:
    """Return the per-variable stats dict; raise if missing.

    A variable can be missing from the JSON either because
    `compute_normalization_stats.py` was not given that variable, or
    because no training files matched. In both cases, fail loudly with
    instructions rather than silently fall back to a Swiss constant.
    """
    stats = _load_normalization_stats()
    block = stats.get("variables", {}).get(var)
    if block is None:
        raise KeyError(
            f"Variable {var!r} is missing from normalization_stats.json "
            f"at {_NORMALIZATION_PATH}. Re-run compute_normalization_stats.py "
            f"and confirm {var!r} appears in the output (no spec entry or "
            f"no training files would suppress it)."
        )
    # First-time encounter warnings: near-constant
    if (block.get("near_constant") and
            var not in _NORMALIZATION_WARNED):
        _NORMALIZATION_WARNED.add(var)
        print(
            f"WARNING [{var}]: {block.get('near_constant_warning', '')} "
            f"({_NORMALIZATION_PATH.name})",
            file=sys.stderr,
        )
    return block


def _apply_log_zscore(x: np.ndarray, var: str) -> np.ndarray:
    """`(log10(clip(x, clip_min)) - mean) / std` with fill + missing handling."""
    spec = _norm(var)
    fill = spec["fill"]
    clip_min = spec.get("clip_min", fill)
    missing_above = spec.get("missing_above")
    mean = spec["mean"]
    std = spec["std"]
    x = np.where(np.isnan(x), fill, x)
    if missing_above is not None:
        x = np.where(x > missing_above, fill, x)
    x = np.clip(x, clip_min, None)
    return (np.log10(x) - mean) / std


def _apply_linear_zscore(x: np.ndarray, var: str) -> np.ndarray:
    """`(x - mean) / std` with fill + missing handling."""
    spec = _norm(var)
    fill = spec["fill"]
    missing_above = spec.get("missing_above")
    mean = spec["mean"]
    std = spec["std"]
    x = np.where(np.isnan(x), fill, x)
    if missing_above is not None:
        x = np.where(x > missing_above, fill, x)
    return (x - mean) / std


# ---- Radar ----
def transform_rzc(x):
    """RZC rain rate: log10 + z-score; heavy-tailed, zero-inflated."""
    return _apply_log_zscore(x, "RZC")


def transform_czc(x):
    """CZC composite reflectivity (dBZ): linear z-score."""
    return _apply_linear_zscore(x, "CZC")


def transform_lzc(x):
    """LZC liquid water content: log10 + z-score; heavy-tailed."""
    return _apply_log_zscore(x, "LZC")


def transform_ezc(x):
    """EZC-20 echo-top height: simple physical scaling (x / 1.97), fill=0.

    Hardcoded scale — not data-driven. Echo-top is a physical altitude
    measure; the divisor expresses an empirically chosen unit conversion
    rather than a statistical centring, so it stays out of the JSON.
    """
    x = np.where(np.isnan(x), 0.0, x)
    return x / 1.97


def transform_bzc(x):
    """BZC base reflectivity: simple physical scaling (x / 100), fill=0.

    Hardcoded. BZC stores integer-coded dBZ * 100 in the source files
    so the `/100` recovers the physical unit; this is a unit conversion,
    not a statistical normalisation.
    """
    x = np.where(np.isnan(x), 0.0, x)
    return x / 100.0


def transform_cpch(x):
    """CPCH precipitation: log10(x) only, fill=0.01, threshold=0.1.

    Hardcoded — pure log transform (no z-score). The threshold drops
    sub-noise rates to the fill value to keep the log finite.
    """
    x = np.where(np.isnan(x), 0.01, x)
    x = np.where(x < 0.1, 0.01, x)
    x = np.clip(x, 0.01, None)
    return np.log10(x)


# ---- Lightning ----
def transform_lightning_density(x):
    """Lightning density: log10 + z-score."""
    return _apply_log_zscore(x, "density")


def transform_lightning_current(x):
    """Lightning current: log10 + z-score."""
    return _apply_log_zscore(x, "current")


def transform_occurrence(x):
    """Lightning occurrence: binary 0/1 — no normalisation.

    Hardcoded. The variable is already on its natural scale {0, 1};
    z-scoring it would destroy the binary interpretation.
    """
    x = np.where(np.isnan(x), 0.0, x)
    return np.clip(x, 0.0, 1.0)


# ---- Satellite (per-channel data-driven stats) ----
def transform_vis(x):
    """Solar visible channels (VIS006, vis_06, ...): x / 100.

    Hardcoded — same unit-conversion logic as BZC: the source data is
    stored as integer reflectance % * 100, so `/100` recovers a value
    already in a sensible model-friendly range. No z-score needed.
    """
    x = np.where(np.isnan(x), 0.0, x)
    return x / 100.0


def transform_ir039(x):
    """MSG IR_039 / MTG ir_38 (solar+thermal): linear z-score."""
    # MSG and MTG share the transform body but use different per-channel
    # stats. The two configs below bind the right variable name.
    return _apply_linear_zscore(x, "IR_039")


def transform_ir38(x):
    """MTG ir_38: linear z-score."""
    return _apply_linear_zscore(x, "ir_38")


def transform_ir108(x):
    """MSG IR_108 thermal channel: linear z-score."""
    return _apply_linear_zscore(x, "IR_108")


def transform_ir105(x):
    """MTG ir_105 thermal channel: linear z-score."""
    return _apply_linear_zscore(x, "ir_105")


def transform_wv062(x):
    """MSG WV_062 water-vapour channel: linear z-score."""
    return _apply_linear_zscore(x, "WV_062")


def transform_wv63(x):
    """MTG wv_63 water-vapour channel: linear z-score."""
    return _apply_linear_zscore(x, "wv_63")


def transform_wv073(x):
    """MSG WV_073 water-vapour channel: linear z-score."""
    return _apply_linear_zscore(x, "WV_073")


def transform_wv73(x):
    """MTG wv_73 water-vapour channel: linear z-score."""
    return _apply_linear_zscore(x, "wv_73")


# ---- NWCSAF ----
def transform_ctth_alti(x):
    """NWCSAF cloud-top altitude: linear z-score (sentinel 65535 dropped)."""
    return _apply_linear_zscore(x, "ctth_alti")


def transform_ctth_tempe(x):
    """NWCSAF cloud-top temperature: linear z-score (sentinel 65535 dropped)."""
    return _apply_linear_zscore(x, "ctth_tempe")


def transform_cmic_phase(x):
    """Cloud-top phase: one-hot.

    Hardcoded. Categorical variable with five classes — no continuous
    normalisation makes sense.

    Input is expected to be int8 codes 0–4 from reproject.py (NaN replaced
    with 0 = "no cloud / missing" at reproject time, so this function no
    longer needs to handle NaN). 4×4 LR pooling in extract_patches.py
    can still produce fractional values when a block mixes categories,
    so a final round-to-nearest still happens here before the one-hot.
    """
    x = np.round(x).astype(np.int32)
    h, w = x.shape
    one_hot = np.zeros((h, w, 5), dtype=np.float32)
    mapping = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}
    for val, ch in mapping.items():
        one_hot[:, :, ch] = (x == val).astype(np.float32)
    known = np.isin(x, list(mapping.keys()))
    one_hot[:, :, 4] = np.where(~known, 1.0, one_hot[:, :, 4])
    return one_hot


def transform_cmic_cot(x):
    """NWCSAF cloud optical thickness: log10 + z-score (sentinel dropped)."""
    return _apply_log_zscore(x, "cmic_cot")


# ---- OPERA radar (new) ----
def transform_opera_reflectivity(x):
    """OPERA max reflectivity (dBZ): linear z-score.

    Reflectivity is already on a logarithmic decibel scale and is roughly
    Gaussian-distributed in non-zero regions, so linear z-scoring is
    appropriate.
    """
    return _apply_linear_zscore(x, "opera_reflectivity")


def transform_opera_rainfall_rate(x):
    """OPERA instantaneous rain rate (mm/h): log10 + z-score.

    Heavy-tailed, zero-inflated, same family as RZC; clip-then-log
    flattens the distribution before z-scoring.
    """
    return _apply_log_zscore(x, "opera_rainfall_rate")


# Label transforms
def label_transform_occurrence(x):
    """Binary lightning occurrence label"""
    x = np.where(np.isnan(x), 0.0, x)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def label_transform_rzc_multiclass(x):
    """RZC rain rate → 5-class one-hot label.
    Classes (mm/h):  0: R<10,  1: 10≤R<20,  2: 20≤R<30,  3: 30≤R<40,  4: R≥40
    Returns (H, W, 5) float32 array.
    """
    x = np.where(np.isnan(x), 0.0, x)
    x = np.clip(x, 0.0, None)
    h, w = x.shape
    one_hot = np.zeros((h, w, 5), dtype=np.float32)
    one_hot[:, :, 0] = (x < 10.0).astype(np.float32)
    one_hot[:, :, 1] = ((x >= 10.0) & (x < 20.0)).astype(np.float32)
    one_hot[:, :, 2] = ((x >= 20.0) & (x < 30.0)).astype(np.float32)
    one_hot[:, :, 3] = ((x >= 30.0) & (x < 40.0)).astype(np.float32)
    one_hot[:, :, 4] = (x >= 40.0).astype(np.float32)
    return one_hot


def label_transform_opera_rainfall_multiclass(x):
    """OPERA instantaneous rain rate → 5-class one-hot label.

    Uses the same bin boundaries as `label_transform_rzc_multiclass`
    (<10, 10–20, 20–30, 30–40, ≥40 mm/h) so that COALITION-4 trained on
    OPERA labels stays comparable with the RZC-trained baseline. The
    label patch is loaded from `opera_rainfall_rate_hr` (HR alias of the
    same reprojected file) so the output shape matches the 256×256 HR head.
    """
    x = np.where(np.isnan(x), 0.0, x)
    x = np.clip(x, 0.0, None)
    h, w = x.shape
    one_hot = np.zeros((h, w, 5), dtype=np.float32)
    one_hot[:, :, 0] = (x < 10.0).astype(np.float32)
    one_hot[:, :, 1] = ((x >= 10.0) & (x < 20.0)).astype(np.float32)
    one_hot[:, :, 2] = ((x >= 20.0) & (x < 30.0)).astype(np.float32)
    one_hot[:, :, 3] = ((x >= 30.0) & (x < 40.0)).astype(np.float32)
    one_hot[:, :, 4] = (x >= 40.0).astype(np.float32)
    return one_hot


def label_transform_rzc_continuous(x):
    """RZC rain rate → continuous label in [0, 1] via min-max normalization.
    NaN → 0, clip to [0, 70], divide by 70.
    Returns (H, W) float32 array.
    """
    x = np.where(np.isnan(x), 0.0, x)
    x = np.clip(x, 0.0, 70.0)
    return (x / 70.0).astype(np.float32)


# Number of label channels per target type
LABEL_CHANNELS = {
    "lightning": 1,           # binary occurrence
    "radar": 5,               # 5-class precipitation
    "radar_continuous": 1,    # continuous rain rate [0, 1]
}


# ============================================================================
# Variable configuration per mode
# ============================================================================

# Variable name → (transform_function, produces_extra_channels)
# produces_extra_channels: None for scalar transforms, int for one-hot etc.

HR_RADAR_CONFIG = {
    "RZC":    (transform_rzc, None),
    "CZC":    (transform_czc, None),
    "EZC-20": (transform_ezc, None),
    "LZC":    (transform_lzc, None),
    "BZC":    (transform_bzc, None),
    "CPCH":   (transform_cpch, None),
}

HR_LIGHTNING_CONFIG = {
    "density":    (transform_lightning_density, None),
    "current":    (transform_lightning_current, None),
    "occurrence": (transform_occurrence, None),
}

MSG_SAT_CONFIG = {
    "VIS006": (transform_vis, None),
    "IR_039": (transform_ir039, None),
    "IR_108": (transform_ir108, None),
    "WV_062": (transform_wv062, None),
    "WV_073": (transform_wv073, None),
}

MTG_HR_SAT_CONFIG = {
    "vis_06": (transform_vis, None),
}

MTG_MR_SAT_CONFIG = {
    "ir_38":  (transform_ir38, None),
    "ir_105": (transform_ir105, None),
    "wv_63":  (transform_wv63, None),
    "wv_73":  (transform_wv73, None),
}

# OPERA radar — both products are 2 km native (MR tier with 2× pool). The
# `opera_rainfall_rate` channel doubles as the label source at the future
# steps; here it is loaded as a past-input feature.
OPERA_MR_CONFIG = {
    "opera_reflectivity":  (transform_opera_reflectivity, None),
    "opera_rainfall_rate": (transform_opera_rainfall_rate, None),
}

NWCSAF_CONFIG = {
    "ctth_alti":  (transform_ctth_alti, None),
    "ctth_tempe": (transform_ctth_tempe, None),
    "cmic_phase": (transform_cmic_phase, 5),  # one-hot → 5 channels
    "cmic_cot":   (transform_cmic_cot, None),
}


def get_mode_config(mode):
    """Return input group configurations and label config for a given mode.

    Returns:
        dict with keys for each input tensor group:
            "past_hr":  (var_config_dict, resolution, suffix)
            "past_lr":  (var_config_dict, resolution, suffix)
            "past_mr":  (var_config_dict, resolution, suffix) or None
        and:
            "label_var": str — variable name for labels
            "label_transform": callable
            "label_suffix": str — HR or LR
    """
    # Common: radar + lightning at HR
    hr_base = {**HR_RADAR_CONFIG, **HR_LIGHTNING_CONFIG}

    # MSG modes are disabled in this build (MSG SEVIRI ingestion was
    # removed from pipeline_msg_mtg.py). The recipes are kept commented
    # so they can be re-enabled if the MSG branch is brought back.
    # if mode == "msg_lightning":
    #     return {
    #         "past_hr": (hr_base, 256, "HR"),
    #         "past_lr": ({**MSG_SAT_CONFIG, **NWCSAF_CONFIG}, 64, "LR"),
    #         "past_mr": None,
    #         "label_var": "occurrence",
    #         "label_transform": label_transform_occurrence,
    #         "label_suffix": "HR",
    #         "label_type": "lightning",
    #     }
    # elif mode == "msg_radar":
    #     return {
    #         "past_hr": (hr_base, 256, "HR"),
    #         "past_lr": ({**MSG_SAT_CONFIG, **NWCSAF_CONFIG}, 64, "LR"),
    #         "past_mr": None,
    #         "label_var": "RZC",
    #         "label_transform": label_transform_rzc_multiclass,
    #         "label_suffix": "HR",
    #         "label_type": "radar",
    #     }
    if mode == "mtg_lightning":
        hr_with_vis = {**hr_base, **MTG_HR_SAT_CONFIG}
        return {
            "past_hr": (hr_with_vis, 256, "HR"),
            "past_mr": (MTG_MR_SAT_CONFIG, 128, "LR"),  # 2km stored as LR
            "past_lr": (NWCSAF_CONFIG, 64, "LR"),
            "label_var": "occurrence",
            "label_transform": label_transform_occurrence,
            "label_suffix": "HR",
            "label_type": "lightning",
        }
    elif mode == "mtg_radar":
        hr_with_vis = {**hr_base, **MTG_HR_SAT_CONFIG}
        return {
            "past_hr": (hr_with_vis, 256, "HR"),
            "past_mr": (MTG_MR_SAT_CONFIG, 128, "LR"),
            "past_lr": (NWCSAF_CONFIG, 64, "LR"),
            "label_var": "RZC",
            "label_transform": label_transform_rzc_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }
    # elif mode == "msg_radar_continuous":
    #     return {
    #         "past_hr": (hr_base, 256, "HR"),
    #         "past_lr": ({**MSG_SAT_CONFIG, **NWCSAF_CONFIG}, 64, "LR"),
    #         "past_mr": None,
    #         "label_var": "RZC",
    #         "label_transform": label_transform_rzc_continuous,
    #         "label_suffix": "HR",
    #         "label_type": "radar_continuous",
    #     }
    elif mode == "mtg_radar_continuous":
        hr_with_vis = {**hr_base, **MTG_HR_SAT_CONFIG}
        return {
            "past_hr": (hr_with_vis, 256, "HR"),
            "past_mr": (MTG_MR_SAT_CONFIG, 128, "LR"),
            "past_lr": (NWCSAF_CONFIG, 64, "LR"),
            "label_var": "RZC",
            "label_transform": label_transform_rzc_continuous,
            "label_suffix": "HR",
            "label_type": "radar_continuous",
        }
    # ------------------------------------------------------------------
    # OPERA-driven modes for the NWCSAF Shapley study (4-model coalition).
    # OPERA is always present in MR; MTG IR/WV and NWCSAF are toggled.
    # HR carries only MTG vis_06 (no legacy radar/lightning channels).
    # Label is `opera_rainfall_rate_hr` (HR alias) so the 256×256 head
    # stays compatible with the existing decoder.
    # ------------------------------------------------------------------
    elif mode == "mtg_opera_radar_only":
        return {
            "past_hr": (MTG_HR_SAT_CONFIG, 256, "HR"),
            "past_mr": (OPERA_MR_CONFIG, 128, "LR"),
            "past_lr": None,
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }
    elif mode == "mtg_opera_mtgmr":
        return {
            "past_hr": (MTG_HR_SAT_CONFIG, 256, "HR"),
            "past_mr": ({**OPERA_MR_CONFIG, **MTG_MR_SAT_CONFIG}, 128, "LR"),
            "past_lr": None,
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }
    elif mode == "mtg_opera_nwcsaf":
        return {
            "past_hr": (MTG_HR_SAT_CONFIG, 256, "HR"),
            "past_mr": (OPERA_MR_CONFIG, 128, "LR"),
            "past_lr": (NWCSAF_CONFIG, 64, "LR"),
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }
    elif mode == "mtg_opera_full":
        return {
            "past_hr": (MTG_HR_SAT_CONFIG, 256, "HR"),
            "past_mr": ({**OPERA_MR_CONFIG, **MTG_MR_SAT_CONFIG}, 128, "LR"),
            "past_lr": (NWCSAF_CONFIG, 64, "LR"),
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use: mtg_lightning, mtg_radar, "
            f"mtg_radar_continuous, mtg_opera_radar_only, mtg_opera_mtgmr, "
            f"mtg_opera_nwcsaf, mtg_opera_full. "
            f"(MSG modes are currently disabled.)"
        )


# ============================================================================
# Timestep utilities
# ============================================================================
#
# Dataset layout is driven by `sequence_meta_<source>.json` (written by
# extract_patch_seq_for_datasets.py) which records the chosen step_minutes,
# past_steps, and future_steps. INPUT_COLS / LABEL_COLS / T_OFFSETS /
# N_INPUT / N_LABEL are populated by init_sequence_config(data_root,
# source) - called once from main() before any function below uses
# them - so the DBSCAN-driven and lightning-driven tracks can coexist on
# disk without colliding on a single sequence_meta.json.

PROJECT_ROOT_FOR_SEQ = Path(__file__).resolve().parent

# Placeholders populated by init_sequence_config(). Keeping them at module
# scope so the existing functions (generate_samples, get_output_signature,
# create_and_save_datasets, ...) can keep referring to them by name once
# the caller has initialised the schema.
SEQUENCE_META_PATH: Path | None = None
STEP_MINUTES: int | None = None
PAST_STEPS: int | None = None
FUTURE_STEPS: int | None = None
SEQ_SOURCE: str | None = None
SOURCE_STEP_MINUTES_NATIVE: int | None = None
INPUT_COLS: list[str] | None = None
LABEL_COLS: list[str] | None = None
T_OFFSETS: list[int] | None = None
N_INPUT: int | None = None
N_LABEL: int | None = None


def init_sequence_config(data_root, source: str = "dbscan") -> None:
    """Load `sequence_meta_<source>.json` and populate module globals.

    Must be called exactly once before any function that depends on
    `STEP_MINUTES`, `INPUT_COLS`, `LABEL_COLS`, `T_OFFSETS`, `N_INPUT`,
    `N_LABEL`. The CLI's main() does this immediately after parsing
    `--source`; callers that import functions from this module (e.g.
    evaluate_coalition.py) only use `get_mode_config` /
    `load_and_transform_group` / `load_label` and don't depend on these
    globals, so they can skip init.
    """
    global SEQUENCE_META_PATH, STEP_MINUTES, PAST_STEPS, FUTURE_STEPS
    global SEQ_SOURCE, SOURCE_STEP_MINUTES_NATIVE
    global INPUT_COLS, LABEL_COLS, T_OFFSETS, N_INPUT, N_LABEL

    data_root = Path(data_root)
    SEQUENCE_META_PATH = data_root / f"sequence_meta_{source}.json"
    if not SEQUENCE_META_PATH.exists():
        print(
            f"ERROR: {SEQUENCE_META_PATH} not found.\n"
            f"Run from the project root:\n"
            f"    python validate_timestep.py --step_minutes <N>\n"
            f"    python extract_patch_seq_for_datasets.py --source {source}",
            file=sys.stderr,
        )
        sys.exit(2)
    seq = json.loads(SEQUENCE_META_PATH.read_text())

    STEP_MINUTES = int(seq["step_minutes"])
    PAST_STEPS = int(seq["past_steps"])
    FUTURE_STEPS = int(seq["future_steps"])
    # Source of the activity index used by extract_patch_seq_for_datasets.py.
    SEQ_SOURCE = seq.get("source", source)
    # Native source cadence (step_minutes from timestep_config.json). For
    # the lightning source, STEP_MINUTES above is the aggregation window,
    # while this is the underlying lightning-map cadence.
    SOURCE_STEP_MINUTES_NATIVE = int(
        seq.get("source_step_minutes_native", STEP_MINUTES)
    )
    INPUT_COLS = [seq["step_columns"][i] for i in range(PAST_STEPS + 1)]
    LABEL_COLS = [seq["step_columns"][i]
                  for i in range(PAST_STEPS + 1, len(seq["step_columns"]))]
    T_OFFSETS = [k * STEP_MINUTES
                 for k in range(-PAST_STEPS, FUTURE_STEPS + 1)]
    N_INPUT = len(INPUT_COLS)
    N_LABEL = len(LABEL_COLS)


def reference_to_hhmm(reference_utc_str, offset_minutes):
    """Convert reference_utc 'HH:MM' + offset to 'HHMM' string.
    Handles day wrapping (e.g., 23:45 + 30 → 0015).
    """
    parts = reference_utc_str.strip().split(":")
    ref = datetime(2000, 1, 1, int(parts[0]), int(parts[1]))
    target = ref + timedelta(minutes=offset_minutes)
    return target.strftime("%H%M")


def row_to_hhmm_list(row):
    """Return list of 6 HHMM strings for all timesteps in a sample row."""
    ref = row["reference_utc"]
    return [reference_to_hhmm(ref, offset) for offset in T_OFFSETS]


# ============================================================================
# Patch loading
# ============================================================================

def load_npy(patches_dir, date_str, var_name, hhmm, suffix):
    """Load a .npy patch file. Returns array of shape (N_patches, H, W)."""
    fn = f"{var_name}_{hhmm}_{suffix}.npy"
    path = os.path.join(patches_dir, date_str, fn)
    if not os.path.isfile(path):
        return None
    return np.load(path)


def load_and_transform_group(patches_dir, date_str, hhmm, suffix,
                             var_config, patch_idx, resolution):
    """Load all variables in a group for one timestep, apply transforms,
    extract the patch at `patch_idx`, and concatenate along channel axis.

    Returns: np.ndarray of shape (resolution, resolution, total_channels), float32.
             Returns None if any critical variable is missing.
    """
    channels = []
    for var_name, (transform_fn, extra_ch) in var_config.items():
        data = load_npy(patches_dir, date_str, var_name, hhmm, suffix)
        if data is None:
            # Variable missing — fill with zeros
            if extra_ch is not None:
                channels.append(np.zeros((resolution, resolution, extra_ch),
                                         dtype=np.float32))
            else:
                channels.append(np.zeros((resolution, resolution, 1),
                                         dtype=np.float32))
            continue

        # Extract the specific patch
        if patch_idx >= data.shape[0]:
            # Index out of range — fill with zeros
            if extra_ch is not None:
                channels.append(np.zeros((resolution, resolution, extra_ch),
                                         dtype=np.float32))
            else:
                channels.append(np.zeros((resolution, resolution, 1),
                                         dtype=np.float32))
            continue

        patch = data[patch_idx].astype(np.float32)

        # Apply transform
        transformed = transform_fn(patch)

        # Ensure (H, W, C) shape
        if transformed.ndim == 2:
            transformed = transformed[:, :, np.newaxis]

        channels.append(transformed.astype(np.float32))

    return np.concatenate(channels, axis=-1)


def load_label(patches_dir, date_str, hhmm, label_var, label_transform,
               label_suffix, patch_idx, n_label_channels=1):
    """Load and transform the label variable for one timestep.

    Returns: np.ndarray of shape (256, 256, n_label_channels), float32.
    """
    data = load_npy(patches_dir, date_str, label_var, hhmm, label_suffix)
    if data is None or patch_idx >= data.shape[0]:
        return np.zeros((256, 256, n_label_channels), dtype=np.float32)

    patch = data[patch_idx].astype(np.float32)
    transformed = label_transform(patch)

    if transformed.ndim == 2:
        transformed = transformed[:, :, np.newaxis]

    return transformed.astype(np.float32)


# ============================================================================
# Sample generation
# ============================================================================

def generate_samples(csv_path, patches_dir, mode_config):
    """Generator that yields one (inputs_dict, label) per qualifying patch.

    Yields:
        inputs_dict: dict with keys like "past_hr", "past_lr", ("past_mr")
                     each value is np.ndarray (T, H, W, C)
        label:       np.ndarray (T_future, 256, 256, C_label)
                     C_label=1 for lightning, C_label=5 for radar
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    label_var = mode_config["label_var"]
    label_transform = mode_config["label_transform"]
    label_suffix = mode_config["label_suffix"]
    label_type = mode_config["label_type"]
    n_label_channels = LABEL_CHANNELS[label_type]

    # Determine input groups (past_hr, past_lr, optionally past_mr)
    input_groups = {}
    for key in ["past_hr", "past_lr", "past_mr"]:
        cfg = mode_config.get(key)
        if cfg is not None:
            input_groups[key] = cfg  # (var_config, resolution, suffix)

    n_skipped = 0
    n_yielded = 0

    for row_idx, row in df.iterrows():
        date_str = row["date"]
        hhmm_list = row_to_hhmm_list(row)

        # Parse patch_numbers and index lists
        patch_numbers = ast.literal_eval(row["patch_numbers"])
        idx_cols = INPUT_COLS + LABEL_COLS
        idx_lists = {}
        for col in idx_cols:
            idx_lists[col] = ast.literal_eval(row[col])

        n_patches = len(patch_numbers)

        # Each qualifying patch becomes one sample
        for p_pos in range(n_patches):
            sample_valid = True

            # --- Build inputs (first 3 timesteps) ---
            input_tensors = {key: [] for key in input_groups}

            for t_idx in range(N_INPUT):
                col = INPUT_COLS[t_idx]
                hhmm = hhmm_list[t_idx]
                npy_idx = idx_lists[col][p_pos]

                for group_key, (var_config, resolution, suffix) in input_groups.items():
                    tensor_slice = load_and_transform_group(
                        patches_dir, date_str, hhmm, suffix,
                        var_config, npy_idx, resolution
                    )
                    if tensor_slice is None:
                        sample_valid = False
                        break
                    input_tensors[group_key].append(tensor_slice)

                if not sample_valid:
                    break

            if not sample_valid:
                n_skipped += 1
                continue

            # --- Build labels (last 3 timesteps) ---
            label_frames = []
            for t_idx in range(N_LABEL):
                col = LABEL_COLS[t_idx]
                hhmm = hhmm_list[N_INPUT + t_idx]
                npy_idx = idx_lists[col][p_pos]

                lbl = load_label(
                    patches_dir, date_str, hhmm,
                    label_var, label_transform, label_suffix, npy_idx,
                    n_label_channels=n_label_channels
                )
                label_frames.append(lbl)

            # Stack along time axis: (T, H, W, C)
            stacked_inputs = {}
            for key in input_tensors:
                stacked_inputs[key] = np.stack(input_tensors[key], axis=0)

            stacked_label = np.stack(label_frames, axis=0)

            n_yielded += 1
            yield stacked_inputs, stacked_label

    print(f"  Generated {n_yielded} samples, skipped {n_skipped}")


# ============================================================================
# TF Dataset creation and saving
# ============================================================================

def get_output_signature(mode_config):
    """Build the tf output signature for the generator."""
    input_specs = {}
    for key in ["past_hr", "past_lr", "past_mr"]:
        cfg = mode_config.get(key)
        if cfg is None:
            continue
        var_config, resolution, _ = cfg
        # Count channels
        n_channels = 0
        for var_name, (tfn, extra_ch) in var_config.items():
            n_channels += (extra_ch if extra_ch is not None else 1)

        input_specs[key] = tf.TensorSpec(
            shape=(N_INPUT, resolution, resolution, n_channels),
            dtype=tf.float32
        )

    label_type = mode_config["label_type"]
    n_label_channels = LABEL_CHANNELS[label_type]
    label_spec = tf.TensorSpec(
        shape=(N_LABEL, 256, 256, n_label_channels),
        dtype=tf.float32
    )

    return (input_specs, label_spec)


def build_dataset(csv_path, patches_dir, mode_config):
    """Build a tf.data.Dataset from a CSV and patch files."""
    sig = get_output_signature(mode_config)
    ds = tf.data.Dataset.from_generator(
        lambda: generate_samples(csv_path, patches_dir, mode_config),
        output_signature=sig
    )
    return ds


# ============================================================================
# TFRecord I/O
# ============================================================================
#
# `tf.data.Dataset.save / load` builds a single monolithic snapshot. Every
# row of the eventual shuffle buffer is materialised in host RAM, so on
# datasets larger than ~10 GB total it can OOM the writer (at save time)
# and the reader (at training time, when filling the shuffle buffer).
# TFRecord shards stream cleanly in both directions: the writer flushes
# one sample at a time, the reader pulls records on demand and the
# training-time shuffle buffer can be a fraction of the dataset.

# How many samples we pack into a single .tfrecord shard.
# - Too few -> too many small files, file-system overhead dominates.
# - Too many -> a single shard barely fits in shuffle buffer.
# 500 hits a sweet spot for our 5-MB samples: ~2.5 GB per shard.
TFRECORD_SAMPLES_PER_SHARD = 500


def _serialize_sample(stacked_inputs, stacked_label):
    """Serialize one (inputs_dict, label) sample as a tf.train.Example."""
    feature: dict[str, tf.train.Feature] = {}
    for key, tensor in stacked_inputs.items():
        # tf.io.serialize_tensor preserves dtype + shape inside the
        # serialised bytes blob, so the parse side doesn't need to know
        # them ahead of time.
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(
                    tf.convert_to_tensor(tensor)
                ).numpy()],
            )
        )
    feature["label"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(
                tf.convert_to_tensor(stacked_label)
            ).numpy()],
        )
    )
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord_shards(csv_path, patches_dir, mode_config,
                          out_dir: Path,
                          samples_per_shard: int = TFRECORD_SAMPLES_PER_SHARD,
                          ) -> tuple[int, int]:
    """Drive `generate_samples` and write samples into per-shard
    `.tfrecord` files under `out_dir`. Returns (n_samples, n_shards).

    Memory cost is one serialised sample at a time — the writer never
    holds the full dataset in RAM, in contrast to tf.data.Dataset.save.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    shard_idx = 0
    writer: tf.io.TFRecordWriter | None = None

    def _shard_path(i: int) -> str:
        return str(out_dir / f"shard_{i:05d}.tfrecord")

    try:
        for stacked_inputs, stacked_label in generate_samples(
                csv_path, patches_dir, mode_config):
            if writer is None or n_samples % samples_per_shard == 0:
                if writer is not None:
                    writer.close()
                    shard_idx += 1
                writer = tf.io.TFRecordWriter(_shard_path(shard_idx))
                if n_samples == 0:
                    print(f"    -> shard 0: {_shard_path(0)}")
            writer.write(_serialize_sample(stacked_inputs, stacked_label))
            n_samples += 1
            if n_samples % 100 == 0:
                print(f"    written {n_samples:,} samples "
                      f"(shard {shard_idx})", flush=True)
    finally:
        if writer is not None:
            writer.close()

    n_shards = shard_idx + 1 if n_samples > 0 else 0
    return n_samples, n_shards


def _make_parse_fn(input_specs, label_spec):
    """Build the parse function that reverses `_serialize_sample`.

    Returns a callable mapping (serialised_example) -> (inputs_dict, label),
    with shapes set so downstream `model.fit` sees the same tensor shapes
    as the old tf.data.Dataset.save path.
    """
    feature_description = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in input_specs
    }
    feature_description["label"] = tf.io.FixedLenFeature([], tf.string)

    def parse(serialised):
        parsed = tf.io.parse_single_example(serialised, feature_description)
        inputs = {}
        for key, spec in input_specs.items():
            t = tf.io.parse_tensor(parsed[key], out_type=spec.dtype)
            t.set_shape(spec.shape)
            inputs[key] = t
        label = tf.io.parse_tensor(parsed["label"], out_type=label_spec.dtype)
        label.set_shape(label_spec.shape)
        return inputs, label

    return parse


def load_tfrecord_dataset(shard_dir: Path,
                           mode_config: dict) -> tf.data.Dataset:
    """Load a split's TFRecord shards into a tf.data.Dataset whose
    output signature matches the one used during write."""
    shard_paths = sorted(str(p) for p in shard_dir.glob("shard_*.tfrecord"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard_*.tfrecord files in {shard_dir}. "
            f"Re-run create_datasets.py for this mode."
        )
    input_specs, label_spec = get_output_signature(mode_config)
    parse_fn = _make_parse_fn(input_specs, label_spec)
    files_ds = tf.data.Dataset.from_tensor_slices(shard_paths)
    ds = files_ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)


def create_and_save_datasets(data_root, mode, source="dbscan", output_root=None):
    """Create and save train, validation, and test datasets.

    Args:
        data_root: path to our_data/ directory containing CSVs and patches/
        mode: one of mtg_lightning, mtg_radar, mtg_radar_continuous,
            mtg_opera_radar_only, mtg_opera_mtgmr
        source: which extract_patch_seq source the sample CSVs came from
            ('dbscan' = patch_index.csv from identify_patches, or
            'lightning' = lightning_patches.csv from
            identify_lightning_periods). The dataset directory is suffixed
            by source so the two tracks coexist on disk.
        output_root: where to save datasets (default: data_root/datasets/)
    """
    data_root = Path(data_root)
    patches_dir = data_root / "patches"

    if output_root is None:
        output_root = data_root / "datasets"
    else:
        output_root = Path(output_root)

    # Point the lazy stats loader at this run's data_root before any
    # transform fires; required because the transforms can be called
    # from worker threads later.
    set_normalization_stats_path(data_root / "normalization_stats.json")

    mode_config = get_mode_config(mode)
    # Suffix the dataset dir with the source so radar- and lightning-
    # driven runs don't clobber each other (the domain-adaptation
    # pipeline trains both and uses them as separate feature extractors).
    save_dir = output_root / f"{mode}_{source}"

    # Print configuration summary
    print("=" * 70)
    print(f"COALITION-4 Dataset Creation - Mode: {mode}  Source: {source}")
    print("=" * 70)
    print(f"Data root:    {data_root}")
    print(f"Patches dir:  {patches_dir}")
    print(f"Stats file:   {data_root / 'normalization_stats.json'}")
    print(f"Output dir:   {save_dir}")
    print(f"Label:        {mode_config['label_var']}")
    print()

    # Print channel counts per group
    sig = get_output_signature(mode_config)
    for key, spec in sig[0].items():
        print(f"  {key}: shape={spec.shape}, dtype={spec.dtype}")
    print(f"  label: shape={sig[1].shape}, dtype={sig[1].dtype}")
    print()

    # Process each split. CSVs are suffixed by source to match
    # extract_patch_seq_for_datasets.py's output convention.
    splits = {
        "train":      f"train_data_{source}.csv",
        "validation": f"validation_data_{source}.csv",
        "test":       f"test_data_{source}.csv",
    }

    for split_name, csv_name in splits.items():
        csv_path = data_root / csv_name
        if not csv_path.is_file():
            print(f"WARNING: {csv_path} not found — skipping {split_name}")
            continue

        print(f"\n--- Processing {split_name} ({csv_name}) ---")

        split_dir = save_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Write TFRecord shards directly from the generator. One sample
        # at a time, so peak RAM is one serialised sample (~5 MB) +
        # whatever the generator's per-row state needs — not the full
        # dataset like the old tf.data.Dataset.save path required.
        print(f"  Writing TFRecord shards to {split_dir} ...")
        n_samples, n_shards = write_tfrecord_shards(
            str(csv_path), str(patches_dir), mode_config, split_dir,
        )
        print(f"  Wrote {n_samples:,} samples across {n_shards} shard(s)")

        # Sidecar metadata. `format` lets the loader know which on-disk
        # layout to expect; the eventual reader path is
        # `load_tfrecord_dataset(split_dir, mode_config)`.
        meta = {
            "mode": mode,
            "source": source,
            "split": split_name,
            "csv": csv_name,
            "format": "tfrecord",
            "n_samples": n_samples,
            "n_shards": n_shards,
            "label_var": mode_config["label_var"],
            "label_type": mode_config["label_type"],
            "label_channels": LABEL_CHANNELS[mode_config["label_type"]],
            "input_shapes": {k: list(v.shape) for k, v in sig[0].items()},
            "label_shape": list(sig[1].shape),
            "step_minutes": STEP_MINUTES,
            "source_step_minutes_native": SOURCE_STEP_MINUTES_NATIVE,
            "sequence_source": SEQ_SOURCE,
            "past_steps": PAST_STEPS,
            "future_steps": FUTURE_STEPS,
            "input_cols": INPUT_COLS,
            "label_cols": LABEL_COLS,
        }
        meta_path = split_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved {split_name} dataset + metadata")

    print("\n" + "=" * 70)
    print("Dataset creation complete.")
    print(f"Saved to: {save_dir}")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create COALITION-4 TF datasets from pre-extracted patches."
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["mtg_lightning", "mtg_radar", "mtg_radar_continuous",
                 "mtg_opera_radar_only", "mtg_opera_mtgmr"],
        help="Dataset mode (MSG modes are disabled in this build)."
    )
    parser.add_argument(
        "--source", type=str, default="dbscan",
        choices=["dbscan", "lightning"],
        help="Which extract_patch_seq source the sample CSVs came from. "
             "Selects which sequence_meta_<source>.json and "
             "{train,validation,test}_data_<source>.csv are read, and "
             "lands the output dataset at datasets/<mode>_<source>/. "
             "'dbscan' (default) = patch_index.csv from identify_patches "
             "(either --source radar/RZC or --source opera at that step). "
             "'lightning' = lightning_patches.csv from "
             "identify_lightning_periods.",
    )
    parser.add_argument(
        "--data_root", type=str, default="./our_data",
        help="Root directory containing CSVs and patches/ subfolder"
    )
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Output root for saved datasets (default: data_root/datasets/)"
    )
    args = parser.parse_args()

    # Populate the module-level schema constants from the per-source
    # sequence metadata before any sample-generating function is called.
    init_sequence_config(args.data_root, args.source)

    create_and_save_datasets(
        data_root=args.data_root,
        mode=args.mode,
        source=args.source,
        output_root=args.output_root
    )


if __name__ == "__main__":
    main()
    