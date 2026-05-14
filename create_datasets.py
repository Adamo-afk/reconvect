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

    # MSG modes are disabled in this build — see _MSG_DISABLED block in
    # pipeline_msg_mtg.py for context. Re-enable both at once if needed.
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
# Dataset layout is driven by sequence_meta.json (written by
# extract_patch_seq_for_datasets.py) which records the chosen step_minutes,
# past_steps, and future_steps. INPUT_COLS / LABEL_COLS / T_OFFSETS / N_INPUT
# / N_LABEL are populated at module load time from that file so all loaders
# see the same schema regardless of cadence.

PROJECT_ROOT_FOR_SEQ = Path(__file__).resolve().parent
SEQUENCE_META_PATH = PROJECT_ROOT_FOR_SEQ / "our_data" / "sequence_meta.json"


def _load_sequence_meta():
    if not SEQUENCE_META_PATH.exists():
        print(
            f"ERROR: sequence_meta.json not found at {SEQUENCE_META_PATH}.\n"
            f"Run from the project root:\n"
            f"    python validate_timestep.py --step_minutes <N>\n"
            f"    python extract_patch_seq_for_datasets.py",
            file=sys.stderr,
        )
        sys.exit(2)
    return json.loads(SEQUENCE_META_PATH.read_text())


_SEQ = _load_sequence_meta()
STEP_MINUTES = int(_SEQ["step_minutes"])
PAST_STEPS = int(_SEQ["past_steps"])
FUTURE_STEPS = int(_SEQ["future_steps"])
# Source of the activity index used by extract_patch_seq_for_datasets.py:
# 'radar' (default, patch_index.csv) or 'lightning' (lightning_patches.csv).
SEQ_SOURCE = _SEQ.get("source", "radar")
# Native source cadence (step_minutes from timestep_config.json). For the
# lightning source, STEP_MINUTES above is the aggregation window, while this
# is the underlying lightning-map cadence.
SOURCE_STEP_MINUTES_NATIVE = int(
    _SEQ.get("source_step_minutes_native", STEP_MINUTES)
)

# Input columns are the past + current step indices; labels are the future ones.
INPUT_COLS = [_SEQ["step_columns"][i] for i in range(PAST_STEPS + 1)]
LABEL_COLS = [_SEQ["step_columns"][i] for i in range(PAST_STEPS + 1, len(_SEQ["step_columns"]))]
# Minute offsets relative to reference_utc (T) — derived from step indices.
T_OFFSETS = [k * STEP_MINUTES for k in range(-PAST_STEPS, FUTURE_STEPS + 1)]
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


def create_and_save_datasets(data_root, mode, output_root=None):
    """Create and save train, validation, and test datasets.

    Args:
        data_root: path to our_data/ directory containing CSVs and patches/
        mode: one of msg_lightning, msg_radar, mtg_lightning, mtg_radar
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
    save_dir = output_root / mode

    # Print configuration summary
    print("=" * 70)
    print(f"COALITION-4 Dataset Creation — Mode: {mode}")
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

    # Process each split
    splits = {
        "train":      "train_data.csv",
        "validation": "validation_data.csv",
        "test":       "test_data.csv",
    }

    for split_name, csv_name in splits.items():
        csv_path = data_root / csv_name
        if not csv_path.is_file():
            print(f"WARNING: {csv_path} not found — skipping {split_name}")
            continue

        print(f"\n--- Processing {split_name} ({csv_name}) ---")

        ds = build_dataset(str(csv_path), str(patches_dir), mode_config)

        # Materialize to count samples and save
        split_dir = save_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        print(f"  Saving to {split_dir} ...")
        tf.data.Dataset.save(ds, str(split_dir))

        # Also save a metadata file with shapes and mode info
        meta = {
            "mode": mode,
            "split": split_name,
            "csv": csv_name,
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
                 "mtg_opera_radar_only", "mtg_opera_mtgmr",
                 "mtg_opera_nwcsaf", "mtg_opera_full"],
        help="Dataset mode (MSG modes are disabled in this build)."
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

    create_and_save_datasets(
        data_root=args.data_root,
        mode=args.mode,
        output_root=args.output_root
    )


if __name__ == "__main__":
    main()
    