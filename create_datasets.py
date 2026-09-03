"""
create_datasets.py — COALITION-4 Romanian Adaptation
=====================================================
Creates train, validation, and test TF datasets from pre-extracted .npy patches.

Every mode reads OPERA in MR and MTG vis_06 in HR; the lightning modes
add the three LINET channels to HR. Mode names state their own track:
`_rainfall` = OPERA rainfall 5-class, `_logz` = OPERA rainfall in
log_zscore space, `_occurrence` = lightning binary. See BUILDABLE_MODES.

Usage:
    # Rainfall track (5-class)
    python create_datasets.py --mode mtg_opera_radar_only_rainfall  --data_root ./our_data
    python create_datasets.py --mode mtg_opera_mtgmr_rainfall       --data_root ./our_data
    python create_datasets.py --mode mtg_lightning_opera_rainfall   --data_root ./our_data

    # Lightning track (binary occurrence)
    python create_datasets.py --mode mtg_lightning_opera_occurrence --data_root ./our_data

The training cadence is read from our_data/timestep_config.json (set via
validate_timestep.py) and the per-sample window from
our_data/sequence_meta_dbscan.json (written by
extract_patch_seq_for_datasets.py).

Inputs (<source> is always `dbscan` — see pipeline_config.SOURCE):
    - train_data_<source>.csv, validation_data_<source>.csv,
      test_data_<source>.csv in data_root/
    - sequence_meta_<source>.json in data_root/
      (step_minutes, past_steps, future_steps)
    - normalization_stats_<source>.json in data_root/
    - .npy patch files in data_root/patches/{date}/{variable}_{HHMM}_{HR|MR}.npy

Outputs (per --mode):
    - TFRecord shards in data_root/datasets/{mode}_{source}/train/
    - TFRecord shards in data_root/datasets/{mode}_{source}/validation/
    - TFRecord shards in data_root/datasets/{mode}_{source}/test/
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

from pipeline_config import (
    SOURCE,
    add_root_arguments,
    resolve_data_root,
    resolve_datasets_root,
)

# Sentinel: `stats_period=None` legitimately means "global statistics", so
# None cannot double as "not supplied".
_UNSET = object()

from periods import (
    Period,
    load_seasons,
    normalization_stats_name,
    sequence_meta_name,
    split_csv_name,
)
from compress_datasets import (
    DEFAULT_LEVEL as _ARCHIVE_LEVEL,
    DEFAULT_MAX_CONCURRENT as _ARCHIVE_MAX_CONCURRENT,
    array_exists,
    default_workers as _default_archive_workers,
    load_array,
)
from ensemble_plan import (
    append_state,
    enumerate_members,
    format_plan,
    load_index_dates,
    registry_path,
    require_last_state,
    state_member,
    state_period,
)

_ARCHIVE_WORKERS = _default_archive_workers()


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
# Variables that are categorical/binary (occurrence) are NOT driven by
# the JSON. Their transforms are hardcoded because the scaling is a
# property of the physical units, not the training distribution.

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
    """Solar visible channel (vis_06): x / 100.

    Hardcoded unit conversion: the source data is
    stored as integer reflectance % * 100, so `/100` recovers a value
    already in a sensible model-friendly range. No z-score needed.
    """
    x = np.where(np.isnan(x), 0.0, x)
    return x / 100.0



def transform_ir38(x):
    """MTG ir_38: linear z-score."""
    return _apply_linear_zscore(x, "ir_38")



def transform_ir105(x):
    """MTG ir_105 thermal channel: linear z-score."""
    return _apply_linear_zscore(x, "ir_105")



def transform_wv63(x):
    """MTG wv_63 water-vapour channel: linear z-score."""
    return _apply_linear_zscore(x, "wv_63")



def transform_wv73(x):
    """MTG wv_73 water-vapour channel: linear z-score."""
    return _apply_linear_zscore(x, "wv_73")


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

    Heavy-tailed and zero-inflated; clip-then-log
    flattens the distribution before z-scoring.
    """
    return _apply_log_zscore(x, "opera_rainfall_rate")


# Label transforms
def label_transform_occurrence(x):
    """Binary lightning occurrence label"""
    x = np.where(np.isnan(x), 0.0, x)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# Shared 5-class rain-rate bin edges (mm/h). The RECONVECT label and the
# SepConv baseline's predictions are both binned at these, so the two
# models cannot be told apart by their thresholds.
RAINFALL_CLASS_EDGES = (10.0, 20.0, 30.0, 40.0)


def label_transform_opera_rainfall_multiclass(x):
    """OPERA instantaneous rain rate → 5-class one-hot label.

    Classes (mm/h):  0: R<10, 1: 10≤R<20, 2: 20≤R<30, 3: 30≤R<40, 4: R≥40.
    The label patch is loaded from `opera_rainfall_rate_hr` (HR alias of
    the same reprojected file) so the output shape matches the 256×256
    HR head. Returns (H, W, 5) float32.
    """
    x = np.where(np.isnan(x), 0.0, x)
    x = np.clip(x, 0.0, None)
    h, w = x.shape
    e0, e1, e2, e3 = RAINFALL_CLASS_EDGES
    one_hot = np.zeros((h, w, 5), dtype=np.float32)
    one_hot[:, :, 0] = (x < e0).astype(np.float32)
    one_hot[:, :, 1] = ((x >= e0) & (x < e1)).astype(np.float32)
    one_hot[:, :, 2] = ((x >= e1) & (x < e2)).astype(np.float32)
    one_hot[:, :, 3] = ((x >= e2) & (x < e3)).astype(np.float32)
    one_hot[:, :, 4] = (x >= e3).astype(np.float32)
    return one_hot


def label_transform_opera_rainfall_logz(x):
    """OPERA instantaneous rain rate → log_zscore target (SepConv baseline).

    Applies exactly the same spec the field gets as an *input* channel —
    fill 0.01, clip 0.01, log10, then the global train-split mean/std.
    That symmetry is the point: the SepConv baseline predicts the field
    in the space it consumes it, so a composed rollout can feed its own
    output back in without a change of units.

    Deliberately not a bounded linear /max scaling: that kind sits in
    [0, 1] and is linear in mm/h, which crushes the 10-70 mm/h band this
    comparison is about into the top 14% of its range.

    Inverting for physical-space thresholding is `10 ** (z * std + mean)`
    — see `logz_to_mmh`. Never threshold in z-space.

    Returns (H, W) float32.
    """
    return _apply_log_zscore(x, "opera_rainfall_rate").astype(np.float32)


def logz_to_mmh(z):
    """Inverse of `label_transform_opera_rainfall_logz`: z → mm/h.

    Calibration and verification both happen in physical space, so this
    is the only sanctioned way back out of a SepConv prediction.
    """
    spec = _norm("opera_rainfall_rate")
    return np.power(10.0, np.asarray(z) * spec["std"] + spec["mean"])


def mmh_to_logz(mmh):
    """Forward map, for putting class edges into z-space (reporting only)."""
    spec = _norm("opera_rainfall_rate")
    clipped = np.clip(np.asarray(mmh, dtype=np.float64), 0.01, None)
    return (np.log10(clipped) - spec["mean"]) / spec["std"]


# Number of label channels per target type
LABEL_CHANNELS = {
    "lightning": 1,           # binary occurrence
    "radar": 5,               # 5-class precipitation
    "radar_logz": 1,          # rain rate in log_zscore space (SepConv)
}


# ============================================================================
# Variable configuration per mode
# ============================================================================

# Variable name → (transform_function, produces_extra_channels)
# produces_extra_channels: None for scalar transforms, int for one-hot etc.

HR_LIGHTNING_CONFIG = {
    "density":    (transform_lightning_density, None),
    "current":    (transform_lightning_current, None),
    "occurrence": (transform_occurrence, None),
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

# ============================================================================
# Resolution tiers
# ============================================================================
# Two input tiers survive. The names are historical — see the note below.
#
#   past_hr  "high resolution"    1 km native, 256x256 px patch, no pooling
#            Channels: MTG vis_06; LINET density / current / occurrence
#            (LINET is rasterised straight onto the 1 km Romania grid).
#
#   past_mr  "medium resolution"  2 km native, 128x128 px patch, 2x2 pooling
#            Channels: OPERA reflectivity + rainfall_rate;
#                      MTG ir_38 / ir_105 / wv_63 / wv_73.
#
# WHY "MEDIUM" WHEN IT IS NOW THE COARSEST TIER:
# There used to be a third tier, `past_lr` ("low resolution", 3 km native,
# 64x64 patch, 4x4 pooling), carrying MSG SEVIRI and NWCSAF. Both products
# were retired, so that tier is gone and MR is now the coarsest input. The
# name is kept because it is baked into the input-tensor names of every
# trained checkpoint and into every dataset's metadata.json — renaming it
# would invalidate existing models.
#
# Extracted patches are named `{variable}_{HHMM}_{HR|MR}.npy` — two
# suffixes, matching the two tiers exactly. The mode config carries the
# suffix explicitly as the third element of each group tuple, so nothing
# infers it from the resolution.

# Iteration order of the input groups. This order sets the model's input
# ordering, so it must stay stable across dataset builds and checkpoints.
INPUT_GROUP_KEYS = ("past_hr", "past_mr")


# ============================================================================
# Mode registry
# ============================================================================
# Single source of truth for which modes exist. Every `--mode` CLI in the
# project builds its `choices=` from these tuples rather than hardcoding
# its own copy, so adding a mode means editing `get_mode_config` below
# plus one entry here.
#
# Naming: the mode name states its own track — `_rainfall` = OPERA
# rainfall 5-class, `_logz` = OPERA rainfall in log_zscore space,
# `_occurrence` = lightning binary. train_models.build_run_tag appends
# `_<source>` to get the on-disk artefact tag.

# Modes create_datasets.py can build a dataset for.
BUILDABLE_MODES = (
    "mtg_opera_radar_only_rainfall",
    "mtg_opera_mtgmr_rainfall",
    "mtg_lightning_opera_rainfall",
    "mtg_lightning_opera_occurrence",
    # Baseline comparison pair — radar-only, see get_mode_config.
    # `opera_sepconv_logz` needs a past=4/future=8 sequence set;
    # `opera_radar_only_rainfall` uses the standard RECONVECT window.
    "opera_sepconv_logz",
    "opera_radar_only_rainfall",
)

# The KD student has no dataset of its own — train_lightning_kd.py feeds
# it the teacher's dataset with past_hr sliced to the student's channels.
# It is a valid --mode for evaluate / predict / visualize / validate.
KD_STUDENT_MODES = (
    "mtg_opera_occurrence",
)

# Everything get_mode_config understands.
MODE_NAMES = BUILDABLE_MODES + KD_STUDENT_MODES


def get_mode_config(mode):
    """Return input group configurations and label config for a given mode.

    Returns:
        dict with keys for each input tensor group:
            "past_hr":  (var_config_dict, resolution, suffix)
            "past_mr":  (var_config_dict, resolution, suffix)
        and:
            "label_var": str — variable name for labels
            "label_transform": callable
            "label_suffix": str — HR or MR
    """
    # HR (256 px) always carries MTG vis_06; the lightning modes add the
    # three LINET channels alongside it. MR (128 px, 2 km native, 2x pool)
    # always carries OPERA, optionally joined by MTG IR/WV.
    hr_vis = MTG_HR_SAT_CONFIG
    hr_lightning_vis = {**HR_LIGHTNING_CONFIG, **MTG_HR_SAT_CONFIG}
    mr_opera = OPERA_MR_CONFIG
    mr_opera_mtg = {**OPERA_MR_CONFIG, **MTG_MR_SAT_CONFIG}

    # --- Baseline comparison modes -------------------------------------
    # Both are radar-only: no MTG, no LINET. Modality enrichment is the
    # thing RECONVECT is being credited for, so the baseline and the
    # ablation must not receive it.
    if mode == "opera_sepconv_logz":
        # SepConv-ens target. The rainfall field is BOTH the input and the
        # label, in the same log_zscore space, so a composed rollout can
        # feed its own output straight back in. Carried in the HR group at
        # 256 px because the label is HR — an autoregressive step must not
        # change resolution between output and input.
        return {
            "past_hr": ({"opera_rainfall_rate_hr":
                         (transform_opera_rainfall_rate, None)}, 256, "HR"),
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_logz,
            "label_suffix": "HR",
            "label_type": "radar_logz",
        }
    elif mode == "opera_radar_only_rainfall":
        # RECONVECT architecture on the baseline's inputs: OPERA
        # rainfall_rate alone, same 5-class head, same training.
        #
        # Input parity with SepConv-ens is what this mode is for. Both
        # then see one field, so a gap between them is attributable to
        # the architecture rather than to a channel one of them was
        # handed. Reflectivity is deliberately absent: including it would
        # confound the comparison, and the modality question - what MTG
        # and lightning add - is answered against the full model by
        # `mtg_opera_radar_only_rainfall`, which keeps both OPERA fields.
        #
        # Dropping reflectivity also closes a coverage trap: the manifest
        # represents OPERA by `opera_rainfall_rate`, so a timestep
        # carrying rainfall but not reflectivity passes the gate and
        # would then be dropped at build time. Rainfall-only makes the
        # gate and the mode agree.
        #
        # Carried in the HR group at 256 px, exactly as the baseline
        # carries it, for two reasons that turn out to be the same
        # reason. First, parity has to mean the same TENSOR, not just
        # the same field: the MR form is 2x2 average-pooled, and while
        # that is near-lossless over the domain as a whole (97% of it is
        # dry), inside wet blocks 79% are non-constant and 4.3% of their
        # pixels change class - because the 2 km OPERA cells do not
        # align with the 1 km grid's 2x2 boundaries. Handing the
        # baseline that detail and not the ablation would put part of
        # any gap down to input rather than architecture.
        #
        # Second, the model's output resolution is its FINEST INPUT
        # (encoder halves three times, decoder doubles three times), and
        # this is the only mode with no HR input to hold it at 256. With
        # the MR form the head emitted 128x128 against a 256x256 label
        # and training died on the shape mismatch. At HR the shapes line
        # up with no model change, and the task stays pure temporal
        # extrapolation rather than forecasting plus a learned upsample
        # the baseline never performs.
        #
        # Note both sides are 2 km information on a 1 km lattice: OPERA
        # is 2 km native and reproject.py nearest-neighbours it onto the
        # grid, so 98% of adjacent label pixels are identical copies.
        # The 256 is the lattice the HR branch defines, not a claim of
        # 1 km rainfall.
        return {
            "past_hr": ({"opera_rainfall_rate_hr":
                         (transform_opera_rainfall_rate, None)}, 256, "HR"),
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }

    # --- Rainfall track: OPERA rainfall_rate 5-class -------------------
    if mode == "mtg_opera_radar_only_rainfall":
        # Lightest baseline: MTG vis_06 + OPERA. No MTG IR/WV, no lightning.
        return {
            "past_hr": (hr_vis, 256, "HR"),
            "past_mr": (mr_opera, 128, "MR"),
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }
    elif mode == "mtg_opera_mtgmr_rainfall":
        # Baseline + MTG IR/WV in MR.
        return {
            "past_hr": (hr_vis, 256, "HR"),
            "past_mr": (mr_opera_mtg, 128, "MR"),
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }
    elif mode == "mtg_lightning_opera_rainfall":
        # Heaviest input stack: LINET in HR alongside MTG vis_06, plus
        # OPERA + MTG IR/WV in MR. Pair with
        # `mtg_lightning_opera_occurrence` (same inputs, lightning label)
        # for the dual-target experiment.
        return {
            "past_hr": (hr_lightning_vis, 256, "HR"),
            "past_mr": (mr_opera_mtg, 128, "MR"),
            "label_var": "opera_rainfall_rate_hr",
            "label_transform": label_transform_opera_rainfall_multiclass,
            "label_suffix": "HR",
            "label_type": "radar",
        }

    # --- Lightning track: binary occurrence ---------------------------
    elif mode == "mtg_lightning_opera_occurrence":
        # Same input stack as `mtg_lightning_opera_rainfall`; the label
        # head predicts binary lightning occurrence at T+future_steps
        # instead of OPERA rainfall. Loss switches to WeightedFocalLoss
        # (label_type == 'lightning'), whose prior reads
        # lightning_fraction_<source>.json at training time.
        return {
            "past_hr": (hr_lightning_vis, 256, "HR"),
            "past_mr": (mr_opera_mtg, 128, "MR"),
            "label_var": "occurrence",
            "label_transform": label_transform_occurrence,
            "label_suffix": "HR",
            "label_type": "lightning",
        }
    elif mode == "mtg_opera_occurrence":
        # KNOWLEDGE-DISTILLATION STUDENT (see train_lightning_kd.py).
        # Same MR/label stack as the teacher
        # `mtg_lightning_opera_occurrence`, so predictions from both land
        # on the exact same 768x1536 binary-occurrence canvases and can be
        # diffed directly; the ONLY difference is the HR branch, which
        # loses LINET here. Rationale: the student produces a lightning
        # prognosis from satellite + OPERA alone, useful whenever the
        # LINET feed is late, missing, or being validated.
        return {
            "past_hr": (hr_vis, 256, "HR"),
            "past_mr": (mr_opera_mtg, 128, "MR"),
            "label_var": "occurrence",
            "label_transform": label_transform_occurrence,
            "label_suffix": "HR",
            "label_type": "lightning",
        }
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use one of: "
            f"{', '.join(sorted(MODE_NAMES))}."
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
# disk under separate sequence_meta_<source>.json files.

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


def init_sequence_config(data_root, source: str = SOURCE,
                         period=None) -> None:
    """Load `sequence_meta_<source>[_<period>].json` and populate globals.

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
    SEQUENCE_META_PATH = data_root / sequence_meta_name(source, period)
    if not SEQUENCE_META_PATH.exists():
        label = period.label if isinstance(period, Period) else period
        hint = (f"    python extract_patch_seq_for_datasets.py "
                f"--period {label}" if label else
                f"    python extract_patch_seq_for_datasets.py")
        print(
            f"ERROR: {SEQUENCE_META_PATH} not found.\n"
            f"Run from the project root:\n"
            f"    python validate_timestep.py --step_minutes <N>\n"
            f"{hint}",
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

    Wraps the CLOCK only - 23:45 + 30 gives 0015 - and deliberately says
    nothing about the date. Anything that opens a file must use
    `row_to_datetime_list`, which carries the day across midnight.
    """
    parts = reference_utc_str.strip().split(":")
    ref = datetime(2000, 1, 1, int(parts[0]), int(parts[1]))
    target = ref + timedelta(minutes=offset_minutes)
    return target.strftime("%H%M")


def row_to_datetime_list(row):
    """(date, HHMM) for every timestep in a sample row.

    The row's `date` column dates the REFERENCE, not the window, so a
    window through midnight has steps on two different days. Pairing
    every step with `row["date"]` - as this did until the patch pool
    guard caught it - reads the previous day's file for every
    post-midnight step: wrong tiles, wrong time, and silent whenever the
    index happens to be in range.
    """
    base = datetime.strptime(str(row["date"]).strip(), "%Y-%m-%d")
    parts = str(row["reference_utc"]).strip().split(":")
    ref = base.replace(hour=int(parts[0]), minute=int(parts[1]))
    out = []
    for offset in T_OFFSETS:
        t = ref + timedelta(minutes=offset)
        out.append((t.strftime("%Y-%m-%d"), t.strftime("%H%M")))
    return out


def row_to_hhmm_list(row):
    """Clock times only. Prefer `row_to_datetime_list` for file access."""
    ref = row["reference_utc"]
    return [reference_to_hhmm(ref, offset) for offset in T_OFFSETS]


# ============================================================================
# Patch loading
# ============================================================================

class StalePatchPool(RuntimeError):
    """The patch pool disagrees with the split CSVs about patch activity.

    `idx_t*` is a POSITION in a timestep's active-patch list, and that
    list lives only in patch_index.csv. Both the split CSVs and the patch
    files derive from it, so an out-of-range position is never a data
    condition - it can only mean the pool was built from a different
    index than the CSVs were.

    This is raised rather than zero-filled because the out-of-range slot
    is the *visible* symptom of a problem that is mostly invisible: when
    a patch becomes active it inserts into the middle of the list and
    shifts every later slot by one. Those shifted slots stay in range and
    read cleanly, silently pairing one region's input with another
    region's label. Padding the one detectable case with zeros would hide
    the only evidence that the rest are wrong.
    """

    def __init__(self, patches_dir, date_str, hhmm, var_name, suffix,
                 patch_idx, n_available):
        super().__init__(
            f"Stale patch pool at {patches_dir}\n"
            f"  {date_str} {hhmm} {var_name} ({suffix}): the split CSV "
            f"asks for slot {patch_idx}, the file holds {n_available}.\n"
            f"  The pool and the split CSVs disagree about what was "
            f"active here. Usually that means the pool was built from a "
            f"different patch_index.csv - but check the DATE first: a "
            f"window through midnight whose steps are read under the "
            f"row's date lands on the wrong day's file, which looks "
            f"identical to a stale pool.\n"
            f"  Slots that are still in range are NOT safe either - they "
            f"may point at the wrong tile.\n"
            f"  Diagnose : python extract_patches.py --audit_pool "
            f"[--period TAG]\n"
            f"  Repair   : delete the affected dates from {patches_dir} "
            f"and re-run extract_patches.py"
        )


def load_npy(patches_dir, date_str, var_name, hhmm, suffix):
    """Load a .npy patch file. Returns array of shape (N_patches, H, W)."""
    fn = f"{var_name}_{hhmm}_{suffix}.npy"
    path = os.path.join(patches_dir, date_str, fn)
    if not array_exists(path):
        return None
    return load_array(path)


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
            raise StalePatchPool(patches_dir, date_str, hhmm, var_name,
                                 suffix, patch_idx, data.shape[0])

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
    if data is None:
        return np.zeros((256, 256, n_label_channels), dtype=np.float32)
    if patch_idx >= data.shape[0]:
        raise StalePatchPool(patches_dir, date_str, hhmm, label_var,
                             label_suffix, patch_idx, data.shape[0])

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
        inputs_dict: dict with keys "past_hr" and "past_mr",
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

    # Determine input groups (past_hr, past_mr)
    input_groups = {}
    for key in INPUT_GROUP_KEYS:
        cfg = mode_config.get(key)
        if cfg is not None:
            input_groups[key] = cfg  # (var_config, resolution, suffix)

    n_skipped = 0
    n_yielded = 0

    for row_idx, row in df.iterrows():
        # Per-step (date, HHMM): a window through midnight spans two
        # days, and each step must be read from its OWN day's patches.
        step_keys = row_to_datetime_list(row)

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
                date_str, hhmm = step_keys[t_idx]
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
                date_str, hhmm = step_keys[N_INPUT + t_idx]
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

            # The patch number rides along so the writer can record which
            # of the 18 grid slots a sample came from. Per-patch ensemble
            # scoring needs that identity, and it is unrecoverable from the
            # tensors alone.
            n_yielded += 1
            # (date, reference_utc, patch) is the verification key. It is
            # unrecoverable from the tensors, and samples can be skipped
            # (see n_skipped), so position in the shard identifies
            # nothing - the key has to ride along with the sample.
            yield (stacked_inputs, stacked_label,
                   int(patch_numbers[p_pos]), str(row["date"]),
                   str(row.get("reference_utc", "")))

    print(f"  Generated {n_yielded} samples, skipped {n_skipped}")


# ============================================================================
# TF Dataset creation and saving
# ============================================================================

def get_output_signature(mode_config):
    """Build the tf output signature for the generator."""
    input_specs = {}
    for key in INPUT_GROUP_KEYS:
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


def _serialize_sample(stacked_inputs, stacked_label, patch_number=-1,
                      date_str="", reference_utc=""):
    """Serialize one (inputs_dict, label) sample as a tf.train.Example.

    `patch_number` identifies which of the 18 grid slots the sample came
    from. Training ignores it; the per-patch ensemble scorer reads it via
    `load_tfrecord_with_patch`.

    `date_str` + `reference_utc` complete the verification key
    `(date, reference_utc, patch)`, which is what verification_keys.py
    freezes. Without them a shard cannot be restricted to the
    leakage-free set, and the baseline comparison can only be scored on
    each model's own test split - different populations, with each
    model's test keys partly inside the other's training data.

    Shards written before any of these fields existed parse back as the
    sentinels (-1 / "") rather than failing.
    """
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
    feature["patch_number"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(patch_number)])
    )
    feature["date"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[str(date_str).encode()])
    )
    feature["reference_utc"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[str(reference_utc).encode()])
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
        for (stacked_inputs, stacked_label, patch_number,
             date_str, reference_utc) in generate_samples(
                csv_path, patches_dir, mode_config):
            if writer is None or n_samples % samples_per_shard == 0:
                if writer is not None:
                    writer.close()
                    shard_idx += 1
                writer = tf.io.TFRecordWriter(_shard_path(shard_idx))
                if n_samples == 0:
                    print(f"    -> shard 0: {_shard_path(0)}")
            writer.write(
                _serialize_sample(stacked_inputs, stacked_label,
                                  patch_number, date_str, reference_utc)
            )
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


def _make_parse_fn_with_patch(input_specs, label_spec):
    """Like `_make_parse_fn`, but also returns the sample's patch number.

    Kept separate so the training path's element structure stays exactly
    (inputs, label) — only the ensemble scorer wants the third element.
    `default_value=-1` makes shards written before the field existed parse
    cleanly instead of raising, so an old dataset degrades to "unknown
    patch" rather than being unreadable.
    """
    feature_description = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in input_specs
    }
    feature_description["label"] = tf.io.FixedLenFeature([], tf.string)
    feature_description["patch_number"] = tf.io.FixedLenFeature(
        [], tf.int64, default_value=-1)

    def parse(serialised):
        parsed = tf.io.parse_single_example(serialised, feature_description)
        inputs = {}
        for key, spec in input_specs.items():
            t = tf.io.parse_tensor(parsed[key], out_type=spec.dtype)
            t.set_shape(spec.shape)
            inputs[key] = t
        label = tf.io.parse_tensor(parsed["label"], out_type=label_spec.dtype)
        label.set_shape(label_spec.shape)
        return inputs, label, parsed["patch_number"]

    return parse


def _make_parse_fn_with_key(input_specs, label_spec):
    """Like `_make_parse_fn`, but also returns the verification key.

    Yields (inputs, label, date, reference_utc, patch_number) so a split
    can be restricted to the frozen leakage-free set. Empty-string and
    -1 defaults mean shards written before these fields existed parse
    cleanly; they simply match no key, which is the safe direction - an
    unidentifiable sample is excluded from a leakage-free comparison
    rather than silently admitted to it.
    """
    feature_description = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in input_specs
    }
    feature_description["label"] = tf.io.FixedLenFeature([], tf.string)
    feature_description["patch_number"] = tf.io.FixedLenFeature(
        [], tf.int64, default_value=-1)
    feature_description["date"] = tf.io.FixedLenFeature(
        [], tf.string, default_value="")
    feature_description["reference_utc"] = tf.io.FixedLenFeature(
        [], tf.string, default_value="")

    def parse(serialised):
        parsed = tf.io.parse_single_example(serialised, feature_description)
        inputs = {}
        for key, spec in input_specs.items():
            t = tf.io.parse_tensor(parsed[key], out_type=spec.dtype)
            t.set_shape(spec.shape)
            inputs[key] = t
        label = tf.io.parse_tensor(parsed["label"], out_type=label_spec.dtype)
        label.set_shape(label_spec.shape)
        return (inputs, label, parsed["date"], parsed["reference_utc"],
                parsed["patch_number"])

    return parse


def count_key_matches(shard_dir: Path, keys) -> tuple[int, int]:
    """(kept, dropped) for `keys` over a split, without reading tensors.

    Parses only date / reference_utc / patch_number, so this is a cheap
    pre-pass. It exists so an empty match can be reported BEFORE the
    dataset is handed to tf.data: a raise inside `from_generator` is
    re-thrown as an opaque UnknownError, burying the one message that
    tells the user what to do.
    """
    shard_paths = sorted(str(p) for p in shard_dir.glob("shard_*.tfrecord"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.tfrecord files in {shard_dir}")
    desc = {
        "date": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "reference_utc": tf.io.FixedLenFeature([], tf.string,
                                               default_value=""),
        "patch_number": tf.io.FixedLenFeature([], tf.int64,
                                              default_value=-1),
    }
    ds = tf.data.TFRecordDataset(shard_paths,
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(lambda r: tf.io.parse_single_example(r, desc),
                num_parallel_calls=tf.data.AUTOTUNE)
    kept = dropped = 0
    for rec in ds:
        k = (rec["date"].numpy().decode(),
             rec["reference_utc"].numpy().decode(),
             int(rec["patch_number"].numpy()))
        if k in keys:
            kept += 1
        else:
            dropped += 1
    return kept, dropped


def require_key_matches(shard_dir: Path, keys) -> tuple[int, int]:
    """count_key_matches, but fail loudly and usefully on zero."""
    kept, dropped = count_key_matches(shard_dir, keys)
    if kept == 0:
        raise SystemExit(
            f"No sample in {shard_dir} matched the frozen key set "
            f"({len(keys):,} keys, {dropped:,} samples scanned).\n"
            f"  Either this dataset predates the date/reference_utc shard "
            f"fields - rebuild it with create_datasets.py - or the frozen "
            f"set describes a different pair of windows than this run tag."
        )
    return kept, dropped


def load_tfrecord_with_key(shard_dir: Path,
                           mode_config: dict) -> tf.data.Dataset:
    """Load a split yielding (inputs, label, date, reference_utc, patch)."""
    shard_paths = sorted(str(p) for p in shard_dir.glob("shard_*.tfrecord"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard_*.tfrecord files in {shard_dir}")
    input_specs, label_spec = get_output_signature(mode_config)
    ds = tf.data.TFRecordDataset(shard_paths,
                                 num_parallel_reads=tf.data.AUTOTUNE)
    return ds.map(_make_parse_fn_with_key(input_specs, label_spec),
                  num_parallel_calls=tf.data.AUTOTUNE)


def load_tfrecord_with_patch(shard_dir: Path,
                             mode_config: dict) -> tf.data.Dataset:
    """Load a split yielding (inputs, label, patch_number) triples."""
    shard_paths = sorted(str(p) for p in shard_dir.glob("shard_*.tfrecord"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard_*.tfrecord files in {shard_dir}")
    input_specs, label_spec = get_output_signature(mode_config)
    ds = tf.data.TFRecordDataset(shard_paths, num_parallel_reads=tf.data.AUTOTUNE)
    return ds.map(_make_parse_fn_with_patch(input_specs, label_spec),
                  num_parallel_calls=tf.data.AUTOTUNE)


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


def create_and_save_datasets(data_root, mode, source=SOURCE, output_root=None,
                             datasets_root=None,
                             period=None, stats_period=_UNSET):
    """Create and save train, validation, and test datasets.

    Args:
        data_root: path to our_data/ directory containing CSVs and patches/
        mode: one of the names registered in get_mode_config, e.g.
            mtg_lightning, mtg_radar_rainfall,
            mtg_opera_mtgmr_rainfall, mtg_lightning_opera_occurrence
        source: which extract_patch_seq source the sample CSVs came from
            ('dbscan' = patch_index.csv from identify_patches, or
            always pipeline_config.SOURCE). The dataset directory is
            suffixed by source.
        output_root: where to save datasets (default: data_root/datasets/)
    """
    data_root = resolve_data_root(data_root)
    datasets_root = resolve_datasets_root(data_root, datasets_root)
    patches_dir = data_root / "patches"

    if output_root is None:
        output_root = datasets_root
    else:
        output_root = Path(output_root)

    # Point the lazy stats loader at this run's data_root before any
    # transform fires; required because the transforms can be called
    # from worker threads later.
    # Per-source normalization: stats are computed from the matching
    # train_data_<source>.csv, so the file is suffixed too.
    # With a period, every input defaults to the member-scoped variant:
    # its own split CSVs and its own statistics. Mixing a member's samples
    # with whole-archive statistics would leak dates the member never saw.
    #
    # `stats_period` decouples the two, because the default is wrong for
    # the baseline comparison. There the period tag names a sequence
    # WINDOW (`w48`), not a date range, and RECONVECT and SepConv-ens must
    # share one normalization or their outputs are not in the same space —
    # which would make the physical-space calibration meaningless.
    if stats_period is _UNSET:
        stats_period = period
    set_normalization_stats_path(
        data_root / normalization_stats_name(source, stats_period)
    )

    mode_config = get_mode_config(mode)
    # Suffix the dataset dir with the source so radar- and lightning-
    # driven runs don't clobber each other (the domain-adaptation
    # pipeline trains both and uses them as separate feature extractors).
    # The mode name already states its track, so
    # `datasets/mtg_lightning_opera_rainfall_dbscan/` is self-describing.
    # See train_models.build_run_tag for the single source of truth.
    from train_models import build_run_tag
    period_label = period.label if isinstance(period, Period) else period
    save_dir = output_root / build_run_tag(mode, source, period_label)

    # Print configuration summary
    print("=" * 70)
    print(f"COALITION-4 Dataset Creation - Mode: {mode}  Source: {source}")
    print("=" * 70)
    print(f"Data root:    {data_root}")
    print(f"Patches dir:  {patches_dir}")
    _stats_file = normalization_stats_name(source, stats_period)
    _stats_note = ("" if stats_period == period
                   else "   <- overridden, decoupled from --period")
    print(f"Stats file:   {data_root / _stats_file}{_stats_note}")
    print(f"Period:       {period if period else 'none (whole archive)'}")
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
        split: split_csv_name(split, source, period)
        for split in ("train", "validation", "test")
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
        # `track` and `run_tag` mirror the naming convention documented in
        # train_models.build_run_tag - `track` is the human-facing label
        # ("rainfall"/"occurrence") and `run_tag` is the full artefact tag
        # `<mode>_<source>`.
        _track = ("occurrence" if mode_config["label_type"] == "lightning"
                  else "rainfall")
        meta = {
            "mode": mode,
            "source": source,
            "track": _track,
            "run_tag": build_run_tag(mode, source, period_label),
            # Bounds, not just the label: the feature-extractor overlap
            # check compares dates and must never have to trust a filename.
            "period": period.to_dict() if isinstance(period, Period) else None,
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
    return save_dir


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create COALITION-4 TF datasets from pre-extracted patches."
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=list(BUILDABLE_MODES),
        help="Dataset mode. The KD student (mtg_opera_occurrence) is "
             "absent by design: it trains on the teacher's dataset with "
             "past_hr sliced — see train_lightning_kd.py."
    )
    parser.add_argument(
        "--datasets_root", type=str, default=None, metavar="PATH",
        help="Root holding the built TFRecord datasets (default: "
             "<data_root>/datasets, or $COALITION4_DATASETS_ROOT). Point "
             "it at another disk to keep the datasets off the one "
             "holding the patch pool."
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Root directory containing CSVs and patches/ subfolder"
    )
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Output root for saved datasets (default: data_root/datasets/)"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Enumerate the seasonal ensemble from the [seasons] block of "
             "training.config crossed with the years present in "
             "patch_index.csv, report each member's period and coverage "
             "plus any discrepancy, and register the plan in "
             "our_data/ensemble_registry.json. Builds nothing — build each "
             "member afterwards with --period <label>."
    )
    parser.add_argument(
        "--config", type=str, default="training.config",
        help="Config file holding the [seasons] block, read by --ensemble "
             "(default: training.config)."
    )
    parser.add_argument(
        "--period", type=str, default=None,
        help="Build one registered ensemble member, e.g. --period 2025warm. "
             "The label must appear in the registry's most recent state; "
             "its bounds come from there. Omit to build the unscoped, "
             "whole-archive dataset."
    )
    parser.add_argument(
        "--global_stats", action="store_true",
        help="Normalise with the unsuffixed whole-archive statistics even "
             "when --period is given, instead of the period-suffixed "
             "file. For a period tag that names a sequence WINDOW rather "
             "than a date range, where the model belongs in the "
             "whole-archive space. The SepConv-ens baseline does NOT use "
             "this: it is normalised by its own split (w48), because "
             "RECONVECT's training split contains 36 of the baseline's "
             "test timestamps, so RECONVECT's constants would be defined "
             "partly by data the baseline is tested on. What has to match "
             "is training and inversion, not the two models."
    )
    parser.add_argument(
        "--no-archive", action="store_true",
        help="Do not spawn the background 7-Zip job after building. By "
             "default a finished dataset is archived immediately (~4.8%% "
             "of its size) and the shards deleted once verified."
    )
    parser.add_argument(
        "--archive_level", type=int, default=_ARCHIVE_LEVEL,
        choices=[0, 1, 3, 5, 7, 9],
        help=f"7-Zip -mx level for the background archive job "
             f"(default: {_ARCHIVE_LEVEL})."
    )
    parser.add_argument(
        "--archive_workers", type=int, default=_ARCHIVE_WORKERS,
        help=f"7-Zip threads for the background archive job (default: "
             f"{_ARCHIVE_WORKERS} = half the logical cores)."
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=_ARCHIVE_MAX_CONCURRENT,
        help=f"Maximum simultaneous background archive jobs "
             f"(default: {_ARCHIVE_MAX_CONCURRENT})."
    )
    args = parser.parse_args()

    # Resolve the roots ONCE, so every use below - including the plain
    # `Path(args.data_root)` ones - sees a real path rather than None.
    args.data_root = str(resolve_data_root(args.data_root))
    args.datasets_root = str(resolve_datasets_root(args.data_root,
                                                  args.datasets_root))

    # --- Plan mode: register the ensemble and stop -----------------------
    if args.ensemble:
        if args.period:
            parser.error("--ensemble registers the plan; --period builds a "
                         "member from it. Use one or the other.")
        seasons = load_seasons(args.config)
        counts = load_index_dates(
            Path(args.data_root) / "patch_index" / "patch_index.csv"
        )
        plan = enumerate_members(counts, seasons)
        print(format_plan(plan, mode=args.mode, source=SOURCE))

        # An overlap means two members would train on shared dates, which
        # defeats the entire point of the split. Refuse to register it.
        if plan.overlaps:
            raise SystemExit(
                "\nERROR: members overlap — fix the [seasons] block before "
                "registering. Nothing was written."
            )

        state = append_state(args.data_root, plan,
                             mode=args.mode, source=SOURCE)
        print(f"\nRegistered {state['n_members']} member(s) "
              f"({state['n_buildable']} buildable) in "
              f"{registry_path(args.data_root)}")
        buildable = [m.label for m in plan.buildable]
        if buildable:
            print("\nBuild them one at a time:")
            for label in buildable:
                print(f"    python create_datasets.py --mode {args.mode} "
                      f"--period {label}")
        return

    # --- Build mode ------------------------------------------------------
    # A period must come from the registry, so a member can never be built
    # over bounds that differ from the ones the plan recorded.
    period = None
    if args.period:
        # Two ways a label can resolve, checked in this order:
        #
        #  1. The sequence metadata on disk. Whatever
        #     extract_patch_seq_for_datasets.py actually wrote is the
        #     authority on what exists, and it records the bounds it used.
        #     This also covers window tags such as `w48`, which name a
        #     sequence window rather than an ensemble member and so never
        #     appear in the registry.
        #  2. The registered ensemble plan, for genuine members.
        seq_meta = Path(args.data_root) / sequence_meta_name(SOURCE,
                                                             args.period)
        if seq_meta.is_file():
            blob = json.loads(seq_meta.read_text()).get("period")
            period = Period.from_dict(blob)
            if period is None:
                parser.error(
                    f"{seq_meta} exists but records no period block, so "
                    f"{args.period!r} cannot be resolved. Rebuild it with "
                    f"--period {args.period}."
                )
        else:
            state = require_last_state(args.data_root)
            period = state_period(state, args.period)
            if period is None:
                known = [m["label"] for m in state.get("members", [])]
                parser.error(
                    f"Period {args.period!r} has neither a sequence "
                    f"metadata file at {seq_meta} nor an entry in the "
                    f"registered ensemble plan (members: "
                    f"{known or '(none)'})."
                )
            member = state_member(state, args.period)
            if member and member.get("status") == "no-data":
                raise SystemExit(
                    f"ERROR: member {args.period!r} was registered with no "
                    f"data ({member['start']} .. {member['end']}). Download "
                    f"those dates and re-register before building it."
                )

    # Populate the module-level schema constants from the sequence
    # metadata before any sample-generating function is called.
    init_sequence_config(args.data_root, SOURCE, period=period)

    save_dir = create_and_save_datasets(
        data_root=args.data_root,
        datasets_root=args.datasets_root,
        mode=args.mode,
        source=SOURCE,
        output_root=args.output_root,
        period=period,
        stats_period=None if args.global_stats else _UNSET,
    )

    # Compression happens once, here, and only here: training never
    # modifies a dataset, so this archive stays valid for its lifetime.
    # Detached so the next member can start building immediately - a
    # 66 GB member takes ~20 min to compress at -mx=5.
    if not args.no_archive and save_dir is not None:
        from compress_datasets import spawn_job
        print("\nArchiving in the background ...")
        spawn_job("compress", save_dir.name, save_dir.parent,
                  level=args.archive_level, workers=args.archive_workers,
                  max_concurrent=args.max_concurrent)
        print("  Dataset creation is done; you can start the next member "
              "or begin training now.")


if __name__ == "__main__":
    main()
    