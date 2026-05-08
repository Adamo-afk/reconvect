"""
create_datasets.py — COALITION-4 Romanian Adaptation
=====================================================
Creates train, validation, and test TF datasets from pre-extracted .npy patches.

Usage:
    python create_datasets.py --mode mtg_lightning --data_root ./our_data
    python create_datasets.py --mode mtg_radar --data_root ./our_data
    python create_datasets.py --mode mtg_radar_continuous --data_root ./our_data

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
# Transform functions (Table A1 from Leinonen et al. 2022)
# Applied to physical float values at dataset creation time.
# ============================================================================

def transform_rzc(x):
    """RZC rain rate: [log10(x) + 0.051] / 0.528, fill=0.01 mm/h"""
    x = np.where(np.isnan(x), 0.01, x)
    x = np.clip(x, 0.01, None)
    return (np.log10(x) + 0.051) / 0.528


def transform_czc(x):
    """CZC composite reflectivity: (x - 21.3) / 8.71, fill=-5 dBZ"""
    x = np.where(np.isnan(x), -5.0, x)
    return (x - 21.3) / 8.71


def transform_ezc(x):
    """EZC-20 echo-top height: x / 1.97, fill=0"""
    x = np.where(np.isnan(x), 0.0, x)
    return x / 1.97


def transform_lzc(x):
    """LZC liquid water content: [log10(x) + 0.274] / 0.135, fill=0.5"""
    x = np.where(np.isnan(x), 0.5, x)
    x = np.clip(x, 0.5, None)
    return (np.log10(x) + 0.274) / 0.135


def transform_bzc(x):
    """BZC base reflectivity: x / 100"""
    x = np.where(np.isnan(x), 0.0, x)
    return x / 100.0


def transform_cpch(x):
    """CPCH precipitation: log10(x), fill=0.01, threshold=0.1"""
    x = np.where(np.isnan(x), 0.01, x)
    x = np.where(x < 0.1, 0.01, x)
    x = np.clip(x, 0.01, None)
    return np.log10(x)


def transform_lightning_density(x):
    """Lightning density: [log10(x) + 0.593] / 0.640, fill=1e-4"""
    x = np.where(np.isnan(x), 1e-4, x)
    x = np.clip(x, 1e-4, None)
    return (np.log10(x) + 0.593) / 0.640


def transform_lightning_current(x):
    """Lightning current: [log10(x) - 0.0718] / 0.731, fill=1e-8"""
    x = np.where(np.isnan(x), 1e-8, x)
    x = np.clip(x, 1e-8, None)
    return (np.log10(x) - 0.0718) / 0.731


def transform_occurrence(x):
    """Lightning occurrence: binary 0/1, cast to float32"""
    x = np.where(np.isnan(x), 0.0, x)
    return np.clip(x, 0.0, 1.0)


def transform_vis(x):
    """Solar visible channels (VIS006, vis_06, etc.): x / 100"""
    x = np.where(np.isnan(x), 0.0, x)
    return x / 100.0


def transform_ir039(x):
    """IR-039 / ir_38 (solar+thermal): (x - 274) / 17.5"""
    x = np.where(np.isnan(x), 274.0, x)
    return (x - 274.0) / 17.5


def transform_thermal(x):
    """Thermal IR/WV channels: (x - 250) / 10"""
    x = np.where(np.isnan(x), 250.0, x)
    return (x - 250.0) / 10.0


def transform_ctth_alti(x):
    """Cloud-top height: (x - 5260) / 2810, fill=-1000, missing=65535"""
    x = np.where(np.isnan(x), -1000.0, x)
    x = np.where(x > 60000, -1000.0, x)  # handle 65535 missing
    return (x - 5260.0) / 2810.0


def transform_ctth_tempe(x):
    """Cloud-top temperature: (x - 260) / 19.1, fill=330, missing=65535"""
    x = np.where(np.isnan(x), 330.0, x)
    x = np.where(x > 60000, 330.0, x)
    return (x - 260.0) / 19.1


def transform_cmic_phase(x):
    """Cloud-top phase: one-hot encode. Input categories: 1,2,3,4 + 0/NaN=missing.
    Returns (H, W, 5) array with channels: [no_cloud, liquid, ice, mixed, missing].
    """
    x = np.where(np.isnan(x), 0.0, x)
    x = np.round(x).astype(np.int32)
    h, w = x.shape
    one_hot = np.zeros((h, w, 5), dtype=np.float32)
    # Map: 0→missing(ch4), 1→no_cloud(ch0), 2→liquid(ch1), 3→ice(ch2), 4→mixed(ch3)
    mapping = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}
    for val, ch in mapping.items():
        one_hot[:, :, ch] = (x == val).astype(np.float32)
    # Anything not in mapping → missing channel
    known = np.isin(x, list(mapping.keys()))
    one_hot[:, :, 4] = np.where(~known, 1.0, one_hot[:, :, 4])
    return one_hot


def transform_cmic_cot(x):
    """Cloud optical thickness: [log10(x) - 0.94] / 0.588, fill=0.1, missing=65535"""
    x = np.where(np.isnan(x), 0.1, x)
    x = np.where(x > 60000, 0.1, x)
    x = np.clip(x, 0.1, None)
    return (np.log10(x) - 0.94) / 0.588


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
    "IR_108": (transform_thermal, None),
    "WV_062": (transform_thermal, None),
    "WV_073": (transform_thermal, None),
}

MTG_HR_SAT_CONFIG = {
    "vis_06": (transform_vis, None),
}

MTG_MR_SAT_CONFIG = {
    "ir_38":  (transform_ir039, None),
    "ir_105": (transform_thermal, None),
    "wv_63":  (transform_thermal, None),
    "wv_73":  (transform_thermal, None),
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
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use: mtg_lightning, mtg_radar, "
            f"mtg_radar_continuous. (MSG modes are currently disabled.)"
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

    mode_config = get_mode_config(mode)
    save_dir = output_root / mode

    # Print configuration summary
    print("=" * 70)
    print(f"COALITION-4 Dataset Creation — Mode: {mode}")
    print("=" * 70)
    print(f"Data root:    {data_root}")
    print(f"Patches dir:  {patches_dir}")
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
        choices=["mtg_lightning", "mtg_radar", "mtg_radar_continuous"],
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
    