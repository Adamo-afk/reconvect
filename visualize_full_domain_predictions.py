"""
visualize_full_domain_predictions.py
=====================================
Top-N timestep aggregator: read a per-source split CSV, pick the
reference timesteps with the most qualifying patches, build full
768x1536 Romania-canvas GT and prediction maps for every future lead
time (t+15, t+30, t+45), and save a combined plot per timestep.

Per top-N reference timestep:
  1. Read the row's `patch_numbers` + `idx_t-30/-15/0` columns.
  2. Build per-patch input tensors in memory (no disk writes) for every
     qualifying patch.
  3. Run a single batched model.predict over all qualifying patches.
  4. For each future lead time, paste each patch's GT label and the
     model's prediction onto independent 768x1536 canvases. Patches
     that are NOT qualifying (i.e. not in the row's patch_numbers
     list) stay zero so they read as visually "no data".
  5. For lightning (`label_type='lightning'`) the prediction map is
     thresholded against the optimal_threshold from the matching
     evaluation_results.json (override via --threshold).
  6. Overlay Romania's national border via the (lat, lon) grids if
     cartopy is installed; otherwise draw a coarse hardcoded polygon
     converted to pixel coords via pyproj.

Example commands
----------------
    # Top 5 lightning-occurrence base-model timesteps, OPERA-driven
    python visualize_full_domain_predictions.py \
        --csv our_data/test_data_dbscan.csv \
        --mode mtg_lightning_opera_occurrence \
        --source dbscan --top_n 5

    # Same but fine-tuned model, manual threshold override
    python visualize_full_domain_predictions.py \
        --csv our_data/test_data_dbscan.csv \
        --mode mtg_lightning_opera_occurrence \
        --source dbscan --top_n 5 --finetuned --threshold 0.5

    # OPERA 5-class on the lightning-driven split CSV
    python visualize_full_domain_predictions.py \
        --csv our_data/validation_data_lightning.csv \
        --mode mtg_opera_mtgmr \
        --source lightning --top_n 3

Outputs land under
    <output_dir>/full_domain_<run_tag>[_finetuned]/
        ts01_<date>_<HHMM>.png
        ts02_<date>_<HHMM>.png
        ...
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import tensorflow as tf

# Reuse the data-loading and mode-config helpers from create_datasets so the
# input tensors we feed the model match the training/eval pipeline byte-for-byte.
from create_datasets import (
    get_mode_config,
    load_and_transform_group,
    load_label,
    LABEL_CHANNELS,
)

# ============================================================================
# Geometry constants (must mirror identify_patches / extract_patches)
# ============================================================================
N_ROWS, N_COLS = 3, 6
PATCH_SIZE = 256
N_PATCHES = N_ROWS * N_COLS
H_FULL = N_ROWS * PATCH_SIZE              # 768
W_FULL = N_COLS * PATCH_SIZE              # 1536

# UTM zone 35N (EPSG:31700) extent of the Romania grid, taken from
# c4dl/projection.py. Order: (lower_left_x, upper_right_x, lower_left_y,
# upper_right_y) in metres.
ROMANIA_EXTENT_UTM = (-177324.0, 1331353.0, 77148.0, 723370.0)

# Column names in the per-source split CSVs use STEP indices, not minute
# offsets - the schema is stable across step_minutes values. The actual
# minute offsets are derived at runtime by multiplying by step_minutes
# from our_data/timestep_config.json.
INPUT_STEP_OFFSETS = [-2, -1, 0]
LABEL_STEP_OFFSETS = [1, 2, 3]


def _step_column_name(offset: int) -> str:
    """Mirror of extract_patch_seq_for_datasets.step_column_name."""
    if offset < 0:
        return f"idx_t{offset}"
    if offset == 0:
        return "idx_t0"
    return f"idx_t+{offset}"


INPUT_TIME_COLS = [_step_column_name(o) for o in INPUT_STEP_OFFSETS]
LABEL_TIME_COLS = [_step_column_name(o) for o in LABEL_STEP_OFFSETS]
RADAR_CLASS_NAMES = ["R<10", "10≤R<20", "20≤R<30",
                     "30≤R<40", "R≥40"]


# ============================================================================
# Patch geometry
# ============================================================================
def get_patch_bounds(patch_number: int) -> tuple[int, int, int, int]:
    """1-indexed patch_number -> (r0, r1, c0, c1) in the 768x1536 grid."""
    idx = patch_number - 1
    row = idx // N_COLS
    col = idx % N_COLS
    r0 = row * PATCH_SIZE
    c0 = col * PATCH_SIZE
    return r0, r0 + PATCH_SIZE, c0, c0 + PATCH_SIZE


# ============================================================================
# Country outline (cartopy preferred, pyproj fallback)
# ============================================================================
# Coarse hardcoded Romania boundary in (lon, lat) used only when cartopy is
# not installed. Smooth enough to read as "this is Romania" but not
# politically precise.
_ROMANIA_OUTLINE_LONLAT = [
    (22.69, 47.99), (23.14, 48.10), (24.30, 47.91), (25.41, 47.93),
    (26.40, 48.22), (27.05, 47.99), (27.55, 47.40), (28.10, 46.81),
    (28.21, 45.97), (28.83, 45.30), (29.65, 45.18), (29.69, 44.81),
    (28.84, 44.05), (28.05, 43.81), (27.00, 44.13), (25.65, 43.69),
    (24.50, 43.68), (23.27, 43.83), (22.65, 44.22), (22.42, 44.71),
    (21.56, 44.77), (21.36, 45.04), (20.79, 45.46), (20.25, 46.10),
    (20.79, 46.30), (21.06, 46.83), (22.13, 47.59), (22.69, 47.99),
]


def latlon_to_pixel(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lon, lat) arrays to (col_px, row_px) on the 768x1536 grid.
    Uses pyproj to go lat/lon -> EPSG:31700 -> pixel via the UTM extent.
    Pixels are returned as floats so matplotlib can interpolate the line.
    """
    import pyproj
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:31700", always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xmin, xmax, ymin, ymax = ROMANIA_EXTENT_UTM
    # Image y axis is flipped (row 0 is the top of the canvas = top of the
    # UTM extent), so map ymax -> row 0 and ymin -> row H_FULL.
    col = (x - xmin) / (xmax - xmin) * W_FULL
    row = (ymax - y) / (ymax - ymin) * H_FULL
    return col, row


def overlay_romania(ax, *, color="black", linewidth=1.2):
    """Draw the Romania border onto an ax that already shows the 768x1536
    canvas via imshow with default (pixel) coords."""
    lonlat = np.asarray(_ROMANIA_OUTLINE_LONLAT, dtype=np.float64)
    col, row = latlon_to_pixel(lonlat[:, 0], lonlat[:, 1])
    ax.plot(col, row, color=color, linewidth=linewidth, zorder=5)


# ============================================================================
# Top-N selection
# ============================================================================
def load_top_n_rows(csv_path: Path, top_n: int) -> pd.DataFrame:
    """Read the split CSV, score each row by `len(patch_numbers)`, return
    the top-N rows sorted by descending count (ties broken by chronological
    order to keep the output reproducible)."""
    df = pd.read_csv(csv_path)
    if "patch_numbers" not in df.columns:
        raise ValueError(
            f"{csv_path} is missing the `patch_numbers` column. Is this "
            f"actually one of the per-source train/validation/test CSVs?"
        )
    df["n_patches"] = df["patch_numbers"].apply(
        lambda s: len(ast.literal_eval(s))
    )
    df = df.sort_values(
        ["n_patches", "date", "reference_utc"],
        ascending=[False, True, True],
    ).head(top_n).reset_index(drop=True)
    return df


# ============================================================================
# Per-row tensor assembly
# ============================================================================
def _ref_to_hhmm(ref_utc: str, offset_min: int) -> str:
    parts = ref_utc.split(":")
    ref_dt = datetime(2000, 1, 1, int(parts[0]), int(parts[1]))
    return (ref_dt + timedelta(minutes=offset_min)).strftime("%H%M")


def _load_step_minutes(data_root: Path) -> int:
    """Read step_minutes from our_data/timestep_config.json. This is what
    multiplies the step indices in the CSV columns to recover minute
    offsets used to build HHMM filenames."""
    config_path = data_root / "timestep_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found - run validate_timestep.py first "
            f"or pass --step_minutes manually."
        )
    with open(config_path) as f:
        return int(json.load(f)["step_minutes"])


def build_batch_inputs(row, patches_dir: str, mode_config: dict,
                       step_minutes: int) -> tuple[dict, list[int]]:
    """Build a single batched inputs dict for every qualifying patch in `row`.

    Returns:
        inputs_dict: keys are model input names ("past_hr", "past_mr",
                     "past_lr"), each value an np.ndarray of shape
                     (N_valid, 3, H, W, C). Skipped patches (any input
                     timestep missing) are dropped from the batch.
        valid_patches: list of patch_numbers (1-indexed) actually in the
                       batch, in batch order.
    """
    date_str = row["date"]
    ref_utc = row["reference_utc"].strip()
    patches = ast.literal_eval(row["patch_numbers"])
    idx_lists = {col: ast.literal_eval(row[col]) for col in INPUT_TIME_COLS}

    input_hhmms = [_ref_to_hhmm(ref_utc, off * step_minutes)
                   for off in INPUT_STEP_OFFSETS]

    input_groups: dict[str, tuple] = {}
    for key in ("past_hr", "past_mr", "past_lr"):
        cfg = mode_config.get(key)
        if cfg is not None:
            input_groups[key] = cfg

    accum: dict[str, list] = {key: [] for key in input_groups}
    valid_patches: list[int] = []

    for p_pos, patch_num in enumerate(patches):
        per_group: dict[str, list] = {key: [] for key in input_groups}
        ok = True
        for t_idx, col in enumerate(INPUT_TIME_COLS):
            hhmm = input_hhmms[t_idx]
            npy_idx = idx_lists[col][p_pos]
            for gk, (vc, res, sfx) in input_groups.items():
                ts = load_and_transform_group(
                    patches_dir, date_str, hhmm, sfx, vc, npy_idx, res,
                )
                if ts is None:
                    ok = False
                    break
                per_group[gk].append(ts)
            if not ok:
                break
        if not ok:
            continue
        for gk in input_groups:
            accum[gk].append(np.stack(per_group[gk], axis=0))  # (T, H, W, C)
        valid_patches.append(patch_num)

    inputs_dict = {
        gk: np.stack(arr, axis=0).astype(np.float32)        # (N, T, H, W, C)
        for gk, arr in accum.items() if arr
    }
    return inputs_dict, valid_patches


def build_full_gt(row, patches_dir: str, mode_config: dict,
                  label_type: str, step_minutes: int) -> list[np.ndarray]:
    """Return a list of 3 full-domain GT canvases (one per lead time).

    Lightning: float32 in [0, 1], shape (H_FULL, W_FULL).
    Radar:     int32 class index in {0..4}; -1 marks "no qualifying patch".
    """
    date_str = row["date"]
    ref_utc = row["reference_utc"].strip()
    patches = ast.literal_eval(row["patch_numbers"])

    label_var = mode_config["label_var"]
    label_transform = mode_config["label_transform"]
    label_suffix = mode_config["label_suffix"]
    n_label_ch = LABEL_CHANNELS[label_type]

    canvases: list[np.ndarray] = []
    for t, col in enumerate(LABEL_TIME_COLS):
        label_hhmm = _ref_to_hhmm(ref_utc, LABEL_STEP_OFFSETS[t] * step_minutes)
        label_idxs = ast.literal_eval(row[col])

        if label_type == "lightning":
            canvas = np.zeros((H_FULL, W_FULL), dtype=np.float32)
        else:
            canvas = np.full((H_FULL, W_FULL), -1, dtype=np.int32)

        for p_pos, patch_num in enumerate(patches):
            r0, r1, c0, c1 = get_patch_bounds(patch_num)
            npy_idx = label_idxs[p_pos]
            gt_patch = load_label(
                patches_dir, date_str, label_hhmm,
                label_var, label_transform, label_suffix,
                npy_idx, n_label_channels=n_label_ch,
            )
            if label_type == "lightning":
                canvas[r0:r1, c0:c1] = gt_patch[..., 0]
            else:
                canvas[r0:r1, c0:c1] = np.argmax(gt_patch, axis=-1)
        canvases.append(canvas)
    return canvases


def build_full_pred(predictions: np.ndarray, valid_patches: list[int],
                    label_type: str) -> list[np.ndarray]:
    """Project model predictions (N, T_future, H, W, C_out) back onto
    three full-domain canvases.

    Lightning: float32 probability in [0, 1] (raw, not thresholded).
    Radar:     int32 class index 0..4; -1 marks "no qualifying patch".
    """
    T_future = predictions.shape[1]
    canvases: list[np.ndarray] = []
    for t in range(T_future):
        if label_type == "lightning":
            canvas = np.zeros((H_FULL, W_FULL), dtype=np.float32)
        else:
            canvas = np.full((H_FULL, W_FULL), -1, dtype=np.int32)

        for p_pos, patch_num in enumerate(valid_patches):
            r0, r1, c0, c1 = get_patch_bounds(patch_num)
            pred_patch = predictions[p_pos, t]                # (256, 256, C)
            if label_type == "lightning":
                canvas[r0:r1, c0:c1] = pred_patch[..., 0]
            else:
                canvas[r0:r1, c0:c1] = np.argmax(pred_patch, axis=-1)
        canvases.append(canvas)
    return canvases


# ============================================================================
# Plotting
# ============================================================================
def _plot_inactive_mask(ax, valid_patches: list[int]):
    """Hatch-shade the patch slots that are NOT in `valid_patches` so the
    viewer can tell which regions have no model output."""
    valid_set = set(valid_patches)
    for p in range(1, N_PATCHES + 1):
        if p in valid_set:
            continue
        r0, _, c0, _ = get_patch_bounds(p)
        rect = Rectangle(
            (c0, r0), PATCH_SIZE, PATCH_SIZE,
            facecolor="lightgray", alpha=0.55,
            edgecolor="none", zorder=2,
        )
        ax.add_patch(rect)


def plot_full_domain(
    gt_canvases: list[np.ndarray],
    pred_canvases: list[np.ndarray],
    valid_patches: list[int],
    label_type: str,
    *,
    date_str: str,
    ref_utc: str,
    threshold: float | None,
    output_path: Path,
    step_minutes: int,
):
    lead_titles = [f"t+{o * step_minutes}" for o in LABEL_STEP_OFFSETS]
    label_offsets_min = [o * step_minutes for o in LABEL_STEP_OFFSETS]
    """Draw the GT-over-Pred 2x3 figure for one reference timestep."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)

    # --- Top row: ground truth -------------------------------------------
    if label_type == "lightning":
        gt_cmap = mcolors.LinearSegmentedColormap.from_list(
            "gt_red", ["#fff5f0", "#67000d"]
        )
        gt_kwargs = dict(cmap=gt_cmap, vmin=0.0, vmax=1.0,
                         aspect="equal", interpolation="nearest")
    else:
        cmap_radar = plt.get_cmap("viridis", 5)
        gt_kwargs = dict(cmap=cmap_radar, vmin=0, vmax=4,
                         aspect="equal", interpolation="nearest")

    for t in range(3):
        ax = axes[0, t]
        canvas = gt_canvases[t]
        if label_type == "radar":
            display = np.where(canvas < 0, np.nan, canvas.astype(float))
            im_gt = ax.imshow(display, **gt_kwargs)
        else:
            im_gt = ax.imshow(canvas, **gt_kwargs)
        _plot_inactive_mask(ax, valid_patches)
        try:
            overlay_romania(ax)
        except Exception:
            pass
        active_px = int(np.sum(canvas > 0)) if label_type == "lightning" \
            else int(np.sum((canvas != 0) & (canvas != -1)))
        ax.text(
            8, H_FULL - 12, f"pixels={active_px}",
            color="white", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="black", alpha=0.55, edgecolor="none"),
            va="bottom", ha="left", zorder=6,
        )
        ax.set_title(f"GT — {lead_titles[t]} "
                     f"({_ref_to_hhmm(ref_utc, label_offsets_min[t])} UTC)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, W_FULL); ax.set_ylim(H_FULL, 0)

    # --- Bottom row: predictions -----------------------------------------
    if label_type == "lightning":
        # Diverging blue->white->red centred on the operative threshold so
        # the visual encodes "above threshold" without hiding the raw scores.
        thr = threshold if threshold is not None else 0.5
        norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=thr, vmax=1.0)
        pred_kwargs = dict(cmap="RdYlBu_r", norm=norm,
                           aspect="equal", interpolation="nearest")
    else:
        pred_kwargs = dict(cmap=plt.get_cmap("viridis", 5),
                           vmin=0, vmax=4,
                           aspect="equal", interpolation="nearest")

    for t in range(3):
        ax = axes[1, t]
        canvas = pred_canvases[t]
        if label_type == "radar":
            display = np.where(canvas < 0, np.nan, canvas.astype(float))
            im_pred = ax.imshow(display, **pred_kwargs)
        else:
            im_pred = ax.imshow(canvas, **pred_kwargs)
        _plot_inactive_mask(ax, valid_patches)
        try:
            overlay_romania(ax)
        except Exception:
            pass
        if label_type == "lightning":
            thr_for_count = threshold if threshold is not None else 0.5
            above = int(np.sum(canvas >= thr_for_count))
            ax.text(
                8, H_FULL - 12,
                f"pixels≥{thr_for_count:.2f}={above}  max={canvas.max():.3f}",
                color="white", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="black", alpha=0.55, edgecolor="none"),
                va="bottom", ha="left", zorder=6,
            )
            title_suffix = f"(≥{thr_for_count:.2f})"
        else:
            title_suffix = ""
        ax.set_title(f"Pred {title_suffix} — {lead_titles[t]} "
                     f"({_ref_to_hhmm(ref_utc, label_offsets_min[t])} UTC)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, W_FULL); ax.set_ylim(H_FULL, 0)

    # Colorbars (one per row).
    if label_type == "lightning":
        cax_gt = fig.colorbar(im_gt, ax=axes[0, :].ravel().tolist(),
                              shrink=0.7, pad=0.01, location="right")
        cax_gt.set_label("Occurrence")
        cax_pred = fig.colorbar(im_pred, ax=axes[1, :].ravel().tolist(),
                                shrink=0.7, pad=0.01, location="right")
        cax_pred.set_label("Probability")
    else:
        cax = fig.colorbar(im_gt, ax=axes.ravel().tolist(),
                           ticks=[0, 1, 2, 3, 4],
                           shrink=0.7, pad=0.01, location="right")
        cax.set_ticklabels(RADAR_CLASS_NAMES)
        cax.set_label("Rain-rate class")

    fig.suptitle(
        f"{'Lightning' if label_type == 'lightning' else 'OPERA 5-class'} "
        f"prediction — Date: {date_str}  |  Ref: {ref_utc} UTC  |  "
        f"Patches: {len(valid_patches)}/{N_PATCHES}",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Model loading
# ============================================================================
def _custom_objects():
    # Import here so importing this module never forces TF graph construction.
    from evaluate_coalition import (
        ReflectionPadding2D, ConvBlock, ResBlock, GRUResBlock, ResGRU,
        SwinBlock, WeightedFocalLoss,
        iou_metric, true_pos, false_pos, false_neg,
    )
    return {
        "ReflectionPadding2D": ReflectionPadding2D,
        "ConvBlock":           ConvBlock,
        "ResBlock":            ResBlock,
        "GRUResBlock":         GRUResBlock,
        "ResGRU":              ResGRU,
        "SwinBlock":           SwinBlock,
        "WeightedFocalLoss":   WeightedFocalLoss,
        "iou_metric":          iou_metric,
        "true_pos":            true_pos,
        "false_pos":           false_pos,
        "false_neg":           false_neg,
    }


def load_model_artifact(model_dir: Path, mode: str, source: str,
                        finetuned: bool) -> tf.keras.Model:
    """Load either the base or the Swin-head model for (mode, source).

    For fine-tuned models, rebuild the architecture via
    train_models.build_finetune_model (same rebuild + load_weights trick
    evaluate_coalition.py uses to dodge the nested-sub-Model load_model
    shape mismatch in TF 2.10).
    """
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    run_tag = f"{mode}_{source}"
    base_path = model_dir / f"coalition_{run_tag}.keras"

    if not finetuned:
        if not base_path.is_file():
            raise FileNotFoundError(f"Base model not found: {base_path}")
        return tf.keras.models.load_model(
            str(base_path), custom_objects=_custom_objects(),
        )

    ft_path = model_dir / f"coalition_{run_tag}_finetuned.keras"
    history_path = model_dir / f"history_{run_tag}_finetuned.json"
    if not ft_path.is_file():
        raise FileNotFoundError(f"Fine-tuned model not found: {ft_path}")
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Fine-tune rebuild needs the base checkpoint at {base_path}. "
            f"`--stage both` keeps both files side by side; copy the base "
            f"into --model_dir if you trained the stages separately."
        )
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Fine-tune rebuild needs Swin hyperparameters from "
            f"{history_path} (written by train_finetune)."
        )

    with open(history_path) as f:
        hist_meta = json.load(f)
    swin = hist_meta.get("swin", {})
    finetune_cfg = {
        "optimizer":     hist_meta.get("optimizer", "adamw"),
        "weight_decay":  hist_meta.get("weight_decay", 0.01),
        "initial_lr":    hist_meta.get("initial_lr", 3e-4),
        "warmup_epochs": 1,
        "min_lr":        1e-6,
        "epochs":        hist_meta.get("epochs_completed", 1),
        "window_size":   swin.get("window_size", 8),
        "n_swin_blocks": swin.get("n_blocks", 2),
        "num_heads":     swin.get("num_heads", 4),
        "c_shared":      swin.get("c_shared", 64),
        "head_dropout":  swin.get("head_dropout", 0.1),
    }
    ones_fraction = hist_meta.get("ones_fraction") or 0.0106

    from train_models import build_finetune_model
    model, _loss, _metrics = build_finetune_model(
        base_model_path=str(base_path),
        finetune_cfg=finetune_cfg,
        ones_fraction=ones_fraction,
    )
    model.load_weights(str(ft_path))
    return model


def resolve_threshold(label_type: str, mode: str, source: str,
                      finetuned: bool, override: float | None,
                      eval_results_path: Path | None) -> float | None:
    """Pick the operative threshold for the lightning prediction map.

    Order:
      1. --threshold (manual override) if set.
      2. --eval_results JSON if explicitly passed.
      3. evaluation/eval_<run_tag>[_finetuned]/evaluation_results.json
         (the path evaluate_coalition.py writes by default).
      4. 0.5 fallback with a warning.
    """
    if label_type != "lightning":
        return None
    if override is not None:
        print(f"  Using manual threshold = {override:.3f}")
        return float(override)

    if eval_results_path is None:
        run_tag = f"{mode}_{source}"
        if finetuned:
            run_tag = f"{run_tag}_finetuned"
        eval_results_path = Path("evaluation") / f"eval_{run_tag}" \
                            / "evaluation_results.json"

    if eval_results_path.is_file():
        with open(eval_results_path) as f:
            data = json.load(f)
        thr = data.get("optimal_threshold")
        if thr is not None:
            print(f"  Using optimal_threshold = {thr:.3f} "
                  f"from {eval_results_path}")
            return float(thr)
        print(f"  WARNING: {eval_results_path} has no `optimal_threshold` "
              f"key; falling back to 0.5")
    else:
        print(f"  WARNING: {eval_results_path} not found; falling back to 0.5. "
              f"Run evaluate_coalition.py first or pass --threshold.")
    return 0.5


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Top-N reference-timestep full-domain visualisation.",
    )
    parser.add_argument("--csv", required=True, type=str,
                        help="Path to a per-source split CSV "
                             "(train/validation/test_data_<source>.csv).")
    parser.add_argument("--mode", required=True, type=str,
                        choices=["mtg_lightning", "mtg_radar",
                                 "mtg_radar_continuous",
                                 "mtg_opera_radar_only", "mtg_opera_mtgmr",
                                 "mtg_lightning_opera",
                                 "mtg_lightning_opera_occurrence"])
    parser.add_argument("--source", type=str, default="dbscan",
                        choices=["dbscan", "lightning"])
    parser.add_argument("--top_n", type=int, default=5,
                        help="How many of the highest-patch-count rows to "
                             "plot (default 5).")
    parser.add_argument("--data_root", type=str, default="./our_data")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--output_dir", type=str, default="./full_domain_plots")
    parser.add_argument("--finetuned", action="store_true",
                        help="Use coalition_<run_tag>_finetuned.keras "
                             "(rebuilt + load_weights, same trick as "
                             "evaluate_coalition).")
    parser.add_argument("--eval_results", type=str, default=None,
                        help="Path to an evaluation_results.json to read "
                             "optimal_threshold from. Default is "
                             "evaluation/eval_<run_tag>[_finetuned]/"
                             "evaluation_results.json.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual probability threshold for the lightning "
                             "prediction map (overrides optimal_threshold).")
    parser.add_argument("--batch_size", type=int, default=18,
                        help="Per-row batch size (default 18 = max patches).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: csv {csv_path} not found.")
        return 1
    data_root = Path(args.data_root)
    patches_dir = data_root / "patches"
    if not patches_dir.is_dir():
        print(f"ERROR: patches dir {patches_dir} not found "
              f"(run extract_patches.py first).")
        return 1

    mode_config = get_mode_config(args.mode)
    label_type = mode_config["label_type"]
    step_minutes = _load_step_minutes(data_root)

    run_tag = f"{args.mode}_{args.source}"
    artifact_tag = f"{run_tag}_finetuned" if args.finetuned else run_tag
    output_dir = Path(args.output_dir) / f"full_domain_{artifact_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Full-domain top-N visualisation")
    print("=" * 70)
    print(f"  CSV:         {csv_path}")
    print(f"  Mode:        {args.mode}  (label_type={label_type})")
    print(f"  Source:      {args.source}  "
          f"{'(finetuned)' if args.finetuned else '(base)'}")
    print(f"  Top N:       {args.top_n}")
    print(f"  Step (min):  {step_minutes}")
    print(f"  Output dir:  {output_dir}")

    threshold = resolve_threshold(
        label_type, args.mode, args.source, args.finetuned,
        args.threshold,
        Path(args.eval_results) if args.eval_results else None,
    )

    print(f"\nLoading model...")
    model = load_model_artifact(
        Path(args.model_dir), args.mode, args.source, args.finetuned,
    )
    print(f"  Loaded: {model.count_params():,} parameters")

    print(f"\nSelecting top {args.top_n} timesteps from {csv_path}")
    df_top = load_top_n_rows(csv_path, args.top_n)
    print(df_top[["date", "reference_utc", "n_patches"]].to_string(index=False))

    for rank, (_, row) in enumerate(df_top.iterrows(), start=1):
        date_str = row["date"]
        ref_utc = row["reference_utc"].strip()
        print(f"\n[{rank}/{len(df_top)}] {date_str} {ref_utc} UTC  "
              f"({row['n_patches']} qualifying patches)")

        inputs, valid_patches = build_batch_inputs(
            row, str(patches_dir), mode_config, step_minutes,
        )
        if not valid_patches:
            print(f"  No usable patches (all input timesteps were missing). "
                  f"Skipping.")
            continue
        n_valid = len(valid_patches)
        print(f"  Built input tensors for {n_valid} patches "
              f"({', '.join(str(p) for p in valid_patches)})")

        preds = model.predict(inputs, batch_size=args.batch_size, verbose=0)
        print(f"  Model output shape: {preds.shape}")

        gt_canvases = build_full_gt(
            row, str(patches_dir), mode_config, label_type, step_minutes,
        )
        pred_canvases = build_full_pred(preds, valid_patches, label_type)

        safe_ref = ref_utc.replace(":", "")
        out_png = output_dir / f"ts{rank:02d}_{date_str}_{safe_ref}.png"
        plot_full_domain(
            gt_canvases, pred_canvases, valid_patches, label_type,
            date_str=date_str, ref_utc=ref_utc,
            threshold=threshold, output_path=out_png,
            step_minutes=step_minutes,
        )
        print(f"  Saved -> {out_png}")

    print(f"\nDone. {len(df_top)} figure(s) written under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
