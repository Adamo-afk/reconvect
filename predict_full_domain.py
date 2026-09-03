"""
predict_full_domain.py
======================
THE inference-only script for COALITION-4 nowcasting.

Runs a trained model against reprojected full-domain fields WITHOUT
touching any training-pipeline artefact. In particular this script does
NOT read from `patch_index.csv`, from `{train,validation,test}_data_<source>.csv`,
or from the pre-extracted 256x256 `.npy` tiles under `our_data/patches/`.
Instead it slices patches on the fly from the full 1536x768 reprojected
canvas the moment inference runs.

Use this when you want to produce predictions for a NEW date without
running (or re-running) `identify_patches.py`, `extract_patch_seq_for_datasets.py`,
`extract_patches.py`, or the Czibula 80/10/10 split. The training-time
tooling stays untouched.

For a training-adjacent visualiser that ALSO renders ground-truth labels
alongside predictions (top-N reference selection driven by qualifying
patch counts in a split CSV, plus a highest-activity patch zoom-in), use
`visualize_gt_vs_pred.py` instead. That script assumes the
full pipeline has run and reads from the per-source split CSVs. This one
does not.

Minimum on-disk requirements (nothing else):
  our_data/reprojected_data/satellite_data/MTG/<var>/nc4_<date>-Romania_<var>/*.npy
  our_data/reprojected_data/opera_data/<var>/nc4_<date>-Romania_<var>/*.npy
  our_data/lightning_data/<var>/nc4_<date>-Romania_<var>/*.npy   (or the reprojected_data mirror)
  our_data/normalization_stats_<source>.json
  our_data/sequence_meta_<source>.json
  our_data/timestep_config.json
  models/coalition_<mode>_<source>.keras   (+ _finetuned.keras + history JSON for --finetuned)

Usage:
    # Full day of predictions at every reference step (96 timesteps at step=15)
    python predict_full_domain.py --mode mtg_lightning_opera_rainfall --source dbscan \
        --date 2026-06-30

    # One hour only (references :00 :15 :30 :45)
    python predict_full_domain.py --mode mtg_lightning_opera_rainfall --source dbscan \
        --date 2026-06-30 --hour 14

    # One specific reference time
    python predict_full_domain.py --mode mtg_lightning_opera_rainfall --source dbscan \
        --date 2026-06-30 --time 14:30

    # Custom range and the Swin fine-tuned model
    python predict_full_domain.py --mode mtg_lightning_opera_occurrence \
        --source dbscan --date 2026-06-30 \
        --start-time 12:00 --end-time 15:45 --finetuned
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import tensorflow as tf

from create_datasets import (
    get_mode_config,
    init_sequence_config,
    set_normalization_stats_path,
    LABEL_CHANNELS,
)
from extract_patches import (
    find_reprojected_file,
    load_reprojected,
    average_pool,
    _resolve_hhmm as snap_hhmm_to_product,
)
from visualize_gt_vs_pred import (
    PATCH_SIZE, N_PATCHES, H_FULL, W_FULL,
    get_patch_bounds,
    overlay_borders,
    _ensure_view_cached,
    _load_country_borders_pixels,
    _gt_kwargs_for,
    load_model_artifact,
    plot_full_domain_predictions_only,
)
import visualize_gt_vs_pred as _vf
from train_models import build_run_tag

from periods import normalization_stats_name
from pipeline_config import SOURCE, resolve_data_root, resolve_model_dir

def _mode_choices():
    """Every mode with trained weights, derived from the registries.

    Never a hardcoded literal: the list this replaces still named three
    modes that had been retired, and omitted ones that had been added,
    so a valid mode was rejected while dead names were accepted.
    """
    from train_lightning_kd import STUDENT_MODE
    from train_models import TRAINING_MODES
    return sorted(set(TRAINING_MODES) | {STUDENT_MODE})



# ============================================================================
# Variable -> product-group registry
# ============================================================================
# Mirrors the config dicts in create_datasets.py so we know which
# `find_reprojected_file_*` dispatcher to hit for each variable name.

_VARIABLE_TO_GROUP: dict[str, str] = {
    # LINET lightning (already on the Romania grid via read_kml)
    "density": "lightning", "current": "lightning", "occurrence": "lightning",
    # MTG FCI
    "vis_06": "satellite_MTG", "ir_38": "satellite_MTG",
    "ir_105": "satellite_MTG", "wv_63": "satellite_MTG", "wv_73": "satellite_MTG",
    # OPERA composite
    "opera_reflectivity": "opera", "opera_rainfall_rate": "opera",
}

INPUT_STEP_OFFSETS = [-2, -1, 0]      # t-2, t-1, t0 relative to reference
LEAD_STEP_OFFSETS = [1, 2, 3]         # t+1, t+2, t+3 (labels on the output side)


# ============================================================================
# Timestep helpers
# ============================================================================
def _load_step_minutes(data_root: Path) -> int:
    """Read step_minutes from our_data/timestep_config.json."""
    cfg_path = data_root / "timestep_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"{cfg_path} not found - run validate_timestep.py first."
        )
    with open(cfg_path) as f:
        return int(json.load(f)["step_minutes"])


def _all_step_aligned_times(step_minutes: int) -> list[str]:
    """Every reference HH:MM in a day at step_minutes cadence.
    step=15 -> ['00:00','00:15','00:30','00:45', ..., '23:45']."""
    result = []
    for hour in range(24):
        for minute in range(0, 60, step_minutes):
            result.append(f"{hour:02d}:{minute:02d}")
    return result


def _resolve_reference_times(args, step_minutes: int) -> list[str]:
    """Turn the mutually-exclusive time flags into a list of reference HH:MM."""
    if args.time is not None:
        return [args.time.strip()]
    if args.hour is not None:
        return [f"{int(args.hour):02d}:{m:02d}"
                for m in range(0, 60, step_minutes)]
    if args.start_time or args.end_time:
        start = args.start_time or "00:00"
        end = args.end_time or "23:59"
        all_times = _all_step_aligned_times(step_minutes)
        s_h, s_m = int(start[:2]), int(start[3:])
        e_h, e_m = int(end[:2]), int(end[3:])
        start_key = s_h * 60 + s_m
        end_key = e_h * 60 + e_m
        return [t for t in all_times
                if start_key <= (int(t[:2]) * 60 + int(t[3:])) <= end_key]
    return _all_step_aligned_times(step_minutes)


def _ref_to_hhmm(ref_utc: str, offset_min: int,
                 date_str: str) -> tuple[str, str]:
    """Turn (reference HH:MM, offset minutes, date) into (HHMM, date_str).
    Handles day rollover for the t-30/t+45 windows around midnight."""
    parts = ref_utc.split(":")
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=int(parts[0]), minute=int(parts[1])
    )
    target = base + timedelta(minutes=offset_min)
    return target.strftime("%H%M"), target.strftime("%Y-%m-%d")


# ============================================================================
# Full-domain patch assembly (no on-disk .npy patches required)
# ============================================================================
def _load_and_slice_patches(data_root: Path, variable: str, date_str: str,
                            hhmm: str, target_res: int) -> list[np.ndarray] | None:
    """Load the reprojected full-domain field, slice all 18 patches, pool
    down to target_res. Returns list of 18 arrays each shape (target_res,
    target_res) as float32, or None if the field file is missing."""
    group = _VARIABLE_TO_GROUP.get(variable)
    if group is None:
        return None
    hhmm_snapped = snap_hhmm_to_product(hhmm, group)
    path = find_reprojected_file(
        str(data_root), variable, group, date_str, hhmm_snapped,
    )
    if path is None:
        return None
    field = load_reprojected(path)          # (768, 1536), float32
    pool_factor = PATCH_SIZE // target_res

    patches = []
    for p in range(1, N_PATCHES + 1):
        r0, r1, c0, c1 = get_patch_bounds(p)
        tile = field[r0:r1, c0:c1]
        if pool_factor > 1:
            tile = average_pool(tile, pool_factor)
        patches.append(tile.astype(np.float32))
    return patches


def _build_group_batch(data_root: Path, mode_config: dict, group_key: str,
                      date_str: str, ref_utc: str,
                      step_minutes: int,
                      restrict_to_patches: list[int] | None = None,
                      ) -> tuple[np.ndarray, list[int]]:
    """Build the (N, T, H, W, C) input tensor for one input group across
    all 18 (or filtered) patches at one reference time.

    Missing variables at any timestep produce a zero-fill for that
    channel-timestep, matching create_datasets.load_and_transform_group.
    A patch is dropped only if EVERY variable at EVERY input step is
    missing (i.e. no data at all for that patch position across the
    whole 45-min window) - that mirrors the training-time behaviour
    where load_and_transform_group returns None when a whole variable
    is unavailable.

    Returns (batch, kept_patch_numbers).
    """
    group = mode_config[group_key]
    if group is None:
        return np.empty((0,), dtype=np.float32), []
    var_config, resolution, _suffix = group

    # For each input timestep, load one (patches x H x W) plate per variable.
    # Structure per timestep: dict[var_name -> list of 18 (H, W) arrays].
    per_step_channels: list[list[np.ndarray | None]] = []
    for offset in INPUT_STEP_OFFSETS:
        hhmm, day = _ref_to_hhmm(
            ref_utc, offset * step_minutes, date_str,
        )
        step_channels: list[np.ndarray | None] = []
        for var_name, (transform_fn, _extra) in var_config.items():
            patches = _load_and_slice_patches(
                data_root, var_name, day, hhmm, resolution,
            )
            if patches is None:
                step_channels.append(None)
                continue
            # Apply per-variable transform pixel-wise on each patch.
            transformed = []
            for tile in patches:
                out = transform_fn(tile)
                if out.ndim == 2:
                    out = out[:, :, np.newaxis]
                transformed.append(out.astype(np.float32))
            step_channels.append(np.stack(transformed, axis=0))   # (18, H, W, C_var)
        per_step_channels.append(step_channels)

    # Concatenate variables along channel dim per timestep, zero-filling
    # any that were missing (so the channel count stays constant).
    zero_shape_lookup: dict[int, tuple[int, int, int]] = {}
    for vi, plates in enumerate(zip(*per_step_channels)):
        for plate in plates:
            if plate is not None:
                zero_shape_lookup[vi] = plate.shape[1:]  # (H, W, C_var)
                break

    if not zero_shape_lookup:
        # No data anywhere for this group at this reference -> zero batch
        return np.empty((0,), dtype=np.float32), []

    # Reshape per_step_channels into (T, 18, H, W, C_group) with zero-fills
    T = len(per_step_channels)
    plates_per_var = list(zip(*per_step_channels))  # var -> tuple of T (18,H,W,C_var) or None

    # Pick sample var shape for the (H, W) size (all vars in a group share the same resolution)
    first_shape = next(iter(zero_shape_lookup.values()))
    H, W = first_shape[0], first_shape[1]

    n_valid_vars = len(plates_per_var)
    filled_per_var = []
    for vi, plates in enumerate(plates_per_var):
        c_var = zero_shape_lookup.get(vi, (H, W, 1))[2]
        filled_ts = []
        for t in range(T):
            if plates[t] is None:
                filled_ts.append(np.zeros((N_PATCHES, H, W, c_var),
                                          dtype=np.float32))
            else:
                filled_ts.append(plates[t])
        filled_per_var.append(np.stack(filled_ts, axis=0))  # (T, 18, H, W, C_var)

    stacked = np.concatenate(filled_per_var, axis=-1)  # (T, 18, H, W, C_group)
    # Reorder to (18, T, H, W, C_group) so we batch over patches
    batch = np.transpose(stacked, (1, 0, 2, 3, 4))

    all_patches = list(range(1, N_PATCHES + 1))
    if restrict_to_patches is not None:
        keep = [p for p in restrict_to_patches
                if 1 <= p <= N_PATCHES]
        idxs = [p - 1 for p in keep]
        return batch[idxs], keep
    return batch, all_patches


def build_inputs_for_reference(data_root: Path, mode_config: dict,
                               date_str: str, ref_utc: str,
                               step_minutes: int,
                               restrict_to_patches: list[int] | None = None,
                               ) -> tuple[dict[str, np.ndarray], list[int]]:
    """Assemble the model-ready inputs dict for one reference timestep.

    Every input group present in the mode config is built independently
    (each may resolve to a different resolution). The set of kept patches
    is the intersection - a patch survives only if every group produced
    a non-empty batch for it. This matches what the training pipeline
    would have kept.
    """
    inputs: dict[str, np.ndarray] = {}
    kept_sets: list[set[int]] = []
    per_group_batches: dict[str, tuple[np.ndarray, list[int]]] = {}
    for group_key in ("past_hr", "past_mr"):
        if mode_config.get(group_key) is None:
            continue
        batch, kept = _build_group_batch(
            data_root, mode_config, group_key,
            date_str, ref_utc, step_minutes,
            restrict_to_patches=restrict_to_patches,
        )
        if len(kept) == 0:
            return {}, []
        per_group_batches[group_key] = (batch, kept)
        kept_sets.append(set(kept))

    valid_patches = sorted(set.intersection(*kept_sets))
    if not valid_patches:
        return {}, []

    for group_key, (batch, kept) in per_group_batches.items():
        idxs = [kept.index(p) for p in valid_patches]
        inputs[group_key] = batch[idxs].astype(np.float32)
    return inputs, valid_patches


# ============================================================================
# Canvas + plotting
# ============================================================================
def paste_predictions_to_canvas(predictions: np.ndarray,
                                valid_patches: list[int],
                                label_type: str) -> list[np.ndarray]:
    """Project model predictions (N, T_future, H, W, C_out) onto three
    768x1536 canvases. lightning -> float probability; radar -> int class,
    -1 means "no data at this patch position"."""
    T_future = predictions.shape[1]
    canvases: list[np.ndarray] = []
    for t in range(T_future):
        if label_type == "lightning":
            canvas = np.zeros((H_FULL, W_FULL), dtype=np.float32)
        else:
            canvas = np.full((H_FULL, W_FULL), -1, dtype=np.int32)
        for p_pos, patch_num in enumerate(valid_patches):
            r0, r1, c0, c1 = get_patch_bounds(patch_num)
            pred_patch = predictions[p_pos, t]         # (256, 256, C)
            if label_type == "lightning":
                canvas[r0:r1, c0:c1] = pred_patch[..., 0]
            else:
                canvas[r0:r1, c0:c1] = np.argmax(pred_patch, axis=-1)
        canvases.append(canvas)
    return canvases


# ============================================================================
# Lightning post-processing helpers (Hann overlap + hysteresis + GT/overlay plot)
# ============================================================================
def _load_gt_rainfall_canvas(data_root: Path, date_str: str,
                             hhmm: str) -> np.ndarray | None:
    """Load the OPERA rainfall_rate reprojected field for one (date, HHMM).
    Returns raw mm/h array (768, 1536) float32, or None when missing. NaNs
    are turned into 0.0 so the caller can classify without extra masking.

    Duplicated (not imported) from validate_predictions to keep the two
    modules import-cycle-free — validate_predictions already imports from
    this file, so we cannot import from it at module scope.
    """
    from extract_patches import find_reprojected_file, load_reprojected
    path = find_reprojected_file(
        str(data_root), "opera_rainfall_rate", "opera", date_str, hhmm,
    )
    if path is None:
        return None
    field = load_reprojected(path)
    if field.ndim == 3:
        field = np.squeeze(field, axis=0)
    return np.where(np.isnan(field), 0.0, field).astype(np.float32)


def _print_prob_stats(prob_canvases: list[np.ndarray],
                      step_minutes: int,
                      low: float,
                      high_per_lead: dict[int, float]) -> None:
    """Emit per-lead raw-probability summary stats to stdout.

    Purpose: when the post-processed binary canvas comes out empty
    (`pred px=0` in the 2x3 subtitles), the reader has no way to tell
    from the plot whether the raw model output was uniformly near zero
    or was hovering just below the hysteresis LOW threshold. Printing
    max / mean / percentiles and the count of pixels crossing LOW gives
    that visibility without opening the .npy artefacts.
    """
    print("  Raw-prob stats per lead (before Hann-blend + hysteresis):")
    for k, offset in enumerate(LEAD_STEP_OFFSETS):
        p = prob_canvases[k]
        lead_min = offset * step_minutes
        p_max = float(p.max()); p_mean = float(p.mean())
        p_p99 = float(np.percentile(p, 99.0))
        p_p999 = float(np.percentile(p, 99.9))
        above_low = int((p >= low).sum())
        above_high = int((p >= high_per_lead[offset]).sum())
        print(f"    t+{lead_min:>3d}: max={p_max:.3f} mean={p_mean:.4f} "
              f"p99={p_p99:.3f} p99.9={p_p999:.3f} "
              f"px>={low:.2f}: {above_low}  px>={high_per_lead[offset]:.2f}: "
              f"{above_high}")


def _load_gt_lightning_canvas(data_root: Path, date_str: str,
                              hhmm: str) -> np.ndarray | None:
    """Load the LINET binary occurrence canvas for one (date, HHMM). Returns
    a (768, 1536) int8 array, or None when the file is not on disk.

    Lightning is emitted on the Romania grid natively (read_kml_version2
    bins strokes into GridProjection pixels), so we route the lookup through
    find_reprojected_file with group='lightning' and it walks the standard
    per-date directory tree - same discovery logic every other product uses.
    """
    from extract_patches import find_reprojected_file, load_reprojected  # local import: avoid circular
    path = find_reprojected_file(
        str(data_root), "occurrence", "lightning", date_str, hhmm,
    )
    if path is None:
        return None
    field = load_reprojected(path)
    if field.ndim == 3:
        field = np.squeeze(field, axis=0)
    # Occurrence maps come out as int8 (0/1) or float (0.0/1.0) depending on
    # the writer; normalise to int8.
    return (field > 0).astype(np.int8)


def _resolve_high_threshold_per_lead(
    args: argparse.Namespace,
    step_minutes: int,
) -> dict[int, float]:
    """Return {lead_offset_steps: high_threshold} for hysteresis.

    Priority:
      1. --validation_summary <path> -> read `high_threshold_per_lead`
         from the JSON produced by validate_predictions.py --track lightning.
         Values are indexed by "t+<minutes>" strings; we map them back to
         step-offsets via LEAD_STEP_OFFSETS + step_minutes.
      2. --lightning_high_threshold <x> -> same x used for every lead.
      3. Fallback: 0.95 for every lead (lightning_postproc.DEFAULT_HIGH_THRESHOLD).
    """
    from lightning_postproc import DEFAULT_HIGH_THRESHOLD  # local import: keep top clean
    if args.validation_summary:
        summary_path = Path(args.validation_summary)
        if not summary_path.is_file():
            raise SystemExit(
                f"--validation_summary points at a missing file: {summary_path}"
            )
        with open(summary_path) as f:
            summary = json.load(f)
        pp = summary.get("post_processing") or {}
        per_lead_named = pp.get("high_threshold_per_lead") or {}
        result: dict[int, float] = {}
        for offset in LEAD_STEP_OFFSETS:
            key = f"t+{offset * step_minutes}"
            if key not in per_lead_named:
                raise SystemExit(
                    f"validation summary {summary_path} is missing "
                    f"post_processing.high_threshold_per_lead[{key}]."
                )
            result[offset] = float(per_lead_named[key])
        return result
    if args.lightning_high_threshold is not None:
        return {offset: float(args.lightning_high_threshold)
                for offset in LEAD_STEP_OFFSETS}
    return {offset: DEFAULT_HIGH_THRESHOLD for offset in LEAD_STEP_OFFSETS}


def _plot_lightning_2x3(
    prob_canvases: list[np.ndarray],
    bin_canvases: list[np.ndarray],
    gt_canvases: list[np.ndarray | None],
    *,
    date_str: str,
    ref_utc: str,
    step_minutes: int,
    low: float,
    high_per_lead: dict[int, float],
    output_path: Path,
    suptitle_prefix: str = "Lightning inference",
    suptitle_color: str = "black",
) -> None:
    """Render the 2x3 lightning figure and save it to `output_path`.

    Row 1 (columns = t+15/+30/+45): GT lightning occurrence rendered in the
    same base "gt_red" colormap the training-scope visualiser uses. Where GT
    is not on disk, the panel is a blank canvas with a "GT unavailable" tag.

    Row 2 (columns = t+15/+30/+45): the SAME GT rendered underneath, then
    every post-processed (Hann + hysteresis) positive pixel drawn on top as
    opaque red. So: black-on-red pattern = miss (GT lit, pred missed), red
    on top of red-gradient = hit, red on background = false alarm.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from visualize_gt_vs_pred import _plot_patch_grid

    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT
    gt_kwargs = _gt_kwargs_for("lightning")

    fig, axes = plt.subplots(2, 3, figsize=(21, 8.5),
                             constrained_layout=True)

    def _apply_frame(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(c_lo, c_hi)
        ax.set_ylim(r_hi, r_lo)
        ax.set_aspect("equal")

    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        lead_min = offset * step_minutes
        lead_hhmm, _ = _ref_to_hhmm(ref_utc, lead_min, date_str)
        lead_wall = f"{lead_hhmm[:2]}:{lead_hhmm[2:]}"

        # ------------------------------ Row 1: GT
        ax_gt = axes[0, i]
        gt = gt_canvases[i]
        if gt is None:
            ax_gt.imshow(np.zeros((H_FULL, W_FULL), dtype=np.float32),
                         cmap="gray", vmin=0.0, vmax=1.0,
                         aspect="equal", interpolation="nearest")
            ax_gt.text(0.5, 0.5, "GT unavailable\n(no LINET file for this lead)",
                       transform=ax_gt.transAxes, ha="center", va="center",
                       fontsize=13, color="#555")
        else:
            ax_gt.imshow(gt.astype(np.float32), **gt_kwargs)
        _plot_patch_grid(ax_gt)
        try:
            overlay_borders(ax_gt)
        except Exception:
            pass
        _apply_frame(ax_gt)
        gt_pixels = int((gt > 0).sum()) if gt is not None else 0
        ax_gt.set_title(
            f"GT lightning occurrence - t+{lead_min} ({lead_wall} UTC)",
            fontsize=11,
        )

        # ------------------------------ Row 2: hits / misses / false-alarm
        # 3-colour palette on a plain white base (the gt_red cmap's
        # white low end, matching Row 1). Hits=orange, misses=blue,
        # false alarms=red; correct-dry pixels stay white.
        from validate_predictions import (
            _ZONE_ORANGE, _ZONE_BLUE, _ZONE_RED, _format_hmf_pct,
        )
        _GT_RED_LOW = (1.0, 1.0, 1.0, 1.0)   # white (was #fff5f0)
        ax_ov = axes[1, i]
        H, W = (gt.shape if gt is not None else (H_FULL, W_FULL))
        rgba = np.empty((H, W, 4), dtype=np.float32)
        rgba[:] = _GT_RED_LOW
        if gt is not None:
            gt_pos = gt > 0
            pr_pos = bin_canvases[i] > 0
            rgba[gt_pos & pr_pos] = _ZONE_ORANGE     # hit
            rgba[gt_pos & ~pr_pos] = _ZONE_BLUE      # miss
            rgba[~gt_pos & pr_pos] = _ZONE_RED       # false alarm
            hits = int((gt_pos & pr_pos).sum())
            misses = int((gt_pos & ~pr_pos).sum())
            false_alarms = int((~gt_pos & pr_pos).sum())
        else:
            # No GT: every post-processed positive counts as an FA-like
            # marker (no way to distinguish hit vs FA without truth).
            pr_pos = bin_canvases[i] > 0
            rgba[pr_pos] = _ZONE_RED
            hits = misses = false_alarms = 0
        ax_ov.imshow(rgba, aspect="equal", interpolation="nearest")
        _plot_patch_grid(ax_ov)
        try:
            overlay_borders(ax_ov)
        except Exception:
            pass
        _apply_frame(ax_ov)
        if gt is not None:
            subtitle = (
                f"Occurrence overlap  (low={low:.2f}, high={high_per_lead[offset]:.2f})\n"
                f"{_format_hmf_pct(hits, misses, false_alarms)}"
            )
        else:
            subtitle = (
                f"Occurrence overlap  (low={low:.2f}, high={high_per_lead[offset]:.2f})"
            )
        ax_ov.set_title(subtitle, fontsize=11)

    fig.suptitle(
        f"{suptitle_prefix}  |  {date_str}  ref={ref_utc}",
        fontsize=14, fontweight="bold", color=suptitle_color,
    )
    # 3-swatch colour legend (hit / miss / false alarm) + plain-language
    # percentage-formula footer, stacked below the plots.
    from validate_predictions import _add_hmf_legend, _add_zone_color_legend
    _add_zone_color_legend(fig)
    _add_hmf_legend(fig, y=-0.14)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_lightning_hits_1x3(
    bin_canvases: list[np.ndarray],
    gt_canvases: list[np.ndarray | None],
    *,
    date_str: str,
    ref_utc: str,
    step_minutes: int,
    output_path: Path,
) -> None:
    """Companion lightning-hits figure (1 row × 3 leads).

    Mirrors the rainfall `_plot_rainfall_perclass_hits_2x3` companion but
    for the binary lightning track — a single "did we detect it here?"
    view per lead. Only the pixels where BOTH GT and post-processed pred
    are active render (in the orange 'hit' colour from the zone palette);
    everything else stays white. Subtitle carries the hit count and the
    hit rate (hits / GT-active).

    Panels with no GT on disk render a placeholder — no hit computation
    possible without truth.
    """
    import matplotlib.pyplot as plt
    from validate_predictions import _ZONE_ORANGE
    from visualize_gt_vs_pred import _plot_patch_grid

    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5),
                             constrained_layout=True)

    def _apply_frame(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(c_lo, c_hi)
        ax.set_ylim(r_hi, r_lo)
        ax.set_aspect("equal")

    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        lead_min = offset * step_minutes
        lead_hhmm, _ = _ref_to_hhmm(ref_utc, lead_min, date_str)
        lead_wall = f"{lead_hhmm[:2]}:{lead_hhmm[2:]}"
        ax = axes[i]
        gt = gt_canvases[i]
        if gt is None:
            ax.imshow(np.zeros((H_FULL, W_FULL), dtype=np.float32),
                      cmap="gray", vmin=0.0, vmax=1.0,
                      aspect="equal", interpolation="nearest")
            ax.text(0.5, 0.5, "GT unavailable\n(no LINET file for this lead)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="#555")
            _plot_patch_grid(ax)
            _apply_frame(ax)
            ax.set_title(f"Hits - t+{lead_min} ({lead_wall} UTC)",
                         fontsize=11)
            continue
        H, W = gt.shape
        rgba = np.ones((H, W, 4), dtype=np.float32)   # white base
        gt_pos = gt > 0
        pr_pos = bin_canvases[i] > 0
        hits_mask = gt_pos & pr_pos
        rgba[hits_mask] = _ZONE_ORANGE
        ax.imshow(rgba, aspect="equal", interpolation="nearest")
        _plot_patch_grid(ax)
        try:
            overlay_borders(ax)
        except Exception:
            pass
        _apply_frame(ax)
        n_hits = int(hits_mask.sum())
        n_gt = int(gt_pos.sum())
        hit_rate = (100.0 * n_hits / n_gt) if n_gt > 0 else 0.0
        ax.set_title(
            f"Hits - t+{lead_min} ({lead_wall} UTC)\n"
            f"hits = {n_hits}  ({hit_rate:.1f}% of GT-active)",
            fontsize=11,
        )

    fig.suptitle(
        f"Lightning inference — hits only  |  {date_str}  ref={ref_utc}",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _build_rainfall_gt_class_canvas(
    gt_mmh: np.ndarray,
    valid_patches: list[int],
) -> np.ndarray:
    """Turn a raw OPERA mm/h canvas into a class canvas (-1 for
    out-of-domain slots) restricted to `valid_patches`. Shared by
    every rainfall-inference plotter in this module."""
    from validate_predictions import _mmh_to_class
    gt_full_cls = _mmh_to_class(gt_mmh).astype(np.int32)
    gt_canvas = np.full((H_FULL, W_FULL), -1, dtype=np.int32)
    for p in valid_patches:
        r0, r1, c0, c1 = get_patch_bounds(p)
        gt_canvas[r0:r1, c0:c1] = gt_full_cls[r0:r1, c0:c1]
    return gt_canvas


def _plot_rainfall_zone_prepost_3x3(
    pred_canvases: list[np.ndarray],
    hyst_canvases: list[np.ndarray],
    gt_rainfall_canvases: list[np.ndarray | None],
    valid_patches: list[int],
    *,
    date_str: str,
    ref_utc: str,
    step_minutes: int,
    output_path: Path,
    rainfall_low: float,
    rainfall_high: float,
) -> None:
    """Rainfall inference main figure (3 rows × 3 leads).

    Row 1 — GT class canvas (viridis-5) restricted to the valid patches.
    Row 2 — Zone overlap of the RAW argmax pred against GT (pre post-
            processing). Orange = hit, blue = miss, red = false alarm.
    Row 3 — Same zone overlap but computed against the HYSTERESIS-
            cleaned pred (`rainfall_hysteresis` with the caller-supplied
            low/high thresholds), so the reader can compare 'raw' vs
            'blob-cleaned' at a glance.

    A single viridis-5 colourbar labels Row 1's class palette on the
    right; Rows 2 and 3 share the standard zone colour legend +
    hit/miss/FA formula footer at the bottom of the figure. Panels
    without GT (missing OPERA file) render placeholders in every row.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    # Lazy imports keep the module cycle-free.
    from validate_predictions import (
        _plot_zone_overlap_axis, _add_zone_color_legend,
        _add_hmf_legend, _format_hmf_pct,
    )
    from visualize_gt_vs_pred import (
        _gt_kwargs_for, _plot_patch_grid, RADAR_CLASS_NAMES,
    )

    fig, axes = plt.subplots(3, 3, figsize=(21, 15),
                             constrained_layout=True)
    gt_kwargs = _gt_kwargs_for("radar")  # viridis-5, vmin=0, vmax=4

    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        lead_min = offset * step_minutes
        lead_hhmm, _ = _ref_to_hhmm(ref_utc, lead_min, date_str)
        lead_wall = f"{lead_hhmm[:2]}:{lead_hhmm[2:]}"
        pred_canvas = pred_canvases[i]
        hyst_canvas = hyst_canvases[i]
        gt_mmh = gt_rainfall_canvases[i]

        ax_r1 = axes[0, i]
        ax_r2 = axes[1, i]
        ax_r3 = axes[2, i]

        if gt_mmh is None:
            for ax in (ax_r1, ax_r2, ax_r3):
                ax.imshow(np.zeros((H_FULL, W_FULL), dtype=np.float32),
                          cmap="gray", vmin=0.0, vmax=1.0,
                          aspect="equal", interpolation="nearest")
                ax.text(0.5, 0.5, "GT unavailable\n(no OPERA file for this lead)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=12, color="#555")
                ax.set_xticks([]); ax.set_yticks([])
            ax_r1.set_title(f"GT class - t+{lead_min} ({lead_wall} UTC)",
                            fontsize=11)
            ax_r2.set_title(f"Zone overlap (pre) - t+{lead_min}",
                            fontsize=11)
            ax_r3.set_title(f"Zone overlap (post) - t+{lead_min}",
                            fontsize=11)
            continue

        gt_canvas = _build_rainfall_gt_class_canvas(gt_mmh, valid_patches)

        # --- Row 1: GT class canvas (viridis-5, NaN for out-of-domain).
        gt_display = np.where(gt_canvas < 0, np.nan, gt_canvas.astype(float))
        ax_r1.imshow(gt_display, **gt_kwargs)
        _plot_patch_grid(ax_r1)
        try:
            overlay_borders(ax_r1)
        except Exception:
            pass
        _ensure_view_cached()
        c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT
        ax_r1.set_xticks([]); ax_r1.set_yticks([])
        ax_r1.set_xlim(c_lo, c_hi); ax_r1.set_ylim(r_hi, r_lo)
        ax_r1.set_aspect("equal")
        ax_r1.set_title(
            f"GT class - t+{lead_min} ({lead_wall} UTC)",
            fontsize=11,
        )

        # --- Row 2: zone overlap (raw pred, pre post-processing).
        stats_pre = _plot_zone_overlap_axis(ax_r2, gt_canvas, pred_canvas)
        _plot_patch_grid(ax_r2)
        ax_r2.set_title(
            f"Zone overlap (pre post-proc) - t+{lead_min} "
            f"({lead_wall} UTC)\n"
            f"{_format_hmf_pct(stats_pre['hits'], stats_pre['misses'], stats_pre['false_alarms'])}",
            fontsize=10,
        )

        # --- Row 3: zone overlap after rainfall hysteresis.
        stats_post = _plot_zone_overlap_axis(ax_r3, gt_canvas, hyst_canvas)
        _plot_patch_grid(ax_r3)
        ax_r3.set_title(
            f"Post-processing (hysteresis) zone-overlap  "
            f"(low={rainfall_low:.2f}, high={rainfall_high:.2f}) - "
            f"t+{lead_min}\n"
            f"{_format_hmf_pct(stats_post['hits'], stats_post['misses'], stats_post['false_alarms'])}",
            fontsize=10,
        )

    # Row 1 gets its own viridis colourbar; Rows 2 and 3 share the
    # zone colour legend + hmf formula footer (no scalar colourbar,
    # the palette is categorical).
    sm = mcm.ScalarMappable(
        cmap=plt.get_cmap("viridis", 5),
        norm=mcolors.Normalize(vmin=0, vmax=4),
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes[0, :].ravel().tolist(),
        ticks=[0, 1, 2, 3, 4],
        shrink=0.75, pad=0.02, location="right",
    )
    cbar.set_ticklabels(RADAR_CLASS_NAMES)
    cbar.set_label("Rain-rate class (Row 1)")
    # Keep Rows 2 and 3 the same width as Row 1 under constrained_layout
    # by attaching same-shape invisible colourbars to them.
    for row in (1, 2):
        spacer = fig.colorbar(
            sm, ax=axes[row, :].ravel().tolist(),
            shrink=0.75, pad=0.02, location="right",
        )
        spacer.ax.set_visible(False)

    _add_zone_color_legend(fig)
    _add_hmf_legend(fig, y=-0.09)
    fig.suptitle(
        f"Rainfall inference — GT + zone overlap pre/post hysteresis  |  "
        f"{date_str}  ref={ref_utc}",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_rainfall_perclass_hits_2x3(
    pred_canvases: list[np.ndarray],
    hyst_canvases: list[np.ndarray],
    gt_rainfall_canvases: list[np.ndarray | None],
    valid_patches: list[int],
    *,
    date_str: str,
    ref_utc: str,
    step_minutes: int,
    output_path: Path,
    rainfall_low: float,
    rainfall_high: float,
) -> None:
    """Per-class hits companion figure (2 rows × 3 leads).

    Row 1 — Per-class hits using the RAW argmax pred (pre post-proc).
    Row 2 — Per-class hits using the hysteresis-cleaned pred.

    Each panel is `_plot_red_hits_axis`: post-processed GT rendered
    with a viridis-5 base and the pixels where pred_class == gt_class
    highlighted in their matching viridis colour. The per-class hit
    rate `C1:X% C2:Y% C3:Z% C4:W%` sits in the subtitle so the reader
    can see which rain-rate bands each variant recovered.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from validate_predictions import (
        _plot_red_hits_axis, _format_per_class_pct,
    )
    from visualize_gt_vs_pred import RADAR_CLASS_NAMES

    fig, axes = plt.subplots(2, 3, figsize=(21, 10),
                             constrained_layout=True)

    row_labels = [
        ("Per-class hits (pre post-proc)", pred_canvases, ""),
        ("Per-class hits (post-processing / hysteresis)", hyst_canvases,
         f"  (low={rainfall_low:.2f}, high={rainfall_high:.2f})"),
    ]

    for row_idx, (row_title, canvases_for_row, extra) in enumerate(row_labels):
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            lead_min = offset * step_minutes
            lead_hhmm, _ = _ref_to_hhmm(ref_utc, lead_min, date_str)
            lead_wall = f"{lead_hhmm[:2]}:{lead_hhmm[2:]}"
            gt_mmh = gt_rainfall_canvases[i]
            ax = axes[row_idx, i]
            if gt_mmh is None:
                ax.imshow(np.zeros((H_FULL, W_FULL), dtype=np.float32),
                          cmap="gray", vmin=0.0, vmax=1.0,
                          aspect="equal", interpolation="nearest")
                ax.text(0.5, 0.5,
                        "GT unavailable\n(no OPERA file for this lead)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=12, color="#555")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{row_title} - t+{lead_min}", fontsize=10)
                continue
            gt_canvas = _build_rainfall_gt_class_canvas(gt_mmh, valid_patches)
            red_stats = _plot_red_hits_axis(
                ax, gt_canvas, canvases_for_row[i],
            )
            ax.set_title(
                f"{row_title}{extra} - t+{lead_min} ({lead_wall} UTC)\n"
                f"{_format_per_class_pct(red_stats['per_class_pct'])}",
                fontsize=9,
            )

    # Shared viridis-5 colourbar (both rows use the same palette).
    sm = mcm.ScalarMappable(
        cmap=plt.get_cmap("viridis", 5),
        norm=mcolors.Normalize(vmin=0, vmax=4),
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(),
        ticks=[0, 1, 2, 3, 4],
        shrink=0.7, pad=0.02, location="right",
    )
    cbar.set_ticklabels(RADAR_CLASS_NAMES)
    cbar.set_label("Rain-rate class")

    fig.suptitle(
        f"Rainfall inference — per-class hits pre vs post hysteresis  |  "
        f"{date_str}  ref={ref_utc}",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="COALITION-4 inference-only script. Runs a trained "
                    "model against the reprojected full-domain fields "
                    "without touching training-pipeline artefacts.",
    )
    parser.add_argument("--mode", required=True, type=str,
                        choices=_mode_choices(),
                        help="Model variant. The name states its own track: "
                             "`_rainfall` for the OPERA 5-class head, "
                             "`_occurrence` for the lightning binary head.")
    parser.add_argument("--date", required=True, type=str,
                        help="Reference date (YYYY-MM-DD).")
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument("--time", type=str, default=None,
                            help="Single reference HH:MM.")
    time_group.add_argument("--hour", type=int, default=None,
                            help="Every step-aligned reference within one "
                                 "UTC hour (0..23).")
    time_group.add_argument("--start-time", type=str, default=None,
                            help="Range mode: inclusive lower HH:MM. "
                                 "Combine with --end-time; omits default to "
                                 "00:00 / 23:59.")
    parser.add_argument("--end-time", type=str, default=None,
                        help="Range mode: inclusive upper HH:MM.")
    parser.add_argument("--data_root", type=str, default=str(resolve_data_root()))
    parser.add_argument("--period", type=str, default=None, metavar="LABEL",
                        help="Period label the model was trained under, "
                             "e.g. --period w34. Selects the weights, the "
                             "normalization statistics and the sequence "
                             "metadata together. Omit for an untagged "
                             "whole-archive run.")
    parser.add_argument("--model_dir", type=str, default=str(resolve_model_dir()))
    parser.add_argument("--output_dir", type=str, default="./inference")
    parser.add_argument("--finetuned", action="store_true",
                        help="Load coalition_<run_tag>_finetuned.keras "
                             "(rebuilt + load_weights, same trick as "
                             "evaluate_coalition).")
    parser.add_argument("--kd", action="store_true",
                        help="Load the knowledge-distillation student weights "
                             "coalition_<run_tag>_kd.keras produced by "
                             "train_lightning_kd.py. Mirrors --finetuned; "
                             "the two flags are mutually exclusive (KD "
                             "student is trained fresh, no swin head). "
                             "Only meaningful for the student mode "
                             "mtg_opera_occurrence.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual probability threshold for the lightning "
                             "prediction map (defaults to 0.5 when the "
                             "evaluation_results.json is not present). "
                             "Ignored for lightning modes when the Hann-"
                             "overlap + hysteresis path is used (see "
                             "--lightning_low_threshold / "
                             "--lightning_high_threshold / "
                             "--validation_summary).")
    parser.add_argument("--batch_size", type=int, default=18,
                        help="Per-reference-time batch size passed to "
                             "model.predict (default 18 = max patches; "
                             "for lightning Hann overlap you may want to "
                             "raise this to 32 or 64 since ~55 positions "
                             "are batched per reference).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip PNG rendering; save raw predictions only.")
    parser.add_argument("--save-npy", action="store_true",
                        help="Save the raw prediction canvases to .npy "
                             "next to each PNG. For lightning modes, this "
                             "saves BOTH the raw probability canvas and the "
                             "post-processed binary canvas as separate files.")
    parser.add_argument("--patches", type=str, default=None,
                        help="Comma-separated 1-indexed patch numbers to "
                             "restrict inference to (default: all 18). "
                             "Ignored for lightning modes (Hann-overlap "
                             "always covers the full canvas).")
    # --- Lightning post-processing controls (ignored for radar/rainfall) ---
    parser.add_argument("--stride", type=int, default=None,
                        help="Overlap stride for the Hann-blended inference "
                             "path (lightning only). Default 128 = 50%% "
                             "overlap, giving 55 patches on a 768x1536 "
                             "canvas.")
    parser.add_argument("--lightning_low_threshold", type=float, default=None,
                        help="Hysteresis LOW threshold for lightning post-"
                             "processing. Default 0.90 (operational).")
    parser.add_argument("--lightning_high_threshold", type=float, default=None,
                        help="Hysteresis HIGH threshold applied to every "
                             "lead time (lightning only). Superseded by "
                             "--validation_summary if that flag is also "
                             "given. Default 0.95.")
    parser.add_argument("--validation_summary", type=str, default=None,
                        help="Path to the {track}_{yyyy}_{mm}_summary.json "
                             "produced by validate_predictions.py --track "
                             "lightning. When present, the per-lead tuned "
                             "high-threshold values are read from it and "
                             "override --lightning_high_threshold.")
    # --- Rainfall post-processing controls (ignored for lightning) ---
    # Defaults imported lazily below to avoid loading visualize_gt_vs_pred
    # at argparse-build time.
    parser.add_argument("--rainfall_low_threshold", type=float,
                        default=None,
                        help="Rainfall hysteresis LOW threshold on p(argmax) "
                             "when argmax is a rainy class. Default 0.35 — "
                             "lower than lightning because probability is "
                             "split across 5 classes.")
    parser.add_argument("--rainfall_high_threshold", type=float,
                        default=None,
                        help="Rainfall hysteresis HIGH threshold. "
                             "Default 0.55.")
    args = parser.parse_args()

    if args.kd and args.finetuned:
        parser.error("--kd and --finetuned are mutually exclusive "
                     "(the KD student has no swin head).")

    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    variant_suffix = ("_finetuned" if args.finetuned
                      else "_kd" if args.kd
                      else "")
    run_tag = build_run_tag(args.mode, SOURCE, args.period)
    output_dir = Path(args.output_dir) / (
        f"predict_{run_tag}{variant_suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    restrict_to_patches = None
    if args.patches:
        restrict_to_patches = [int(x) for x in args.patches.split(",")
                               if x.strip()]

    # 1. Init sequence config + normalization stats (per-source paths)
    init_sequence_config(str(data_root), SOURCE, period=args.period)
    set_normalization_stats_path(
        data_root / normalization_stats_name(SOURCE, args.period)
    )

    mode_config = get_mode_config(args.mode)
    label_type = mode_config["label_type"]
    step_minutes = _load_step_minutes(data_root)

    ref_times = _resolve_reference_times(args, step_minutes)

    print("=" * 70)
    print("COALITION-4 Inference (full-domain)")
    print("=" * 70)
    variant_label = ("finetuned" if args.finetuned
                     else "KD student" if args.kd
                     else "base")
    print(f"  Mode:            {args.mode}  (label_type={label_type})")
    print(f"  Source:          {SOURCE}  ({variant_label})")
    print(f"  Date:            {args.date}")
    print(f"  Reference times: {len(ref_times)} step-aligned slots "
          f"(step={step_minutes} min)")
    print(f"  Output dir:      {output_dir}")

    # Country borders + view extent
    _load_country_borders_pixels()  # populate cache eagerly
    _ensure_view_cached()

    # 2. Model
    # NOTE: we no longer call resolve_threshold() here. The legacy 0.5-
    # style sigmoid threshold has no consumer in this script: lightning
    # goes through Hann-blend + hysteresis (driven by
    # --lightning_low_threshold / --lightning_high_threshold /
    # --validation_summary) and rainfall uses the argmax over 5 classes.
    # --threshold survives only as an optional override for the raw-
    # probability heatmap's colormap centering.
    print("\nLoading model...")
    model = load_model_artifact(
        model_dir, args.mode, SOURCE, args.finetuned, kd=args.kd,
    )
    print(f"  Loaded: {model.count_params():,} parameters")

    # 3. Per-reference-time inference
    # Prepare lightning post-processing state once (no cost when unused).
    is_lightning = (label_type == "lightning")
    stride = args.stride if args.stride is not None else 128
    low_threshold = (args.lightning_low_threshold
                     if args.lightning_low_threshold is not None
                     else 0.90)
    if is_lightning:
        high_per_lead = _resolve_high_threshold_per_lead(args, step_minutes)
        print(f"  Lightning post-proc: stride={stride}  low={low_threshold:.2f}  "
              f"high per lead={{{', '.join(f't+{o*step_minutes}={h:.2f}' for o, h in high_per_lead.items())}}}")
        if restrict_to_patches is not None:
            print("  (ignoring --patches: Hann-overlap always covers the full canvas)")
        # Import here so radar/rainfall runs pay zero import cost.
        from lightning_postproc import (
            run_hann_overlapped_inference,
            hysteresis_binary,
        )

    for i, ref_utc in enumerate(ref_times, 1):
        print(f"\n[{i}/{len(ref_times)}] {args.date} {ref_utc} UTC")

        if is_lightning:
            # ---- Lightning: Hann-overlap inference + hysteresis + 2x3 plot
            prob_canvases = run_hann_overlapped_inference(
                model, data_root, mode_config, args.date, ref_utc,
                step_minutes, stride=stride, batch_size=args.batch_size,
            )
            if prob_canvases is None:
                print("  No reprojected data available for the input window. "
                      "Skipping.")
                continue
            bin_canvases = [
                hysteresis_binary(
                    prob_canvases[k], low=low_threshold,
                    high=high_per_lead[LEAD_STEP_OFFSETS[k]],
                )
                for k in range(len(prob_canvases))
            ]
            gt_canvases = []
            for offset in LEAD_STEP_OFFSETS:
                gt_hhmm, gt_day = _ref_to_hhmm(
                    ref_utc, offset * step_minutes, args.date,
                )
                gt_canvases.append(
                    _load_gt_lightning_canvas(data_root, gt_day, gt_hhmm)
                )
            print(f"  Predicted {len(prob_canvases)} lead(s); "
                  f"canvas shape {prob_canvases[0].shape}; "
                  f"GT panels available: "
                  f"{sum(1 for g in gt_canvases if g is not None)}/{len(gt_canvases)}")
            _print_prob_stats(prob_canvases, step_minutes,
                              low_threshold, high_per_lead)

            safe_ref = ref_utc.replace(":", "")
            out_stem = output_dir / f"predict_{args.date}_{safe_ref}"
            if not args.no_plot:
                _plot_lightning_2x3(
                    prob_canvases, bin_canvases, gt_canvases,
                    date_str=args.date, ref_utc=ref_utc,
                    step_minutes=step_minutes,
                    low=low_threshold, high_per_lead=high_per_lead,
                    output_path=out_stem.with_suffix(".png"),
                    suptitle_prefix="Lightning inference (Hann + hysteresis)",
                )
                print(f"  Saved plot -> {out_stem.with_suffix('.png').name}")
                # Companion hits-only figure (single row × 3 leads,
                # only pixels where GT and post-processed pred both
                # fire — mirrors the rainfall per-class hits sibling).
                hits_out = out_stem.parent / (
                    out_stem.name + "_hits.png"
                )
                _plot_lightning_hits_1x3(
                    bin_canvases, gt_canvases,
                    date_str=args.date, ref_utc=ref_utc,
                    step_minutes=step_minutes,
                    output_path=hits_out,
                )
                print(f"  Saved hits -> {hits_out.name}")
            if args.save_npy:
                # Per-lead per-artefact filenames so downstream tooling can
                # pick either one without re-splitting a stacked array.
                for k, offset in enumerate(LEAD_STEP_OFFSETS):
                    lead_min = offset * step_minutes
                    raw_path = out_stem.parent / (
                        out_stem.name + f"_prob_raw_t+{lead_min}.npy"
                    )
                    bin_path = out_stem.parent / (
                        out_stem.name + f"_bin_hyst_t+{lead_min}.npy"
                    )
                    np.save(raw_path, prob_canvases[k].astype(np.float32))
                    np.save(bin_path, bin_canvases[k].astype(np.int8))
                print(f"  Saved 2 x {len(prob_canvases)} .npy artefacts "
                      f"(*_prob_raw_t+*.npy and *_bin_hyst_t+*.npy)")
            continue

        # ---- Radar / rainfall: unchanged patch-based path
        inputs, valid_patches = build_inputs_for_reference(
            data_root, mode_config, args.date, ref_utc,
            step_minutes, restrict_to_patches=restrict_to_patches,
        )
        if not valid_patches:
            print("  No reprojected data available for the input window. "
                  "Skipping.")
            continue

        preds = model.predict(inputs, batch_size=args.batch_size, verbose=0)
        canvases = paste_predictions_to_canvas(preds, valid_patches, label_type)
        print(f"  Predicted {len(valid_patches)} patch(es); "
              f"canvases {canvases[0].shape}")

        # Rainfall-only: build the (H, W, 5) soft canvases + apply the
        # class-argmax hysteresis so both pre- and post-processing views
        # can be drawn side by side. Defaults for the two knobs live in
        # visualize_gt_vs_pred so the two scripts share the same
        # operational thresholds.
        hyst_canvases: list[np.ndarray] | None = None
        rainfall_low_used = rainfall_high_used = None
        if label_type == "radar":
            from visualize_gt_vs_pred import (
                build_full_soft_pred, rainfall_hysteresis,
                DEFAULT_RAIN_LOW, DEFAULT_RAIN_HIGH,
            )
            rainfall_low_used = (args.rainfall_low_threshold
                                 if args.rainfall_low_threshold is not None
                                 else DEFAULT_RAIN_LOW)
            rainfall_high_used = (args.rainfall_high_threshold
                                  if args.rainfall_high_threshold is not None
                                  else DEFAULT_RAIN_HIGH)
            soft_canvases = build_full_soft_pred(
                preds, valid_patches, n_classes=preds.shape[-1],
            )
            hyst_canvases = [
                rainfall_hysteresis(
                    soft_canvases[k],
                    low=rainfall_low_used,
                    high=rainfall_high_used,
                )
                for k in range(len(soft_canvases))
            ]
            print(f"  Rainfall hysteresis "
                  f"(low={rainfall_low_used:.2f}, "
                  f"high={rainfall_high_used:.2f}): "
                  f"selected px per lead = "
                  f"{[int(np.sum(c > 0)) for c in hyst_canvases]}")

        # Attempt to load OPERA GT for each lead time so the operator sees
        # the same "overlapping" structure/zoom figures validate_predictions
        # produces per lead. When GT is missing (typical for new-date
        # inference), fall back to the pred-only 1x3 heatmap.
        gt_class_canvases: list[np.ndarray | None] = []
        for offset in LEAD_STEP_OFFSETS:
            gt_hhmm, gt_day = _ref_to_hhmm(
                ref_utc, offset * step_minutes, args.date,
            )
            gt_class_canvases.append(
                _load_gt_rainfall_canvas(data_root, gt_day, gt_hhmm)
            )
        n_gt_avail = sum(1 for g in gt_class_canvases if g is not None)
        print(f"  OPERA GT panels available: {n_gt_avail}/{len(gt_class_canvases)}")

        safe_ref = ref_utc.replace(":", "")
        out_stem = output_dir / f"predict_{args.date}_{safe_ref}"
        if not args.no_plot:
            if label_type == "radar" and n_gt_avail > 0:
                # Main 3x3 figure: GT / zone overlap (raw pred) /
                # zone overlap (hysteresis-cleaned pred). Companion
                # 2x3 figure: per-class hits pre + post-processing.
                _plot_rainfall_zone_prepost_3x3(
                    canvases, hyst_canvases, gt_class_canvases,
                    valid_patches,
                    date_str=args.date, ref_utc=ref_utc,
                    step_minutes=step_minutes,
                    output_path=out_stem.with_suffix(".png"),
                    rainfall_low=rainfall_low_used,
                    rainfall_high=rainfall_high_used,
                )
                print(f"  Saved plot -> {out_stem.with_suffix('.png').name}")
                hits_out = out_stem.parent / (
                    out_stem.name + "_perclass_hits.png"
                )
                _plot_rainfall_perclass_hits_2x3(
                    canvases, hyst_canvases, gt_class_canvases,
                    valid_patches,
                    date_str=args.date, ref_utc=ref_utc,
                    step_minutes=step_minutes,
                    output_path=hits_out,
                    rainfall_low=rainfall_low_used,
                    rainfall_high=rainfall_high_used,
                )
                print(f"  Saved per-class hits -> {hits_out.name}")
            else:
                # No OPERA GT on disk (typical for a genuinely new date):
                # fall back to the pred-only 1x3 heatmap as the sole
                # output. No GT means no Row 2/Row 3/Zone content to
                # render anyway.
                plot_full_domain_predictions_only(
                    canvases, valid_patches, label_type,
                    date_str=args.date, ref_utc=ref_utc,
                    step_minutes=step_minutes,
                    threshold=args.threshold,
                    output_path=out_stem.with_suffix(".png"),
                    suptitle_prefix="Inference",
                )
                print(f"  Saved plot -> {out_stem.with_suffix('.png').name}")
        if args.save_npy:
            npy_path = out_stem.with_suffix(".npy")
            np.save(npy_path, np.stack(canvases, axis=0))
            print(f"  Saved canvases -> {npy_path.name}")
            if hyst_canvases is not None:
                hyst_npy = out_stem.parent / (out_stem.name + "_hyst.npy")
                np.save(hyst_npy, np.stack(hyst_canvases, axis=0))
                print(f"  Saved hyst canvases -> {hyst_npy.name}")

    print(f"\nDone. Wrote {len(ref_times)} reference time(s) to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
