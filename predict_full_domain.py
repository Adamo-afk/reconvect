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
    python predict_full_domain.py --mode mtg_lightning_opera --source dbscan \
        --date 2026-06-30

    # One hour only (references :00 :15 :30 :45)
    python predict_full_domain.py --mode mtg_lightning_opera --source dbscan \
        --date 2026-06-30 --hour 14

    # One specific reference time
    python predict_full_domain.py --mode mtg_lightning_opera --source dbscan \
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
    _ensure_view_cached,
    _load_country_borders_pixels,
    load_model_artifact,
    resolve_threshold,
    plot_full_domain_predictions_only,
)


# ============================================================================
# Variable -> product-group registry
# ============================================================================
# Mirrors the config dicts in create_datasets.py so we know which
# `find_reprojected_file_*` dispatcher to hit for each variable name.

_VARIABLE_TO_GROUP: dict[str, str] = {
    # Legacy ANM radar (RZC and friends)
    "RZC": "radar", "CZC": "radar", "EZC-20": "radar", "LZC": "radar",
    "BZC": "radar", "CPCH": "radar",
    # LINET lightning (already on the Romania grid via read_kml)
    "density": "lightning", "current": "lightning", "occurrence": "lightning",
    # MSG SEVIRI (currently disabled in the active build)
    "VIS006": "satellite_MSG", "IR_039": "satellite_MSG",
    "IR_108": "satellite_MSG", "WV_062": "satellite_MSG",
    "WV_073": "satellite_MSG",
    # MTG FCI
    "vis_06": "satellite_MTG", "ir_38": "satellite_MTG",
    "ir_105": "satellite_MTG", "wv_63": "satellite_MTG", "wv_73": "satellite_MTG",
    # OPERA composite
    "opera_reflectivity": "opera", "opera_rainfall_rate": "opera",
    # NWCSAF (dropped from active build but kept for completeness)
    "ctth_alti": "nwcsaf", "ctth_tempe": "nwcsaf",
    "cmic_phase": "nwcsaf", "cmic_cot": "nwcsaf",
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
    for group_key in ("past_hr", "past_mr", "past_lr"):
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
# CLI
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="COALITION-4 inference-only script. Runs a trained "
                    "model against the reprojected full-domain fields "
                    "without touching training-pipeline artefacts.",
    )
    parser.add_argument("--mode", required=True, type=str,
                        choices=["mtg_lightning", "mtg_radar",
                                 "mtg_radar_continuous",
                                 "mtg_opera_radar_only", "mtg_opera_mtgmr",
                                 "mtg_lightning_opera",
                                 "mtg_lightning_opera_occurrence"])
    parser.add_argument("--source", type=str, default="dbscan",
                        choices=["dbscan", "lightning"],
                        help="Selects normalization_stats_<source>.json and "
                             "the trained model file suffix.")
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
    parser.add_argument("--data_root", type=str, default="./our_data")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--output_dir", type=str, default="./inference")
    parser.add_argument("--finetuned", action="store_true",
                        help="Load coalition_<run_tag>_finetuned.keras "
                             "(rebuilt + load_weights, same trick as "
                             "evaluate_coalition).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual probability threshold for the lightning "
                             "prediction map (defaults to 0.5 when the "
                             "evaluation_results.json is not present).")
    parser.add_argument("--batch_size", type=int, default=18,
                        help="Per-reference-time batch size passed to "
                             "model.predict (default 18 = max patches).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip PNG rendering; save raw predictions only.")
    parser.add_argument("--save-npy", action="store_true",
                        help="Save the raw prediction canvases to .npy "
                             "next to each PNG.")
    parser.add_argument("--patches", type=str, default=None,
                        help="Comma-separated 1-indexed patch numbers to "
                             "restrict inference to (default: all 18).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) / (
        f"predict_{args.mode}_{args.source}"
        + ("_finetuned" if args.finetuned else "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    restrict_to_patches = None
    if args.patches:
        restrict_to_patches = [int(x) for x in args.patches.split(",")
                               if x.strip()]

    # 1. Init sequence config + normalization stats (per-source paths)
    init_sequence_config(str(data_root), args.source)
    set_normalization_stats_path(
        data_root / f"normalization_stats_{args.source}.json"
    )

    mode_config = get_mode_config(args.mode)
    label_type = mode_config["label_type"]
    step_minutes = _load_step_minutes(data_root)

    ref_times = _resolve_reference_times(args, step_minutes)

    print("=" * 70)
    print("COALITION-4 Inference (full-domain)")
    print("=" * 70)
    print(f"  Mode:            {args.mode}  (label_type={label_type})")
    print(f"  Source:          {args.source}  "
          f"{'(finetuned)' if args.finetuned else '(base)'}")
    print(f"  Date:            {args.date}")
    print(f"  Reference times: {len(ref_times)} step-aligned slots "
          f"(step={step_minutes} min)")
    print(f"  Output dir:      {output_dir}")

    # Country borders + view extent
    _load_country_borders_pixels()  # populate cache eagerly
    _ensure_view_cached()

    # 2. Threshold + model
    threshold = resolve_threshold(
        label_type, args.mode, args.source, args.finetuned,
        args.threshold, None,
    )
    print("\nLoading model...")
    model = load_model_artifact(
        model_dir, args.mode, args.source, args.finetuned,
    )
    print(f"  Loaded: {model.count_params():,} parameters")

    # 3. Per-reference-time inference
    for i, ref_utc in enumerate(ref_times, 1):
        print(f"\n[{i}/{len(ref_times)}] {args.date} {ref_utc} UTC")
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

        safe_ref = ref_utc.replace(":", "")
        out_stem = output_dir / f"predict_{args.date}_{safe_ref}"
        if not args.no_plot:
            # Shared renderer from visualize_gt_vs_pred so the pred
            # panels are pixel-identical to the bottom row of the 2x3
            # figure the training-scope script produces.
            plot_full_domain_predictions_only(
                canvases, valid_patches, label_type,
                date_str=args.date, ref_utc=ref_utc,
                step_minutes=step_minutes,
                threshold=threshold,
                output_path=out_stem.with_suffix(".png"),
                suptitle_prefix="Inference",
            )
            print(f"  Saved plot -> {out_stem.with_suffix('.png').name}")
        if args.save_npy:
            npy_path = out_stem.with_suffix(".npy")
            np.save(npy_path, np.stack(canvases, axis=0))
            print(f"  Saved canvases -> {npy_path.name}")

    print(f"\nDone. Wrote {len(ref_times)} reference time(s) to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
