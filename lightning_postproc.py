"""
lightning_postproc.py
=====================
Post-processing primitives for the lightning-occurrence prediction head:

  1. Overlapping inference with Hann blending
     - Enumerate patch positions at stride < patch_size across the
       768x1536 canvas (default stride 128 -> 5x11 = 55 positions).
     - Slice HR / MR input tiles at each position (MR is
       pooled versions of the same HR window, matching what the model
       consumed during training).
     - Run the model once per batch.
     - Multiply each patch prediction by a 2-D Hann window and
       accumulate a weighted-sum canvas + weight canvas. The final
       probability canvas is weighted_sum / weight_sum, which kills
       the 256-px tiling seams the non-overlapping paste path leaves.

  2. Hysteresis thresholding
     - A pixel is positive iff its probability is >= low AND is part
       of a connected component (8-neighbourhood) that also contains
       at least one pixel >= high. Implemented via scipy.ndimage.label
       + component filtering, which is O(H*W) and beats a per-pixel
       flood fill.

Both are used by predict_full_domain.py (operational) and by
validate_predictions.py (per-lead high-threshold tuning against LINET
observations).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import label as _cc_label

from extract_patches import (
    find_reprojected_file,
    load_reprojected,
    average_pool,
)
from predict_full_domain import (
    _VARIABLE_TO_GROUP,
    INPUT_STEP_OFFSETS,
    _ref_to_hhmm,
    sync_window_from_sequence_config,
)
from extract_patches import _resolve_hhmm as snap_hhmm_to_product


# ============================================================================
# Constants
# ============================================================================
DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE = 128            # 50% overlap -> 55 positions on 768x1536
DEFAULT_LOW_THRESHOLD = 0.90    # operational (matches --threshold in predict)
DEFAULT_HIGH_THRESHOLD = 0.95   # placeholder if no tuned value is supplied


# ============================================================================
# 2-D Hann window
# ============================================================================
def hann_window_2d(size: int = DEFAULT_PATCH_SIZE) -> np.ndarray:
    """2-D Hann window of shape (size, size).

    Uses np.hanning(size + 2)[1:-1] to drop the exact-zero endpoints of
    the reference Hann. The exact zeros would leave the very first and
    last row/column of each patch with zero blending weight, which is a
    problem for edge patches whose extremity is on the canvas boundary
    (no overlapping patch on that side, so those pixels would divide by
    zero in the weighted mean). The 1-pixel trim gives a tiny non-zero
    weight there.
    """
    w = np.hanning(size + 2)[1:-1].astype(np.float32)
    return np.outer(w, w)


# ============================================================================
# Position enumeration
# ============================================================================
def enumerate_positions(
    canvas_shape: tuple[int, int],
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> list[tuple[int, int]]:
    """Return the (r0, c0) top-left positions of every patch that tiles
    the canvas with the given stride. The last row/column are snapped so
    they end at H/W exactly - so an off-stride remainder is handled by a
    slightly-larger-than-stride overlap on the trailing edge (rather than
    dropping the last strip)."""
    H, W = canvas_shape
    if patch_size > H or patch_size > W:
        raise ValueError(
            f"patch_size {patch_size} exceeds canvas {canvas_shape}"
        )
    rows = list(range(0, H - patch_size + 1, stride))
    if rows[-1] != H - patch_size:
        rows.append(H - patch_size)
    cols = list(range(0, W - patch_size + 1, stride))
    if cols[-1] != W - patch_size:
        cols.append(W - patch_size)
    return [(r, c) for r in rows for c in cols]


# ============================================================================
# Overlapping input builder (analogue of build_inputs_for_reference)
# ============================================================================
def _load_full_field(data_root: Path, variable: str,
                     date_str: str, hhmm: str) -> np.ndarray | None:
    """Load the reprojected 768x1536 field for one variable at one HHMM.

    The variable's product group is looked up via predict_full_domain's
    _VARIABLE_TO_GROUP; hhmm is snapped to the product's native cadence
    before find_reprojected_file walks the disk layout. Returns None
    when the file is missing (missing-input policy identical to
    _build_group_batch: zero-fill that channel-timestep for every patch).
    """
    group = _VARIABLE_TO_GROUP.get(variable)
    if group is None:
        return None
    hhmm_snapped = snap_hhmm_to_product(hhmm, group)
    path = find_reprojected_file(
        str(data_root), variable, group, date_str, hhmm_snapped,
    )
    if path is None:
        return None
    field = load_reprojected(path)
    if field.ndim == 3:
        field = np.squeeze(field, axis=0)
    return field.astype(np.float32)


FIELD_READERS = 8      # threads reading one reference's input files


def read_fields_parallel(fn, jobs: list) -> list:
    """fn(job) for every job, in FIELD_READERS threads, results in job
    order. Used for the file reads of one reference."""
    if len(jobs) <= 1 or FIELD_READERS <= 1:
        return [fn(j) for j in jobs]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(FIELD_READERS, len(jobs))) as ex:
        return list(ex.map(fn, jobs))


def transform_full_field(field: np.ndarray, transform_fn, pool_factor: int
                         ) -> np.ndarray:
    """One variable's field, pooled to the group's resolution and
    transformed ONCE, as (h, w, C_var) float32. Every input transform is
    pixelwise with global statistics and the pooling averages fixed
    blocks, so slicing this field at a patch gives exactly what pooling
    and transforming the raw tile gave; the work is done once instead of
    once per position."""
    x = average_pool(field, pool_factor) if pool_factor > 1 else field
    out = transform_fn(x.astype(np.float32))
    if out.ndim == 2:
        out = out[:, :, np.newaxis]
    return out.astype(np.float32)


def _build_group_batch_overlapped(
    data_root: Path,
    mode_config: dict,
    group_key: str,
    date_str: str,
    ref_utc: str,
    step_minutes: int,
    positions: list[tuple[int, int]],
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> np.ndarray:
    """Analogue of predict_full_domain._build_group_batch for the
    overlapping-position layout.

    Loads each variable's full 768x1536 field ONCE per input timestep,
    then slices at all overlapping positions. Missing variables become
    zero-fills for that (channel, timestep) across every position, so
    the tensor's channel count matches training.
    """
    group = mode_config[group_key]
    if group is None:
        raise ValueError(f"group_key {group_key} is not present in mode config")
    var_config, resolution, _suffix = group

    N = len(positions)
    T = len(INPUT_STEP_OFFSETS)
    pool = patch_size // resolution
    var_names = list(var_config)

    # fields[t][var] -> the transformed (h, w, C_var) field, or None
    # when the file is missing at that timestep. The T x V files are
    # read in threads: reading and decompressing release the
    # interpreter lock, and the result is assembled by index, so the
    # order of completion does not matter.
    def _one(job):
        t, var_name, day, hhmm, transform_fn = job
        full_field = _load_full_field(data_root, var_name, day, hhmm)
        return (None if full_field is None
                else transform_full_field(full_field, transform_fn, pool))

    jobs = []
    for t, offset in enumerate(INPUT_STEP_OFFSETS):
        hhmm, day = _ref_to_hhmm(
            ref_utc, offset * step_minutes, date_str,
        )
        for var_name, (transform_fn, _extra) in var_config.items():
            jobs.append((t, var_name, day, hhmm, transform_fn))
    fields: list[dict[str, np.ndarray | None]] = [{} for _ in INPUT_STEP_OFFSETS]
    for job, out in zip(jobs, read_fields_parallel(_one, jobs)):
        fields[job[0]][job[1]] = out

    # Channel count per variable from its first available timestep
    # (one channel when it was never on disk, the zero-fill width).
    per_var_c: dict[str, int] = {}
    for vn in var_names:
        for t in range(T):
            arr = fields[t][vn]
            if arr is not None:
                per_var_c[vn] = arr.shape[-1]
                break
    if not per_var_c:
        # No data anywhere - caller decides what to do (usually skip).
        return np.empty((0,), dtype=np.float32)

    # Assemble (N, T, res, res, C_group) contiguously; a missing
    # (variable, timestep) stays zero.
    c_of = {vn: per_var_c.get(vn, 1) for vn in var_names}
    batch = np.zeros((N, T, resolution, resolution, sum(c_of.values())),
                     dtype=np.float32)
    ch0 = 0
    for vn in var_names:
        ch1 = ch0 + c_of[vn]
        for t in range(T):
            arr = fields[t][vn]
            if arr is None:
                continue
            for p, (r0, c0) in enumerate(positions):
                r, c = r0 // pool, c0 // pool
                batch[p, t, :, :, ch0:ch1] = arr[r:r + resolution, c:c + resolution]
        ch0 = ch1
    return batch


def build_inputs_for_reference_overlapped(
    data_root: Path,
    mode_config: dict,
    date_str: str,
    ref_utc: str,
    step_minutes: int,
    stride: int = DEFAULT_STRIDE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    canvas_shape: tuple[int, int] = (768, 1536),
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    """Build model inputs for every overlapping position on the canvas.

    Returns (inputs_dict, positions) where inputs_dict has "past_hr" and
    "past_mr" keys, each shaped (N_positions, T, H_res, W_res, C_group). `positions`
    is the list of (r0, c0) top-left coords, in the same order as the
    batch axis so paste_predictions_hann_blended can zip them back
    onto the canvas.

    Unlike the non-overlapping build_inputs_for_reference, this does
    NOT drop positions with missing inputs - it zero-fills them at the
    channel level so the (r0, c0) list stays aligned with the batch.
    That matches the operational-inference expectation ("give me a
    canvas even in the presence of some missing feeds") rather than
    the training-time "skip incomplete patches" behaviour.
    """
    # The window of the period in force (create_datasets holds it after
    # init_sequence_config); this module's copy of the offsets is synced
    # here so a script run as __main__ cannot leave it on the fallback.
    sync_window_from_sequence_config()
    positions = enumerate_positions(canvas_shape, patch_size, stride)
    inputs: dict[str, np.ndarray] = {}
    all_missing = True
    for group_key in ("past_hr", "past_mr"):
        if mode_config.get(group_key) is None:
            continue
        batch = _build_group_batch_overlapped(
            data_root, mode_config, group_key,
            date_str, ref_utc, step_minutes,
            positions, patch_size,
        )
        if batch.size == 0:
            # Not a single variable in this group was on disk at any
            # of the input timesteps - synthesise the expected shape
            # from the group config so the model still runs (with all
            # zeros). Rare in practice; keeps the operational path
            # robust to partial outages.
            var_config, resolution, _suffix = mode_config[group_key]
            n_chan = sum(1 for _ in var_config.items())
            batch = np.zeros(
                (len(positions), len(INPUT_STEP_OFFSETS),
                 resolution, resolution, n_chan),
                dtype=np.float32,
            )
        else:
            all_missing = False
        inputs[group_key] = batch

    if all_missing:
        return {}, []
    return inputs, positions


# ============================================================================
# Hann-weighted paste (analogue of paste_predictions_to_canvas for lightning)
# ============================================================================
def paste_predictions_hann_blended(
    predictions: np.ndarray,
    positions: list[tuple[int, int]],
    canvas_shape: tuple[int, int] = (768, 1536),
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> list[np.ndarray]:
    """Paste overlapping patch predictions onto per-lead canvases with
    Hann-window blending.

    Args:
        predictions: (N_positions, T_future, patch_size, patch_size, 1)
                     lightning-probability output of model.predict.
                     Any singleton channel is squeezed.
        positions:   list of (r0, c0) top-left coords aligned with the
                     batch axis, from build_inputs_for_reference_overlapped.
        canvas_shape: target canvas (H, W).
        patch_size:  patch side length (must match the model's input).

    Returns:
        list of T_future canvases (float32, shape canvas_shape), each
        with values in [0, 1] where at least one patch contributed and
        exactly 0 where no patch overlapped (which only happens if the
        position list is empty).
    """
    if predictions.ndim == 5 and predictions.shape[-1] == 1:
        predictions = predictions[..., 0]
    # predictions is now (N, T, H, W).
    T_future = predictions.shape[1]
    H, W = canvas_shape
    hann = hann_window_2d(patch_size)  # (patch, patch), non-negative

    canvases: list[np.ndarray] = []
    for t in range(T_future):
        acc = np.zeros((H, W), dtype=np.float32)
        wgt = np.zeros((H, W), dtype=np.float32)
        for p_pos, (r0, c0) in enumerate(positions):
            patch_pred = predictions[p_pos, t]  # (patch, patch)
            acc[r0:r0 + patch_size, c0:c0 + patch_size] += patch_pred * hann
            wgt[r0:r0 + patch_size, c0:c0 + patch_size] += hann
        canvas = np.zeros((H, W), dtype=np.float32)
        nonzero = wgt > 0.0
        canvas[nonzero] = acc[nonzero] / wgt[nonzero]
        canvases.append(canvas)
    return canvases


# ============================================================================
# Hysteresis thresholding
# ============================================================================
_STRUCT_8CONN = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)


def hysteresis_binary(
    prob: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Two-threshold binarisation with 8-neighbourhood connectivity.

    A pixel is positive iff prob >= low AND its 8-connected component
    (over the >= low mask) contains at least one pixel with prob >= high.
    This preserves weak-but-locally-anchored predictions while
    rejecting isolated below-high noise pixels.

    Args:
        prob: probability canvas, float, values roughly in [0, 1].
        low:  lower threshold. Common: 0.90 (operational).
        high: upper threshold. Common: tuned per lead by
              validate_predictions.py --track lightning.

    Returns:
        int8 canvas (0 or 1) with the same shape as `prob`. Guarantees:
        high > low; if the caller violates this an error is raised
        (both bounds equal is technically legal but reduces to a
        single-threshold binarisation, which caller probably didn't
        mean).
    """
    if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0):
        raise ValueError(f"low/high must be in [0, 1]; got low={low}, high={high}")
    if high < low:
        raise ValueError(
            f"hysteresis needs high >= low; got low={low}, high={high}"
        )
    mask_low = prob >= low
    seed = prob >= high
    if not seed.any():
        return np.zeros_like(prob, dtype=np.int8)
    labeled, _n = _cc_label(mask_low, structure=_STRUCT_8CONN)
    # Which components contain at least one seed pixel?
    seed_labels = np.unique(labeled[seed])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return np.zeros_like(prob, dtype=np.int8)
    keep = np.isin(labeled, seed_labels)
    return keep.astype(np.int8)


# ============================================================================
# Convenience wrapper
# ============================================================================
def run_hann_overlapped_inference(
    model,
    data_root: Path,
    mode_config: dict,
    date_str: str,
    ref_utc: str,
    step_minutes: int,
    *,
    stride: int = DEFAULT_STRIDE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    canvas_shape: tuple[int, int] = (768, 1536),
    batch_size: int = 16,
) -> list[np.ndarray] | None:
    """End-to-end: build overlapping inputs, run the model, Hann-blend
    the predictions into per-lead probability canvases.

    Returns a list of T_future float32 canvases (values roughly in
    [0, 1]), one per lead time in the order LEAD_STEP_OFFSETS produced.
    Returns None if not a single input variable was on disk at any of
    the input timesteps (no way to synthesise even zeros without
    knowing the group config).
    """
    inputs, positions = build_inputs_for_reference_overlapped(
        data_root, mode_config, date_str, ref_utc, step_minutes,
        stride=stride, patch_size=patch_size, canvas_shape=canvas_shape,
    )
    if not positions:
        return None
    preds = model.predict(inputs, batch_size=batch_size, verbose=0)
    canvases = paste_predictions_hann_blended(
        preds, positions, canvas_shape=canvas_shape, patch_size=patch_size,
    )
    return canvases
