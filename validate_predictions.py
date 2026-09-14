"""
validate_predictions.py
=======================
Validation branch for two tracks selectable via --track:

  RAINFALL (OPERA multiclass, structural + semantic coverage)
  ------------------------------------------------------------
  EXTRACTION MODE (no --date):
    - Scan every OPERA rainfall_rate .npy for the given (year, month).
    - Keep the sample if AT LEAST ONE pixel is >= 10 mm/h.
    - Run the model via the non-overlapping 18-patch path
      (predict_full_domain.build_inputs_for_reference +
      paste_predictions_to_canvas).
    - Two coverage metrics per (sample, lead time):
        * iou_mask -> IoU of the binary >=10 mm/h masks (structure)
        * class_wt -> per-class weighted overlap macro-averaged across
                      the 5 rainfall classes (semantic)
    - Aggregate FAR/POD/CSI per lead on the binary >=10 mm/h event.
    - Emits rainfall_<YYYY>_<MM>_{samples.csv, summary.json, metrics.png}.

  VISUALIZATION MODE (--date given):
    - Per selected reference, saves ONE figure per lead time (3 total):
      left = structure overlay (red pixels where GT class == Pred class
      AND both >= 10 mm/h); right = zoom on the 256x256 patch with
      the most GT-active pixels.
    - Suptitle colour: green if the lead cleared 90% coverage on either
      metric; orange if only in the initial selection.

  LIGHTNING (Hann-blended overlap + hysteresis, per-lead threshold tuning)
  -----------------------------------------------------------------------
  EXTRACTION MODE (no --date):
    - OPERA-driven sample selection (>=10 mm/h anywhere on the canvas at
      the reference timestep), SAME list as the rainfall track. This is a
      parity choice: identical selected references let coupling analysis
      and cross-track comparison line up cleanly. select_samples_lightning
      (LINET-driven, >=1 active pixel) remains available as a Python
      helper if a LINET-only cut is ever needed.
    - Run Hann-blended overlapping inference (default stride 128 -> 55
      patches, weights = 2-D Hann window). Yields a smooth probability
      canvas per lead, seam-free vs the non-overlapping paste path.
    - Per candidate high threshold in a 0.91..0.99 (step 0.01) grid,
      apply hysteresis (low=0.90 by default) and score confusion
      counts against LINET GT.
    - After all samples: pick the high that maximises aggregate CSI
      PER LEAD; persist the full sweep and the choices to the
      summary's `post_processing` block. predict_full_domain.py can
      consume that block via --validation_summary to get the same
      per-lead thresholds at inference time.
    - Emits lightning_<YYYY>_<MM>_{samples.csv, summary.json, metrics.png}
      (metrics figure: FAR/POD/CSI bars at chosen high + CSI sweep curves).

  VISUALIZATION MODE (--date given):
    - Per selected reference on the date, saves ONE 2x3 figure:
        Row 1 (t+15 / +30 / +45): GT lightning occurrence
        Row 2 (t+15 / +30 / +45): GT with post-processed positives
                                  overlaid in red.
      All three lead times on the same figure per user's spec.
    - Colour marker logged (green/orange) using the same convention
      as rainfall.

CLI examples:
    # Rainfall
    python validate_predictions.py --track rainfall --year 2025 --month 5
    python validate_predictions.py --track rainfall --year 2025 --month 5 --date 2025-05-14

    # Lightning (extraction tunes per-lead high threshold; viz reads it back)
    python validate_predictions.py --track lightning --year 2025 --month 5 \
        --mode mtg_lightning_opera_occurrence --source dbscan
    python validate_predictions.py --track lightning --year 2025 --month 5 --date 2025-05-14 \
        --mode mtg_lightning_opera_occurrence --source dbscan
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from compress_datasets import array_exists, list_arrays, load_array
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import tensorflow as tf

from create_datasets import (
    get_mode_config,
    init_sequence_config,
    set_normalization_stats_path,
    LABEL_CHANNELS,
    label_transform_opera_rainfall_multiclass,
)
from extract_patches import (
    find_reprojected_file,
    load_reprojected,
)
from visualize_gt_vs_pred import (
    PATCH_SIZE, N_PATCHES, H_FULL, W_FULL,
    RADAR_CLASS_NAMES,
    get_patch_bounds,
    overlay_borders,
    _ensure_view_cached,
    _load_country_borders_pixels,
    load_model_artifact,
)
import visualize_gt_vs_pred as _vf
from predict_full_domain import (
    build_inputs_for_reference,
    paste_predictions_to_canvas,
    sync_window_from_sequence_config,
    LEAD_STEP_OFFSETS,
    _load_step_minutes,
    _ref_to_hhmm,
    _load_gt_lightning_canvas,
    _plot_lightning_2x3,
)
from lightning_postproc import (
    DEFAULT_STRIDE, DEFAULT_LOW_THRESHOLD,
    build_inputs_for_reference_overlapped,
    paste_predictions_hann_blended,
    hysteresis_binary,
)

from periods import normalization_stats_name
from pipeline_config import SOURCE, resolve_data_root, resolve_model_dir


# Threshold above which a rainfall pixel is considered "active" convection.
# Same value across selection, coverage IoU, and the binary FAR/POD/CSI event.
RAINFALL_THRESHOLD_MMH = 10.0

# --max_samples: cap on the selected references per run, for trial runs
# on a small subset. None = every sample the month holds.
MAX_SAMPLES = None
# Per-month reproducible sampling under --max_samples: every month's
# candidates are shuffled with SEED and MONTH_BATCH are taken from each
# month in turn, cycling until MAX_SAMPLES, never repeating a sample.
MONTH_BATCH = 10
SEED = 1234
# CPU threads for the per-lead sweeps.
WORKERS = 4

# Scope of a run: a dataset split (train / validation / test, read from
# the split CSV), a month scanned from the reprojected archive, or a
# split restricted to one month. Set once in main(); the output stems
# and figure titles read it.
SPLITS = ("train", "validation", "test")
SPLIT: str | None = None
# Which saved state the run scores: the final save or the per-epoch
# checkpoint. Set in main(); the artifact tag says which.
WEIGHTS: str = "best"
# The selection threshold (--rainfall_threshold_mmh) is a property of
# the scored population too, so it is part of the scope and of every
# output name; runs at different thresholds never overwrite each other.
THRESHOLD_MMH: float = RAINFALL_THRESHOLD_MMH


def threshold_tag(mmh: float) -> str:
    """`thr8mmh`, `thr8.5mmh`."""
    return f"thr{mmh:g}mmh"


def run_dir(output_dir: Path, stem: str) -> Path:
    """The figure folder of one validation run, validation/<stem>/,
    created on demand. Only PNGs go there - the figures and the per-date
    overlays; the summary JSON and the samples CSV stay flat in
    validation/ next to it."""
    d = Path(output_dir) / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def scope_stem(year: int | None, month: int | None) -> str:
    """`2026_06_thr8mmh`, `test_thr8mmh`, `test_2026_06_thr8mmh`: the
    piece of every output name that says which samples a run scored."""
    parts = []
    if SPLIT:
        parts.append(SPLIT)
    if year is not None and month is not None:
        parts.append(f"{year:04d}_{month:02d}")
    parts.append(threshold_tag(THRESHOLD_MMH))
    return "_".join(parts)


def scope_label(year: int | None, month: int | None) -> str:
    """Human form of scope_stem for titles and messages."""
    if SPLIT and year is not None and month is not None:
        base = f"{SPLIT} split, {year:04d}-{month:02d}"
    elif SPLIT:
        base = f"{SPLIT} split"
    else:
        base = f"{year:04d}-{month:02d}"
    return f"{base}, >= {THRESHOLD_MMH:g} mm/h"

# Class boundaries mirror create_datasets.label_transform_opera_rainfall_multiclass:
#   class 0: R < 10  (below threshold)
#   class 1: 10-20   (>= threshold, weakest)
#   class 2: 20-30
#   class 3: 30-40
#   class 4: >= 40
N_RAINFALL_CLASSES = 5

# Coverage threshold that decides which samples land in the "high-coverage"
# lists in the JSON.
HIGH_COVERAGE_PCT = 90.0

# ---------------------------------------------------------------------------
# Rainfall hysteresis tuning
# ---------------------------------------------------------------------------
# Tuned on the validation split in two phases (see run_extraction): a
# plain-threshold sweep on p(argmax) picks LOW per lead, then (LOW, HIGH)
# windows slid around it pick the pair per lead. The scope is scored at
# that pair. Defaults of the window width and reach live on the CLI.

# The 18-patch grid the ensemble scorer accumulates over.
N_PATCHES = 18


def artifact_tag(mode: str, source: str, period=None, finetuned: bool = False,
                 kd: bool = False, baseline: bool = False) -> str:
    """The name every output of one model carries.

    <mode>_<source>[_<period>] plus _finetuned / _kd, or sepconv_<run_tag>
    for the baseline. build_patch_ensemble resolves member summaries by
    the same string, and a comparison across models keys on it.
    """
    from train_models import build_run_tag
    tag = build_run_tag(mode, source, period)
    if baseline:
        return f"sepconv_{tag}"
    return (tag + ("_finetuned" if finetuned else "_kd" if kd else "")
            + ("_latest" if WEIGHTS == "latest" else ""))


def _paste_class_canvases(classes_by_step: dict, valid_patches: list[int],
                          n_lead: int) -> list[np.ndarray]:
    """SepConv class maps (step -> (N, 256, 256)) onto full canvases,
    -1 where no patch was predicted, one canvas per lead."""
    canvases = []
    for i in range(n_lead):
        canvas = np.full((H_FULL, W_FULL), -1, dtype=np.int32)
        arr = classes_by_step.get(i + 1)
        if arr is not None:
            for p_pos, patch_num in enumerate(valid_patches):
                r0, r1, c0, c1 = get_patch_bounds(patch_num)
                canvas[r0:r1, c0:c1] = arr[p_pos]
        canvases.append(canvas)
    return canvases


def lead_palette(n: int) -> tuple[list[str], list[str]]:
    """(colours, markers) for n leads; the window decides n, not a constant."""
    import matplotlib.cm as _cm
    cmap = _cm.get_cmap("tab10")
    colours = [_cm.colors.to_hex(cmap(i % 10)) for i in range(n)]
    markers = [["o", "s", "^", "D", "v", "P", "X", "*"][i % 8] for i in range(n)]
    return colours, markers


_PATCH_ID_CANVAS: np.ndarray | None = None


def _patch_id_canvas() -> np.ndarray:
    """(H_FULL, W_FULL) int64 canvas holding patch index 0..N_PATCHES-1."""
    global _PATCH_ID_CANVAS
    if _PATCH_ID_CANVAS is None:
        from predict_full_domain import get_patch_bounds
        canvas = np.zeros((H_FULL, W_FULL), dtype=np.int64)
        for patch in range(1, N_PATCHES + 1):
            r0, r1, c0, c1 = get_patch_bounds(patch)
            canvas[r0:r1, c0:c1] = patch - 1
        _PATCH_ID_CANVAS = canvas
    return _PATCH_ID_CANVAS


def sweep_hysteresis(score: np.ndarray, low: float, highs,
                     gt: np.ndarray, eligible: np.ndarray | None = None,
                     ) -> tuple[dict, dict]:
    """Every HIGH candidate of a hysteresis sweep from one labelling.

    hysteresis_binary keeps a pixel iff score >= low and its 8-connected
    component over the `score >= low` mask holds a pixel with
    score >= high. The components do not depend on HIGH, and "holds a
    seed" is "the component's maximum score reaches HIGH", so the
    labelling and the per-(component, patch) TP/FP counts are done once
    and each candidate is a boolean over components. Results equal the
    per-candidate reference exactly.

    `gt` is the class / binary canvas with -1 marking pixels without
    model output (excluded from every count) and > 0 the positive event.
    `eligible` marks the pixels that count as positive when kept — for
    rainfall the rainy-argmax pixels (rainfall_hysteresis writes 0 for
    the rest), for lightning every pixel.

    Returns ({high: (TP, FP, FN, TN)},
             {high: {patch: (TP, FP, FN, TN)}} over patches with valid
             pixels), the shapes _binary_confusion and
             _accumulate_per_patch produce.
    """
    from scipy import ndimage as _ndi
    from lightning_postproc import _cc_label, _STRUCT_8CONN

    highs = [float(h) for h in highs]
    valid = gt >= 0
    gt_pos = (gt > 0) & valid
    if eligible is None:
        eligible = np.ones(gt.shape, dtype=bool)
    patch = _patch_id_canvas()
    valid_p = np.bincount(patch[valid], minlength=N_PATCHES)
    gtpos_p = np.bincount(patch[gt_pos], minlength=N_PATCHES)

    mask_low = score >= low
    labeled, n = _cc_label(mask_low, structure=_STRUCT_8CONN)
    if n:
        comp_max = np.asarray(
            _ndi.maximum(score, labeled, index=np.arange(1, n + 1)),
            dtype=score.dtype)
    else:
        comp_max = np.zeros(0, dtype=score.dtype)

    # TP / FP per (component, patch) over the pixels a kept component
    # would turn positive.
    sel = (labeled > 0) & eligible & valid
    key = labeled[sel] * N_PATCHES + patch[sel]
    is_tp = gt_pos[sel]
    size = (n + 1) * N_PATCHES
    tp_cp = np.bincount(key[is_tp], minlength=size).reshape(n + 1, N_PATCHES)
    fp_cp = np.bincount(key[~is_tp], minlength=size).reshape(n + 1, N_PATCHES)

    conf: dict = {}
    per_patch: dict = {}
    scored_patches = [q for q in range(N_PATCHES) if valid_p[q] > 0]
    for h in highs:
        kept = np.zeros(n + 1, dtype=bool)
        kept[1:] = comp_max >= h
        tp_p = tp_cp[kept].sum(axis=0)
        fp_p = fp_cp[kept].sum(axis=0)
        fn_p = gtpos_p - tp_p
        tn_p = valid_p - gtpos_p - fp_p
        conf[h] = (int(tp_p.sum()), int(fp_p.sum()),
                   int(fn_p.sum()), int(tn_p.sum()))
        per_patch[h] = {q + 1: (int(tp_p[q]), int(fp_p[q]),
                                int(fn_p[q]), int(tn_p[q]))
                        for q in scored_patches}
    return conf, per_patch


N_SCORE_BINS = 1000


def score_histograms(score: np.ndarray, gt: np.ndarray, eligible: np.ndarray,
                     n_bins: int = N_SCORE_BINS) -> tuple[np.ndarray, np.ndarray]:
    """(pos, neg) int64 histograms of `score` over the eligible, valid
    pixels, split by whether the event (gt > 0) occurred. A plain
    threshold t then keeps bins >= round(t * n_bins), so every threshold
    on the 1/n_bins grid is exact."""
    valid = (gt >= 0) & eligible
    idx = np.clip((score[valid] * n_bins).astype(np.int64), 0, n_bins - 1)
    truth = gt[valid] > 0
    pos = np.bincount(idx[truth], minlength=n_bins)
    neg = np.bincount(idx[~truth], minlength=n_bins)
    return pos, neg


def csi_from_histograms(pos: np.ndarray, neg: np.ndarray,
                        thresholds, gt_positive_total: float) -> np.ndarray:
    """Pooled CSI at each threshold from the histograms. `gt_positive_total`
    is the count of event pixels among ALL valid pixels (eligible or not):
    event pixels the model called dry are misses at every threshold."""
    n_bins = pos.shape[0]
    k = np.clip(np.rint(np.asarray(thresholds, dtype=np.float64) * n_bins)
                .astype(np.int64), 0, n_bins)
    pos_tail = np.concatenate([np.cumsum(pos[::-1])[::-1], [0]]).astype(np.float64)
    neg_tail = np.concatenate([np.cumsum(neg[::-1])[::-1], [0]]).astype(np.float64)
    tp = pos_tail[k]
    fp = neg_tail[k]
    fn = float(gt_positive_total) - tp
    return tp / (tp + fp + fn + 1e-7)


def window_pairs(low: float, width: float, reach: float) -> list[tuple[float, float]]:
    """The (LOW, HIGH) candidates around a phase-1 LOW: windows of
    `width` sliding left and right out to `reach`. For 0.35, 0.02, 0.10:
    (0.25, 0.27) ... (0.33, 0.35) on the left, (0.35, 0.37) ...
    (0.43, 0.45) on the right. Pairs leaving (0, 1) are dropped."""
    n = int(round(reach / width))
    pairs = []
    for k in range(n, 0, -1):
        pairs.append((round(low - k * width, 4), round(low - (k - 1) * width, 4)))
    for k in range(1, n + 1):
        pairs.append((round(low + (k - 1) * width, 4), round(low + k * width, 4)))
    return [(lo, hi) for lo, hi in pairs if lo > 0.0 and hi < 1.0]


def _map_leads(fn, n: int):
    """fn(i) for i in range(n), in WORKERS threads; returns the list of
    results in lead order. The per-lead sweeps are numpy / scipy heavy
    and release the interpreter lock, so they overlap."""
    if WORKERS <= 1 or n <= 1:
        return [fn(i) for i in range(n)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(WORKERS, n)) as ex:
        return list(ex.map(fn, range(n)))


def save_per_sample(path: Path, **arrays) -> None:
    """Every per-sample count of a run, compressed, next to the summary,
    so the figures can be redrawn and new ones made without re-running."""
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"  Wrote per-sample data to {path}")


def _merge_patch_counts(acc: dict, lead_idx: int, per_patch: dict) -> None:
    """Add one sample's per-patch (TP, FP, FN, TN) into the pooled
    accumulator, the structure _accumulate_per_patch builds."""
    for patch, (tp, fp, fn, tn) in per_patch.items():
        cell = acc.setdefault(patch, {}).setdefault(
            lead_idx, {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "n": 0})
        cell["TP"] += tp
        cell["FP"] += fp
        cell["FN"] += fn
        cell["TN"] += tn
        cell["n"] += 1


def _prefetch(items, load, ahead: int = 2):
    """Yield (item, load(item)) in order, with `load` of the next `ahead`
    items running in a background thread. The loader does disk reads,
    decompression and numpy work only, so it overlaps with the GPU call
    in the main thread."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as ex:
        it = iter(items)
        pending: deque = deque()
        for item in it:
            pending.append((item, ex.submit(load, item)))
            if len(pending) >= ahead:
                break
        while pending:
            item, fut = pending.popleft()
            for nxt in it:
                pending.append((nxt, ex.submit(load, nxt)))
                break
            yield item, fut.result()



def _predict_batches(model, inputs: dict, batch_size: int) -> np.ndarray:
    """model.predict_on_batch over explicit chunks of a dict of arrays.

    Keras' predict() wraps every call in a data adapter and traces a
    new graph the first time; on a single 18-patch batch that costs
    more than the network does. predict_on_batch is the plain forward
    pass, so the loop calls it chunk by chunk and concatenates.
    """
    n = len(next(iter(inputs.values())))
    outs = []
    for start in range(0, n, batch_size):
        chunk = {k: a[start:start + batch_size] for k, a in inputs.items()}
        outs.append(np.asarray(model.predict_on_batch(chunk)))
    return outs[0] if len(outs) == 1 else np.concatenate(outs, axis=0)

def _accumulate_per_patch(gt_bin: np.ndarray, pred_bin: np.ndarray,
                          acc: dict, lead_idx: int) -> None:
    """Pool contingency counts per patch for one lead time.

    Scored on the POST-PROCESSED canvases, so a member is judged on the
    product actually shipped rather than on raw model output. Counts are
    pooled rather than averaged because CSI is not additive.
    """
    from predict_full_domain import get_patch_bounds

    for patch in range(1, N_PATCHES + 1):
        r0, r1, c0, c1 = get_patch_bounds(patch)
        g = gt_bin[r0:r1, c0:c1]
        p = pred_bin[r0:r1, c0:c1]
        valid = g >= 0                      # -1 marks an unfilled patch slot
        if not np.any(valid):
            continue
        g_pos = (g > 0) & valid
        p_pos = (p > 0) & valid
        cell = acc.setdefault(patch, {}).setdefault(
            lead_idx, {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "n": 0})
        cell["TP"] += int(np.count_nonzero(g_pos & p_pos))
        cell["FP"] += int(np.count_nonzero(~g_pos & p_pos & valid))
        cell["FN"] += int(np.count_nonzero(g_pos & ~p_pos))
        cell["TN"] += int(np.count_nonzero(~g_pos & ~p_pos & valid))
        cell["n"] += 1


def per_patch_scores(acc: dict) -> dict:
    """Collapse pooled per-patch counts into CSI, POD and FAR."""
    eps = 1e-7
    out: dict[str, dict] = {}
    for patch, leads in sorted(acc.items()):
        tp = sum(c["TP"] for c in leads.values())
        fp = sum(c["FP"] for c in leads.values())
        fn = sum(c["FN"] for c in leads.values())
        per_lead = {
            str(i): round(c["TP"] / (c["TP"] + c["FP"] + c["FN"] + eps), 6)
            for i, c in sorted(leads.items())
        }
        out[str(patch)] = {
            "csi": round(tp / (tp + fp + fn + eps), 6),
            "pod": round(tp / (tp + fn + eps), 6),
            "far": round(fp / (tp + fp + eps), 6),
            "csi_per_lead": per_lead,
            "n_samples": max((c["n"] for c in leads.values()), default=0),
            "TP": tp, "FP": fp, "FN": fn,
        }
    return out


# ============================================================================
# OPERA sample selection
# ============================================================================
_OPERA_FILENAME_RE = re.compile(
    r"^nc4_(\d{4}-\d{2}-\d{2})-Romania_(\d{4})_rainfall_rate\.npy$"
)


def _iter_opera_files(data_root: Path, year: int, month: int):
    """Yield (date_str, hhmm, path) for every OPERA rainfall_rate .npy
    matching the given (year, month). Uses the folder-per-day layout
    from extract_patches: nc4_YYYY-MM-DD-Romania_rainfall_rate/*.npy."""
    root = data_root / "reprojected_data" / "opera_data" / "rainfall_rate"
    if not root.is_dir():
        raise FileNotFoundError(
            f"OPERA reprojected root not found: {root}. "
            f"Run reproject.py --opera first."
        )
    date_prefix = f"nc4_{year:04d}-{month:02d}-"
    for day_folder in sorted(root.iterdir()):
        if not day_folder.is_dir() or not day_folder.name.startswith(date_prefix):
            continue
        for name in list_arrays(day_folder):
            m = _OPERA_FILENAME_RE.match(name)
            if m is None:
                continue
            yield m.group(1), m.group(2), day_folder / name


def _opera_file_for(data_root: Path, date_str: str, hhmm: str) -> Path:
    """The reprojected OPERA rain-rate file of one reference timestep."""
    return (data_root / "reprojected_data" / "opera_data" / "rainfall_rate"
            / f"nc4_{date_str}-Romania_rainfall_rate"
            / f"nc4_{date_str}-Romania_{hhmm}_rainfall_rate.npy")


def split_references(data_root: Path, source: str, period,
                     split: str, year: int | None = None,
                     month: int | None = None) -> list[tuple[str, str]]:
    """Distinct (date, hhmm) reference timesteps of a dataset split,
    read from its CSV (<split>_data_<source>[_<period>].csv), optionally
    restricted to one month. These are the timesteps the split's
    samples were cut from, so scoring them is scoring the split."""
    from periods import split_csv_name
    path = data_root / split_csv_name(split, source, period)
    if not path.is_file():
        raise FileNotFoundError(
            f"split CSV not found: {path}. Build the datasets for this "
            f"source/period first (create_datasets.py).")
    refs: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date_str = row["date"].strip()
            if year is not None and month is not None:
                if not date_str.startswith(f"{year:04d}-{month:02d}-"):
                    continue
            refs.add((date_str, row["reference_utc"].strip().replace(":", "")))
    print(f"  {split} split: {len(refs)} reference timesteps in {path.name}"
          + (f" for {year:04d}-{month:02d}" if year is not None and month is not None
             else ""))
    return sorted(refs)


def subsample_by_month(kept: list[tuple[str, str]], max_samples: int,
                       month_batch: int = MONTH_BATCH, seed: int = SEED,
                       ) -> list[tuple[str, str]]:
    """Reproducible per-month sampling: shuffle each month's candidates
    with `seed`, then take `month_batch` from every month in turn,
    cycling over the months until `max_samples` are drawn. No sample is
    drawn twice, months with fewer candidates drop out when exhausted,
    and the result is returned in chronological order."""
    if max_samples is None or len(kept) <= max_samples:
        return kept
    rng = np.random.default_rng(seed)
    by_month: dict[str, list] = {}
    for d, h in kept:
        by_month.setdefault(d[:7], []).append((d, h))
    queues = []
    for m in sorted(by_month):
        items = by_month[m]
        rng.shuffle(items)
        queues.append(items)
    chosen: list[tuple[str, str]] = []
    while len(chosen) < max_samples and any(queues):
        for q in queues:
            take = q[:month_batch]
            del q[:month_batch]
            chosen.extend(take)
            if len(chosen) >= max_samples:
                break
    chosen = chosen[:max_samples]
    print(f"  --max_samples {max_samples}: {len(chosen)} drawn from "
          f"{len(by_month)} month(s), {month_batch} per month per round, "
          f"seed {seed}")
    return sorted(chosen)


def select_samples(data_root: Path, year: int | None, month: int | None,
                   threshold_mmh: float = RAINFALL_THRESHOLD_MMH,
                   *, source: str | None = None, period=None,
                   split: str | None = None, limit: int | None = -1,
                   ) -> list[tuple[str, str]]:
    """The reference timesteps a run scores, with at least one pixel
    >= threshold: every OPERA sample of the month, or, when a split is
    set (`split`, else the run's SPLIT), the split's own timesteps
    (within the month if one is given). `limit` caps them with the
    per-month sampling; -1 means MAX_SAMPLES. Returns (date_str, hhmm)
    tuples sorted chronologically."""
    split = split or SPLIT
    if limit == -1:
        limit = MAX_SAMPLES
    if split:
        if source is None:
            raise ValueError("select_samples needs `source` on a split run")
        candidates = []
        missing = 0
        for date_str, hhmm in split_references(data_root, source, period,
                                               split, year, month):
            path = _opera_file_for(data_root, date_str, hhmm)
            if array_exists(path):
                candidates.append((date_str, hhmm, path))
            else:
                missing += 1
        if missing:
            print(f"  NOTE: {missing} split timestep(s) have no reprojected "
                  f"OPERA file and are skipped.")
    else:
        if year is None or month is None:
            raise ValueError("select_samples needs a year and month "
                             "without a split")
        candidates = _iter_opera_files(data_root, year, month)
    kept: list[tuple[str, str]] = []
    scanned = 0
    for date_str, hhmm, path in candidates:
        scanned += 1
        data = load_array(path)
        if data.ndim == 3:
            data = np.squeeze(data, axis=0)
        # NaN -> 0 to mirror the label transform in create_datasets.
        finite_max = np.nanmax(data) if data.size else 0.0
        if finite_max >= threshold_mmh:
            kept.append((date_str, hhmm))
    print(f"  Scanned {scanned} OPERA files; "
          f"kept {len(kept)} with >= {threshold_mmh:g} mm/h")
    return subsample_by_month(kept, limit, MONTH_BATCH, SEED)


# ============================================================================
# GT loading and canvas assembly
# ============================================================================
def _load_gt_rainfall_canvas(data_root: Path, date_str: str,
                             hhmm: str) -> np.ndarray | None:
    """Load the OPERA rainfall_rate reprojected field for a single (date,
    hhmm). Returns the raw 768x1536 mm/h array, or None if missing."""
    path = find_reprojected_file(
        str(data_root), "opera_rainfall_rate", "opera", date_str, hhmm,
    )
    if path is None:
        return None
    field = load_reprojected(path)
    if field.ndim == 3:
        field = np.squeeze(field, axis=0)
    return np.where(np.isnan(field), 0.0, field).astype(np.float32)


def _mmh_to_class(mmh: np.ndarray) -> np.ndarray:
    """Vectorised version of label_transform_opera_rainfall_multiclass,
    returning integer class indices instead of one-hot."""
    mmh = np.where(np.isnan(mmh), 0.0, mmh)
    mmh = np.clip(mmh, 0.0, None)
    cls = np.zeros_like(mmh, dtype=np.int32)
    cls[(mmh >= 10.0) & (mmh < 20.0)] = 1
    cls[(mmh >= 20.0) & (mmh < 30.0)] = 2
    cls[(mmh >= 30.0) & (mmh < 40.0)] = 3
    cls[mmh >= 40.0] = 4
    return cls


def _paste_gt_class_canvas(gt_field: np.ndarray | None,
                           valid_patches: list[int]) -> np.ndarray:
    """Turn a full 768x1536 mm/h field into a per-pixel class-index
    canvas, restricted to the valid patch slots (others = -1 for
    'no model output here')."""
    canvas = np.full((H_FULL, W_FULL), -1, dtype=np.int32)
    if gt_field is None:
        return canvas
    cls = _mmh_to_class(gt_field)
    for p in valid_patches:
        r0, r1, c0, c1 = get_patch_bounds(p)
        canvas[r0:r1, c0:c1] = cls[r0:r1, c0:c1]
    return canvas


# ============================================================================
# Coverage metrics per (sample, lead time)
# ============================================================================
def _binary_confusion(gt_cls: np.ndarray, pred_cls: np.ndarray
                      ) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) on the binary >=10 mm/h event, over the
    pixels where GT has a valid class (not -1)."""
    valid = gt_cls != -1
    gt_pos = (gt_cls >= 1) & valid
    pred_pos = (pred_cls >= 1) & valid
    tp = int(np.sum(gt_pos & pred_pos))
    fp = int(np.sum(~gt_pos & pred_pos & valid))
    fn = int(np.sum(gt_pos & ~pred_pos))
    tn = int(np.sum(~gt_pos & ~pred_pos & valid))
    return tp, fp, fn, tn


def _iou_binary(gt_cls: np.ndarray, pred_cls: np.ndarray) -> float:
    """IoU of the binary >=10 mm/h masks over the whole valid region."""
    tp, fp, fn, _ = _binary_confusion(gt_cls, pred_cls)
    denom = tp + fp + fn
    return (tp / denom) * 100.0 if denom > 0 else 0.0


def _per_class_weighted(gt_cls: np.ndarray, pred_cls: np.ndarray) -> float:
    """Macro-average across the 5 rainfall classes of the per-class same-
    class overlap rate: for each class k, fraction of GT pixels in class
    k that Pred also labelled k. Robust to class-0 dominance. Returns
    percentage. Classes with zero GT presence contribute 0 (skipped
    from the average denominator when ALL classes are empty, to avoid
    NaN)."""
    valid = gt_cls != -1
    scores = []
    for k in range(N_RAINFALL_CLASSES):
        gt_k = (gt_cls == k) & valid
        n_gt = int(np.sum(gt_k))
        if n_gt == 0:
            continue
        n_hit = int(np.sum(gt_k & (pred_cls == k)))
        scores.append(n_hit / n_gt)
    if not scores:
        return 0.0
    return float(np.mean(scores)) * 100.0


# ============================================================================
# JSON / CSV writers
# ============================================================================
def _summarise_confusion(counts: dict) -> dict:
    """Compute FAR/POD/CSI from aggregated (TP, FP, FN, TN)."""
    tp, fp, fn, _tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    eps = 1e-10
    return {
        "FAR": fp / (tp + fp + eps),
        "POD": tp / (tp + fn + eps),
        "CSI": tp / (tp + fp + fn + eps),
        "TP":  tp, "FP": fp, "FN": fn, "TN": _tn,
    }


def _write_csv(rows: list[dict], path: Path):
    """Per-sample CSV with one row per (date, reference_utc) and
    columns for both metrics x each lead time, plus CSI and the hits /
    misses / false-alarm percentages at the tuned HIGH."""
    if not rows:
        print(f"  No rows to write for {path}")
        return
    fieldnames = ["date", "reference_utc"]
    for offset in LEAD_STEP_OFFSETS:
        fieldnames.append(f"iou_mask_t+{offset}")
        fieldnames.append(f"class_wt_t+{offset}")
        fieldnames.append(f"csi_t+{offset}")
        for name in HMF_NAMES:
            fieldnames.append(f"{name}_pct_t+{offset}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def _write_json(track: str, year: int, month: int,
                selected: list[tuple[str, str]],
                rows: list[dict],
                confusion_per_lead: dict[int, dict],
                step_minutes: int, path: Path,
                *,
                rainfall_threshold_mmh: float = RAINFALL_THRESHOLD_MMH,
                high_coverage_pct: float = HIGH_COVERAGE_PCT,
                post_processing: dict | None = None,
                per_patch: dict | None = None,
                model_tag: str | None = None,
                hmf_pooled: dict | None = None,
                extra: dict | None = None):
    """Aggregate summary with per-lead-time counts + metrics + the
    lists of (date, reference_utc) that met the high-coverage threshold.
    Both thresholds are recorded in the JSON so a run's outputs are
    self-documenting when the CLI overrides the defaults.

    `post_processing` mirrors the lightning track's block: the swept
    hysteresis grid and the per-lead winner. `per_patch` carries the
    per-patch CSI table that build_patch_ensemble.py selects members
    from - scored on the post-processed canvases, so it describes the
    shipped product rather than raw argmax."""
    lead_titles = [f"t+{o * step_minutes}" for o in LEAD_STEP_OFFSETS]
    total = len(rows)
    above = {lt: {"iou_mask": 0, "class_wt": 0} for lt in lead_titles}
    high_cov_lists = {lt: {"iou_mask": [], "class_wt": []}
                      for lt in lead_titles}
    for r in rows:
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            lt = lead_titles[i]
            iou = r[f"iou_mask_t+{offset}"]
            cwt = r[f"class_wt_t+{offset}"]
            if iou >= high_coverage_pct:
                above[lt]["iou_mask"] += 1
                high_cov_lists[lt]["iou_mask"].append(
                    [r["date"], r["reference_utc"]]
                )
            if cwt >= high_coverage_pct:
                above[lt]["class_wt"] += 1
                high_cov_lists[lt]["class_wt"].append(
                    [r["date"], r["reference_utc"]]
                )
    diff_pct = {}
    for lt in lead_titles:
        diff_pct[lt] = {}
        for metric in ("iou_mask", "class_wt"):
            if total == 0:
                diff_pct[lt][metric] = 0.0
                continue
            missing = total - above[lt][metric]
            diff_pct[lt][metric] = (missing / total) * 100.0
    metrics = {
        lead_titles[i]: _summarise_confusion(confusion_per_lead[i])
        for i in range(len(LEAD_STEP_OFFSETS))
    }
    doc = {
        "track": track,
        "year": year,
        "month": month,
        "split": SPLIT,
        "model": model_tag,
        "threshold_mmh": rainfall_threshold_mmh,
        "hits_misses_false_alarms_pooled_per_lead": hmf_pooled or {},
        "high_coverage_threshold_pct": high_coverage_pct,
        "total_selected_samples": total,
        "initial_selection": [[d, h] for d, h in selected],
        "samples_above_threshold_per_lead": above,
        "difference_pct_per_lead": diff_pct,
        "metrics_per_lead": metrics,
        "high_coverage_samples_per_lead": high_cov_lists,
    }
    if post_processing is not None:
        doc["post_processing"] = post_processing
    if per_patch is not None:
        doc["per_patch"] = per_patch
    if extra:
        doc.update(extra)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Wrote summary to {path}")


# ============================================================================
# Metrics figure (extraction mode side-effect)
# ============================================================================
HMF_NAMES = ("hits", "misses", "false_alarms")
PICK_MODES = ("csi",)


def picks_from_summary(summary_path, mode: str = "csi", top_n: int | None = None
                       ) -> list[tuple[str, str, str]]:
    """(label, date, reference_utc) of the top N samples of a run.

    mode "csi": ranked by the mean of the per-sample CSI over leads,
    descending, from the samples CSV next to the summary; N defaults to
    5. Labels csi_top01, csi_top02, ... Used by --pick in
    visualize_gt_vs_pred and predict_full_domain.
    """
    if mode not in PICK_MODES:
        raise SystemExit(f"unknown pick mode {mode!r}; choose from {PICK_MODES}")
    top_n = 5 if top_n is None else int(top_n)
    if top_n < 1:
        raise SystemExit("--top_n must be at least 1")
    summary_path = Path(summary_path)
    summary, rows, offsets, step, _stem = _load_run(summary_path)
    if not rows:
        raise SystemExit(f"no samples CSV next to {summary_path}")
    track = summary.get("track", "rainfall")
    cols = ([f"csi_t+{o}" for o in offsets] if track == "rainfall"
            else [f"csi_t+{o * step}" for o in offsets])
    scored = []
    for r in rows:
        vals = [r[c] for c in cols if r.get(c) is not None]
        if vals:
            scored.append((sum(vals) / len(vals), r["date"], r["reference_utc"]))
    if not scored:
        raise SystemExit(f"{summary_path}: the samples CSV has no per-sample CSI")
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(f"csi_top{i:02d}", d, ref)
            for i, (_, d, ref) in enumerate(scored[:top_n], 1)]

def _hmf_percentages(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """hits % and misses % over the GT-active pixels, false alarms % over
    the predicted-active pixels — the denominators _format_hmf_pct uses
    on the per-date figures. None where the denominator is empty."""
    gt_active = tp + fn
    pred_active = tp + fp
    return {
        "hits": (100.0 * tp / gt_active) if gt_active else None,
        "misses": (100.0 * fn / gt_active) if gt_active else None,
        "false_alarms": (100.0 * fp / pred_active) if pred_active else None,
    }


def _load_run(summary_path: Path):
    """(summary, rows, offsets, step_minutes, stem) from a saved run.

    Rows come from the sibling <stem>_samples.csv with numbers parsed
    (empty cells -> None); the leads and the step from the summary's
    metrics_per_lead keys (t+15, t+30, ...)."""
    import math
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stem = (summary_path.name[:-len("_summary.json")]
            if summary_path.name.endswith("_summary.json") else summary_path.stem)
    csv_path = summary_path.with_name(f"{stem}_samples.csv")
    rows: list[dict] = []
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                row = {}
                for k, v in r.items():
                    if k in ("date", "reference_utc"):
                        row[k] = v
                        continue
                    try:
                        row[k] = float(v) if v not in ("", None) else None
                    except ValueError:
                        row[k] = v
                rows.append(row)
    minutes = sorted(int(k[2:]) for k in (summary.get("metrics_per_lead") or {}))
    if not minutes:
        raise SystemExit(f"{summary_path} has no metrics_per_lead block")
    step = math.gcd(*minutes)
    offsets = [m // step for m in minutes]
    return summary, rows, offsets, step, stem


def _run_title(summary: dict) -> str:
    """`Validation - rainfall - test split, 2026-06, >= 8 mm/h  |  <model>  |  N samples`."""
    parts = [f"Validation - {summary.get('track', '')} - "
             f"{scope_label(summary.get('year'), summary.get('month'))}"]
    if summary.get("model"):
        parts.append(str(summary["model"]))
    parts.append(f"{summary.get('total_selected_samples', len(summary.get('initial_selection', [])))} selected samples")
    return "  |  ".join(parts)


def _apply_scope(summary: dict) -> None:
    """scope_label reads module state; take it from the summary so a
    figure redrawn later says what the run said."""
    global SPLIT, THRESHOLD_MMH
    SPLIT = summary.get("split")
    if summary.get("threshold_mmh") is not None:
        THRESHOLD_MMH = float(summary["threshold_mmh"])


def _hmf_series(rows: list[dict], track: str, offset: int, step: int
                ) -> dict[str, list[tuple[int, float]]]:
    """{hits|misses|false_alarms: [(sample index, percent)]} for one lead.
    Rainfall rows carry the three columns; lightning rows carry pod and
    far ratios, so hits = POD, misses = 1 - POD, false alarms = FAR."""
    out: dict[str, list] = {name: [] for name in HMF_NAMES}
    for x, r in enumerate(rows):
        if track == "rainfall":
            for name in HMF_NAMES:
                v = r.get(f"{name}_pct_t+{offset}")
                if v is not None:
                    out[name].append((x, v))
        else:
            m = offset * step
            pod, far = r.get(f"pod_t+{m}"), r.get(f"far_t+{m}")
            if pod is not None:
                out["hits"].append((x, 100.0 * pod))
                out["misses"].append((x, 100.0 * (1.0 - pod)))
            if far is not None:
                out["false_alarms"].append((x, 100.0 * far))
    return out


def _date_ticks(ax, rows: list[dict]) -> None:
    """First, median and last sample as the x-axis labels."""
    n = len(rows)
    if n == 0:
        return
    idx = sorted({0, (n - 1) // 2, n - 1})
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{rows[i]['date']}\n{rows[i]['reference_utc']}" for i in idx],
                       fontsize=8)
    ax.set_xlabel("Sample (chronological)")


def _plot_metrics_bars(summary: dict, offsets: list[int], step: int,
                       path: Path) -> None:
    """Grouped FAR / POD / CSI bars, one group per metric, a bar per lead."""
    metrics_per_lead = summary["metrics_per_lead"]
    lead_titles = [f"t+{o * step}" for o in offsets]
    names = ["FAR", "POD", "CSI"]
    n_lead = len(lead_titles)
    colors, _ = lead_palette(n_lead)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    x = np.arange(len(names))
    width = 0.8 / n_lead
    for i, lt in enumerate(lead_titles):
        vals = [float(metrics_per_lead.get(lt, {}).get(m, 0.0)) for m in names]
        ax.bar(x + (i - (n_lead - 1) / 2) * width, vals, width, label=lt,
               color=colors[i], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    if summary.get("track") == "rainfall":
        ax.set_title("FAR / POD / CSI on the >= 10 mm/h event (class >= 1)\n"
                     f"samples selected at >= {THRESHOLD_MMH:g} mm/h")
    else:
        ax.set_title("FAR / POD / CSI at the tuned HIGH per lead")
    fig.suptitle(_run_title(summary), fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


def _plot_coverage_scatter(summary: dict, rows: list[dict], offsets: list[int],
                           step: int, path: Path) -> None:
    """Per-sample IoU against class-weighted overlap, marker per lead;
    red lines at 50 %, gray dotted at the coverage grade."""
    high_cov = float(summary.get("high_coverage_threshold_pct", HIGH_COVERAGE_PCT))
    lead_titles = [f"t+{o * step}" for o in offsets]
    colors, markers = lead_palette(len(offsets))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, offset in enumerate(offsets):
        pts = [(r[f"iou_mask_t+{offset}"], r[f"class_wt_t+{offset}"]) for r in rows
               if r.get(f"iou_mask_t+{offset}") is not None
               and r.get(f"class_wt_t+{offset}") is not None]
        if pts:
            ax.scatter([a for a, _ in pts], [b for _, b in pts], marker=markers[i],
                       color=colors[i], alpha=0.55, s=25, edgecolor="none",
                       label=lead_titles[i])
    ax.axhline(high_cov, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.axvline(high_cov, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.axhline(50.0, color="red", linestyle="-", alpha=0.8, linewidth=1)
    ax.axvline(50.0, color="red", linestyle="-", alpha=0.8, linewidth=1)
    ax.set_xlabel("IoU on the >= 10 mm/h binary mask (%)")
    ax.set_ylabel("Per-class weighted overlap (%)")
    ax.set_title("Per-sample coverage scatter")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(_run_title(summary), fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


def _plot_hmf_figure(summary: dict, rows: list[dict], offsets: list[int],
                     step: int, path: Path) -> None:
    """Three panels - hits %, misses %, false alarms % - each a per-sample
    scatter in chronological order with one marker per lead and a red
    line at 50 %. The x axis names the first, median and last sample."""
    track = summary.get("track", "rainfall")
    lead_titles = [f"t+{o * step}" for o in offsets]
    colors, markers = lead_palette(len(offsets))
    titles = {"hits": "Hits (% of GT-active pixels detected)",
              "misses": "Misses (% of GT-active pixels missed)",
              "false_alarms": "False alarms (% of predicted-active pixels)"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for ax, name in zip(axes, HMF_NAMES):
        for i, offset in enumerate(offsets):
            pts = _hmf_series(rows, track, offset, step)[name]
            ax.scatter([x for x, _ in pts], [v for _, v in pts],
                       marker=markers[i], color=colors[i], alpha=0.55, s=25,
                       edgecolor="none", label=lead_titles[i])
        ax.axhline(50.0, color="red", linestyle="-", alpha=0.8, linewidth=1)
        ax.set_title(titles[name])
        ax.set_ylabel("%")
        ax.set_ylim(-2, 102)
        ax.set_xlim(-1, max(len(rows), 1))
        ax.grid(alpha=0.3)
        ax.legend(title="lead", fontsize=8)
        _date_ticks(ax, rows)
    fig.suptitle(f"{_run_title(summary)}  |  post-processed map at the tuned HIGH",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, _hmf_legend_text(), fontsize=8, family="monospace",
             va="top")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


def _plot_hmf_percentiles(summary: dict, rows: list[dict], offsets: list[int],
                          step: int, path: Path) -> None:
    """Per panel (hits, misses, false alarms) and per lead: the p10-p90
    whiskers, the p25-p75 box and the median of the per-sample rates,
    the red 50 % line, and above each box the share of samples at or
    above 50 % and below it."""
    track = summary.get("track", "rainfall")
    lead_titles = [f"t+{o * step}" for o in offsets]
    colors, _ = lead_palette(len(offsets))
    titles = {"hits": "Hits %", "misses": "Misses %", "false_alarms": "False alarms %"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for ax, name in zip(axes, HMF_NAMES):
        data, labels = [], []
        for i, offset in enumerate(offsets):
            vals = [v for _, v in _hmf_series(rows, track, offset, step)[name]]
            data.append(vals if vals else [np.nan])
            labels.append(lead_titles[i])
        bp = ax.boxplot(data, positions=np.arange(len(offsets)), widths=0.55,
                        whis=(10, 90), showfliers=False, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for i, vals in enumerate(data):
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            above = 100.0 * sum(v >= 50.0 for v in vals) / len(vals)
            ax.text(i, 101, f">= 50 %: {above:.0f} %\n< 50 %: {100 - above:.0f} %\nn={len(vals)}",
                    ha="center", va="bottom", fontsize=8)
        ax.axhline(50.0, color="red", linestyle="-", alpha=0.8, linewidth=1)
        ax.set_xticks(np.arange(len(offsets)))
        ax.set_xticklabels(labels)
        ax.set_ylim(-2, 118)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel("%")
        ax.set_title(f"{titles[name]} - p10 / p25 / median / p75 / p90 over samples")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{_run_title(summary)}  |  distribution of the per-sample rates",
                 fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path.name}")


def plot_rainfall_tuning(summary: dict, out_low: Path, out_high: Path) -> None:
    """The two rainfall tuning figures from the summary: phase 1, pooled
    CSI against the plain threshold per lead with the chosen LOW starred;
    phase 2, pooled CSI of every (LOW, HIGH) window per lead with the
    chosen pair starred. Both on the validation-split samples."""
    pp = summary["post_processing"]
    leads = list(pp["low_sweep"])
    colors, markers = lead_palette(len(leads))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, lead in enumerate(leads):
        thr = sorted(float(t) for t in pp["low_sweep"][lead])
        csi = [pp["low_sweep"][lead][f"{t:.2f}"] for t in thr]
        ax.plot(thr, csi, color=colors[i], linewidth=1.5, label=lead)
        low = pp["phase1_low_per_lead"][lead]
        ax.plot([low], [pp["low_sweep"][lead][f"{low:.2f}"]], marker='*',
                markersize=13, color=colors[i], linestyle='none')
    ax.set_xlabel("threshold on p(argmax) of rainy-argmax pixels")
    ax.set_ylabel("pooled CSI (validation split)")
    ax.set_title("Phase 1 - LOW per lead (star = chosen)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(_run_title(summary), fontsize=12, fontweight="bold")
    fig.savefig(out_low, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {Path(out_low).name}")

    fig, axes = plt.subplots(1, len(leads), figsize=(5 * len(leads), 4.8),
                             squeeze=False, constrained_layout=True)
    for i, (ax, lead) in enumerate(zip(axes[0], leads)):
        table = pp["window_sweep"][lead]
        xs = np.arange(len(table))
        csi = [t["CSI"] for t in table]
        ax.bar(xs, csi, color=colors[i], alpha=0.75)
        best = max(range(len(table)), key=lambda j: (table[j]["CSI"], -table[j]["low"]))
        ax.plot([best], [csi[best]], marker='*', markersize=14, color='black')
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{t['low']:.2f}\n{t['high']:.2f}" for t in table],
                           fontsize=7)
        ax.set_xlabel("window (LOW over HIGH)")
        ax.set_ylabel("pooled CSI (validation split)")
        ax.set_title(f"{lead}: chosen low={table[best]['low']:.2f} "
                     f"high={table[best]['high']:.2f}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{_run_title(summary)}  |  Phase 2 - (LOW, HIGH) windows, "
                 f"width {pp['window']:g}, reach {pp['reach']:g}",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_high, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {Path(out_high).name}")


def plot_tuning_from_summary(summary_path: Path, out_path: Path) -> None:
    """CSI against the swept HIGH threshold, one line per lead, a dashed
    line at each lead's winner, for either track, drawn from
    post_processing.tuning_scores of the saved summary."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    pp = summary.get("post_processing") or {}
    scores = pp.get("tuning_scores") or {}
    if not scores:
        raise SystemExit(f"{summary_path} has no post_processing.tuning_scores "
                         f"(a baseline run, or written before tuning existed)")
    winners = pp.get("high_threshold_per_lead") or {}
    low = pp.get("low_threshold")
    leads = list(scores)
    colors, _markers = lead_palette(len(leads))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, lead in enumerate(leads):
        highs = sorted(float(h) for h in scores[lead])
        csis = [scores[lead][f"{h:.2f}"]["CSI"] for h in highs]
        ax.plot(highs, csis, marker="o", color=colors[i], label=lead,
                linewidth=1.5)
        if winners.get(lead) is not None:
            ax.axvline(float(winners[lead]), color=colors[i], linestyle="--",
                       alpha=0.5, linewidth=1)
    ax.set_xlabel("High threshold")
    ax.set_ylabel("Pooled CSI (validation split)")
    ax.set_title("CSI vs high-threshold sweep  "
                 + (f"(low={float(low):.2f} fixed)" if low is not None else ""))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(_run_title(summary), fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {Path(out_path).name}")


def make_plots(summary_path: Path) -> list[Path]:
    """Every figure the saved run allows, next to the summary: metrics
    bars, coverage scatter (rainfall), hits / misses / false alarms,
    their percentiles, and the tuning curves (runs with a sweep).
    Called at the end of an extraction and by --plots."""
    summary_path = Path(summary_path)
    summary, rows, offsets, step, stem = _load_run(summary_path)
    _apply_scope(summary)
    # Figures go to the run's own folder next to the summary.
    here = run_dir(summary_path.parent, stem)
    made: list[Path] = []

    def out(suffix):
        made.append(here / f"{stem}_{suffix}.png")
        return made[-1]

    _plot_metrics_bars(summary, offsets, step, out("metrics"))
    if rows and summary.get("track") == "rainfall" \
            and f"iou_mask_t+{offsets[0]}" in rows[0]:
        _plot_coverage_scatter(summary, rows, offsets, step, out("coverage"))
    if rows:
        _plot_hmf_figure(summary, rows, offsets, step, out("hmf"))
        _plot_hmf_percentiles(summary, rows, offsets, step, out("hmf_percentiles"))
    pp = summary.get("post_processing") or {}
    if pp.get("low_sweep"):
        plot_rainfall_tuning(summary, out("tuning_low"), out("tuning_high"))
    elif pp.get("tuning_scores"):
        plot_tuning_from_summary(summary_path, out("tuning"))
    return made


# ============================================================================
# Extraction mode
# ============================================================================
def run_extraction(track: str, year: int, month: int,
                   mode: str, source: str, finetuned: bool,
                   data_root: Path, model_dir: Path, output_dir: Path,
                   *,
                   rainfall_threshold_mmh: float = RAINFALL_THRESHOLD_MMH,
                   high_coverage_pct: float = HIGH_COVERAGE_PCT,
                   rainfall_low: float | None = None,
                   rainfall_window: float = 0.02,
                   rainfall_reach: float = 0.10,
                   period=None,
                   baseline: bool = False):
    """Extraction mode for the rainfall track, in three phases.

    1. LOW per lead, tuned on samples drawn from the VALIDATION split:
       a plain threshold on p(argmax) of the rainy-argmax pixels, swept
       0.01..0.99, the best pooled CSI wins. --rainfall_low_threshold
       skips this and fixes one LOW for every lead.
    2. (LOW, HIGH) per lead, on the same samples: windows of
       --rainfall_window slid left and right of the phase-1 LOW out to
       --rainfall_reach; every window is one hysteresis run; the best
       pooled CSI wins.
    3. The scope samples (test split, month, ...) are scored at the
       chosen pair per lead: FAR/POD/CSI, coverage, hits / misses /
       false alarms, per-patch CSI, and every per-sample count is saved.

    The selection threshold (`rainfall_threshold_mmh`) only decides
    which timesteps are considered; the scored event is class >= 1,
    the model's own 10 mm/h boundary. The baseline has no
    post-processing: its class map is scored as it is (phase 3 only).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Validation extraction - track={track}  {scope_label(year, month)}")
    print("=" * 70)
    if baseline:
        from sepconv_ensemble_training import SEPCONV_MODE
        mode = SEPCONV_MODE
        if period is None:
            raise SystemExit("--baseline needs --period (the window the "
                             "baseline was trained on, e.g. w44)")
    tag = artifact_tag(mode, source, period, finetuned, baseline=baseline)
    print(f"  Data root: {data_root}")
    print(f"  Model:     {mode} ({source}"
          f"{' finetuned' if finetuned else ''}"
          f"{' - SepConv-ens baseline' if baseline else ''})  -> {tag}")
    print(f"  Thresholds: rainfall_threshold_mmh={rainfall_threshold_mmh:g}  "
          f"high_coverage_pct={high_coverage_pct:g}")

    init_sequence_config(str(data_root), source, period=period)
    sync_window_from_sequence_config()
    L = len(LEAD_STEP_OFFSETS)
    set_normalization_stats_path(
        data_root / normalization_stats_name(source, period)
    )
    mode_config = get_mode_config(mode)
    step_minutes = _load_step_minutes(data_root)

    print(f"\nScope samples (>= {rainfall_threshold_mmh:g} mm/h) ...")
    selected = select_samples(data_root, year, month,
                              threshold_mmh=rainfall_threshold_mmh,
                              source=source, period=period)
    if not selected:
        print("No samples selected. Nothing to do.")
        return
    tuning_selected: list = []
    if not baseline:
        print(f"\nTuning samples from the validation split "
              f"(>= {rainfall_threshold_mmh:g} mm/h) ...")
        tuning_selected = select_samples(
            data_root, year, month, threshold_mmh=rainfall_threshold_mmh,
            source=source, period=period, split="validation")
        if not tuning_selected:
            raise SystemExit("no validation-split samples to tune on")
        if SPLIT == "validation":
            print("  NOTE: the scope IS the validation split; thresholds are "
                  "tuned on the same population they are reported on.")

    print(f"\nLoading model ...")
    if baseline:
        from sepconv_predict import load_base_models, predict_classes
        from sepconv_compose import MAX_STEP as _SEPCONV_MAX_STEP
        from train_models import build_run_tag
        if L != _SEPCONV_MAX_STEP:
            raise SystemExit(
                f"the {period} window has {L} future steps but the "
                f"composition forecasts {_SEPCONV_MAX_STEP}")
        base_models = load_base_models(model_dir,
                                       build_run_tag(mode, source, period))
        print(f"  Loaded base models: {sorted(base_models)}")
        model = None
    else:
        model = load_model_artifact(model_dir, mode, source, finetuned,
                                    period=period, weights=WEIGHTS)
        print(f"  Loaded: {model.count_params():,} parameters")
    from visualize_gt_vs_pred import build_full_soft_pred

    def _load_sample(sel):
        """Inputs and the ground-truth field of every lead for one
        reference timestep; runs in the loader thread."""
        date_str, hhmm = sel
        ref_utc = f"{hhmm[:2]}:{hhmm[2:]}"
        inputs, valid_patches = build_inputs_for_reference(
            data_root, mode_config, date_str, ref_utc, step_minutes,
        )
        gt_fields = []
        if valid_patches:
            for offset in LEAD_STEP_OFFSETS:
                gt_hhmm, gt_day = _resolve_gt(
                    ref_utc, offset * step_minutes, date_str,
                )
                gt_fields.append(
                    _load_gt_rainfall_canvas(data_root, gt_day, gt_hhmm))
        return inputs, valid_patches, gt_fields

    def _predict(inputs, valid_patches):
        """(argmax canvases, score canvases or None, eligible masks or
        None) for one sample. Score = p(argmax) on rainy-argmax pixels,
        the quantity the hysteresis thresholds act on."""
        if baseline:
            past = np.asarray(inputs["past_hr"])
            frames = [past[:, t, :, :, 0] for t in range(past.shape[1])]
            classes, _mmh = predict_classes(
                base_models, frames, period, max_step=L,
                data_root=str(data_root), source=source,
                batch_size=18, batched=True)
            return _paste_class_canvases(classes, valid_patches, L), None, None
        preds = _predict_batches(model, inputs, batch_size=18)
        pred_canvases = paste_predictions_to_canvas(
            preds, valid_patches, label_type="radar")
        soft = build_full_soft_pred(preds, valid_patches,
                                    n_classes=preds.shape[-1])
        scores, eligible = [], []
        for i in range(L):
            argmax = np.argmax(soft[i], axis=-1)
            p_arg = np.take_along_axis(soft[i], argmax[..., None],
                                       axis=-1).squeeze(-1)
            scores.append(np.where(argmax > 0, p_arg, 0.0).astype(np.float32))
            eligible.append(argmax > 0)
        return pred_canvases, scores, eligible

    def _run(samples, label, per_sample):
        """Loop `samples` through the model, calling per_sample(k, date,
        ref, pred_canvases, scores, eligible, gt_canvases). Returns the
        (date, ref) of every sample that produced predictions."""
        done = []
        n_skipped = 0
        print(f"\n{label}: {len(samples)} samples ...")
        for k, ((date_str, hhmm), (inputs, valid_patches, gt_fields)) in enumerate(
                _prefetch(samples, _load_sample), 1):
            ref_utc = f"{hhmm[:2]}:{hhmm[2:]}"
            if k == 1 or k % 20 == 0 or k == len(samples):
                print(f"  [{k}/{len(samples)}] {date_str} {ref_utc}")
            if not valid_patches:
                n_skipped += 1
                continue
            pred_canvases, scores, eligible = _predict(inputs, valid_patches)
            gt_canvases = [_paste_gt_class_canvas(gt_fields[i], valid_patches)
                           for i in range(L)]
            per_sample(len(done), date_str, ref_utc, pred_canvases, scores,
                       eligible, gt_canvases)
            done.append((date_str, ref_utc))
        print(f"  done: {len(done)} scored, {n_skipped} skipped (missing inputs)")
        return done

    low_grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    low_per_lead: dict[int, float] = {}
    pair_per_lead: dict[int, tuple] = {}
    low_sweep: dict = {}
    window_sweep: dict = {}
    tune_dates: list = []
    tune_hist: list = []        # [n_t][L] (pos, neg, gt_pos_total)
    tune_window: list = []      # [n_t][L][n_pairs] (tp, fp, fn, tn)
    pairs_per_lead: dict[int, list] = {}

    if not baseline:
        # ---- Phase 1: LOW per lead from a plain threshold sweep --------
        def _phase1(n, date_str, ref_utc, pred_canvases, scores, eligible, gts):
            def one(i):
                pos, neg = score_histograms(scores[i], gts[i], eligible[i])
                gt_pos_total = int(((gts[i] > 0) & (gts[i] >= 0)).sum())
                return pos, neg, gt_pos_total
            tune_hist.append(_map_leads(one, L))

        tune_dates = _run(tuning_selected, "Phase 1 - LOW sweep on the "
                          "validation split", _phase1)
        if not tune_dates:
            raise SystemExit("no tuning sample produced predictions")
        if rainfall_low is not None:
            low_per_lead = {i: float(rainfall_low) for i in range(L)}
            print(f"  LOW fixed at {rainfall_low:.2f} for every lead "
                  f"(--rainfall_low_threshold)")
        print("\nPhase 1 result (plain threshold on p(argmax), pooled CSI):")
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            pos = sum(h[i][0] for h in tune_hist)
            neg = sum(h[i][1] for h in tune_hist)
            gt_tot = sum(h[i][2] for h in tune_hist)
            csi = csi_from_histograms(pos, neg, low_grid, gt_tot)
            low_sweep[f"t+{offset}"] = {f"{t:.2f}": float(c)
                                        for t, c in zip(low_grid, csi)}
            if rainfall_low is None:
                low_per_lead[i] = float(low_grid[int(np.argmax(csi))])
            print(f"  t+{offset}: LOW={low_per_lead[i]:.2f}  "
                  f"CSI={csi[int(np.argmax(csi))]:.4f}")

        # ---- Phase 2: (LOW, HIGH) windows around each lead's LOW -------
        pairs_per_lead = {i: window_pairs(low_per_lead[i], rainfall_window,
                                          rainfall_reach) for i in range(L)}
        print(f"\nPhase 2 - {len(pairs_per_lead[0])} (LOW, HIGH) windows per "
              f"lead, width {rainfall_window:g}, reach {rainfall_reach:g}")

        def _phase2(n, date_str, ref_utc, pred_canvases, scores, eligible, gts):
            def one(i):
                out = []
                for lo, hi in pairs_per_lead[i]:
                    conf, _ = sweep_hysteresis(scores[i], lo, [hi], gts[i],
                                               eligible=eligible[i])
                    out.append(conf[hi])
                return out
            tune_window.append(_map_leads(one, L))

        _run(tuning_selected, "Phase 2 - window sweep on the validation "
             "split", _phase2)
        print("\nPhase 2 result (hysteresis (LOW, HIGH), pooled CSI):")
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            table = []
            for j, (lo, hi) in enumerate(pairs_per_lead[i]):
                tp = sum(w[i][j][0] for w in tune_window)
                fp = sum(w[i][j][1] for w in tune_window)
                fn = sum(w[i][j][2] for w in tune_window)
                tn = sum(w[i][j][3] for w in tune_window)
                m = _summarise_confusion({"TP": tp, "FP": fp, "FN": fn, "TN": tn})
                table.append({"low": lo, "high": hi, "CSI": m["CSI"],
                              "POD": m["POD"], "FAR": m["FAR"]})
            window_sweep[f"t+{offset}"] = table
            best = max(range(len(table)),
                       key=lambda j: (table[j]["CSI"], -table[j]["low"]))
            pair_per_lead[i] = (table[best]["low"], table[best]["high"])
            print(f"  t+{offset}: low={table[best]['low']:.2f} "
                  f"high={table[best]['high']:.2f}  CSI={table[best]['CSI']:.4f}")

    # ---- Phase 3: score the scope at the chosen pair per lead ----------
    rows: list[dict] = []
    confusion_raw = {i: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for i in range(L)}
    confusion_post = {i: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for i in range(L)}
    patch_acc: dict = {}
    conf_raw_rows: list = []
    conf_post_rows: list = []
    patch_rows: list = []
    cover_rows: list = []

    def _phase3(n, date_str, ref_utc, pred_canvases, scores, eligible, gts):
        row = {"date": date_str, "reference_utc": ref_utc}

        def one(i):
            gt = gts[i]
            raw = pred_canvases[i]
            iou = _iou_binary(gt, raw)
            cwt = _per_class_weighted(gt, raw)
            r_conf = _binary_confusion(gt, raw)
            if baseline:
                p_conf = r_conf
                acc = {}
                _accumulate_per_patch(gt, raw, acc, i)
                per_patch = {q: (c[i]["TP"], c[i]["FP"], c[i]["FN"], c[i]["TN"])
                             for q, c in acc.items()}
            else:
                lo, hi = pair_per_lead[i]
                conf, patches = sweep_hysteresis(scores[i], lo, [hi], gt,
                                                 eligible=eligible[i])
                p_conf, per_patch = conf[hi], patches[hi]
            return iou, cwt, r_conf, p_conf, per_patch

        results = _map_leads(one, L)
        patch_arr = np.zeros((L, N_PATCHES, 4), dtype=np.int64)
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            iou, cwt, r_conf, p_conf, per_patch = results[i]
            row[f"iou_mask_t+{offset}"] = iou
            row[f"class_wt_t+{offset}"] = cwt
            for key, val in zip(("TP", "FP", "FN", "TN"), r_conf):
                confusion_raw[i][key] += val
            for key, val in zip(("TP", "FP", "FN", "TN"), p_conf):
                confusion_post[i][key] += val
            tp, fp, fn, _ = p_conf
            pct = _hmf_percentages(tp, fp, fn)
            for name in HMF_NAMES:
                row[f"{name}_pct_t+{offset}"] = pct[name]
            row[f"csi_t+{offset}"] = (tp / (tp + fp + fn) if (tp + fp + fn) else None)
            _merge_patch_counts(patch_acc, i, per_patch)
            for q, counts in per_patch.items():
                patch_arr[i, q - 1] = counts
        rows.append(row)
        conf_raw_rows.append([results[i][2] for i in range(L)])
        conf_post_rows.append([results[i][3] for i in range(L)])
        patch_rows.append(patch_arr)
        cover_rows.append([(results[i][0], results[i][1]) for i in range(L)])

    scored = _run(selected, "Phase 3 - scoring the scope", _phase3)
    if not rows:
        print("No sample produced predictions. Nothing to write.")
        return

    hmf_pooled = {i: _hmf_percentages(confusion_post[i]["TP"],
                                      confusion_post[i]["FP"],
                                      confusion_post[i]["FN"]) for i in range(L)}
    print("\nScope results at the chosen thresholds:")
    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        m = _summarise_confusion(confusion_post[i])
        r = _summarise_confusion(confusion_raw[i])
        print(f"  t+{offset}: CSI={m['CSI']:.4f} POD={m['POD']:.4f} "
              f"FAR={m['FAR']:.4f}  (raw argmax CSI={r['CSI']:.4f})")

    if baseline:
        post_processing = {
            "method": "none - SepConv-ens class map is the product",
            "low_threshold_per_lead": {f"t+{off}": None for off in LEAD_STEP_OFFSETS},
            "high_threshold_per_lead": {f"t+{off}": None for off in LEAD_STEP_OFFSETS},
        }
    else:
        post_processing = {
            "method": "rainfall_hysteresis on p(argmax)",
            "tuned_on": "validation split",
            "low_threshold": None,
            "low_threshold_per_lead": {
                f"t+{off}": pair_per_lead[i][0] for i, off in enumerate(LEAD_STEP_OFFSETS)},
            "high_threshold_per_lead": {
                f"t+{off}": pair_per_lead[i][1] for i, off in enumerate(LEAD_STEP_OFFSETS)},
            "phase1_low_per_lead": {
                f"t+{off}": low_per_lead[i] for i, off in enumerate(LEAD_STEP_OFFSETS)},
            "low_fixed": rainfall_low,
            "window": rainfall_window,
            "reach": rainfall_reach,
            "low_sweep": low_sweep,
            "window_sweep": window_sweep,
        }
    extra = {
        "sampling": {"max_samples": MAX_SAMPLES, "month_batch": MONTH_BATCH,
                     "seed": SEED},
        "tuning": {"split": "validation" if not baseline else None,
                   "n_samples": len(tune_dates),
                   "samples": [[d, r] for d, r in tune_dates]},
        "metrics_per_lead_raw": {
            f"t+{off * step_minutes}": _summarise_confusion(confusion_raw[i])
            for i, off in enumerate(LEAD_STEP_OFFSETS)},
    }

    stem = f"{track}_{scope_stem(year, month)}_{tag}"
    _write_csv(rows, output_dir / f"{stem}_samples.csv")
    _write_json(track, year, month, selected, rows, confusion_post,
                step_minutes, output_dir / f"{stem}_summary.json",
                rainfall_threshold_mmh=rainfall_threshold_mmh,
                high_coverage_pct=high_coverage_pct,
                post_processing=post_processing,
                per_patch=per_patch_scores(patch_acc),
                model_tag=tag,
                hmf_pooled={f"t+{off * step_minutes}": hmf_pooled[i]
                            for i, off in enumerate(LEAD_STEP_OFFSETS)},
                extra=extra)
    per_sample = dict(
        dates=[d for d, _ in scored], refs=[r for _, r in scored],
        lead_offsets=list(LEAD_STEP_OFFSETS),
        conf_raw=np.asarray(conf_raw_rows, dtype=np.int64),
        conf_post=np.asarray(conf_post_rows, dtype=np.int64),
        patch_post=np.asarray(patch_rows, dtype=np.int64),
        coverage=np.asarray(cover_rows, dtype=np.float64),
    )
    if not baseline:
        per_sample.update(
            tune_dates=[d for d, _ in tune_dates],
            tune_refs=[r for _, r in tune_dates],
            low_grid=low_grid,
            tune_hist_pos=np.asarray([[h[i][0] for i in range(L)] for h in tune_hist],
                                     dtype=np.int64),
            tune_hist_neg=np.asarray([[h[i][1] for i in range(L)] for h in tune_hist],
                                     dtype=np.int64),
            tune_gt_positive=np.asarray([[h[i][2] for i in range(L)] for h in tune_hist],
                                        dtype=np.int64),
            window_pairs=np.asarray([pairs_per_lead[i] for i in range(L)], dtype=np.float64),
            tune_window_conf=np.asarray(tune_window, dtype=np.int64),
        )
    save_per_sample(output_dir / f"{stem}_per_sample.npz", **per_sample)
    make_plots(output_dir / f"{stem}_summary.json")


def _resolve_gt(ref_utc: str, offset_min: int,
                date_str: str) -> tuple[str, str]:
    """Same as predict_full_domain._ref_to_hhmm but returns (hhmm, day)
    with day rolled over if the offset crosses midnight."""
    parts = ref_utc.split(":")
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=int(parts[0]), minute=int(parts[1])
    )
    target = base + timedelta(minutes=offset_min)
    return target.strftime("%H%M"), target.strftime("%Y-%m-%d")


# ============================================================================
# Visualization mode
# ============================================================================
def _load_summary_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"Summary JSON not found: {path}\n"
            f"Run the extraction phase first (drop --date to invoke it)."
        )
    with open(path) as f:
        return json.load(f)


def _date_is_in(date_str: str, entries: list[list[str]]) -> bool:
    for e in entries:
        if len(e) >= 1 and e[0] == date_str:
            return True
    return False


def _colour_for_title(date_in_selection: bool,
                      date_in_high_cov: bool) -> str:
    """Green if both, orange if only initial selection, black otherwise."""
    if not date_in_selection:
        return "black"
    return "#2ca02c" if date_in_high_cov else "#ff7f0e"


def _hit_canvas_and_pct(gt_cls: np.ndarray, pred_cls: np.ndarray
                        ) -> tuple[np.ndarray, float]:
    """Boolean canvas where GT class == Pred class AND GT is active
    (class >= 1). Plus the percentage of GT-active pixels that match."""
    valid = gt_cls != -1
    active = (gt_cls >= 1) & valid
    hit = active & (pred_cls == gt_cls)
    denom = int(np.sum(active))
    pct = (int(np.sum(hit)) / denom * 100.0) if denom > 0 else 0.0
    return hit, pct


def _find_highest_activity_patch(gt_cls: np.ndarray) -> int:
    """Patch (1-indexed) with the most GT-active pixels (class >= 1)."""
    best_patch, best_score = 1, -1
    for p in range(1, N_PATCHES + 1):
        r0, r1, c0, c1 = get_patch_bounds(p)
        tile = gt_cls[r0:r1, c0:c1]
        score = int(np.sum((tile >= 1) & (tile != -1)))
        if score > best_score:
            best_score = score
            best_patch = p
    return best_patch


DEFAULT_GT_MIN_CELL_PIXELS = 10


def _postproc_gt_class_canvas(gt_cls: np.ndarray,
                              min_cell_pixels: int = DEFAULT_GT_MIN_CELL_PIXELS
                              ) -> np.ndarray:
    """Return a copy of gt_cls with class-active (>=1) 8-connected components
    smaller than min_cell_pixels demoted to class 0 (dry).

    Removes single-pixel and sub-scale GT specks so pixel-wise hit counts
    aren't inflated by noise that no spatially-smooth prediction can be
    expected to match. Matches the coupled-cell filtering already used in
    generate_report.py (MIN_CELL_SIZE_PIXELS = 10). -1 sentinels stay -1.
    """
    if min_cell_pixels <= 1:
        return gt_cls.copy()
    active = (gt_cls >= 1) & (gt_cls != -1)
    if not active.any():
        return gt_cls.copy()
    from scipy.ndimage import label as _cc_label
    labeled, n_cc = _cc_label(active, structure=np.ones((3, 3), dtype=bool))
    if n_cc == 0:
        return gt_cls.copy()
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    small_ids = np.where(sizes < min_cell_pixels)[0]
    if small_ids.size == 0:
        return gt_cls.copy()
    small_mask = np.isin(labeled, small_ids) & active
    out = gt_cls.copy()
    out[small_mask] = 0
    return out


def _plot_class_overlap_axis(ax, gt_cls: np.ndarray, pred_cls: np.ndarray,
                             *,
                             min_cell_pixels: int = DEFAULT_GT_MIN_CELL_PIXELS,
                             gt_postproc: bool = True,
                             ) -> dict:
    """Class-aware overlap map on the full canvas.

    Per pixel:
      - hit  (pred_class == gt_class, gt >= 1): viridis-coloured by class
        (same 5-class palette as the pred-only plot, so hit clusters
        read at the same intensity a viewer already associates with the
        rain-rate band).
      - miss (gt >= 1 but pred != gt / pred == 0):        light blue
      - false alarm (gt == 0 but pred >= 1):              orange
      - correct dry (both == 0):                          white
      - out of domain (gt == -1):                         light gray

    When gt_postproc is True (default), GT is first passed through
    _postproc_gt_class_canvas to strip sub-scale connected components.
    That prevents lone GT pixels from swamping the overall matched-%
    denominator.

    Returns a dict with per-class counts and the overall matched-%,
    which the caller can drop into the subtitle:
        {"matched_pct": float, "hits": int, "misses": int,
         "false_alarms": int, "n_gt_active": int,
         "per_class": {1: {"hits": ..., "n_gt": ...}, ...}}
    """
    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT

    gt_eff = (_postproc_gt_class_canvas(gt_cls, min_cell_pixels)
              if gt_postproc else gt_cls)

    valid = gt_eff != -1
    gt_pos = (gt_eff >= 1) & valid
    pr_pos = (pred_cls >= 1) & valid
    hits = gt_pos & (pred_cls == gt_eff)
    misses = gt_pos & ~hits
    fas = pr_pos & ~gt_pos

    H, W = gt_eff.shape
    viridis_5 = plt.get_cmap("viridis", 5)
    # Base: viridis GT class canvas across the whole valid area (class 0
    # dry pixels included — they render as the darkest viridis, same as
    # the pred-only plot). Out-of-domain patches stay as light gray so
    # they don't read as "predicted dry".
    rgba = np.ones((H, W, 4), dtype=np.float32)
    rgba[~valid] = (0.90, 0.90, 0.90, 1.0)
    for k in range(0, 5):
        mask = valid & (gt_eff == k)
        if mask.any():
            rgba[mask] = viridis_5(k / 4.0)
    # Misses painted first: light blue over the viridis base.
    rgba[misses] = (0.20, 0.45, 0.85, 1.0)
    # False alarms painted in RED (was orange in the pre-viridis-base
    # design) for better contrast against the viridis backdrop.
    rgba[fas] = (0.84, 0.15, 0.16, 1.0)
    # Hits: intentionally NOT overpainted — the viridis base already
    # colours them by their (true == predicted) class, and the absence
    # of a miss/FA overlay tells the reader "this pixel matched".

    ax.imshow(rgba, aspect="equal", interpolation="nearest")

    for p in range(1, N_PATCHES + 1):
        r0, _, c0, _ = get_patch_bounds(p)
        ax.add_patch(Rectangle(
            (c0, r0), PATCH_SIZE, PATCH_SIZE,
            linewidth=0.7, edgecolor="black",
            linestyle=(0, (1, 3)), facecolor="none", zorder=3,
        ))
    try:
        overlay_borders(ax)
    except Exception:
        pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_hi, r_lo)
    ax.set_aspect("equal")

    n_gt = int(gt_pos.sum())
    n_hits = int(hits.sum())
    per_class = {}
    for k in range(1, 5):
        gt_k = (gt_eff == k) & valid
        n_gt_k = int(gt_k.sum())
        n_hit_k = int((gt_k & hits).sum())
        per_class[k] = {"hits": n_hit_k, "n_gt": n_gt_k}
    return {
        "matched_pct": (n_hits / n_gt * 100.0) if n_gt > 0 else 0.0,
        "hits": n_hits,
        "misses": int(misses.sum()),
        "false_alarms": int(fas.sum()),
        "n_gt_active": n_gt,
        "per_class": per_class,
    }


_ZONE_ORANGE = (1.00, 0.55, 0.10, 1.0)   # hit  (GT-active + correctly predicted)
_ZONE_BLUE   = (0.20, 0.45, 0.85, 1.0)   # miss (GT-active but not detected)
_ZONE_RED    = (0.84, 0.15, 0.16, 1.0)   # false alarm (predicted, no GT)
_ZONE_WHITE  = (1.00, 1.00, 1.00, 1.0)   # correct dry
_ZONE_GRAY   = (0.90, 0.90, 0.90, 1.0)   # out-of-domain


def _plot_zone_overlap_axis(ax, gt_cls: np.ndarray, pred_cls: np.ndarray,
                            *,
                            min_cell_pixels: int = DEFAULT_GT_MIN_CELL_PIXELS,
                            gt_postproc: bool = True,
                            ) -> dict:
    """Zone-overlap map — treats GT and pred as BINARY masks (any class
    >= 1 counts as "active"), so a predicted blob never mixes hits and
    misses on the same continuous region. This is the "detection" view:
    did we or did we not fire in the right place, regardless of the
    specific rain-rate class we picked.

    Semantics:
      hit         = GT-active AND pred-active    (any class match)
      miss        = GT-active AND pred-dry
      false alarm = pred-active AND GT-dry
      correct dry = both dry
      out of domain = gt == -1

    Palette (only three colours have meaning; the other two are neutral
    canvas / masking):
      orange = hit         (GT-active, correctly predicted)
      blue   = miss        (GT-active, not detected)
      red    = false alarm (predicted, no GT)
      white  = correct dry (base canvas)
      gray   = out of domain
    """
    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT

    gt_eff = (_postproc_gt_class_canvas(gt_cls, min_cell_pixels)
              if gt_postproc else gt_cls)

    valid = gt_eff != -1
    gt_pos = (gt_eff >= 1) & valid
    pr_pos = (pred_cls >= 1) & valid
    # BINARY semantic: hits = both active (any class). Every pixel inside
    # a pred blob is either a hit (GT also active) or a false alarm (GT
    # dry). No "miss inside a pred blob" contradiction.
    hits = gt_pos & pr_pos
    misses = gt_pos & ~pr_pos
    fas = pr_pos & ~gt_pos

    H, W = gt_eff.shape
    rgba = np.ones((H, W, 4), dtype=np.float32)  # base = white (correct dry)
    rgba[~valid] = _ZONE_GRAY
    rgba[gt_pos] = _ZONE_ORANGE             # GT-active: hits + misses base
    rgba[misses] = _ZONE_BLUE                # miss overlay covers orange
    rgba[fas] = _ZONE_RED                    # FA overlay covers white
    # Hits: intentionally NO overlay — orange base showing through says
    # "GT-active pixel, correctly detected".

    ax.imshow(rgba, aspect="equal", interpolation="nearest")

    for p in range(1, N_PATCHES + 1):
        r0, _, c0, _ = get_patch_bounds(p)
        ax.add_patch(Rectangle(
            (c0, r0), PATCH_SIZE, PATCH_SIZE,
            linewidth=0.7, edgecolor="black",
            linestyle=(0, (1, 3)), facecolor="none", zorder=3,
        ))
    try:
        overlay_borders(ax)
    except Exception:
        pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_hi, r_lo)
    ax.set_aspect("equal")

    return {
        "hits": int(hits.sum()),
        "misses": int(misses.sum()),
        "false_alarms": int(fas.sum()),
    }


def _add_zone_color_legend(fig, *, y: float = -0.02,
                           fontsize: int = 10) -> None:
    """Add the 3-swatch zone-overlap colour legend BELOW the plot grid.

    Only the three meaningful categories are listed — correct-dry
    (white) and out-of-domain (gray) are just neutral canvas / masking
    colours, not semantic outcomes worth explaining.

    Anchored at `y=-0.02` in figure coords via `loc="upper center"`,
    so the TOP of the legend sits just below the plot area. Pair with
    `_add_hmf_legend(fig, y=-0.14)` (or similar) to stack the formula
    footer further down; both are captured by `bbox_inches="tight"` at
    savefig time.
    """
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_ZONE_ORANGE[:3], edgecolor="#666", linewidth=0.5,
              label="hit"),
        Patch(facecolor=_ZONE_BLUE[:3], edgecolor="#666", linewidth=0.5,
              label="miss"),
        Patch(facecolor=_ZONE_RED[:3], edgecolor="#666", linewidth=0.5,
              label="false alarm"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=3, fontsize=fontsize, frameon=True,
        framealpha=0.90, edgecolor="#bbb",
    )


def _plot_red_hits_axis(ax, gt_cls: np.ndarray, pred_cls: np.ndarray,
                        *,
                        min_cell_pixels: int = DEFAULT_GT_MIN_CELL_PIXELS,
                        gt_postproc: bool = True,
                        ) -> dict:
    """Per-class hits map — surfaces ONLY the correctly-classified rainfall
    pixels, coloured by their (matching pred == GT) class.

    Every non-hit pixel is transparent, so the panel highlights the
    subset of the domain the model got right at the class level. Misses,
    false alarms and correct-dry pixels are intentionally omitted from
    the visual — the per-class hit-rate breakdown in the subtitle
    (produced from the returned dict) covers the "how much did we
    catch" side, and the zone-overlap sibling file covers the "where
    did we miss / over-predict" side.

        C1: X.X%  C2: Y.Y%  C3: Z.Z%  C4: W.W%

    where each `Ck: p%` is `#hits_k / #gt_active_k * 100`, so classes
    with no GT pixels for that lead print as `n/a`.

    Returns:
        {"per_class_pct": {1: float|None, 2: ..., 3: ..., 4: ...},
         "per_class_counts": {k: {"hits": int, "n_gt": int}},
         "total_matched_pct": float,
         "total_hits": int, "total_gt_active": int}
    """
    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT

    gt_eff = (_postproc_gt_class_canvas(gt_cls, min_cell_pixels)
              if gt_postproc else gt_cls)

    valid = gt_eff != -1
    gt_pos = (gt_eff >= 1) & valid
    hits = gt_pos & (pred_cls == gt_eff)

    # Base: fill the WHOLE valid domain with class 0 (darkest viridis =
    # "R<10"), matching the background rectangle that Rows 1 and 2 show
    # when a pixel is dry / out-of-domain. Out-of-domain patches stay
    # NaN so they render transparent (same behaviour as Rows 1 and 2's
    # nan-masked fallback).
    viridis_kwargs = dict(cmap=plt.get_cmap("viridis", 5),
                          vmin=0, vmax=4,
                          aspect="equal", interpolation="nearest")
    base_display = np.where(valid, 0.0, np.nan)
    ax.imshow(base_display, **viridis_kwargs)

    # Overlay: hit pixels painted in their (matching) viridis class
    # colour — same exact colours the pred / GT rows use, so a class-3
    # hit here reads as the same yellow-green as a class-3 GT pixel in
    # Row 1. Misses / false alarms sit under the dark-viridis background
    # (visually indistinguishable from correct-dry); the per-class
    # hit-rate in the subtitle covers how much of each class was caught.
    hit_class_display = np.where(hits, gt_eff.astype(float), np.nan)
    ax.imshow(hit_class_display, **viridis_kwargs)

    for p in range(1, N_PATCHES + 1):
        r0, _, c0, _ = get_patch_bounds(p)
        ax.add_patch(Rectangle(
            (c0, r0), PATCH_SIZE, PATCH_SIZE,
            linewidth=0.7, edgecolor="black",
            linestyle=(0, (1, 3)), facecolor="none", zorder=3,
        ))
    try:
        overlay_borders(ax)
    except Exception:
        pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_hi, r_lo)
    ax.set_aspect("equal")

    per_class_pct: dict[int, float | None] = {}
    per_class_counts: dict[int, dict] = {}
    for k in range(1, 5):
        gt_k = (gt_eff == k) & valid
        n_gt_k = int(gt_k.sum())
        n_hit_k = int((gt_k & hits).sum())
        per_class_counts[k] = {"hits": n_hit_k, "n_gt": n_gt_k}
        per_class_pct[k] = ((n_hit_k / n_gt_k * 100.0)
                            if n_gt_k > 0 else None)

    n_gt_total = int(gt_pos.sum())
    n_hit_total = int(hits.sum())
    return {
        "per_class_pct": per_class_pct,
        "per_class_counts": per_class_counts,
        "total_matched_pct": ((n_hit_total / n_gt_total * 100.0)
                              if n_gt_total > 0 else 0.0),
        "total_hits": n_hit_total,
        "total_gt_active": n_gt_total,
    }


def _format_per_class_pct(per_class_pct: dict[int, float | None]) -> str:
    """Two-line rendering of {1..4: pct} for a subtitle.

    Each class label carries the actual rain-rate range (from
    visualize_gt_vs_pred.RADAR_CLASS_NAMES) so a viewer knows which mm/h
    band a `C1` / `C2` / ... number refers to without opening the code.
    The full C1..C4 line was overflowing between adjacent panels of the
    3x3 rainfall figure and the 1x4 validation row, so we split it
    across two rows — C1/C2 above, C3/C4 below — which fits per-panel
    at fontsize=10 without touching neighbouring subtitles:

        C1 [10≤R<20 mm/h]: X.X%   C2 [20≤R<30 mm/h]: Y.Y%
        C3 [30≤R<40 mm/h]: Z.Z%   C4 [R≥40 mm/h]: W.W%

    Classes with zero GT-active pixels for the lead print `n/a`.
    """
    def _one(k: int) -> str:
        label = RADAR_CLASS_NAMES[k]  # e.g. "10≤R<20"
        v = per_class_pct.get(k)
        val = "n/a" if v is None else f"{v:.1f}%"
        return f"C{k} [{label} mm/h]: {val}"

    row1 = "   ".join(_one(k) for k in (1, 2))
    row2 = "   ".join(_one(k) for k in (3, 4))
    return f"{row1}\n{row2}"


def _hmf_legend_text() -> str:
    """One-block plain-language explanation of how the hits / misses /
    false-alarms percentages are computed. Rendered as a small footer
    on every figure that uses `_format_hmf_pct` so a reader can decode
    the numbers without opening the code.
    """
    return (
        "How the percentages are computed:\n"
        "  hits %          = hits / (hits + misses)                 "
        "→ fraction of GT-active pixels the model correctly detected\n"
        "  misses %        = misses / (hits + misses)               "
        "→ fraction of GT-active pixels the model missed\n"
        "  false alarms %  = false alarms / (hits + false alarms)   "
        "→ fraction of predicted-active pixels that were wrong"
    )


def _add_hmf_legend(fig, *, y: float = 0.0, fontsize: int = 8) -> None:
    """Anchor `_hmf_legend_text()` below the plot grid.

    `y` is the figure-coord y for the TOP of the text block (va="top"),
    so `y=0.0` puts it flush against the plot bottom, and negative
    values push it further down. Zone-overlap figures use `y=-0.14`
    to leave room for the zone colour legend at y=-0.02 above it.
    Both are captured by `bbox_inches="tight"` at savefig time.
    """
    fig.text(
        0.5, y, _hmf_legend_text(),
        ha="center", va="top",
        fontsize=fontsize, color="#333",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#f5f5f5", edgecolor="#cccccc",
                  linewidth=0.5),
    )


def _format_hmf_pct(hits: int, misses: int, false_alarms: int) -> str:
    """Render (hits, misses, false alarms) as percentages of their natural
    denominators — the format every rainfall/lightning overlap subtitle
    uses so raw pixel counts don't have to be mentally normalised.

    Denominators:
      - hits, misses:   GT-active total (hits + misses)   [recall / miss-rate]
      - false alarms:   pred-active total (hits + FA)     [1 - precision / FAR]

    Undefined denominators (no GT active or no pred active) print `n/a`
    so an accidental "0.0%" isn't misread as a genuine perfect result.
    """
    gt_active = hits + misses
    pred_active = hits + false_alarms
    hits_pct = (hits / gt_active * 100.0) if gt_active > 0 else None
    miss_pct = (misses / gt_active * 100.0) if gt_active > 0 else None
    fa_pct = (false_alarms / pred_active * 100.0) if pred_active > 0 else None
    def _p(v): return "n/a" if v is None else f"{v:.1f}%"
    return (f"hits = {_p(hits_pct)}   "
            f"misses = {_p(miss_pct)}   "
            f"false alarms = {_p(fa_pct)}")


def _plot_structure_axis(ax, hit_mask: np.ndarray, gt_cls: np.ndarray):
    """Left panel: white background, red pixels where hit_mask is True."""
    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT
    # Base layer: light-gray for "no data" (gt_cls == -1) so the reader
    # can tell "empty patch" apart from "predicted no rain".
    base = np.zeros_like(gt_cls, dtype=np.float32)  # 0 = white background
    base[gt_cls == -1] = 0.5   # gray for no-data patches
    ax.imshow(base, cmap="gray_r", vmin=0.0, vmax=1.0,
              aspect="equal", interpolation="nearest")
    # Overlay red hits.
    hit_display = np.where(hit_mask, 1.0, np.nan)
    ax.imshow(hit_display, cmap=mcolors.ListedColormap(["#d62728"]),
              vmin=0.5, vmax=1.5, aspect="equal", interpolation="nearest")
    for p in range(1, N_PATCHES + 1):
        r0, _, c0, _ = get_patch_bounds(p)
        ax.add_patch(Rectangle(
            (c0, r0), PATCH_SIZE, PATCH_SIZE,
            linewidth=0.7, edgecolor="black",
            linestyle=(0, (1, 3)), facecolor="none", zorder=3,
        ))
    try:
        overlay_borders(ax)
    except Exception:
        pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_hi, r_lo)
    ax.set_aspect("equal")


def _plot_zoom_axis(ax, gt_cls: np.ndarray, pred_cls: np.ndarray,
                    patch_num: int):
    """Right panel: 256x256 zoom of the highest-activity patch,
    coloured by (a) matched pixels red, (b) GT-only pixels blue,
    (c) Pred-only pixels orange, (d) background grey."""
    r0, r1, c0, c1 = get_patch_bounds(patch_num)
    gt_tile = gt_cls[r0:r1, c0:c1]
    pr_tile = pred_cls[r0:r1, c0:c1]
    gt_pos = (gt_tile >= 1) & (gt_tile != -1)
    pr_pos = (pr_tile >= 1) & (pr_tile != -1)
    both = gt_pos & pr_pos & (gt_tile == pr_tile)
    only_gt = gt_pos & ~pr_pos
    only_pr = pr_pos & ~gt_pos
    display = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32) + 0.95
    display[only_gt] = np.array([0.20, 0.45, 0.85])  # blue: GT missed
    display[only_pr] = np.array([1.00, 0.55, 0.10])  # orange: Pred FA
    display[both] = np.array([0.84, 0.15, 0.16])     # red: matched
    ax.imshow(display, extent=(c0, c1, r1, r0),
              aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)
    ax.set_aspect("equal")
    n_gt = int(np.sum(gt_pos))
    n_hit = int(np.sum(both))
    pct = (n_hit / n_gt * 100.0) if n_gt > 0 else 0.0
    ax.text(
        c0 + 4, r1 - 6,
        f"patch #{patch_num}  hit={n_hit}/{n_gt}  ({pct:.1f}%)",
        color="white", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="black", alpha=0.55, edgecolor="none"),
        va="bottom", ha="left", zorder=6,
    )


def run_visualization(track: str, year: int, month: int, date_str: str,
                      mode: str, source: str, finetuned: bool,
                      data_root: Path, model_dir: Path, output_dir: Path, period=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{track}_{scope_stem(year, month)}_"
            f"{artifact_tag(mode, source, period, finetuned)}")
    summary = _load_summary_json(output_dir / f"{stem}_summary.json")

    date_in_selection = _date_is_in(date_str, summary["initial_selection"])
    if not date_in_selection:
        raise SystemExit(
            f"Date {date_str} not present in the initial selection for "
            f"{scope_label(year, month)}. Nothing to visualise. "
            f"Check {output_dir / (stem + '_summary.json')} for the "
            f"list of selected dates."
        )

    # Load prediction pipeline the same way extraction does.
    init_sequence_config(str(data_root), source, period=period)
    sync_window_from_sequence_config()
    set_normalization_stats_path(
        data_root / normalization_stats_name(source, period)
    )
    mode_config = get_mode_config(mode)
    step_minutes = _load_step_minutes(data_root)

    # Pick every reference on this date that survived the selection.
    refs = sorted(
        h for d, h in summary["initial_selection"] if d == date_str
    )
    if not refs:
        raise SystemExit(f"Selection has no references for {date_str}.")

    print(f"Loading model ...")
    model = load_model_artifact(model_dir, mode, source, finetuned,
                                period=period, weights=WEIGHTS)
    print(f"  Loaded: {model.count_params():,} parameters")

    lead_titles = [f"t+{o * step_minutes}" for o in LEAD_STEP_OFFSETS]
    label_offsets_min = [o * step_minutes for o in LEAD_STEP_OFFSETS]

    for ref_utc in refs:
        print(f"\n{date_str} {ref_utc} - building figures ...")
        inputs, valid_patches = build_inputs_for_reference(
            data_root, mode_config, date_str, ref_utc, step_minutes,
        )
        if not valid_patches:
            print("  No inputs available - skipping.")
            continue
        preds = model.predict(inputs, batch_size=18, verbose=0)
        pred_canvases = paste_predictions_to_canvas(
            preds, valid_patches, label_type="radar",
        )

        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            gt_hhmm, gt_day = _resolve_gt(
                ref_utc, offset * step_minutes, date_str,
            )
            gt_field = _load_gt_rainfall_canvas(data_root, gt_day, gt_hhmm)
            gt_canvas = _paste_gt_class_canvas(gt_field, valid_patches)
            pred_canvas = pred_canvases[i]

            # Colour picker: green if this lead time cleared 90%, orange if
            # only in selection. Same date can be green on one lead and
            # orange on another.
            lt = lead_titles[i]
            high_cov_iou = summary["high_coverage_samples_per_lead"][lt]["iou_mask"]
            high_cov_cwt = summary["high_coverage_samples_per_lead"][lt]["class_wt"]
            in_high = _date_is_in(date_str, high_cov_iou) or \
                      _date_is_in(date_str, high_cov_cwt)
            title_color = _colour_for_title(True, in_high)

            # 1x3 layout per lead (main file):
            #   [GT class | Pred class (+hit/miss/FA %) | Per-class red hits (per-class %)]
            # + shared viridis-5 rain-rate colourbar on the right. Mirrors
            # predict_full_domain's 3x3 inference figure column-for-column,
            # just laid out horizontally because validation writes one PNG
            # per lead. The zone-overlap view lives in a sibling
            # `<stem>_..._zone.png` file (orange/blue/red palette, its own
            # legend) so the two colour schemes don't compete.
            import matplotlib.cm as mcm
            import matplotlib.colors as mcolors
            from visualize_gt_vs_pred import (
                _render_gt_axes, _render_pred_axes,
                _gt_kwargs_for, _pred_kwargs_for, RADAR_CLASS_NAMES,
            )
            wall = _resolve_gt(ref_utc, label_offsets_min[i], date_str)[0]
            wall_hm = f"{wall[:2]}:{wall[2:]}"

            fig, axes = plt.subplots(1, 3, figsize=(26, 8),
                                     constrained_layout=True)
            _render_gt_axes(
                axes[0], gt_canvas, "radar",
                lead_title=lt, lead_hhmm=wall_hm,
                gt_kwargs=_gt_kwargs_for("radar"),
                valid_patches=valid_patches,
            )
            _render_pred_axes(
                axes[1], pred_canvas, valid_patches, "radar",
                threshold=None,
                lead_title=lt, lead_hhmm=wall_hm,
                pred_kwargs=_pred_kwargs_for("radar", None),
            )
            # Aggregate hit/miss/FA (class-strict, post-processed GT) —
            # same numbers the sibling `_zone.png` reports.
            gt_pp = _postproc_gt_class_canvas(gt_canvas)
            valid_pp = gt_pp != -1
            gt_pp_pos = (gt_pp >= 1) & valid_pp
            pr_pp_pos = (pred_canvas >= 1) & valid_pp
            hits_agg = int(((pred_canvas == gt_pp) & gt_pp_pos).sum())
            misses_agg = int((gt_pp_pos & ~((pred_canvas == gt_pp) & gt_pp_pos)).sum())
            fas_agg = int((pr_pp_pos & ~gt_pp_pos).sum())
            axes[1].set_title(
                f"Pred - {lt} ({wall_hm} UTC)\n"
                f"{_format_hmf_pct(hits_agg, misses_agg, fas_agg)}",
                fontsize=10,
            )
            red_stats = _plot_red_hits_axis(
                axes[2], gt_canvas, pred_canvas,
            )
            axes[2].set_title(
                f"Per-class hits  |  "
                f"{_format_per_class_pct(red_stats['per_class_pct'])}",
                fontsize=10,
            )

            sm = mcm.ScalarMappable(
                cmap=plt.get_cmap("viridis", 5),
                norm=mcolors.Normalize(vmin=0, vmax=4),
            )
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=axes.tolist(),
                ticks=[0, 1, 2, 3, 4],
                shrink=0.7, pad=0.01, location="right",
            )
            cbar.set_ticklabels(RADAR_CLASS_NAMES)
            cbar.set_label("Rain-rate class")

            fig.suptitle(
                f"Validation ({track})  |  {date_str}  ref={ref_utc}  "
                f"|  {lt} ({wall_hm} UTC)",
                fontsize=14, fontweight="bold", color=title_color,
            )
            _add_hmf_legend(fig)

            safe_ref = ref_utc.replace(":", "")
            out_png = output_dir / (
                f"{stem}_{date_str}_{safe_ref}_{lt.replace('+', 'p')}.png"
            )
            fig.savefig(out_png, dpi=130, bbox_inches="tight")
            plt.close(fig)

            # Sibling zone-overlap file per lead: orange = hit,
            # blue = miss, red = FA, white = correct dry, gray = out-of-
            # domain. Same class-strict hit/miss/FA numbers as the Pred
            # column above, presented under the detection palette with
            # an explicit colour legend at the top.
            zone_fig, zone_ax = plt.subplots(
                1, 1, figsize=(14, 8), constrained_layout=True,
            )
            zone_stats = _plot_zone_overlap_axis(
                zone_ax, gt_canvas, pred_canvas,
            )
            zone_ax.set_title(
                f"Zone overlap - {lt} ({wall_hm} UTC)\n"
                f"{_format_hmf_pct(zone_stats['hits'], zone_stats['misses'], zone_stats['false_alarms'])}",
                fontsize=11,
            )
            zone_fig.suptitle(
                f"Validation ({track}) - Zone overlap  |  "
                f"{date_str}  ref={ref_utc}  |  {lt} ({wall_hm} UTC)",
                fontsize=13, fontweight="bold", color=title_color,
            )
            _add_zone_color_legend(zone_fig)
            _add_hmf_legend(zone_fig, y=-0.14)
            zone_png = output_dir / (
                f"{stem}_{date_str}_{safe_ref}_{lt.replace('+', 'p')}_zone.png"
            )
            zone_fig.savefig(zone_png, dpi=130, bbox_inches="tight")
            plt.close(zone_fig)
            print(f"  Saved {lt} -> {out_png.name}")


# ============================================================================
# ==========================  LIGHTNING TRACK  ===============================
# ============================================================================
# Structural clone of the rainfall track, differing only in:
#   - sample selection driven by LINET occurrence (>= N active pixels)
#   - inference via Hann-blended overlap + hysteresis (from lightning_postproc)
#   - per-lead high-threshold TUNING as part of extraction (sweep CSI over a
#     grid of candidate high values, pick argmax per lead, persist to JSON)
#   - binary GT and binary post-processed prediction; no per-class weighting
#   - visualization reuses predict_full_domain._plot_lightning_2x3
#     (2 rows x 3 leads on ONE figure per reference)
# ============================================================================

LIGHTNING_LOW_THRESHOLD = DEFAULT_LOW_THRESHOLD          # 0.90 (operational)
LIGHTNING_HIGH_GRID = tuple(round(x, 2)
                            for x in np.arange(0.91, 1.00, 0.01))
LIGHTNING_MIN_ACTIVE_PIXELS = 1


_LIGHTNING_FILENAME_RE = re.compile(
    r"^lightning_occurrence_(\d{8})_(\d{4})\.npy$"
)


def _iter_lightning_files(data_root: Path, year: int, month: int):
    """Yield (date_str, hhmm, path) for every LINET occurrence .npy matching
    (year, month). Layout: {data_root}/lightning_data/occurrence/
    nc4_YYYY-MM-DD-Romania_occurrence/lightning_occurrence_YYYYMMDD_HHMM.npy
    (that's the native-grid write path from read_kml_version2)."""
    root = data_root / "lightning_data" / "occurrence"
    if not root.is_dir():
        raise FileNotFoundError(
            f"LINET occurrence root not found: {root}. "
            f"Run read_kml_version2.py first."
        )
    date_prefix = f"nc4_{year:04d}-{month:02d}-"
    for day_folder in sorted(root.iterdir()):
        if not day_folder.is_dir() or not day_folder.name.startswith(date_prefix):
            continue
        for name in list_arrays(day_folder):
            m = _LIGHTNING_FILENAME_RE.match(name)
            if m is None:
                continue
            f = day_folder / name
            ymd = m.group(1)
            date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
            yield date_str, m.group(2), f


def select_samples_lightning(data_root: Path, year: int, month: int,
                              min_active_pixels: int = LIGHTNING_MIN_ACTIVE_PIXELS,
                              ) -> list[tuple[str, str]]:
    """Iterate every LINET occurrence file in the month, keep those with
    at least `min_active_pixels` active pixels. Returns list of
    (date_str, hhmm) tuples sorted chronologically."""
    kept: list[tuple[str, str]] = []
    scanned = 0
    for date_str, hhmm, path in _iter_lightning_files(data_root, year, month):
        scanned += 1
        data = load_array(path)
        if data.ndim == 3:
            data = np.squeeze(data, axis=0)
        n_active = int((data > 0).sum())
        if n_active >= min_active_pixels:
            kept.append((date_str, hhmm))
    print(f"  Scanned {scanned} LINET occurrence files; "
          f"kept {len(kept)} with >= {min_active_pixels} active pixel(s)")
    return kept


def _binary_confusion_lightning(gt_bin: np.ndarray,
                                pred_bin: np.ndarray
                                ) -> tuple[int, int, int, int]:
    """(TP, FP, FN, TN) over the full 768x1536 canvas. Both inputs are
    treated as binary via a `> 0` cast, so any dtype works."""
    gt_pos = gt_bin > 0
    pr_pos = pred_bin > 0
    tp = int((gt_pos & pr_pos).sum())
    fp = int((~gt_pos & pr_pos).sum())
    fn = int((gt_pos & ~pr_pos).sum())
    tn = int((~gt_pos & ~pr_pos).sum())
    return tp, fp, fn, tn


def _iou_lightning(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return (tp / denom) * 100.0 if denom > 0 else 0.0


def _write_csv_lightning(rows: list[dict], path: Path, step_minutes: int):
    """Per-sample CSV: (date, reference_utc) + IoU/FAR/POD/CSI per lead at
    the CHOSEN high-threshold. Complements the summary JSON which persists
    the full tuning sweep."""
    if not rows:
        print(f"  No rows to write for {path}")
        return
    fieldnames = ["date", "reference_utc"]
    for offset in LEAD_STEP_OFFSETS:
        m = offset * step_minutes
        fieldnames += [
            f"iou_t+{m}", f"far_t+{m}", f"pod_t+{m}", f"csi_t+{m}",
        ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def _write_json_lightning(
    year: int, month: int,
    selected: list[tuple[str, str]],
    rows: list[dict],
    aggregate_confusion_per_lead: dict[int, dict],
    tuning_scores: dict[int, dict[float, dict]],
    best_high_per_lead: dict[int, float],
    low_threshold: float,
    step_minutes: int,
    path: Path,
    *,
    rainfall_threshold_mmh: float = RAINFALL_THRESHOLD_MMH,
    high_coverage_pct: float = HIGH_COVERAGE_PCT,
    per_patch: dict | None = None,
    extra: dict | None = None,
):
    """Aggregate summary that mirrors the rainfall JSON schema and adds
    the `post_processing` block predict_full_domain.py consumes for the
    tuned per-lead high thresholds. `tuning_scores` is the full grid so
    the choice can be re-audited later. Selection is OPERA-driven (>=10
    mm/h) - the field `selection_criterion` in the JSON documents that,
    matching the rainfall track for cross-track coupling analysis."""
    lead_titles = [f"t+{o * step_minutes}" for o in LEAD_STEP_OFFSETS]
    total = len(rows)
    above = {lt: {"iou": 0} for lt in lead_titles}
    high_cov_lists = {lt: {"iou": []} for lt in lead_titles}
    for r in rows:
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            lt = lead_titles[i]
            iou = r[f"iou_t+{offset * step_minutes}"]
            if iou >= high_coverage_pct:
                above[lt]["iou"] += 1
                high_cov_lists[lt]["iou"].append(
                    [r["date"], r["reference_utc"]]
                )
    diff_pct = {}
    for lt in lead_titles:
        diff_pct[lt] = {}
        if total == 0:
            diff_pct[lt]["iou"] = 0.0
            continue
        diff_pct[lt]["iou"] = ((total - above[lt]["iou"]) / total) * 100.0
    metrics_at_best = {
        lead_titles[i]: _summarise_confusion(aggregate_confusion_per_lead[i])
        for i in range(len(LEAD_STEP_OFFSETS))
    }
    # Reshape tuning_scores from {lead_idx: {high: agg_dict}} to
    # {lead_title: {high_str: agg_dict}} so JSON keeps stable string keys.
    tuning_scores_named = {}
    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        lt = lead_titles[i]
        tuning_scores_named[lt] = {
            f"{h:.2f}": tuning_scores[i][h] for h in sorted(tuning_scores[i])
        }
    high_named = {
        f"t+{offset * step_minutes}": best_high_per_lead[offset]
        for offset in LEAD_STEP_OFFSETS
    }
    doc = {
        "track": "lightning",
        "year": year,
        "month": month,
        "split": SPLIT,
        "threshold_mmh": rainfall_threshold_mmh,
        "selection_criterion": (
            f"OPERA-driven: >= {rainfall_threshold_mmh:g} mm/h anywhere on the "
            f"768x1536 canvas at the reference timestep (shared with the "
            f"rainfall track for parity)"
        ),
        "high_coverage_threshold_pct": high_coverage_pct,
        "total_selected_samples": total,
        "initial_selection": [[d, h] for d, h in selected],
        "samples_above_threshold_per_lead": above,
        "difference_pct_per_lead": diff_pct,
        "metrics_per_lead": metrics_at_best,
        "high_coverage_samples_per_lead": high_cov_lists,
        "post_processing": {
            "low_threshold": low_threshold,
            "high_grid": sorted(float(h) for h in next(iter(tuning_scores.values()), {})),
            "high_threshold_per_lead": high_named,
            "tuning_scores": tuning_scores_named,
            "tuning_metric": "csi",
        },
    }
    if per_patch is not None:
        doc["per_patch"] = per_patch
    if extra:
        doc.update(extra)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Wrote summary to {path}")


def lightning_low_from_evaluation(eval_root: Path, tag: str) -> float:
    """The decision threshold the lightning evaluation tuned
    (optimal_threshold in evaluation/eval_<tag>/evaluation_results.json);
    it is the hysteresis LOW of the validation track."""
    path = Path(eval_root) / f"eval_{tag}" / "evaluation_results.json"
    if not path.is_file():
        raise SystemExit(
            f"no evaluation for {tag} at {path}: run evaluate_coalition.py "
            f"for this model first (its tuned threshold is the hysteresis "
            f"LOW), or pass --lightning_low_threshold.")
    blob = json.loads(path.read_text(encoding="utf-8"))
    value = blob.get("optimal_threshold")
    if value is None:
        raise SystemExit(f"{path} has no optimal_threshold")
    print(f"  LOW = {float(value):.3f} from {path} "
          f"({blob.get('threshold_source', '?')})")
    return float(value)


def run_extraction_lightning(
    year: int, month: int, mode: str, source: str, finetuned: bool,
    data_root: Path, model_dir: Path, output_dir: Path,
    *,
    stride: int = DEFAULT_STRIDE,
    low_threshold: float | None = None,
    batch_size: int = 32,
    rainfall_threshold_mmh: float = RAINFALL_THRESHOLD_MMH,
    high_coverage_pct: float = HIGH_COVERAGE_PCT,
    kd: bool = False,
    period=None,
    eval_root: Path = Path("./evaluation"),
):
    """Extraction mode for the lightning track.

    LOW is the threshold the evaluation tuned for this model (read from
    its evaluation_results.json) unless --lightning_low_threshold says
    otherwise. HIGH is swept above LOW in 0.01 steps, per lead, on
    samples drawn from the VALIDATION split; the scope samples are then
    scored at the chosen HIGH per lead, and every per-sample count is
    saved. Sample selection is OPERA-driven, shared with the rainfall
    track.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = artifact_tag(mode, source, period, finetuned, kd)

    print("=" * 70)
    print(f"Validation extraction - track=lightning  {scope_label(year, month)}")
    print("=" * 70)
    print(f"  Data root: {data_root}")
    print(f"  Model:     {mode} ({source}{' finetuned' if finetuned else ''}"
          f"{' KD student' if kd else ''})  -> {tag}")
    if low_threshold is None:
        low_threshold = lightning_low_from_evaluation(eval_root, tag)
    else:
        print(f"  LOW = {low_threshold:.3f} (--lightning_low_threshold)")
    high_grid = [round(low_threshold + 0.01 * k, 2)
                 for k in range(1, int(round((0.99 - low_threshold) / 0.01)) + 1)]
    if not high_grid:
        raise SystemExit(f"LOW {low_threshold:.2f} leaves no room for a HIGH sweep")
    print(f"  Post-proc: stride={stride}  low={low_threshold:.2f}  "
          f"high {high_grid[0]:.2f}..{high_grid[-1]:.2f} ({len(high_grid)} candidates)")
    print(f"  Thresholds: rainfall_threshold_mmh={rainfall_threshold_mmh:g}  "
          f"high_coverage_pct={high_coverage_pct:g}")

    init_sequence_config(str(data_root), source, period=period)
    sync_window_from_sequence_config()
    L = len(LEAD_STEP_OFFSETS)
    set_normalization_stats_path(
        data_root / normalization_stats_name(source, period)
    )
    mode_config = get_mode_config(mode)
    if mode_config["label_type"] != "lightning":
        raise SystemExit(
            f"--mode {mode} has label_type={mode_config['label_type']!r}; "
            f"--track lightning requires a lightning-headed mode.")
    step_minutes = _load_step_minutes(data_root)

    print(f"\nScope samples via OPERA (>= {rainfall_threshold_mmh:g} mm/h) ...")
    selected = select_samples(data_root, year, month,
                              threshold_mmh=rainfall_threshold_mmh,
                              source=source, period=period)
    if not selected:
        print("No samples selected. Nothing to do.")
        return
    print(f"\nTuning samples from the validation split ...")
    tuning_selected = select_samples(
        data_root, year, month, threshold_mmh=rainfall_threshold_mmh,
        source=source, period=period, split="validation")
    if not tuning_selected:
        raise SystemExit("no validation-split samples to tune on")
    if SPLIT == "validation":
        print("  NOTE: the scope IS the validation split; HIGH is tuned on "
              "the same population it is reported on.")

    print(f"\nLoading model ...")
    model = load_model_artifact(model_dir, mode, source, finetuned,
                                kd=kd, period=period, weights=WEIGHTS)
    print(f"  Loaded: {model.count_params():,} parameters")

    def _load_sample(sel):
        date_str, hhmm = sel
        ref_utc = f"{hhmm[:2]}:{hhmm[2:]}"
        inputs, positions = build_inputs_for_reference_overlapped(
            data_root, mode_config, date_str, ref_utc, step_minutes,
            stride=stride,
        )
        gt_bins = []
        if positions:
            for offset in LEAD_STEP_OFFSETS:
                gt_hhmm, gt_day = _resolve_gt(
                    ref_utc, offset * step_minutes, date_str,
                )
                gt_bins.append(
                    _load_gt_lightning_canvas(data_root, gt_day, gt_hhmm))
        return inputs, positions, gt_bins

    def _run(samples, label, per_sample):
        done = []
        n_skipped = 0
        print(f"\n{label}: {len(samples)} samples (Hann overlap, stride={stride}) ...")
        for k, ((date_str, hhmm), (inputs, positions, gt_bins)) in enumerate(
                _prefetch(samples, _load_sample), 1):
            ref_utc = f"{hhmm[:2]}:{hhmm[2:]}"
            if k == 1 or k % 20 == 0 or k == len(samples):
                print(f"  [{k}/{len(samples)}] {date_str} {ref_utc}")
            if not positions:
                n_skipped += 1
                continue
            preds = _predict_batches(model, inputs, batch_size=batch_size)
            prob_canvases = paste_predictions_hann_blended(preds, positions)
            gts = [None if g is None else g.astype(np.int32) for g in gt_bins]
            per_sample(len(done), date_str, ref_utc, prob_canvases, gts)
            done.append((date_str, ref_utc))
        print(f"  done: {len(done)} scored, {n_skipped} skipped (missing inputs)")
        return done

    # ---- Phase A: HIGH per lead on the validation-split samples --------
    tune_conf: list = []   # [n_t][L][n_high] (tp, fp, fn, tn); zeros when no GT

    def _phase_a(n, date_str, ref_utc, probs, gts):
        def one(i):
            if gts[i] is None:
                return [(0, 0, 0, 0)] * len(high_grid)
            conf, _ = sweep_hysteresis(probs[i], low_threshold, high_grid, gts[i])
            return [conf[h] for h in high_grid]
        tune_conf.append(_map_leads(one, L))

    tune_dates = _run(tuning_selected, "Phase A - HIGH sweep on the validation split",
                      _phase_a)
    if not tune_dates:
        raise SystemExit("no tuning sample produced predictions")
    tuning_scores: dict[int, dict[float, dict]] = {i: {} for i in range(L)}
    best_high_per_lead: dict[int, float] = {}
    print("\nPhase A result (pooled CSI on the validation split):")
    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        for j, h in enumerate(high_grid):
            tp = sum(c[i][j][0] for c in tune_conf)
            fp = sum(c[i][j][1] for c in tune_conf)
            fn = sum(c[i][j][2] for c in tune_conf)
            tn = sum(c[i][j][3] for c in tune_conf)
            tuning_scores[i][float(h)] = _summarise_confusion(
                {"TP": tp, "FP": fp, "FN": fn, "TN": tn})
        best_h = max(high_grid, key=lambda h: (tuning_scores[i][float(h)]["CSI"], -h))
        best_high_per_lead[offset] = float(best_h)
        print(f"  t+{offset * step_minutes}: high={best_h:.2f}  "
              f"CSI={tuning_scores[i][float(best_h)]['CSI']:.4f}")

    # ---- Phase B: score the scope at the chosen HIGH per lead ----------
    rows: list[dict] = []
    aggregate_confusion_per_lead = {i: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                                    for i in range(L)}
    patch_acc: dict = {}
    conf_rows: list = []
    patch_rows: list = []

    def _phase_b(n, date_str, ref_utc, probs, gts):
        row = {"date": date_str, "reference_utc": ref_utc}

        def one(i):
            if gts[i] is None:
                return None
            h = best_high_per_lead[LEAD_STEP_OFFSETS[i]]
            conf, patches = sweep_hysteresis(probs[i], low_threshold, [h], gts[i])
            return conf[h], patches[h]

        results = _map_leads(one, L)
        patch_arr = np.zeros((L, N_PATCHES, 4), dtype=np.int64)
        conf_arr = np.zeros((L, 4), dtype=np.int64)
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            m = offset * step_minutes
            if results[i] is None:
                for name in ("iou", "far", "pod", "csi"):
                    row[f"{name}_t+{m}"] = None
                continue
            (tp, fp, fn, tn), per_patch = results[i]
            for key, val in zip(("TP", "FP", "FN", "TN"), (tp, fp, fn, tn)):
                aggregate_confusion_per_lead[i][key] += val
            per = _summarise_confusion({"TP": tp, "FP": fp, "FN": fn, "TN": tn})
            row[f"iou_t+{m}"] = _iou_lightning(tp, fp, fn)
            row[f"far_t+{m}"] = per["FAR"]
            row[f"pod_t+{m}"] = per["POD"]
            row[f"csi_t+{m}"] = per["CSI"]
            _merge_patch_counts(patch_acc, i, per_patch)
            conf_arr[i] = (tp, fp, fn, tn)
            for q, counts in per_patch.items():
                patch_arr[i, q - 1] = counts
        rows.append(row)
        conf_rows.append(conf_arr)
        patch_rows.append(patch_arr)

    scored = _run(selected, "Phase B - scoring the scope", _phase_b)
    if not rows:
        print("No sample produced predictions. Nothing to write.")
        return
    print("\nScope results at the chosen thresholds:")
    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        m = _summarise_confusion(aggregate_confusion_per_lead[i])
        print(f"  t+{offset * step_minutes}: CSI={m['CSI']:.4f} POD={m['POD']:.4f} "
              f"FAR={m['FAR']:.4f}")

    stem = f"lightning_{scope_stem(year, month)}_{tag}"
    _write_csv_lightning(rows, output_dir / f"{stem}_samples.csv", step_minutes)
    _write_json_lightning(
        year, month, selected, rows,
        aggregate_confusion_per_lead, tuning_scores,
        best_high_per_lead, low_threshold, step_minutes,
        output_dir / f"{stem}_summary.json",
        rainfall_threshold_mmh=rainfall_threshold_mmh,
        high_coverage_pct=high_coverage_pct,
        per_patch=per_patch_scores(patch_acc),
        extra={
            "sampling": {"max_samples": MAX_SAMPLES, "month_batch": MONTH_BATCH,
                         "seed": SEED},
            "tuning": {"split": "validation", "n_samples": len(tune_dates),
                       "samples": [[d, r] for d, r in tune_dates],
                       "low_source": ("--lightning_low_threshold"
                                      if low_threshold is not None and False
                                      else "evaluation optimal_threshold")},
        },
    )
    save_per_sample(
        output_dir / f"{stem}_per_sample.npz",
        dates=[d for d, _ in scored], refs=[r for _, r in scored],
        lead_offsets=list(LEAD_STEP_OFFSETS),
        conf_post=np.asarray(conf_rows, dtype=np.int64),
        patch_post=np.asarray(patch_rows, dtype=np.int64),
        tune_dates=[d for d, _ in tune_dates], tune_refs=[r for _, r in tune_dates],
        high_grid=np.asarray(high_grid, dtype=np.float64),
        tune_conf=np.asarray(tune_conf, dtype=np.int64),
    )
    make_plots(output_dir / f"{stem}_summary.json")


def run_visualization_lightning(
    year: int, month: int, date_str: str,
    mode: str, source: str, finetuned: bool,
    data_root: Path, model_dir: Path, output_dir: Path,
    *,
    stride: int = DEFAULT_STRIDE,
    low_threshold: float | None = None,
    batch_size: int = 32,
    kd: bool = False,
    period=None,
):
    """One figure per selected reference on the given date. Layout:
      Row 1 (columns = t+15/+30/+45): GT lightning occurrence
      Row 2 (columns = t+15/+30/+45): GT rendered underneath + post-processed
                                       positive pixels overlaid in red

    The per-lead high threshold is read from
    post_processing.high_threshold_per_lead in the summary JSON produced
    by run_extraction_lightning. The stem is the SAME as extraction wrote
    (base / _finetuned / _kd suffix chosen by the corresponding flag) so
    visualisation reads the same JSON its own extraction produced."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"lightning_{scope_stem(year, month)}_"
            f"{artifact_tag(mode, source, period, finetuned, kd)}")
    summary = _load_summary_json(output_dir / f"{stem}_summary.json")
    if low_threshold is None:
        low_threshold = float((summary.get("post_processing") or {}).get(
            "low_threshold", LIGHTNING_LOW_THRESHOLD))
    if "post_processing" not in summary:
        raise SystemExit(
            f"Summary {stem}_summary.json is missing the post_processing "
            f"block. Re-run extraction (--track lightning without --date)."
        )
    step_minutes = _load_step_minutes(data_root)
    high_named = summary["post_processing"]["high_threshold_per_lead"]
    high_per_lead: dict[int, float] = {}
    for offset in LEAD_STEP_OFFSETS:
        key = f"t+{offset * step_minutes}"
        if key not in high_named:
            raise SystemExit(
                f"post_processing.high_threshold_per_lead is missing {key}"
            )
        high_per_lead[offset] = float(high_named[key])

    date_in_selection = _date_is_in(date_str, summary["initial_selection"])
    if not date_in_selection:
        raise SystemExit(
            f"Date {date_str} is not in the initial selection for "
            f"{scope_label(year, month)}. Nothing to visualise."
        )

    init_sequence_config(str(data_root), source, period=period)
    sync_window_from_sequence_config()
    set_normalization_stats_path(
        data_root / normalization_stats_name(source, period)
    )
    mode_config = get_mode_config(mode)
    if mode_config["label_type"] != "lightning":
        raise SystemExit(
            f"--mode {mode} has label_type={mode_config['label_type']!r}; "
            f"--track lightning requires a lightning-headed mode."
        )

    refs = sorted(
        h for d, h in summary["initial_selection"] if d == date_str
    )
    if not refs:
        raise SystemExit(f"Selection has no references for {date_str}.")

    variant_label = ("finetuned" if finetuned
                     else "KD student" if kd
                     else "base")
    print(f"Loading model ({variant_label}) ...")
    model = load_model_artifact(model_dir, mode, source, finetuned,
                                kd=kd, period=period, weights=WEIGHTS)
    print(f"  Loaded: {model.count_params():,} parameters")

    lead_titles = [f"t+{o * step_minutes}" for o in LEAD_STEP_OFFSETS]
    for ref_utc in refs:
        print(f"\n{date_str} {ref_utc} - building figure ...")
        inputs, positions = build_inputs_for_reference_overlapped(
            data_root, mode_config, date_str, ref_utc, step_minutes,
            stride=stride,
        )
        if not positions:
            print("  No inputs available - skipping.")
            continue
        preds = model.predict(inputs, batch_size=batch_size, verbose=0)
        prob_canvases = paste_predictions_hann_blended(preds, positions)
        bin_canvases = [
            hysteresis_binary(
                prob_canvases[i], low=low_threshold,
                high=high_per_lead[LEAD_STEP_OFFSETS[i]],
            )
            for i in range(len(prob_canvases))
        ]
        gt_canvases = []
        for offset in LEAD_STEP_OFFSETS:
            gt_hhmm, gt_day = _resolve_gt(
                ref_utc, offset * step_minutes, date_str,
            )
            gt_canvases.append(
                _load_gt_lightning_canvas(data_root, gt_day, gt_hhmm)
            )

        # 90% coverage title colour, matching the rainfall track:
        #   green  = this date cleared >=90% IoU on AT LEAST ONE lead
        #            (post-processed binary vs GT binary; see the note
        #            about lightning's post-processing step turning
        #            surviving pixels into 1s)
        #   orange = this date is in the initial selection but no lead
        #            cleared 90%
        any_high = any(
            _date_is_in(date_str,
                        summary["high_coverage_samples_per_lead"][lt]["iou"])
            for lt in lead_titles
        )
        suptitle_color = _colour_for_title(True, any_high)

        safe_ref = ref_utc.replace(":", "")
        out_png = run_dir(output_dir, stem) / f"{stem}_{date_str}_{safe_ref}.png"
        _plot_lightning_2x3(
            prob_canvases, bin_canvases, gt_canvases,
            date_str=date_str, ref_utc=ref_utc,
            step_minutes=step_minutes,
            low=low_threshold, high_per_lead=high_per_lead,
            output_path=out_png,
            suptitle_prefix=f"Validation lightning ({date_str} ref={ref_utc})",
            suptitle_color=suptitle_color,
        )
        marker = ("green (>= 90% IoU on some lead)" if any_high
                  else "orange (in selection only, no lead cleared 90%)")
        print(f"  Saved -> {out_png.name}  [{marker}]")


# ============================================================================
# ==========================  KD TRACK (teacher vs student)  =================
# ============================================================================
# Structural clone of the lightning track, wrapped around BOTH models:
#   * OPERA-driven sample selection (same shared list as rainfall/lightning)
#   * Hann-overlapped inputs built once per reference; TEACHER consumes them
#     as-is (LINET + MTG vis_06 in HR), STUDENT consumes a slice of the
#     LAST N HR channels (vis_06 only). Same MR pass through.
#   * Per candidate high in the sweep grid we hysteresis-binarise EACH model
#     against LINET GT independently, so each ends up with its own tuned
#     per-lead high threshold.
#   * Outputs: kd_<yyyy>_<mm>_{samples.csv, summary.json, metrics_{FAR,POD,
#     CSI,IoU}.png}. summary.json has BOTH teacher.post_processing and
#     student.post_processing blocks.

KD_TEACHER_MODE = "mtg_lightning_opera_occurrence"
KD_STUDENT_MODE = "mtg_opera_occurrence"
# Same constant as train_lightning_kd.STUDENT_HR_CHANNELS - kept in sync by
# a comment (no runtime import to keep this module free of TF at CLI time).
KD_STUDENT_HR_CHANNELS = 1

KD_METRIC_NAMES = ("FAR", "POD", "CSI", "IoU")


def _kd_slice_student_inputs(inputs: dict) -> dict:
    """Derive the student's inputs from a teacher-format Hann-overlapped
    inputs dict by keeping only the LAST N HR channels (= MTG vis_06)."""
    student_hr = inputs["past_hr"][..., -KD_STUDENT_HR_CHANNELS:]
    out = dict(inputs)
    out["past_hr"] = student_hr
    return out


def _write_csv_kd(rows: list[dict], path: Path, step_minutes: int):
    """Per-sample CSV with teacher + student columns side by side per lead.

    Columns: date, reference_utc,
             iou_teacher_t+{m}, iou_student_t+{m},
             far_teacher_t+{m}, far_student_t+{m}, ... (per lead).
    """
    if not rows:
        print(f"  No rows to write for {path}")
        return
    fieldnames = ["date", "reference_utc"]
    for offset in LEAD_STEP_OFFSETS:
        m = offset * step_minutes
        for metric in ("iou", "far", "pod", "csi"):
            fieldnames.append(f"{metric}_teacher_t+{m}")
            fieldnames.append(f"{metric}_student_t+{m}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def _write_json_kd(
    year: int, month: int,
    selected: list[tuple[str, str]],
    rows: list[dict],
    teacher_conf_per_lead: dict[int, dict],
    student_conf_per_lead: dict[int, dict],
    teacher_tuning: dict[int, dict[float, dict]],
    student_tuning: dict[int, dict[float, dict]],
    teacher_best_high: dict[int, float],
    student_best_high: dict[int, float],
    low_threshold: float,
    step_minutes: int,
    path: Path,
    *,
    rainfall_threshold_mmh: float = RAINFALL_THRESHOLD_MMH,
    high_coverage_pct: float = HIGH_COVERAGE_PCT,
):
    """KD summary JSON: rainfall-track-style metadata + BOTH tracks'
    metrics_per_lead / high-coverage lists + BOTH post_processing blocks
    (per-lead high tuned independently per model)."""
    lead_titles = [f"t+{o * step_minutes}" for o in LEAD_STEP_OFFSETS]

    def _build_side(name_col: str) -> tuple[dict, dict, dict, dict]:
        """Compute (above, diff_pct, high_cov_lists, metrics_per_lead) for
        one side (teacher or student) from the rows + confusion dicts."""
        above = {lt: {"iou": 0} for lt in lead_titles}
        high_cov = {lt: {"iou": []} for lt in lead_titles}
        for r in rows:
            for i, offset in enumerate(LEAD_STEP_OFFSETS):
                lt = lead_titles[i]
                if r[f"iou_{name_col}_t+{offset * step_minutes}"] >= high_coverage_pct:
                    above[lt]["iou"] += 1
                    high_cov[lt]["iou"].append(
                        [r["date"], r["reference_utc"]]
                    )
        total = len(rows)
        diff_pct = {}
        for lt in lead_titles:
            diff_pct[lt] = {}
            if total == 0:
                diff_pct[lt]["iou"] = 0.0
                continue
            diff_pct[lt]["iou"] = ((total - above[lt]["iou"]) / total) * 100.0
        conf_source = (teacher_conf_per_lead if name_col == "teacher"
                       else student_conf_per_lead)
        metrics_at_best = {
            lead_titles[i]: _summarise_confusion(conf_source[i])
            for i in range(len(LEAD_STEP_OFFSETS))
        }
        return above, diff_pct, high_cov, metrics_at_best

    def _pp_block(tuning: dict, best_high: dict) -> dict:
        tuning_named = {}
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            lt = lead_titles[i]
            tuning_named[lt] = {
                f"{h:.2f}": tuning[i][h] for h in sorted(tuning[i])
            }
        high_named = {
            f"t+{offset * step_minutes}": best_high[offset]
            for offset in LEAD_STEP_OFFSETS
        }
        return {
            "low_threshold": low_threshold,
            "high_grid": list(LIGHTNING_HIGH_GRID),
            "high_threshold_per_lead": high_named,
            "tuning_scores": tuning_named,
            "tuning_metric": "csi",
        }

    t_above, t_diff, t_hcov, t_metrics = _build_side("teacher")
    s_above, s_diff, s_hcov, s_metrics = _build_side("student")

    doc = {
        "track": "kd",
        "year": year, "month": month, "split": SPLIT,
        "threshold_mmh": rainfall_threshold_mmh,
        "selection_criterion": (
            f"OPERA-driven: >= {rainfall_threshold_mmh:g} mm/h anywhere on the "
            f"768x1536 canvas at the reference timestep (shared with rainfall + "
            f"lightning tracks for parity)."
        ),
        "high_coverage_threshold_pct": high_coverage_pct,
        "total_selected_samples": len(rows),
        "initial_selection": [[d, h] for d, h in selected],
        "teacher_mode": KD_TEACHER_MODE,
        "student_mode": KD_STUDENT_MODE,
        "student_hr_channels": KD_STUDENT_HR_CHANNELS,
        "teacher": {
            "samples_above_threshold_per_lead": t_above,
            "difference_pct_per_lead":         t_diff,
            "metrics_per_lead":                t_metrics,
            "high_coverage_samples_per_lead":  t_hcov,
            "post_processing":                 _pp_block(teacher_tuning, teacher_best_high),
        },
        "student": {
            "samples_above_threshold_per_lead": s_above,
            "difference_pct_per_lead":         s_diff,
            "metrics_per_lead":                s_metrics,
            "high_coverage_samples_per_lead":  s_hcov,
            "post_processing":                 _pp_block(student_tuning, student_best_high),
        },
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Wrote KD summary to {path}")


def _plot_metrics_figure_kd_per_metric(
    year: int, month: int,
    teacher_conf_per_lead: dict[int, dict],
    student_conf_per_lead: dict[int, dict],
    rows: list[dict],
    step_minutes: int,
    output_dir: Path,
):
    """Emit ONE figure per metric (FAR / POD / CSI / IoU%) with teacher
    vs student bars grouped per lead. Four files:
       kd_{yyyy}_{mm}_{FAR,POD,CSI,IoU}.png
    """
    lead_titles = [f"t+{o * step_minutes}" for o in LEAD_STEP_OFFSETS]
    stem = f"kd_{scope_stem(year, month)}"

    # Bar values per (metric, model, lead).
    t_agg = [_summarise_confusion(teacher_conf_per_lead[i])
             for i in range(len(LEAD_STEP_OFFSETS))]
    s_agg = [_summarise_confusion(student_conf_per_lead[i])
             for i in range(len(LEAD_STEP_OFFSETS))]

    def _iou_series(rows_side_key: str) -> list[float]:
        # Aggregate IoU per lead as the mean of the per-sample column so
        # the IoU figure lines up with what the CSV shows. FAR/POD/CSI use
        # the aggregate-confusion form since that's how the summary metric
        # is defined; IoU is a coverage ratio per sample, better averaged.
        vals = []
        for offset in LEAD_STEP_OFFSETS:
            col = f"iou_{rows_side_key}_t+{offset * step_minutes}"
            xs = [r[col] for r in rows]
            vals.append(float(np.mean(xs)) if xs else 0.0)
        return vals

    x = np.arange(len(lead_titles))
    width = 0.35
    t_color = "#1f77b4"   # blue
    s_color = "#ff7f0e"   # orange

    for metric in KD_METRIC_NAMES:
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        if metric in ("FAR", "POD", "CSI"):
            t_vals = [agg[metric] for agg in t_agg]
            s_vals = [agg[metric] for agg in s_agg]
            ylabel = f"{metric} (0..1)"
            ylim = (0.0, 1.0)
        else:  # IoU is a percentage per sample -> average
            t_vals = _iou_series("teacher")
            s_vals = _iou_series("student")
            ylabel = "Mean IoU per sample (%)"
            ylim = (0.0, 100.0)
        ax.bar(x - width / 2, t_vals, width,
               label="Teacher (mtg_lightning_opera_occurrence)",
               color=t_color, edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, s_vals, width,
               label="Student (mtg_opera_occurrence, KD)",
               color=s_color, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(lead_titles)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.set_title(
            f"{metric} - teacher vs student (KD) - "
            f"{scope_label(year, month)}  |  {len(rows)} selected samples",
            fontsize=12, fontweight="bold",
        )
        out = run_dir(output_dir, stem) / f"{stem}_metrics_{metric}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out.name}")


def _plot_kd_3x3(
    gt_canvases: list[np.ndarray | None],
    teacher_bin: list[np.ndarray],
    student_bin: list[np.ndarray],
    *,
    date_str: str, ref_utc: str, step_minutes: int,
    output_path: Path, suptitle_color: str = "black",
) -> None:
    """3 rows x 3 lead cols. Row 1 = GT alone; row 2 = GT + teacher red
    overlay; row 3 = GT + student red overlay. Same base colormap +
    border overlay as _plot_lightning_2x3 for visual continuity."""
    import matplotlib.colors as mcolors
    from visualize_gt_vs_pred import (
        H_FULL, W_FULL, overlay_borders, _ensure_view_cached,
        _gt_kwargs_for,
    )
    import visualize_gt_vs_pred as _vf

    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT
    gt_kwargs = _gt_kwargs_for("lightning")

    n_lead = len(LEAD_STEP_OFFSETS)
    fig, axes = plt.subplots(3, n_lead, figsize=(7 * n_lead, 12.5),
                             constrained_layout=True, squeeze=False)

    def _apply_frame(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(c_lo, c_hi)
        ax.set_ylim(r_hi, r_lo)
        ax.set_aspect("equal")

    row_labels = ("GT", "GT + teacher (red)", "GT + student (red)")
    for i, offset in enumerate(LEAD_STEP_OFFSETS):
        lead_min = offset * step_minutes
        gt = gt_canvases[i]

        # --- Row 1: GT alone ---
        ax_gt = axes[0, i]
        if gt is None:
            ax_gt.imshow(np.zeros((H_FULL, W_FULL), dtype=np.float32),
                         cmap="gray", vmin=0.0, vmax=1.0,
                         aspect="equal", interpolation="nearest")
            ax_gt.text(0.5, 0.5, "GT unavailable",
                       transform=ax_gt.transAxes, ha="center", va="center",
                       fontsize=13, color="#555")
        else:
            ax_gt.imshow(gt.astype(np.float32), **gt_kwargs)
        try: overlay_borders(ax_gt)
        except Exception: pass
        _apply_frame(ax_gt)
        ax_gt.set_title(f"GT - t+{lead_min} min", fontsize=11)

        for row_idx, (bin_can, colour) in enumerate(
            ((teacher_bin[i], "#d62728"),
             (student_bin[i], "#d62728")), start=1,
        ):
            ax = axes[row_idx, i]
            if gt is not None:
                ax.imshow(gt.astype(np.float32), **gt_kwargs)
            else:
                ax.imshow(np.zeros((H_FULL, W_FULL), dtype=np.float32),
                          cmap="gray", vmin=0.0, vmax=1.0,
                          aspect="equal", interpolation="nearest")
            overlay = np.where(bin_can > 0, 1.0, np.nan)
            ax.imshow(overlay, cmap=mcolors.ListedColormap([colour]),
                      vmin=0.5, vmax=1.5, aspect="equal", interpolation="nearest")
            try: overlay_borders(ax)
            except Exception: pass
            _apply_frame(ax)
            if gt is not None:
                gt_pos = gt > 0
                pr_pos = bin_can > 0
                hits = int((gt_pos & pr_pos).sum())
                misses = int((gt_pos & ~pr_pos).sum())
                fa = int((~gt_pos & pr_pos).sum())
                subtitle = (f"{row_labels[row_idx]}  t+{lead_min}\n"
                            f"{_format_hmf_pct(hits, misses, fa)}")
            else:
                subtitle = f"{row_labels[row_idx]}  t+{lead_min}"
            ax.set_title(subtitle, fontsize=10)

    fig.suptitle(
        f"KD comparison  |  {date_str}  ref={ref_utc}",
        fontsize=14, fontweight="bold", color=suptitle_color,
    )
    _add_hmf_legend(fig)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_extraction_kd(
    year: int, month: int,
    teacher_mode: str, student_mode: str,
    source: str,
    teacher_finetuned: bool, student_kd: bool,
    data_root: Path, model_dir: Path, output_dir: Path,
    *,
    stride: int = DEFAULT_STRIDE,
    low_threshold: float = LIGHTNING_LOW_THRESHOLD,
    high_grid: tuple[float, ...] = LIGHTNING_HIGH_GRID,
    batch_size: int = 32,
    rainfall_threshold_mmh: float = RAINFALL_THRESHOLD_MMH,
    high_coverage_pct: float = HIGH_COVERAGE_PCT,
    period=None,
):
    """KD extraction: load both models, run each on the same OPERA-selected
    samples with the same Hann-overlapped inputs (student sees only the
    last HR channel = vis_06), tune each model's per-lead high threshold
    independently, emit joint CSV / JSON / 4 metric figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Validation extraction - track=kd  {scope_label(year, month)}")
    print("=" * 70)
    print(f"  Teacher: {teacher_mode} ({source}"
          f"{' finetuned' if teacher_finetuned else ''})")
    print(f"  Student: {student_mode} ({source}"
          f"{' KD' if student_kd else ' base'})")
    print(f"  Post-proc: stride={stride}  low={low_threshold:.2f}  "
          f"high grid={list(high_grid)}")
    print(f"  Thresholds: rainfall_threshold_mmh={rainfall_threshold_mmh:g}  "
          f"high_coverage_pct={high_coverage_pct:g}")

    init_sequence_config(str(data_root), source, period=period)
    sync_window_from_sequence_config()
    set_normalization_stats_path(
        data_root / normalization_stats_name(source, period)
    )
    teacher_cfg = get_mode_config(teacher_mode)
    student_cfg = get_mode_config(student_mode)
    if teacher_cfg["label_type"] != "lightning":
        raise SystemExit(f"Teacher --mode {teacher_mode} is not lightning-headed.")
    if student_cfg["label_type"] != "lightning":
        raise SystemExit(f"Student --mode {student_mode} is not lightning-headed.")
    step_minutes = _load_step_minutes(data_root)

    print(f"\nSelecting samples via OPERA (>= {rainfall_threshold_mmh:g} mm/h) ...")
    selected = select_samples(data_root, year, month,
                              threshold_mmh=rainfall_threshold_mmh,
                              source=source, period=period)
    if not selected:
        print("No samples selected. Nothing to do.")
        return

    print(f"\nLoading TEACHER ...")
    teacher = load_model_artifact(model_dir, teacher_mode, source,
                                  teacher_finetuned)
    print(f"  Teacher params: {teacher.count_params():,}")
    print(f"Loading STUDENT ...")
    student = load_model_artifact(model_dir, student_mode, source,
                                  finetuned=False, kd=student_kd)
    print(f"  Student params: {student.count_params():,}")

    # Store per-(sample, side, lead_idx, high) confusion tuples.
    per_sample: list[dict] = []
    n_skipped = 0

    print(f"\nRunning both models on {len(selected)} samples "
          f"(Hann overlap, stride={stride}) ...")
    for k, (date_str, hhmm) in enumerate(selected, 1):
        ref_utc = f"{hhmm[:2]}:{hhmm[2:]}"
        if k == 1 or k % 20 == 0 or k == len(selected):
            print(f"  [{k}/{len(selected)}] {date_str} {ref_utc}")

        inputs, positions = build_inputs_for_reference_overlapped(
            data_root, teacher_cfg, date_str, ref_utc, step_minutes,
            stride=stride,
        )
        if not positions:
            n_skipped += 1
            continue
        student_inputs = _kd_slice_student_inputs(inputs)

        t_preds = teacher.predict(inputs, batch_size=batch_size, verbose=0)
        s_preds = student.predict(student_inputs, batch_size=batch_size, verbose=0)
        t_prob = paste_predictions_hann_blended(t_preds, positions)
        s_prob = paste_predictions_hann_blended(s_preds, positions)

        conf_t: dict[tuple[int, float], tuple[int, int, int, int]] = {}
        conf_s: dict[tuple[int, float], tuple[int, int, int, int]] = {}
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            gt_hhmm, gt_day = _resolve_gt(
                ref_utc, offset * step_minutes, date_str,
            )
            gt_bin = _load_gt_lightning_canvas(data_root, gt_day, gt_hhmm)
            if gt_bin is None:
                # No GT for this lead -> zero confusion so per-sample IoU
                # collapses to 0 without crashing the aggregation.
                for h in high_grid:
                    conf_t[(i, float(h))] = (0, 0, 0, H_FULL * W_FULL)
                    conf_s[(i, float(h))] = (0, 0, 0, H_FULL * W_FULL)
                continue
            for h in high_grid:
                t_bin = hysteresis_binary(
                    t_prob[i], low=low_threshold, high=float(h),
                )
                s_bin = hysteresis_binary(
                    s_prob[i], low=low_threshold, high=float(h),
                )
                conf_t[(i, float(h))] = _binary_confusion_lightning(gt_bin, t_bin)
                conf_s[(i, float(h))] = _binary_confusion_lightning(gt_bin, s_bin)
        per_sample.append({
            "date": date_str, "reference_utc": ref_utc,
            "conf_teacher": conf_t, "conf_student": conf_s,
        })

    print(f"\nDone Phase 1. {len(per_sample)} samples scored, "
          f"{n_skipped} skipped (missing inputs).")
    if not per_sample:
        print("No samples produced predictions.")
        return

    # ---- Phase 2: independent per-lead sweep for each model -----------
    def _tune(side_key: str) -> tuple[dict, dict, dict]:
        tuning: dict[int, dict[float, dict]] = {
            i: {} for i in range(len(LEAD_STEP_OFFSETS))
        }
        for i in range(len(LEAD_STEP_OFFSETS)):
            for h in high_grid:
                tp = fp = fn = tn = 0
                for s in per_sample:
                    t, f, n, tt = s[side_key][(i, float(h))]
                    tp += t; fp += f; fn += n; tn += tt
                tuning[i][float(h)] = _summarise_confusion(
                    {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
                )
        best: dict[int, float] = {}
        agg_conf: dict[int, dict] = {}
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            best_h = max(tuning[i], key=lambda h: tuning[i][h]["CSI"])
            best[offset] = best_h
            a = tuning[i][best_h]
            agg_conf[i] = {"TP": a["TP"], "FP": a["FP"],
                           "FN": a["FN"], "TN": a["TN"]}
        return tuning, best, agg_conf

    t_tuning, t_best, t_agg = _tune("conf_teacher")
    s_tuning, s_best, s_agg = _tune("conf_student")

    for offset in LEAD_STEP_OFFSETS:
        print(f"  t+{offset * step_minutes} min | "
              f"teacher best_high={t_best[offset]:.2f} "
              f"(CSI={t_tuning[LEAD_STEP_OFFSETS.index(offset)][t_best[offset]]['CSI']:.3f})  "
              f"student best_high={s_best[offset]:.2f} "
              f"(CSI={s_tuning[LEAD_STEP_OFFSETS.index(offset)][s_best[offset]]['CSI']:.3f})")

    # ---- Emit per-sample rows at each model's chosen best_high per lead
    rows: list[dict] = []
    for s in per_sample:
        row = {"date": s["date"], "reference_utc": s["reference_utc"]}
        for i, offset in enumerate(LEAD_STEP_OFFSETS):
            m = offset * step_minutes
            for side_key, best_map, out_key in (
                ("conf_teacher", t_best, "teacher"),
                ("conf_student", s_best, "student"),
            ):
                tp, fp, fn, _ = s[side_key][(i, best_map[offset])]
                per = _summarise_confusion({"TP": tp, "FP": fp, "FN": fn, "TN": 0})
                row[f"iou_{out_key}_t+{m}"] = _iou_lightning(tp, fp, fn)
                row[f"far_{out_key}_t+{m}"] = per["FAR"]
                row[f"pod_{out_key}_t+{m}"] = per["POD"]
                row[f"csi_{out_key}_t+{m}"] = per["CSI"]
        rows.append(row)

    stem = f"kd_{scope_stem(year, month)}"
    _write_csv_kd(rows, output_dir / f"{stem}_samples.csv", step_minutes)
    _write_json_kd(
        year, month, selected, rows,
        t_agg, s_agg, t_tuning, s_tuning, t_best, s_best,
        low_threshold, step_minutes,
        output_dir / f"{stem}_summary.json",
        rainfall_threshold_mmh=rainfall_threshold_mmh,
        high_coverage_pct=high_coverage_pct,
    )
    _plot_metrics_figure_kd_per_metric(
        year, month, t_agg, s_agg, rows, step_minutes, output_dir,
    )


def run_visualization_kd(
    year: int, month: int, date_str: str,
    teacher_mode: str, student_mode: str,
    source: str,
    teacher_finetuned: bool, student_kd: bool,
    data_root: Path, model_dir: Path, output_dir: Path,
    *,
    stride: int = DEFAULT_STRIDE,
    low_threshold: float = LIGHTNING_LOW_THRESHOLD,
    batch_size: int = 32,
    period=None,
):
    """One 3x3 figure per selected reference on `date_str`:
       Row 1 GT / Row 2 GT+teacher red / Row 3 GT+student red, cols = leads.
    Per-lead high thresholds come from the KD summary JSON (tuned during
    the extraction run)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kd_{scope_stem(year, month)}"
    summary = _load_summary_json(output_dir / f"{stem}_summary.json")
    if "teacher" not in summary or "student" not in summary:
        raise SystemExit(
            f"{stem}_summary.json is missing teacher/student blocks. "
            f"Re-run extraction (`--track kd` without --date)."
        )
    step_minutes = _load_step_minutes(data_root)

    def _load_pp_map(side: str) -> dict[int, float]:
        m: dict[int, float] = {}
        pp = summary[side]["post_processing"]["high_threshold_per_lead"]
        for offset in LEAD_STEP_OFFSETS:
            key = f"t+{offset * step_minutes}"
            if key not in pp:
                raise SystemExit(
                    f"{side}.post_processing.high_threshold_per_lead is missing {key}"
                )
            m[offset] = float(pp[key])
        return m

    t_high_per_lead = _load_pp_map("teacher")
    s_high_per_lead = _load_pp_map("student")

    if not _date_is_in(date_str, summary["initial_selection"]):
        raise SystemExit(
            f"Date {date_str} not in the initial selection for "
            f"{scope_label(year, month)}. Nothing to visualise."
        )

    init_sequence_config(str(data_root), source, period=period)
    sync_window_from_sequence_config()
    set_normalization_stats_path(
        data_root / normalization_stats_name(source, period)
    )
    teacher_cfg = get_mode_config(teacher_mode)

    refs = sorted(h for d, h in summary["initial_selection"] if d == date_str)
    if not refs:
        raise SystemExit(f"Selection has no references for {date_str}.")

    print("Loading TEACHER + STUDENT ...")
    teacher = load_model_artifact(model_dir, teacher_mode, source,
                                  teacher_finetuned)
    student = load_model_artifact(model_dir, student_mode, source,
                                  finetuned=False, kd=student_kd)
    print(f"  Teacher params: {teacher.count_params():,}")
    print(f"  Student params: {student.count_params():,}")

    lead_titles = [f"t+{o * step_minutes}" for o in LEAD_STEP_OFFSETS]
    for ref_utc in refs:
        print(f"\n{date_str} {ref_utc} - building 3x3 KD figure ...")
        inputs, positions = build_inputs_for_reference_overlapped(
            data_root, teacher_cfg, date_str, ref_utc, step_minutes,
            stride=stride,
        )
        if not positions:
            print("  No inputs available - skipping.")
            continue
        student_inputs = _kd_slice_student_inputs(inputs)
        t_preds = teacher.predict(inputs, batch_size=batch_size, verbose=0)
        s_preds = student.predict(student_inputs, batch_size=batch_size,
                                   verbose=0)
        t_prob = paste_predictions_hann_blended(t_preds, positions)
        s_prob = paste_predictions_hann_blended(s_preds, positions)
        t_bin = [hysteresis_binary(t_prob[i], low=low_threshold,
                                    high=t_high_per_lead[LEAD_STEP_OFFSETS[i]])
                 for i in range(len(t_prob))]
        s_bin = [hysteresis_binary(s_prob[i], low=low_threshold,
                                    high=s_high_per_lead[LEAD_STEP_OFFSETS[i]])
                 for i in range(len(s_prob))]
        gt_canvases = []
        for offset in LEAD_STEP_OFFSETS:
            gt_hhmm, gt_day = _resolve_gt(
                ref_utc, offset * step_minutes, date_str,
            )
            gt_canvases.append(
                _load_gt_lightning_canvas(data_root, gt_day, gt_hhmm)
            )

        # Colour rule: green if EITHER model cleared 90% on any lead;
        # orange if only in the initial selection (matches lightning viz).
        any_high = False
        for side in ("teacher", "student"):
            for lt in lead_titles:
                if _date_is_in(date_str,
                               summary[side]["high_coverage_samples_per_lead"][lt]["iou"]):
                    any_high = True; break
            if any_high: break
        suptitle_color = _colour_for_title(True, any_high)

        safe_ref = ref_utc.replace(":", "")
        out_png = run_dir(output_dir, stem) / f"{stem}_{date_str}_{safe_ref}.png"
        _plot_kd_3x3(
            gt_canvases, t_bin, s_bin,
            date_str=date_str, ref_utc=ref_utc, step_minutes=step_minutes,
            output_path=out_png, suptitle_color=suptitle_color,
        )
        marker = ("green (a model cleared 90% IoU on some lead)"
                  if any_high else "orange (in selection only)")
        print(f"  Saved -> {out_png.name}  [{marker}]")


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    global SPLIT, THRESHOLD_MMH, WEIGHTS, MONTH_BATCH, SEED, WORKERS
    parser = argparse.ArgumentParser(
        description="COALITION-4 validation branch. Extraction mode "
                    "takes the reference timesteps of a dataset split "
                    "(--split) or of a calendar month (--year --month), "
                    "keeps those with any OPERA pixel at or above the "
                    "selected threshold, runs inference, computes per-"
                    "sample coverage, and emits CSV + JSON + metrics "
                    "figure. Visualization mode reads the JSON and "
                    "plots structure-overlay + zoom for a given date.",
    )
    parser.add_argument("--plots", type=str, default=None,
                        metavar="SUMMARY_JSON",
                        help="Redraw every figure of a finished run from its "
                             "summary JSON and sibling samples CSV, then "
                             "exit: metrics bars, coverage scatter "
                             "(rainfall), hits / misses / false alarms, "
                             "their percentiles, tuning curves. Either "
                             "track. No model, no data; every other flag "
                             "is ignored.")
    parser.add_argument("--track", type=str, default=None,
                        choices=["rainfall", "lightning", "kd"],
                        help="Validation track. 'rainfall' is the OPERA "
                             "multiclass pipeline; 'lightning' runs the "
                             "Hann-blended overlap + hysteresis pipeline "
                             "and tunes the high threshold per lead time; "
                             "'kd' runs BOTH teacher (mtg_lightning_opera_"
                             "occurrence) and student (mtg_opera_occurrence, "
                             "loaded from the _kd checkpoint) on the same "
                             "OPERA-selected samples and tunes each model's "
                             "per-lead high threshold independently.")
    parser.add_argument("--split", type=str, default=None, choices=SPLITS,
                        help="Score the reference timesteps of this dataset "
                             "split (read from <split>_data_<source>"
                             "[_<period>].csv) instead of scanning a month. "
                             "With --year/--month, only the split's "
                             "timesteps of that month. Outputs are named "
                             "<track>_<split>[_<yyyy>_<mm>]_<tag>_*.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None,
                        help="Month as an integer 1..12. Required without "
                             "--split; with --split it restricts the split "
                             "to that month.")
    parser.add_argument("--date", type=str, default=None,
                        help="If given (YYYY-MM-DD), switches to "
                             "visualization mode against the JSON "
                             "produced by an earlier extraction run.")
    parser.add_argument("--mode", type=str, nargs="+",
                        default=["mtg_lightning_opera_rainfall"],
                        help="Model mode name(s). Several run one after the "
                             "other on the same track, each with its own "
                             "outputs; pair them with --period.")
    parser.add_argument("--finetuned", action="store_true",
                        help="Load coalition_<mode>_<source>_finetuned.keras "
                             "(rebuilt + load_weights via train_models."
                             "build_finetune_model). Mutually exclusive with "
                             "--kd.")
    parser.add_argument("--kd", action="store_true",
                        help="Load coalition_<mode>_<source>_kd.keras - the "
                             "knowledge-distillation student produced by "
                             "train_lightning_kd.py. Only meaningful for "
                             "--track lightning (evaluates the student "
                             "standalone) and --mode mtg_opera_occurrence. "
                             "Mutually exclusive with --finetuned.")
    parser.add_argument("--data_root", type=str, default=str(resolve_data_root()))
    parser.add_argument("--period", type=str, nargs="+", default=None,
                        metavar="LABEL",
                        help="Period label(s) the model(s) were trained "
                             "under, e.g. --period f34 w34, one per --mode "
                             "(one label serves every mode). Selects the "
                             "weights, the normalization statistics and the "
                             "sequence metadata together.")
    parser.add_argument("--model_dir", type=str, default=str(resolve_model_dir()))
    parser.add_argument("--weights", type=str, default="best",
                        choices=["best", "latest"],
                        help="Which saved state to load: `best`, the final save (best epoch, restored by early stopping), or `latest`, the per-epoch checkpoint under models/checkpoints/ (the last epoch run). Outputs of a `latest` run carry a _latest suffix.")
    parser.add_argument("--output_dir", type=str, default="./validation",
                        help="Where the summary JSON and samples CSV go "
                             "(default ./validation); a run's figures go to "
                             "<output_dir>/<stem>/.")
    # --- Lightning-only knobs (ignored when --track rainfall) ---
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="Overlap stride for Hann inference (lightning). "
                             f"Default {DEFAULT_STRIDE} = 50%% overlap.")
    parser.add_argument("--lightning_low_threshold", type=float, default=None,
                        help="Hysteresis LOW threshold (lightning). Default: "
                             "the threshold the evaluation tuned for this "
                             "model, read from --eval_root/eval_<tag>/"
                             "evaluation_results.json.")
    parser.add_argument("--eval_root", type=str, default="./evaluation",
                        help="Where evaluate_coalition wrote eval_<tag>/ "
                             "(the lightning LOW comes from there).")
    # NOTE: --min_active_pixels was removed - lightning selection is now
    # OPERA-driven for parity with the rainfall track. select_samples_lightning
    # (LINET-driven, >=1 active pixel) remains callable from Python for anyone
    # who deliberately wants a LINET-only cut.
    parser.add_argument("--rainfall_threshold_mmh", type=float,
                        default=RAINFALL_THRESHOLD_MMH,
                        help=f"OPERA rainfall threshold in mm/h for sample "
                             f"selection (both tracks - selection is shared). "
                             f"Default {RAINFALL_THRESHOLD_MMH:g}. NOTE: this "
                             f"overrides ONLY the selection cut and the "
                             f"metrics-figure label text; the binary event "
                             f"used for FAR/POD/CSI/IoU stays anchored to "
                             f"class >= 1 (10 mm/h), which is the trained "
                             f"model's decision boundary.")
    parser.add_argument("--high_coverage_pct", type=float,
                        default=HIGH_COVERAGE_PCT,
                        help=f"Coverage %% above which a sample is added to "
                             f"the high-coverage list per lead in the summary "
                             f"JSON, and above which the visualisation "
                             f"suptitle is coloured green. Default "
                             f"{HIGH_COVERAGE_PCT:g}. Lowering makes the "
                             f"grading more lenient; raising is stricter.")
    parser.add_argument("--rainfall_low_threshold", type=float, default=None,
                        help="Skip phase 1 and fix this LOW for every lead; "
                             "the window sweep still runs around it.")
    parser.add_argument("--rainfall_window", type=float, default=0.02,
                        help="Width of the (LOW, HIGH) windows of the "
                             "phase-2 sweep (default 0.02).")
    parser.add_argument("--rainfall_reach", type=float, default=0.10,
                        help="How far the windows slide on each side of "
                             "the phase-1 LOW (default 0.10).")
    parser.add_argument("--baseline", action="store_true",
                        help="Rainfall track: validate the SepConv-ens "
                             "baseline instead of a RECONVECT model. Its "
                             "class map is scored as is (no hysteresis). "
                             "Needs --period (the baseline's window, e.g. "
                             "w44); --mode is ignored.")
    parser.add_argument("--baseline_period", type=str, default=None,
                        metavar="LABEL",
                        help="Rainfall track: also validate the SepConv-ens "
                             "baseline trained on this window, after the "
                             "--mode models, in the same run.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap the scored samples (and the tuning "
                             "samples) at N, drawn reproducibly per month: "
                             "--month_batch from each month in turn, "
                             "shuffled with --seed, no repeats.")
    parser.add_argument("--month_batch", type=int, default=MONTH_BATCH,
                        help=f"Samples taken from each month per round of "
                             f"the draw (default {MONTH_BATCH}).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Seed of the draw (default {SEED}).")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"CPU threads for the per-lead sweeps "
                             f"(default {WORKERS}).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="model.predict batch size. For lightning the "
                             "Hann overlap produces ~55 patches per reference "
                             "so 32-64 is a good range.")
    # ---- KD-track-only knobs (ignored when --track != kd) ----
    parser.add_argument("--teacher_mode", type=str, default=KD_TEACHER_MODE,
                        help=f"KD teacher mode. Default {KD_TEACHER_MODE}.")
    parser.add_argument("--student_mode", type=str, default=KD_STUDENT_MODE,
                        help=f"KD student mode. Default {KD_STUDENT_MODE}.")
    parser.add_argument("--teacher_finetuned", action="store_true",
                        help="Load coalition_<teacher_mode>_<source>_finetuned.keras "
                             "instead of the base for the KD teacher.")
    parser.add_argument("--no_student_kd", action="store_true",
                        help="Load the STUDENT from coalition_<student_mode>_<source>"
                             ".keras (plain, no _kd suffix). Default is to load "
                             "the KD-trained student produced by train_lightning_kd.py.")
    args = parser.parse_args()

    if args.plots:
        src = Path(args.plots)
        if not src.is_file():
            parser.error(f"--plots: {src} not found")
        make_plots(src)
        return 0
    if args.track is None:
        parser.error("--track is required")

    if args.kd and args.finetuned:
        parser.error("--kd and --finetuned are mutually exclusive "
                     "(the KD student is trained fresh, no swin head).")

    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    global MAX_SAMPLES
    MAX_SAMPLES = args.max_samples

    if (args.year is None) != (args.month is None):
        parser.error("--year and --month go together")
    if args.split is None and args.year is None:
        parser.error("give --split, or --year and --month, or both")
    if args.month is not None and not (1 <= args.month <= 12):
        raise SystemExit(f"--month must be 1..12, got {args.month}")
    SPLIT = args.split
    THRESHOLD_MMH = float(args.rainfall_threshold_mmh)
    WEIGHTS = args.weights
    MONTH_BATCH = max(1, int(args.month_batch))
    SEED = int(args.seed)
    WORKERS = max(1, int(args.workers))

    # Prime the border cache once at process start (visualization uses it,
    # extraction ignores it but the cost is a few ms).
    _load_country_borders_pixels()

    # One job per model: (mode, period, baseline). Periods are zipped
    # with the modes; one period serves every mode.
    periods = args.period or [None]
    if len(periods) == 1:
        periods = periods * len(args.mode)
    if len(periods) != len(args.mode):
        parser.error(f"{len(args.mode)} --mode value(s) but {len(periods)} "
                     f"--period value(s); give one per mode or one for all")
    if args.baseline:
        jobs = [(args.mode[0], periods[0], True)]
    else:
        jobs = [(m, p, False) for m, p in zip(args.mode, periods)]
        if args.baseline_period:
            jobs.append((args.mode[0], args.baseline_period, True))
    if args.date is not None and len(jobs) != 1:
        parser.error("visualisation mode (--date) takes one model")

    def _release():
        """Free the GPU between models of one run."""
        import gc
        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()

    if args.track == "rainfall":
        if args.date is None:
            for n_job, (mode, period, baseline) in enumerate(jobs, 1):
                if len(jobs) > 1:
                    print(f"\n##### model {n_job}/{len(jobs)}: "
                          f"{'SepConv-ens baseline' if baseline else mode} "
                          f"({period}) #####")
                run_extraction(
                    args.track, args.year, args.month,
                    mode, SOURCE, args.finetuned,
                    data_root, model_dir, output_dir,
                    rainfall_threshold_mmh=args.rainfall_threshold_mmh,
                    high_coverage_pct=args.high_coverage_pct,
                    rainfall_low=args.rainfall_low_threshold,
                    rainfall_window=args.rainfall_window,
                    rainfall_reach=args.rainfall_reach,
                    period=period,
                    baseline=baseline,
                )
                _release()
        else:
            # Visualization mode reads high-coverage lists from the JSON
            # produced by extraction, so it inherits whatever
            # --high_coverage_pct was in effect then. --rainfall_threshold_mmh
            # is likewise a selection-time knob and has no effect here.
            run_visualization(
                args.track, args.year, args.month, args.date,
                jobs[0][0], SOURCE, args.finetuned,
                data_root, model_dir, output_dir,
                period=jobs[0][1],
            )
    elif args.track == "lightning":
        if args.baseline or args.baseline_period:
            parser.error("there is no lightning baseline")
        if args.date is None:
            for n_job, (mode, period, _b) in enumerate(jobs, 1):
                if len(jobs) > 1:
                    print(f"\n##### model {n_job}/{len(jobs)}: {mode} ({period}) #####")
                run_extraction_lightning(
                    args.year, args.month,
                    mode, SOURCE, args.finetuned,
                    data_root, model_dir, output_dir,
                    stride=args.stride,
                    low_threshold=args.lightning_low_threshold,
                    batch_size=args.batch_size,
                    rainfall_threshold_mmh=args.rainfall_threshold_mmh,
                    high_coverage_pct=args.high_coverage_pct,
                    kd=args.kd,
                    period=period,
                    eval_root=Path(args.eval_root),
                )
                _release()
        else:
            run_visualization_lightning(
                args.year, args.month, args.date,
                jobs[0][0], SOURCE, args.finetuned,
                data_root, model_dir, output_dir,
                stride=args.stride,
                low_threshold=args.lightning_low_threshold,
                batch_size=args.batch_size,
                kd=args.kd,
                period=jobs[0][1],
            )
    else:  # kd
        if args.date is None:
            run_extraction_kd(
                args.year, args.month,
                args.teacher_mode, args.student_mode, SOURCE,
                teacher_finetuned=args.teacher_finetuned,
                student_kd=(not args.no_student_kd),
                data_root=data_root, model_dir=model_dir,
                output_dir=output_dir,
                stride=args.stride,
                low_threshold=(args.lightning_low_threshold
                               if args.lightning_low_threshold is not None
                               else LIGHTNING_LOW_THRESHOLD),
                batch_size=args.batch_size,
                rainfall_threshold_mmh=args.rainfall_threshold_mmh,
                high_coverage_pct=args.high_coverage_pct,
                period=jobs[0][1],
            )
        else:
            run_visualization_kd(
                args.year, args.month, args.date,
                args.teacher_mode, args.student_mode, SOURCE,
                teacher_finetuned=args.teacher_finetuned,
                student_kd=(not args.no_student_kd),
                data_root=data_root, model_dir=model_dir,
                output_dir=output_dir,
                stride=args.stride,
                low_threshold=(args.lightning_low_threshold
                               if args.lightning_low_threshold is not None
                               else LIGHTNING_LOW_THRESHOLD),
                batch_size=args.batch_size,
                period=jobs[0][1],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
