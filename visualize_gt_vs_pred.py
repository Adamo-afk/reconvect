"""
visualize_gt_vs_pred.py
=======================
Training-scope visualiser: reads the per-source split CSVs the
training pipeline produces (train_data_<source>.csv /
validation_data_<source>.csv / test_data_<source>.csv), picks the
top-N reference timesteps by qualifying-patch count, runs one
batched model.predict per timestep, and saves a 2x3 GT (top row) vs
Pred (bottom row) figure per timestep plus a companion zoom-in on
the highest-activity patch.

For inference-only work on a date the training pipeline has never
touched (no split CSV, no extracted .npy patches), use
`predict_full_domain.py` instead. That script slices patches on the
fly from the reprojected full-domain fields and reuses the pred-panel
rendering here via `plot_full_domain_predictions_only` (defined
below), so both scripts produce pixel-identical Pred panels - only
the 2x3 GT vs Pred layout is exclusive to this file.

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
    python visualize_gt_vs_pred.py \
        --csv our_data/test_data_dbscan.csv \
        --mode mtg_lightning_opera_occurrence \
        --source dbscan --top_n 5

    # Same but fine-tuned model, manual threshold override
    python visualize_gt_vs_pred.py \
        --csv our_data/test_data_dbscan.csv \
        --mode mtg_lightning_opera_occurrence \
        --source dbscan --top_n 5 --finetuned --threshold 0.5

    # OPERA 5-class on the lightning-driven split CSV
    python visualize_gt_vs_pred.py \
        --csv our_data/validation_data_lightning.csv \
        --mode mtg_opera_mtgmr_rainfall \
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
    init_sequence_config,
    load_and_transform_group,
    load_label,
    set_normalization_stats_path,
    LABEL_CHANNELS,
)

from pipeline_config import SOURCE

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
# Country outline (cartopy Natural Earth preferred, pyproj fallback)
# ============================================================================
# Coarse hardcoded Romania boundary in (lon, lat) used as fallback when
# cartopy is not installed. Smooth enough to read as "this is Romania"
# but obviously rough vs. the Natural Earth 10m boundaries.
_ROMANIA_OUTLINE_LONLAT_FALLBACK = [
    (22.69, 47.99), (23.14, 48.10), (24.30, 47.91), (25.41, 47.93),
    (26.40, 48.22), (27.05, 47.99), (27.55, 47.40), (28.10, 46.81),
    (28.21, 45.97), (28.83, 45.30), (29.65, 45.18), (29.69, 44.81),
    (28.84, 44.05), (28.05, 43.81), (27.00, 44.13), (25.65, 43.69),
    (24.50, 43.68), (23.27, 43.83), (22.65, 44.22), (22.42, 44.71),
    (21.56, 44.77), (21.36, 45.04), (20.79, 45.46), (20.25, 46.10),
    (20.79, 46.30), (21.06, 46.83), (22.13, 47.59), (22.69, 47.99),
]

# Module-level cache for country boundaries in pixel coords. Populated
# lazily on first overlay call. With cartopy installed: all visible
# European countries within range of the data canvas; without cartopy:
# only Romania's coarse hardcoded polygon.
# Each entry is (col_px, row_px, name).
_BORDERS_PIXEL_CACHE: list[tuple[np.ndarray, np.ndarray, str]] | None = None
_BORDERS_SOURCE: str | None = None
# Romania-centred view extent (col_lo, col_hi, row_lo, row_hi), populated
# lazily alongside the border cache. The view is sized so the entire
# data canvas plus VIEW_EXTRA_PAD pixels of breathing room is always
# visible regardless of how off-centre Romania sits within the canvas.
_VIEW_EXTENT: tuple[float, float, float, float] | None = None

# Pixels of breathing room beyond the farthest data-canvas edge from
# Romania's centroid. Sized to roughly one patch-width (~256 px / 250 km
# in UTM 35N at the grid's 1 km/pixel resolution).
VIEW_EXTRA_PAD = 80

# Names of countries whose borders we want to render (alongside Romania).
# Names match Natural Earth 10m admin_0_countries; aliases included
# because the shapefile attribute key varies across NE versions.
NEIGHBOUR_NAMES = {
    "Hungary", "Serbia", "Bulgaria", "Moldova", "Republic of Moldova",
    "Ukraine", "Slovakia", "Austria", "Czech Republic", "Czechia",
    "Poland", "Belarus", "Russia", "Croatia", "Bosnia and Herz.",
    "Bosnia and Herzegovina", "Greece", "North Macedonia", "Macedonia",
    "Albania", "Italy", "Slovenia", "Turkey", "Montenegro", "Kosovo",
    "Republic of Serbia",
}


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


def _load_country_borders_pixels() -> tuple[list[tuple[np.ndarray, np.ndarray, str]], str]:
    """Resolve Romania and its neighbours to (col, row, name) tuples in
    canvas pixel coords. Each polygon ring becomes one entry; multi-
    polygons (e.g. islands - Romania has none but its neighbours do)
    become multiple entries.

    Strategy:
      1. If cartopy is importable: read Natural Earth 10m
         `admin_0_countries`, keep Romania + listed neighbours,
         convert every exterior + interior ring through pyproj to the
         shared (col_px, row_px) coordinate system. Pixels may go
         negative or exceed W_FULL/H_FULL - the caller is expected to
         set xlim/ylim wide enough to include them so the neighbour
         borders show up around the data canvas.
      2. Otherwise: only Romania's hardcoded polygon. Tagged
         "Romania" so the caller can still style it specifically.
    """
    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth(
            resolution="10m", category="cultural",
            name="admin_0_countries",
        )
        wanted = NEIGHBOUR_NAMES | {"Romania"}
        result: list[tuple[np.ndarray, np.ndarray, str]] = []
        for country in shpreader.Reader(shp).records():
            name = country.attributes.get("NAME") \
                or country.attributes.get("ADMIN") \
                or country.attributes.get("name")
            if name not in wanted:
                continue
            geom = country.geometry
            geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" \
                else [geom]
            for poly in geoms:
                lon = np.asarray(poly.exterior.coords.xy[0], dtype=np.float64)
                lat = np.asarray(poly.exterior.coords.xy[1], dtype=np.float64)
                col, row = latlon_to_pixel(lon, lat)
                result.append((col, row, name))
                for interior in poly.interiors:
                    lon_i = np.asarray(interior.coords.xy[0],
                                       dtype=np.float64)
                    lat_i = np.asarray(interior.coords.xy[1],
                                       dtype=np.float64)
                    c_i, r_i = latlon_to_pixel(lon_i, lat_i)
                    result.append((c_i, r_i, name))
        if result:
            return result, "natural_earth_10m"
    except Exception:
        pass

    lonlat = np.asarray(_ROMANIA_OUTLINE_LONLAT_FALLBACK, dtype=np.float64)
    col, row = latlon_to_pixel(lonlat[:, 0], lonlat[:, 1])
    return [(col, row, "Romania")], "hardcoded_coarse"


def overlay_borders(ax, *, color="black", linewidth=1.3):
    """Draw Romania + neighbour-country borders onto an axis that already
    shows the 768x1536 data canvas in default (pixel) coords. All borders
    use the same solid style; Romania is drawn last so it sits on top at
    boundary meetings. Caches geometry at module level."""
    global _BORDERS_PIXEL_CACHE, _BORDERS_SOURCE
    if _BORDERS_PIXEL_CACHE is None:
        _BORDERS_PIXEL_CACHE, _BORDERS_SOURCE = \
            _load_country_borders_pixels()
    # Neighbours first (under), Romania last (on top) so it stays
    # unambiguous when borders meet.
    for col, row, name in _BORDERS_PIXEL_CACHE:
        if name == "Romania":
            continue
        ax.plot(col, row, color=color, linewidth=linewidth, zorder=5)
    for col, row, name in _BORDERS_PIXEL_CACHE:
        if name != "Romania":
            continue
        ax.plot(col, row, color=color, linewidth=linewidth, zorder=6)


def _compute_view_extent() -> tuple[float, float, float, float]:
    """Romania-centred view extent in pixel coords.

    Strategy: place Romania's bbox centroid at the figure centre, then
    pick a half-extent large enough to contain BOTH the full data canvas
    (so no patch ever clips out of view) and a VIEW_EXTRA_PAD buffer of
    additional padding on every side. When Romania is off-centre within
    the data canvas (which it is - the canvas extends north past
    Romania's actual border), this gives more padding on the
    closer-to-Romania side and less on the farther side, which visually
    re-centres the country.

    Returns (col_lo, col_hi, row_lo, row_hi). row_lo < row_hi (call sites
    flip the y-axis afterwards via set_ylim(row_hi, row_lo) for image
    coords).
    """
    assert _BORDERS_PIXEL_CACHE is not None, \
        "Border cache must be populated before computing the view extent"
    ro_cols, ro_rows = [], []
    for col, row, name in _BORDERS_PIXEL_CACHE:
        if name == "Romania":
            ro_cols.append(col)
            ro_rows.append(row)
    if not ro_cols:
        # Pathological fallback: just centre on the canvas.
        c_x, c_y = W_FULL / 2.0, H_FULL / 2.0
    else:
        all_col = np.concatenate(ro_cols)
        all_row = np.concatenate(ro_rows)
        c_x = float((all_col.min() + all_col.max()) / 2)
        c_y = float((all_row.min() + all_row.max()) / 2)

    half_w = max(c_x - 0.0, W_FULL - c_x) + VIEW_EXTRA_PAD
    half_h = max(c_y - 0.0, H_FULL - c_y) + VIEW_EXTRA_PAD
    return (c_x - half_w, c_x + half_w,
            c_y - half_h, c_y + half_h)


def _ensure_view_cached():
    """Populate _BORDERS_PIXEL_CACHE / _BORDERS_SOURCE / _VIEW_EXTENT.
    Cheap when already cached."""
    global _BORDERS_PIXEL_CACHE, _BORDERS_SOURCE, _VIEW_EXTENT
    if _BORDERS_PIXEL_CACHE is None:
        _BORDERS_PIXEL_CACHE, _BORDERS_SOURCE = \
            _load_country_borders_pixels()
    if _VIEW_EXTENT is None:
        _VIEW_EXTENT = _compute_view_extent()


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
        inputs_dict: keys are model input names ("past_hr", "past_mr"),
                     each value an np.ndarray of shape
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
    for key in ("past_hr", "past_mr"):
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

    Active patches (those in `row['patch_numbers']`) are populated from
    the pre-extracted per-patch .npy tiles in `patches_dir` — the exact
    bytes the model was trained on. Inactive patches (the ones DBSCAN
    selection dropped) are filled from the reprojected full-domain
    canvas on disk so all 18 tiles carry real observational data
    instead of a placeholder, keeping the plot consistent between the
    two tracks (rainfall inactive shows class 0 dark viridis; lightning
    inactive shows any real occurrence that happened there, which is
    usually near-white "no lightning" but occasionally sparse strikes).

    If the full-canvas loader can't find a file for the label HHMM,
    the inactive tiles fall back to their pre-fix defaults (float32
    zeros for lightning, -1 sentinels for radar).

    Lightning: float32 in [0, 1], shape (H_FULL, W_FULL).
    Radar:     int32 class index in {0..4}; -1 marks "no reprojected
               data on disk for this lead" (rare fallback).
    """
    # Lazy imports: predict_full_domain and validate_predictions both
    # pull TF at module load. Deferring keeps this function callable
    # in TF-less contexts (tests, type checks) as long as it isn't hit.
    from predict_full_domain import (
        _load_gt_lightning_canvas, _load_gt_rainfall_canvas,
        _ref_to_hhmm as _pf_ref_to_hhmm,  # handles day rollover
    )
    from validate_predictions import _mmh_to_class

    date_str = row["date"]
    ref_utc = row["reference_utc"].strip()
    patches = ast.literal_eval(row["patch_numbers"])
    active_set = set(patches)

    label_var = mode_config["label_var"]
    label_transform = mode_config["label_transform"]
    label_suffix = mode_config["label_suffix"]
    n_label_ch = LABEL_CHANNELS[label_type]

    data_root = Path(patches_dir).parent  # patches_dir = data_root/patches

    canvases: list[np.ndarray] = []
    for t, col in enumerate(LABEL_TIME_COLS):
        label_hhmm = _ref_to_hhmm(ref_utc, LABEL_STEP_OFFSETS[t] * step_minutes)
        label_idxs = ast.literal_eval(row[col])

        if label_type == "lightning":
            canvas = np.zeros((H_FULL, W_FULL), dtype=np.float32)
        else:
            canvas = np.full((H_FULL, W_FULL), -1, dtype=np.int32)

        # --- Active patches from pre-extracted .npy tiles.
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

        # --- Inactive patches from the reprojected full canvas so the
        # plot shows real observed data everywhere (not a placeholder).
        # Uses predict_full_domain._ref_to_hhmm for day rollover on
        # labels that cross midnight.
        inactive_patches = [p for p in range(1, N_PATCHES + 1)
                            if p not in active_set]
        if inactive_patches:
            gt_hhmm, gt_day = _pf_ref_to_hhmm(
                ref_utc, LABEL_STEP_OFFSETS[t] * step_minutes, date_str,
            )
            if label_type == "lightning":
                full_canvas = _load_gt_lightning_canvas(
                    data_root, gt_day, gt_hhmm,
                )
                if full_canvas is not None:
                    for p in inactive_patches:
                        r0, r1, c0, c1 = get_patch_bounds(p)
                        canvas[r0:r1, c0:c1] = full_canvas[r0:r1, c0:c1] \
                            .astype(np.float32)
            else:  # radar
                mmh = _load_gt_rainfall_canvas(
                    data_root, gt_day, gt_hhmm,
                )
                if mmh is not None:
                    full_cls = _mmh_to_class(mmh).astype(np.int32)
                    for p in inactive_patches:
                        r0, r1, c0, c1 = get_patch_bounds(p)
                        canvas[r0:r1, c0:c1] = full_cls[r0:r1, c0:c1]

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


def build_full_soft_pred(predictions: np.ndarray,
                         valid_patches: list[int],
                         n_classes: int) -> list[np.ndarray]:
    """Radar-only companion to `build_full_pred` that keeps the raw
    softmax probabilities per class, so downstream code can run the
    rainfall hysteresis post-processing (which needs `p(argmax)` per
    pixel, not just the argmax label).

    Returns one (H_FULL, W_FULL, n_classes) float32 canvas per lead
    time. Non-qualifying patch slots stay zeroed — no valid probability
    distribution exists there, and 0 across every class is a natural
    'no prediction' marker (subsequent `argmax` still returns 0/dry so
    the pixel drops out of the hysteresis selection anyway).
    """
    T_future = predictions.shape[1]
    canvases: list[np.ndarray] = []
    for t in range(T_future):
        canvas = np.zeros((H_FULL, W_FULL, n_classes), dtype=np.float32)
        for p_pos, patch_num in enumerate(valid_patches):
            r0, r1, c0, c1 = get_patch_bounds(patch_num)
            canvas[r0:r1, c0:c1, :] = predictions[p_pos, t].astype(np.float32)
        canvases.append(canvas)
    return canvases


# ============================================================================
# Rainfall post-processing (hysteresis on p(argmax) when argmax is rainy)
# ============================================================================
DEFAULT_RAIN_LOW = 0.35
DEFAULT_RAIN_HIGH = 0.55


def rainfall_hysteresis(
    soft_canvas: np.ndarray,
    *,
    low: float = DEFAULT_RAIN_LOW,
    high: float = DEFAULT_RAIN_HIGH,
) -> np.ndarray:
    """Apply hysteresis thresholding to a rainfall softmax canvas.

    Rule: for every pixel, take the argmax over the 5-class softmax. If
    argmax == 0 (dry / R<10) the pixel is not a rain candidate. Otherwise
    the pixel's score for the hysteresis test is `p(argmax)` — the
    confidence with which the model picked that specific rainy class.
    We then apply the same connected-component hysteresis
    (`lightning_postproc.hysteresis_binary`) with the caller-supplied
    (low, high) thresholds and keep only the pixels selected by that
    mask. Selected pixels retain their argmax class label (1..4),
    everything else is written to 0 (dry).

    The default thresholds (low=0.35, high=0.55) are lower than the
    lightning ones because probability mass is split across 5 classes
    and a confident rainy prediction rarely exceeds 0.6 for any single
    class.

    Args:
        soft_canvas: (H, W, 5) softmax probabilities.
        low:  hysteresis lower threshold on p(argmax).
        high: hysteresis upper threshold on p(argmax).

    Returns:
        (H, W) int32 class canvas in {0..4}. All rejected pixels are 0.
    """
    from lightning_postproc import hysteresis_binary
    argmax = np.argmax(soft_canvas, axis=-1).astype(np.int32)
    p_argmax = np.take_along_axis(
        soft_canvas, argmax[..., None], axis=-1
    ).squeeze(-1)
    # Only rainy-class argmax pixels are eligible for the hysteresis
    # selection; dry-class argmax pixels start at score 0 so they never
    # cross even the low threshold.
    score = np.where(argmax > 0, p_argmax, 0.0).astype(np.float32)
    keep = hysteresis_binary(score, low=low, high=high).astype(bool)
    out = np.where(keep, argmax, 0).astype(np.int32)
    return out


# ============================================================================
# Plotting
# ============================================================================
def _plot_patch_grid(ax, *, color="black", linewidth=0.7,
                     linestyle=(0, (1, 3)),
                     valid_patches: list[int] | None = None,
                     number_active_color: str = "#1b7a1b",
                     number_inactive_color: str = "#c11515"):
    """Outline every 256x256 patch slot with a dashed/dotted rectangle so
    the viewer can read the 6x3 tile structure of the Romania canvas.
    Default linestyle is loosely-dotted (`(0, (1, 3))` = 1px on, 3px off).

    When `valid_patches` is supplied, each patch is additionally labelled
    with its 1-indexed number (1..18, row-major from upper-left) in
    green for active patches (present in `valid_patches`) and red for
    inactive ones. Replaces the older `_plot_radar_inactive_mask` hatch
    overlay — the numbers make it obvious which patches contributed to
    the prediction without hiding the underlying pixels.
    """
    valid_set = set(valid_patches) if valid_patches is not None else None
    for p in range(1, N_PATCHES + 1):
        r0, _, c0, _ = get_patch_bounds(p)
        rect = Rectangle(
            (c0, r0), PATCH_SIZE, PATCH_SIZE,
            linewidth=linewidth, edgecolor=color,
            linestyle=linestyle, facecolor="none",
            zorder=3,
        )
        ax.add_patch(rect)
        if valid_set is not None:
            txt_color = (number_active_color if p in valid_set
                         else number_inactive_color)
            ax.text(
                c0 + 3, r0 + 3, str(p),
                color=txt_color, fontsize=6, fontweight="bold",
                ha="left", va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.08",
                          facecolor="white", alpha=0.75,
                          edgecolor="none"),
            )


def _plot_radar_inactive_mask(ax, valid_patches: list[int]):
    """DEPRECATED: no longer called from the main plotters. Kept as a
    utility for callers that still want the classic hatch overlay on
    non-qualifying patches. The new-style patch numbering via
    `_plot_patch_grid(valid_patches=...)` (green/red per-patch labels)
    now conveys the same information without hiding the underlying
    pixel values."""
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


def _gt_kwargs_for(label_type: str) -> dict:
    """imshow kwargs for the GT panel of a given label type. Shared
    between plot_full_domain (2x3 GT vs Pred) and any other renderer
    that wants the same GT styling."""
    if label_type == "lightning":
        # Ramp starts at pure white so the domain rectangle reads as
        # 'no reading' rather than a subtle pink cast (user request:
        # drop the pink background from every lightning plot).
        gt_cmap = mcolors.LinearSegmentedColormap.from_list(
            "gt_red", ["#ffffff", "#67000d"]
        )
        return dict(cmap=gt_cmap, vmin=0.0, vmax=1.0,
                    aspect="equal", interpolation="nearest")
    return dict(cmap=plt.get_cmap("viridis", 5), vmin=0, vmax=4,
                aspect="equal", interpolation="nearest")


def _pred_kwargs_for(label_type: str, threshold: float | None) -> dict:
    """imshow kwargs for the Pred panel of a given label type. Shared
    between plot_full_domain (2x3) and plot_full_domain_predictions_only
    (1x3), so the inference-only script's single-row figure is
    guaranteed to match the bottom row of the 2x3 figure the training-
    scope script produces."""
    if label_type == "lightning":
        thr = threshold if threshold is not None else 0.5
        norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=thr, vmax=1.0)
        return dict(cmap="RdYlBu_r", norm=norm,
                    aspect="equal", interpolation="nearest")
    return dict(cmap=plt.get_cmap("viridis", 5), vmin=0, vmax=4,
                aspect="equal", interpolation="nearest")


def _apply_view_and_frame(ax):
    """Set the Romania-centred view extent + hide ticks. Shared so both
    the 2x3 and 1x3 figures use the identical framing."""
    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _VIEW_EXTENT
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_hi, r_lo)  # image y is flipped
    ax.set_aspect("equal")


def _render_gt_axes(ax, canvas: np.ndarray, label_type: str,
                    *, lead_title: str, lead_hhmm: str,
                    gt_kwargs: dict,
                    valid_patches: list[int] | None = None,
                    show_patch_numbers: bool = False):
    """Render a single GT panel onto `ax`. Returns the imshow handle so
    the caller can wire a colorbar.

    When `show_patch_numbers=True` AND `valid_patches` is supplied, the
    patch grid is annotated with green (active) / red (inactive) 1..18
    numbers. This is the training-scope 2x3 use case
    (plot_full_domain / finetuned): the CSV drives a partial set of
    qualifying patches, so numbering makes the coverage explicit.
    Inference callers reconstruct the whole grid — all 18 patches are
    always active — so they should leave the default False and skip
    the redundant numbering.
    """
    if label_type == "radar":
        # Fill non-qualifying slots (canvas == -1) with class 0 (darkest
        # viridis / "R<10") so all 18 patches paint in the viridis-5
        # palette rather than showing as blank pixels. The green/red
        # numbering from _plot_patch_grid then tells the reader which
        # tiles were actually processed vs. filled in as dark background.
        display = np.where(canvas < 0, 0.0, canvas.astype(float))
        im = ax.imshow(display, **gt_kwargs)
    else:
        im = ax.imshow(canvas, **gt_kwargs)
    _plot_patch_grid(
        ax,
        valid_patches=(valid_patches if show_patch_numbers else None),
    )
    try:
        overlay_borders(ax)
    except Exception:
        pass
    ax.set_title(f"GT — {lead_title} ({lead_hhmm} UTC)", fontsize=11)
    _apply_view_and_frame(ax)
    return im


def _render_pred_lightning_zone_axes(
    ax,
    gt_canvas: np.ndarray,
    bin_canvas: np.ndarray,
    *,
    lead_title: str,
    lead_hhmm: str,
    extent: tuple[float, float, float, float] | None = None,
    valid_patches: list[int] | None = None,
    show_patch_numbers: bool = False,
) -> dict:
    """Row 3 renderer for lightning: post-processed hit/miss/FA overlay.

    Palette matches `predict_full_domain._plot_lightning_2x3`:
      orange   = hit  (GT-active + post-proc positive)
      blue     = miss (GT-active + post-proc negative)
      red      = false alarm  (GT-inactive + post-proc positive)
      white    = correct dry  (matches Row 1's gt_red cmap low end)

    When `extent` is set the RGBA image is placed in those data coords —
    used by `plot_zoom_patch` so every panel draws in the full-canvas
    coordinate system but is cropped to a single patch via xlim/ylim.

    Returns {"hits", "misses", "false_alarms"} px counts so callers can
    re-use them (e.g. for aggregate tallies).
    """
    from validate_predictions import (
        _ZONE_ORANGE, _ZONE_BLUE, _ZONE_RED, _format_hmf_pct,
    )
    _GT_RED_LOW = (1.0, 1.0, 1.0, 1.0)   # white (was #fff5f0)
    gt_pos = gt_canvas > 0
    pr_pos = bin_canvas > 0
    H, W = gt_canvas.shape
    rgba = np.empty((H, W, 4), dtype=np.float32)
    rgba[:] = _GT_RED_LOW
    rgba[gt_pos & pr_pos] = _ZONE_ORANGE
    rgba[gt_pos & ~pr_pos] = _ZONE_BLUE
    rgba[~gt_pos & pr_pos] = _ZONE_RED
    hits = int((gt_pos & pr_pos).sum())
    misses = int((gt_pos & ~pr_pos).sum())
    false_alarms = int((~gt_pos & pr_pos).sum())
    imshow_kwargs = dict(aspect="equal", interpolation="nearest")
    if extent is not None:
        imshow_kwargs["extent"] = extent
    ax.imshow(rgba, **imshow_kwargs)
    _plot_patch_grid(
        ax,
        valid_patches=(valid_patches if show_patch_numbers else None),
    )
    try:
        overlay_borders(ax)
    except Exception:
        pass
    ax.set_title(
        f"Post-processing (hysteresis) overlap — {lead_title} "
        f"({lead_hhmm} UTC)\n"
        f"{_format_hmf_pct(hits, misses, false_alarms)}",
        fontsize=10,
    )
    return {"hits": hits, "misses": misses, "false_alarms": false_alarms}


def _render_pred_rainfall_hyst_axes(
    ax,
    gt_class_canvas: np.ndarray,
    hyst_class_canvas: np.ndarray,
    *,
    lead_title: str,
    lead_hhmm: str,
    low: float,
    high: float,
    extent: tuple[float, float, float, float] | None = None,
    valid_patches: list[int] | None = None,
    show_patch_numbers: bool = False,
    stats_crop: tuple[int, int, int, int] | None = None,
) -> dict:
    """Row 3 renderer for rainfall: zone overlap between the
    hysteresis-cleaned pred and GT.

    Reuses `validate_predictions._plot_zone_overlap_axis` so the palette
    (orange = hit, blue = miss, red = false alarm, white = correct dry)
    and the figure-level colour + formula legends stay identical to
    the lightning Row 3 and the predict_full_domain rainfall figure.

    Patch numbering (green = active, red = inactive) is opt-in via
    `show_patch_numbers=True` — the training-scope caller keeps it on
    so the reader still sees which patches DBSCAN selected.

    `stats_crop=(r0, r1, c0, c1)` restricts the hits/misses/false-alarms
    numbers in the subtitle to that pixel window WITHOUT changing what
    gets rendered (the full canvas is still drawn — the zoom caller
    then crops the view via xlim/ylim). Used by `plot_zoom_patch` so
    the reported percentages describe the zoomed patch alone.

    Returns the zone stats dict — either whole-canvas or crop-restricted
    depending on `stats_crop`.
    """
    from validate_predictions import (
        _plot_zone_overlap_axis, _format_hmf_pct,
        _postproc_gt_class_canvas,
    )
    # NOTE: _plot_zone_overlap_axis draws the RGBA canvas itself and
    # doesn't honour an `extent` kwarg (zoom callers work by cropping
    # gt_canvas and hyst_canvas before calling — see plot_zoom_patch).
    stats = _plot_zone_overlap_axis(ax, gt_class_canvas, hyst_class_canvas)
    _plot_patch_grid(
        ax,
        valid_patches=(valid_patches if show_patch_numbers else None),
    )
    if stats_crop is not None:
        # Recompute hits/misses/false-alarms restricted to the zoom
        # window, mirroring _plot_zone_overlap_axis's logic exactly
        # (GT is post-processed on the FULL canvas first — small-blob
        # rejection has to see the whole neighbourhood before we crop —
        # then we slice both the post-processed GT and the pred).
        r0, r1, c0, c1 = stats_crop
        gt_eff_full = _postproc_gt_class_canvas(gt_class_canvas)
        gt_eff = gt_eff_full[r0:r1, c0:c1]
        pred_crop = hyst_class_canvas[r0:r1, c0:c1]
        valid = gt_eff != -1
        gt_pos = (gt_eff >= 1) & valid
        pr_pos = (pred_crop >= 1) & valid
        stats = {
            "hits": int((gt_pos & pr_pos).sum()),
            "misses": int((gt_pos & ~pr_pos).sum()),
            "false_alarms": int((pr_pos & ~gt_pos).sum()),
        }
    ax.set_title(
        f"Post-processing (hysteresis) zone-overlap\n"
        f"(low={low:.2f}, high={high:.2f}) — {lead_title} "
        f"({lead_hhmm} UTC)\n"
        f"{_format_hmf_pct(stats['hits'], stats['misses'], stats['false_alarms'])}",
        fontsize=10,
    )
    return stats


def _render_pred_axes(ax, canvas: np.ndarray, valid_patches: list[int],
                     label_type: str, threshold: float | None,
                     *, lead_title: str, lead_hhmm: str,
                     pred_kwargs: dict,
                     show_patch_numbers: bool = False):
    """Render a single Pred panel onto `ax`. Returns the imshow handle.

    Patch grid numbering (green = active, red = inactive) is opt-in via
    `show_patch_numbers=True`. Turn it on for the training-scope 2x3
    plot_full_domain figure (both base and finetuned) where the CSV
    drives a partial patch set; leave it off for inference/validation
    where the whole 18-patch grid is always in play and the numbers
    would just be noise.
    """
    if label_type == "radar":
        # Same "fill inactive with class 0 dark viridis" treatment as
        # _render_gt_axes so all 18 tiles read as filled.
        display = np.where(canvas < 0, 0.0, canvas.astype(float))
        im = ax.imshow(display, **pred_kwargs)
    else:
        im = ax.imshow(canvas, **pred_kwargs)
    _plot_patch_grid(
        ax,
        valid_patches=(valid_patches if show_patch_numbers else None),
    )
    try:
        overlay_borders(ax)
    except Exception:
        pass
    if label_type == "lightning":
        thr_for_count = threshold if threshold is not None else 0.5
        title_suffix = f"(≥{thr_for_count:.2f})"
    else:
        title_suffix = ""
    ax.set_title(f"Pred {title_suffix} — {lead_title} ({lead_hhmm} UTC)",
                 fontsize=11)
    _apply_view_and_frame(ax)
    return im


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
    row3_canvases: list[np.ndarray] | None = None,
    postproc_low: float | None = None,
    postproc_high_per_lead: dict[int, float] | None = None,
    postproc_high: float | None = None,
):
    """Draw the GT / Pred / Post-proc 3x3 figure for one reference timestep.

    Row 1 (GT), Row 2 (Pred, raw) — unchanged from the historical 2x3
    layout. Row 3 shows the post-processed output:
      lightning: Hann-blended + hysteresis binary rendered as a
                 hit / miss / false-alarm overlay on the same
                 gt_red-low white base Row 1 uses.
      rainfall:  hysteresis-cleaned argmax (rainfall_hysteresis), same
                 viridis-5 palette as Rows 1/2 but dry pixels rendered
                 in light grey so 'not selected' reads distinctly from
                 the darkest viridis end.

    `row3_canvases` is either:
      - a list of 3 int8 binary canvases (lightning), or
      - a list of 3 int32 class canvases (rainfall), or
      - None, in which case the figure falls back to the historical
        2-row layout (used only when the caller can't provide post-
        processed output for the given label_type).

    `postproc_low` + `postproc_high_per_lead` (lightning) or
    `postproc_low` + `postproc_high` (rainfall) drive the Row 3 subtitle
    annotations.
    """
    lead_titles = [f"t+{o * step_minutes}" for o in LABEL_STEP_OFFSETS]
    label_offsets_min = [o * step_minutes for o in LABEL_STEP_OFFSETS]

    has_row3 = row3_canvases is not None
    n_rows = 3 if has_row3 else 2
    fig_height = 14 if has_row3 else 10
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, fig_height),
                             constrained_layout=True)

    gt_kwargs = _gt_kwargs_for(label_type)
    pred_kwargs = _pred_kwargs_for(label_type, threshold)

    im_gt = im_pred = None
    for t in range(3):
        lead_hhmm = _ref_to_hhmm(ref_utc, label_offsets_min[t])
        im_gt = _render_gt_axes(
            axes[0, t], gt_canvases[t], label_type,
            lead_title=lead_titles[t], lead_hhmm=lead_hhmm,
            gt_kwargs=gt_kwargs,
            valid_patches=valid_patches,
            show_patch_numbers=True,
        )
        im_pred = _render_pred_axes(
            axes[1, t], pred_canvases[t], valid_patches, label_type,
            threshold, lead_title=lead_titles[t], lead_hhmm=lead_hhmm,
            pred_kwargs=pred_kwargs,
            show_patch_numbers=True,
        )
        if has_row3:
            ax_r3 = axes[2, t]
            if label_type == "lightning":
                _render_pred_lightning_zone_axes(
                    ax_r3, gt_canvases[t], row3_canvases[t],
                    lead_title=lead_titles[t], lead_hhmm=lead_hhmm,
                    valid_patches=valid_patches,
                    show_patch_numbers=True,
                )
                _apply_view_and_frame(ax_r3)
            else:  # rainfall
                # Per-lead high threshold isn't a concept for rainfall
                # (single --rainfall_high_threshold across leads), so
                # fall back to the scalar value when the dict form isn't
                # provided.
                hi = (postproc_high if postproc_high is not None
                      else (postproc_high_per_lead or {}).get(
                          LABEL_STEP_OFFSETS[t], 0.0))
                _render_pred_rainfall_hyst_axes(
                    ax_r3, gt_canvases[t], row3_canvases[t],
                    lead_title=lead_titles[t], lead_hhmm=lead_hhmm,
                    low=(postproc_low if postproc_low is not None else 0.0),
                    high=hi,
                    valid_patches=valid_patches,
                    show_patch_numbers=True,
                )
                _apply_view_and_frame(ax_r3)

    # Colorbars (one per row, but Row 3 shares Row 2's colorbar where
    # the palette matches).
    if label_type == "lightning":
        # GT is a binary 0/1 canvas — the gt_red gradient colourbar
        # was redundant, so we drop it. To keep all three rows the same
        # width under constrained_layout, we attach an INVISIBLE
        # colourbar of the same shape as Row 2's; matplotlib reserves
        # the width but renders nothing.
        cax_gt_spacer = fig.colorbar(
            im_gt, ax=axes[0, :].ravel().tolist(),
            shrink=0.7, pad=0.01, location="right",
        )
        cax_gt_spacer.ax.set_visible(False)
        cax_pred = fig.colorbar(im_pred, ax=axes[1, :].ravel().tolist(),
                                shrink=0.7, pad=0.01, location="right")
        cax_pred.set_label("Probability")
        if has_row3:
            # Row 3 uses the orange / blue / red zone palette — no
            # scalar colourbar. Add the standard hit/miss/FA legend +
            # formula footer instead.
            from validate_predictions import (
                _add_zone_color_legend, _add_hmf_legend,
            )
            _add_zone_color_legend(fig)
            _add_hmf_legend(fig, y=-0.10)
            # Row 3 also needs a same-width invisible colourbar slot
            # so it stays aligned with Rows 1 and 2.
            cax_spacer = fig.colorbar(
                im_pred, ax=axes[2, :].ravel().tolist(),
                shrink=0.7, pad=0.01, location="right",
            )
            cax_spacer.ax.set_visible(False)
    else:
        # Rainfall: Rows 1 and 2 read against the same viridis-5 class
        # palette, so they share ONE colourbar. Row 3 now uses the
        # zone-overlap palette (orange/blue/red) — no scalar colourbar
        # for it; instead we attach a same-shape invisible spacer so
        # its panels align with Rows 1/2, and place the standard zone
        # colour legend + HMF formula footer below the figure.
        upper_axes = (axes[0:2, :].ravel().tolist()
                      if has_row3 else axes.ravel().tolist())
        cax = fig.colorbar(im_gt, ax=upper_axes,
                           ticks=[0, 1, 2, 3, 4],
                           shrink=0.7, pad=0.01, location="right")
        cax.set_ticklabels(RADAR_CLASS_NAMES)
        cax.set_label("Rain-rate class")
        if has_row3:
            spacer = fig.colorbar(
                im_gt, ax=axes[2, :].ravel().tolist(),
                shrink=0.7, pad=0.01, location="right",
            )
            spacer.ax.set_visible(False)
            from validate_predictions import (
                _add_zone_color_legend, _add_hmf_legend,
            )
            _add_zone_color_legend(fig)
            _add_hmf_legend(fig, y=-0.10)

    fig.suptitle(
        f"{'Lightning' if label_type == 'lightning' else 'OPERA 5-class'} "
        f"prediction — Date: {date_str}  |  Ref: {ref_utc} UTC  |  "
        f"Patches: {len(valid_patches)}/{N_PATCHES}",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_full_domain_predictions_only(
    pred_canvases: list[np.ndarray],
    valid_patches: list[int],
    label_type: str,
    *,
    date_str: str,
    ref_utc: str,
    threshold: float | None,
    output_path: Path,
    step_minutes: int,
    suptitle_prefix: str = "Inference",
):
    """Draw a 1x3 Pred-only figure for one reference timestep.

    Uses the SAME _render_pred_axes helper as plot_full_domain, so the
    output is guaranteed to be pixel-identical to the bottom row of the
    2x3 figure that plot_full_domain would produce. Meant for
    inference-only callers (predict_full_domain.py) that have no ground
    truth to render.

    `suptitle_prefix` lets the caller distinguish the figure title from
    the training-scope "prediction" wording, e.g. "Inference" vs the
    default 2x3 "Lightning prediction" phrasing.
    """
    lead_titles = [f"t+{o * step_minutes}" for o in LABEL_STEP_OFFSETS]
    label_offsets_min = [o * step_minutes for o in LABEL_STEP_OFFSETS]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5),
                             constrained_layout=True)
    pred_kwargs = _pred_kwargs_for(label_type, threshold)

    im_pred = None
    for t in range(3):
        lead_hhmm = _ref_to_hhmm(ref_utc, label_offsets_min[t])
        im_pred = _render_pred_axes(
            axes[t], pred_canvases[t], valid_patches, label_type,
            threshold, lead_title=lead_titles[t], lead_hhmm=lead_hhmm,
            pred_kwargs=pred_kwargs,
        )

    if label_type == "lightning":
        cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(),
                            shrink=0.7, pad=0.01, location="right")
        cbar.set_label("Probability")
    else:
        cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(),
                            ticks=[0, 1, 2, 3, 4],
                            shrink=0.7, pad=0.01, location="right")
        cbar.set_ticklabels(RADAR_CLASS_NAMES)
        cbar.set_label("Rain-rate class")

    kind = "Lightning" if label_type == "lightning" else "OPERA 5-class"
    fig.suptitle(
        f"{suptitle_prefix} - {kind} - Date: {date_str}  |  "
        f"Ref: {ref_utc} UTC  |  Patches: {len(valid_patches)}/{N_PATCHES}",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Zoom plot for the highest-activity patch
# ============================================================================
def find_highest_activity_patch(
    gt_canvases: list[np.ndarray],
    valid_patches: list[int],
    label_type: str,
) -> tuple[int, int]:
    """Pick the qualifying patch with the most GT activity summed across
    all lead times.

    Activity =
      - lightning: count of pixels where the GT occurrence flag is 1
        (anything above 0 - the label transform clips to {0, 1}).
      - radar: count of pixels classified to ANY non-zero rain class
        (0 = R<10 means dry, -1 = no qualifying patch).

    Returns (patch_number, activity_score). If no patch has any
    activity, returns the first valid patch with score 0.
    """
    best_patch = valid_patches[0]
    best_score = -1
    for p in valid_patches:
        r0, r1, c0, c1 = get_patch_bounds(p)
        score = 0
        for canvas in gt_canvases:
            tile = canvas[r0:r1, c0:c1]
            if label_type == "lightning":
                score += int(np.sum(tile > 0))
            else:
                score += int(np.sum((tile > 0) & (tile != -1)))
        if score > best_score:
            best_score = score
            best_patch = p
    return best_patch, max(best_score, 0)


def _format_pixel_to_latlon(col_px: float, row_px: float) -> str:
    """Reverse the pixel->lat/lon mapping for a corner annotation."""
    try:
        import pyproj
        xmin, xmax, ymin, ymax = ROMANIA_EXTENT_UTM
        x = xmin + (col_px / W_FULL) * (xmax - xmin)
        y = ymax - (row_px / H_FULL) * (ymax - ymin)
        transformer = pyproj.Transformer.from_crs(
            "EPSG:31700", "EPSG:4326", always_xy=True,
        )
        lon, lat = transformer.transform(x, y)
        return f"{lat:.2f}°N, {lon:.2f}°E"
    except Exception:
        return ""


def plot_zoom_patch(
    gt_canvases: list[np.ndarray],
    pred_canvases: list[np.ndarray],
    patch_num: int,
    activity_score: int,
    label_type: str,
    *,
    date_str: str,
    ref_utc: str,
    threshold: float | None,
    output_path: Path,
    step_minutes: int,
    row3_canvases: list[np.ndarray] | None = None,
    postproc_low: float | None = None,
    postproc_high_per_lead: dict[int, float] | None = None,
    postproc_high: float | None = None,
):
    """Save a zoomed 3-row figure for one specific 256x256 patch.

    Same GT / Pred / Post-proc 3-leadtime layout as plot_full_domain but
    cropped to the chosen patch. Country borders that intersect the
    patch are drawn inside the cropped frame so the location stays
    identifiable; the patch's lat/lon centre is reported in the figure
    suptitle so the operator can match it to a real-world region.

    Passing `row3_canvases=None` falls back to the historical 2-row
    layout (no post-processing available for the label_type).
    """
    r0, r1, c0, c1 = get_patch_bounds(patch_num)
    lead_titles = [f"t+{o * step_minutes}" for o in LABEL_STEP_OFFSETS]
    label_offsets_min = [o * step_minutes for o in LABEL_STEP_OFFSETS]

    has_row3 = row3_canvases is not None
    n_rows = 3 if has_row3 else 2
    fig_height = 13 if has_row3 else 9.5
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, fig_height),
                             constrained_layout=True)

    # GT styling identical to plot_full_domain so the zoom reads as a
    # consistent panel of the same figure.
    if label_type == "lightning":
        # White low end (was #fff5f0) — pink background removed from
        # every lightning figure per user request.
        gt_cmap = mcolors.LinearSegmentedColormap.from_list(
            "gt_red", ["#ffffff", "#67000d"]
        )
        gt_kwargs = dict(cmap=gt_cmap, vmin=0.0, vmax=1.0,
                         aspect="equal", interpolation="nearest",
                         extent=(c0, c1, r1, r0))
    else:
        gt_kwargs = dict(cmap=plt.get_cmap("viridis", 5),
                         vmin=0, vmax=4,
                         aspect="equal", interpolation="nearest",
                         extent=(c0, c1, r1, r0))

    for t in range(3):
        ax = axes[0, t]
        canvas = gt_canvases[t]
        tile = canvas[r0:r1, c0:c1]
        if label_type == "radar":
            display = np.where(tile < 0, np.nan, tile.astype(float))
            im_gt = ax.imshow(display, **gt_kwargs)
        else:
            im_gt = ax.imshow(tile, **gt_kwargs)
        # Country borders inside the patch frame (matplotlib auto-clips).
        try:
            overlay_borders(ax)
        except Exception:
            pass
        ax.set_title(f"GT — {lead_titles[t]} "
                     f"({_ref_to_hhmm(ref_utc, label_offsets_min[t])} UTC)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)

    # Predictions row
    if label_type == "lightning":
        thr = threshold if threshold is not None else 0.5
        pred_norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=thr, vmax=1.0)
        pred_kwargs = dict(cmap="RdYlBu_r", norm=pred_norm,
                           aspect="equal", interpolation="nearest",
                           extent=(c0, c1, r1, r0))
    else:
        pred_kwargs = dict(cmap=plt.get_cmap("viridis", 5),
                           vmin=0, vmax=4,
                           aspect="equal", interpolation="nearest",
                           extent=(c0, c1, r1, r0))

    for t in range(3):
        ax = axes[1, t]
        canvas = pred_canvases[t]
        tile = canvas[r0:r1, c0:c1]
        if label_type == "radar":
            display = np.where(tile < 0, np.nan, tile.astype(float))
            im_pred = ax.imshow(display, **pred_kwargs)
        else:
            im_pred = ax.imshow(tile, **pred_kwargs)
        try:
            overlay_borders(ax)
        except Exception:
            pass
        if label_type == "lightning":
            thr_for_count = threshold if threshold is not None else 0.5
            above = int(np.sum(tile >= thr_for_count))
            ax.text(
                c0 + 4, r1 - 6,
                f"pixels≥{thr_for_count:.2f}={above}  "
                f"max={tile.max():.3f}",
                color="white", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="black", alpha=0.55, edgecolor="none"),
                va="bottom", ha="left", zorder=7,
            )
            title_suffix = f"(≥{thr_for_count:.2f})"
        else:
            title_suffix = ""
        ax.set_title(f"Pred {title_suffix} — {lead_titles[t]} "
                     f"({_ref_to_hhmm(ref_utc, label_offsets_min[t])} UTC)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)

    # Row 3 (post-processed) — cropped to the patch via extent + xlim.
    if has_row3:
        for t in range(3):
            ax = axes[2, t]
            lead_hhmm = _ref_to_hhmm(ref_utc, label_offsets_min[t])
            row3_tile = row3_canvases[t][r0:r1, c0:c1]
            if label_type == "lightning":
                gt_tile = gt_canvases[t][r0:r1, c0:c1]
                _render_pred_lightning_zone_axes(
                    ax, gt_tile, row3_tile,
                    lead_title=lead_titles[t], lead_hhmm=lead_hhmm,
                    extent=(c0, c1, r1, r0),
                )
            else:
                hi = (postproc_high if postproc_high is not None
                      else (postproc_high_per_lead or {}).get(
                          LABEL_STEP_OFFSETS[t], 0.0))
                # Pass FULL canvases (not the cropped tile) — the
                # underlying `_plot_zone_overlap_axis` draws the RGBA
                # at natural (0..W, 0..H) coords and doesn't honour
                # extent. The post-render xlim/ylim below then zooms
                # into the patch window. `stats_crop` restricts the
                # subtitle numbers to that same window so the reported
                # hits/misses/false-alarms match the zoomed view.
                _render_pred_rainfall_hyst_axes(
                    ax, gt_canvases[t], row3_canvases[t],
                    lead_title=lead_titles[t], lead_hhmm=lead_hhmm,
                    low=(postproc_low if postproc_low is not None else 0.0),
                    high=hi,
                    stats_crop=(r0, r1, c0, c1),
                )
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)

    # Colorbars (one per row for lightning; single shared for rainfall).
    if label_type == "lightning":
        # GT colourbar dropped (binary canvas — the gradient bar was
        # noise); an invisible spacer of the same size keeps Row 1
        # aligned with Rows 2/3 under constrained_layout.
        cax_gt_spacer = fig.colorbar(
            im_gt, ax=axes[0, :].ravel().tolist(),
            shrink=0.85, pad=0.01, location="right",
        )
        cax_gt_spacer.ax.set_visible(False)
        cax_pred = fig.colorbar(im_pred, ax=axes[1, :].ravel().tolist(),
                                shrink=0.85, pad=0.01, location="right")
        cax_pred.set_label("Probability")
        if has_row3:
            from validate_predictions import (
                _add_zone_color_legend, _add_hmf_legend,
            )
            _add_zone_color_legend(fig)
            _add_hmf_legend(fig, y=-0.10)
            cax_spacer = fig.colorbar(
                im_pred, ax=axes[2, :].ravel().tolist(),
                shrink=0.85, pad=0.01, location="right",
            )
            cax_spacer.ax.set_visible(False)
    else:
        # Rainfall Row 3 now renders the zone palette (orange/blue/red),
        # not viridis — so the shared viridis colourbar only makes
        # sense for Rows 1 and 2. Row 3 gets an invisible spacer and
        # the standard zone legend + HMF footer instead.
        upper_axes = (axes[0:2, :].ravel().tolist()
                      if has_row3 else axes.ravel().tolist())
        cax = fig.colorbar(im_gt, ax=upper_axes,
                           ticks=[0, 1, 2, 3, 4],
                           shrink=0.85, pad=0.01, location="right")
        cax.set_ticklabels(RADAR_CLASS_NAMES)
        cax.set_label("Rain-rate class")
        if has_row3:
            spacer = fig.colorbar(
                im_gt, ax=axes[2, :].ravel().tolist(),
                shrink=0.85, pad=0.01, location="right",
            )
            spacer.ax.set_visible(False)
            from validate_predictions import (
                _add_zone_color_legend, _add_hmf_legend,
            )
            _add_zone_color_legend(fig)
            _add_hmf_legend(fig, y=-0.10)

    centre_latlon = _format_pixel_to_latlon((c0 + c1) / 2, (r0 + r1) / 2)
    fig.suptitle(
        f"ZOOM patch #{patch_num}  ({'Lightning' if label_type == 'lightning' else 'OPERA 5-class'})  "
        f"— Date: {date_str}  |  Ref: {ref_utc} UTC  |  "
        f"GT activity: {activity_score} px"
        + (f"  |  Centre: {centre_latlon}" if centre_latlon else ""),
        fontsize=13, fontweight="bold",
    )

    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Aggregate rainfall comparison graphs (top-N timesteps × 3 leads)
# ============================================================================
def _per_patch_class_counts(canvas: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Return an (N_PATCHES, n_classes) int64 count matrix. Slots that
    are out-of-domain (canvas < 0) contribute to no class — the row sum
    can therefore be smaller than PATCH_SIZE ** 2."""
    counts = np.zeros((N_PATCHES, n_classes), dtype=np.int64)
    for pi, p in enumerate(range(1, N_PATCHES + 1)):
        r0, r1, c0, c1 = get_patch_bounds(p)
        tile = canvas[r0:r1, c0:c1]
        for c in range(n_classes):
            counts[pi, c] = int(np.sum(tile == c))
    return counts


def _per_patch_class_pct(canvas: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Return an (N_PATCHES, n_classes) float64 percentage matrix. Zero
    out-of-domain contributions still divide by the full patch area
    (PATCH_SIZE ** 2), keeping the % scale interpretable when
    non-qualifying slots are present."""
    denom = float(PATCH_SIZE * PATCH_SIZE)
    counts = _per_patch_class_counts(canvas, n_classes)
    return 100.0 * counts.astype(np.float64) / denom


def _total_class_counts(canvas: np.ndarray, n_classes: int = 5) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        counts[c] = int(np.sum(canvas == c))
    return counts


def _total_class_pct(canvas: np.ndarray, n_classes: int = 5) -> np.ndarray:
    denom = float(H_FULL * W_FULL)
    counts = _total_class_counts(canvas, n_classes)
    return 100.0 * counts.astype(np.float64) / denom


# ---- p(class) histogram helpers ------------------------------------------
# Bin edges shared by every rainfall p(c) aggregate graph (2D + 3D). 10
# equal 10-percentage-point bins over the [0, 1] softmax range.
PMAX_BIN_EDGES = np.linspace(0.0, 1.0, 11)         # 0.0, 0.1, …, 1.0
PMAX_BIN_LABELS = [f"{int(100*PMAX_BIN_EDGES[i])}-"
                   f"{int(100*PMAX_BIN_EDGES[i+1])}%"
                   for i in range(len(PMAX_BIN_EDGES) - 1)]
PMAX_N_BINS = len(PMAX_BIN_LABELS)                 # 10


def _per_class_pc_hist_aggregate(
    raw_soft_samples: list[np.ndarray],
    raw_class_samples: list[np.ndarray],
    hyst_class_samples: list[np.ndarray],
) -> dict:
    """Aggregate counts + per-class denominators for the rainfall p(c)
    histogram graphs (2D + 3D).

    The denominator is per-CLASS (not per-domain): for row c, we look
    ONLY at pixels the model predicted as class c
      raw:  pixels where argmax(soft) == c   (from `raw_class_samples`)
      hyst: pixels where hyst_class     == c (from `hyst_class_samples`)
    and count how many have `soft[..., c]` in each softmax bin. The
    resulting histogram row therefore describes the distribution of
    confidence WITHIN class c — normalising by the row's own total
    (which the caller shows on the plot as a bonus annotation).

    Returned dict keys:
      raw_counts_pp   (n_classes, N_PATCHES, PMAX_N_BINS) int64
      hyst_counts_pp  (n_classes, N_PATCHES, PMAX_N_BINS) int64
      raw_denom_pp    (n_classes, N_PATCHES)              int64
      hyst_denom_pp   (n_classes, N_PATCHES)              int64

    Whole-domain (global) tallies for the 2D graph are just row-sums
    over the patch axis — the caller reduces as needed.
    """
    if not raw_soft_samples:
        empty_hist = np.zeros((5, N_PATCHES, PMAX_N_BINS), dtype=np.int64)
        empty_denom = np.zeros((5, N_PATCHES), dtype=np.int64)
        return dict(
            raw_counts_pp=empty_hist, hyst_counts_pp=empty_hist.copy(),
            raw_denom_pp=empty_denom, hyst_denom_pp=empty_denom.copy(),
        )
    n_classes = raw_soft_samples[0].shape[-1]
    raw_counts = np.zeros((n_classes, N_PATCHES, PMAX_N_BINS),
                          dtype=np.int64)
    hyst_counts = np.zeros_like(raw_counts)
    raw_denom = np.zeros((n_classes, N_PATCHES), dtype=np.int64)
    hyst_denom = np.zeros_like(raw_denom)

    for soft, raw_cls, hyst_cls in zip(
        raw_soft_samples, raw_class_samples, hyst_class_samples,
    ):
        for pi, p in enumerate(range(1, N_PATCHES + 1)):
            r0, r1, c0, c1 = get_patch_bounds(p)
            raw_tile = raw_cls[r0:r1, c0:c1]
            hyst_tile = hyst_cls[r0:r1, c0:c1]
            soft_tile = soft[r0:r1, c0:c1, :]
            for c in range(n_classes):
                p_c = soft_tile[..., c]
                r_mask = (raw_tile == c)
                r_n = int(r_mask.sum())
                if r_n > 0:
                    raw_denom[c, pi] += r_n
                    hist, _ = np.histogram(p_c[r_mask].ravel(),
                                           bins=PMAX_BIN_EDGES)
                    raw_counts[c, pi] += hist
                h_mask = (hyst_tile == c)
                h_n = int(h_mask.sum())
                if h_n > 0:
                    hyst_denom[c, pi] += h_n
                    hist, _ = np.histogram(p_c[h_mask].ravel(),
                                           bins=PMAX_BIN_EDGES)
                    hyst_counts[c, pi] += hist
    return dict(
        raw_counts_pp=raw_counts, hyst_counts_pp=hyst_counts,
        raw_denom_pp=raw_denom, hyst_denom_pp=hyst_denom,
    )


def _plotly_bar3d_mesh(x_pos, y_pos, z_top, *,
                       dx: float = 0.7, dy: float = 0.7,
                       colorscale: str = "Blues",
                       colorbar_title: str = "% pixels"):
    """Build a single plotly Mesh3d trace representing N 3D bars.

    Each bar is a rectangular prism from (x-dx/2, y-dy/2, 0) to
    (x+dx/2, y+dy/2, z_top[i]). 8 vertices × N bars, 12 triangles ×
    N bars combined into one Mesh3d — faster than N separate traces.

    Vertex intensity = the bar's height, colour-mapped through
    `colorscale`, so taller bars pop against shorter ones on the same
    axes. The colourbar shows the raw % value (not a normalised
    quantity) — plotly rescales automatically.
    """
    import plotly.graph_objects as go
    x_pos = np.asarray(x_pos, dtype=np.float64)
    y_pos = np.asarray(y_pos, dtype=np.float64)
    z_top = np.asarray(z_top, dtype=np.float64)
    n = len(x_pos)
    all_x = np.empty(8 * n, dtype=np.float64)
    all_y = np.empty(8 * n, dtype=np.float64)
    all_z = np.empty(8 * n, dtype=np.float64)
    all_intensity = np.empty(8 * n, dtype=np.float64)

    face_offsets = np.array([
        (0, 1, 2), (0, 2, 3),   # bottom
        (4, 5, 6), (4, 6, 7),   # top
        (0, 1, 5), (0, 5, 4),   # front  (y=y0)
        (1, 2, 6), (1, 6, 5),   # right  (x=x1)
        (2, 3, 7), (2, 7, 6),   # back   (y=y1)
        (3, 0, 4), (3, 4, 7),   # left   (x=x0)
    ], dtype=np.int64)

    all_i = np.empty(12 * n, dtype=np.int64)
    all_j = np.empty(12 * n, dtype=np.int64)
    all_k = np.empty(12 * n, dtype=np.int64)

    for b in range(n):
        x0 = x_pos[b] - dx / 2
        x1 = x_pos[b] + dx / 2
        y0 = y_pos[b] - dy / 2
        y1 = y_pos[b] + dy / 2
        z0 = 0.0
        z1 = float(z_top[b])
        base = b * 8
        # Local corner order:
        #   0 (x0,y0,z0)  1 (x1,y0,z0)  2 (x1,y1,z0)  3 (x0,y1,z0)
        #   4 (x0,y0,z1)  5 (x1,y0,z1)  6 (x1,y1,z1)  7 (x0,y1,z1)
        all_x[base:base + 8] = [x0, x1, x1, x0, x0, x1, x1, x0]
        all_y[base:base + 8] = [y0, y0, y1, y1, y0, y0, y1, y1]
        all_z[base:base + 8] = [z0, z0, z0, z0, z1, z1, z1, z1]
        # Uniform per-bar colour intensity (all 8 vertices share the value).
        all_intensity[base:base + 8] = z1
        tri_base = b * 12
        all_i[tri_base:tri_base + 12] = face_offsets[:, 0] + base
        all_j[tri_base:tri_base + 12] = face_offsets[:, 1] + base
        all_k[tri_base:tri_base + 12] = face_offsets[:, 2] + base

    z_max = float(max(z_top.max(), 1e-6))
    return go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        intensity=all_intensity,
        colorscale=colorscale,
        cmin=0.0, cmax=z_max,
        showscale=True,
        colorbar=dict(title=colorbar_title),
        flatshading=True,
        opacity=1.0,
    )


def plot_rainfall_pc_hist(
    raw_soft_samples: list[np.ndarray],
    raw_class_samples: list[np.ndarray],
    hyst_class_samples: list[np.ndarray],
    *,
    output_dir: Path,
    top_n: int,
) -> list[Path]:
    """Aggregate rainfall p(c) histogram — TWO separate PNGs (raw / hyst).

    Each PNG has 5 rows (one per class). Per row (class c):
      X-axis      10 bins of p(c) in 10% ranges (0-10% .. 90-100%).
                  p(c) is the softmax value for class c, BEFORE argmax.
      Y-axis      % of PIXELS-PREDICTED-AS-CLASS-C whose p(c) fell in
                  that bin (denominator is class-c pixels only — the
                  distribution INSIDE the class, not against the whole
                  domain).
      Bar label   raw pixel count in that bin (bonus annotation on top
                  of each bar).
      Row title   total number of class-c pixels considered.

    Denominator per row:
      raw  → pixels where argmax(soft) == c   (from `raw_class_samples`)
      hyst → pixels where hyst_class     == c (from `hyst_class_samples`)

    Returns paths to the two files written.
    """
    if (not raw_soft_samples or not raw_class_samples
            or not hyst_class_samples):
        print(f"  Skipping p(class) hist graph — no samples collected.")
        return []
    agg = _per_class_pc_hist_aggregate(
        raw_soft_samples, raw_class_samples, hyst_class_samples,
    )
    # Sum patch axis → global counts + denoms.
    raw_counts = agg["raw_counts_pp"].sum(axis=1)    # (n_classes, bins)
    hyst_counts = agg["hyst_counts_pp"].sum(axis=1)
    raw_denom = agg["raw_denom_pp"].sum(axis=1)      # (n_classes,)
    hyst_denom = agg["hyst_denom_pp"].sum(axis=1)
    n_classes = raw_counts.shape[0]
    n_samples = len(raw_soft_samples)

    def _annotate_bars(ax, x, counts, y_max):
        """Print the raw pixel count centred over each bar in
        thousands-separated form."""
        for xi, cnt in zip(x, counts):
            if cnt <= 0:
                continue
            ax.text(
                xi, y_max * 0.02 + (100.0 * cnt / max(counts.sum(), 1)),
                f"{int(cnt):,}",
                ha="center", va="bottom", fontsize=7,
                color="#222",
            )

    written: list[Path] = []
    for source_name, counts_by_class, denom_by_class, color in [
        ("raw", raw_counts, raw_denom, "steelblue"),
        ("hyst", hyst_counts, hyst_denom, "darkorange"),
    ]:
        fig, axes = plt.subplots(
            n_classes, 1, figsize=(13, 2.8 * n_classes),
            sharex=True, constrained_layout=True,
        )
        x = np.arange(PMAX_N_BINS)
        for c in range(n_classes):
            ax = axes[c]
            denom = int(denom_by_class[c])
            pct = (100.0 * counts_by_class[c] / denom
                   if denom > 0
                   else np.zeros(PMAX_N_BINS, dtype=np.float64))
            ax.bar(x, pct, width=0.8, color=color, alpha=0.9,
                   edgecolor=color)
            ax.set_ylabel(
                f"class {c} — {RADAR_CLASS_NAMES[c]}\n"
                f"(% of class-{c} pixels)",
                fontsize=10,
            )
            ax.set_title(
                f"class {c} — {RADAR_CLASS_NAMES[c]}  |  "
                f"total pixels considered: {denom:,}",
                fontsize=11, loc="left",
            )
            ax.grid(True, alpha=0.3, axis="y")
            # Bonus: annotate each bar with the raw pixel count that
            # produced it (integer, thousands-separated).
            y_max = max(float(pct.max()), 1.0)
            for xi, cnt, pc in zip(x, counts_by_class[c], pct):
                if int(cnt) <= 0:
                    continue
                ax.text(
                    xi, pc + 0.02 * y_max,
                    f"{int(cnt):,}",
                    ha="center", va="bottom", fontsize=7, color="#222",
                )
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(PMAX_BIN_LABELS, rotation=0, fontsize=9)
        axes[-1].set_xlabel(
            "p(class) bin  (softmax value for that class, before argmax)"
        )
        fig.suptitle(
            f"Rainfall — per-class p(c) histogram ({source_name} pred)  "
            f"|  top-{top_n} timesteps × 3 leads = {n_samples} samples",
            fontsize=14, fontweight="bold",
        )
        out_path = output_dir / f"aggregate_pc_hist_{source_name}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def plot_rainfall_pc_hist_3d(
    raw_soft_samples: list[np.ndarray],
    raw_class_samples: list[np.ndarray],
    hyst_class_samples: list[np.ndarray],
    *,
    output_dir: Path,
    top_n: int,
    filename_prefix: str = "aggregate_pc_hist_3d",
) -> list[Path]:
    """Interactive 3D companion to `plot_rainfall_pc_hist`.

    Writes ONE plotly HTML per (class, source) pair — 5 classes ×
    (raw, hyst) = 10 files. Each figure is a rotatable 3D BAR chart
    (built from a single plotly Mesh3d trace):
      X  p(c) bin (0..9 → 0-10% .. 90-100%)
      Y  Patch #  (1..18)
      Z  % of that patch's class-c pixels whose p(c) fell in the bin
         — per-patch denominator is 'pixels in the patch predicted as
         class c across all samples'. Each patch's bars therefore sum
         to 100% independently (when the patch has any class-c pixels).

    Title carries the total class-c denominator across the whole
    domain so the reader knows how many pixels the row summarises.

    Returns the list of written file paths.
    """
    if (not raw_soft_samples or not raw_class_samples
            or not hyst_class_samples):
        print(f"  Skipping p(class) 3D hist graphs — no samples collected.")
        return []
    import plotly.graph_objects as go
    agg = _per_class_pc_hist_aggregate(
        raw_soft_samples, raw_class_samples, hyst_class_samples,
    )
    raw_counts_pp = agg["raw_counts_pp"]     # (n_classes, N_PATCHES, bins)
    hyst_counts_pp = agg["hyst_counts_pp"]
    raw_denom_pp = agg["raw_denom_pp"]       # (n_classes, N_PATCHES)
    hyst_denom_pp = agg["hyst_denom_pp"]
    n_classes = raw_counts_pp.shape[0]
    n_samples = len(raw_soft_samples)
    written: list[Path] = []

    # Per-(class, patch) percentage matrices. Skip divisions by zero
    # cleanly with np.divide's `where`.
    def _pct(counts, denom):
        pct = np.zeros_like(counts, dtype=np.float64)
        for c in range(counts.shape[0]):
            for pi in range(counts.shape[1]):
                d = int(denom[c, pi])
                if d > 0:
                    pct[c, pi] = 100.0 * counts[c, pi] / d
        return pct

    raw_pct_pp = _pct(raw_counts_pp, raw_denom_pp)
    hyst_pct_pp = _pct(hyst_counts_pp, hyst_denom_pp)
    raw_denom_total = raw_denom_pp.sum(axis=1)    # (n_classes,)
    hyst_denom_total = hyst_denom_pp.sum(axis=1)

    x_ticks = list(range(PMAX_N_BINS))
    y_ticks = list(range(1, N_PATCHES + 1))

    for c in range(n_classes):
        for source_name, pct_pp, denom_total, colorscale in [
            ("raw",  raw_pct_pp[c],  int(raw_denom_total[c]),  "Blues"),
            ("hyst", hyst_pct_pp[c], int(hyst_denom_total[c]), "Oranges"),
        ]:
            # Flatten (N_PATCHES, PMAX_N_BINS) → per-bar positions.
            xx, yy = np.meshgrid(
                np.arange(PMAX_N_BINS),
                np.arange(1, N_PATCHES + 1),
                indexing="xy",
            )
            x_flat = xx.ravel().astype(np.float64)
            y_flat = yy.ravel().astype(np.float64)
            z_flat = pct_pp.ravel().astype(np.float64)

            trace = _plotly_bar3d_mesh(
                x_flat, y_flat, z_flat,
                dx=0.7, dy=0.7,
                colorscale=colorscale,
                colorbar_title="% of class pixels",
            )
            fig = go.Figure(data=[trace])
            fig.update_layout(
                title=(
                    f"Rainfall — p(class {c} - {RADAR_CLASS_NAMES[c]}) "
                    f"histogram per patch — {source_name} pred  |  "
                    f"total class-{c} pixels: {denom_total:,}  |  "
                    f"top-{top_n} timesteps × 3 leads = {n_samples} "
                    f"samples"
                ),
                scene=dict(
                    xaxis=dict(
                        title="p(class) bin",
                        tickmode="array",
                        tickvals=x_ticks,
                        ticktext=PMAX_BIN_LABELS,
                    ),
                    yaxis=dict(
                        title="Patch #",
                        tickmode="array",
                        tickvals=y_ticks,
                    ),
                    zaxis=dict(title="% of class pixels (per patch)"),
                    aspectratio=dict(x=1.4, y=1.6, z=1.0),
                ),
                margin=dict(l=0, r=0, b=0, t=80),
            )
            out_path = (output_dir
                        / f"{filename_prefix}_class{c}_{source_name}.html")
            fig.write_html(str(out_path), include_plotlyjs="cdn")
            written.append(out_path)
    return written


def plot_rainfall_class_count_distribution(
    gt_samples: list[np.ndarray],
    raw_samples: list[np.ndarray],
    hyst_samples: list[np.ndarray],
    *,
    output_path: Path,
    top_n: int,
):
    """Aggregate rainfall graph: whole-domain class-count distribution
    (TOTAL column only) across top-N × 3 leads.

    Sources: GT, raw pred, hyst pred → 3 boxes per class subplot. One
    subplot per class (5 total), each subplot shows the distribution of
    whole-domain pixel counts for that class across every sample. The
    per-patch column series has been dropped — reader wanted the
    right-of-panel TOTAL summary only.
    """
    if not raw_samples or not hyst_samples or not gt_samples:
        print(f"  Skipping class-count distribution graph — "
              f"no samples collected.")
        return
    gt_tot = np.stack([_total_class_counts(c) for c in gt_samples])
    raw_tot = np.stack([_total_class_counts(c) for c in raw_samples])
    hyst_tot = np.stack([_total_class_counts(c) for c in hyst_samples])

    colors = {"gt": "#2ca02c", "raw": "steelblue", "hyst": "darkorange"}
    labels = {"gt": "GT", "raw": "raw pred", "hyst": "hyst pred"}
    positions = [0, 1, 2]

    fig, axes = plt.subplots(1, 5, figsize=(20, 6),
                             constrained_layout=True)
    for c in range(5):
        ax = axes[c]
        data = [gt_tot[:, c], raw_tot[:, c], hyst_tot[:, c]]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black"),
        )
        for box, key in zip(bp["boxes"], ("gt", "raw", "hyst")):
            box.set_facecolor(colors[key])
            box.set_alpha(0.7)
            box.set_edgecolor(colors[key])
        ax.set_xticks(positions)
        ax.set_xticklabels(["GT", "raw", "hyst"], fontsize=10)
        ax.set_title(f"class {c} — {RADAR_CLASS_NAMES[c]}",
                     fontsize=11)
        if c == 0:
            ax.set_ylabel("pixel count (whole domain)", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=colors[k],
                      alpha=0.7, edgecolor=colors[k], label=labels[k])
            for k in ("gt", "raw", "hyst")
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.suptitle(
        f"Rainfall — whole-domain class-count distribution  |  "
        f"top-{top_n} timesteps × 3 leads = "
        f"{len(raw_samples)} samples",
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
        SwinBlock, WeightedFocalLoss, WeightedFocalCategoricalCrossentropy,
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
        "WeightedFocalCategoricalCrossentropy": WeightedFocalCategoricalCrossentropy,
        "iou_metric":          iou_metric,
        "true_pos":            true_pos,
        "false_pos":           false_pos,
        "false_neg":           false_neg,
    }


def load_model_artifact(model_dir: Path, mode: str, source: str,
                        finetuned: bool, kd: bool = False) -> tf.keras.Model:
    """Load a saved model checkpoint. Three variants, mutually exclusive:

      kd=True                 -> coalition_<run_tag>_kd.keras
                                 (KD student, saved by train_lightning_kd.py -
                                  plain tf.keras save, no swin rebuild dance)
      finetuned=True          -> coalition_<run_tag>_finetuned.keras
                                 (rebuilt via train_models.build_finetune_model
                                 + load_weights; TF 2.10 nested-sub-Model
                                 shape workaround)
      neither                 -> coalition_<run_tag>.keras   (base)
    """
    if kd and finetuned:
        raise ValueError(
            "kd=True and finetuned=True are mutually exclusive: the KD "
            "student is trained fresh from scratch (no swin head), so a "
            "'finetuned KD' variant does not exist. Pass one or neither."
        )
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # Central naming helper. Deferred import so this module doesn't
    # depend on train_models at import time.
    from train_models import build_run_tag
    run_tag = build_run_tag(mode, source)

    def _resolve(kind_suffix: str) -> Path:
        """Artefact path for this run tag. `kind_suffix` is "" for base,
        "_finetuned", "_kd", etc."""
        return model_dir / f"coalition_{run_tag}{kind_suffix}.keras"

    base_path = _resolve("")

    if kd:
        # KD student is saved via .save() so a straight load_model works;
        # the custom_objects list handles ResBlock / ResGRU / ConvBlock /
        # WeightedFocalLoss registration same as the base path.
        kd_path = _resolve("_kd")
        if not kd_path.is_file():
            raise FileNotFoundError(
                f"KD-student checkpoint not found: {kd_path}. "
                f"Train it via `train_lightning_kd.py`."
            )
        return tf.keras.models.load_model(
            str(kd_path), custom_objects=_custom_objects(),
        )

    if not finetuned:
        if not base_path.is_file():
            raise FileNotFoundError(f"Base model not found: {base_path}")
        return tf.keras.models.load_model(
            str(base_path), custom_objects=_custom_objects(),
        )

    ft_path = _resolve("_finetuned")
    # History JSON follows the same tag as the finetuned checkpoint.
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
                      eval_results_path: Path | None,
                      kd: bool = False) -> float | None:
    """Pick the operative threshold for the lightning prediction map.

    Order:
      1. --threshold (manual override) if set.
      2. --eval_results JSON if explicitly passed.
      3. evaluation/eval_<run_tag>[_finetuned|_kd]/evaluation_results.json
         (the path evaluate_coalition.py writes by default).
         Falls back to the legacy pre-rename dir if the new-name dir is
         absent, so historical evaluation runs still resolve.
      4. 0.5 fallback with a warning.

    Note: this legacy threshold is only meaningful for the direct sigmoid-
    threshold path. The Hann-blend + hysteresis lightning inference in
    predict_full_domain.py drives binarisation from
    --lightning_low_threshold / --lightning_high_threshold instead, so
    predict_full_domain no longer calls this function.
    """
    if label_type != "lightning":
        return None
    if override is not None:
        print(f"  Using manual threshold = {override:.3f}")
        return float(override)

    from train_models import build_run_tag  # local import: keep viz-only path clean
    if eval_results_path is None:
        run_tag = build_run_tag(mode, source)
        if finetuned:
            run_tag = f"{run_tag}_finetuned"
        elif kd:
            run_tag = f"{run_tag}_kd"
        eval_results_path = (Path("evaluation") / f"eval_{run_tag}"
                             / "evaluation_results.json")

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
                        choices=[
                            "mtg_lightning",
                            "mtg_radar_rainfall",
                            "mtg_radar_continuous_rainfall",
                            "mtg_opera_radar_only_rainfall",
                            "mtg_opera_mtgmr_rainfall",
                            "mtg_lightning_opera_rainfall",
                            "mtg_lightning_opera_occurrence",
                            "mtg_opera_occurrence",
                        ],
                        help="Model variant. The name states its own track: "
                             "`_rainfall` for the OPERA 5-class head, "
                             "`_occurrence` for the lightning binary head.")
    parser.add_argument("--top_n", type=int, default=5,
                        help="How many of the highest-patch-count rows to "
                             "plot (default 5).")
    parser.add_argument("--data_root", type=str, default="./our_data")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--output_dir", type=str, default="./full_domain_plots")
    parser.add_argument("--finetuned", action="store_true",
                        help="Use coalition_<run_tag>_finetuned.keras "
                             "(rebuilt + load_weights, same trick as "
                             "evaluate_coalition). Mutually exclusive "
                             "with --kd.")
    parser.add_argument("--kd", action="store_true",
                        help="Use coalition_<run_tag>_kd.keras — the "
                             "knowledge-distillation student saved by "
                             "train_lightning_kd.py. Only meaningful for "
                             "the student mode `mtg_opera_occurrence`; "
                             "the student's mode config (satellite-only "
                             "past_hr) drives input construction so no "
                             "extra slicing step is needed. Mutually "
                             "exclusive with --finetuned.")
    parser.add_argument("--eval_results", type=str, default=None,
                        help="Path to an evaluation_results.json to read "
                             "optimal_threshold from. Default is "
                             "evaluation/eval_<run_tag>[_finetuned|_kd]/"
                             "evaluation_results.json.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual probability threshold for the lightning "
                             "prediction map (overrides optimal_threshold).")
    parser.add_argument("--batch_size", type=int, default=18,
                        help="Per-row batch size (default 18 = max patches).")
    parser.add_argument("--no_zoom", action="store_true",
                        help="Skip the per-timestep zoom-in plot of the "
                             "patch with the highest GT activity. Default "
                             "is to save BOTH the full-domain figure and "
                             "the zoom figure side by side.")
    # ---- Lightning post-processing (Row 3, lightning modes only) ----
    parser.add_argument("--stride", type=int, default=128,
                        help="Hann-overlap stride for the lightning "
                             "post-processing path used to build Row 3. "
                             "Default 128 = 50%% overlap (55 patches on "
                             "the 768x1536 canvas).")
    parser.add_argument("--lightning_low_threshold", type=float, default=0.90,
                        help="Hysteresis LOW threshold for lightning "
                             "Row 3. Default 0.90 (operational).")
    parser.add_argument("--lightning_high_threshold", type=float,
                        default=None,
                        help="Hysteresis HIGH threshold applied to every "
                             "lead time for lightning Row 3. Superseded by "
                             "--validation_summary. Default falls back to "
                             "lightning_postproc.DEFAULT_HIGH_THRESHOLD.")
    parser.add_argument("--validation_summary", type=str, default=None,
                        help="Path to a {track}_{yyyy}_{mm}_summary.json "
                             "produced by validate_predictions.py --track "
                             "lightning. When present, the per-lead tuned "
                             "high-threshold values are read from it and "
                             "override --lightning_high_threshold.")
    # ---- Rainfall post-processing (Row 3, rainfall/radar modes only) ----
    parser.add_argument("--rainfall_low_threshold", type=float,
                        default=DEFAULT_RAIN_LOW,
                        help=f"Hysteresis LOW threshold for rainfall Row 3 "
                             f"(applied to p(argmax) when argmax is a rainy "
                             f"class). Default {DEFAULT_RAIN_LOW} — lower "
                             f"than the lightning threshold because "
                             f"probability is split across 5 classes.")
    parser.add_argument("--rainfall_high_threshold", type=float,
                        default=DEFAULT_RAIN_HIGH,
                        help=f"Hysteresis HIGH threshold for rainfall Row 3. "
                             f"Default {DEFAULT_RAIN_HIGH}.")
    parser.add_argument("--no_aggregate_graphs", action="store_true",
                        help="Skip the rainfall aggregate comparison "
                             "graphs (avg per-patch class % and per-patch "
                             "class-count distribution). Only meaningful "
                             "for rainfall/radar modes; ignored otherwise.")
    args = parser.parse_args()

    if args.kd and args.finetuned:
        parser.error("--kd and --finetuned are mutually exclusive "
                     "(the KD student is trained fresh from scratch, "
                     "no swin head).")

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

    # Point create_datasets at the per-source artifacts before any input
    # transform runs. init_sequence_config populates the step/cols globals
    # from sequence_meta_<source>.json; set_normalization_stats_path
    # points the lazy stats loader at normalization_stats_<source>.json
    # (required - the transforms in create_datasets read those stats).
    init_sequence_config(str(data_root), SOURCE)
    set_normalization_stats_path(
        data_root / f"normalization_stats_{SOURCE}.json"
    )

    mode_config = get_mode_config(args.mode)
    label_type = mode_config["label_type"]
    step_minutes = _load_step_minutes(data_root)

    from train_models import build_run_tag  # local import: keep TF-heavy load lazy
    run_tag = build_run_tag(args.mode, SOURCE)
    variant_suffix = ("_finetuned" if args.finetuned
                      else "_kd" if args.kd
                      else "")
    artifact_tag = f"{run_tag}{variant_suffix}"
    output_dir = Path(args.output_dir) / f"full_domain_{artifact_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_label = ("finetuned" if args.finetuned
                     else "KD student" if args.kd
                     else "base")
    print("=" * 70)
    print("Full-domain top-N visualisation")
    print("=" * 70)
    print(f"  CSV:         {csv_path}")
    print(f"  Mode:        {args.mode}  (label_type={label_type})")
    print(f"  Source:      {SOURCE}  ({variant_label})")
    print(f"  Top N:       {args.top_n}")
    print(f"  Step (min):  {step_minutes}")
    print(f"  Output dir:  {output_dir}")

    threshold = resolve_threshold(
        label_type, args.mode, SOURCE, args.finetuned,
        args.threshold,
        Path(args.eval_results) if args.eval_results else None,
        kd=args.kd,
    )

    print(f"\nLoading model...")
    model = load_model_artifact(
        Path(args.model_dir), args.mode, SOURCE, args.finetuned,
        kd=args.kd,
    )
    print(f"  Loaded: {model.count_params():,} parameters")

    # Resolve country borders + Romania-centred view extent once now so
    # the script reports both the border source (cartopy Natural Earth
    # vs hardcoded fallback) and the countries it loaded, without
    # having to wait for the first plot.
    _ensure_view_cached()
    countries = sorted({n for _, _, n in _BORDERS_PIXEL_CACHE})
    print(f"  Border src:  {_BORDERS_SOURCE}  "
          f"({len(countries)} countries: {', '.join(countries)})")

    print(f"\nSelecting top {args.top_n} timesteps from {csv_path}")
    df_top = load_top_n_rows(csv_path, args.top_n)
    print(df_top[["date", "reference_utc", "n_patches"]].to_string(index=False))

    # Lazy import: predict_full_domain pulls TF, but we're already past
    # the model load so the cost is paid. Use its full-canvas input
    # builder + paste helper to run the model on ALL 18 patches (from
    # the reprojected data on disk), so pred canvases carry real model
    # output on inactive patches too — not a placeholder. The CSV's
    # active-patch list is kept only to drive the green/red numbering
    # (which patches DBSCAN actually selected).
    from predict_full_domain import (
        build_inputs_for_reference,
        paste_predictions_to_canvas,
        LEAD_STEP_OFFSETS as _PF_LEAD_STEP_OFFSETS,
    )

    # ---- Lightning post-processing setup (Row 3, lightning modes) ----
    is_lightning = (label_type == "lightning")
    is_rainfall = (label_type == "radar")
    run_hann_overlapped_inference = hysteresis_binary = None
    high_per_lead: dict[int, float] | None = None
    if is_lightning:
        from lightning_postproc import (
            run_hann_overlapped_inference,
            hysteresis_binary,
            DEFAULT_HIGH_THRESHOLD,
        )
        # Reuse the same resolver predict_full_domain uses so a shared
        # validation-summary file drives both scripts identically.
        from predict_full_domain import _resolve_high_threshold_per_lead
        import argparse as _argparse
        _resolver_args = _argparse.Namespace(
            validation_summary=args.validation_summary,
            lightning_high_threshold=args.lightning_high_threshold,
        )
        high_per_lead = _resolve_high_threshold_per_lead(
            _resolver_args, step_minutes,
        )
        print(f"  Lightning post-proc: stride={args.stride}  "
              f"low={args.lightning_low_threshold:.2f}  "
              f"high per lead={{{', '.join(f't+{o*step_minutes}={h:.2f}' for o, h in high_per_lead.items())}}}")

    # ---- Aggregate accumulators (rainfall only; per-sample canvases) ----
    # agg_raw_soft carries the raw (H, W, 5) softmax canvases needed by
    # the p(argmax) histogram graphs; the *_class arrays are int32
    # argmax / hysteresis-cleaned canvases used by the count-based graph.
    agg_gt: list[np.ndarray] = []
    agg_raw: list[np.ndarray] = []
    agg_hyst: list[np.ndarray] = []
    agg_raw_soft: list[np.ndarray] = []

    for rank, (_, row) in enumerate(df_top.iterrows(), start=1):
        date_str = row["date"]
        ref_utc = row["reference_utc"].strip()
        print(f"\n[{rank}/{len(df_top)}] {date_str} {ref_utc} UTC  "
              f"({row['n_patches']} qualifying patches)")

        # DBSCAN-selected patches from the CSV — drives green/red
        # numbering only; every patch still gets a model prediction.
        csv_active = ast.literal_eval(row["patch_numbers"])

        gt_canvases = build_full_gt(
            row, str(patches_dir), mode_config, label_type, step_minutes,
        )

        # -------------------- Model inference + Row 3 canvases --------------------
        pred_canvases: list[np.ndarray]
        row3_canvases: list[np.ndarray] | None = None
        raw_class_canvases: list[np.ndarray] | None = None
        if is_lightning:
            # Hann-overlap raw probability canvas → Row 2, then hysteresis
            # binary → Row 3. Matches predict_full_domain._plot_lightning_2x3
            # semantics so the visualiser and the operational inference
            # script show the same post-processed output on the same data.
            prob_canvases = run_hann_overlapped_inference(
                model, data_root, mode_config, date_str, ref_utc,
                step_minutes, stride=args.stride,
                batch_size=args.batch_size,
            )
            if prob_canvases is None:
                print(f"  No data available for the Hann-overlap input "
                      f"window. Skipping.")
                continue
            pred_canvases = prob_canvases
            row3_canvases = [
                hysteresis_binary(
                    prob_canvases[k], low=args.lightning_low_threshold,
                    high=high_per_lead[_PF_LEAD_STEP_OFFSETS[k]],
                )
                for k in range(len(prob_canvases))
            ]
            print(f"  Hann-overlap produced {len(prob_canvases)} lead(s); "
                  f"canvas shape {prob_canvases[0].shape}")
        else:
            inputs, all_patches = build_inputs_for_reference(
                data_root, mode_config, date_str, ref_utc, step_minutes,
            )
            if not all_patches:
                print(f"  No reprojected data available for the input "
                      f"window. Skipping.")
                continue
            print(f"  Built inputs for {len(all_patches)} / {N_PATCHES} "
                  f"patches (from reprojected data)  |  "
                  f"DBSCAN-selected: {len(csv_active)}")
            preds = model.predict(inputs, batch_size=args.batch_size,
                                  verbose=0)
            print(f"  Model output shape: {preds.shape}")
            pred_canvases = paste_predictions_to_canvas(
                preds, all_patches, label_type,
            )
            if is_rainfall:
                # Soft canvases → rainfall_hysteresis → Row 3 class map.
                # We reuse the model output rather than re-running the
                # 5-class softmax path.
                soft_canvases = build_full_soft_pred(
                    preds, all_patches, n_classes=preds.shape[-1],
                )
                row3_canvases = [
                    rainfall_hysteresis(
                        soft_canvases[k],
                        low=args.rainfall_low_threshold,
                        high=args.rainfall_high_threshold,
                    )
                    for k in range(len(soft_canvases))
                ]
                raw_class_canvases = pred_canvases   # int32 argmax
                raw_soft_canvases = soft_canvases    # (H, W, 5) softmax
                print(f"  Rainfall hysteresis "
                      f"(low={args.rainfall_low_threshold:.2f}, "
                      f"high={args.rainfall_high_threshold:.2f}): "
                      f"selected px per lead = "
                      f"{[int(np.sum((c > 0))) for c in row3_canvases]}")

        # -------------------- Aggregate collection (rainfall) --------------------
        # Each (timestep, lead) is one sample. GT tiles come from
        # build_full_gt (int32 class canvas with -1 for missing lead
        # data — clip to 0 for the aggregate so out-of-domain reads as
        # dry rather than an invalid class).
        if (is_rainfall and row3_canvases is not None
                and raw_class_canvases is not None):
            for t in range(len(gt_canvases)):
                gt_clean = np.where(gt_canvases[t] < 0, 0,
                                    gt_canvases[t]).astype(np.int32)
                raw_clean = np.where(raw_class_canvases[t] < 0, 0,
                                     raw_class_canvases[t]).astype(np.int32)
                agg_gt.append(gt_clean)
                agg_raw.append(raw_clean)
                agg_hyst.append(row3_canvases[t].astype(np.int32))
                agg_raw_soft.append(raw_soft_canvases[t])

        safe_ref = ref_utc.replace(":", "")
        out_png = output_dir / f"ts{rank:02d}_{date_str}_{safe_ref}.png"
        plot_full_domain(
            gt_canvases, pred_canvases, csv_active, label_type,
            date_str=date_str, ref_utc=ref_utc,
            threshold=threshold, output_path=out_png,
            step_minutes=step_minutes,
            row3_canvases=row3_canvases,
            postproc_low=(args.lightning_low_threshold if is_lightning
                          else args.rainfall_low_threshold),
            postproc_high_per_lead=(high_per_lead if is_lightning else None),
            postproc_high=(args.rainfall_high_threshold
                           if is_rainfall else None),
        )
        print(f"  Saved -> {out_png}")

        # Zoom plot: the qualifying patch with the most GT activity.
        if not args.no_zoom:
            best_patch, best_score = find_highest_activity_patch(
                gt_canvases, csv_active, label_type,
            )
            zoom_png = (
                output_dir
                / f"ts{rank:02d}_{date_str}_{safe_ref}_zoom_p{best_patch:02d}.png"
            )
            plot_zoom_patch(
                gt_canvases, pred_canvases, best_patch, best_score,
                label_type, date_str=date_str, ref_utc=ref_utc,
                threshold=threshold, output_path=zoom_png,
                step_minutes=step_minutes,
                row3_canvases=row3_canvases,
                postproc_low=(args.lightning_low_threshold if is_lightning
                              else args.rainfall_low_threshold),
                postproc_high_per_lead=(high_per_lead if is_lightning
                                        else None),
                postproc_high=(args.rainfall_high_threshold
                               if is_rainfall else None),
            )
            print(f"  Zoom  -> {zoom_png}  "
                  f"(patch #{best_patch}, GT activity={best_score} px)")

    # -------------------- Aggregate rainfall comparison graphs --------------------
    if is_rainfall and not args.no_aggregate_graphs and agg_raw:
        n_samples = len(agg_raw)
        print(f"\nBuilding aggregate rainfall graphs over {n_samples} samples "
              f"({args.top_n} timesteps × 3 leads)...")
        pc_paths = plot_rainfall_pc_hist(
            agg_raw_soft, agg_raw, agg_hyst,
            output_dir=output_dir, top_n=args.top_n,
        )
        for p in pc_paths:
            print(f"  Saved -> {p}")
        html_dir = output_dir / "aggregate_pc_hist_3d"
        html_dir.mkdir(exist_ok=True)
        written = plot_rainfall_pc_hist_3d(
            agg_raw_soft, agg_raw, agg_hyst,
            output_dir=html_dir, top_n=args.top_n,
        )
        if written:
            print(f"  Saved {len(written)} interactive 3D HTML(s) "
                  f"-> {html_dir}")
        dist_png = output_dir / "aggregate_class_count_distribution.png"
        plot_rainfall_class_count_distribution(
            agg_gt, agg_raw, agg_hyst,
            output_path=dist_png, top_n=args.top_n,
        )
        print(f"  Saved -> {dist_png}")

    print(f"\nDone. {len(df_top)} figure(s) written under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
