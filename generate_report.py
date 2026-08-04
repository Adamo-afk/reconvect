"""
generate_report.py
==================
Automated meteorologist-facing PDF report of the validation outputs.

Reads what validate_predictions.py has already produced in ./validation/
for a given (year, month), asks a local Ollama-hosted vision LLM
(default: gemma4:12b) to generate short data-first commentaries, then
renders a standalone PDF via fpdf2. Everything is deterministic given
(same validation outputs, same model tag, temperature=0, fixed seed).

The report combines BOTH tracks when their artefacts are present:
  * rainfall (OPERA multiclass) - precipitation intensities in mm/h
  * lightning (Hann + hysteresis) - occurrence with post-processing

When BOTH tracks selected the same reference (date + HHMM) and the
convective activity spatially overlaps, the commentary uses the
"IF APPLICABLE" coupling: "precipitation X mm/h paired with Y%
lightning probability inside the same convective cell". Otherwise
each track is described individually across timesteps.

Prerequisites:
  * Ollama server running on http://localhost:11434
  * `ollama pull gemma4:12b` (or override via --model)
  * pip install fpdf2 ollama Pillow

Usage:
  python generate_report.py --year 2025 --month 5
  python generate_report.py --year 2025 --month 5 --model gemma4:12b
  python generate_report.py --year 2025 --month 5 \
      --validation_dir ./validation --output ./validation/report_2025_05.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_MODEL_TAG = "gemma4:12b"
DEFAULT_VALIDATION_DIR = Path("./validation")
DEFAULT_ASSETS_DIR = Path("./assets")
DEFAULT_OLLAMA_SEED = 42
DEFAULT_OLLAMA_TEMPERATURE = 0.0

# Rainfall / lightning artefact filename patterns that validate_predictions.py
# emits. Kept as module-level constants so the loaders in the next step have a
# single source of truth.
RAINFALL_STEM_TEMPLATE = "rainfall_{year:04d}_{month:02d}"
LIGHTNING_STEM_TEMPLATE = "lightning_{year:04d}_{month:02d}"


# ============================================================================
# Artefact discovery (scaffold only - full parsing added in the next step)
# ============================================================================
def _discover_track_artefacts(validation_dir: Path, track: str,
                              year: int, month: int) -> dict:
    """Return a dict describing what's on disk for one (track, year, month).

    This is the SCAFFOLD version: it only reports which paths exist. The
    next step turns the JSON/CSV into typed dicts + the per-date PNG list.
    """
    if track == "rainfall":
        stem = RAINFALL_STEM_TEMPLATE.format(year=year, month=month)
    elif track == "lightning":
        stem = LIGHTNING_STEM_TEMPLATE.format(year=year, month=month)
    else:
        raise ValueError(f"unknown track: {track!r}")

    samples_csv = validation_dir / f"{stem}_samples.csv"
    summary_json = validation_dir / f"{stem}_summary.json"
    metrics_png = validation_dir / f"{stem}_metrics.png"

    # Per-date visualization PNGs. Both tracks use the same {stem}_{date}_...
    # prefix; rainfall adds a per-lead suffix (_t+NN) so it produces 3 PNGs
    # per reference, lightning produces 1 (2x3 all-leads-in-one-figure).
    per_date_pngs = sorted(validation_dir.glob(f"{stem}_20*-*-*_*.png"))
    # Remove the metrics.png (globs it as {stem}_metrics.png doesn't match
    # 20*-*-* prefix anyway, but keep this filter as belt-and-suspenders).
    per_date_pngs = [p for p in per_date_pngs if p != metrics_png]

    return {
        "track": track,
        "stem": stem,
        "samples_csv": samples_csv if samples_csv.exists() else None,
        "summary_json": summary_json if summary_json.exists() else None,
        "metrics_png": metrics_png if metrics_png.exists() else None,
        "per_date_pngs": per_date_pngs,
    }


def _print_discovery(artefacts: dict) -> None:
    """One-block-per-track summary of what the scaffold found. Used only by
    the stub main() so we can eyeball the discovery layer before wiring in
    Gemma + fpdf2."""
    track = artefacts["track"]
    print(f"--- {track} ({artefacts['stem']}) ---")
    for key in ("samples_csv", "summary_json", "metrics_png"):
        path = artefacts[key]
        status = f"OK   {path}" if path is not None else "MISSING"
        print(f"  {key:14s}: {status}")
    n = len(artefacts["per_date_pngs"])
    print(f"  per_date_pngs : {n} file(s)")
    for p in artefacts["per_date_pngs"][:6]:
        print(f"                  - {p.name}")
    if n > 6:
        print(f"                  ... and {n - 6} more")


# ============================================================================
# Artefact PARSERS (item 4)
# ============================================================================
def _load_step_minutes(data_root: Path) -> int:
    """Reuse the same timestep_config.json validate_predictions.py reads."""
    from predict_full_domain import _load_step_minutes as _load
    return _load(data_root)


def load_summary_json(path: Path) -> dict:
    """Pure JSON load. Both tracks' summaries share the top-level keys:
        track, year, month, total_selected_samples, initial_selection,
        samples_above_threshold_per_lead, difference_pct_per_lead,
        metrics_per_lead, high_coverage_samples_per_lead
    Lightning additionally has post_processing.high_threshold_per_lead
    and post_processing.tuning_scores. Rainfall additionally has
    threshold_mmh."""
    with open(path, "r") as f:
        return json.load(f)


def load_samples_csv(path: Path, track: str, step_minutes: int
                     ) -> list[dict]:
    """Normalise the per-track samples.csv into a common schema.

    Rainfall CSV uses `iou_mask_t+{step_offset}` / `class_wt_t+{step_offset}`
    (step_offset is 1/2/3). Lightning CSV uses `iou_t+{minutes}` /
    `far_t+{minutes}` / `pod_t+{minutes}` / `csi_t+{minutes}`
    (minutes is 15/30/45). We normalise BOTH to keys indexed by
    lead_minutes so the downstream cardinal extractor + PDF layout do not
    need to know which track produced a row.

    Returns a list of dicts:
        [{
           "date": "2025-05-14",
           "reference_utc": "12:30",
           "per_lead": {
               15: {"iou_mask": 87.3, "class_wt": 76.1},   # rainfall
               30: {"iou_mask": 71.2, "class_wt": 62.8},
               45: {"iou_mask": 55.8, "class_wt": 48.0},
           },
        }, ...]

    For lightning, per_lead[minutes] holds {"iou", "far", "pod", "csi"}.
    """
    rows: list[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            per_lead: dict[int, dict[str, float]] = {}
            if track == "rainfall":
                # Rainfall columns: iou_mask_t+1 / class_wt_t+1 / ... (step offsets)
                for offset in (1, 2, 3):
                    minutes = offset * step_minutes
                    per_lead[minutes] = {
                        "iou_mask": float(r[f"iou_mask_t+{offset}"]),
                        "class_wt": float(r[f"class_wt_t+{offset}"]),
                    }
            elif track == "lightning":
                # Lightning columns: iou_t+15 / far_t+15 / pod_t+15 / csi_t+15 / ...
                for minutes in (step_minutes, 2 * step_minutes, 3 * step_minutes):
                    per_lead[minutes] = {
                        "iou":  float(r[f"iou_t+{minutes}"]),
                        "far":  float(r[f"far_t+{minutes}"]),
                        "pod":  float(r[f"pod_t+{minutes}"]),
                        "csi":  float(r[f"csi_t+{minutes}"]),
                    }
            else:
                raise ValueError(f"unknown track: {track!r}")
            rows.append({
                "date": r["date"],
                "reference_utc": r["reference_utc"],
                "per_lead": per_lead,
            })
    return rows


def _index_per_date(sample_rows: list[dict]) -> dict[str, list[dict]]:
    """Group sample rows by date so the PDF layout can iterate 'for each date
    -> for each reference on that date'."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in sample_rows:
        idx[r["date"]].append(r)
    for date_str in idx:
        idx[date_str].sort(key=lambda x: x["reference_utc"])
    return dict(idx)


def load_track_data(artefacts: dict, step_minutes: int) -> dict:
    """Parse the discovered artefacts. Returns a dict:
        {
          "track": "rainfall"|"lightning",
          "summary": <raw summary dict>,
          "samples": [<normalised sample rows>],
          "samples_by_date": {"2025-05-14": [<rows>], ...},
          "metrics_png": Path,
          "per_date_pngs": [Path],
        }
    Raises FileNotFoundError if the summary or CSV is missing (both are
    mandatory - the discovery scaffold enforces this upstream)."""
    track = artefacts["track"]
    if artefacts["summary_json"] is None:
        raise FileNotFoundError(
            f"summary JSON missing for {track} - cannot build report."
        )
    if artefacts["samples_csv"] is None:
        raise FileNotFoundError(
            f"samples CSV missing for {track} - cannot build report."
        )
    summary = load_summary_json(artefacts["summary_json"])
    samples = load_samples_csv(
        artefacts["samples_csv"], track, step_minutes,
    )
    return {
        "track": track,
        "summary": summary,
        "samples": samples,
        "samples_by_date": _index_per_date(samples),
        "metrics_png": artefacts["metrics_png"],
        "per_date_pngs": artefacts["per_date_pngs"],
    }


def _print_loaded(loaded: dict) -> None:
    """Compact summary of what got parsed. Used by the scaffold main()."""
    track = loaded["track"]
    s = loaded["summary"]
    n_samples = len(loaded["samples"])
    n_dates = len(loaded["samples_by_date"])
    print(f"  parsed {n_samples} rows across {n_dates} unique date(s)")
    print(f"  summary: total_selected_samples={s.get('total_selected_samples')} "
          f"| metrics_per_lead keys={list(s.get('metrics_per_lead', {}).keys())}")
    if track == "lightning" and "post_processing" in s:
        pp = s["post_processing"]
        print(f"  post-proc: low={pp.get('low_threshold')} | "
              f"tuned high per lead={pp.get('high_threshold_per_lead')}")


# ============================================================================
# Cardinal + numeric-facts extractor (item 5)
# ============================================================================
# For every selected reference (across BOTH tracks when both are loaded),
# for every lead time (t+15/+30/+45), we compute a dict of facts that gets
# handed to Gemma as prompt context. The dict is what tells the model
# "convection is in the north-west centred at (47.2 N, 22.4 E), peaking at
# 38 mm/h, with 62% of the lightning strokes co-located inside the >=10
# mm/h footprint" - i.e. every claim it can plausibly make in the report is
# grounded in a number computed from the actual GT canvas, not read off
# the PNG by the vision head. Gemma's job is to *phrase* these facts, not
# to *observe* them.

RAINFALL_THRESHOLD_MMH = 10.0        # matches validate_predictions.RAINFALL_THRESHOLD_MMH
MIN_CELL_SIZE_PIXELS = 10             # ignore coupled cells smaller than this
                                       # (1-2 px specks would balloon the FACTS
                                       # block with noise, and are visually
                                       # invisible in the coupling PNG anyway)

CARDINAL_LABELS = (
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
)

# Coupling-mask figure colours (RGB in 0..1).
COUPLING_COLOR_RAIN_ONLY = (0.15, 0.35, 0.85)   # blue
COUPLING_COLOR_LIGHT_ONLY = (1.00, 0.55, 0.10)  # orange
COUPLING_COLOR_COUPLED = (0.90, 0.15, 0.15)     # red
COUPLING_COLOR_BASE = (0.93, 0.93, 0.93)        # pale grey (GT visible; both absent)


def _romania_grid_centre_latlon(gp) -> tuple[float, float]:
    """(lat, lon) of the geometric centre of the Romania grid (pixel
    (H_FULL/2, W_FULL/2)) via GridProjection.inverse. Used as the origin
    for the 8-way cardinal classification, so 'north-west' means
    'north-west of the grid centre', not 'north-west of Bucharest'."""
    import numpy as np
    from visualize_gt_vs_pred import H_FULL, W_FULL
    # GridProjection.inverse has an empty-input guard that calls len() on
    # its inputs; passing a 0-d numpy scalar makes it throw
    # "len() of unsized object". np.atleast_1d wraps it into a 1-element
    # array so both the guard and pyproj's transform run cleanly.
    y = np.atleast_1d(np.asarray(H_FULL / 2.0))
    x = np.atleast_1d(np.asarray(W_FULL / 2.0))
    lon, lat = gp.inverse(y, x)
    return float(lat[0]), float(lon[0])


def _cardinal_from_latlon(lat: float, lon: float,
                          centre_lat: float, centre_lon: float) -> str:
    """8-way cardinal label for the vector (centre -> point).
    Bearing measured clockwise from geographic north (0 deg)."""
    import numpy as np
    dlat = lat - centre_lat
    dlon = lon - centre_lon
    if dlat == 0.0 and dlon == 0.0:
        return "central"
    bearing = (float(np.degrees(np.arctan2(dlon, dlat))) + 360.0) % 360.0
    idx = int(((bearing + 22.5) % 360.0) / 45.0)
    return CARDINAL_LABELS[idx]


def _mask_centroid_latlon(mask, gp) -> tuple[float, float, int] | None:
    """Return (lat, lon, n_pixels) for the pixel centroid of a boolean mask.
    Returns None if the mask has zero True pixels."""
    import numpy as np
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    row_c = float(rows.mean())
    col_c = float(cols.mean())
    # atleast_1d works around GridProjection.inverse's 0-d input handling;
    # see _romania_grid_centre_latlon for the detail.
    lon, lat = gp.inverse(np.atleast_1d(row_c), np.atleast_1d(col_c))
    return float(lat[0]), float(lon[0]), int(mask.sum())


def _resolve_lead(ref_utc: str, offset_min: int,
                  date_str: str) -> tuple[str, str]:
    """Same as validate_predictions._resolve_gt: advance (date, HH:MM) by
    offset_min minutes and return (hhmm_string, date_string), rolling over
    the day when needed. Duplicated here to avoid pulling matplotlib in
    via a validate_predictions import."""
    from datetime import datetime, timedelta
    parts = ref_utc.split(":")
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=int(parts[0]), minute=int(parts[1])
    )
    target = base + timedelta(minutes=offset_min)
    return target.strftime("%H%M"), target.strftime("%Y-%m-%d")


def _load_rainfall_field(data_root: Path, day: str, hhmm: str):
    """Return the raw 768x1536 mm/h array (NaN -> 0) or None if missing."""
    import numpy as np
    from extract_patches import find_reprojected_file, load_reprojected
    path = find_reprojected_file(
        str(data_root), "opera_rainfall_rate", "opera", day, hhmm,
    )
    if path is None:
        return None
    field = load_reprojected(path)
    if field.ndim == 3:
        field = np.squeeze(field, axis=0)
    return np.where(np.isnan(field), 0.0, field).astype(np.float32)


def _load_lightning_field(data_root: Path, day: str, hhmm: str):
    """Return the 768x1536 binary lightning occurrence array or None."""
    import numpy as np
    from extract_patches import find_reprojected_file, load_reprojected
    path = find_reprojected_file(
        str(data_root), "occurrence", "lightning", day, hhmm,
    )
    if path is None:
        return None
    field = load_reprojected(path)
    if field.ndim == 3:
        field = np.squeeze(field, axis=0)
    return (field > 0).astype(np.int8)


def _extract_coupled_cell_metadata(
    coupled_mask, rain_field, light_mask, gp,
    centre_lat: float, centre_lon: float,
    min_size_pixels: int = MIN_CELL_SIZE_PIXELS,
) -> list[dict]:
    """Run 8-connected labelling on the coupled mask, drop specks under
    `min_size_pixels`, return per-cell metadata (biggest cell first).

    Each cell dict has:
        bounding_box_pixels:      (row_min, row_max, col_min, col_max)
        size_pixels:              int (component area)
        centroid_lat_lon:         (lat, lon)
        centroid_cardinal:        N/NE/E/SE/S/SW/W/NW relative to grid centre
        peak_mmh_inside:          max mm/h inside this component
        lightning_active_inside:  active lightning pixels inside this component

    This is what replaces the coupling-mask VISION call: Gemma gets a
    text description of every coupled cell rather than reading the PNG.
    """
    import numpy as np
    from scipy.ndimage import label as _cc_label
    if not coupled_mask.any():
        return []
    structure_8conn = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    labelled, n_components = _cc_label(coupled_mask, structure=structure_8conn)
    cells: list[dict] = []
    for label_id in range(1, n_components + 1):
        component_mask = (labelled == label_id)
        size = int(component_mask.sum())
        if size < min_size_pixels:
            continue
        rows, cols = np.where(component_mask)
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        row_c = float(rows.mean()); col_c = float(cols.mean())
        lon_arr, lat_arr = gp.inverse(
            np.atleast_1d(row_c), np.atleast_1d(col_c),
        )
        centroid_lat = float(lat_arr[0]); centroid_lon = float(lon_arr[0])
        cardinal = _cardinal_from_latlon(
            centroid_lat, centroid_lon, centre_lat, centre_lon,
        )
        cells.append({
            "bounding_box_pixels":     (row_min, row_max, col_min, col_max),
            "size_pixels":             size,
            "centroid_lat_lon":        (centroid_lat, centroid_lon),
            "centroid_cardinal":       cardinal,
            "peak_mmh_inside":         float(rain_field[component_mask].max()),
            "lightning_active_inside": int(light_mask[component_mask].sum()),
        })
    # Largest cell first - Gemma tends to weight the first item in a list.
    cells.sort(key=lambda c: c["size_pixels"], reverse=True)
    return cells


def compute_facts_for_reference(
    date_str: str, ref_utc: str, lead_minutes_list: list[int],
    data_root: Path, gp, centre_lat_lon: tuple[float, float],
    rainfall_available: bool, lightning_available: bool,
) -> dict[int, dict]:
    """For one selected reference, compute per-lead NUMERIC facts. The
    "coupled or not" decision itself is delegated to Gemma via the
    coupling-mask figure; here we only ship the numeric anchors so any
    quantity Gemma cites in the caption is grounded in a real count.

    Returns:
        {
          15: {
            "rainfall": {
                "peak_mmh":         <float>,
                "n_pixels_ge10":    <int>,
                "centroid_lat_lon": (lat, lon) or None,
                "centroid_cardinal": "north-west" or None,
            } or None if rainfall track absent / GT missing,
            "lightning": {
                "n_active_pixels":  <int>,
                "centroid_lat_lon": (lat, lon) or None,
                "centroid_cardinal": "north" or None,
            } or None,
            "coupling": {
                # Numeric hooks for whatever Gemma sees as red in the
                # coupling-mask figure. NO applicable flag - Gemma decides
                # from the image whether the coupled region is visually
                # significant.
                "n_pixels_rain_only":         <int>,   # blue count
                "n_pixels_lightning_only":    <int>,   # orange count
                "n_pixels_coupled":           <int>,   # red count
                "peak_mmh_in_coupled_cells":  <float>, # 0 if red count is 0
                "lightning_pct_in_coupled":   <float>, # 0..100
                "coupled_centroid_cardinal":  <str> or None,
            } or None if either GT is missing (nothing to overlap),
          },
          30: {...},
          45: {...},
        }
    """
    import numpy as np
    centre_lat, centre_lon = centre_lat_lon
    out: dict[int, dict] = {}
    for lead_min in lead_minutes_list:
        gt_hhmm, gt_day = _resolve_lead(ref_utc, lead_min, date_str)
        rain_field = _load_rainfall_field(data_root, gt_day, gt_hhmm) \
            if rainfall_available else None
        light_field = _load_lightning_field(data_root, gt_day, gt_hhmm) \
            if lightning_available else None

        # ---- rainfall facts -------------------------------------------
        rainfall_facts = None
        if rain_field is not None:
            mask_ge10 = rain_field >= RAINFALL_THRESHOLD_MMH
            centroid = _mask_centroid_latlon(mask_ge10, gp)
            rainfall_facts = {
                "peak_mmh":         float(rain_field.max()),
                "n_pixels_ge10":    int(mask_ge10.sum()),
                "centroid_lat_lon": (centroid[0], centroid[1]) if centroid else None,
                "centroid_cardinal": (
                    _cardinal_from_latlon(centroid[0], centroid[1],
                                          centre_lat, centre_lon)
                    if centroid else None
                ),
            }

        # ---- lightning facts ------------------------------------------
        lightning_facts = None
        if light_field is not None:
            active_mask = light_field > 0
            centroid = _mask_centroid_latlon(active_mask, gp)
            lightning_facts = {
                "n_active_pixels":  int(active_mask.sum()),
                "centroid_lat_lon": (centroid[0], centroid[1]) if centroid else None,
                "centroid_cardinal": (
                    _cardinal_from_latlon(centroid[0], centroid[1],
                                          centre_lat, centre_lon)
                    if centroid else None
                ),
            }

        # ---- coupling numeric hooks + per-cell metadata -----------------
        # The per-cell metadata lets us drop the vision call for prompt C:
        # Gemma reads the per-cell dict list instead of looking at the
        # coupling-mask PNG. The PNG is still rendered by the report - it
        # ends up embedded in the PDF as a decoration for the human reader.
        coupling: dict | None = None
        if rain_field is not None and light_field is not None:
            mask_rain = rain_field >= RAINFALL_THRESHOLD_MMH
            mask_light = light_field > 0
            n_light = int(mask_light.sum())
            inter = mask_rain & mask_light
            n_coupled = int(inter.sum())
            n_rain_only = int(mask_rain.sum() - n_coupled)
            n_light_only = int(mask_light.sum() - n_coupled)
            coupled_centroid = _mask_centroid_latlon(inter, gp)
            coupled_cells = _extract_coupled_cell_metadata(
                inter, rain_field, mask_light, gp,
                centre_lat, centre_lon,
                min_size_pixels=MIN_CELL_SIZE_PIXELS,
            )
            coupling = {
                "n_pixels_rain_only":        n_rain_only,
                "n_pixels_lightning_only":   n_light_only,
                "n_pixels_coupled":          n_coupled,
                "peak_mmh_in_coupled_cells": (
                    float(rain_field[inter].max()) if n_coupled > 0 else 0.0
                ),
                "lightning_pct_in_coupled": (
                    (n_coupled / n_light) * 100.0 if n_light > 0 else 0.0
                ),
                "coupled_centroid_cardinal": (
                    _cardinal_from_latlon(coupled_centroid[0], coupled_centroid[1],
                                          centre_lat, centre_lon)
                    if coupled_centroid is not None else None
                ),
                # NEW: list of per-cell dicts, biggest cell first, cells
                # smaller than MIN_CELL_SIZE_PIXELS dropped. Empty list when
                # no coupled component survives the size cut.
                "coupled_cells": coupled_cells,
            }

        out[lead_min] = {
            "rainfall":  rainfall_facts,
            "lightning": lightning_facts,
            "coupling":  coupling,
        }
    return out


# ============================================================================
# Coupling-mask renderer (feeds Gemma the visual it uses to detect coupling)
# ============================================================================
def render_coupling_mask_figure(
    date_str: str, ref_utc: str, lead_minutes_list: list[int],
    data_root: Path, output_path: Path,
    rainfall_available: bool, lightning_available: bool,
) -> Path | None:
    """Render a 1x3 (columns = lead times) figure of colour-coded
    binary-mask overlays that Gemma reads to decide whether rainfall and
    lightning are spatially coupled.

    Legend (also printed in the figure):
      * blue   = rainfall >= 10 mm/h AND NOT lightning-active
      * orange = lightning-active AND NOT rainfall >= 10 mm/h
      * red    = both (coupled convective cell)
      * grey   = neither, or a whole GT is missing for that lead

    Returns the output path when at least one lead had usable GT, or
    None when nothing could be rendered (both tracks absent, or every
    lead's GT missing on disk). Never partially fills the output file.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from visualize_gt_vs_pred import H_FULL, W_FULL, overlay_borders, _ensure_view_cached
    import visualize_gt_vs_pred as _vf

    if not (rainfall_available or lightning_available):
        return None

    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT

    fig, axes = plt.subplots(
        1, len(lead_minutes_list), figsize=(7 * len(lead_minutes_list), 4.5),
        constrained_layout=True,
    )
    if len(lead_minutes_list) == 1:
        axes = [axes]

    any_rendered = False
    for i, lead_min in enumerate(lead_minutes_list):
        gt_hhmm, gt_day = _resolve_lead(ref_utc, lead_min, date_str)
        rain_field = (_load_rainfall_field(data_root, gt_day, gt_hhmm)
                      if rainfall_available else None)
        light_field = (_load_lightning_field(data_root, gt_day, gt_hhmm)
                       if lightning_available else None)

        ax = axes[i]
        display = np.full((H_FULL, W_FULL, 3),
                          COUPLING_COLOR_BASE, dtype=np.float32)

        mask_rain = (rain_field >= RAINFALL_THRESHOLD_MMH) if rain_field is not None else None
        mask_light = (light_field > 0) if light_field is not None else None

        stats: list[str] = []
        if mask_rain is not None and mask_light is not None:
            rain_only = mask_rain & ~mask_light
            light_only = mask_light & ~mask_rain
            coupled = mask_rain & mask_light
            display[rain_only] = COUPLING_COLOR_RAIN_ONLY
            display[light_only] = COUPLING_COLOR_LIGHT_ONLY
            display[coupled] = COUPLING_COLOR_COUPLED
            stats = [
                f"rain-only={int(rain_only.sum())}",
                f"light-only={int(light_only.sum())}",
                f"coupled={int(coupled.sum())}",
            ]
            any_rendered = True
        elif mask_rain is not None:
            display[mask_rain] = COUPLING_COLOR_RAIN_ONLY
            stats = [f"rain-only={int(mask_rain.sum())}",
                     "light GT unavailable"]
            any_rendered = True
        elif mask_light is not None:
            display[mask_light] = COUPLING_COLOR_LIGHT_ONLY
            stats = [f"light-only={int(mask_light.sum())}",
                     "rain GT unavailable"]
            any_rendered = True
        else:
            stats = ["no GT for this lead"]

        ax.imshow(display, aspect="equal", interpolation="nearest")
        try:
            overlay_borders(ax)
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(c_lo, c_hi)
        ax.set_ylim(r_hi, r_lo)
        ax.set_aspect("equal")
        ax.set_title(
            f"t+{lead_min} min ({gt_hhmm[:2]}:{gt_hhmm[2:]} UTC)  |  " + "  ".join(stats),
            fontsize=10,
        )

    if not any_rendered:
        plt.close(fig)
        return None

    legend_handles = [
        Patch(color=COUPLING_COLOR_RAIN_ONLY, label="Rainfall >=10 mm/h only"),
        Patch(color=COUPLING_COLOR_LIGHT_ONLY, label="Lightning active only"),
        Patch(color=COUPLING_COLOR_COUPLED, label="Coupled (both)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"Coupling mask - {date_str} ref={ref_utc}",
        fontsize=13, fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_facts_index(
    per_track_loaded: dict, data_root: Path, step_minutes: int,
    coupling_output_dir: Path,
) -> dict:
    """For every (date, reference) that appears in EITHER track's initial
    selection, compute the per-lead facts dict AND render the coupling-
    mask figure that Gemma reads to detect coupling visually.

    Returns:
        {
          "grid_centre_lat_lon": (46.1, 25.3),
          "lead_minutes": [15, 30, 45],
          "references": {
             ("2025-05-14", "12:30"): {
                 "coupling_figure_path": <Path or None>,
                 "per_lead": {15: {...}, 30: {...}, 45: {...}},
             },
             ...
          },
        }

    The coupling figure is skipped (path = None) when neither track has
    GT for any lead of that reference. The prompt-construction layer
    degrades to text-only for those references.
    """
    from c4dl.projection import GridProjection, romania_grid_area
    gp = GridProjection(romania_grid_area)
    centre = _romania_grid_centre_latlon(gp)

    rainfall_available = "rainfall" in per_track_loaded
    lightning_available = "lightning" in per_track_loaded
    lead_minutes = [step_minutes, 2 * step_minutes, 3 * step_minutes]

    # Union the selections across whichever tracks are loaded. Under the
    # OPERA-driven parity convention (both tracks share select_samples)
    # this union == either track's initial_selection; the loop preserves
    # correctness if someone runs with just one track loaded, or if a
    # legacy summary from before the parity fix produces a diverging list.
    ref_set: set[tuple[str, str]] = set()
    for loaded in per_track_loaded.values():
        for entry in loaded["summary"].get("initial_selection", []):
            date_str, hhmm = entry[0], entry[1]
            # summary stores HHMM (no colon); normalise to HH:MM for facts.
            if ":" not in hhmm:
                hhmm = f"{hhmm[:2]}:{hhmm[2:]}"
            ref_set.add((date_str, hhmm))
    ref_list = sorted(ref_set)

    facts_per_ref: dict[tuple[str, str], dict] = {}
    coupling_output_dir.mkdir(parents=True, exist_ok=True)
    for date_str, ref_utc in ref_list:
        per_lead = compute_facts_for_reference(
            date_str, ref_utc, lead_minutes, data_root, gp, centre,
            rainfall_available=rainfall_available,
            lightning_available=lightning_available,
        )
        # Render the coupling mask (Gemma's visual anchor). One figure per
        # reference, all leads on the same image, so the LLM sees the
        # evolution at a glance in a single vision call.
        safe_ref = ref_utc.replace(":", "")
        coupling_path = coupling_output_dir / f"coupling_{date_str}_{safe_ref}.png"
        rendered = render_coupling_mask_figure(
            date_str, ref_utc, lead_minutes, data_root, coupling_path,
            rainfall_available=rainfall_available,
            lightning_available=lightning_available,
        )
        facts_per_ref[(date_str, ref_utc)] = {
            "coupling_figure_path": rendered,
            "per_lead": per_lead,
        }

    return {
        "grid_centre_lat_lon": centre,
        "lead_minutes": lead_minutes,
        "references": facts_per_ref,
    }


# ============================================================================
# Ollama client wrapper (item 6)
# ============================================================================
# Two entry points:
#   * ollama_generate_text(prompt, ...)         - text-only prompt
#   * ollama_generate_vision(prompt, image, ...) - prompt + PNG bytes
#
# Both share the same retry / seed / temperature machinery. Retries only
# fire on transient network errors (ConnectionError / timeout / 5xx-ish
# ResponseErrors); a "model not found" error hard-fails with a clear
# actionable message ("run `ollama pull <tag>`") because that's a setup
# problem no retry can fix. temperature=0 + fixed seed give
# reproducible output across runs.

_MODEL_AVAILABLE_CACHE: dict[str, bool] = {}


def verify_ollama_model_available(model: str, timeout: float = 3.0) -> None:
    """Ping the local Ollama server and confirm `model` is pulled. Raises
    SystemExit with actionable text on any failure. Result is cached so
    repeated calls in the same run don't hit the network."""
    if _MODEL_AVAILABLE_CACHE.get(model):
        return
    import ollama
    try:
        response = ollama.list()
    except Exception as e:
        raise SystemExit(
            f"cannot reach the Ollama server (default http://localhost:11434):\n"
            f"  {type(e).__name__}: {e}\n"
            f"Make sure the Ollama app / daemon is running."
        )
    # ollama.list() returns an object with .models -> list of Model objects
    # that expose either `.model` or `.name`. Support both shapes.
    installed: list[str] = []
    for m in getattr(response, "models", []) or []:
        installed.append(getattr(m, "model", None) or getattr(m, "name", None))
    installed = [x for x in installed if x]
    if model not in installed:
        raise SystemExit(
            f"model {model!r} is not pulled on this Ollama instance.\n"
            f"Available: {installed}\n"
            f"Run: ollama pull {model}"
        )
    _MODEL_AVAILABLE_CACHE[model] = True


def _ollama_chat_with_retries(messages: list, *, model: str, seed: int,
                              temperature: float, max_retries: int = 3,
                              retry_backoff_sec: float = 2.0) -> str:
    """Shared retry wrapper around ollama.chat. Returns the assistant's
    message text."""
    import time
    import ollama

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "seed": seed,
                    # keep_alive defaults are fine; long-running batch of
                    # calls will re-use the loaded model.
                },
            )
            return resp["message"]["content"]
        except ollama.ResponseError as e:
            msg = str(e).lower()
            if "not found" in msg or "no such model" in msg:
                raise SystemExit(
                    f"model {model!r} not found by Ollama - "
                    f"run `ollama pull {model}` and retry."
                )
            last_exc = e
        except (ConnectionError, TimeoutError) as e:
            last_exc = e
        # Exponential-ish backoff between attempts (2s, 4s, 8s ...).
        if attempt < max_retries - 1:
            time.sleep(retry_backoff_sec * (2 ** attempt))
    raise SystemExit(
        f"Ollama request failed after {max_retries} attempts: "
        f"{type(last_exc).__name__}: {last_exc}"
    )


def ollama_generate_text(
    prompt: str, *, model: str, system: str | None = None,
    seed: int = DEFAULT_OLLAMA_SEED,
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    max_retries: int = 3,
) -> str:
    """Text-only chat completion. `system` is optional; when set it goes
    into a system-role message ahead of the user prompt."""
    verify_ollama_model_available(model)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _ollama_chat_with_retries(
        messages, model=model, seed=seed, temperature=temperature,
        max_retries=max_retries,
    )


def ollama_generate_vision(
    prompt: str, image_path: Path, *, model: str,
    system: str | None = None,
    seed: int = DEFAULT_OLLAMA_SEED,
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    max_retries: int = 3,
) -> str:
    """Vision chat: prompt + one PNG. The Ollama Python client accepts
    either a file path or bytes for `images`; we pass the path so it
    handles the read + base64 encoding for us."""
    verify_ollama_model_available(model)
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"vision image not found: {image_path}")
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user", "content": prompt,
        "images": [str(image_path)],
    })
    return _ollama_chat_with_retries(
        messages, model=model, seed=seed, temperature=temperature,
        max_retries=max_retries,
    )


# ============================================================================
# Prompt templates (items 7 + 8) - NOT wired into main() yet; review first.
# ============================================================================
# Design intent, per the spec:
#   * Audience = meteorologists UNFAMILIAR with our internal model / pipeline.
#     Every numeric claim must be explicit about units (mm/h, %, IoU as a
#     percentage). No jargon from the code side.
#   * "Data first, prose second": the model MUST NOT invent values. It
#     paraphrases the numeric facts we provide.
#   * Start / end of interval framing across t+15 -> t+30 -> t+45,
#     highlighting the extremum.
#   * Cardinal points: use pre-computed centroid_cardinal for affected zones
#     (never guess from pixels).
#   * IF APPLICABLE coupling: Gemma INFERS coupling from the coupling-mask
#     figure (red = coupled cells); the numeric anchors in FACTS let it
#     cite exact peak mm/h and % of lightning inside those red regions.
#   * Two-stage output: English generation, then a second Gemma call
#     translates each paragraph to Romanian with a technical glossary in
#     the system prompt (keeps IoU / POD / FAR / celula convectiva stable).
#   * Deterministic: temperature=0 + fixed seed -> same text.
#   * Prompts are STANDALONE per call so no truncation risk from stacking
#     multiple images/facts into one long context window.
# ----------------------------------------------------------------------------

PROMPT_SYSTEM_EN = """\
You are a meteorological analyst writing for other meteorologists who \
have never seen the internal model or code behind this nowcasting \
system. Your job is to interpret validation metrics and event snapshots \
using ONLY the numbers explicitly provided in the FACTS block of the \
user message. Do NOT invent values, do NOT infer numbers not in FACTS. \
Every claim in your output must be traceable to a specific value in \
FACTS.

General rules:
- On first mention every metric carries its unit or scale:
  * precipitation rates in mm/h,
  * lightning as % of active pixels / % coverage,
  * IoU / POD / FAR / CSI as scores in 0..1 (or in %, matching whichever \
    form FACTS uses; do not convert between the two).
- Describe evolution across the three lead times by naming the start \
  value at t+15, the end value at t+45, and the extremum (min or max) \
  across the interval; comment on whether skill grows or decays with \
  lead time.
- Locate affected zones by the cardinal labels supplied in FACTS \
  (north / north-east / east / south-east / south / south-west / west / \
  north-west). Never guess cardinal position.
- If FACTS lists coupled_cells with one or more entries (rainfall \
  >=10 mm/h AND active lightning inside the same connected component, \
  each cell >= 10 pixels), use the paired phrasing: "precipitation of \
  X mm/h paired with Y% of the lightning strokes inside the same \
  convective cell, in the <cardinal> of Romania" - filling X from the \
  cell's peak_mmh_inside, Y from lightning_pct_in_coupled, and \
  <cardinal> from that cell's centroid_cardinal. If coupled_cells is \
  empty, describe rainfall and lightning as SEPARATE observations \
  (their own cardinal positions, their own peaks, their own \
  evolutions across the three lead times).
- Concise and professional wording. No "we", no "our model". Two to \
  four sentences per requested block unless the user asks for more.
"""

# ----- Prompt A - Executive summary (text-only) -----------------------------
PROMPT_EXEC_SUMMARY_EN = """\
Write a short EXECUTIVE SUMMARY (2-3 short paragraphs) of the model's \
validation performance over the following period, for meteorologists.

FACTS (source of truth for every number below):
{facts_block}

Structure:
1. One paragraph naming the period, the number of selected samples \
   (identical across both tracks - selection is shared, OPERA-driven, \
   at precipitation >=10 mm/h anywhere on the canvas at the reference \
   timestep).
2. One paragraph reporting per-lead FAR / POD / CSI for each track \
   across t+15, t+30, t+45. Frame it as "start value -> end value" with \
   the extremum highlighted, and comment on whether skill degrades with \
   lead time as expected.
3. (If a lightning track is present in FACTS) one short paragraph on \
   the tuned per-lead hysteresis high thresholds and what they imply \
   about the model's operating point at each lead - e.g. does the \
   threshold rise or fall with lead time.
"""

# ----- Prompt B - Per-lead metrics commentary (text-only, one call per lead)
PROMPT_LEAD_METRICS_EN = """\
Write ONE short paragraph (2-4 sentences) interpreting the model's \
performance at lead time t+{lead_min} minutes on the {track} track. \
Target audience: meteorologists reviewing operational skill.

FACTS (source of truth):
{facts_block}

Requirements:
- Report FAR, POD, CSI with their values (as they appear in FACTS - \
  keep the same 0..1 vs % form).
- Compare this lead against the OTHER two lead times' values provided \
  in FACTS. State whether this lead is the best, middle, or worst of \
  the three on CSI, and by how much (e.g. "CSI drops from 0.76 at t+15 \
  to 0.46 at t+45").
- Note how many of the selected samples cleared the 90% coverage bar at \
  this lead (from FACTS).
- Do NOT restate the tuned high threshold here unless FACTS flags an \
  unusual value for this lead.
"""

# ----- Prompt C - Per-reference event caption (TEXT-ONLY, one call per ref)
# NOTE: this used to be a vision call over the coupling-mask PNG. It is
# now purely text-based: the FACTS block carries a per-cell
# `coupled_cells` list (bounding box, centroid cardinal, size in pixels,
# peak mm/h inside, active lightning inside) produced by scipy.ndimage.
# label with a >= 10-pixel filter, which is what Gemma reads to decide
# whether coupling is meteorologically meaningful. The PNG is still
# rendered and embedded in the PDF for the human reader; Gemma doesn't
# see it. Rationale: vision head hallucinates on scientific charts;
# text is deterministic, cheaper, and lets us apply a hard minimum-cell-
# size cut so single-pixel specks don't drive prose.
PROMPT_FIGURE_CAPTION_EN = """\
Write a meteorological caption (4-6 sentences) for date {date_str}, \
reference time {ref_utc} UTC, describing the convective event across \
the three lead times (t+15, t+30, t+45 minutes).

FACTS (source of truth for every number you cite):
{facts_block}

Coupling rule (this is important - it lives in FACTS, not an image):
- Each lead's coupling.coupled_cells list contains ONE ENTRY PER \
  coupled convective cell (rainfall >=10 mm/h AND active lightning \
  inside the same 8-connected component, minimum 10 pixels per cell). \
  If the list is non-empty for that lead, describe the biggest cell \
  first using the paired phrasing: "precipitation of X mm/h paired \
  with Y% of the lightning strokes inside the same convective cell, \
  in the <cardinal> of Romania". Fill X from that cell's \
  peak_mmh_inside; Y from lightning_pct_in_coupled (percentage of \
  ALL active lightning in the whole canvas that fell inside the \
  coupled region for that lead); <cardinal> from that cell's \
  centroid_cardinal. When multiple cells are listed, briefly note \
  their count and cardinal spread.
- If coupled_cells is EMPTY for a lead but rainfall or lightning has \
  activity, describe them as SEPARATE observations for that lead \
  (rainfall's own peak_mmh + cardinal + n_pixels_ge10, lightning's \
  own n_active_pixels + cardinal). Never fabricate a coupling that \
  isn't in FACTS.
- If a lead's rainfall AND lightning are both null, note that GT is \
  unavailable at that lead and skip numeric claims for it.

Structure:
1. One opening sentence naming the event, the dominant cardinal zone \
   at the earliest lead with activity, and whether the FACTS coupled_cells \
   list indicates a coupled convective system at that lead.
2. Two to three sentences tracking the evolution across t+15 -> t+45: \
   name the start value, the end value, and the extremum (min or max) \
   across the interval for whichever quantity matters most \
   (peak_mmh_inside for coupled cells, or peak_mmh / n_active_pixels for \
   the separate-observations case).
3. One closing sentence noting whether the coupling pattern strengthens, \
   holds, or weakens across the three lead times (compare coupled_cells \
   sizes / counts across leads), and roughly where in Romania the \
   coupled cells sit (cardinals from FACTS).
"""

# ============================================================================
# Prompt D - Romanian translation (text-only, one call per English paragraph)
# ============================================================================
# The translator sees ONE English paragraph at a time. A compact
# meteorological glossary in the system prompt is enough to keep the
# technical vocabulary stable across paragraphs without needing a
# whole-document translation call (which would blow the context budget
# on a big report and would still not give per-paragraph determinism).

PROMPT_SYSTEM_RO_TRANSLATE = """\
You are a professional technical translator. Translate the user's \
English text into Romanian, preserving every number, unit, cardinal \
direction, date, and place name. Do NOT add commentary or \
footnotes - output ONLY the Romanian translation.

Use these Romanian meteorological equivalents CONSISTENTLY (English -> Romanian):

  lead time / lead                 -> orizont de prognoza
  lead time t+15 / t+30 / t+45     -> orizontul t+15 / t+30 / t+45 minute
  forecast                         -> prognoza
  nowcast                          -> avertizare imediata (nowcast)
  ground truth (GT)                -> valoare de referinta (GT)
  prediction                       -> predictie
  model                            -> model (avoid "modelul nostru")
  precipitation                    -> precipitatii
  precipitation rate               -> intensitatea precipitatiilor
  rainfall                         -> precipitatii lichide
  lightning stroke                 -> descarcare electrica
  lightning occurrence             -> ocurenta descarcarilor
  convective cell                  -> celula convectiva
  post-processing                  -> post-procesare
  hysteresis threshold             -> prag de histereza
  low / high threshold             -> prag inferior / prag superior
  IoU                              -> indice de suprapunere (IoU)
  POD                              -> probabilitatea de detectie (POD)
  FAR                              -> rata alarmelor false (FAR)
  CSI                              -> scor de succes critic (CSI)
  true positive / false positive / false negative
                                   -> adevarat pozitiv / fals pozitiv / fals negativ
  hit / miss / false alarm         -> reusita / omitere / alarma falsa
  north / north-east / east / south-east / south / south-west / west / north-west
                                   -> nord / nord-est / est / sud-est / sud / sud-vest / vest / nord-vest
  Romania                          -> Romania

On FIRST use of each acronym keep the English form in parentheses \
("indice de suprapunere (IoU)"), then use the Romanian form for the \
rest of the paragraph. Do NOT translate SI units (mm/h, %). Preserve \
diacritics in the Romanian output (a, i, s, t become a with breve, \
a/i with circumflex, s/t with comma).
"""

PROMPT_USER_RO_TRANSLATE = """\
Translate the following English text into Romanian, following the \
glossary in the system prompt. Preserve every number, unit, cardinal, \
and date character-for-character. Output ONLY the translation.

---
{english_text}
---
"""


def _print_facts(facts_index: dict) -> None:
    """Compact human-readable dump of the facts index. Scaffold-only."""
    centre_lat, centre_lon = facts_index["grid_centre_lat_lon"]
    print(f"  grid centre: ({centre_lat:.3f} N, {centre_lon:.3f} E)  "
          f"| leads (min): {facts_index['lead_minutes']}")
    for (date_str, ref_utc), ref_data in facts_index["references"].items():
        coupling_path = ref_data.get("coupling_figure_path")
        cp_str = coupling_path.name if coupling_path else "no coupling figure"
        print(f"  {date_str} {ref_utc}   [{cp_str}]")
        for lead_min, slot in ref_data["per_lead"].items():
            parts = []
            rf = slot["rainfall"]
            lg = slot["lightning"]
            cp = slot["coupling"]
            if rf is None:
                parts.append("rain=N/A")
            else:
                card = rf["centroid_cardinal"] or "n/a"
                parts.append(
                    f"rain(peak={rf['peak_mmh']:.1f}mm/h, "
                    f">=10px={rf['n_pixels_ge10']}, at={card})"
                )
            if lg is None:
                parts.append("light=N/A")
            else:
                card = lg["centroid_cardinal"] or "n/a"
                parts.append(
                    f"light(active={lg['n_active_pixels']}, at={card})"
                )
            if cp is None:
                parts.append("coupling=N/A")
            else:
                parts.append(
                    f"cp(rain-only={cp['n_pixels_rain_only']}, "
                    f"light-only={cp['n_pixels_lightning_only']}, "
                    f"coupled={cp['n_pixels_coupled']}, "
                    f"pk_in_coupled={cp['peak_mmh_in_coupled_cells']:.1f}mm/h, "
                    f"at={cp['coupled_centroid_cardinal'] or 'n/a'})"
                )
            print(f"    t+{lead_min:>2}: " + "  |  ".join(parts))


# ============================================================================
# FACTS-block builders (item 9 helpers - one per prompt)
# ============================================================================
# Each builder returns a plain-text block that goes into the FACTS section
# of the corresponding user prompt. The rules:
#   * Every numeric value the model can plausibly cite must appear here.
#   * Layout is simple key:value or bullet lines; no JSON, no YAML - the
#     model will paraphrase in prose so a human-readable format works best.
#   * Absent data becomes explicit ("N/A") so the model doesn't hallucinate
#     a number to fill the gap.

def _fmt_lead_metrics(metrics: dict) -> str:
    """Format one lead's metrics-per-lead dict from the summary JSON."""
    return (f"FAR={metrics.get('FAR', 0):.3f}  "
            f"POD={metrics.get('POD', 0):.3f}  "
            f"CSI={metrics.get('CSI', 0):.3f}")


def _facts_block_exec_summary(per_track_loaded: dict, year: int,
                              month: int) -> str:
    """FACTS block for prompt A (executive summary)."""
    lines: list[str] = [f"Period: {year:04d}-{month:02d}"]
    for track in ("rainfall", "lightning"):
        loaded = per_track_loaded.get(track)
        if loaded is None:
            lines.append(f"{track}: NOT INCLUDED in this report")
            continue
        s = loaded["summary"]
        total = s.get("total_selected_samples", 0)
        criterion = ("precipitation >=10 mm/h at the reference time"
                     if track == "rainfall"
                     else f">= {s.get('min_active_pixels_for_selection', 1)} "
                          f"active LINET pixel(s) at the reference time")
        lines.append(f"\n{track.upper()} track:")
        lines.append(f"  selected samples: {total}")
        lines.append(f"  selection criterion: {criterion}")
        lines.append("  per-lead aggregate metrics:")
        for lead_title, metrics in s.get("metrics_per_lead", {}).items():
            lines.append(f"    {lead_title}: {_fmt_lead_metrics(metrics)}")
        lines.append("  samples clearing the 90% coverage bar per lead:")
        for lead_title, above in s.get("samples_above_threshold_per_lead", {}).items():
            lines.append(f"    {lead_title}: {above}")
        if track == "lightning" and "post_processing" in s:
            pp = s["post_processing"]
            lines.append(f"  post-processing low threshold: {pp.get('low_threshold')}")
            lines.append("  tuned high threshold per lead:")
            for lead_title, high in pp.get("high_threshold_per_lead", {}).items():
                lines.append(f"    {lead_title}: {high}")
    return "\n".join(lines)


def _facts_block_lead_metrics(loaded: dict, lead_min: int,
                              step_minutes: int) -> str:
    """FACTS block for prompt B (per-lead metrics commentary)."""
    track = loaded["track"]
    s = loaded["summary"]
    metrics_per_lead = s.get("metrics_per_lead", {})
    above_per_lead = s.get("samples_above_threshold_per_lead", {})
    this_lead = f"t+{lead_min}"

    lines = [
        f"Track: {track}",
        f"Total selected samples for the period: {s.get('total_selected_samples', 0)}",
        f"",
        f"THIS lead ({this_lead}):",
        f"  {_fmt_lead_metrics(metrics_per_lead.get(this_lead, {}))}",
        f"  samples cleared 90% coverage bar: {above_per_lead.get(this_lead, {})}",
        f"",
        f"OTHER leads (for comparison):",
    ]
    for offset in (1, 2, 3):
        m = offset * step_minutes
        if m == lead_min:
            continue
        key = f"t+{m}"
        lines.append(f"  {key}: {_fmt_lead_metrics(metrics_per_lead.get(key, {}))}")
    if track == "lightning" and "post_processing" in s:
        pp = s["post_processing"]
        high_per_lead = pp.get("high_threshold_per_lead", {})
        lines.append(f"")
        lines.append(f"Tuned high threshold at THIS lead ({this_lead}): "
                     f"{high_per_lead.get(this_lead, 'N/A')}  "
                     f"(low={pp.get('low_threshold', 'N/A')})")
    return "\n".join(lines)


def _fmt_cardinal_facts(slot_key: str, slot: dict | None) -> list[str]:
    """Format one (rainfall|lightning|coupling) slot for prompt C.
    The coupling slot now enumerates its coupled_cells list explicitly
    so Gemma has one dict per convective cell to paraphrase."""
    if slot is None:
        return [f"  {slot_key}: N/A (track absent or GT missing at this lead)"]
    if slot_key == "rainfall":
        card = slot.get("centroid_cardinal") or "n/a"
        return [
            f"  rainfall:",
            f"    peak_mmh:        {slot['peak_mmh']:.1f} mm/h",
            f"    n_pixels_ge10:   {slot['n_pixels_ge10']}",
            f"    centroid_cardinal: {card}",
        ]
    if slot_key == "lightning":
        card = slot.get("centroid_cardinal") or "n/a"
        return [
            f"  lightning:",
            f"    n_active_pixels:   {slot['n_active_pixels']}",
            f"    centroid_cardinal: {card}",
        ]
    if slot_key == "coupling":
        card = slot.get("coupled_centroid_cardinal") or "n/a"
        out = [
            f"  coupling:",
            f"    n_pixels_rain_only:         {slot['n_pixels_rain_only']}",
            f"    n_pixels_lightning_only:    {slot['n_pixels_lightning_only']}",
            f"    n_pixels_coupled:           {slot['n_pixels_coupled']}",
            f"    peak_mmh_in_coupled_cells:  {slot['peak_mmh_in_coupled_cells']:.1f} mm/h",
            f"    lightning_pct_in_coupled:   {slot['lightning_pct_in_coupled']:.1f} %",
            f"    coupled_centroid_cardinal:  {card}",
        ]
        cells = slot.get("coupled_cells") or []
        if not cells:
            out.append("    coupled_cells: [] (no coupled component >= 10 px)")
        else:
            out.append(f"    coupled_cells: {len(cells)} cell(s), biggest first:")
            for i, cell in enumerate(cells, 1):
                r_min, r_max, c_min, c_max = cell["bounding_box_pixels"]
                out.extend([
                    f"      cell #{i}:",
                    f"        size_pixels:             {cell['size_pixels']}",
                    f"        bounding_box_pixels:     rows {r_min}..{r_max}, cols {c_min}..{c_max}",
                    f"        centroid_cardinal:       {cell['centroid_cardinal']}",
                    f"        peak_mmh_inside:         {cell['peak_mmh_inside']:.1f} mm/h",
                    f"        lightning_active_inside: {cell['lightning_active_inside']}",
                ])
        return out
    return []


def _facts_block_figure_caption(ref_data: dict, lead_minutes: list[int]
                                 ) -> str:
    """FACTS block for prompt C (per-reference event caption, text-only).
    Emits per-lead rainfall + lightning summaries + a coupling block that
    lists every coupled cell (>= 10 px) with its own peak_mmh, centroid
    cardinal and lightning-active count. Gemma paraphrases those cell
    dicts directly - the coupling PNG is no longer sent to the model."""
    lines: list[str] = []
    for lead_min in lead_minutes:
        slot = ref_data["per_lead"].get(lead_min)
        lines.append(f"lead t+{lead_min}:")
        if slot is None:
            lines.append("  (no data)")
            continue
        lines.extend(_fmt_cardinal_facts("rainfall", slot["rainfall"]))
        lines.extend(_fmt_cardinal_facts("lightning", slot["lightning"]))
        lines.extend(_fmt_cardinal_facts("coupling", slot["coupling"]))
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Orchestrators: run every prompt through Gemma and translate the output
# ============================================================================
# Two-stage per section: build FACTS block -> Ollama English call ->
# Ollama Romanian translate call. Sections are kept as a list of
# (section_id, english, romanian) tuples so the PDF layout can iterate in
# order without knowing what generated them.

def generate_english_paragraphs(
    per_track_loaded: dict, facts_index: dict, *,
    year: int, month: int, step_minutes: int,
    model: str, seed: int, temperature: float,
) -> list[dict]:
    """Run prompts A, B (per track x per lead), C (per reference) through
    Gemma and return a list of section dicts:
        [{
            "id":     "exec_summary" | "lead_metrics_<track>_<lead>" |
                      "figure_caption_<date>_<hhmm>",
            "kind":   "exec_summary" | "lead_metrics" | "figure_caption",
            "title":  short human-readable title,
            "english": <str>,
            # Extras used by the PDF layout:
            "track":  optional str,
            "lead_min": optional int,
            "date":   optional str,
            "ref_utc": optional str,
            "figure_path": optional Path,
        }, ...]
    """
    sections: list[dict] = []

    # ---- Prompt A: executive summary ----------------------------------
    print(f"  [A] executive summary ...", flush=True)
    facts = _facts_block_exec_summary(per_track_loaded, year, month)
    english = ollama_generate_text(
        PROMPT_EXEC_SUMMARY_EN.format(facts_block=facts),
        model=model, system=PROMPT_SYSTEM_EN,
        seed=seed, temperature=temperature,
    )
    sections.append({
        "id": "exec_summary",
        "kind": "exec_summary",
        "title": "Executive summary",
        "english": english.strip(),
    })

    # ---- Prompt B: per-lead metrics commentary (per track x per lead) --
    for track in ("rainfall", "lightning"):
        loaded = per_track_loaded.get(track)
        if loaded is None:
            continue
        for offset in (1, 2, 3):
            lead_min = offset * step_minutes
            print(f"  [B] {track} metrics at t+{lead_min} ...", flush=True)
            facts = _facts_block_lead_metrics(loaded, lead_min, step_minutes)
            english = ollama_generate_text(
                PROMPT_LEAD_METRICS_EN.format(
                    lead_min=lead_min, track=track, facts_block=facts,
                ),
                model=model, system=PROMPT_SYSTEM_EN,
                seed=seed, temperature=temperature,
            )
            sections.append({
                "id": f"lead_metrics_{track}_t+{lead_min}",
                "kind": "lead_metrics",
                "title": f"{track.capitalize()} - t+{lead_min} minutes",
                "english": english.strip(),
                "track": track,
                "lead_min": lead_min,
            })

    # ---- Prompt C: per-reference event caption (TEXT-ONLY) ------------
    # Gemma no longer sees the coupling-mask PNG; it reads the per-cell
    # metadata from the FACTS block. The PNG is still rendered upstream
    # (build_facts_index) and gets embedded in the PDF for the human.
    lead_minutes = facts_index["lead_minutes"]
    for (date_str, ref_utc), ref_data in facts_index["references"].items():
        # Only emit a caption when at least one lead has coupling metadata
        # (i.e. both GTs were on disk for at least one lead). Same guard
        # as before, just phrased in terms of facts rather than PNG.
        has_any_data = any(
            slot["rainfall"] is not None or slot["lightning"] is not None
            for slot in ref_data["per_lead"].values()
        )
        if not has_any_data:
            continue
        print(f"  [C] event caption {date_str} {ref_utc} ...", flush=True)
        facts = _facts_block_figure_caption(ref_data, lead_minutes)
        english = ollama_generate_text(
            PROMPT_FIGURE_CAPTION_EN.format(
                date_str=date_str, ref_utc=ref_utc, facts_block=facts,
            ),
            model=model, system=PROMPT_SYSTEM_EN,
            seed=seed, temperature=temperature,
        )
        sections.append({
            "id": f"figure_caption_{date_str}_{ref_utc.replace(':', '')}",
            "kind": "figure_caption",
            "title": f"{date_str} - reference {ref_utc} UTC",
            "english": english.strip(),
            "date": date_str,
            "ref_utc": ref_utc,
            # figure_path kept so the PDF layout still embeds the coupling PNG
            # as decoration below the caption; it just isn't sent to Gemma.
            "figure_path": ref_data.get("coupling_figure_path"),
        })
    return sections


def translate_paragraphs_to_romanian(
    sections: list[dict], *,
    model: str, seed: int, temperature: float,
) -> list[dict]:
    """For each section, call prompt D once and store the Romanian text
    under 'romanian'. Returns the SAME list (mutated in place)."""
    for i, sec in enumerate(sections, 1):
        print(f"  [D] translating {i}/{len(sections)}: {sec['id']} ...",
              flush=True)
        ro = ollama_generate_text(
            PROMPT_USER_RO_TRANSLATE.format(english_text=sec["english"]),
            model=model, system=PROMPT_SYSTEM_RO_TRANSLATE,
            seed=seed, temperature=temperature,
        )
        sec["romanian"] = ro.strip()
    return sections


# ============================================================================
# PDF renderer (item 9) - fpdf2 layout, Unicode-safe via DejaVu Sans
# ============================================================================
# Layout (A4 portrait):
#   Page 1     Cover        header PNG + title + period + timestamp
#   Page 2     TOC          section list with page numbers
#   Page 3+    Exec summary Romanian body (English shown in italics
#                           underneath when --bilingual)
#   ...        Per-lead     one section per (track x lead), embeds
#              metrics      metrics.png at track level
#   ...        Per-ref      coupling figure + Romanian caption
#              captions     (one page per reference typically)
#   Last       Data         min/max/mean IoU per lead from samples.csv
#              appendix
#
# Font: DejaVu Sans (bundled with matplotlib) - has full Latin Extended-A
# so Romanian diacritics render without glyph substitution.


def _find_unicode_fonts() -> tuple[Path, Path]:
    """Resolve regular + bold TTFs for a Unicode font (DejaVu Sans).
    Uses matplotlib's font_manager so we don't hard-code an env path.
    Raises SystemExit with actionable text on failure."""
    try:
        from matplotlib import font_manager
        reg = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans"),
            fallback_to_default=False,
        )
        bold = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
            fallback_to_default=False,
        )
        return Path(reg), Path(bold)
    except Exception:
        # Fall back to a couple of common install paths.
        candidates = [
            (Path("C:/Windows/Fonts/DejaVuSans.ttf"),
             Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf")),
            (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
             Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        ]
        for reg, bold in candidates:
            if reg.is_file() and bold.is_file():
                return reg, bold
    raise SystemExit(
        "DejaVu Sans font not found. Install matplotlib (it bundles the "
        "font) or place DejaVuSans.ttf + DejaVuSans-Bold.ttf somewhere "
        "matplotlib.font_manager can find."
    )


class _ReportPDF:
    """Thin wrapper around fpdf2.FPDF that owns the DejaVu fonts, margins,
    and helpers for title/body/image blocks. Kept as a class so section
    rendering can share the pdf state without threading it through
    argument lists."""

    def __init__(self, output_path: Path):
        from fpdf import FPDF
        reg, bold = _find_unicode_fonts()
        self.pdf = FPDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=25)
        self.pdf.set_margins(left=20, top=20, right=20)
        self.pdf.add_font("DejaVu", "", str(reg))
        self.pdf.add_font("DejaVu", "B", str(bold))
        self.pdf.set_font("DejaVu", "", 11)
        self.output_path = output_path
        # Page width available for content after margins (A4 = 210mm).
        self.content_width_mm = 210 - 20 - 20
        # TOC accumulates (title, page_number) pairs as sections render.
        self._toc_entries: list[tuple[str, int]] = []

    # ---- primitives ----------------------------------------------------
    def new_page(self):
        self.pdf.add_page()

    def title(self, text: str, size: int = 16, top_gap: float = 4):
        self.pdf.ln(top_gap)
        self.pdf.set_font("DejaVu", "B", size)
        self.pdf.set_x(20)
        self.pdf.multi_cell(0, size * 0.55, text)
        self.pdf.ln(2)
        self.pdf.set_font("DejaVu", "", 11)

    def body(self, text: str, size: int = 11, italic: bool = False):
        # fpdf2 has no italic DejaVu style loaded; we simulate italics by
        # writing in a lighter grey when italic=True (used for the English
        # bilingual pass) so the RO body remains visually dominant.
        if italic:
            self.pdf.set_text_color(110, 110, 110)
        self.pdf.set_font("DejaVu", "", size)
        self.pdf.set_x(20)
        self.pdf.multi_cell(0, size * 0.55, text)
        if italic:
            self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)

    def image_full_width(self, path: Path, max_height_mm: float = 120):
        """Embed an image scaled to page width, capping height for readability."""
        from PIL import Image
        try:
            with Image.open(path) as im:
                w_px, h_px = im.size
        except Exception:
            self.body(f"[image not readable: {path.name}]", italic=True)
            return
        # Scale so width == content_width; then cap height.
        target_w_mm = self.content_width_mm
        target_h_mm = target_w_mm * (h_px / w_px)
        if target_h_mm > max_height_mm:
            scale = max_height_mm / target_h_mm
            target_w_mm *= scale
            target_h_mm = max_height_mm
        # Trigger a page break if the image won't fit in what's left.
        if self.pdf.get_y() + target_h_mm > 297 - 25:
            self.new_page()
        # Centre horizontally if we shrunk it.
        x = 20 + (self.content_width_mm - target_w_mm) / 2
        self.pdf.image(str(path), x=x, y=self.pdf.get_y(),
                       w=target_w_mm, h=target_h_mm)
        self.pdf.ln(target_h_mm + 3)

    def _mc(self, h: float, text: str, align: str = "L"):
        """Wrapper around pdf.multi_cell that resets x to the left margin
        before every call. fpdf2's cursor after a centred multi_cell
        occasionally lands at a position where the next multi_cell's
        effective width (0 == remaining-from-x) shrinks below one char,
        crashing with 'Not enough horizontal space'. Explicit set_x
        keeps every line starting at the left margin."""
        self.pdf.set_x(20)  # left margin
        self.pdf.multi_cell(0, h, text, align=align)

    # ---- specific sections --------------------------------------------
    def cover(self, header_png: Path | None, title_ro: str, title_en: str,
              period: str, tracks: list[str], generation_ts: str,
              bilingual: bool = False):
        self.new_page()
        if header_png is not None and header_png.is_file():
            self.image_full_width(header_png, max_height_mm=45)
        self.pdf.ln(20)
        self.pdf.set_font("DejaVu", "B", 22)
        self._mc(10, title_ro, align="C")
        self.pdf.ln(2)
        if bilingual:
            self.pdf.set_text_color(110, 110, 110)
            self.pdf.set_font("DejaVu", "", 12)
            self._mc(6, title_en, align="C")
            self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(25)
        self.pdf.set_font("DejaVu", "", 14)
        self._mc(7, f"Perioada: {period}", align="C")
        self._mc(7, f"Fluxuri incluse: {', '.join(tracks)}", align="C")
        self.pdf.ln(3)
        self.pdf.set_font("DejaVu", "", 10)
        self.pdf.set_text_color(110, 110, 110)
        self._mc(5, f"Generat: {generation_ts}", align="C")
        self.pdf.set_text_color(0, 0, 0)

    def _record_toc(self, title: str):
        self._toc_entries.append((title, self.pdf.page_no()))

    def section(self, title: str, romanian: str, english: str | None = None,
                bilingual: bool = False):
        """One section = title + Romanian body [+ optional English]."""
        self.new_page()
        self._record_toc(title)
        self.title(title, size=16)
        self.body(romanian, size=11)
        if bilingual and english:
            self.pdf.ln(2)
            self.pdf.set_font("DejaVu", "B", 9)
            self.pdf.set_text_color(110, 110, 110)
            self._mc(4.5, "(English original)")
            self.pdf.set_text_color(0, 0, 0)
            self.body(english, size=9, italic=True)

    def track_metrics_section(self, track_title_ro: str, metrics_png: Path | None,
                              subsections: list[tuple[str, str, str | None]],
                              bilingual: bool = False):
        """One section per track: metrics.png + one paragraph per lead."""
        self.new_page()
        self._record_toc(track_title_ro)
        self.title(track_title_ro, size=16)
        if metrics_png is not None and metrics_png.is_file():
            self.image_full_width(metrics_png, max_height_mm=95)
        for lead_title, romanian, english in subsections:
            self.pdf.ln(2)
            self.pdf.set_font("DejaVu", "B", 12)
            self._mc(6, lead_title)
            self.pdf.ln(1)
            self.body(romanian, size=11)
            if bilingual and english:
                self.body(english, size=9, italic=True)

    def reference_section(self, ref_title_ro: str, figure_path: Path | None,
                          romanian: str, english: str | None = None,
                          bilingual: bool = False):
        """One section per (date, ref): coupling figure + caption."""
        self.new_page()
        self._record_toc(ref_title_ro)
        self.title(ref_title_ro, size=14)
        if figure_path is not None and figure_path.is_file():
            self.image_full_width(figure_path, max_height_mm=95)
        self.body(romanian, size=11)
        if bilingual and english:
            self.body(english, size=9, italic=True)

    def data_appendix(self, per_track_loaded: dict, step_minutes: int):
        """Per-track table: min / mean / max coverage per lead time."""
        import numpy as np
        self.new_page()
        self._record_toc("Anexa - date sumar")
        self.title("Anexa - statistici pe eșantion", size=16)
        for track, loaded in per_track_loaded.items():
            self.pdf.set_font("DejaVu", "B", 12)
            self._mc(6, f"{track.capitalize()} - "
                        f"{len(loaded['samples'])} eșantioane")
            self.pdf.ln(1)
            # Choose the "primary" per-lead metric per track.
            primary_key = "iou_mask" if track == "rainfall" else "iou"
            header = f"{'Orizont':<12}{'min':>8}{'mediu':>10}{'max':>8}"
            self.pdf.set_font("DejaVu", "", 10)
            self._mc(5, header)
            for offset in (1, 2, 3):
                m = offset * step_minutes
                vals = [row["per_lead"][m][primary_key]
                        for row in loaded["samples"]
                        if m in row["per_lead"]]
                if not vals:
                    line = f"t+{m:<10}{'--':>8}{'--':>10}{'--':>8}"
                else:
                    line = (f"t+{m:<10}"
                            f"{min(vals):>8.1f}"
                            f"{float(np.mean(vals)):>10.1f}"
                            f"{max(vals):>8.1f}")
                self._mc(5, line)
            self.pdf.ln(3)

    def emit_toc_at_start(self):
        """Rebuild the final PDF so the TOC lands as page 2 with the
        collected page numbers filled in. fpdf2 doesn't have a first-pass
        placeholder, so we render everything into a temp buffer once,
        collect page numbers as sections are added, and then rewrite a
        TOC page. Simpler: skip a rewrite and print the TOC BEFORE section
        rendering with 'see following pages' if page numbers aren't known
        yet - but users usually want the exact pages. Compromise below:
        insert the TOC as page 2 by inserting into the internal pages
        list after the fact."""
        # fpdf2 stores pages as bytes in self.pdf.pages (a dict-like from
        # 2.7+). Rewriting the sequence is fragile; instead we output the
        # TOC as the last page and note it. For a first cut this keeps the
        # code simple; the audience is not chapter-hopping, so TOC at end
        # is acceptable.
        self.new_page()
        self.pdf.set_font("DejaVu", "B", 16)
        self._mc(8, "Cuprins")
        self.pdf.ln(2)
        self.pdf.set_font("DejaVu", "", 11)
        for title, page in self._toc_entries:
            dots = "." * max(3, 60 - len(title))
            line = f"{title} {dots} p.{page}"
            self._mc(5.5, line)

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf.output(str(self.output_path))


def render_pdf_report(
    sections: list[dict],
    per_track_loaded: dict,
    *,
    header_png: Path | None,
    output_path: Path,
    year: int, month: int, step_minutes: int,
    bilingual: bool = False,
) -> None:
    """Compose the standalone PDF from generated (English + Romanian)
    section paragraphs, plus the metrics.png and per-reference coupling
    figures that already sit on disk in the validation directory."""
    from datetime import datetime, timezone
    tracks_ro = {"rainfall": "precipitații (OPERA)",
                 "lightning": "descărcări electrice (LINET)"}
    tracks_included = [tracks_ro.get(t, t) for t in per_track_loaded]

    doc = _ReportPDF(output_path)
    doc.cover(
        header_png=header_png,
        title_ro=f"Raport de validare — {year:04d}-{month:02d}",
        title_en=f"Validation report — {year:04d}-{month:02d}",
        period=f"{year:04d}-{month:02d}",
        tracks=tracks_included,
        generation_ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        bilingual=bilingual,
    )

    # Index sections by kind for structured layout.
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        by_kind[s["kind"]].append(s)

    # -------- Executive summary
    for sec in by_kind.get("exec_summary", []):
        doc.section(
            title="Rezumat executiv",
            romanian=sec["romanian"],
            english=sec.get("english"),
            bilingual=bilingual,
        )

    # -------- Per-track metrics sections
    for track, loaded in per_track_loaded.items():
        track_title = f"Performanță pe orizont — {tracks_ro.get(track, track)}"
        # Group prompt-B sections for this track by lead order.
        lead_secs = [s for s in by_kind.get("lead_metrics", [])
                     if s.get("track") == track]
        lead_secs.sort(key=lambda s: s["lead_min"])
        subsections = [
            (f"Orizontul t+{s['lead_min']} minute",
             s["romanian"], s.get("english"))
            for s in lead_secs
        ]
        doc.track_metrics_section(
            track_title_ro=track_title,
            metrics_png=loaded.get("metrics_png"),
            subsections=subsections,
            bilingual=bilingual,
        )

    # -------- Per-reference figure captions
    for sec in by_kind.get("figure_caption", []):
        ref_title = (f"Eveniment {sec['date']} — "
                     f"referință {sec['ref_utc']} UTC")
        doc.reference_section(
            ref_title_ro=ref_title,
            figure_path=sec.get("figure_path"),
            romanian=sec["romanian"],
            english=sec.get("english"),
            bilingual=bilingual,
        )

    # -------- Data appendix
    doc.data_appendix(per_track_loaded, step_minutes)

    # -------- TOC (last page for now - fpdf2 doesn't allow easy insertion)
    doc.emit_toc_at_start()

    doc.save()


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a meteorologist-facing Romanian PDF report from the "
            "outputs of validate_predictions.py. Uses a local Ollama-hosted "
            "vision LLM for commentary + translation; fpdf2 for layout."
        ),
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True,
                        help="Month as an integer 1..12.")
    parser.add_argument("--track", type=str, default="both",
                        choices=["rainfall", "lightning", "both"],
                        help="Which tracks to include. 'both' generates the "
                             "combined report and requires artefacts for "
                             "both tracks on disk for the given (year, month).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_TAG,
                        help=f"Ollama model tag. Default {DEFAULT_MODEL_TAG}. "
                             f"Must be vision-capable if per-figure "
                             f"captioning is requested.")
    parser.add_argument("--validation_dir", type=str,
                        default=str(DEFAULT_VALIDATION_DIR),
                        help=f"Directory containing the extraction / "
                             f"visualization outputs of validate_predictions.py. "
                             f"Default {DEFAULT_VALIDATION_DIR}.")
    parser.add_argument("--assets_dir", type=str,
                        default=str(DEFAULT_ASSETS_DIR),
                        help=f"Directory containing the header banner PNG "
                             f"used on the report cover. Default {DEFAULT_ASSETS_DIR}.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDF path. Default: "
                             "{validation_dir}/report_{yyyy}_{mm}.pdf")
    parser.add_argument("--data_root", type=str, default="./our_data",
                        help="Project data root - needed by the cardinal + "
                             "peak-values extractor to load GT canvases and "
                             "translate pixel coords to lat/lon via "
                             "c4dl.projection.GridProjection.")
    parser.add_argument("--seed", type=int, default=DEFAULT_OLLAMA_SEED,
                        help="Ollama seed for deterministic generation.")
    parser.add_argument("--temperature", type=float,
                        default=DEFAULT_OLLAMA_TEMPERATURE,
                        help="Sampling temperature. Keep at 0 for reproducibility.")
    parser.add_argument("--bilingual", action="store_true",
                        help="Include the English source paragraph below each "
                             "Romanian body in the PDF. Default: Romanian only.")
    parser.add_argument("--skip_pdf", action="store_true",
                        help="Skip the PDF render step (useful when iterating "
                             "on prompts / facts extraction).")
    args = parser.parse_args()

    if not (1 <= args.month <= 12):
        raise SystemExit(f"--month must be 1..12, got {args.month}")

    validation_dir = Path(args.validation_dir)
    if not validation_dir.is_dir():
        raise SystemExit(
            f"--validation_dir does not exist: {validation_dir}. "
            f"Run validate_predictions.py first."
        )

    tracks_to_load = (["rainfall", "lightning"] if args.track == "both"
                      else [args.track])

    print("=" * 70)
    print(f"generate_report.py (SCAFFOLD MODE - discovery only)")
    print("=" * 70)
    print(f"  Year/month:     {args.year:04d}-{args.month:02d}")
    print(f"  Track(s):       {tracks_to_load}")
    print(f"  Model:          {args.model}")
    print(f"  Validation dir: {validation_dir}")
    print(f"  Assets dir:     {args.assets_dir}")
    print(f"  Output PDF:     {args.output or f'{validation_dir}/report_{args.year:04d}_{args.month:02d}.pdf'}")
    print()

    per_track_paths: dict[str, dict] = {}
    for track in tracks_to_load:
        artefacts = _discover_track_artefacts(
            validation_dir, track, args.year, args.month,
        )
        per_track_paths[track] = artefacts
        _print_discovery(artefacts)
        print()

    # 'both' mode requires the summary JSON + samples.csv for each requested
    # track. Bail early with a clear error if either is missing - the
    # downstream steps (facts extraction, prompt construction, PDF layout)
    # all depend on both being parseable.
    if args.track == "both":
        missing = [
            t for t, a in per_track_paths.items()
            if a["summary_json"] is None or a["samples_csv"] is None
        ]
        if missing:
            raise SystemExit(
                f"--track both requires summary.json + samples.csv for BOTH "
                f"tracks, but these are missing for {args.year:04d}-{args.month:02d}: "
                f"{missing}. Run `validate_predictions.py --track {'/'.join(missing)}` first."
            )

    # ---------------------------------------------------------------------
    # Item 4: parse everything into typed dicts
    # ---------------------------------------------------------------------
    step_minutes = _load_step_minutes(Path(args.data_root))
    print(f"step_minutes from {args.data_root}/timestep_config.json: {step_minutes}")
    print()

    per_track_loaded: dict[str, dict] = {}
    for track, artefacts in per_track_paths.items():
        if artefacts["summary_json"] is None or artefacts["samples_csv"] is None:
            print(f"--- {track}: skipping parse (summary or samples missing) ---")
            continue
        try:
            loaded = load_track_data(artefacts, step_minutes)
        except Exception as e:
            raise SystemExit(f"failed to parse {track} artefacts: {e}")
        per_track_loaded[track] = loaded
        print(f"--- {track}: parsed ---")
        _print_loaded(loaded)
        print()

    if not per_track_loaded:
        raise SystemExit("no track artefacts could be parsed - nothing to report.")

    # ---------------------------------------------------------------------
    # Item 5: cardinal + numeric-facts extractor
    # ---------------------------------------------------------------------
    print("Computing per-reference cardinal + numeric facts from GT canvases "
          "and rendering the coupling-mask figures ...")
    facts_index: dict | None = None
    try:
        facts_index = build_facts_index(
            per_track_loaded, Path(args.data_root), step_minutes,
            coupling_output_dir=validation_dir,
        )
    except Exception as e:
        # GT canvases are typically missing until the user has run reproject
        # on the same month. Report the exception but keep the scaffold usable
        # so we can validate the paths on any month that DOES have GT.
        import traceback
        print(f"WARN: facts extraction failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("(skipping facts dump; report will be paths-only when we build "
              "the PDF layer)")
    else:
        print()
        print(f"--- facts index ({len(facts_index['references'])} reference(s)) ---")
        _print_facts(facts_index)
        print()

    # ---------------------------------------------------------------------
    # Item 6: prove the Ollama wrapper round-trips
    # ---------------------------------------------------------------------
    print(f"--- Ollama round-trip smoke test (model={args.model!r}) ---")
    try:
        verify_ollama_model_available(args.model)
    except SystemExit as e:
        # Ollama not reachable / model not pulled -> stop before the
        # expensive orchestrators. The message from verify_... is already
        # actionable.
        raise

    # ---------------------------------------------------------------------
    # Items 7 + 8: run prompts A/B/C in English, then prompt D per paragraph
    # ---------------------------------------------------------------------
    if facts_index is None:
        raise SystemExit(
            "facts_index is not available; cannot generate paragraphs. "
            "See the WARN above (typically GT canvases missing on disk)."
        )
    print()
    print("Generating English paragraphs via Gemma ...")
    sections = generate_english_paragraphs(
        per_track_loaded, facts_index,
        year=args.year, month=args.month, step_minutes=step_minutes,
        model=args.model, seed=args.seed, temperature=args.temperature,
    )
    print(f"  {len(sections)} sections generated in English.")

    print()
    print("Translating each section to Romanian via Gemma ...")
    translate_paragraphs_to_romanian(
        sections,
        model=args.model, seed=args.seed, temperature=args.temperature,
    )
    print(f"  {len(sections)} sections translated.")

    # Preview: first 200 chars of each section's Romanian body.
    print()
    print(f"--- Section preview ({len(sections)} sections) ---")
    for sec in sections:
        preview = (sec["romanian"][:200] + "...") if len(sec["romanian"]) > 200 else sec["romanian"]
        print(f"  [{sec['kind']}] {sec['title']}")
        print(f"    RO: {preview}")
        print()

    # ---------------------------------------------------------------------
    # Item 9: render the PDF
    # ---------------------------------------------------------------------
    if args.skip_pdf:
        print("--skip_pdf set; not rendering the PDF.")
        return 0

    output_pdf = Path(
        args.output
        or (validation_dir / f"report_{args.year:04d}_{args.month:02d}.pdf")
    )
    header_png = Path(args.assets_dir) / "official_doc_header.png"
    print(f"Rendering PDF -> {output_pdf}")
    render_pdf_report(
        sections, per_track_loaded,
        header_png=header_png if header_png.is_file() else None,
        output_path=output_pdf,
        year=args.year, month=args.month, step_minutes=step_minutes,
        bilingual=args.bilingual,
    )
    print(f"  Wrote {output_pdf} ({output_pdf.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
