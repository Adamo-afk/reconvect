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
DEFAULT_MODEL_TAG = "gemma3:27b-it-q4_K_M"

# Hard upper bound on tokens Gemma is allowed to emit for a single call.
# Ollama's default `num_predict` can be -1 (unbounded) on some models —
# combined with a bad decode state or a repetition loop, that can hang
# the whole report on a single paragraph. 2000 tokens is generous for
# every prompt we send (executive summary, per-lead metrics, KD, per-
# reference caption, period conclusion) while still tripping fast if
# Gemma stalls. Override via `--max_tokens` on the CLI.
DEFAULT_OLLAMA_MAX_TOKENS = 2000
DEFAULT_VALIDATION_DIR = Path("./validation")
DEFAULT_ASSETS_DIR = Path("./assets")
DEFAULT_OLLAMA_SEED = 42
DEFAULT_OLLAMA_TEMPERATURE = 0.1   # 0.0 makes Gemma collapse to empty
                                   # completions; 0.1 is low enough to
                                   # stay near-deterministic (paired
                                   # with the fixed seed) without
                                   # triggering the empty-output path.

# Rainfall / lightning artefact filename patterns that validate_predictions.py
# emits. Kept as module-level constants so the loaders in the next step have a
# single source of truth.
RAINFALL_STEM_TEMPLATE = "rainfall_{year:04d}_{month:02d}"
LIGHTNING_STEM_TEMPLATE = "lightning_{year:04d}_{month:02d}"
KD_STEM_TEMPLATE = "kd_{year:04d}_{month:02d}"


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
# Knowledge-distillation artefact discovery (auto-added to report)
# ============================================================================
# The KD comparison section is included whenever kd_{yyyy}_{mm}_summary.json
# is present in the validation dir. Distinct from the tracks[] lookup: KD
# is orthogonal to which base tracks the user asked for (--track rainfall
# etc.) - it's a separate section that piggybacks on the same report.

def discover_kd_artefacts(validation_dir: Path, year: int, month: int) -> dict:
    """Return {stem, summary_json, samples_csv, metric_pngs (dict metric->Path),
    per_ref_pngs (list)}. Every field is None / empty when the corresponding
    file is not on disk, so the caller can degrade cleanly."""
    stem = KD_STEM_TEMPLATE.format(year=year, month=month)
    summary_json = validation_dir / f"{stem}_summary.json"
    samples_csv = validation_dir / f"{stem}_samples.csv"

    metric_pngs: dict[str, Path] = {}
    for metric in ("FAR", "POD", "CSI", "IoU"):
        p = validation_dir / f"{stem}_metrics_{metric}.png"
        if p.is_file():
            metric_pngs[metric] = p

    # Per-reference 3x3 comparison PNGs land at kd_YYYY_MM_YYYY-MM-DD_HHMM.png
    per_ref_pngs = sorted(validation_dir.glob(f"{stem}_20*-*-*_*.png"))

    return {
        "stem": stem,
        "summary_json": summary_json if summary_json.is_file() else None,
        "samples_csv":  samples_csv if samples_csv.is_file() else None,
        "metric_pngs":  metric_pngs,
        "per_ref_pngs": per_ref_pngs,
    }


def _facts_block_kd_summary(kd_summary: dict) -> str:
    """FACTS block for the KD comparison paragraph. Compact side-by-side
    table of teacher vs student per-lead FAR/POD/CSI + the tuned high
    thresholds that each model landed on."""
    lines: list[str] = [
        f"Total selected samples: {kd_summary.get('total_selected_samples', 0)}",
        f"Teacher mode: {kd_summary.get('teacher_mode', 'unknown')}",
        f"Student mode: {kd_summary.get('student_mode', 'unknown')} "
        f"(HR channels: {kd_summary.get('student_hr_channels', '?')} "
        f"- LINET dropped; MTG vis_06 only)",
        f"Selection criterion: {kd_summary.get('selection_criterion', 'n/a')}",
        "",
        "Per-lead aggregate metrics (teacher | student):",
    ]
    t_metrics = kd_summary["teacher"]["metrics_per_lead"]
    s_metrics = kd_summary["student"]["metrics_per_lead"]
    for lead_title in t_metrics:
        t = t_metrics[lead_title]
        s = s_metrics.get(lead_title, {})
        lines.append(
            f"  {lead_title}: "
            f"FAR {t.get('FAR', 0):.3f} | {s.get('FAR', 0):.3f}    "
            f"POD {t.get('POD', 0):.3f} | {s.get('POD', 0):.3f}    "
            f"CSI {t.get('CSI', 0):.3f} | {s.get('CSI', 0):.3f}"
        )
    lines.append("")
    lines.append("High-coverage-bar (>=90% IoU) count per lead:")
    t_hi = kd_summary["teacher"]["samples_above_threshold_per_lead"]
    s_hi = kd_summary["student"]["samples_above_threshold_per_lead"]
    for lead_title in t_hi:
        lines.append(
            f"  {lead_title}: teacher={t_hi[lead_title]}   "
            f"student={s_hi.get(lead_title, {})}"
        )
    lines.append("")
    lines.append("Tuned hysteresis high threshold per lead:")
    t_pp = kd_summary["teacher"]["post_processing"]["high_threshold_per_lead"]
    s_pp = kd_summary["student"]["post_processing"]["high_threshold_per_lead"]
    for lead_title in t_pp:
        lines.append(
            f"  {lead_title}: teacher={t_pp[lead_title]}   "
            f"student={s_pp.get(lead_title, '?')}"
        )
    return "\n".join(lines)


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
            lightning_only = mask_light & ~mask_rain
            coupled = mask_rain & mask_light
            display[rain_only] = COUPLING_COLOR_RAIN_ONLY
            display[lightning_only] = COUPLING_COLOR_LIGHT_ONLY
            display[coupled] = COUPLING_COLOR_COUPLED
            # Percentages relative to each batch's own reference set:
            #  - rain-only        = fraction of rain-active pixels that
            #                        DIDN'T co-occur with lightning
            #  - lightning-only   = fraction of lightning-active pixels
            #                        that DIDN'T co-occur with rain
            #  - coupled          = fraction of ANY-active pixels that
            #                        had BOTH rain and lightning
            # Each ratio is a "did the other quantity accompany me here?"
            # question, which is what the reader is actually trying to
            # judge from this panel — raw counts made you mentally
            # normalise against the visible blob sizes.
            n_rain = int(mask_rain.sum())
            n_light = int(mask_light.sum())
            n_union = int((mask_rain | mask_light).sum())
            pct_rain_only = (int(rain_only.sum()) / n_rain * 100.0
                             if n_rain > 0 else None)
            pct_light_only = (int(lightning_only.sum()) / n_light * 100.0
                              if n_light > 0 else None)
            pct_coupled = (int(coupled.sum()) / n_union * 100.0
                           if n_union > 0 else None)
            def _p(v): return "n/a" if v is None else f"{v:.1f}%"
            stats = [
                f"rain-only={_p(pct_rain_only)}",
                f"lightning-only={_p(pct_light_only)}",
                f"coupled={_p(pct_coupled)}",
            ]
            any_rendered = True
        elif mask_rain is not None:
            display[mask_rain] = COUPLING_COLOR_RAIN_ONLY
            stats = ["rain-only=100.0% (no lightning GT for this lead)"]
            any_rendered = True
        elif mask_light is not None:
            display[mask_light] = COUPLING_COLOR_LIGHT_ONLY
            stats = ["lightning-only=100.0% (no rain GT for this lead)"]
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


# ============================================================================
# Predicted-coupling figure (Row 1 GT + Row 2 pred-area + Row 3 pred-class)
# ============================================================================
# Optional 3x3 companion to render_coupling_mask_figure that adds the
# MODEL's take on the same lead times so the reader can compare observed
# vs predicted coupling side by side. Renders only when the caller
# supplies both a rainfall-multiclass model and a lightning-occurrence
# model checkpoint on disk; the coupling numeric-facts pipeline above
# already operates on GT only, so this figure sits alongside the GT
# figure as a separate PNG.
#
# Row 1: GT — reuses the same rain-only / lightning-only / coupled
#        palette as render_coupling_mask_figure (blue / orange / red).
# Row 2: pred-area  — pred rain class >= 1 (binary "any rain") vs
#        post-processed pred lightning (Hann + hysteresis binary),
#        painted in the same 3-color palette as Row 1.
# Row 3: pred-class — pred rain class canvas rendered in viridis-5,
#        BUT only where pred lightning is also active (i.e. the coupled
#        subset), so the reader sees WHICH rain intensity the model
#        thinks fell in the coupled region.
def render_pred_coupling_figure(
    date_str: str,
    ref_utc: str,
    lead_minutes: list[int],
    data_root: Path,
    output_path: Path,
    *,
    model_dir: Path,
    rainfall_mode: str,
    rainfall_source: str,
    rainfall_finetuned: bool,
    lightning_mode: str,
    lightning_source: str,
    lightning_finetuned: bool,
    lightning_kd: bool,
    lightning_low_threshold: float,
    lightning_high_per_lead: dict[int, float] | None,
    rainfall_available: bool,
    lightning_available: bool,
) -> Path | None:
    """Render the 3x3 GT-vs-predicted coupling figure. Returns the saved
    path, or None if inputs / models are unavailable at this reference.

    The rainfall model is expected to be a 5-class rainfall-rate head
    (mode registered in create_datasets.get_mode_config; e.g.
    `mtg_lightning_opera_rainfall`). The lightning model is the occurrence head
    (`mtg_lightning_opera_occurrence`) or the KD student
    (`mtg_opera_occurrence`) — post-processing is Hann-blended overlap
    inference + per-lead hysteresis (see lightning_postproc.py). Per-
    lead high thresholds come from the caller (typically the tuned
    values in `lightning_{yyyy}_{mm}_summary.json`); when absent, the
    lightning_postproc.DEFAULT_HIGH_THRESHOLD (0.95) is used.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    from predict_full_domain import (
        build_inputs_for_reference,
        paste_predictions_to_canvas,
        LEAD_STEP_OFFSETS,
    )
    from visualize_gt_vs_pred import (
        H_FULL, W_FULL, overlay_borders, _ensure_view_cached,
        load_model_artifact,
    )
    import visualize_gt_vs_pred as _vf
    from create_datasets import (
        get_mode_config, init_sequence_config, set_normalization_stats_path,
    )
    from lightning_postproc import (
        run_hann_overlapped_inference, hysteresis_binary,
        DEFAULT_HIGH_THRESHOLD, DEFAULT_STRIDE,
    )
    from validate_predictions import _load_step_minutes

    _ensure_view_cached()
    c_lo, c_hi, r_lo, r_hi = _vf._VIEW_EXTENT

    # Init sequence config + normalization stats before any transform runs.
    init_sequence_config(str(data_root), rainfall_source)
    set_normalization_stats_path(
        data_root / f"normalization_stats_{rainfall_source}.json"
    )
    step_minutes = _load_step_minutes(data_root)
    lead_minutes_list = list(lead_minutes)

    # ---- Load both models (once per reference; cost is dominated by
    # the two model.predict calls below).
    rainfall_model = load_model_artifact(
        model_dir, rainfall_mode, rainfall_source,
        finetuned=rainfall_finetuned,
    )
    lightning_model = load_model_artifact(
        model_dir, lightning_mode, lightning_source,
        finetuned=lightning_finetuned, kd=lightning_kd,
    )

    # ---- Rainfall inference: mode_config drives input build; output is
    # per-lead (H, W) int32 class canvases (0..4, or -1 for out-of-domain).
    rainfall_config = get_mode_config(rainfall_mode)
    rain_inputs, rain_valid_patches = build_inputs_for_reference(
        data_root, rainfall_config, date_str, ref_utc, step_minutes,
    )
    if not rain_valid_patches:
        return None
    rain_preds = rainfall_model.predict(rain_inputs, batch_size=18, verbose=0)
    rain_class_canvases = paste_predictions_to_canvas(
        rain_preds, rain_valid_patches, "radar",
    )

    # ---- Lightning inference: Hann-overlap + hysteresis produces per-lead
    # (H, W) int8 binary canvases matching the operational path.
    lightning_config = get_mode_config(lightning_mode)
    prob_canvases = run_hann_overlapped_inference(
        lightning_model, data_root, lightning_config,
        date_str, ref_utc, step_minutes,
        stride=DEFAULT_STRIDE, batch_size=32,
    )
    if prob_canvases is None:
        return None
    high_per_lead = lightning_high_per_lead or {}
    lightning_bin_canvases = []
    for k, offset in enumerate(LEAD_STEP_OFFSETS):
        h = high_per_lead.get(offset, DEFAULT_HIGH_THRESHOLD)
        lightning_bin_canvases.append(
            hysteresis_binary(prob_canvases[k],
                              low=lightning_low_threshold, high=h)
        )

    # ---- Figure setup.
    fig, axes = plt.subplots(3, len(lead_minutes_list),
                             figsize=(5.6 * len(lead_minutes_list), 12),
                             constrained_layout=True)
    if len(lead_minutes_list) == 1:
        axes = axes.reshape(3, 1)

    def _frame(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(c_lo, c_hi); ax.set_ylim(r_hi, r_lo)
        ax.set_aspect("equal")
        try:
            overlay_borders(ax)
        except Exception:
            pass

    any_rendered = False
    for i, lead_min in enumerate(lead_minutes_list):
        gt_hhmm, gt_day = _resolve_lead(ref_utc, lead_min, date_str)
        lead_wall = f"{gt_hhmm[:2]}:{gt_hhmm[2:]}"

        # ---- Row 1: GT coupling (rain-only / lightning-only / coupled),
        # same palette as render_coupling_mask_figure.
        rain_field = (_load_rainfall_field(data_root, gt_day, gt_hhmm)
                      if rainfall_available else None)
        light_field = (_load_lightning_field(data_root, gt_day, gt_hhmm)
                       if lightning_available else None)
        ax_r1 = axes[0, i]
        r1_display = np.full((H_FULL, W_FULL, 3),
                             COUPLING_COLOR_BASE, dtype=np.float32)
        if rain_field is not None and light_field is not None:
            mr = rain_field >= RAINFALL_THRESHOLD_MMH
            ml = light_field > 0
            r1_display[mr & ~ml] = COUPLING_COLOR_RAIN_ONLY
            r1_display[~mr & ml] = COUPLING_COLOR_LIGHT_ONLY
            r1_display[mr & ml] = COUPLING_COLOR_COUPLED
            any_rendered = True
        ax_r1.imshow(r1_display, aspect="equal", interpolation="nearest")
        _frame(ax_r1)
        ax_r1.set_title(f"GT - t+{lead_min} ({lead_wall} UTC)", fontsize=10)

        # ---- Row 2: pred-area (pred rain binary) intersected with
        # post-processed pred lightning binary. Same palette as Row 1.
        pred_rain = rain_class_canvases[i]
        pred_light = lightning_bin_canvases[i]
        pr_rain_bin = (pred_rain >= 1)   # any rain class
        pr_light_bin = (pred_light > 0)
        ax_r2 = axes[1, i]
        r2_display = np.full((H_FULL, W_FULL, 3),
                             COUPLING_COLOR_BASE, dtype=np.float32)
        r2_display[pr_rain_bin & ~pr_light_bin] = COUPLING_COLOR_RAIN_ONLY
        r2_display[~pr_rain_bin & pr_light_bin] = COUPLING_COLOR_LIGHT_ONLY
        r2_display[pr_rain_bin & pr_light_bin] = COUPLING_COLOR_COUPLED
        ax_r2.imshow(r2_display, aspect="equal", interpolation="nearest")
        _frame(ax_r2)
        ax_r2.set_title(f"Pred (per-area) - t+{lead_min}", fontsize=10)

        # ---- Row 3: pred-class rainfall coloured by class, restricted
        # to pixels where lightning was also predicted (the coupled
        # subset). Non-coupled pixels transparent.
        ax_r3 = axes[2, i]
        coupled_mask = pr_rain_bin & pr_light_bin
        cls_display = np.where(coupled_mask, pred_rain.astype(float), np.nan)
        ax_r3.imshow(cls_display, cmap=plt.get_cmap("viridis", 5),
                     vmin=0, vmax=4,
                     aspect="equal", interpolation="nearest")
        _frame(ax_r3)
        ax_r3.set_title(f"Pred (per-class, coupled only) - t+{lead_min}",
                        fontsize=10)

    if not any_rendered:
        plt.close(fig)
        return None

    # Legend for the top two rows (Row 3 uses the class colourbar).
    legend_handles = [
        Patch(color=COUPLING_COLOR_RAIN_ONLY, label="Rainfall only"),
        Patch(color=COUPLING_COLOR_LIGHT_ONLY, label="Lightning only"),
        Patch(color=COUPLING_COLOR_COUPLED, label="Coupled (both)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        f"Coupling: GT vs predicted  |  {date_str} ref={ref_utc}",
        fontsize=13, fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# Coupling aggregation helpers (peak-activity picker + period-wide stats)
# ============================================================================
# The report used to emit ONE coupling PNG per reference; that scales to
# hundreds of figures on a full-month run. The new layout keeps a single
# hero image (the peak-activity reference) in `rainfall_lightning_coupling/`
# and folds the rest of the period into an aggregate conclusion paragraph
# (hour-of-day distribution + rainfall-band -> lightning-coupling
# probability cross-tab).

# Bins for the rainfall-band cross-tab. Same class edges as the model's
# 5-class rainfall head so a reader lining up the report with the pred
# plots recognises the labels immediately.
_RAINFALL_BAND_EDGES = [10.0, 20.0, 30.0, 40.0, float("inf")]
_RAINFALL_BAND_LABELS = ["10-20 mm/h", "20-30 mm/h", "30-40 mm/h", ">=40 mm/h"]


def _reference_coupling_activity(ref_data: dict) -> int:
    """Total coupled pixels for a reference, summed across all leads.

    Zero when no lead has any coupling metadata (no GT on disk, or GT
    on disk but no coupled cell survived the 10-pixel minimum size).
    Used as the ranker for the peak-activity image and as the aggregate
    input for the period-wide summary.
    """
    total = 0
    for slot in ref_data.get("per_lead", {}).values():
        coupling = slot.get("coupling") if slot else None
        if coupling is not None:
            total += int(coupling.get("n_pixels_coupled", 0) or 0)
    return total


def _find_most_active_coupling_ref(
    facts_per_ref: dict,
) -> tuple[str, str] | None:
    """Pick the (date, ref_utc) with the most total coupled pixels
    across its three leads. Returns None when no reference has any
    coupling — the caller then skips the hero image entirely.
    """
    best_ref: tuple[str, str] | None = None
    best_score = 0
    for ref_key, ref_data in facts_per_ref.items():
        score = _reference_coupling_activity(ref_data)
        if score > best_score:
            best_score = score
            best_ref = ref_key
    return best_ref


def _compute_period_coupling_summary(
    facts_per_ref: dict,
    most_active_ref: tuple[str, str] | None,
) -> dict:
    """Aggregate coupling activity across every reference in the month
    into a text-friendly summary the conclusion paragraph reads from.

    Fields:
      n_refs_scanned         - how many references had any GT loaded
      n_refs_with_coupling   - how many had at least one coupled pixel
      total_coupled_pixels   - sum across all references and leads
      per_hour               - {hour_utc: total_coupled_pixels} (only
                                hours with non-zero activity)
      peak_hour_utc          - argmax of per_hour; None if all zero
      peak_hour_pixels       - the value at that hour
      rainfall_band_stats    - list of dicts, one per band, giving the
                                # of coupled cells whose peak_mmh fell
                                in that band + the sum of coupled pixels
                                in cells peaking in that band
      most_likely_band_label - band label with the highest coupled-pixel
                                total (proxy for "at which rainfall
                                intensity is lightning coupling most
                                likely"); None if no coupling occurred
      most_active_ref        - echoed back from the caller for prose
                                anchoring
    """
    n_refs_scanned = len(facts_per_ref)
    n_refs_with_coupling = 0
    total_coupled_pixels = 0
    per_hour: dict[int, int] = {}
    band_counts = [0] * len(_RAINFALL_BAND_LABELS)
    band_pixel_sums = [0] * len(_RAINFALL_BAND_LABELS)

    for ref_key, ref_data in facts_per_ref.items():
        ref_utc = ref_key[1]
        score = _reference_coupling_activity(ref_data)
        if score == 0:
            continue
        n_refs_with_coupling += 1
        total_coupled_pixels += score

        # Hour bucket from the reference HH:MM.
        try:
            hour = int(ref_utc.split(":")[0])
        except (ValueError, IndexError):
            hour = -1
        if hour >= 0:
            per_hour[hour] = per_hour.get(hour, 0) + score

        # Rainfall-band tally: iterate every coupled cell across every
        # lead and drop it into the band its peak_mmh falls into.
        for slot in ref_data.get("per_lead", {}).values():
            coupling = slot.get("coupling") if slot else None
            if coupling is None:
                continue
            for cell in coupling.get("coupled_cells", []) or []:
                peak = float(cell.get("peak_mmh_inside", 0.0) or 0.0)
                if peak < _RAINFALL_BAND_EDGES[0]:
                    continue
                for i, edge in enumerate(_RAINFALL_BAND_EDGES[1:], start=0):
                    if peak < edge:
                        band_counts[i] += 1
                        band_pixel_sums[i] += int(cell.get("n_pixels", 0) or 0)
                        break

    peak_hour_utc = None
    peak_hour_pixels = 0
    if per_hour:
        peak_hour_utc = max(per_hour, key=per_hour.get)
        peak_hour_pixels = per_hour[peak_hour_utc]

    rainfall_band_stats = [
        {
            "band": _RAINFALL_BAND_LABELS[i],
            "n_cells": band_counts[i],
            "coupled_pixels": band_pixel_sums[i],
        }
        for i in range(len(_RAINFALL_BAND_LABELS))
    ]
    most_likely_band_label: str | None = None
    if any(s > 0 for s in band_pixel_sums):
        most_likely_band_label = _RAINFALL_BAND_LABELS[
            max(range(len(band_pixel_sums)), key=lambda i: band_pixel_sums[i])
        ]

    return {
        "n_refs_scanned":         n_refs_scanned,
        "n_refs_with_coupling":   n_refs_with_coupling,
        "total_coupled_pixels":   total_coupled_pixels,
        "per_hour":               dict(sorted(per_hour.items())),
        "peak_hour_utc":          peak_hour_utc,
        "peak_hour_pixels":       peak_hour_pixels,
        "rainfall_band_stats":    rainfall_band_stats,
        "most_likely_band_label": most_likely_band_label,
        "most_active_ref":        most_active_ref,
    }


def build_facts_index(
    per_track_loaded: dict, data_root: Path, step_minutes: int,
    coupling_output_dir: Path,
    *,
    pred_coupling: bool = False,
    model_dir: Path | None = None,
    pred_lightning_high_per_lead: dict[int, float] | None = None,
) -> dict:
    """For every (date, reference) that appears in EITHER track's initial
    selection, compute the per-lead facts dict. Renders the coupling-
    mask hero figure ONLY for the peak-activity reference (into
    `coupling_output_dir/`), and computes a period-wide coupling
    summary (hour-of-day + rainfall-band cross-tab) for the conclusion
    paragraph. All other references keep `coupling_figure_path=None`;
    the prompt layer emits a single figure caption for the hero + one
    aggregate conclusion, no per-ref image spam.

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
          "most_active_ref": ("2025-05-14", "12:30") | None,
          "period_coupling_summary": {...},   # from
                                              # _compute_period_coupling_summary
        }
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

    # Pass 1: numeric facts for every reference (no image rendering).
    facts_per_ref: dict[tuple[str, str], dict] = {}
    for date_str, ref_utc in ref_list:
        per_lead = compute_facts_for_reference(
            date_str, ref_utc, lead_minutes, data_root, gp, centre,
            rainfall_available=rainfall_available,
            lightning_available=lightning_available,
        )
        facts_per_ref[(date_str, ref_utc)] = {
            "coupling_figure_path": None,
            "per_lead": per_lead,
        }

    # Pass 2: pick the peak-activity reference and render ONE coupling
    # PNG for it. When --pred_coupling is set, the 3x3 GT-vs-predicted
    # figure IS that image (its Row 1 already contains the GT view, so
    # a separate GT-only file would just duplicate content). Otherwise
    # we fall back to the 1x3 GT-only render.
    most_active_ref = _find_most_active_coupling_ref(facts_per_ref)
    coupling_output_dir.mkdir(parents=True, exist_ok=True)
    if most_active_ref is not None:
        date_str, ref_utc = most_active_ref
        safe_ref = ref_utc.replace(":", "")

        pred_rendered = None
        if pred_coupling:
            if model_dir is None:
                print("  pred_coupling: skipped (no --model_dir set)")
            else:
                pred_coupling_path = (
                    coupling_output_dir
                    / f"coupling_pred_{date_str}_{safe_ref}.png"
                )
                # Operational defaults for the model config — matches what
                # predict_full_domain / validate_predictions use by default.
                # No user knobs on these because the report always wants
                # the base operational models; a variant comparison
                # (finetuned / KD student) belongs to the per-track
                # validation figures, not the coupling image.
                try:
                    pred_rendered = render_pred_coupling_figure(
                        date_str, ref_utc, lead_minutes,
                        data_root, pred_coupling_path,
                        model_dir=model_dir,
                        rainfall_mode="mtg_lightning_opera_rainfall",
                        rainfall_source="dbscan",
                        rainfall_finetuned=False,
                        lightning_mode="mtg_lightning_opera_occurrence",
                        lightning_source="dbscan",
                        lightning_finetuned=False,
                        lightning_kd=False,
                        lightning_low_threshold=0.90,
                        lightning_high_per_lead=pred_lightning_high_per_lead,
                        rainfall_available=rainfall_available,
                        lightning_available=lightning_available,
                    )
                    if pred_rendered is not None:
                        print(f"  pred_coupling: {pred_rendered.name}")
                except Exception as e:  # noqa: BLE001
                    import traceback
                    print(f"  pred_coupling: failed "
                          f"({type(e).__name__}: {e}); "
                          f"falling back to GT-only figure.")
                    traceback.print_exc()
                    pred_rendered = None

        if pred_rendered is not None:
            # Pred figure contains GT as Row 1 — use it as the single
            # coupling image, no separate GT render.
            facts_per_ref[most_active_ref]["coupling_figure_path"] = pred_rendered
        else:
            # Fallback / --pred_coupling not set: GT-only 1x3 render.
            coupling_path = (
                coupling_output_dir / f"coupling_{date_str}_{safe_ref}.png"
            )
            rendered = render_coupling_mask_figure(
                date_str, ref_utc, lead_minutes, data_root, coupling_path,
                rainfall_available=rainfall_available,
                lightning_available=lightning_available,
            )
            facts_per_ref[most_active_ref]["coupling_figure_path"] = rendered

    period_coupling_summary = _compute_period_coupling_summary(
        facts_per_ref, most_active_ref,
    )

    return {
        "grid_centre_lat_lon": centre,
        "lead_minutes": lead_minutes,
        "references": facts_per_ref,
        "most_active_ref": most_active_ref,
        "period_coupling_summary": period_coupling_summary,
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
                              retry_backoff_sec: float = 2.0,
                              max_tokens: int = DEFAULT_OLLAMA_MAX_TOKENS
                              ) -> str:
    """Shared retry wrapper around ollama.chat. Returns the assistant's
    message text.

    `max_tokens` maps to Ollama's `num_predict` option — a hard cap on
    the number of tokens the model may emit. Prevents a single Gemma
    call from wedging the whole report if it enters a repetition /
    hallucination loop. On truncation the returned string is silently
    cut mid-sentence, which is fine for our downstream (fpdf2 wraps
    whatever text it gets) and infinitely preferable to a stall.

    Empty-response handling: Ollama occasionally returns an empty
    message with a non-"stop" `done_reason` — most commonly on the
    first call after loading (`done_reason == "load"`), and sometimes
    with Gemma at `temperature=0.0`. We detect that, log the metadata
    for diagnosis, and retry with a tiny temperature bump (0.05 floor)
    on the next attempt so the model has a non-degenerate path forward.
    Never merges silently: if all retries produce empty text, we raise
    SystemExit with the last response metadata attached so the caller
    knows why the run failed.
    """
    import time
    import ollama

    last_exc: Exception | None = None
    last_resp: dict | None = None
    effective_temp = temperature
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": effective_temp,
                    "seed": seed,
                    "num_predict": int(max_tokens),
                    # keep_alive defaults are fine; long-running batch of
                    # calls will re-use the loaded model.
                },
            )
            # ollama.chat may return either a plain dict or a typed
            # ChatResponse; both expose ["message"]["content"] and
            # a top-level "done_reason".
            try:
                content = resp["message"]["content"]
            except (TypeError, KeyError):
                content = getattr(resp.message, "content", "")  # type: ignore[union-attr]
            content = content or ""
            if content.strip():
                return content
            # Empty response — dig out enough metadata to explain why
            # and try again with a nudge on temperature.
            try:
                done_reason = resp.get("done_reason")
            except AttributeError:
                done_reason = getattr(resp, "done_reason", None)
            last_resp = {
                "done_reason": done_reason,
                "attempt": attempt + 1,
                "temperature": effective_temp,
                "seed": seed,
            }
            print(
                f"    [warn] Ollama returned an empty message on "
                f"attempt {attempt + 1}/{max_retries} "
                f"(done_reason={done_reason!r}, temp={effective_temp}). "
                f"Retrying with a small temperature nudge.",
                flush=True,
            )
            # Nudge temperature so Gemma has a non-degenerate sampling
            # distribution on the retry. 0.05 is enough to break the
            # deterministic empty-output path without materially
            # changing the tone of the answer.
            effective_temp = max(effective_temp + 0.05, 0.05)
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
    if last_exc is not None:
        raise SystemExit(
            f"Ollama request failed after {max_retries} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        )
    raise SystemExit(
        f"Ollama returned an empty message on all "
        f"{max_retries} attempts.  Last response metadata: {last_resp}\n"
        f"Common causes:\n"
        f"  - Cold-start ('load') on the first call: run the request "
        f"once manually to warm the model, or restart Ollama.\n"
        f"  - Gemma + temperature=0.0 sampling collapse: pass "
        f"--temperature 0.05 (or higher) on the report CLI.\n"
        f"  - System prompt not accepted by this Gemma build: try a "
        f"different --model (e.g. gemma3:12b)."
    )


def ollama_generate_text(
    prompt: str, *, model: str, system: str | None = None,
    seed: int = DEFAULT_OLLAMA_SEED,
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    max_retries: int = 3,
    max_tokens: int = DEFAULT_OLLAMA_MAX_TOKENS,
) -> str:
    """Text-only chat completion. `system` is optional; when set it goes
    into a system-role message ahead of the user prompt. `max_tokens`
    caps the response length so runaway generations can't stall a run."""
    verify_ollama_model_available(model)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _ollama_chat_with_retries(
        messages, model=model, seed=seed, temperature=temperature,
        max_retries=max_retries, max_tokens=max_tokens,
    )


def ollama_generate_vision(
    prompt: str, image_path: Path, *, model: str,
    system: str | None = None,
    seed: int = DEFAULT_OLLAMA_SEED,
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    max_retries: int = 3,
    max_tokens: int = DEFAULT_OLLAMA_MAX_TOKENS,
) -> str:
    """Vision chat: prompt + one PNG. The Ollama Python client accepts
    either a file path or bytes for `images`; we pass the path so it
    handles the read + base64 encoding for us. `max_tokens` caps the
    response length so a runaway vision decode can't stall a run."""
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
        max_retries=max_retries, max_tokens=max_tokens,
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
Write a SHORT meteorological caption (2-3 sentences, no more) for date \
{date_str}, reference time {ref_utc} UTC, describing the peak-activity \
coupled convective event across the three lead times (t+15, t+30, t+45).

FACTS (source of truth for every number you cite):
{facts_block}

Sentence 1 - one line naming the event and its location: cite the biggest \
coupled cell at the lead with the strongest coupling using the phrasing \
"precipitation of X mm/h paired with Y% of the lightning strokes inside \
the same convective cell, in the <cardinal> of Romania". Fill X from \
that cell's peak_mmh_inside, Y from lightning_pct_in_coupled, <cardinal> \
from centroid_cardinal.

Sentence 2 (and optional sentence 3) - track how the coupling evolves \
across t+15 -> t+45: does the coupled cell strengthen, hold, or weaken \
(compare peak_mmh_inside and coupled_cells count across leads).

Hard limits:
- 2-3 sentences total. No opening filler, no closing filler.
- Never cite a number that isn't in FACTS.
- If a lead has no coupled_cells, just say "no coupling at t+N" in a \
  clause; don't spend a whole sentence on it.
"""


# ============================================================================
# Prompt C2 - period-wide rainfall-lightning coupling conclusion (TEXT-ONLY)
# ============================================================================
# The hero image (Prompt C above) is one snapshot. This prompt writes the
# closing paragraph that contextualises it against the rest of the month:
# when in the day did coupling peak, and at what rainfall intensity was
# lightning most likely to fire alongside the rain. Reader flow: skim the
# per-track metrics, look at the hero coupling PNG, then read THIS
# paragraph to know how representative the hero image is of the period.
PROMPT_PERIOD_COUPLING_EN = """\
Write a VERY SHORT summary paragraph (2-3 sentences, no more) placing \
the report's single coupling image in the context of the whole period \
{year:04d}-{month:02d}.

FACTS (source of truth for every number you cite):
{facts_block}

Sentence 1 - the diurnal + intensity pattern: cite peak_hour_utc (the \
UTC hour with the most aggregate coupling) and \
most_likely_rainfall_band_for_coupling (the rainfall intensity band with \
the most coupled pixels across the period).

Sentence 2 (and optional sentence 3) - the hero image's context: does \
hero_reference land inside the peak_hour_utc window and the \
most_likely_rainfall_band_for_coupling? If yes, call it "representative \
of the period's dominant regime"; if no, call it a "high-magnitude \
outlier vs. the typical period pattern".

Hard limits:
- 2-3 sentences total. No filler.
- Never invent hours, bands, or counts.
- If peak_hour_utc, most_likely_rainfall_band_for_coupling, or \
  hero_reference is n/a, output ONE sentence: "the period lacked \
  measurable rainfall-lightning coupling."
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


# ============================================================================
# Gemma output cache — persistent JSON keyed by section id + language
# ============================================================================
# Every Gemma call (English generation AND Romanian translation) is
# checked against this cache first; on a hit the LLM is skipped
# entirely, which lets the user iterate on the PDF layout / prompts /
# prose without paying the LLM cost on every run. On a miss the call
# runs normally and the response is written back to the cache.
#
# Layout on disk: `validation/report_gemma_cache_{yyyy}_{mm}.json`
#     {
#       "english:exec_summary":                 "...",
#       "english:lead_metrics_rainfall_t+15":   "...",
#       "romanian:exec_summary":                "...",
#       "romanian:lead_metrics_rainfall_t+15":  "...",
#       ...
#     }
#
# CLI:
#     --refresh_cache          -- discard every entry before running
#                                  (equivalent to deleting the JSON)
#     --no_cache               -- skip the cache entirely (don't read,
#                                  don't write)
#
# `text` in the cache is stored EXACTLY as Gemma returned it (no
# `.strip()`) so the cache is a faithful record of what the model
# said. Callers still strip on consumption if they want to.
class GemmaCache:
    """Thin JSON-backed cache. Keys are `"{lang}:{section_id}"`; values
    are the raw Gemma response strings."""

    def __init__(self, path: Path, *, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.data: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        if enabled and path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
                print(f"  Gemma cache: loaded {len(self.data)} entries "
                      f"from {path.name}")
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Gemma cache: could not read {path.name} "
                      f"({type(e).__name__}: {e}); starting fresh.")
                self.data = {}

    @staticmethod
    def _key(lang: str, section_id: str) -> str:
        return f"{lang}:{section_id}"

    def get(self, lang: str, section_id: str) -> str | None:
        if not self.enabled:
            return None
        val = self.data.get(self._key(lang, section_id))
        if val is not None:
            self.hits += 1
        return val

    def put(self, lang: str, section_id: str, value: str) -> None:
        if not self.enabled:
            return
        self.data[self._key(lang, section_id)] = value
        self.misses += 1

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"  Gemma cache: wrote {len(self.data)} entries "
              f"({self.hits} hits, {self.misses} new) -> {self.path.name}")


# ----- Prompt E - KD teacher-vs-student comparison (text-only, 1 call) ------
# One paragraph that summarises the KD experiment: how the student
# (satellite-only inputs) compares to the teacher (with LINET) across
# every lead time, whether the gap widens or narrows with lead, and how
# each model's tuned high-threshold behaves. Emitted only when a
# kd_{yyyy}_{mm}_summary.json is present.
PROMPT_KD_SUMMARY_EN = """\
Write ONE meteorological paragraph (4-6 sentences) comparing the \
teacher and student models in the knowledge-distillation experiment \
for the validation period. Audience: meteorologists reviewing whether \
the LINET-free student is operationally usable.

FACTS (source of truth for every number below):
{facts_block}

Requirements:
- Report per-lead FAR / POD / CSI for BOTH models. Frame each metric \
  as "teacher X vs student Y" using the exact values in FACTS.
- Comment on how the gap between teacher and student EVOLVES across \
  t+15 -> t+30 -> t+45: does the student catch up, hold the gap, or \
  fall further behind at longer leads?
- Note how many samples cleared the 90% coverage bar for each model at \
  each lead - this is a proxy for how often the student is "good \
  enough" to substitute for the teacher.
- Briefly comment on the tuned per-lead hysteresis high thresholds: \
  are the student's operating points more or less conservative than \
  the teacher's, and does that make meteorological sense given the \
  student's reduced input stack (no LINET at inference)?
- Do NOT use "we" or "our model"; write as an independent analyst.
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
                    f"lightning-only={cp['n_pixels_lightning_only']}, "
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


def _facts_block_period_coupling(period_summary: dict,
                                  year: int, month: int) -> str:
    """FACTS block for Prompt C2 (period-wide coupling conclusion).

    Text-only render of the aggregated stats produced by
    `_compute_period_coupling_summary`: overall counts, hour-of-day
    activity distribution, rainfall-band -> lightning-coupling cross-tab,
    and a pointer to the hero-image reference. Gemma reads these numbers
    to write the closing paragraph — no images involved.
    """
    lines: list[str] = [
        f"period:                 {year:04d}-{month:02d}",
        f"references_scanned:     {period_summary.get('n_refs_scanned', 0)}",
        f"references_with_coupling: {period_summary.get('n_refs_with_coupling', 0)}",
        f"total_coupled_pixels:   {period_summary.get('total_coupled_pixels', 0)}",
        "",
    ]

    peak_hour = period_summary.get("peak_hour_utc")
    peak_pixels = period_summary.get("peak_hour_pixels", 0)
    if peak_hour is not None:
        lines.append(
            f"peak_hour_utc:          {peak_hour:02d}:00 UTC  "
            f"({peak_pixels} coupled pixels aggregated at this hour)"
        )
    else:
        lines.append("peak_hour_utc:          n/a (no coupling in the period)")

    per_hour = period_summary.get("per_hour") or {}
    if per_hour:
        lines.append("per_hour_utc  (hour -> total coupled pixels):")
        for h, px in sorted(per_hour.items()):
            lines.append(f"  {h:02d}:00 UTC -> {px}")
    else:
        lines.append("per_hour_utc: (empty)")
    lines.append("")

    most_likely = period_summary.get("most_likely_band_label")
    lines.append(
        f"most_likely_rainfall_band_for_coupling: {most_likely or 'n/a'}  "
        f"(band with the most coupled pixels across the period)"
    )
    lines.append("rainfall_band_stats  (peak intensity of each coupled cell -> tally):")
    for entry in period_summary.get("rainfall_band_stats", []):
        lines.append(
            f"  {entry['band']:<12}  n_cells={entry['n_cells']}  "
            f"coupled_pixels={entry['coupled_pixels']}"
        )
    lines.append("")

    hero = period_summary.get("most_active_ref")
    if hero is not None:
        d, r = hero
        lines.append(f"hero_reference: {d} {r} UTC "
                     "(the ONE coupling image included in the report)")
    else:
        lines.append("hero_reference: n/a")

    return "\n".join(lines)


# ============================================================================
# Orchestrators: run every prompt through Gemma and translate the output
# ============================================================================
# Two-stage per section: build FACTS block -> Ollama English call ->
# Ollama Romanian translate call. Sections are kept as a list of
# (section_id, english, romanian) tuples so the PDF layout can iterate in
# order without knowing what generated them.

def _strip_leading_markdown_heading(text: str) -> str:
    """Drop a leading `# … / ## … / ### …` line if Gemma prefixed the
    body with its own section title (already redundant with the PDF
    section heading rendered above it). Also eats one blank line right
    after the heading so the visible body doesn't start with an
    empty paragraph."""
    import re
    stripped = text.lstrip()
    lines = stripped.split("\n")
    if lines and re.match(r"^\s*#{1,6}\s+", lines[0]):
        i = 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        return "\n".join(lines[i:])
    return stripped


def _english_call(cache: "GemmaCache | None", section_id: str, prompt: str,
                  *, model: str, seed: int, temperature: float,
                  max_tokens: int) -> str:
    """Cache-checked English generation. Prints response length either
    way so an empty/short Gemma reply is immediately obvious in the log
    (was the root cause of a run where PDF bodies came out blank).

    Post-processes with `_strip_leading_markdown_heading` — Gemma
    sometimes prefixes the body with its own '## …' title even when the
    prompt asks for paragraphs; that duplicates the section header the
    PDF renders above the body and reads badly."""
    if cache is not None and (cached := cache.get("english", section_id)) is not None:
        preview = cached[:80].replace("\n", " ")
        print(f"    [cache HIT english:{section_id}] len={len(cached)} "
              f"preview={preview!r}", flush=True)
        return cached
    text = ollama_generate_text(
        prompt, model=model, system=PROMPT_SYSTEM_EN,
        seed=seed, temperature=temperature, max_tokens=max_tokens,
    )
    text = _strip_leading_markdown_heading(text)
    preview = text[:80].replace("\n", " ")
    print(f"    [english:{section_id}] len={len(text)} "
          f"preview={preview!r}", flush=True)
    if cache is not None:
        cache.put("english", section_id, text)
    return text


def generate_english_paragraphs(
    per_track_loaded: dict, facts_index: dict, *,
    year: int, month: int, step_minutes: int,
    model: str, seed: int, temperature: float,
    max_tokens: int = DEFAULT_OLLAMA_MAX_TOKENS,
    kd_artefacts: dict | None = None,
    cache: "GemmaCache | None" = None,
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
    english = _english_call(
        cache, "exec_summary",
        PROMPT_EXEC_SUMMARY_EN.format(facts_block=facts),
        model=model, seed=seed, temperature=temperature,
        max_tokens=max_tokens,
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
            section_id = f"lead_metrics_{track}_t+{lead_min}"
            print(f"  [B] {track} metrics at t+{lead_min} ...", flush=True)
            facts = _facts_block_lead_metrics(loaded, lead_min, step_minutes)
            english = _english_call(
                cache, section_id,
                PROMPT_LEAD_METRICS_EN.format(
                    lead_min=lead_min, track=track, facts_block=facts,
                ),
                model=model, seed=seed, temperature=temperature,
                max_tokens=max_tokens,
            )
            sections.append({
                "id": section_id,
                "kind": "lead_metrics",
                "title": f"{track.capitalize()} - t+{lead_min} minutes",
                "english": english.strip(),
                "track": track,
                "lead_min": lead_min,
            })

    # ---- Prompt C: HERO coupling caption (only the peak-activity ref) --
    # Prior versions of the report emitted one figure_caption section per
    # reference (hundreds of Gemma calls per month). We now render + caption
    # only the PEAK reference and fold everything else into a period-wide
    # conclusion (Prompt C2) so the reader gets ONE hero image contextualised
    # against the whole month rather than a wall of near-empty panels.
    lead_minutes = facts_index["lead_minutes"]
    hero_ref = facts_index.get("most_active_ref")
    if hero_ref is not None:
        date_str, ref_utc = hero_ref
        ref_data = facts_index["references"][hero_ref]
        section_id = f"figure_caption_{date_str}_{ref_utc.replace(':', '')}"
        print(f"  [C] hero coupling caption "
              f"{date_str} {ref_utc} (peak-activity) ...", flush=True)
        facts = _facts_block_figure_caption(ref_data, lead_minutes)
        english = _english_call(
            cache, section_id,
            PROMPT_FIGURE_CAPTION_EN.format(
                date_str=date_str, ref_utc=ref_utc, facts_block=facts,
            ),
            model=model, seed=seed, temperature=temperature,
            max_tokens=max_tokens,
        )
        sections.append({
            "id": section_id,
            "kind": "figure_caption",
            "title": f"Peak coupling event - {date_str} "
                     f"reference {ref_utc} UTC",
            "english": english.strip(),
            "date": date_str,
            "ref_utc": ref_utc,
            # figure_path is the ONLY coupling PNG that build_facts_index
            # rendered; every other reference's path is None so the PDF
            # layer implicitly skips them (belt-and-braces: the Prompt C
            # loop above never emits sections for them either).
            # Single coupling image. When --pred_coupling is set this
            # is the 3x3 GT-vs-predicted figure (its Row 1 IS the GT
            # view); otherwise it's the 1x3 GT-only figure. Either way
            # the PDF layout embeds exactly one image for this section.
            "figure_path": ref_data.get("coupling_figure_path"),
        })

    # ---- Prompt C2: period-wide coupling conclusion (TEXT-ONLY) --------
    # Aggregates hour-of-day activity + rainfall-band -> lightning-coupling
    # cross-tab across the whole month. Answers "when in the day were storms
    # most active" and "at what rainfall intensity was lightning coupling
    # most likely" — the reader can then re-contextualise the hero image
    # against the period as a whole without needing per-reference captions.
    # Skipped when there was no coupling anywhere in the month (the
    # aggregate has nothing to say).
    period_summary = facts_index.get("period_coupling_summary") or {}
    if period_summary.get("n_refs_with_coupling", 0) > 0:
        print(f"  [C2] period-wide coupling conclusion ...", flush=True)
        facts = _facts_block_period_coupling(period_summary, year, month)
        english = _english_call(
            cache, "coupling_period_conclusion",
            PROMPT_PERIOD_COUPLING_EN.format(
                year=year, month=month, facts_block=facts,
            ),
            model=model, seed=seed, temperature=temperature,
            max_tokens=max_tokens,
        )
        sections.append({
            "id": "coupling_period_conclusion",
            "kind": "coupling_period_conclusion",
            "title": f"Rainfall-lightning coupling - period summary "
                     f"({year:04d}-{month:02d})",
            "english": english.strip(),
        })

    # ---- Prompt E: KD teacher-vs-student comparison (auto-detected) ---
    # Emitted only when kd_{yyyy}_{mm}_summary.json is on disk. One
    # additional Gemma call + one translation. The 4 metric PNGs and
    # per-reference 3x3 PNGs are embedded in the PDF as decoration; no
    # per-figure vision call.
    if kd_artefacts and kd_artefacts.get("summary_json"):
        print(f"  [E] KD teacher vs student summary ...", flush=True)
        with open(kd_artefacts["summary_json"]) as f:
            kd_summary = json.load(f)
        facts = _facts_block_kd_summary(kd_summary)
        english = _english_call(
            cache, "kd_summary",
            PROMPT_KD_SUMMARY_EN.format(facts_block=facts),
            model=model, seed=seed, temperature=temperature,
            max_tokens=max_tokens,
        )
        sections.append({
            "id": "kd_summary",
            "kind": "kd_summary",
            "title": f"Knowledge distillation - teacher vs student "
                     f"({year:04d}-{month:02d})",
            "english": english.strip(),
            "kd_summary": kd_summary,
            "kd_metric_pngs": kd_artefacts["metric_pngs"],
            "kd_per_ref_pngs": kd_artefacts["per_ref_pngs"],
        })
    return sections


def translate_paragraphs_to_romanian(
    sections: list[dict], *,
    model: str, seed: int, temperature: float,
    max_tokens: int = DEFAULT_OLLAMA_MAX_TOKENS,
    cache: "GemmaCache | None" = None,
) -> list[dict]:
    """For each section, call prompt D once and store the Romanian text
    under 'romanian'. Returns the SAME list (mutated in place). Cache-
    checked per section id so a re-run reuses prior translations."""
    for i, sec in enumerate(sections, 1):
        sid = sec["id"]
        print(f"  [D] translating {i}/{len(sections)}: {sid} ...",
              flush=True)
        cached = cache.get("romanian", sid) if cache is not None else None
        if cached is not None:
            preview = cached[:80].replace("\n", " ")
            print(f"    [cache HIT romanian:{sid}] len={len(cached)} "
                  f"preview={preview!r}", flush=True)
            ro = cached
        else:
            ro = ollama_generate_text(
                PROMPT_USER_RO_TRANSLATE.format(english_text=sec["english"]),
                model=model, system=PROMPT_SYSTEM_RO_TRANSLATE,
                seed=seed, temperature=temperature, max_tokens=max_tokens,
            )
            preview = ro[:80].replace("\n", " ")
            print(f"    [romanian:{sid}] len={len(ro)} "
                  f"preview={preview!r}", flush=True)
            if cache is not None:
                cache.put("romanian", sid, ro)
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

    def __init__(self, output_path: Path, *, toc_title: str = "Cuprins"):
        from fpdf import FPDF
        reg, bold = _find_unicode_fonts()

        # FPDF subclass that renders a bottom-right page number on
        # every non-cover page. `footer()` is called automatically
        # after each `add_page()` and each auto-page-break.
        class _NumberedFPDF(FPDF):
            def footer(inner_self):   # noqa: N805 — fpdf2 convention
                # Skip the cover page (page 1) — convention is to number
                # from page 2 (TOC) onward.
                if inner_self.page_no() <= 1:
                    return
                inner_self.set_y(-15)  # 15mm from the bottom edge
                try:
                    inner_self.set_font("DejaVu", "", 9)
                except Exception:
                    inner_self.set_font("Helvetica", "", 9)
                inner_self.set_text_color(120, 120, 120)
                inner_self.cell(0, 8, str(inner_self.page_no()),
                                align="R")
                inner_self.set_text_color(0, 0, 0)

        self.pdf = _NumberedFPDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=25)
        self.pdf.set_margins(left=20, top=20, right=20)
        self.pdf.add_font("DejaVu", "", str(reg))
        self.pdf.add_font("DejaVu", "B", str(bold))
        self.pdf.set_font("DejaVu", "", 11)
        self.output_path = output_path
        self._toc_title = toc_title
        # Page width available for content after margins (A4 = 210mm).
        self.content_width_mm = 210 - 20 - 20
        # TOC accumulates (title, page_number) pairs as sections render.
        self._toc_entries: list[tuple[str, int]] = []
        # Placeholder page number reserved for the TOC (populated by
        # `reserve_toc_page`, filled by `_write_toc_content` at save).
        self._toc_page_no: int | None = None

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
              bilingual: bool = False,
              *,
              period_label: str = "Perioada",
              tracks_label: str = "Fluxuri incluse",
              generated_label: str = "Generat"):
        """Cover page. Labels default to Romanian but are overridden
        by `render_pdf_report` from the active language pack so an
        English run reads 'Period / Streams included / Generated'
        instead of the Romanian originals."""
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
        self._mc(7, f"{period_label}: {period}", align="C")
        self._mc(7, f"{tracks_label}: {', '.join(tracks)}", align="C")
        self.pdf.ln(3)
        self.pdf.set_font("DejaVu", "", 10)
        self.pdf.set_text_color(110, 110, 110)
        self._mc(5, f"{generated_label}: {generation_ts}", align="C")
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
        """One section per (date, ref): coupling figure + caption. The
        figure is either the 1x3 GT-only render or the 3x3 GT-vs-
        predicted render (whichever `build_facts_index` produced); the
        layout is the same either way — one image, then the caption."""
        self.new_page()
        self._record_toc(ref_title_ro)
        self.title(ref_title_ro, size=14)
        if figure_path is not None and figure_path.is_file():
            # 3x3 pred-coupling image is taller than the 1x3 GT-only
            # figure, so we give it more vertical room; the aspect-
            # preserving scaler inside image_full_width falls back to
            # width-capped for the shorter 1x3 image automatically.
            self.image_full_width(figure_path, max_height_mm=170)
        self.body(romanian, size=11)
        if bilingual and english:
            self.body(english, size=9, italic=True)

    def kd_section(self, title_ro: str, romanian: str,
                   metric_pngs: dict, per_ref_pngs: list,
                   english: str | None = None,
                   bilingual: bool = False):
        """Knowledge-distillation comparison section:
             1 title,
             1 Romanian body paragraph,
             (optional) 1 English body paragraph in italics,
             4 metric PNGs (FAR / POD / CSI / IoU) embedded in fixed order,
             per-reference 3x3 comparison figures embedded one-per-page.
        The metric PNGs live in the SAME validation dir as the other artefacts;
        we resolve them by the (metric -> Path) dict produced by
        discover_kd_artefacts.
        """
        self.new_page()
        self._record_toc(title_ro)
        self.title(title_ro, size=16)
        self.body(romanian, size=11)
        if bilingual and english:
            self.body(english, size=9, italic=True)
        # Metric bars (4 files, canonical order)
        for metric in ("FAR", "POD", "CSI", "IoU"):
            path = metric_pngs.get(metric)
            if path is not None and path.is_file():
                self.pdf.ln(2)
                self.pdf.set_font("DejaVu", "B", 11)
                self._mc(6, f"Metrica {metric}")
                self.image_full_width(path, max_height_mm=95)
        # Per-reference 3x3 comparison figures — one per page so each
        # renders large enough to actually read.
        for p in per_ref_pngs:
            if not p.is_file():
                continue
            self.new_page()
            self.pdf.set_font("DejaVu", "B", 12)
            # e.g. kd_2025_05_2025-05-14_1230.png -> "2025-05-14 12:30 UTC"
            stem = p.stem
            parts = stem.split("_")
            if len(parts) >= 4:
                caption = f"{parts[-2]} ref {parts[-1][:2]}:{parts[-1][2:]} UTC"
            else:
                caption = p.name
            self._mc(6, f"Comparație KD - {caption}")
            self.image_full_width(p, max_height_mm=200)

    def data_appendix(self, per_track_loaded: dict, step_minutes: int,
                       *, appendix_title: str = "Anexa - statistici pe eșantion",
                       appendix_toc: str = "Anexa - date sumar",
                       samples_word: str = "eșantioane",
                       header_labels: tuple[str, str, str, str] = (
                           "Orizont", "min", "mediu", "max"),
                       ):
        """Per-track table: min / mean / max coverage per lead time.
        Localisable — pass the label kwargs to switch to English."""
        import numpy as np
        self.new_page()
        self._record_toc(appendix_toc)
        self.title(appendix_title, size=16)
        for track, loaded in per_track_loaded.items():
            self.pdf.set_font("DejaVu", "B", 12)
            self._mc(6, f"{track.capitalize()} - "
                        f"{len(loaded['samples'])} {samples_word}")
            self.pdf.ln(1)
            # Choose the "primary" per-lead metric per track.
            primary_key = "iou_mask" if track == "rainfall" else "iou"
            h0, h1, h2, h3 = header_labels
            header = f"{h0:<12}{h1:>8}{h2:>10}{h3:>8}"
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

    def reserve_toc_page(self) -> None:
        """Reserve page 2 for the table of contents. Adds an empty page
        immediately so subsequent sections start on page 3+; the actual
        TOC content is written by `_write_toc_content` right before
        `save()` so the entries reflect every section's final page
        number.

        Manual navigation (`pdf.page = self._toc_page_no`) is used
        instead of fpdf2's `insert_toc_placeholder(callback, pages=1)`
        — the callback was silently producing an empty page 2 in this
        codebase; manual navigation is bulletproof.
        """
        self.new_page()
        self._toc_page_no = self.pdf.page_no()

    def _write_toc_content(self) -> None:
        """Navigate back to the reserved TOC page and render the TOC
        entries there. Called once, right before `save()`.

        Setting `self.pdf.page = N` is the low-level fpdf2 mechanism for
        switching the active page buffer — every `multi_cell` / `set_xy`
        that follows writes into page N. We restore the previous page
        afterwards so downstream save state stays sane.

        Each entry is a fpdf2 internal link (`add_link` + `set_link`)
        stamped over the whole 'title … p.NN' line via `multi_cell(
        link=…)`, so the reader can click the entry in the PDF viewer
        to jump straight to that section's first page.
        """
        if self._toc_page_no is None:
            return
        saved_page = self.pdf.page
        try:
            self.pdf.page = self._toc_page_no
            # Reset the cursor to the top of the page — pdf.page = N
            # doesn't reset x/y, and by the time we get here the
            # cursor might be halfway down the last-rendered page.
            self.pdf.set_xy(20, 20)
            self.pdf.set_font("DejaVu", "B", 16)
            self.pdf.set_x(20)
            self.pdf.multi_cell(0, 8, self._toc_title)
            self.pdf.ln(2)
            self.pdf.set_font("DejaVu", "", 11)
            for title, page in self._toc_entries:
                dots = "." * max(3, 60 - len(title))
                line = f"{title} {dots} p.{page}"
                link_id = self.pdf.add_link()
                self.pdf.set_link(link_id, page=page)
                self.pdf.set_x(20)
                self.pdf.multi_cell(0, 5.5, line, link=link_id)
        finally:
            self.pdf.page = saved_page

    def save(self):
        # Fill the reserved TOC page (if any) with entries collected
        # while sections rendered.
        self._write_toc_content()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf.output(str(self.output_path))


# ---------------------------------------------------------------------------
# Localised month names + local-time helpers used by the cover page.
# Locale-agnostic (Windows Python ships without proper locale data by
# default, so `%B` isn't reliable) — we hard-code the two supported
# languages.
# ---------------------------------------------------------------------------
_MONTH_NAMES = {
    "ro": ["", "ianuarie", "februarie", "martie", "aprilie", "mai",
           "iunie", "iulie", "august", "septembrie", "octombrie",
           "noiembrie", "decembrie"],
    "en": ["", "January", "February", "March", "April", "May",
           "June", "July", "August", "September", "October",
           "November", "December"],
}


def _format_period_month_year(year: int, month: int, language: str) -> str:
    """`(2026, 8, 'ro') → 'august 2026'`, `(2026, 8, 'en') → 'August 2026'`."""
    names = _MONTH_NAMES.get(language, _MONTH_NAMES["en"])
    if not (1 <= month <= 12):
        return f"{year:04d}-{month:02d}"
    return f"{names[month]} {year}"


def _romanian_local_now_str() -> str:
    """Current wall time in Europe/Bucharest, formatted
    `'YYYY-MM-DD HH:MM EET'` or `'... EEST'` depending on DST.

    Uses `zoneinfo` when available (Python 3.9+ with the `tzdata`
    package on Windows). Falls back to a hand-rolled EU DST rule
    (last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC
    = EEST +03:00, otherwise EET +02:00) so the report still generates
    on stock Windows Python installs that don't ship system tzdata.
    """
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Bucharest")
        local = now_utc.astimezone(tz)
        return local.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        # Manual DST fallback.
        from calendar import monthrange

        def _last_sunday(y: int, m: int) -> datetime:
            last_day = monthrange(y, m)[1]
            d = datetime(y, m, last_day)
            while d.weekday() != 6:   # 6 == Sunday for weekday()
                d = d.replace(day=d.day - 1)
            return d

        y = now_utc.year
        dst_start = _last_sunday(y, 3).replace(
            hour=1, tzinfo=timezone.utc,
        )
        dst_end = _last_sunday(y, 10).replace(
            hour=1, tzinfo=timezone.utc,
        )
        if dst_start <= now_utc < dst_end:
            offset = timedelta(hours=3)
            tzname = "EEST"
        else:
            offset = timedelta(hours=2)
            tzname = "EET"
        local = now_utc.astimezone(timezone(offset))
        return local.strftime("%Y-%m-%d %H:%M ") + tzname


_REPORT_LABELS = {
    "ro": {
        "cover_title":         "Raport de validare — {period}",
        "toc_title":           "Cuprins",
        "exec_summary":        "Rezumat executiv",
        "track_metrics_pref":  "Performanță pe orizont — ",
        "rainfall":            "clasificarea cantităților de precipitații instantanee (OPERA)",
        "lightning":           "detecția descărcărilor electrice (LINET)",
        "lead_prefix":         "Orizontul t+",
        "lead_suffix":         " minute",
        "peak_coupling":       "Eveniment de vârf al cuplării — {date} referință {ref} UTC",
        "kd_title":            "Distilare cunoștințe — profesor vs. student",
        "period_coupling":     "Cuplare precipitații-fulgere — sinteza perioadei",
        "appendix_title":      "Anexa - statistici pe eșantion",
        "appendix_toc":        "Anexa - date sumar",
        "cover_period":        "Perioada",
        "cover_tracks":        "Fluxuri incluse",
        "cover_generated":     "Generat",
    },
    "en": {
        "cover_title":         "Validation report — {period}",
        "toc_title":           "Table of contents",
        "exec_summary":        "Executive summary",
        "track_metrics_pref":  "Per-lead performance — ",
        "rainfall":            "instantaneous rainfall intensity classification (OPERA)",
        "lightning":           "electrical discharge detection (LINET)",
        "lead_prefix":         "Lead time t+",
        "lead_suffix":         " minutes",
        "peak_coupling":       "Peak coupling event — {date} reference {ref} UTC",
        "kd_title":            "Knowledge distillation — teacher vs student",
        "period_coupling":     "Rainfall-lightning coupling — period summary",
        "appendix_title":      "Appendix - per-sample statistics",
        "appendix_toc":        "Appendix - summary data",
        "cover_period":        "Period",
        "cover_tracks":        "Streams included",
        "cover_generated":     "Generated",
    },
}


def render_pdf_report(
    sections: list[dict],
    per_track_loaded: dict,
    *,
    header_png: Path | None,
    output_path: Path,
    year: int, month: int, step_minutes: int,
    bilingual: bool = False,
    language: str = "ro",
) -> None:
    """Compose the standalone PDF from generated section paragraphs,
    plus the metrics.png and per-reference coupling figures that already
    sit on disk in the validation directory.

    `language` picks the label set (Romanian or English) for every
    hardcoded string in the layout (cover title, section headers, TOC
    title, appendix). Body prose is always read from `sec["romanian"]`
    — when `--language en` the orchestrator copies the raw English into
    that slot so the layout code stays language-agnostic.
    """
    L = _REPORT_LABELS.get(language, _REPORT_LABELS["ro"])
    track_names = {"rainfall": L["rainfall"], "lightning": L["lightning"]}
    tracks_included = [track_names.get(t, t) for t in per_track_loaded]

    period_primary = _format_period_month_year(year, month, language)
    period_secondary = _format_period_month_year(
        year, month, "en" if language != "en" else "ro",
    )

    doc = _ReportPDF(output_path, toc_title=L["toc_title"])
    doc.cover(
        header_png=header_png,
        title_ro=L["cover_title"].format(period=period_primary),
        title_en=(_REPORT_LABELS["en"]["cover_title"]
                  .format(period=period_secondary)),
        period=period_primary,
        tracks=tracks_included,
        generation_ts=_romanian_local_now_str(),
        bilingual=bilingual,
        period_label=L["cover_period"],
        tracks_label=L["cover_tracks"],
        generated_label=L["cover_generated"],
    )
    # Reserve page 2 for the TOC. We write the actual TOC content
    # right before `doc.save()` via manual page-2 navigation, so the
    # entries reflect every section's true page number.
    doc.reserve_toc_page()

    # Index sections by kind for structured layout.
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        by_kind[s["kind"]].append(s)

    # -------- Executive summary
    for sec in by_kind.get("exec_summary", []):
        doc.section(
            title=L["exec_summary"],
            romanian=sec["romanian"],
            english=sec.get("english"),
            bilingual=bilingual,
        )

    # -------- Executive summary label swap (uses LABELS)
    # (The exec_summary loop above hardcoded the Romanian title; patched
    # in-place below to reuse the LABELS dict picked at the top of this
    # function.)

    # -------- Per-track metrics sections
    for track, loaded in per_track_loaded.items():
        track_title = f"{L['track_metrics_pref']}{track_names.get(track, track)}"
        # Group prompt-B sections for this track by lead order.
        lead_secs = [s for s in by_kind.get("lead_metrics", [])
                     if s.get("track") == track]
        lead_secs.sort(key=lambda s: s["lead_min"])
        subsections = [
            (f"{L['lead_prefix']}{s['lead_min']}{L['lead_suffix']}",
             s["romanian"], s.get("english"))
            for s in lead_secs
        ]
        doc.track_metrics_section(
            track_title_ro=track_title,
            metrics_png=loaded.get("metrics_png"),
            subsections=subsections,
            bilingual=bilingual,
        )

    # -------- Peak coupling event: ONE hero image + its Gemma caption.
    # `generate_english_paragraphs` only emits a figure_caption section
    # for the peak-activity reference (build_facts_index also only
    # renders that one PNG). The loop keeps the same shape as before
    # but iterates over exactly one element under the new design.
    for sec in by_kind.get("figure_caption", []):
        ref_title = L["peak_coupling"].format(
            date=sec['date'], ref=sec['ref_utc'],
        )
        doc.reference_section(
            ref_title_ro=ref_title,
            figure_path=sec.get("figure_path"),
            romanian=sec["romanian"],
            english=sec.get("english"),
            bilingual=bilingual,
        )

    # -------- KD comparison (auto-detected; may be absent for pure rain/light runs)
    for sec in by_kind.get("kd_summary", []):
        doc.kd_section(
            title_ro=L["kd_title"],
            romanian=sec["romanian"],
            metric_pngs=sec.get("kd_metric_pngs", {}),
            per_ref_pngs=sec.get("kd_per_ref_pngs", []),
            english=sec.get("english"),
            bilingual=bilingual,
        )

    # -------- Period-wide coupling conclusion: LAST narrative section.
    # Text-only, no image. Contextualises the hero image against the
    # whole month (hour-of-day pattern, rainfall-band -> lightning-
    # coupling rate). Placed last so the reader closes with the
    # period-scale takeaway; rendered via reference_section with
    # figure_path=None so the same method handles both the hero + this
    # conclusion consistently.
    for sec in by_kind.get("coupling_period_conclusion", []):
        doc.reference_section(
            ref_title_ro=L["period_coupling"],
            figure_path=None,
            romanian=sec["romanian"],
            english=sec.get("english"),
            bilingual=bilingual,
        )

    # -------- Data appendix
    doc.data_appendix(
        per_track_loaded, step_minutes,
        appendix_title=L["appendix_title"],
        appendix_toc=L["appendix_toc"],
        samples_word=("samples" if language == "en" else "eșantioane"),
        header_labels=(
            ("Lead", "min", "mean", "max") if language == "en"
            else ("Orizont", "min", "mediu", "max")
        ),
    )

    # TOC page 2 is filled in automatically during `doc.save()` via the
    # placeholder callback (`_render_toc_at_placeholder`).
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
    parser.add_argument("--max_tokens", type=int,
                        default=DEFAULT_OLLAMA_MAX_TOKENS,
                        help=f"Hard cap on tokens per Gemma response "
                             f"(Ollama num_predict). Default "
                             f"{DEFAULT_OLLAMA_MAX_TOKENS}. Prevents a "
                             f"single paragraph from wedging the whole "
                             f"run if the model enters a repetition / "
                             f"hallucination loop; the response is cut "
                             f"mid-sentence when the ceiling is hit.")
    parser.add_argument("--refresh_cache", action="store_true",
                        help="Discard the persisted Gemma cache (JSON) "
                             "before running, forcing every English + "
                             "Romanian paragraph to be regenerated from "
                             "scratch. Use when a prompt has changed or "
                             "the previous run's outputs look off.")
    parser.add_argument("--no_cache", action="store_true",
                        help="Do not read from OR write to the Gemma "
                             "cache. Every call hits the LLM and the "
                             "on-disk cache is left untouched.")
    parser.add_argument("--language", type=str, default="ro",
                        choices=["ro", "en"],
                        help="Body language for the report. 'ro' "
                             "(default) generates English via Gemma "
                             "AND translates every paragraph to "
                             "Romanian (two Gemma calls per section). "
                             "'en' skips the translation phase — the "
                             "raw English paragraph is what lands in "
                             "the PDF. Roughly halves generation time "
                             "and doubles as a debug switch: if the "
                             "English-only PDF renders correctly then "
                             "any body-text bug lives in the "
                             "translation stage.")
    parser.add_argument("--bilingual", action="store_true",
                        help="Include the English source paragraph below each "
                             "Romanian body in the PDF. Default: Romanian only.")
    parser.add_argument("--skip_pdf", action="store_true",
                        help="Skip the PDF render step (useful when iterating "
                             "on prompts / facts extraction).")

    # --- Predicted-coupling companion figure -----------------------------
    # Two flags only. Everything else about the pred-coupling figure
    # (rainfall/lightning modes, sources, threshold) is hardcoded to the
    # operational defaults: base checkpoints, `mtg_lightning_opera_rainfall`
    # (rainfall) and `mtg_lightning_opera_occurrence` (lightning teacher),
    # dbscan source, hysteresis low=0.90. Per-lead high thresholds are
    # auto-read from `lightning_{yyyy}_{mm}_summary.json` when present.
    parser.add_argument("--pred_coupling", action="store_true",
                        help="Replace the GT-only coupling image with the "
                             "3x3 GT-vs-predicted coupling figure (Row 1: "
                             "GT, Row 2: pred per-area, Row 3: pred per-"
                             "class). Loads the operational rainfall + "
                             "lightning checkpoints from --model_dir. "
                             "Adds ~10-30s to the run.")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory with the model checkpoints "
                             "(only used when --pred_coupling is set).")

    args = parser.parse_args()

    # Overall wall-clock so the final line can report end-to-end runtime
    # (facts extraction + Gemma calls + Romanian translation + PDF layout).
    import time as _time
    _t_start = _time.time()

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

    # If pred-coupling is requested, dig the tuned per-lead high-threshold
    # values out of the lightning summary (mirrors the operational path in
    # predict_full_domain --validation_summary). Falls back to
    # lightning_postproc.DEFAULT_HIGH_THRESHOLD (0.95) per lead when the
    # summary is absent or missing the block.
    pred_lightning_high_per_lead: dict[int, float] | None = None
    if args.pred_coupling:
        light = per_track_loaded.get("lightning")
        if light is not None:
            pp = light["summary"].get("post_processing") or {}
            named = pp.get("high_threshold_per_lead") or {}
            resolved: dict[int, float] = {}
            for offset in (1, 2, 3):
                key = f"t+{offset * step_minutes}"
                if key in named:
                    resolved[offset] = float(named[key])
            if resolved:
                pred_lightning_high_per_lead = resolved
                print(f"  pred_coupling: per-lead high thresholds from "
                      f"lightning summary: {resolved}")

    facts_index: dict | None = None
    try:
        facts_index = build_facts_index(
            per_track_loaded, Path(args.data_root), step_minutes,
            coupling_output_dir=validation_dir / "rainfall_lightning_coupling",
            pred_coupling=args.pred_coupling,
            model_dir=Path(args.model_dir) if args.pred_coupling else None,
            pred_lightning_high_per_lead=pred_lightning_high_per_lead,
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
    # KD comparison section is auto-detected: if kd_{yyyy}_{mm}_summary.json
    # is present in the validation dir, we include a Teacher-vs-Student
    # section (1 extra Gemma call + Romanian translation, plus embedded
    # metric + per-reference PNGs).
    kd_artefacts = discover_kd_artefacts(validation_dir, args.year, args.month)
    if kd_artefacts["summary_json"] is not None:
        print(f"  KD outputs detected ({kd_artefacts['stem']}): "
              f"{len(kd_artefacts['metric_pngs'])} metric PNG(s), "
              f"{len(kd_artefacts['per_ref_pngs'])} per-reference PNG(s). "
              f"Including KD comparison section.")
    else:
        print("  No KD outputs present; skipping KD comparison section.")

    # Gemma output cache. Persists English + Romanian per section id in
    # `validation/report_gemma_cache_{yyyy}_{mm}.json`, so a re-run with
    # the same prompts / facts skips the LLM entirely and reuses the
    # prior outputs. `--refresh_cache` discards the file before running;
    # `--no_cache` disables both read and write.
    cache_path = (validation_dir /
                  f"report_gemma_cache_{args.year:04d}_{args.month:02d}.json")
    if args.refresh_cache and cache_path.is_file():
        print(f"--refresh_cache: removing {cache_path.name}")
        try:
            cache_path.unlink()
        except OSError as e:
            print(f"  WARN: could not remove {cache_path.name}: {e}")
    cache = GemmaCache(cache_path, enabled=(not args.no_cache))

    print()
    print("Generating English paragraphs via Gemma ...")
    sections = generate_english_paragraphs(
        per_track_loaded, facts_index,
        year=args.year, month=args.month, step_minutes=step_minutes,
        model=args.model, seed=args.seed, temperature=args.temperature,
        max_tokens=args.max_tokens,
        kd_artefacts=kd_artefacts if kd_artefacts["summary_json"] else None,
        cache=cache,
    )
    print(f"  {len(sections)} sections generated in English.")

    if args.language == "en":
        # English-only path: skip the translation phase entirely, copy
        # the raw English body into the layout slot (`sec["romanian"]`
        # is the layout's language-agnostic body slot — repurposed here
        # so the PDF code stays untouched).
        print()
        print("--language en: skipping Romanian translation phase.")
        for sec in sections:
            sec["romanian"] = sec.get("english", "")
    else:
        print()
        print("Translating each section to Romanian via Gemma ...")
        translate_paragraphs_to_romanian(
            sections,
            model=args.model, seed=args.seed, temperature=args.temperature,
            max_tokens=args.max_tokens,
            cache=cache,
        )
        print(f"  {len(sections)} sections translated.")
    cache.save()

    # Preview: first 200 chars of each section's rendered body. When
    # --language en the "romanian" slot actually carries the English
    # text (we copied it in above so the PDF layout stays language-
    # agnostic), so label the preview accordingly.
    preview_label = "EN" if args.language == "en" else "RO"
    print()
    print(f"--- Section preview ({len(sections)} sections) ---")
    for sec in sections:
        body = sec["romanian"]
        preview = (body[:200] + "...") if len(body) > 200 else body
        print(f"  [{sec['kind']}] {sec['title']}")
        print(f"    {preview_label}: {preview}")
        print()

    # ---------------------------------------------------------------------
    # Item 9: render the PDF
    # ---------------------------------------------------------------------
    if args.skip_pdf:
        print("--skip_pdf set; not rendering the PDF.")
        _elapsed = _time.time() - _t_start
        print(f"Total report-generation wall time: "
              f"{_elapsed:.1f}s ({_elapsed / 60.0:.1f} min)")
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
        language=args.language,
    )
    print(f"  Wrote {output_pdf} ({output_pdf.stat().st_size / 1024:.1f} KB)")
    _elapsed = _time.time() - _t_start
    print(f"Total report-generation wall time: "
          f"{_elapsed:.1f}s ({_elapsed / 60.0:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
