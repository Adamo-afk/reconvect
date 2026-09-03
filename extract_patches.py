"""
COALITION-4 patch extraction pipeline.

Driven by the train/val/test_data_<source>.csv files produced by
extract_patch_seq_for_datasets.py - those splits already incorporate the
cross-product manifest gate, so this script just walks the union of
their per-row windows and slices the active patches at each (date, time)
from the cached reprojected data.

Resolution categories:
    HR (1km)  -> no pooling   -> 256x256  (lightning, MTG vis_06)
    MR (2km)  -> 2x2 avg pool -> 128x128  (MTG IR/WV channels)

Output:
    our_data/patches/{date}/{variable}_{HHMM}_{HR|MR}.npy
    Each file has shape (num_active_patches, H, W).
    Patch order matches the active patches from the source patch-index
    (patch_index.csv), so the idx_t* columns in the split CSVs index
    into these files correctly.

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python extract_patches.py
    python extract_patches.py --date 2025-05-15
    python extract_patches.py --products satellite_MTG opera
    python extract_patches.py --period w48 --products opera
"""

import numpy as np
import os
import re
import sys
import csv
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from compress_datasets import array_exists, load_array
from pipeline_config import SOURCE


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)

PATCH_SIZE = 256
N_COLS = 6
N_ROWS = 3
N_PATCHES = N_COLS * N_ROWS

# -----------------------------------------------------------------------------
# Product registry: {variable_name: (product_group, resolution_tag, pool_factor)}
#   pool_factor: 1 = no pooling, 2 = 2×2, 4 = 4×4
# -----------------------------------------------------------------------------

MTG_PRODUCTS = {
    'vis_06': ('satellite_MTG', 'HR', 1),
    'ir_38':  ('satellite_MTG', 'MR', 2),
    'ir_105': ('satellite_MTG', 'MR', 2),
    'wv_63':  ('satellite_MTG', 'MR', 2),
    'wv_73':  ('satellite_MTG', 'MR', 2),
}

LIGHTNING_PRODUCTS = {
    'density':    ('lightning', 'HR', 1),
    'current':    ('lightning', 'HR', 1),
    'occurrence': ('lightning', 'HR', 1),
}

# OPERA: max reflectivity (dBZ) + rainfall_rate (mm/h), 2 km native → 2× pool.
# `opera_rainfall_rate_hr` is an alias of `opera_rainfall_rate` extracted at
# HR (no pooling, 256×256) so it can be used as the multi-class label target
# in OPERA-driven modes. Both aliases point to the same source `.npy`.
OPERA_PRODUCTS = {
    'opera_reflectivity':     ('opera', 'MR', 2),
    'opera_rainfall_rate':    ('opera', 'MR', 2),
    'opera_rainfall_rate_hr': ('opera', 'HR', 1),
}

# Group name → CLI flag mapping
PRODUCT_GROUPS = {
    'satellite_MTG': MTG_PRODUCTS,
    'lightning':     LIGHTNING_PRODUCTS,
    'opera':         OPERA_PRODUCTS,
}

# Map canonical (prefixed) OPERA variable name → on-disk folder/short name.
# Both rainfall variants resolve to the same reprojected file.
OPERA_VAR_TO_DISK = {
    'opera_reflectivity':     'reflectivity',
    'opera_rainfall_rate':    'rainfall_rate',
    'opera_rainfall_rate_hr': 'rainfall_rate',
}


# =============================================================================
# Grid utilities
# =============================================================================

def get_patch_bounds(patch_number):
    """Get (r0, r1, c0, c1) for a 1-indexed patch number."""
    idx = patch_number - 1
    row = idx // N_COLS
    col = idx % N_COLS
    r0 = row * PATCH_SIZE
    c0 = col * PATCH_SIZE
    return r0, r0 + PATCH_SIZE, c0, c0 + PATCH_SIZE


def average_pool(data, factor):
    """
    Downsample 2D array by averaging non-overlapping blocks.

    Args:
        data: 2D array (H, W)
        factor: pooling factor (2 → 2×2 blocks, 4 → 4×4 blocks)

    Returns:
        2D array (H/factor, W/factor)
    """
    if factor == 1:
        return data

    h, w = data.shape
    new_h, new_w = h // factor, w // factor
    return data[:new_h * factor, :new_w * factor].reshape(
        new_h, factor, new_w, factor
    ).mean(axis=(1, 3))


# =============================================================================
# Patch index reader
# =============================================================================

# =============================================================================
# Pool staleness stamp
# =============================================================================
# A patch file is an ARRAY OF TILES with nothing recording which tile sits
# in which slot: slot k means "the k-th active patch at this timestep",
# and that mapping lives only in patch_index.csv. Re-run identify_patches
# and a patch that becomes active inserts into the middle of the list,
# shifting every slot after it. The split CSVs' idx_t* columns then point
# at the wrong tile - and because the shapes still line up, nothing
# raises. A model trained on it learns to predict one region from another
# a thousand kilometres away.
#
# Since extract_patches skips files that already exist, that stale state
# is sticky: nothing regenerates them. So each date carries a stamp of
# the active-patch lists it was built from, and any timestep whose list
# has since changed is re-extracted instead of skipped.

STAMP_NAME = "_patch_index.json"


def _read_stamp(out_dir):
    """Active-patch lists this date's files were built from, or None."""
    path = os.path.join(out_dir, STAMP_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
        return blob.get("active") or None
    except (OSError, ValueError):
        return None


def _write_stamp(out_dir, active_by_hhmm, index_digest):
    """Record what the files in `out_dir` were built from."""
    path = os.path.join(out_dir, STAMP_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "patch_index_sha256": index_digest,
            "written_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "active": active_by_hhmm,
        }, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _index_digest(data_root, source):
    """SHA-256 of the patch index, recorded for provenance."""
    import hashlib

    try:
        csv_path = _resolve_index_csv(data_root, source)
    except ValueError:
        return None
    h = hashlib.sha256()
    try:
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def audit_pool(data_root, source):
    """Report timesteps whose files disagree with the current index.

    Answers the question the stamp exists to prevent, for a pool built
    before stamping: which dates would silently feed the wrong tile?
    """
    index = read_patch_index(data_root, source=source)
    if not index:
        return 1
    output_root = os.path.join(data_root, 'patches')
    by_date: dict[str, dict[str, list[int]]] = {}
    for (date_str, time_str), active in index.items():
        by_date.setdefault(date_str, {})[time_str.replace(':', '')] = active

    stamped = unstamped = drifted = 0
    drift_dates: list[str] = []
    unstamped_dates: list[str] = []
    for date_str in sorted(by_date):
        out_dir = os.path.join(output_root, date_str)
        if not os.path.isdir(out_dir):
            continue
        stamp = _read_stamp(out_dir)
        if stamp is None:
            unstamped += 1
            unstamped_dates.append(date_str)
            continue
        stamped += 1
        bad = [h for h, act in by_date[date_str].items()
               if h in stamp and list(stamp[h]) != list(act)]
        if bad:
            drifted += 1
            drift_dates.append(f"{date_str} ({len(bad)} timestep(s))")

    print(f"Pool audit  : {output_root}")
    print(f"  stamped dates       : {stamped}")
    print(f"  drifted dates       : {drifted}")
    print(f"  UNSTAMPED dates     : {unstamped}")
    if drift_dates:
        print("\n  These dates were built from a different active set and "
              "MUST be re-extracted:")
        for d in drift_dates[:40]:
            print(f"    {d}")
        if len(drift_dates) > 40:
            print(f"    ... and {len(drift_dates)-40} more")
    if unstamped_dates:
        print(f"\n  Unstamped dates predate stamping, so their slot order "
              f"cannot be verified from the pool alone.")
        print(f"  Compare file mtimes against patch_index.csv, or "
              f"re-extract them to be sure.")
    if not drifted and not unstamped:
        print("\n  Every date agrees with the current patch index.")
    return 1 if (drifted or unstamped) else 0


def _resolve_index_csv(data_root, source):
    """Path to the patch-activity index CSV.

    One index serves every period. identify_patches.py has no period
    concept - it marks convective activity from OPERA alone - and each
    period's split CSVs are gated subsets of the same index, so they all
    read it. The tag is still checked, so a typo surfaces here rather
    than as a silently empty extraction.
    """
    if source == SOURCE or source.startswith(f"{SOURCE}_"):
        return os.path.join(data_root, 'patch_index', 'patch_index.csv')
    raise ValueError(
        f"Unknown source: {source!r} "
        f"(expected {SOURCE!r} or {SOURCE}_<period>)"
    )


def read_patch_index(data_root, source='dbscan'):
    """
    Read the per-source patch-activity index and return a dict
    `{(date, 'HH:MM'): [active_patch_numbers]}`.

    The index is `our_data/patch_index/patch_index.csv`, produced by
    identify_patches.py from DBSCAN clusters in OPERA rainfall_rate.
    Schema: `date,time_utc,iso_timestamp,patch_1..patch_18`.

    Returns: dict keyed by (date, 'HH:MM') with the sorted list of
    1-indexed active patch numbers as the value.
    """
    csv_path = _resolve_index_csv(data_root, source)
    if not os.path.isfile(csv_path):
        print(f"ERROR: patch_index.csv not found at {csv_path}")
        print("Run identify_patches.py first.")
        return {}

    index: dict[tuple[str, str], list[int]] = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row['date']
            time_str = row['time_utc']

            active = [
                p for p in range(1, N_PATCHES + 1)
                if row.get(f'patch_{p}', '0') == '1'
            ]
            if active:
                index[(date_str, time_str)] = active

    return index


def _load_step_minutes_from_sequence_meta(data_root, source):
    """Read step_minutes from sequence_meta_<source>.json.

    The split CSVs (train/val/test_data_<source>.csv) were produced by
    extract_patch_seq_for_datasets.py at this cadence; we need it to
    enumerate the timesteps inside each row's `[start_utc, end_utc]`
    window. Fails fast if the metadata is missing — the file must exist
    in lockstep with the split CSVs.
    """
    meta_path = os.path.join(data_root, f'sequence_meta_{source}.json')
    if not os.path.isfile(meta_path):
        print(
            f"ERROR: {meta_path} not found.\n"
            f"Run extract_patch_seq_for_datasets.py first"
            + (f" with --period {source[len(SOURCE) + 1:]}"
               if source != SOURCE else "") + ".",
            file=sys.stderr,
        )
        sys.exit(2)
    with open(meta_path) as f:
        seq = json.load(f)
    return int(seq['step_minutes'])


def load_sequence_timesteps(data_root, source):
    """Return the set of (date, 'HH:MM') timesteps needed across all
    train / validation / test sequences for the chosen source.

    Each split row carries `date, start_utc, end_utc`. The window from
    `start_utc` to `end_utc` (inclusive) at `step_minutes` spacing is
    exactly the set of timesteps the training pipeline will load for
    that sample (past + current + future). The union across all rows in
    all three splits is what extract_patches needs on disk.

    This replaces the older `patch_index ∩ timestep_manifest` flow: the
    splits already incorporate the manifest gate (Step 6 enforces it),
    so any (date, time) here is guaranteed to have survived every
    upstream filter.
    """
    step_minutes = _load_step_minutes_from_sequence_meta(data_root, source)
    step = timedelta(minutes=step_minutes)

    needed: set[tuple[str, str]] = set()
    n_rows = 0
    for split in ('train', 'validation', 'test'):
        csv_path = os.path.join(data_root, f'{split}_data_{source}.csv')
        if not os.path.isfile(csv_path):
            print(f"  WARNING: {csv_path} not found - skipping {split} split.")
            continue
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row['date'].strip()
                start_str = row['start_utc'].strip()
                end_str = row['end_utc'].strip()
                if not (date_str and start_str and end_str):
                    continue
                base = datetime.strptime(date_str, '%Y-%m-%d')
                # start/end_utc are clock times with no date of their
                # own, and `date` dates the REFERENCE. A window through
                # midnight therefore arrives with end_utc numerically
                # BEFORE start_utc; anchoring on reference_utc and
                # walking outwards resolves which side moves, and every
                # step then carries its own date.
                start_h, start_m = (int(x) for x in start_str.split(':'))
                end_h, end_m = (int(x) for x in end_str.split(':'))
                ref_str = (row.get('reference_utc') or '').strip()
                day = 24 * 60
                if ref_str:
                    rh, rm = (int(x) for x in ref_str.split(':'))
                    ref = base.replace(hour=rh, minute=rm)
                    r_min = rh * 60 + rm
                    s_min = start_h * 60 + start_m
                    e_min = end_h * 60 + end_m
                    t = ref - timedelta(minutes=(r_min - s_min) % day)
                    t_end = ref + timedelta(minutes=(e_min - r_min) % day)
                else:
                    t = base.replace(hour=start_h, minute=start_m)
                    t_end = base.replace(hour=end_h, minute=end_m)
                    if t_end < t:
                        t_end += timedelta(days=1)
                if t_end - t > timedelta(days=1):
                    continue
                while t <= t_end:
                    needed.add((t.strftime('%Y-%m-%d'), t.strftime('%H:%M')))
                    t += step
                n_rows += 1

    if needed:
        print(f"  Split CSVs:      train+val+test for --source {source}  "
              f"({n_rows} sequences -> {len(needed)} unique timesteps)")
    return needed, step_minutes


# =============================================================================
# Per-product timestamp snap (sourced from timestep_config.json)
# =============================================================================
#
# Different products have different native cadences. With a 15-min training
# step, `validate_timestep.py` writes a per-product `filter` listing exactly
# which minute marks each product is available at:
#
#   opera_rainfall_rate.filter = [0, 15, 30, 45]   ← OPERA, the patch index driver
#   mtg.filter = [0, 10, 30, 40]   ← 10-min product at 15-min step
#
# The patch index runs on OPERA's grid (:00, :15, :30, :45). When we ask
# for MTG's `vis_06` at OPERA :15, no file exists at that exact minute —
# MTG only wrote :00, :10, :30, :40. Rather than a fuzzy ±tolerance search,
# the filter IS the source of truth: we snap the requested HHMM to the
# nearest minute the product actually has, then load that file. So OPERA
# :15 -> MTG :10, OPERA :45 -> MTG :40, and OPERA :00 / :30 are exact.

_TIMESTEP_CONFIG_PATH = os.path.join(
    DEFAULT_DATA_ROOT, 'timestep_config.json',
)
_PRODUCT_FILTER_CACHE: dict[str, set[int]] = {}


def _load_product_filter(product_key: str) -> set[int] | None:
    """Return the minute filter for `product_key` from timestep_config.json.

    Cached after first read. Returns `None` if the config is missing or the
    product isn't listed there — in which case the caller falls back to an
    exact-match lookup, preserving the legacy behaviour.
    """
    if product_key in _PRODUCT_FILTER_CACHE:
        cached = _PRODUCT_FILTER_CACHE[product_key]
        return cached if cached else None
    try:
        with open(_TIMESTEP_CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        _PRODUCT_FILTER_CACHE[product_key] = set()
        return None
    flt = cfg.get('products', {}).get(product_key, {}).get('filter')
    if not flt:
        _PRODUCT_FILTER_CACHE[product_key] = set()
        return None
    s = {int(m) for m in flt}
    _PRODUCT_FILTER_CACHE[product_key] = s
    return s


# Map the PRODUCT_GROUPS keys to the timestep_config product names used to
# look up minute filters. OPERA's filter is identical across its two
# products (both at 15-min cadence), so we just point at one of them.
#
# Lightning now has a real filter in timestep_config.json
# (`products.lightning.filter = [0, 10, 30, 40]` at the typical 15-min
# step), so master :15 needs to snap to filter :10 and master :45 to
# filter :40 - the only HHMMs `read_kml_version2.py` actually writes.
# `_load_product_filter` falls back to no-snap when the filter is null,
# so this remains correct if the user reverts lightning to a continuous
# cadence in product_cadences.config.
_FILTER_PRODUCT_KEY = {
    'satellite_MTG': 'mtg',
    'lightning':     'lightning',
    'opera':         'opera_rainfall_rate',
}


def _snap_hhmm_to_filter(hhmm: str, filter_minutes: set[int]) -> str:
    """Snap `hhmm` to the nearest minute mark in `filter_minutes`.

    The minute is matched against the per-hour filter (0–59). If the nearest
    mark crosses the hour boundary (e.g. requested :58, filter has :00), the
    hour is adjusted with wrap-around at midnight. Ties prefer the earlier
    minute so OPERA :15 snaps deterministically to MTG :10 rather than :20
    (when MTG happens to have both).
    """
    h, m = int(hhmm[:2]), int(hhmm[2:])
    best = min(filter_minutes, key=lambda fm: (
        min(abs(fm - m), 60 - abs(fm - m)),  # primary: circular distance
        fm,                                  # tiebreaker: earlier minute wins
    ))
    diff = best - m
    if diff > 30:
        diff -= 60   # snapped forward across the hour — really one hour back
    elif diff < -30:
        diff += 60   # snapped backward — one hour forward
    total = (h * 60 + m + diff) % (24 * 60)
    return f"{total // 60:02d}{total % 60:02d}"


def _resolve_hhmm(hhmm: str, group: str) -> str:
    """Snap a requested HHMM to the group's available cadence grid."""
    product_key = _FILTER_PRODUCT_KEY.get(group)
    if product_key is None:
        return hhmm
    flt = _load_product_filter(product_key)
    if not flt:
        return hhmm
    return _snap_hhmm_to_filter(hhmm, flt)


# =============================================================================
# File discovery per product
# =============================================================================



def find_reprojected_file_satellite(data_root, instrument, channel,
                                   date_str, time_str):
    """
    Find a reprojected satellite .npy file.

    Path: reprojected_data/satellite_data/MTG/{channel}/
          nc4_{date}-Romania_{channel}/nc4_{date}-Romania_{HHMM}_{channel}.npy

    HHMM is snapped to the instrument's minute filter — MTG at 10-min
    cadence have {00, 10, 30, 40}; OPERA at 15-min has {00, 15, 30, 45};
    snapping resolves the mismatch when these are mixed in one sample.
    """
    group = f"satellite_{instrument}"
    hhmm = _resolve_hhmm(time_str.replace(':', ''), group)
    day_folder = f"nc4_{date_str}-Romania_{channel}"
    filename = f"nc4_{date_str}-Romania_{hhmm}_{channel}.npy"
    path = os.path.join(
        data_root, 'reprojected_data', 'satellite_data', instrument,
        channel, day_folder, filename
    )
    return path if array_exists(path) else None


def find_reprojected_file_lightning(data_root, product, date_str, time_str):
    """
    Find a lightning .npy file on disk.

    `read_kml_version2.py` writes lightning maps directly onto the
    Romania grid via `GridProjection`, so they live at
        `lightning_data/{product}/nc4_<date>-Romania_<product>/lightning_<product>_<YYYYMMDD>_<HHMM>.npy`
    (no `reprojected_data/` prefix). The legacy
    `reproject.py --lightning` flow used to mirror them into
    `reprojected_data/lightning_data/...`; we try that location first
    for backward compatibility, then fall back to the canonical native
    path.
    """
    hhmm = _resolve_hhmm(time_str.replace(':', ''), 'lightning')
    date_compact = date_str.replace('-', '')
    day_folder = f"nc4_{date_str}-Romania_{product}"
    filename = f"lightning_{product}_{date_compact}_{hhmm}.npy"

    # 1. Legacy mirrored location under reprojected_data/.
    legacy_path = os.path.join(
        data_root, 'reprojected_data', 'lightning_data',
        product, day_folder, filename,
    )
    if array_exists(legacy_path):
        return legacy_path

    # 2. Canonical path: read_kml_version2.py writes here directly.
    native_path = os.path.join(
        data_root, 'lightning_data',
        product, day_folder, filename,
    )
    return native_path if array_exists(native_path) else None




def find_reprojected_file_opera(data_root, variable, date_str, time_str):
    """
    Find a reprojected OPERA .npy file. The on-disk folder uses the short
    product name (reflectivity / rainfall_rate) while the variable name
    passed in by extract_patches uses the `opera_` prefix.

    Path: reprojected_data/opera_data/{short}/nc4_{date}-Romania_{short}/
          nc4_{date}-Romania_{HHMM}_{short}.npy

    The patch index is on OPERA's grid so the snap is a no-op here, but
    it's still applied for consistency.
    """
    short = OPERA_VAR_TO_DISK.get(variable, variable)
    hhmm = _resolve_hhmm(time_str.replace(':', ''), 'opera')
    day_folder = f"nc4_{date_str}-Romania_{short}"
    filename = f"nc4_{date_str}-Romania_{hhmm}_{short}.npy"
    path = os.path.join(
        data_root, 'reprojected_data', 'opera_data',
        short, day_folder, filename
    )
    return path if array_exists(path) else None


def find_reprojected_file(data_root, variable, group, date_str, time_str):
    """
    Dispatch to the correct file finder based on product group.

    Returns:
        str or None: path to the reprojected file, or None if not found
    """
    if group == 'satellite_MTG':
        return find_reprojected_file_satellite(
            data_root, 'MTG', variable, date_str, time_str
        )
    elif group == 'lightning':
        return find_reprojected_file_lightning(
            data_root, variable, date_str, time_str
        )
    elif group == 'opera':
        return find_reprojected_file_opera(
            data_root, variable, date_str, time_str
        )
    return None


# =============================================================================
# Data loading
# =============================================================================

def load_reprojected(filepath, variable=None, group=None):
    """
    Load a reprojected `.npy` file. Every product family writes `.npy`,
    so `variable` and `group` are no longer used here.

    Returns:
        np.ndarray: 2D array (768×1536) as float32.
    """
    if not filepath.endswith(('.npy', '.npy.zst')):
        raise ValueError(f"Unknown file format: {filepath}")
    data = load_array(filepath)
    return np.asarray(data, dtype=np.float32)


# =============================================================================
# Patch extraction
# =============================================================================

def extract_and_pool(data, active_patches, pool_factor):
    """
    Extract active patches from a full grid and apply pooling.

    Args:
        data: 2D array (768×1536) — the full reprojected field
        active_patches: sorted list of 1-indexed patch numbers
        pool_factor: 1 (no pooling), 2 (2×2), or 4 (4×4)

    Returns:
        np.ndarray: shape (num_patches, patch_h, patch_w)
            patch_h = patch_w = 256 / pool_factor
    """
    out_size = PATCH_SIZE // pool_factor
    patches = np.zeros(
        (len(active_patches), out_size, out_size), dtype=np.float32
    )

    for i, p in enumerate(active_patches):
        r0, r1, c0, c1 = get_patch_bounds(p)
        patch = data[r0:r1, c0:c1]
        patches[i] = average_pool(patch, pool_factor)

    return patches


# =============================================================================
# Main pipeline
# =============================================================================

def run_extraction(data_root, output_root, source='dbscan',
                   date_filter=None, product_filter=None):
    """
    Run the patch extraction pipeline.

    Step 6 (`extract_patch_seq_for_datasets.py`) already produced the
    train / validation / test_data_<source>.csv files with the manifest
    gate enforced - those CSVs are the authoritative list of (date, time)
    pairs the training pipeline will load. We walk the union of the
    three splits, look up the active patches at each timestep from the
    per-source patch-activity index, and slice + save. No separate
    manifest read is needed.

    Args:
        data_root:      path to our_data directory
        output_root:    path to output patches directory
        source:         sample-selection source (always
                        pipeline_config.SOURCE). Picks both the
                        activity-index file and the split CSV suffix.
        date_filter:    optional YYYY-MM-DD to restrict processing
        product_filter: optional list of group names to process
    """
    print("=" * 70)
    print("COALITION-4 Patch Extraction Pipeline")
    print("=" * 70)
    print(f"Data root   : {data_root}")
    print(f"Output root : {output_root}")
    print(f"Source      : {source}")

    # Per-timestep active-patches lookup. The same index file the split
    # CSVs were built from, so the saved-patch ORDER matches the idx_t*
    # column values in those CSVs.
    index = read_patch_index(data_root, source=source)
    if not index:
        return

    # Union of (date, time) pairs across train / val / test_<source>.csv.
    # Each split row's [start_utc, end_utc] window at step_minutes
    # spacing contributes past+1+future timesteps.
    needed_timesteps, _step_minutes = load_sequence_timesteps(
        data_root, source
    )
    if not needed_timesteps:
        print(f"No train/val/test_{source}.csv rows found at "
              f"{data_root}. Run extract_patch_seq_for_datasets.py "
              f"--source {source} first.")
        return

    # Filter the index to (date, time) the split CSVs actually need.
    kept_keys = sorted(set(index.keys()) & needed_timesteps)
    if date_filter:
        kept_keys = [(d, t) for (d, t) in kept_keys if d == date_filter]
        print(f"Date filter : {date_filter}")

    if not kept_keys:
        print("No matching timestamps. The split CSVs reference no "
              "(date, time) pairs that exist in the patch index - "
              "did you run Steps 4-6 for this source?")
        return

    index_rows = [(d, t, index[(d, t)]) for d, t in kept_keys]
    dates = sorted({d for d, _, _ in index_rows})
    print(f"Timestamps  : {len(index_rows)} across {len(dates)} dates "
          f"(union of train+val+test for --source {source})")

    # Rebind for the rest of the loop: the older code below iterates
    # `index` as a list of (date, time, active_patches) tuples.
    index = index_rows

    # Determine which product groups to process
    if product_filter:
        groups = {k: v for k, v in PRODUCT_GROUPS.items() if k in product_filter}
    else:
        groups = PRODUCT_GROUPS

    # Collect all variables
    all_vars = {}
    for group_name, products in groups.items():
        for var_name, (group, res_tag, pool_factor) in products.items():
            all_vars[var_name] = (group, res_tag, pool_factor)

    print(f"Products    : {len(all_vars)} variables across "
          f"{len(groups)} groups")
    print()

    # Process each timestamp
    total_files_saved = 0
    total_files_missing = 0

    # Provenance for the stamps written below, and the running record of
    # what each date was actually built from.
    index_digest = _index_digest(data_root, source)
    stamps_seen: dict[str, dict] = {}
    stamp_updates: dict[str, dict[str, list[int]]] = {}
    n_stale_refreshed = 0
    stale_dates: set[str] = set()

    for idx, (date_str, time_str, active_patches) in enumerate(index):
        hhmm = time_str.replace(':', '')
        out_dir = os.path.join(output_root, date_str)
        os.makedirs(out_dir, exist_ok=True)

        # Has this timestep's active set changed since its files were
        # written? If so the slot order in them is wrong, and skipping
        # would keep it wrong forever.
        if date_str not in stamps_seen:
            stamps_seen[date_str] = _read_stamp(out_dir) or {}
        prior = stamps_seen[date_str].get(hhmm)
        stale = prior is not None and list(prior) != list(active_patches)
        if stale:
            n_stale_refreshed += 1
            stale_dates.add(date_str)
        stamp_updates.setdefault(date_str, {})[hhmm] = list(active_patches)

        n_saved = 0
        n_missing = 0

        for var_name, (group, res_tag, pool_factor) in all_vars.items():
            # Build output path
            out_filename = f"{var_name}_{hhmm}_{res_tag}.npy"
            out_path = os.path.join(out_dir, out_filename)

            # Skip if already extracted, in either form on disk - unless
            # the active set moved under it, in which case the cached
            # file's slot order is wrong and must be overwritten.
            if array_exists(out_path) and not stale:
                n_saved += 1
                continue

            # Find source file
            filepath = find_reprojected_file(
                data_root, var_name, group, date_str, time_str
            )

            if filepath is None:
                n_missing += 1
                continue

            try:
                # Load the full 768×1536 grid
                data = load_reprojected(filepath, variable=var_name, group=group)

                # Extract patches and apply pooling
                patches = extract_and_pool(data, active_patches, pool_factor)

                # Save. Overwriting a stale entry must also drop its
                # compressed twin, or load_array would keep resolving to
                # the old tiles the moment the fresh .npy is compressed
                # away again.
                if stale:
                    twin = out_path + '.zst'
                    if os.path.isfile(twin):
                        os.remove(twin)
                np.save(out_path, patches)
                n_saved += 1

            except Exception as e:
                print(f"  ERROR {var_name} @ {date_str} {time_str}: {e}")
                n_missing += 1

        total_files_saved += n_saved
        total_files_missing += n_missing

        patches_str = ','.join(str(p) for p in active_patches)
        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(index) - 1:
            print(f"  [{idx+1}/{len(index)}] {date_str} {time_str} "
                  f"patches=[{patches_str}] -> {n_saved} saved, "
                  f"{n_missing} missing")

    # Stamp each date with the active sets its files were built from, so
    # a later identify_patches run can tell what moved. Merged with any
    # existing stamp: a --date filtered run must not erase the rest.
    for date_str, active_by_hhmm in stamp_updates.items():
        out_dir = os.path.join(output_root, date_str)
        merged = dict(stamps_seen.get(date_str) or {})
        merged.update(active_by_hhmm)
        _write_stamp(out_dir, merged, index_digest)

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"  Timestamps processed : {len(index)}")
    print(f"  Files saved/cached   : {total_files_saved}")
    print(f"  Files missing        : {total_files_missing}")
    if n_stale_refreshed:
        print(f"  STALE re-extracted   : {n_stale_refreshed} timestep(s) "
              f"across {len(stale_dates)} date(s)")
        print(f"                         (their active set changed since "
              f"the files were written)")
    print(f"  Dates stamped        : {len(stamp_updates)}")
    print(f"  Output directory     : {output_root}")

    # Print resolution summary
    print(f"\n  Resolution mapping:")
    for var_name, (group, res_tag, pool_factor) in sorted(all_vars.items()):
        out_size = PATCH_SIZE // pool_factor
        print(f"    {var_name:<14} {group:<16} {res_tag}  "
              f"{out_size}x{out_size}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COALITION-4 patch extraction pipeline. "
                    "Extracts 256x256 patches from reprojected data based on "
                    "the patch index, with resolution-dependent pooling."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="Output directory (default: our_data/patches)"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Process a single date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--products", nargs='+',
        choices=list(PRODUCT_GROUPS),
        default=None,
        help="Product groups to extract (default: all)"
    )
    parser.add_argument(
        "--period", type=str, default=None, metavar="LABEL",
        help="Slice patches for a period-suffixed split set, e.g. "
             f"--period w48 reads {{train,validation,test}}_data_{SOURCE}_w48"
             ".csv. Omit for the unsuffixed whole-archive splits. Lets "
             "split sets built under different gates coexist: the patch "
             "index is shared, but each period's splits select their own "
             "subset of it.",
    )

    parser.add_argument(
        "--audit_pool", action="store_true",
        help="Report which dates in our_data/patches/ were built from a "
             "different active-patch set than the current patch index, "
             "then exit. Extracts nothing. A drifted date silently feeds "
             "the wrong tile into training, so check this after every "
             "identify_patches re-run.",
    )

    args = parser.parse_args()

    output_root = args.output_dir or os.path.join(args.data_root, 'patches')

    # The split CSVs, the sequence metadata and the saved patches are all
    # keyed by this tag, so it has to be assembled once and threaded
    # through rather than re-derived at each call site.
    source = f"{SOURCE}_{args.period}" if args.period else SOURCE

    if args.audit_pool:
        sys.exit(audit_pool(args.data_root, source))

    run_extraction(
        data_root=args.data_root,
        output_root=output_root,
        source=source,
        date_filter=args.date,
        product_filter=args.products,
    )
    