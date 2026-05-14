"""
COALITION-4 patch extraction pipeline.

Reads the patch index (from identify_patches.py) and the cached reprojected
data (from reproject_data.py), extracts the active 256x256 patches for every
product, applies resolution-dependent pooling, and saves stacked .npy files.

Resolution categories:
    HR (1km)  → no pooling   → 256×256  (radar, lightning, MTG vis_06)
    LR (2km)  → 2×2 avg pool → 128×128  (MTG IR/WV channels)
    LR (3km)  → 4×4 avg pool →  64×64   (MSG, NWCSAF)

Output:
    our_data/patches/{date}/{variable}_{HHMM}_{HR|LR}.npy
    Each file has shape (num_active_patches, H, W).
    Patch order matches the active patches from patch_index.csv for that
    timestamp (sorted ascending).

Usage (run from F:\\nowcasting\\coalition4-rcnn):
    python extract_patches.py
    python extract_patches.py --date 2025-05-15
    python extract_patches.py --date 2025-05-15 --products radar lightning
"""

import numpy as np
import os
import re
import csv
import json
import argparse
from pathlib import Path


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

RADAR_PRODUCTS = {
    'RZC':    ('radar', 'HR', 1),
    'BZC':    ('radar', 'HR', 1),
    'CZC':    ('radar', 'HR', 1),
    'EZC-20': ('radar', 'HR', 1),
    'LZC':    ('radar', 'HR', 1),
    'CPCH':   ('radar', 'HR', 1),
}

MSG_PRODUCTS = {
    'VIS006': ('satellite_MSG', 'LR', 4),
    'IR_039': ('satellite_MSG', 'LR', 4),
    'IR_108': ('satellite_MSG', 'LR', 4),
    'WV_062': ('satellite_MSG', 'LR', 4),
    'WV_073': ('satellite_MSG', 'LR', 4),
}

MTG_PRODUCTS = {
    'vis_06': ('satellite_MTG', 'HR', 1),
    'ir_38':  ('satellite_MTG', 'LR', 2),
    'ir_105': ('satellite_MTG', 'LR', 2),
    'wv_63':  ('satellite_MTG', 'LR', 2),
    'wv_73':  ('satellite_MTG', 'LR', 2),
}

LIGHTNING_PRODUCTS = {
    'density':    ('lightning', 'HR', 1),
    'current':    ('lightning', 'HR', 1),
    'occurrence': ('lightning', 'HR', 1),
}

NWCSAF_PRODUCTS = {
    'ctth_alti':  ('nwcsaf', 'LR', 4),
    'ctth_tempe': ('nwcsaf', 'LR', 4),
    'cmic_phase': ('nwcsaf', 'LR', 4),
    'cmic_cot':   ('nwcsaf', 'LR', 4),
}

# OPERA: max reflectivity (dBZ) + rainfall_rate (mm/h), 2 km native → 2× pool.
# `opera_rainfall_rate_hr` is an alias of `opera_rainfall_rate` extracted at
# HR (no pooling, 256×256) so it can be used as the multi-class label target
# in OPERA-driven modes. Both aliases point to the same source `.npy`.
OPERA_PRODUCTS = {
    'opera_reflectivity':     ('opera', 'LR', 2),
    'opera_rainfall_rate':    ('opera', 'LR', 2),
    'opera_rainfall_rate_hr': ('opera', 'HR', 1),
}

# Group name → CLI flag mapping
PRODUCT_GROUPS = {
    'radar':         RADAR_PRODUCTS,
    'satellite_MSG': MSG_PRODUCTS,
    'satellite_MTG': MTG_PRODUCTS,
    'lightning':     LIGHTNING_PRODUCTS,
    'nwcsaf':        NWCSAF_PRODUCTS,
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

def read_patch_index(data_root):
    """
    Read patch_index.csv and return a list of (date, time, active_patches).

    Args:
        data_root: path to our_data directory

    Returns:
        list[tuple]: (date_str, time_str, list[int]) for each active row
    """
    csv_path = os.path.join(data_root, 'patch_index', 'patch_index.csv')
    if not os.path.isfile(csv_path):
        print(f"ERROR: patch_index.csv not found at {csv_path}")
        print("Run identify_patches.py first.")
        return []

    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row['date']
            time_str = row['time_utc']

            active = []
            for p in range(1, N_PATCHES + 1):
                if row.get(f'patch_{p}', '0') == '1':
                    active.append(p)

            if active:
                results.append((date_str, time_str, active))

    return results


def load_manifest_timesteps(manifest_path):
    """
    Read the timestep manifest produced by `intersect_product_coverage.py`
    and return the set of `(date, 'HH:MM')` tuples that survived the
    per-timestep intersection.

    The manifest columns are `date,hhmm,<key>_hhmm,...`. We only need the
    first two — the per-product columns are an audit trail consumed by
    other tooling. Returns an empty set if the file does not exist, in
    which case the caller falls back to processing every entry in the
    patch index.
    """
    if not manifest_path or not os.path.isfile(manifest_path):
        return set()
    used: set[tuple[str, str]] = set()
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'date' not in reader.fieldnames \
                or 'hhmm' not in reader.fieldnames:
            return set()
        for row in reader:
            date_str = row['date'].strip()
            hhmm = row['hhmm'].strip()
            if not date_str or len(hhmm) != 4 or not hhmm.isdigit():
                continue
            used.add((date_str, f"{hhmm[:2]}:{hhmm[2:]}"))
    if used:
        print(f"  Manifest:        {manifest_path}  "
              f"({len(used)} timesteps to extract)")
    return used


# =============================================================================
# Per-product timestamp snap (sourced from timestep_config.json)
# =============================================================================
#
# Different products have different native cadences. With a 15-min training
# step, `validate_timestep.py` writes a per-product `filter` listing exactly
# which minute marks each product is available at:
#
#   opera_rainfall_rate.filter = [0, 15, 30, 45]   ← OPERA, the patch index driver
#   mtg.filter / nwcsaf.filter = [0, 10, 30, 40]   ← 10-min products at 15-min step
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
_FILTER_PRODUCT_KEY = {
    'radar':         'radar',
    'satellite_MSG': None,        # legacy / not in current config — skip snap
    'satellite_MTG': 'mtg',
    'lightning':     None,        # cadence_minutes is null — skip snap
    'nwcsaf':        'nwcsaf',
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

def find_reprojected_file_radar(data_root, variable, date_str, time_str):
    """
    Find a reprojected radar .npy file.

    Path: reprojected_data/radar_data/{var}/nc4_{date}-Romania_{var}/
          nc4_{date}-Romania_{HHMM}_{var}.npy

    HHMM is snapped to the radar minute filter from timestep_config.json
    so a request at e.g. :15 maps to the nearest available :10.
    """
    hhmm = _resolve_hhmm(time_str.replace(':', ''), 'radar')
    day_folder = f"nc4_{date_str}-Romania_{variable}"
    filename = f"nc4_{date_str}-Romania_{hhmm}_{variable}.npy"
    path = os.path.join(
        data_root, 'reprojected_data', 'radar_data',
        variable, day_folder, filename
    )
    return path if os.path.isfile(path) else None


def find_reprojected_file_satellite(data_root, instrument, channel,
                                   date_str, time_str):
    """
    Find a reprojected satellite .npy file.

    Path: reprojected_data/satellite_data/{MSG|MTG}/{channel}/
          nc4_{date}-Romania_{channel}/nc4_{date}-Romania_{HHMM}_{channel}.npy

    HHMM is snapped to the instrument's minute filter — MTG/MSG at 10-min
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
    return path if os.path.isfile(path) else None


def find_reprojected_file_lightning(data_root, product, date_str, time_str):
    """
    Find a reprojected lightning .npy file.

    Path: reprojected_data/lightning_data/{product}/nc4_{date}-Romania_{product}/
          lightning_{product}_{YYYYMMDD}_{HHMM}.npy

    Lightning has no minute filter in timestep_config.json (cadence_minutes
    is null), so the HHMM is used exactly as supplied.
    """
    hhmm = _resolve_hhmm(time_str.replace(':', ''), 'lightning')
    date_compact = date_str.replace('-', '')
    day_folder = f"nc4_{date_str}-Romania_{product}"
    filename = f"lightning_{product}_{date_compact}_{hhmm}.npy"
    path = os.path.join(
        data_root, 'reprojected_data', 'lightning_data',
        product, day_folder, filename
    )
    return path if os.path.isfile(path) else None


def find_reprojected_file_nwcsaf(data_root, variable, date_str, time_str):
    """
    Find a reprojected NWCSAF `.npy` file. After the unification in reproject.py,
    each NWCSAF variable is stored as its own per-variable `.npy` mirroring
    the radar / MTG layout — no more multi-variable `.nc` files.

    Path: reprojected_data/nwcsaf_data/{variable}/nc4_{date}-Romania_{variable}/
          nc4_{date}-Romania_{HHMM}_{variable}.npy

    HHMM is snapped to the NWCSAF minute filter (same shape as MTG's).
    """
    hhmm = _resolve_hhmm(time_str.replace(':', ''), 'nwcsaf')
    day_folder = f"nc4_{date_str}-Romania_{variable}"
    filename = f"nc4_{date_str}-Romania_{hhmm}_{variable}.npy"
    path = os.path.join(
        data_root, 'reprojected_data', 'nwcsaf_data', variable,
        day_folder, filename,
    )
    return path if os.path.isfile(path) else None


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
    return path if os.path.isfile(path) else None


def find_reprojected_file(data_root, variable, group, date_str, time_str):
    """
    Dispatch to the correct file finder based on product group.

    Returns:
        str or None: path to the reprojected file, or None if not found
    """
    if group == 'radar':
        return find_reprojected_file_radar(data_root, variable, date_str, time_str)
    elif group == 'satellite_MSG':
        return find_reprojected_file_satellite(
            data_root, 'MSG', variable, date_str, time_str
        )
    elif group == 'satellite_MTG':
        return find_reprojected_file_satellite(
            data_root, 'MTG', variable, date_str, time_str
        )
    elif group == 'lightning':
        return find_reprojected_file_lightning(
            data_root, variable, date_str, time_str
        )
    elif group == 'nwcsaf':
        return find_reprojected_file_nwcsaf(
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
    Load a reprojected `.npy` file. All product families now write `.npy`;
    multi-variable NWCSAF `.nc` files were removed when reproject.py was
    unified, so `variable` and `group` are no longer used here.

    Returns:
        np.ndarray: 2D array (768×1536) as float32.
    """
    if not filepath.endswith('.npy'):
        raise ValueError(f"Unknown file format: {filepath}")
    data = np.load(filepath)
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

def run_extraction(data_root, output_root, date_filter=None,
                   product_filter=None, manifest=None):
    """
    Run the patch extraction pipeline.

    Args:
        data_root: path to our_data directory
        output_root: path to output patches directory
        date_filter: optional YYYY-MM-DD
        product_filter: optional list of group names to process
        manifest: optional path to `timestep_manifest.csv` (the output
            of `intersect_product_coverage.py`). When supplied (or
            auto-discovered at `<data_root>/timestep_manifest.csv`),
            only timesteps listed in the manifest are processed.
            Pass an empty string or `'none'` to disable and process
            every active timestep in `patch_index.csv`.
    """
    print("=" * 70)
    print("COALITION-4 Patch Extraction Pipeline")
    print("=" * 70)
    print(f"Data root   : {data_root}")
    print(f"Output root : {output_root}")

    # Read patch index — this gives us the per-timestep set of active
    # patches (needed regardless of whether the manifest filter is on).
    index = read_patch_index(data_root)
    if not index:
        return

    # Constrain to (date, time) pairs in the intersect manifest. Skipping
    # un-listed timesteps avoids work on slots where one or more required
    # products had no data.
    if manifest is None:
        manifest = os.path.join(data_root, 'timestep_manifest.csv')
    used_timesteps: set[tuple[str, str]] = set()
    if manifest:
        used_timesteps = load_manifest_timesteps(manifest)
    if used_timesteps:
        before = len(index)
        index = [(d, t, a) for d, t, a in index
                 if (d, t) in used_timesteps]
        print(f"Manifest filter: kept {len(index)}/{before} timesteps")

    if date_filter:
        index = [(d, t, a) for d, t, a in index if d == date_filter]
        print(f"Date filter : {date_filter}")

    if not index:
        print("No matching timestamps in patch index.")
        return

    dates = sorted(set(d for d, _, _ in index))
    print(f"Timestamps  : {len(index)} across {len(dates)} dates")

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

    for idx, (date_str, time_str, active_patches) in enumerate(index):
        hhmm = time_str.replace(':', '')
        out_dir = os.path.join(output_root, date_str)
        os.makedirs(out_dir, exist_ok=True)

        n_saved = 0
        n_missing = 0

        for var_name, (group, res_tag, pool_factor) in all_vars.items():
            # Build output path
            out_filename = f"{var_name}_{hhmm}_{res_tag}.npy"
            out_path = os.path.join(out_dir, out_filename)

            # Skip if already extracted
            if os.path.isfile(out_path):
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

                # Save
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
                  f"patches=[{patches_str}] → {n_saved} saved, "
                  f"{n_missing} missing")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"  Timestamps processed : {len(index)}")
    print(f"  Files saved/cached   : {total_files_saved}")
    print(f"  Files missing        : {total_files_missing}")
    print(f"  Output directory     : {output_root}")

    # Print resolution summary
    print(f"\n  Resolution mapping:")
    for var_name, (group, res_tag, pool_factor) in sorted(all_vars.items()):
        out_size = PATCH_SIZE // pool_factor
        print(f"    {var_name:<14} {group:<16} {res_tag}  "
              f"{out_size}×{out_size}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COALITION-4 patch extraction pipeline. "
                    "Extracts 256×256 patches from reprojected data based on "
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
        choices=['radar', 'satellite_MSG', 'satellite_MTG',
                 'lightning', 'nwcsaf', 'opera'],
        default=None,
        help="Product groups to extract (default: all)"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to the timestep manifest produced by "
             "intersect_product_coverage.py. Default: "
             "<data_root>/timestep_manifest.csv if it exists. Pass "
             "`--manifest none` to skip the filter and process every "
             "active timestep in patch_index.csv."
    )

    args = parser.parse_args()

    output_root = args.output_dir or os.path.join(args.data_root, 'patches')

    # `--manifest none` explicitly opts out; otherwise auto-discover.
    if args.manifest is not None:
        manifest = '' if args.manifest.lower() == 'none' else args.manifest
    else:
        manifest = None  # let run_extraction auto-discover

    run_extraction(
        data_root=args.data_root,
        output_root=output_root,
        date_filter=args.date,
        product_filter=args.products,
        manifest=manifest,
    )
    