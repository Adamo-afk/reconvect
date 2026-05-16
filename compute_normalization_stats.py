"""
compute_normalization_stats.py — Derive per-variable normalization parameters
from the reprojected data in `our_data/reprojected_data/`.

This script is Step 4.3, between extract_patch_seq_for_datasets.py (4.1)
and create_datasets.py (Step 5). It produces
`our_data/normalization_stats_<source>.json` for whichever `--source`
the run targets (dbscan / lightning), which `create_datasets.py
--source <source>` then consumes — there is **no fallback to the
Leinonen Swiss constants**, so this script must be run successfully
before TF datasets can be built. Each source has its own training
distribution, so each gets its own stats file; the two never share.

The training CSV (`train_data_<source>.csv`) from Step 4.1 is the
authoritative source for *which sequences* the model will see at
training time. For each row, the past + current + future timesteps
(read from `sequence_meta_<source>.json`) are expanded into the set of
(date, HHMM) tuples consumed at training time, and only those
contribute to the stats. The cross-product manifest gate has already
been enforced upstream by extract_patch_seq, so no separate manifest
read is needed here.

Policy decisions (also recorded inside the JSON for traceability)
---------------------------------------------------------------

1. **Training set only.** Statistics are computed exclusively from
   timesteps that appear inside *training-eligible windows*. For each
   row of `train_data_<source>.csv`, the past + current + future
   timesteps (read from `sequence_meta_<source>.json`) are expanded
   into the set of (date, HHMM) tuples actually consumed at training
   time; every other timestep (validation, test, or off-grid) is
   excluded. Pass `--no_split_filter` to disable this and use every
   reprojected file.

2. **Single scalar mean/std per variable.** Across all valid pixels of
   all training timesteps, no per-pixel climatology. Per-pixel stats
   would let the model overfit to the training domain's geography
   (e.g. permanent radar beam blockage); a single scalar avoids that.

3. **Source: reprojected data, not patches.** Patches under
   `our_data/patches/` are filtered by RZC-based DBSCAN activity (for
   radar mode) or by lightning activity (for lightning mode). Computing
   stats on them would bias every input variable's distribution toward
   convective scenes. Reading directly from
   `our_data/reprojected_data/.../1536x768.npy` uses the **full Romania
   grid** at each timestep, removing that bias.

4. **Missing-pixel handling.** Pixels that are NaN, or that match a
   per-variable "missing sentinel" (e.g. NWCSAF 65535, OPERA nodata),
   are **dropped before Welford accumulation**. Replacing them with the
   variable's fill value (as the live `create_datasets.py` transforms
   still do at inference) would have silently dragged the mean toward
   that fill value. After the JSON is consumed downstream, the same
   physical fill is still applied to keep tensor shapes consistent;
   the *stats* themselves now reflect signal only.

5. **Domain-informed pre-norm before z-scoring.** Heavy-tailed,
   zero-inflated variables (rain rate, lightning density/current,
   cmic_cot, opera_rainfall_rate) are clipped to a positive floor and
   then `log10`-transformed before mean/std are taken. Linear-scale
   variables (radar reflectivity composites, satellite brightness
   temperatures, cloud-top altitude/temperature, opera_reflectivity)
   are left in physical units. Simple physical scaling (`x/100` for
   BZC/VIS, `x/1.97` for EZC) and categorical/binary variables
   (occurrence, cmic_phase) are not driven by this script.

6. **Near-constant detection.** A variable whose `std < eps*|mean|`
   (default `eps=1e-3`) is flagged `near_constant=true` in the JSON.
   Standardisation amplifies noise in the rare non-zero pixels of
   such variables; the downstream consumer should consider clipping
   or robust scaling instead of z-scoring.

Output schema
-------------
See `_OUTPUT_SCHEMA_NOTES` in the source for the full JSON structure.

Usage
-----
    python compute_normalization_stats.py
    python compute_normalization_stats.py --variables RZC ir_105 cmic_cot
    python compute_normalization_stats.py --no_split_filter
    python compute_normalization_stats.py --sample_fraction 0.1
    python compute_normalization_stats.py --with_percentiles
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"
DEFAULT_REGRID_ROOT = DEFAULT_DATA_ROOT / "reprojected_data"
DEFAULT_TIMESTEP_CONFIG = DEFAULT_DATA_ROOT / "timestep_config.json"
# The training CSV / sequence meta / output are suffixed by the
# `--source` flag at CLI time: train_data_<source>.csv,
# sequence_meta_<source>.json, normalization_stats_<source>.json. The
# constants below are only used when the user passes `--no_split_filter`
# (no training-window scope; ignored otherwise).

# Map a NORMALIZATION_SPEC source name to the matching product key in
# timestep_config.json. OPERA has two entries in the config (reflectivity /
# rainfall_rate) but they share the same filter, so either one works.
_SOURCE_TO_TSCONFIG_PRODUCT = {
    "radar":     "radar",
    "mtg":       "mtg",
    "nwcsaf":    "nwcsaf",
    "opera":     "opera_rainfall_rate",
    "lightning": "lightning",
}

# Per-variable normalization spec.
#
# Each entry declares:
#   source         : which reprojected sub-tree to look in
#                    (radar | mtg | lightning | nwcsaf | opera)
#   transform      : 'log_zscore' or 'linear'
#   fill           : value used to fill NaN at inference time (NOT used by
#                    the stats accumulator — see policy #4)
#   clip_min       : (log_zscore only) floor applied before log10 at
#                    inference time
#   missing_above  : (optional) sentinel — pixels strictly greater than this
#                    are dropped from the stats accumulator AND replaced with
#                    `fill` at inference time
#
# Variables not listed here either use simple physical scaling (BZC, VIS,
# EZC = x/k) or are categorical/binary (occurrence, cmic_phase).
NORMALIZATION_SPEC: dict[str, dict] = {
    # --- Radar composites (Swiss MeteoSwiss conventions) ---
    "RZC":      {"source": "radar",     "transform": "log_zscore",
                 "fill": 0.01,  "clip_min": 0.01},
    "LZC":      {"source": "radar",     "transform": "log_zscore",
                 "fill": 0.5,   "clip_min": 0.5},
    "CZC":      {"source": "radar",     "transform": "linear",
                 "fill": -5.0},
    # --- MTG FCI L1C brightness temperatures ---
    "ir_38":    {"source": "mtg",       "transform": "linear",
                 "fill": 274.0},
    "ir_105":   {"source": "mtg",       "transform": "linear",
                 "fill": 250.0},
    "wv_63":    {"source": "mtg",       "transform": "linear",
                 "fill": 250.0},
    "wv_73":    {"source": "mtg",       "transform": "linear",
                 "fill": 250.0},
    # --- Lightning aggregated rasters ---
    "density":  {"source": "lightning", "transform": "log_zscore",
                 "fill": 1e-4,  "clip_min": 1e-4},
    "current":  {"source": "lightning", "transform": "log_zscore",
                 "fill": 1e-8,  "clip_min": 1e-8},
    # --- NWCSAF L2 cloud products (variables live INSIDE CTTH / CMIC files) ---
    "ctth_alti":  {"source": "nwcsaf",  "transform": "linear",
                   "fill": -1000.0,  "missing_above": 60000},
    "ctth_tempe": {"source": "nwcsaf",  "transform": "linear",
                   "fill": 330.0,    "missing_above": 60000},
    "cmic_cot":   {"source": "nwcsaf",  "transform": "log_zscore",
                   "fill": 0.1, "clip_min": 0.1, "missing_above": 60000},
    # --- OPERA radar composites ---
    "opera_reflectivity":  {"source": "opera", "transform": "linear",
                            "fill": -32.0},
    "opera_rainfall_rate": {"source": "opera", "transform": "log_zscore",
                            "fill": 0.01, "clip_min": 0.01},
}

# Default near-constant threshold — flagged in the JSON when triggered.
NEAR_CONSTANT_EPS = 1e-3


# =============================================================================
# Welford's running statistics
# =============================================================================

class WelfordStats:
    """
    Chunked online mean / variance / min / max using Welford's parallel
    algorithm. NaN and externally-masked pixels are filtered by the caller
    *before* calling update_chunk so this class never sees them.
    """

    def __init__(self, reservoir_size: int = 0, rng: random.Random | None = None):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float('inf')
        self.max = float('-inf')
        self.reservoir_size = reservoir_size
        self.reservoir: list[float] = []
        self._seen = 0
        self._rng = rng or random.Random(0)

    def update_chunk(self, flat: np.ndarray) -> None:
        flat = np.asarray(flat).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return

        m = flat.size
        chunk_sum = float(flat.sum())
        chunk_mean = chunk_sum / m
        chunk_M2 = float(((flat - chunk_mean) ** 2).sum())
        cmin = float(flat.min())
        cmax = float(flat.max())

        self.update_from_partial(
            n=m,
            chunk_sum=chunk_sum,
            chunk_M2=chunk_M2,
            chunk_min=cmin,
            chunk_max=cmax,
        )
        if self.reservoir_size > 0:
            self._update_reservoir(flat)

    def update_from_partial(self, *,
                            n: int,
                            chunk_sum: float,
                            chunk_M2: float,
                            chunk_min: float,
                            chunk_max: float) -> None:
        """Merge a partial-chunk summary (`n`, `sum`, `sum_of_sq_dev`,
        `min`, `max`) into the running totals using the parallel-Welford
        merge. Lets the caller compute these on GPU and feed the scalars
        in without reducing on CPU."""
        if n <= 0:
            return
        m = float(n)
        chunk_mean = chunk_sum / m
        delta = chunk_mean - self.mean
        new_n = self.n + m
        self.mean += delta * m / new_n
        self.M2 += chunk_M2 + (delta ** 2) * self.n * m / new_n
        self.n = new_n
        if chunk_min < self.min:
            self.min = chunk_min
        if chunk_max > self.max:
            self.max = chunk_max

    def _update_reservoir(self, flat: np.ndarray) -> None:
        if len(self.reservoir) < self.reservoir_size:
            n_take = min(self.reservoir_size - len(self.reservoir), flat.size)
            picks = self._rng.sample(range(flat.size), n_take)
            self.reservoir.extend(float(flat[i]) for i in picks)
            self._seen += flat.size
        else:
            for v in flat:
                self._seen += 1
                j = self._rng.randint(0, self._seen - 1)
                if j < self.reservoir_size:
                    self.reservoir[j] = float(v)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.sqrt(self.M2 / (self.n - 1)))

    def percentiles_and_mad(self) -> dict[str, float]:
        if not self.reservoir:
            return {}
        arr = np.asarray(self.reservoir)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return {
            "p01": float(np.percentile(arr, 1)),
            "p50": med,
            "p99": float(np.percentile(arr, 99)),
            "mad": mad,
        }


# =============================================================================
# Pre-normalization (matches the inference-time transform up to z-scoring)
# =============================================================================

# =============================================================================
# GPU acceleration (optional, lazy import)
# =============================================================================
#
# Each file's heavy ops (NaN mask, missing-sentinel mask, optional clip+log10,
# count + sum + sum-of-squared-deviations + min + max) run on the GPU when
# CuPy is installed and a GPU is present. The disk read stays on CPU — that
# is the I/O-bound portion. Per-file compute drops from ~5-15 ms (numpy) to
# ~1-3 ms (CuPy + transfer), giving roughly 1.5-2x wall-clock speedup before
# I/O starts dominating.
#
# CuPy wheels are versioned per CUDA major (`cupy-cuda11x` for the 11.2
# stack TensorFlow already uses here) so they share the existing toolchain
# without a second CUDA install.

_CUPY = None                       # lazily-imported cupy module
_CUPY_AVAILABLE: bool | None = None  # None = not probed yet


def _cupy_available() -> bool:
    """Return True if CuPy imports and a CUDA device is visible."""
    global _CUPY, _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE
    try:
        import cupy as cp  # type: ignore
        # Probe the device — this raises if no CUDA runtime is available.
        cp.cuda.Device(0).compute_capability
        _CUPY = cp
        _CUPY_AVAILABLE = True
    except Exception:
        _CUPY = None
        _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def _gpu_reduce(arr_np: np.ndarray, spec: dict
                ) -> tuple[int, float, float, float, float] | None:
    """Filter + reduce one file on the GPU via CuPy.

    Returns `(n, sum, sum_of_squared_deviations, min, max)`, or None
    when every pixel was masked out. The masking, log10, and reductions
    all happen on-device; only five scalars come back to the host.
    """
    cp = _CUPY
    if cp is None:
        return None

    x = cp.asarray(arr_np, dtype=cp.float64).ravel()
    # NaN + missing-sentinel filter
    x = x[~cp.isnan(x)]
    missing_above = spec.get("missing_above")
    if missing_above is not None:
        x = x[x <= float(missing_above)]
    if int(x.size) == 0:
        return None

    # log_zscore pre-normalisation on-device
    if spec.get("transform") == "log_zscore":
        floor = spec.get("clip_min")
        if floor is not None:
            x = cp.maximum(x, cp.float64(floor))
        x = cp.log10(x)

    n = int(x.size)
    s = float(x.sum().get())
    mean = s / n
    m2 = float(((x - mean) ** 2).sum().get())
    a_min = float(x.min().get())
    a_max = float(x.max().get())
    return n, s, m2, a_min, a_max


def filter_and_pre_normalize(arr: np.ndarray, spec: dict) -> np.ndarray:
    """
    Apply variable-specific *pre-stats* filtering: drop NaN, drop
    `missing_above` sentinel pixels, clip to the floor (if any), and apply
    log10 (if the transform is `log_zscore`).

    The returned 1-D array contains ONLY valid signal pixels — anything
    that would have been masked at inference time is removed entirely so it
    cannot poison the mean.
    """
    flat = arr.astype(np.float64, copy=False).ravel()
    # Drop NaN
    flat = flat[~np.isnan(flat)]
    # Drop missing-sentinel pixels (variable-specific)
    missing_above = spec.get("missing_above")
    if missing_above is not None:
        flat = flat[flat <= float(missing_above)]
    if spec.get("transform") == "log_zscore":
        floor = spec.get("clip_min")
        if floor is not None:
            flat = np.clip(flat, float(floor), None)
        flat = np.log10(flat)
    return flat


# =============================================================================
# Training-window filter
# =============================================================================

def load_sequence_meta(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"ERROR: {path} not found. Run extract_patch_seq_for_datasets.py "
            f"first so the window definition is available."
        )
    return json.loads(path.read_text())


def load_product_filters(timestep_config_path: Path
                         ) -> dict[str, set[int]]:
    """Read every product's minute filter from timestep_config.json.

    Returns a dict mapping NORMALIZATION_SPEC source names ('radar',
    'mtg', 'nwcsaf', 'opera', 'lightning') to the set of valid
    minute-of-hour values. Missing or null filters become empty sets
    (treated as 'no snap' downstream).
    """
    if not timestep_config_path.exists():
        return {}
    cfg = json.loads(timestep_config_path.read_text())
    out: dict[str, set[int]] = {}
    for source, ts_key in _SOURCE_TO_TSCONFIG_PRODUCT.items():
        block = cfg.get("products", {}).get(ts_key, {})
        flt = block.get("filter")
        out[source] = set(int(m) for m in flt) if flt else set()
    return out


def snap_hhmm(hhmm: str, filter_minutes: set[int]) -> str:
    """Snap a 4-digit HHMM string to the nearest minute in the filter,
    with hour-boundary wrap. Tie-break prefers the earlier minute.
    Mirrors the snap rule used in extract_patches.py / intersect."""
    h, m = int(hhmm[:2]), int(hhmm[2:])
    best = min(filter_minutes, key=lambda fm: (
        min(abs(fm - m), 60 - abs(fm - m)),
        fm,
    ))
    diff = best - m
    if diff > 30:
        diff -= 60
    elif diff < -30:
        diff += 60
    total = (h * 60 + m + diff) % (24 * 60)
    return f"{total // 60:02d}{total % 60:02d}"


def snap_keys_to_filter(keys: set[tuple[str, str]],
                        filter_minutes: set[int]) -> set[tuple[str, str]]:
    """Snap every (date, HHMM) in `keys` to the per-product cadence grid.

    Empty filter = no snap (continuous / event-based product).
    """
    if not filter_minutes:
        return keys
    return {(d, snap_hhmm(h, filter_minutes)) for d, h in keys}


def expand_training_window(date_str: str, ref_utc: str,
                           step_minutes: int,
                           past_steps: int,
                           future_steps: int) -> list[tuple[str, str]]:
    """
    Expand a single (date, reference_utc) row into the full list of
    (date, HHMM) tuples covered by its past + current + future window.
    Handles day rollover (window crossing midnight).
    """
    parts = ref_utc.strip().split(":")
    h, m = int(parts[0]), int(parts[1])
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    out: list[tuple[str, str]] = []
    for k in range(-past_steps, future_steps + 1):
        t = base + timedelta(minutes=k * step_minutes)
        out.append((t.strftime("%Y-%m-%d"), t.strftime("%H%M")))
    return out


def load_training_keys(train_csv: Path,
                       seq_meta: dict) -> set[tuple[str, str]]:
    """
    Read train_data.csv and build the full set of (date, HHMM) tuples that
    appear in any training row's past+current+future window.
    """
    if not train_csv.exists():
        print(f"WARNING: {train_csv} not found — falling back to all data.")
        return set()

    step = int(seq_meta["step_minutes"])
    past = int(seq_meta["past_steps"])
    future = int(seq_meta["future_steps"])

    keys: set[tuple[str, str]] = set()
    with open(train_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row.get("reference_utc", "").strip()
            date_str = row.get("date", "").strip()
            if not ref or not date_str:
                continue
            try:
                for k in expand_training_window(date_str, ref,
                                                step, past, future):
                    keys.add(k)
            except ValueError:
                continue
    return keys


# =============================================================================
# Per-source file discovery + loading
# =============================================================================

_NPY_NAME_PATTERN = re.compile(
    r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_(\d{4})_(?P<var>.+?)\.npy$'
)
_LIGHTNING_NAME_PATTERN = re.compile(
    r'^lightning_(?P<var>density|current|occurrence)_'
    r'(?P<date>\d{8})_(?P<hhmm>\d{4})\.npy$'
)


def _walk_radar_or_mtg(root: Path, var: str,
                       training_keys: set[tuple[str, str]]
                       ) -> list[Path]:
    """
    `reprojected_data/radar_data/{VAR}/nc4_{date}-Romania_{VAR}/nc4_{date}-Romania_{HHMM}_{VAR}.npy`
    or
    `reprojected_data/satellite_data/MTG/{ch}/nc4_{date}-Romania_{ch}/nc4_{date}-Romania_{HHMM}_{ch}.npy`
    """
    var_root = root / var
    if not var_root.is_dir():
        return []
    out: list[Path] = []
    for day_dir in sorted(var_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for f in sorted(day_dir.iterdir()):
            m = _NPY_NAME_PATTERN.match(f.name)
            if not m:
                continue
            date_str = m.group(1)
            hhmm = m.group(2)
            if training_keys and (date_str, hhmm) not in training_keys:
                continue
            out.append(f)
    return out


def _walk_lightning(root: Path, var: str,
                    training_keys: set[tuple[str, str]]
                    ) -> list[Path]:
    """
    `reprojected_data/lightning_data/{var}/nc4_{date}-Romania_{var}/lightning_{var}_{YYYYMMDD}_{HHMM}.npy`
    """
    var_root = root / var
    if not var_root.is_dir():
        return []
    out: list[Path] = []
    for day_dir in sorted(var_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for f in sorted(day_dir.iterdir()):
            m = _LIGHTNING_NAME_PATTERN.match(f.name)
            if not m:
                continue
            d = m.group("date")
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            hhmm = m.group("hhmm")
            if training_keys and (date_str, hhmm) not in training_keys:
                continue
            out.append(f)
    return out


def _walk_nwcsaf(root: Path, var: str,
                 training_keys: set[tuple[str, str]]
                 ) -> list[Path]:
    """
    After the reproject pipeline unification, NWCSAF is written as one
    `.npy` per variable, same layout as radar / MTG / OPERA:

      reprojected_data/nwcsaf_data/{variable}/
          nc4_{date}-Romania_{variable}/nc4_{date}-Romania_{HHMM}_{variable}.npy

    Walks that tree and returns the matching file paths (filtered by
    `training_keys` when supplied).
    """
    var_root = root / var
    if not var_root.is_dir():
        return []
    out: list[Path] = []
    for day_dir in sorted(var_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for f in sorted(day_dir.iterdir()):
            m = _NPY_NAME_PATTERN.match(f.name)
            if not m:
                continue
            date_str = m.group(1)
            hhmm = m.group(2)
            if training_keys and (date_str, hhmm) not in training_keys:
                continue
            out.append(f)
    return out


def _walk_opera(root: Path, var: str,
                training_keys: set[tuple[str, str]]
                ) -> list[Path]:
    """
    Reproject writes OPERA as `.npy` (one file per variable, same layout as
    radar / MTG / lightning):

      reprojected_data/opera_data/{product}/
          nc4_{date}-Romania_{product}/nc4_{date}-Romania_{HHMM}_{product}.npy

    `var` is the canonical name used in the stats spec
    (`opera_reflectivity` / `opera_rainfall_rate`); the on-disk folder
    name is the short form (`reflectivity` / `rainfall_rate`).
    """
    product_subdir = "reflectivity" if var == "opera_reflectivity" else "rainfall_rate"
    var_root = root / product_subdir
    if not var_root.is_dir():
        return []
    pattern = re.compile(
        r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_(\d{4})_'
        + re.escape(product_subdir) + r'\.npy$'
    )
    out: list[Path] = []
    for day_dir in sorted(var_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for f in sorted(day_dir.iterdir()):
            m = pattern.match(f.name)
            if not m:
                continue
            date_str = m.group(1)
            hhmm = m.group(2)
            if training_keys and (date_str, hhmm) not in training_keys:
                continue
            out.append(f)
    return out


def discover_inputs(reproject_root: Path, var: str,
                    training_keys: set[tuple[str, str]],
                    product_filters: dict[str, set[int]] | None = None,
                    ) -> list:
    """
    Dispatcher: walks the product's reprojected subtree and returns the
    list of `.npy` paths matching the training keys.

    `training_keys` is on the master grid (e.g. {:00, :15, :30, :45}
    when step=15). Different products live on different minute grids:
    MTG / NWCSAF / radar at {:00, :10, :30, :40}; OPERA at the master
    grid. Without a snap, the matcher would skip MTG/NWCSAF files at
    :10 and :40 even though `extract_patches.py` loads them at runtime
    (via its cadence snap). To keep the stats aligned with the files
    actually used in training, snap the training keys to each product's
    filter from `timestep_config.json` before matching.
    """
    spec = NORMALIZATION_SPEC[var]
    source = spec["source"]

    keys = training_keys
    if training_keys and product_filters is not None:
        flt = product_filters.get(source)
        if flt:
            keys = snap_keys_to_filter(training_keys, flt)

    if source == "radar":
        return _walk_radar_or_mtg(reproject_root / "radar_data", var, keys)
    if source == "mtg":
        return _walk_radar_or_mtg(
            reproject_root / "satellite_data" / "MTG", var, keys,
        )
    if source == "lightning":
        return _walk_lightning(
            reproject_root / "lightning_data", var, keys,
        )
    if source == "nwcsaf":
        return _walk_nwcsaf(
            reproject_root / "nwcsaf_data", var, keys,
        )
    if source == "opera":
        return _walk_opera(
            reproject_root / "opera_data", var, keys,
        )
    raise ValueError(f"Unknown source: {source!r}")


def load_array(item, source: str) -> np.ndarray | None:
    """Load a 2-D array from a reprojected `.npy` file.

    After the reproject pipeline unification, every product family
    (radar, MTG, lightning, NWCSAF, OPERA) writes one `.npy` per
    (variable, timestep), so the loader is the same everywhere.
    """
    try:
        arr = np.load(item, allow_pickle=False)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            return None
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(np.nan)
        return arr.astype(np.float64, copy=False)
    except Exception as e:
        print(f"    WARN: could not load {item}: {e}", file=sys.stderr)
        return None


# =============================================================================
# Per-variable computation
# =============================================================================

def compute_variable_stats(var: str,
                           items: list,
                           spec: dict,
                           sample_fraction: float,
                           reservoir_size: int,
                           rng: random.Random,
                           use_gpu: bool = False) -> dict | None:
    if not items:
        print(f"  {var:22s}: no input files matched")
        return None

    if 0 < sample_fraction < 1.0:
        n_sample = max(1, int(round(len(items) * sample_fraction)))
        items = rng.sample(items, n_sample)

    stats = WelfordStats(reservoir_size=reservoir_size, rng=rng)
    n_files_used = 0
    # When percentiles / MAD are requested the reservoir sampler needs
    # the actual filtered pixel values, so we have to keep the full
    # filtered array on CPU. In that case GPU only saves the per-array
    # reductions (smaller win), and the data round-trip through host
    # memory eats that savings. Fall back to CPU automatically.
    gpu_active = use_gpu and reservoir_size == 0
    for item in items:
        arr = load_array(item, spec["source"])
        if arr is None:
            continue
        if gpu_active:
            partial = _gpu_reduce(arr, spec)
            if partial is None:
                continue
            n, s, m2, a_min, a_max = partial
            stats.update_from_partial(
                n=n, chunk_sum=s, chunk_M2=m2,
                chunk_min=a_min, chunk_max=a_max,
            )
        else:
            flat = filter_and_pre_normalize(arr, spec)
            if flat.size == 0:
                continue
            stats.update_chunk(flat)
        n_files_used += 1

    if stats.n == 0:
        print(f"  {var:22s}: every file empty / unreadable after filtering")
        return None

    result: dict = {
        "source":         spec["source"],
        "transform":      spec["transform"],
        "n_files_used":   n_files_used,
        "n_valid_pixels": int(stats.n),
        "min":            round(stats.min, 6),
        "max":            round(stats.max, 6),
        "mean":           round(stats.mean, 6),
        "std":            round(stats.std, 6),
        "fill":           spec.get("fill"),
    }
    if "clip_min" in spec:
        result["clip_min"] = spec["clip_min"]
    if "missing_above" in spec:
        result["missing_above"] = spec["missing_above"]
    if reservoir_size > 0:
        result.update({k: round(v, 6) for k, v in
                       stats.percentiles_and_mad().items()})
    # Near-constant flag
    abs_mean = abs(stats.mean)
    if abs_mean > 0 and stats.std < NEAR_CONSTANT_EPS * abs_mean:
        result["near_constant"] = True
        result["near_constant_warning"] = (
            f"std ({stats.std:g}) < {NEAR_CONSTANT_EPS}*|mean| "
            f"({abs_mean:g}); z-scoring may amplify noise. "
            f"Consider clipping or robust scaling."
        )
    else:
        result["near_constant"] = False
    return result


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-variable normalization stats (mean/std) "
                    "from the reprojected data in our_data/reprojected_data/. "
                    "create_datasets.py requires the resulting JSON."
    )
    parser.add_argument('--reproject_root', type=str,
                        default=str(DEFAULT_REGRID_ROOT),
                        help=f'Reprojected-data root '
                             f'(default: {DEFAULT_REGRID_ROOT})')
    parser.add_argument('--source', type=str, default='dbscan',
                        choices=['dbscan', 'lightning'],
                        help="Which extract_patch_seq source to compute "
                             "stats for. Selects "
                             "train_data_<source>.csv + "
                             "sequence_meta_<source>.json as inputs and "
                             "writes normalization_stats_<source>.json. "
                             "The two tracks need separate stats because "
                             "their training distributions differ. "
                             "(default: dbscan)")
    parser.add_argument('--train_csv', type=str, default=None,
                        help='Explicit path to the training CSV. '
                             'Defaults to '
                             'our_data/train_data_<source>.csv.')
    parser.add_argument('--sequence_meta', type=str, default=None,
                        help='Explicit path to the sequence metadata '
                             'JSON. Defaults to '
                             'our_data/sequence_meta_<source>.json.')
    parser.add_argument('--no_split_filter', action='store_true',
                        help='Disable the training-window filter and use '
                             'every reprojected file (validation / test data '
                             'WILL leak; only useful for diagnostics).')
    parser.add_argument('--variables', type=str, nargs='+', default=None,
                        help='Subset of variables to process '
                             f'(default: all spec entries).')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='Random fraction of matching files per '
                             'variable (default: 1.0 = use all).')
    parser.add_argument('--with_percentiles', action='store_true',
                        help='Also compute p01/p50/p99 and MAD '
                             '(reservoir-sampled).')
    parser.add_argument('--reservoir_size', type=int, default=200_000,
                        help='Reservoir sample size for percentile / MAD '
                             '(default: 200000). Ignored unless '
                             '--with_percentiles.')
    parser.add_argument('--timestep_config', type=str,
                        default=str(DEFAULT_TIMESTEP_CONFIG),
                        help=f'Path to timestep_config.json (default: '
                             f'{DEFAULT_TIMESTEP_CONFIG}). Used to snap '
                             f'training keys onto each product\'s cadence '
                             f'grid so 10-min products (MTG/NWCSAF) get '
                             f'their :10/:40 files counted, not just :00/:30.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'],
                        default='auto',
                        help="Where the per-array reductions run. 'auto' "
                             "uses CuPy + GPU when available, else CPU; "
                             "'cpu' forces the pure-numpy path; 'gpu' "
                             "errors out if CuPy / a CUDA device is "
                             "missing. With --with_percentiles the GPU "
                             "path falls back to CPU automatically "
                             "(reservoir sampling needs the full filtered "
                             "array on host).")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON path. Defaults to '
                             'our_data/normalization_stats_<source>.json.')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed (default: 0).')

    args = parser.parse_args()
    if not args.with_percentiles:
        args.reservoir_size = 0

    # Resolve --device into a concrete `use_gpu` boolean.
    if args.device == 'gpu':
        if not _cupy_available():
            sys.exit("ERROR: --device gpu requested but CuPy is not "
                     "available. Install `cupy-cuda11x` (matching the "
                     "existing CUDA 11.2 toolkit) or use --device cpu.")
        use_gpu = True
    elif args.device == 'cpu':
        use_gpu = False
    else:  # auto
        use_gpu = _cupy_available()

    # Resolve per-source paths. The user can override any of them with
    # an explicit flag - we only build the default when it's missing.
    reproject_root = Path(args.reproject_root)
    train_csv = Path(
        args.train_csv
        if args.train_csv
        else DEFAULT_DATA_ROOT / f"train_data_{args.source}.csv"
    )
    seq_meta_path = Path(
        args.sequence_meta
        if args.sequence_meta
        else DEFAULT_DATA_ROOT / f"sequence_meta_{args.source}.json"
    )
    output_path = Path(
        args.output
        if args.output
        else DEFAULT_DATA_ROOT / f"normalization_stats_{args.source}.json"
    )

    variables = None
    if args.variables:
        variables = set(args.variables)
        unknown = variables - set(NORMALIZATION_SPEC.keys())
        if unknown:
            print(f"WARNING: unknown variable(s) (no spec): {sorted(unknown)}")
            variables -= unknown
        if not variables:
            sys.exit("ERROR: no known variables selected.")
    else:
        variables = set(NORMALIZATION_SPEC.keys())

    rng = random.Random(args.seed)

    # Build training-key set
    if args.no_split_filter:
        training_keys: set[tuple[str, str]] = set()
        train_n_rows = None
    else:
        seq_meta = load_sequence_meta(seq_meta_path)
        training_keys = load_training_keys(train_csv, seq_meta)
        train_n_rows = (sum(1 for _ in open(train_csv))
                        if train_csv.exists() else 0) - 1

    # Per-product minute filters from timestep_config.json. Used to snap
    # training keys onto each product's own cadence grid so MTG/NWCSAF
    # files at :10 and :40 are matched (not only :00/:30).
    product_filters = load_product_filters(Path(args.timestep_config))

    print("=" * 70)
    print("Normalization Stats Computation")
    print("=" * 70)
    print(f"Reproject root         : {reproject_root}")
    print(f"Training filter     : "
          f"{'DISABLED — using all files' if args.no_split_filter else f'enabled ({len(training_keys)} unique (date, HHMM) keys)'}")
    print(f"Variables           : {sorted(variables)}")
    print(f"Sample fraction     : {args.sample_fraction}")
    print(f"With percentiles    : {args.with_percentiles}")
    print(f"Device              : "
          f"{'GPU (CuPy)' if use_gpu else 'CPU (numpy)'}"
          f"{' [forced cpu]' if args.device == 'cpu' else ''}"
          f"{' [auto-fallback: no GPU]' if args.device == 'auto' and not use_gpu else ''}")
    print(f"Source              : {args.source}")
    print(f"Output              : {output_path}")
    print()

    variable_results: dict[str, dict] = {}
    for var in sorted(variables):
        spec = NORMALIZATION_SPEC[var]
        items = discover_inputs(reproject_root, var, training_keys,
                                product_filters=product_filters)
        print(f"  {var:22s}: {len(items)} file(s) match")
        result = compute_variable_stats(
            var, items, spec, args.sample_fraction,
            args.reservoir_size, rng, use_gpu=use_gpu,
        )
        if result is None:
            continue
        variable_results[var] = result
        nc_flag = "  [near-constant]" if result.get("near_constant") else ""
        print(f"     -> mean={result['mean']:.4f}, std={result['std']:.4f}, "
              f"min={result['min']:.4f}, max={result['max']:.4f}, "
              f"n_px={result['n_valid_pixels']:,}{nc_flag}")

    payload = {
        "computed_utc":   datetime.now(timezone.utc)
                                 .isoformat(timespec="seconds"),
        "source":         args.source,
        "reproject_root":    str(reproject_root),
        "training_filter": {
            "train_csv":       (None if args.no_split_filter else str(train_csv)),
            "n_rows":          (None if args.no_split_filter else train_n_rows),
            "n_unique_keys":   (None if args.no_split_filter else len(training_keys)),
        },
        "policy": {
            "training_set_only":   not args.no_split_filter,
            "scope":               ("single scalar mean/std per variable, "
                                    "over all valid pixels of training "
                                    "timesteps"),
            "missing_handling":    ("NaN and missing-sentinel pixels are "
                                    "dropped before Welford accumulation"),
            "spatial":             "no per-pixel climatology",
            "near_constant_eps":   NEAR_CONSTANT_EPS,
        },
        "sample_fraction": args.sample_fraction,
        "variables":       variable_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Wrote normalization stats to {output_path}")
    print(f"  {len(variable_results)} / {len(variables)} variables ready")
    if len(variable_results) < len(variables):
        missing = sorted(set(variables) - set(variable_results.keys()))
        print(f"  Missing: {missing} — create_datasets.py will refuse "
              f"to use those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
