"""
compute_normalization_stats.py — Derive per-variable normalization parameters
from the reprojected data in `our_data/reprojected_data/`.

This script is Step 4.3, between intersect_product_coverage.py (4.2) and
create_datasets.py (Step 5). It produces `our_data/normalization_stats.json`,
which `create_datasets.py` then consumes — there is **no fallback to the
Leinonen Swiss constants**, so this script must be run successfully before
TF datasets can be built.

When `train_data_consistent.csv` exists in the data root (i.e. Step 4.2
has been run), pass it via `--train_csv our_data/train_data_consistent.csv`
so the stats reflect the actual training set the model will see.

Policy decisions (also recorded inside the JSON for traceability)
---------------------------------------------------------------

1. **Training set only.** Statistics are computed exclusively from
   timesteps that appear inside *training-eligible windows*. For each row
   of `train_data.csv`, the past + current + future timesteps (read from
   `sequence_meta.json`) are expanded into the set of (date, HHMM)
   tuples actually consumed at training time; every other timestep
   (validation, test, or off-grid) is excluded. Pass `--no_split_filter`
   to disable this and use every reprojected file.

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

try:
    from netCDF4 import Dataset as NC4Dataset
except ImportError:
    NC4Dataset = None  # only needed for NWCSAF / OPERA sources


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"
DEFAULT_REGRID_ROOT = DEFAULT_DATA_ROOT / "reprojected_data"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "normalization_stats.json"
DEFAULT_TRAIN_CSV = DEFAULT_DATA_ROOT / "train_data.csv"
DEFAULT_SEQUENCE_META = DEFAULT_DATA_ROOT / "sequence_meta.json"

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

# Which NWCSAF L2 file holds each variable (CTTH or CMIC).
NWCSAF_FILE_FOR_VAR = {
    "ctth_alti":  "CTTH",
    "ctth_tempe": "CTTH",
    "cmic_cot":   "CMIC",
}

# Internal NWCSAF NetCDF variable name (typically lowercase)
NWCSAF_NC_VAR_NAME = {
    "ctth_alti":  "ctth_alti",
    "ctth_tempe": "ctth_tempe",
    "cmic_cot":   "cmic_cot",
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
        chunk_mean = float(flat.mean())
        chunk_M2 = float(((flat - chunk_mean) ** 2).sum())

        delta = chunk_mean - self.mean
        new_n = self.n + m
        self.mean += delta * m / new_n
        self.M2 += chunk_M2 + (delta ** 2) * self.n * m / new_n
        self.n = new_n
        cmin = float(flat.min())
        cmax = float(flat.max())
        if cmin < self.min:
            self.min = cmin
        if cmax > self.max:
            self.max = cmax

        if self.reservoir_size > 0:
            self._update_reservoir(flat)

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
_NWCSAF_NAME_PATTERN = re.compile(
    r'^S_NWC_(?P<product>CMIC|CTTH)_[^_]+_[^_]+_'
    r'(?P<date>\d{8})T(?P<hhmm>\d{4})\d{2}Z(.*?)?\.nc$'
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
                 ) -> list[tuple[Path, str]]:
    """
    NWCSAF variables live inside CTTH or CMIC .nc files in
    `reprojected_data/nwcsaf_data/{date}-Romania/`. Returns
    [(file_path, internal_variable_name), ...] for every (date, HHMM) that
    matches the training keys and whose file holds `var`.
    """
    product = NWCSAF_FILE_FOR_VAR[var]
    nc_var = NWCSAF_NC_VAR_NAME[var]
    out: list[tuple[Path, str]] = []
    for day_dir in sorted(root.iterdir()):
        if not day_dir.is_dir():
            continue
        for f in sorted(day_dir.iterdir()):
            m = _NWCSAF_NAME_PATTERN.match(f.name)
            if not m or m.group("product") != product:
                continue
            d = m.group("date")
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            hhmm = m.group("hhmm")
            if training_keys and (date_str, hhmm) not in training_keys:
                continue
            out.append((f, nc_var))
    return out


def _walk_opera(root: Path, var: str,
                training_keys: set[tuple[str, str]]
                ) -> list[tuple[Path, str]]:
    """
    `reprojected_data/opera_data/{product}/nc4_{date}-Romania_{product}/nc4_{date}-Romania_{HHMM}_{product}.nc`
    The internal NetCDF variable name follows the reproject_opera.py
    convention: 'max_reflectivity' or 'rainfall_rate'.
    """
    product_subdir = "reflectivity" if var == "opera_reflectivity" else "rainfall_rate"
    nc_var = "max_reflectivity" if var == "opera_reflectivity" else "rainfall_rate"
    var_root = root / product_subdir
    if not var_root.is_dir():
        return []
    pattern = re.compile(
        r'^nc4_(\d{4}-\d{2}-\d{2})-Romania_(\d{4})_' + re.escape(product_subdir) + r'\.nc$'
    )
    out: list[tuple[Path, str]] = []
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
            out.append((f, nc_var))
    return out


def discover_inputs(reproject_root: Path, var: str,
                    training_keys: set[tuple[str, str]]
                    ) -> list:
    """
    Dispatcher: returns either list[Path] (radar/mtg/lightning) or
    list[(Path, internal_var_name)] (nwcsaf/opera).
    """
    spec = NORMALIZATION_SPEC[var]
    source = spec["source"]
    if source == "radar":
        return _walk_radar_or_mtg(reproject_root / "radar_data", var, training_keys)
    if source == "mtg":
        return _walk_radar_or_mtg(
            reproject_root / "satellite_data" / "MTG", var, training_keys,
        )
    if source == "lightning":
        return _walk_lightning(
            reproject_root / "lightning_data", var, training_keys,
        )
    if source == "nwcsaf":
        return _walk_nwcsaf(
            reproject_root / "nwcsaf_data", var, training_keys,
        )
    if source == "opera":
        return _walk_opera(
            reproject_root / "opera_data", var, training_keys,
        )
    raise ValueError(f"Unknown source: {source!r}")


def load_array(item, source: str) -> np.ndarray | None:
    """Load a 2-D array for one file, with per-source quirks."""
    try:
        if source in ("radar", "mtg", "lightning"):
            arr = np.load(item, allow_pickle=False)
        else:
            # nwcsaf / opera — item is (path, nc_var)
            if NC4Dataset is None:
                raise RuntimeError("netCDF4 not installed")
            path, nc_var = item
            with NC4Dataset(path, "r") as ds:
                if nc_var not in ds.variables:
                    return None
                arr = np.asarray(ds.variables[nc_var][:])
        # Strip time dim if present
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            return None
        # Mask the netCDF _FillValue and resolve MaskedArray
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(np.nan)
        return arr.astype(np.float64, copy=False)
    except Exception as e:
        path = item if not isinstance(item, tuple) else item[0]
        print(f"    WARN: could not load {path}: {e}", file=sys.stderr)
        return None


# =============================================================================
# Per-variable computation
# =============================================================================

def compute_variable_stats(var: str,
                           items: list,
                           spec: dict,
                           sample_fraction: float,
                           reservoir_size: int,
                           rng: random.Random) -> dict | None:
    if not items:
        print(f"  {var:22s}: no input files matched")
        return None

    if 0 < sample_fraction < 1.0:
        n_sample = max(1, int(round(len(items) * sample_fraction)))
        items = rng.sample(items, n_sample)

    stats = WelfordStats(reservoir_size=reservoir_size, rng=rng)
    n_files_used = 0
    for item in items:
        arr = load_array(item, spec["source"])
        if arr is None:
            continue
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
    parser.add_argument('--train_csv', type=str,
                        default=str(DEFAULT_TRAIN_CSV),
                        help=f'train_data.csv used to derive training '
                             f'(date, HHMM) keys '
                             f'(default: {DEFAULT_TRAIN_CSV})')
    parser.add_argument('--sequence_meta', type=str,
                        default=str(DEFAULT_SEQUENCE_META),
                        help=f'sequence_meta.json used for window expansion '
                             f'(default: {DEFAULT_SEQUENCE_META})')
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
    parser.add_argument('--output', '-o', type=str,
                        default=str(DEFAULT_OUTPUT),
                        help=f'Output JSON path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed (default: 0).')

    args = parser.parse_args()
    if not args.with_percentiles:
        args.reservoir_size = 0

    reproject_root = Path(args.reproject_root)
    train_csv = Path(args.train_csv)
    seq_meta_path = Path(args.sequence_meta)

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

    print("=" * 70)
    print("Normalization Stats Computation")
    print("=" * 70)
    print(f"Reproject root         : {reproject_root}")
    print(f"Training filter     : "
          f"{'DISABLED — using all files' if args.no_split_filter else f'enabled ({len(training_keys)} unique (date, HHMM) keys)'}")
    print(f"Variables           : {sorted(variables)}")
    print(f"Sample fraction     : {args.sample_fraction}")
    print(f"With percentiles    : {args.with_percentiles}")
    print(f"Output              : {args.output}")
    print()

    variable_results: dict[str, dict] = {}
    for var in sorted(variables):
        spec = NORMALIZATION_SPEC[var]
        items = discover_inputs(reproject_root, var, training_keys)
        print(f"  {var:22s}: {len(items)} file(s) match")
        result = compute_variable_stats(
            var, items, spec, args.sample_fraction,
            args.reservoir_size, rng,
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
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Wrote normalization stats to {out_path}")
    print(f"  {len(variable_results)} / {len(variables)} variables ready")
    if len(variable_results) < len(variables):
        missing = sorted(set(variables) - set(variable_results.keys()))
        print(f"  Missing: {missing} — create_datasets.py will refuse "
              f"to use those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
