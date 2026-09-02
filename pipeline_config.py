"""
pipeline_config.py — COALITION-4 Romanian Adaptation
=====================================================
Project-wide constants that more than one pipeline stage needs to agree
on. Deliberately dependency-free (no TensorFlow, no numpy) so the light
CLIs can import it without paying for a TF import.

Sample-selection source
-----------------------
Every artefact this project writes is suffixed with the sample-selection
source: `datasets/<mode>_<source>/`, `coalition_<mode>_<source>.keras`,
`train_data_<source>.csv`, `normalization_stats_<source>.json`, and so
on. See train_models.build_run_tag.

Historically there were two sources — `dbscan` (DBSCAN clusters in the
OPERA rain field, via identify_patches.py) and `lightning`
(lightning-active windows). The lightning-driven track was retired:
patches are now extracted from OPERA only, so `dbscan` is the sole
source and `--source` is no longer a CLI flag anywhere.

The `source` *parameter* is still threaded through the functions that
build paths, defaulting to SOURCE. That keeps the on-disk naming stable
(nothing had to be renamed when the flag went away) and leaves a seam if
a second selection strategy is ever introduced.
"""

# The only sample-selection source. DBSCAN clustering over OPERA
# `rainfall_rate`, produced by identify_patches.py.
SOURCE = "dbscan"


# =============================================================================
# Filesystem roots
# =============================================================================
# Every default here is resolved against THIS FILE, not the working
# directory. `./our_data` as a default meant the scripts only worked when
# invoked from the repository root: run one from anywhere else and it
# would silently create an empty tree beside you rather than find the
# real one.
#
# `datasets/` is resolved separately from `data_root` because the two
# have very different sizes and lifetimes - the patch pool and the
# reprojected archive are terabytes that stay put, while a TFRecord
# dataset is tens of gigabytes that may need to live on whichever disk
# has room this month. Previously the only way to separate them was an
# NTFS junction.
#
# Precedence, most specific first:
#     1. the explicit CLI flag
#     2. the environment variable
#     3. the default derived from the repository location
#
# So a shell can export COALITION4_DATASETS_ROOT once for a whole run
# sequence, and any single command can still override it.

import os
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

DATA_ROOT_ENV = "COALITION4_DATA_ROOT"
DATASETS_ROOT_ENV = "COALITION4_DATASETS_ROOT"
MODEL_DIR_ENV = "COALITION4_MODEL_DIR"

DATASETS_DIRNAME = "datasets"


def _from_env(name):
    value = os.environ.get(name, "").strip()
    return _Path(value) if value else None


def resolve_data_root(explicit=None) -> _Path:
    """Root holding patches, split CSVs, statistics and reprojected data."""
    if explicit:
        return _Path(explicit)
    return _from_env(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT


def resolve_model_dir(explicit=None) -> _Path:
    """Directory holding trained checkpoints and their history JSON."""
    if explicit:
        return _Path(explicit)
    return _from_env(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR


def resolve_datasets_root(data_root=None, explicit=None) -> _Path:
    """Root holding the built TFRecord datasets.

    Falls back to `<data_root>/datasets`, so the layout is unchanged for
    anyone who passes nothing.
    """
    if explicit:
        return _Path(explicit)
    env = _from_env(DATASETS_ROOT_ENV)
    if env:
        return env
    return resolve_data_root(data_root) / DATASETS_DIRNAME


def add_root_arguments(parser, datasets: bool = True,
                       model_dir: bool = False) -> None:
    """Attach the standard root flags, worded identically everywhere.

    Defaults stay None so `resolve_*` can tell "not given" from "given
    the same value as the default" - the env var must not win over an
    explicit flag that happens to match.
    """
    parser.add_argument(
        "--data_root", type=str, default=None, metavar="PATH",
        help=f"Root holding patches/, split CSVs and statistics "
             f"(default: {DEFAULT_DATA_ROOT}, or ${DATA_ROOT_ENV}).")
    if datasets:
        parser.add_argument(
            "--datasets_root", type=str, default=None, metavar="PATH",
            help=f"Root holding the built TFRecord datasets (default: "
                 f"<data_root>/{DATASETS_DIRNAME}, or "
                 f"${DATASETS_ROOT_ENV}). Point it at another disk to "
                 f"keep datasets off the one holding the patch pool.")
    if model_dir:
        parser.add_argument(
            "--model_dir", type=str, default=None, metavar="PATH",
            help=f"Directory holding trained models and history JSON "
                 f"(default: {DEFAULT_MODEL_DIR}, or ${MODEL_DIR_ENV}).")
