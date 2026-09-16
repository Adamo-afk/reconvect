"""
train_models.py — COALITION-4 Romanian Adaptation: Training Script
==================================================================
Builds the recurrent-convolutional architecture from scratch, loads the
pre-built TF datasets, trains, and saves the model + history.

Two ways to run
---------------
1. Loop over every mode listed in a config file (the common case):

       python train_models.py --config training.config

2. Train a single mode (still reads hyperparameters from the config):

       python train_models.py --config training.config \
           --mode mtg_opera_mtgmr_rainfall

Available training modes
------------------------
The mode determines which input groups feed the model and what the target
is. The dataset for each mode must already exist under
`our_data/datasets/<mode>/` (created by create_datasets.py — see its CLI
choices for the full list).

  - mtg_opera_radar_only_rainfall
        Inputs:  MTG vis_06 (HR) + OPERA reflectivity + rainfall_rate (MR).
                 No MTG IR/WV. No lightning.
        Target:  opera_rainfall_rate 5-class.

  - mtg_opera_mtgmr_rainfall
        Inputs:  mtg_opera_radar_only_rainfall + MTG IR/WV in MR.
        Target:  same as mtg_opera_radar_only_rainfall.

  - mtg_lightning_opera_rainfall
        Inputs:  lightning (density / current / occurrence) + MTG vis_06
                 in HR; OPERA reflectivity + rainfall_rate + MTG IR/WV
                 in MR.
        Target:  opera_rainfall_rate 5-class.

        Inputs:  same as mtg_opera_mtgmr_rainfall.
        Target:  opera_rainfall_rate continuous regression in [0, 1].
                 Exists so the SepConv ensemble baseline can be compared
                 against COALITION-4 on identical inputs — either
                 continuous-vs-continuous, or with both binned back to
                 the 5 classes.

  - mtg_lightning_opera_occurrence
        Inputs:  same as mtg_lightning_opera_rainfall.
        Target:  lightning binary occurrence (focal loss). Pairs with
                 mtg_lightning_opera_rainfall as the dual-target experiment
                 on identical inputs.

A sixth mode, `mtg_opera_occurrence`, is the knowledge-distillation
student. It is NOT trainable here — see train_lightning_kd.py.

Mode names carry their own track marker: `_rainfall` (5-class),
`_logz` (log_zscore, the baseline), `_occurrence` (lightning binary). The mode
name is therefore also the artefact tag — `build_run_tag` just appends
`_<source>`.

Hyperparameters, the run list, the LR schedule, and the early-stopping
configuration all live in `training.config`. See the docstring at the top
of that file for the editable fields.

Requires:
    - TensorFlow 2.x with GPU support
    - Pre-built TF datasets from create_datasets.py
    - our_data/lightning_fraction_<source>[_<period>].json
      (for `_occurrence` modes only)
"""

import argparse
import configparser
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline_config import (
    SOURCE,
    resolve_data_root,
    resolve_datasets_root,
    resolve_model_dir,
)
from periods import Period, data_tag, require_no_overlap
from compress_datasets import ensure_available, inuse_for
from ensemble_plan import (
    check_member_datasets,
    format_dataset_check,
    require_last_state,
    state_period,
)
from periods import sequence_meta_name


# =============================================================================
# Documented training modes
# =============================================================================
#
# Used to validate `--mode` / `[modes].run` against a known set and to make
# the registry of available modes discoverable. Kept in sync with
# create_datasets.get_mode_config() — when you add a new mode there, add
# a matching entry here so `--list-modes` and config validation keep working.

TRAINING_MODES: dict[str, dict[str, str]] = {
    "mtg_opera_radar_only_rainfall": {
        "target":  "opera_rainfall_rate 5-class",
        "summary": "Baseline: MTG vis_06 + OPERA. No MTG IR/WV, no lightning.",
    },
    "mtg_opera_mtgmr_rainfall": {
        "target":  "opera_rainfall_rate 5-class",
        "summary": "Baseline + MTG IR/WV in MR.",
    },
    "mtg_lightning_opera_rainfall": {
        "target":  "opera_rainfall_rate 5-class",
        "summary": "Lightning + MTG vis_06 in HR; OPERA + MTG IR/WV in MR; "
                   "OPERA rainfall as label.",
    },
    "mtg_lightning_opera_occurrence": {
        "target":  "lightning binary occurrence",
        "summary": "Same inputs as mtg_lightning_opera_rainfall; target is "
                   "lightning occurrence instead of OPERA rainfall.",
    },
    # The ablation half of the SepConv-ens comparison. It was buildable by
    # create_datasets (BUILDABLE_MODES) but missing here, so the dataset
    # could be created and then not trained. Nothing else was needed: this
    # table is descriptive - used for validation and --list-modes - while
    # the architecture is read from the dataset's metadata.json.
    "opera_radar_only_rainfall": {
        "target":  "opera_rainfall_rate 5-class",
        "summary": "Ablation for the SepConv-ens comparison: OPERA "
                   "rainfall_rate only, no MTG and no lightning, so it "
                   "consumes exactly what the baseline does.",
    },
}


# =============================================================================
# Run-tag naming for saved model artefacts
# =============================================================================
# Every mode name carries its own track marker — rainfall-headed modes end
# in `_rainfall`, lightning-headed modes in `_occurrence` (or are named
# `mtg_lightning`). So the artefact tag is just `<mode>_<source>`:
#
#     mtg_lightning_opera_occurrence + dbscan
#         -> coalition_mtg_lightning_opera_occurrence_dbscan.keras
#     mtg_lightning_opera_rainfall   + dbscan
#         -> coalition_mtg_lightning_opera_rainfall_dbscan.keras
#
# `source` is the sample-selection track (dbscan | lightning), so the two
# tracks never overwrite each other's weights, history, or datasets.


def build_run_tag(mode: str, source: str,
                  period: str | None = None) -> str:
    """Return the filename tag used for saved model artefacts, checkpoints,
    and dataset directories. Single source of truth for the convention —
    every script that reads or writes an artefact routes through here.

    `period` is an ensemble member label (e.g. '2025warm'). Omitting it
    yields the original two-part tag, so every artefact produced before
    period support keeps its name and stays loadable — there are trained
    models on disk under that convention.

        build_run_tag('mtg_opera_mtgmr_rainfall', 'dbscan')
            -> 'mtg_opera_mtgmr_rainfall_dbscan'
        build_run_tag('mtg_opera_mtgmr_rainfall', 'dbscan', '2025warm')
            -> 'mtg_opera_mtgmr_rainfall_dbscan_2025warm'
    """
    tag = f"{mode}_{source}"
    if period:
        tag = f"{tag}_{period}"
    return tag


def model_sidecar_path(model_path) -> Path:
    """`coalition_<tag>.keras` -> `coalition_<tag>.meta.json`."""
    return Path(model_path).with_suffix(".meta.json")


def save_model_period(model_path, period, mode=None, source=None,
                      stage=None, dataset_dir=None) -> Path:
    """Record what a saved model was trained on, next to the weights.

    A `.keras` file carries no provenance, so the period a model was
    trained over would otherwise live only in its filename — and a
    filename is not something a leakage check should trust. This sidecar
    is what `load_model_period` reads when the model is later reused as a
    frozen feature extractor.
    """
    path = model_sidecar_path(model_path)
    payload = {
        "model": Path(model_path).name,
        "mode": mode,
        "source": source,
        "stage": stage,
        "period": period.to_dict() if isinstance(period, Period) else None,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_model_period(model_path) -> "Period | None":
    """Read a model's training period from its sidecar.

    Returns None when the sidecar is absent — which is the honest answer
    for every model trained before period support existed, and is treated
    by the overlap gate as "unknown", not "safe".
    """
    path = model_sidecar_path(model_path)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return Period.from_dict(json.load(fh).get("period"))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"WARNING: could not read period from {path}: {exc}")
        return None


# =============================================================================
# Config loader (INI via configparser)
# =============================================================================

DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parent / "training.config"


def _parse_norm(raw: str | None) -> str | None:
    """Map a config `norm` value to what build_coalition_model expects.

    Accepts 'none', 'None', empty string, or `null` -> Python None.
    Anything else passes through after lower-casing.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "null"):
        return None
    return s


def _coerce(value: str, target_type):
    """Convert a configparser string to the requested Python type."""
    if target_type is bool:
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return target_type(value)


def load_training_config(path: Path) -> dict:
    """Load `training.config` and return a structured config dict.

    Shape of the returned dict:

        {
            "modes":          [...],     # mode names, in run order
            "defaults":       {...},     # epochs, batch_size, dropout, ...
            "lr_schedule":    {...},
            "early_stopping": {...},
            "mode_overrides": {mode: {...}, ...},
        }

    Per-mode overrides under [mode.<name>] are applied on top of the
    defaults at call time via `merge_for_mode()`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training config not found: {path}\n"
            f"Pass --config explicitly or create the default config at "
            f"{DEFAULT_TRAINING_CONFIG}."
        )

    parser = configparser.ConfigParser(
        inline_comment_prefixes=("#", ";"),
    )
    parser.optionxform = str   # preserve key case
    parser.read(path, encoding="utf-8")

    # [modes].run -> list of mode names
    modes: list[str] = []
    if parser.has_section("modes") and parser.has_option("modes", "run"):
        raw = parser.get("modes", "run")
        modes = [m.strip() for m in raw.split(",") if m.strip()]

    unknown = [m for m in modes if m not in TRAINING_MODES]
    if unknown:
        raise ValueError(
            f"[modes].run lists unknown mode(s): {unknown}. "
            f"Known modes: {sorted(TRAINING_MODES)}"
        )

    # [defaults]
    d = parser["defaults"] if parser.has_section("defaults") else {}
    defaults = {
        "epochs":           _coerce(d.get("epochs", "10"), int),
        "batch_size":       _coerce(d.get("batch_size", "4"), int),
        "dropout":          _coerce(d.get("dropout", "0.1"), float),
        "norm":             _parse_norm(d.get("norm", "none")),
        "seed":             _coerce(d.get("seed", "0"), int),
        "shuffle_buffer":   _coerce(d.get("shuffle_buffer", "256"), int),
        "mixed_precision":  _coerce(d.get("mixed_precision", "true"), bool),
    }

    # [lr_schedule]
    s = parser["lr_schedule"] if parser.has_section("lr_schedule") else {}
    lr_schedule = {
        "type":          s.get("type", "cosine_warmup").strip().lower(),
        "initial_lr":    _coerce(s.get("initial_lr", "1e-3"), float),
        "warmup_epochs": _coerce(s.get("warmup_epochs", "2"), int),
        "min_lr":        _coerce(s.get("min_lr", "1e-6"), float),
    }
    if lr_schedule["type"] != "cosine_warmup":
        raise ValueError(
            f"[lr_schedule].type = {lr_schedule['type']!r} is not "
            f"supported. The only schedule wired up today is "
            f"'cosine_warmup'."
        )

    # [early_stopping]
    e = parser["early_stopping"] if parser.has_section("early_stopping") else {}
    early_stopping = {
        "enabled":              _coerce(e.get("enabled", "true"), bool),
        "monitor":              e.get("monitor", "val_loss").strip(),
        "mode":                 e.get("mode", "min").strip().lower(),
        "patience":             _coerce(e.get("patience", "5"), int),
        "min_delta":            _coerce(e.get("min_delta", "1e-4"), float),
        "restore_best_weights": _coerce(e.get("restore_best_weights", "true"), bool),
    }

    # [sepconv]
    # The baseline is a different architecture trained by a different
    # script, so it gets its own section rather than a [mode.*] override
    # (which would also have to be registered in TRAINING_MODES, implying
    # train_models can train it - it cannot). Anything omitted here falls
    # back to the shared values, which is what keeps the two halves of the
    # comparison on the same footing.
    sp = parser["sepconv"] if parser.has_section("sepconv") else {}
    sepconv = {
        "epochs":        _coerce(sp.get("epochs",
                                        str(defaults["epochs"])), int),
        "batch_size":    _coerce(sp.get("batch_size",
                                        str(defaults["batch_size"])), int),
        "learning_rate": _coerce(sp.get("learning_rate",
                                        str(lr_schedule["initial_lr"])), float),
        "es_patience":   _coerce(sp.get("es_patience",
                                        str(early_stopping["patience"])), int),
        # ReduceLROnPlateau is the paper's schedule; RECONVECT uses cosine
        # warmup. This one has no counterpart to inherit from.
        "lr_patience":   _coerce(sp.get("lr_patience", "5"), int),
    }

    # [checkpointing]
    c = parser["checkpointing"] if parser.has_section("checkpointing") else {}
    checkpointing = {
        "enabled": _coerce(c.get("enabled", "true"), bool),
        "resume":  _coerce(c.get("resume", "true"), bool),
    }

    # [radar_loss] - controls the multiclass loss used by radar/OPERA
    # modes. Defaults reproduce the historical plain CCE behaviour
    # (weighting=none, gamma=0) so an unchanged training.config keeps
    # producing the same numbers until the user opts in.
    rl = parser["radar_loss"] if parser.has_section("radar_loss") else {}
    radar_loss = {
        "weighting":       rl.get("weighting", "none").strip().lower(),
        "gamma":           _coerce(rl.get("gamma", "0.0"), float),
        "alpha_max":       _coerce(rl.get("alpha_max", "100.0"), float),
        "label_smoothing": _coerce(rl.get("label_smoothing", "0.01"), float),
    }
    if radar_loss["weighting"] not in ("inverse", "median", "none"):
        raise ValueError(
            f"[radar_loss].weighting = {radar_loss['weighting']!r} is not "
            f"supported. Use 'inverse', 'median', or 'none'."
        )

    # [finetune] - Swin transformer head + AdamW for domain-adaptation
    # stage. Used when train_models.py is run with --stage finetune (or
    # the finetune leg of --stage both). The base stage ignores this
    # section entirely - it keeps the existing Adam optimizer + the
    # main [lr_schedule].
    f = parser["finetune"] if parser.has_section("finetune") else {}
    finetune = {
        "optimizer":      f.get("optimizer", "adamw").strip().lower(),
        "weight_decay":   _coerce(f.get("weight_decay", "0.01"), float),
        "initial_lr":     _coerce(f.get("initial_lr", "3e-4"), float),
        "warmup_epochs":  _coerce(f.get("warmup_epochs", "1"), int),
        "min_lr":         _coerce(f.get("min_lr", "1e-6"), float),
        "epochs":         _coerce(f.get("epochs", "20"), int),
        "window_size":    _coerce(f.get("window_size", "8"), int),
        "n_swin_blocks":  _coerce(f.get("n_swin_blocks", "2"), int),
        "num_heads":      _coerce(f.get("num_heads", "4"), int),
        "c_shared":       _coerce(f.get("c_shared", "64"), int),
        "head_dropout":   _coerce(f.get("head_dropout", "0.1"), float),
    }
    # v2 head (two-level Swin-UNet correction, optional cVAE); the
    # defaults are the values chosen inside the recommended ranges.
    v2 = finetune_head_defaults()

    def _ints(key):
        raw = f.get(key, None)
        return ([int(x) for x in str(raw).split(",")] if raw is not None
                else v2[key])
    finetune.update({
        "head":                  f.get("head", v2["head"]).strip().lower(),
        "stage_dims":            _ints("stage_dims"),
        "blocks_per_stage":      _coerce(f.get("blocks_per_stage", str(v2["blocks_per_stage"])), int),
        "stage_heads":           _ints("stage_heads"),
        "mlp_ratio":             _coerce(f.get("mlp_ratio", str(v2["mlp_ratio"])), float),
        "stem_width":            _coerce(f.get("stem_width", str(v2["stem_width"])), int),
        "expand_dim":            _coerce(f.get("expand_dim", str(v2["expand_dim"])), int),
        "drop_path":             _coerce(f.get("drop_path", str(v2["drop_path"])), float),
        "latent_res":            _coerce(f.get("latent_res", str(v2["latent_res"])), int),
        "latent_channels":       _coerce(f.get("latent_channels", str(v2["latent_channels"])), int),
        "beta":                  _coerce(f.get("beta", str(v2["beta"])), float),
        "beta_warmup_epochs":    _coerce(f.get("beta_warmup_epochs", str(v2["beta_warmup_epochs"])), int),
        "free_bits":             _coerce(f.get("free_bits", str(v2["free_bits"])), float),
        "pos_weight_cap":        _coerce(f.get("pos_weight_cap", str(v2["pos_weight_cap"])), float),
        "lightning_gamma":       _coerce(f.get("lightning_gamma", str(v2["lightning_gamma"])), float),
        "expected_class_weight": _coerce(f.get("expected_class_weight", str(v2["expected_class_weight"])), float),
        "n_members":             _coerce(f.get("n_members", str(v2["n_members"])), int),
        "es_patience":           _coerce(f.get("es_patience", "4"), int),
    })
    if finetune["head"] not in FINETUNE_HEADS:
        raise ValueError(
            f"[finetune].head = {finetune['head']!r} is not supported. "
            f"Use one of {FINETUNE_HEADS}.")
    if finetune["optimizer"] not in ("adam", "adamw"):
        raise ValueError(
            f"[finetune].optimizer = {finetune['optimizer']!r} is not "
            f"supported. Use 'adam' or 'adamw'."
        )

    # [mode.<name>] overrides
    mode_overrides: dict[str, dict] = {}
    for section in parser.sections():
        if not section.startswith("mode."):
            continue
        mode_name = section[len("mode."):]
        if mode_name not in TRAINING_MODES:
            raise ValueError(
                f"[{section}] references unknown mode {mode_name!r}. "
                f"Known modes: {sorted(TRAINING_MODES)}"
            )
        block = parser[section]
        overrides = {}
        for key, raw in block.items():
            if key == "norm":
                overrides[key] = _parse_norm(raw)
            elif key in ("epochs", "batch_size", "seed", "shuffle_buffer"):
                overrides[key] = _coerce(raw, int)
            elif key in ("dropout",):
                overrides[key] = _coerce(raw, float)
            else:
                # Unknown per-mode key — pass through as string. The caller
                # can decide whether to honour it.
                overrides[key] = raw.strip()
        mode_overrides[mode_name] = overrides

    return {
        "modes":          modes,
        "defaults":       defaults,
        "lr_schedule":    lr_schedule,
        "early_stopping": early_stopping,
        "checkpointing":  checkpointing,
        "radar_loss":     radar_loss,
        "sepconv":        sepconv,
        "finetune":       finetune,
        "mode_overrides": mode_overrides,
    }


def merge_for_mode(cfg: dict, mode: str) -> dict:
    """Return the effective hyperparameters for `mode` (defaults + override)."""
    merged = dict(cfg["defaults"])
    merged.update(cfg["mode_overrides"].get(mode, {}))
    return merged


# =============================================================================
# Cosine-with-warmup learning-rate schedule
# =============================================================================

def cosine_warmup_schedule(initial_lr: float,
                            warmup_epochs: int,
                            total_epochs: int,
                            min_lr: float):
    """Return a `(epoch, current_lr) -> next_lr` callable for the
    `tf.keras.callbacks.LearningRateScheduler`.

      - Epochs [0 .. warmup_epochs - 1]: linear ramp `min_lr -> initial_lr`.
      - Epochs [warmup_epochs .. total_epochs - 1]: half-cosine decay
        `initial_lr -> min_lr`.

    `current_lr` is required by Keras's signature but ignored — the schedule
    is a pure function of `epoch`.
    """
    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = max(1, int(total_epochs))

    def schedule(epoch: int, _current_lr: float) -> float:
        if epoch < warmup_epochs:
            # +1 so the schedule reaches `initial_lr` at the START of the
            # post-warmup phase, not partway through it.
            frac = (epoch + 1) / max(1, warmup_epochs)
            return float(min_lr + (initial_lr - min_lr) * frac)
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / denom
        progress = min(1.0, max(0.0, progress))
        return float(
            min_lr + 0.5 * (initial_lr - min_lr)
            * (1.0 + math.cos(math.pi * progress))
        )

    return schedule

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Input, Add, Conv2D, Dropout, Activation, ELU, LeakyReLU, ReLU,
    AveragePooling2D, BatchNormalization, TimeDistributed,
    LayerNormalization, Concatenate, UpSampling2D, Lambda
)
from tensorflow.keras.models import Model


# ============================================================================
# Mixed precision
# ============================================================================

def configure_tf_runtime(use_mixed_precision: bool = True) -> None:
    """Configure TensorFlow for stable GPU training on Windows.

    Three independent settings, each addressing a different failure mode
    that we've actually hit on this hardware (RTX A6000, CUDA 11.2, TF 2.x,
    Windows). Call once before model construction.

    1. **Memory growth** — by default TF allocates the entire GPU VRAM
       block on first use. A mid-batch spike during the ConvGRU backward
       pass that doesn't fit inside that block can then return a null
       pointer that TF dereferences -> Windows "fatal exception: access
       violation" (the OOM-as-segfault we kept seeing). With memory
       growth enabled, TF allocates incrementally and the same condition
       raises a clean `ResourceExhaustedError` we can react to.
    2. **JIT/XLA off** — TF 2.x sometimes auto-compiles subgraphs with
       XLA. XLA's fused kernels have a different memory profile from the
       regular runtime and can OOM in patterns we wouldn't otherwise
       hit. Explicitly disabling JIT keeps the memory footprint
       predictable.
    3. **Mixed precision** — `mixed_float16` is the recommended policy
       for A6000-class GPUs (compute on tensor cores). Off-switch is
       provided because fp16 underflow during loss spikes can also
       crash CUDA kernels on some driver versions.
    """
    # 1. Memory growth — must be set before any GPU is initialised.
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            # Already initialised — happens when train() is called twice
            # in the same process (multi-mode loop). Memory growth was
            # set on the first call and survives across iterations.
            pass
    if gpus:
        print(f"  GPU memory growth: enabled on {len(gpus)} device(s)")
    else:
        print("  GPU memory growth: no GPUs detected, CPU-only run")

    # 2. Disable XLA JIT compilation.
    tf.config.optimizer.set_jit(False)
    print("  JIT/XLA: disabled (predictable memory footprint)")

    # 3. Mixed precision policy.
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"  Mixed precision: {policy.name} "
              f"(compute={policy.compute_dtype}, "
              f"variable={policy.variable_dtype})")
    else:
        # Reset to fp32 in case a previous run in the same process left
        # the global policy at mixed_float16.
        tf.keras.mixed_precision.set_global_policy('float32')
        print("  Mixed precision: disabled (fp32 only)")


# Backwards-compat alias so the rest of the file's existing call site
# keeps working.
def setup_mixed_precision() -> None:
    configure_tf_runtime(use_mixed_precision=True)


# ============================================================================
# Custom Layers — self-contained, no c4dl dependency
# ============================================================================

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, x):
        (i_pad, j_pad) = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]],
                      'REFLECT')

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


class ConvBlock(Layer):
    def __init__(self, channels, conv_size=(3, 3), time_dist=False,
                 norm=None, stride=1, activation='relu', padding='same',
                 order=("conv", "act", "dropout", "norm"), scale_norm=False,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self._channels = channels
        self._conv_size = conv_size
        self._time_dist = time_dist
        self._norm = norm
        self._stride = stride
        self._activation = activation
        self._padding = padding
        self._order = order
        self._scale_norm = scale_norm
        self._dropout = dropout

        TD = TimeDistributed if time_dist else (lambda x: x)

        if padding == 'reflect':
            pad = tuple((s - 1) // 2 for s in conv_size)
            self.padding_layer = TD(ReflectionPadding2D(padding=pad))
        else:
            self.padding_layer = lambda x: x

        self.conv = TD(Conv2D(channels, conv_size,
                              padding='valid' if padding == 'reflect' else padding,
                              strides=(stride, stride)))

        if activation == 'leakyrelu':
            self.act = LeakyReLU(0.2)
        elif activation == 'relu':
            self.act = ReLU()
        elif activation == 'elu':
            self.act = ELU()
        else:
            self.act = Activation(activation)

        if norm == "batch":
            self.norm_layer = BatchNormalization(momentum=0.95, scale=scale_norm)
        elif norm == "layer":
            self.norm_layer = LayerNormalization(scale=scale_norm)
        else:
            self.norm_layer = lambda x: x

        if dropout > 0:
            self.dropout_layer = Dropout(dropout)
        else:
            self.dropout_layer = lambda x: x

    def call(self, x):
        for layer in self._order:
            if layer == "conv":
                x = self.conv(self.padding_layer(x))
            elif layer == "act":
                x = self.act(x)
            elif layer == "norm":
                x = self.norm_layer(x)
            elif layer == "dropout":
                x = self.dropout_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self._channels, "conv_size": self._conv_size,
            "time_dist": self._time_dist, "norm": self._norm,
            "stride": self._stride, "activation": self._activation,
            "padding": self._padding, "order": self._order,
            "scale_norm": self._scale_norm, "dropout": self._dropout,
        })
        return config


class ResBlock(Layer):
    def __init__(self, channels, **kwargs):
        self._init_kwargs = kwargs.copy()
        self._res_channels = channels
        stride = kwargs.pop("stride", 1)
        self._stride = stride
        time_dist = kwargs.get("time_dist", False)
        super().__init__()

        TD = TimeDistributed if time_dist else (lambda x: x)
        if stride > 1:
            self.pool = TD(AveragePooling2D(pool_size=(stride, stride)))
        else:
            self.pool = lambda x: x
        self.proj = TD(Conv2D(channels, kernel_size=(1, 1)))
        self.conv_block_1 = ConvBlock(channels, stride=stride, **kwargs)
        self.conv_block_2 = ConvBlock(channels, activation='leakyrelu', **kwargs)
        self.add = Add()

    def call(self, x):
        x_in = self.pool(x)
        if int(x.shape[-1]) != self._res_channels:
            x_in = self.proj(x_in)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.add([x, x_in])

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self._res_channels})
        config.update(self._init_kwargs)
        return config


class GRUResBlock(ResBlock):
    def __init__(self, channels, final_activation='sigmoid', **kwargs):
        super().__init__(channels, **kwargs)
        self._final_activation = final_activation
        self.final_act = Activation(final_activation)

    def call(self, x):
        x_in = self.proj(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.add([x, x_in])
        return self.final_act(x)

    def get_config(self):
        config = super().get_config()
        config.update({"final_activation": self._final_activation})
        return config


class ResGRU(Layer):
    def __init__(self, channels, conv_size=(3, 3),
                 return_sequences=False, time_steps=1,
                 dropout=0.0, norm=None, **kwargs):
        super().__init__(**kwargs)
        self._channels = channels
        self._conv_size = conv_size
        self._return_sequences = return_sequences
        self._time_steps = time_steps
        self._dropout = dropout
        self._norm_type = norm

        self.update_gate = GRUResBlock(channels, conv_size=conv_size,
                                       final_activation='sigmoid', padding='same',
                                       dropout=dropout, norm=norm)
        self.reset_gate = GRUResBlock(channels, conv_size=conv_size,
                                      final_activation='sigmoid', padding='same',
                                      dropout=dropout, norm=norm)
        self.output_gate = GRUResBlock(channels, conv_size=conv_size,
                                       final_activation='linear', padding='same',
                                       dropout=dropout, norm=norm)

    @tf.function
    def iterate(self, x, h):
        xh = tf.concat((x, h), axis=-1)
        z = self.update_gate(xh)
        r = self.reset_gate(xh)
        o = self.output_gate(tf.concat((x, r * h), axis=-1))
        h = z * h + (1.0 - z) * tf.math.tanh(o)
        return h

    def call(self, inputs):
        (xt, h) = inputs
        h_all = []
        for t in range(self._time_steps):
            x = xt[:, t, ...]
            h = self.iterate(x, h)
            if self._return_sequences:
                h_all.append(h)
        return tf.stack(h_all, axis=1) if self._return_sequences else h

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self._channels, "conv_size": self._conv_size,
            "return_sequences": self._return_sequences,
            "time_steps": self._time_steps,
            "dropout": self._dropout, "norm": self._norm_type,
        })
        return config


# ============================================================================
# Loss functions and metrics
# ============================================================================

class WeightedFocalLoss(tf.keras.losses.Loss):
    """Focal loss with class weighting for binary classification."""

    def __init__(self, ones_fraction=0.0106, gamma=2.0,
                 name='weighted_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ones_fraction = float(ones_fraction)
        self.gamma = float(gamma)
        zeros_fraction = 1.0 - self.ones_fraction
        self.weight_0 = 1.0 / (2.0 * zeros_fraction)
        self.weight_1 = 1.0 / (2.0 * self.ones_fraction)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weights = (1 - y_true) * self.weight_0 + y_true * self.weight_1
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)
        return tf.reduce_mean(focal_weight * weights * bce)

    def get_config(self):
        config = super().get_config()
        config.update({'ones_fraction': self.ones_fraction, 'gamma': self.gamma})
        return config


class WeightedFocalCategoricalCrossentropy(tf.keras.losses.Loss):
    """Per-class focal categorical cross-entropy for the radar multiclass head.

    Multiclass generalisation of WeightedFocalLoss. Each class k gets a
    weight alpha_k derived from its training-distribution pixel fraction
    f_k (read from opera_rainfall_fraction_<source>.json). The focal
    (1 - p_t)^gamma factor down-weights easy pixels - critical for class 0
    ("<10 mm/h") which is ~98% of the data and saturates quickly.

    weighting:
      'inverse': alpha_k = 1 / (K * max(f_k, eps))                normalised by K
      'median':  alpha_k = median(f) / max(f_k, eps)              median-frequency balancing
      'none':    alpha_k = 1                                      pure focal, no class weights

    alpha_max clips per-class weights to keep gradients stable when the
    rarest class (class 4, >=40 mm/h) is so sparse that its inverse-frequency
    weight would otherwise blow up to ~1e4.
    """

    def __init__(self, class_fractions, gamma=2.0, weighting='inverse',
                 alpha_max=100.0, label_smoothing=0.0,
                 name='weighted_focal_cce', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_fractions = [float(f) for f in class_fractions]
        self.gamma = float(gamma)
        self.weighting = str(weighting).lower()
        self.alpha_max = float(alpha_max)
        self.label_smoothing = float(label_smoothing)

        f = np.asarray(self.class_fractions, dtype=np.float64)
        f_safe = np.maximum(f, 1e-8)
        if self.weighting == 'inverse':
            alpha = 1.0 / (len(f) * f_safe)
        elif self.weighting == 'median':
            alpha = np.median(f_safe) / f_safe
        elif self.weighting == 'none':
            alpha = np.ones_like(f_safe)
        else:
            raise ValueError(
                f"weighting must be 'inverse' | 'median' | 'none', "
                f"got {self.weighting!r}"
            )
        if self.alpha_max > 0:
            alpha = np.minimum(alpha, self.alpha_max)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if self.label_smoothing > 0:
            k = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / k
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        w = tf.reduce_sum(y_true * self.alpha, axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal = tf.pow(1.0 - pt, self.gamma)
        return tf.reduce_mean(focal * w * ce)

    def get_config(self):
        config = super().get_config()
        config.update({
            'class_fractions':  self.class_fractions,
            'gamma':            self.gamma,
            'weighting':        self.weighting,
            'alpha_max':        self.alpha_max,
            'label_smoothing':  self.label_smoothing,
        })
        return config


@tf.function
def iou_metric(y_true, y_pred):
    """IoU / CSI metric."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    return intersection / (union + 1e-6)


@tf.function
def true_pos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(tf.cast(y_pred, tf.float32))
    return tf.reduce_mean(y_true * y_pred)


@tf.function
def false_pos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(tf.cast(y_pred, tf.float32))
    return tf.reduce_mean((1 - y_true) * y_pred)


@tf.function
def false_neg(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(tf.cast(y_pred, tf.float32))
    return tf.reduce_mean(y_true * (1 - y_pred))


# ============================================================================
# Model construction
# ============================================================================

def _build_radar_loss(class_fractions, radar_loss_cfg):
    """Pick the radar multiclass loss based on the [radar_loss] config.

    `radar_loss_cfg=None` or `weighting='none'` + `gamma=0` reproduces the
    historical plain CategoricalCrossentropy(label_smoothing=0.01) so
    runs that don't opt in stay numerically identical to before.
    `class_fractions` is required whenever weighting != 'none'.
    """
    cfg = radar_loss_cfg or {}
    weighting = cfg.get("weighting", "none")
    gamma = float(cfg.get("gamma", 0.0))
    smoothing = float(cfg.get("label_smoothing", 0.01))
    alpha_max = float(cfg.get("alpha_max", 100.0))

    if weighting == "none" and gamma == 0.0:
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)

    if weighting != "none" and class_fractions is None:
        raise ValueError(
            "[radar_loss].weighting != 'none' requires class_fractions. "
            "Run `python opera_rainfall_fraction.py` (with --period "
            "<label> for a labelled split) and make sure the radar mode "
            "loads the prior."
        )
    fractions = class_fractions if class_fractions is not None else [0.2] * 5
    return WeightedFocalCategoricalCrossentropy(
        class_fractions=fractions,
        gamma=gamma,
        weighting=weighting,
        alpha_max=alpha_max,
        label_smoothing=smoothing,
    )


# ============================================================================
# SepConv baseline loss: inverse-frequency weights in log_zscore space
# ============================================================================
# The paper's own weighting ("modified MSE ... more weight to higher
# values") is unpublished, so this is ours and is documented as ours.
#
# It has to exist at all: on this training split 99.815% of pixels are
# class 0, and in log_zscore space the dry point mass sits at z = -0.291
# while 10-40 mm/h spans z = +5.55..+6.72. Plain MSE is minimised by
# emitting -0.291 everywhere — a worse failure than the smoothing the
# paper reports.
#
# Weights are derived from the measured class fractions rather than
# hard-coded, so they follow the data instead of a guess. The cap keeps
# the rarest class from dominating the gradient: without it the >=40 mm/h
# bin alone would carry a weight near 4600.
SEPCONV_WEIGHT_CAP = 1000.0


def sepconv_class_weights(fractions, cap: float = SEPCONV_WEIGHT_CAP):
    """Inverse-frequency weights per rainfall class, normalised and capped.

    Args:
        fractions: per-class pixel fractions, class 0 first, as written by
            opera_rainfall_fraction.py.
        cap: maximum weight relative to the most common class.

    Returns:
        list[float] — one weight per class, class 0 normalised to 1.0.
    """
    eps = 1e-12
    inv = [1.0 / max(f, eps) for f in fractions]
    base = inv[0]                      # class 0 is always the most common
    return [min(w / base, cap) for w in inv]


def load_sepconv_class_weights(data_root, source, period=None,
                               cap: float = SEPCONV_WEIGHT_CAP):
    """Read opera_rainfall_fraction_<tag>.json and derive the weights.

    Fails loudly rather than falling back to a constant: training the
    baseline with the wrong weighting silently produces the dry-collapse
    this whole mechanism exists to prevent.
    """
    path = (Path(data_root)
            / f"opera_rainfall_fraction_{data_tag(source, period)}.json")
    if not path.is_file():
        raise SystemExit(
            f"Class fractions not found: {path}\n"
            f"The SepConv baseline's loss weighting is derived from them.\n"
            f"    python opera_rainfall_fraction.py"
        )
    with open(path, encoding="utf-8") as fh:
        fractions = json.load(fh)["fractions"]
    return sepconv_class_weights(fractions, cap=cap)


def build_coalition_model(input_shapes, label_type, past_timesteps=3,
                          future_timesteps=3, dropout=0, norm=None,
                          ones_fraction=0.0106,
                          class_fractions=None, radar_loss_cfg=None):
    """Build the COALITION encoder-forecaster model dynamically.

    The model architecture adapts to whatever inputs the dataset provides.
    Input shapes are read from the dataset metadata so adding or removing
    input groups requires no code changes here.

    Args:
        input_shapes: dict from metadata.json["input_shapes"], e.g.
            {"past_hr": [3, 256, 256, 1], "past_mr": [3, 128, 128, 6]}
            (HR = 1 km / 256 px, MR = 2 km / 128 px — see the resolution
            note at the top of create_datasets.py)
        label_type: "lightning" or "radar" (determines loss + output head)
        past_timesteps: number of input timesteps
        future_timesteps: number of output timesteps
        dropout: dropout rate
        norm: normalization type (None, "batch", "layer")
        ones_fraction: lightning occurrence rate (for focal loss)

    Returns:
        compiled Keras Model
    """
    # Determine the highest resolution (HR) to compute shape divisors
    max_res = max(shape[1] for shape in input_shapes.values())

    # Build Input layers and assign to resolution branches
    all_inputs = []
    xt = {}
    input_divisors = set()

    for name in sorted(input_shapes.keys()):
        shape = input_shapes[name]  # [T, H, W, C]
        res = shape[1]
        channels = shape[-1]
        divisor = max_res // res  # 1 for HR, 2 for MR

        inp = Input(shape=(past_timesteps, res, res, channels), name=name)
        all_inputs.append(inp)
        xt[divisor] = Concatenate(axis=-1)([xt[divisor], inp]) \
            if divisor in xt else inp
        input_divisors.add(divisor)

    print(f"  Dynamic model: {len(all_inputs)} inputs, "
          f"divisors={sorted(input_divisors)}, "
          f"max_res={max_res}")

    block_channels = [32, 64, 128]

    # ==================== ENCODER ====================
    intermediate = []  # skip connections for decoder

    for (i, channels) in enumerate(block_channels):
        s = 2 ** i  # 1, 2, 4

        # Merge branches when their resolution matches
        if (i > 0) and s in input_divisors:
            if 1 in xt:
                xt[1] = Concatenate(axis=-1)([xt[1], xt[s]])
            else:
                xt[1] = xt[s]
            del xt[s]

        for s_key in list(xt.keys()):
            stride = 2 if (s_key == 1) else 1
            xt[s_key] = ResBlock(channels, time_dist=True, stride=stride,
                                 dropout=dropout, norm=norm)(xt[s_key])

            initial_state = Lambda(
                lambda y: tf.zeros_like(y[:, 0, ...])
            )(xt[s_key])

            xt[s_key] = ResGRU(channels, return_sequences=True,
                               time_steps=past_timesteps,
                               dropout=dropout, norm=norm
                               )([xt[s_key], initial_state])

        # Save skip connection: last timestep through a ConvBlock
        intermediate.append(ConvBlock(channels)(xt[1][:, -1, ...]))

    encoded = xt[1]  # (batch, T_past, H_deep, W_deep, C_deep)

    # ==================== FORECASTER (DECODER) ====================
    # No future branch → start from zeros
    xt_dec = Lambda(lambda y: tf.zeros_like(
        tf.repeat(y[:, :1, ...], future_timesteps, axis=1)
    ))(encoded)

    for (i, channels) in reversed(list(enumerate(block_channels))):
        xt_dec = ResGRU(channels, return_sequences=True,
                        time_steps=future_timesteps,
                        dropout=dropout, norm=norm
                        )([xt_dec, intermediate[i]])
        xt_dec = TimeDistributed(
            UpSampling2D(interpolation='bilinear')
        )(xt_dec)
        xt_dec = ResBlock(block_channels[max(i - 1, 0)], time_dist=True,
                          dropout=dropout, norm=norm)(xt_dec)

    # Named graft point. `xt_dec` is the encoder-forecaster's final
    # feature tensor of shape (B, F, H, W, C_deep). The fine-tune stage
    # (train_finetune below) looks up this layer by name to attach a
    # Swin transformer head on top of the frozen backbone.
    xt_dec = Lambda(lambda x: x, name='backbone_output')(xt_dec)

    # ==================== OUTPUT HEAD ====================
    if label_type == "lightning":
        num_outputs = 1
        # Use float32 for numerical stability with mixed precision
        final_conv = Conv2D(num_outputs, kernel_size=(1, 1),
                            activation='sigmoid', dtype='float32')
        loss = WeightedFocalLoss(ones_fraction=ones_fraction, gamma=2.0)
        metrics = [iou_metric, true_pos, false_pos, false_neg]
    else:  # radar multiclass
        num_outputs = 5
        final_conv = Conv2D(num_outputs, kernel_size=(1, 1),
                            activation='softmax', dtype='float32')
        loss = _build_radar_loss(class_fractions, radar_loss_cfg)
        metrics = ['accuracy']

    seq_out = TimeDistributed(final_conv)(xt_dec)

    model = Model(inputs=all_inputs, outputs=[seq_out])

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


# ============================================================================
# Swin transformer head (domain-adaptation fine-tune)
# ============================================================================
#
# Sits between the frozen backbone's `backbone_output` (shape
# (B, F, H, W, C_deep) = (B, 3, 256, 256, 32) for the default model) and
# a fresh per-lead-time output head. The backbone stays frozen; only
# these layers train.
#
# Architecture (option (a) from the design discussion):
#
#   1. Collapse the future-time axis into shared spatial features:
#         (B, F, H, W, C) -> (B, H, W, F*C) -> Conv 1x1 -> (B, H, W, c_shared)
#   2. N Swin blocks (default 2): windowed self-attention with 8x8
#      windows. Block 0 uses regular partition; block 1 uses cyclic
#      shift by `window_size // 2` for cross-window communication.
#      No relative-position bias and no attention mask on the shifted
#      wrap-around (lite variant - the per-window MHA carries enough
#      signal at this depth).
#   3. F independent lightweight projection heads, each producing the
#      prediction for one future step. Output stacked on axis=1 so the
#      shape matches the base model exactly (B, F, H, W, num_outputs).
#
# At 256x256 with window_size=8 we get 1024 windows of 64 tokens, so
# each MHA call sees 64 queries x 64 keys - cheap compared to a global
# self-attention over 65k tokens.


def _window_partition(x, window_size):
    """(B, H, W, C) -> (B*nW, ws*ws, C) where nW = (H/ws)*(W/ws)."""
    shape = tf.shape(x)
    B, H, W = shape[0], shape[1], shape[2]
    C = x.shape[-1]
    ws = window_size
    x = tf.reshape(x, [B, H // ws, ws, W // ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])      # (B, H/ws, W/ws, ws, ws, C)
    x = tf.reshape(x, [-1, ws * ws, C])
    return x


def _window_reverse(x_windows, H, W, window_size, B):
    """(B*nW, ws*ws, C) -> (B, H, W, C)."""
    ws = window_size
    C = x_windows.shape[-1]
    x = tf.reshape(x_windows, [B, H // ws, W // ws, ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])      # (B, H/ws, ws, W/ws, ws, C)
    x = tf.reshape(x, [B, H, W, C])
    return x


class SwinBlock(tf.keras.layers.Layer):
    """Single Swin transformer block (W-MSA or SW-MSA + MLP).

    Operates on `(B, H, W, C)` feature maps with H, W divisible by
    `window_size`. When `shift_size > 0`, the input is cyclic-shifted
    by `(shift, shift)` before window partitioning so different windows
    see each other across blocks. The shifted-wrap attention mask is
    omitted (lite variant); the impact is small at 256x256 with 8x8
    windows because only one window-row and one window-column see a
    wrap-around discontinuity.
    """

    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=2.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

    def build(self, input_shape):
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.dim // self.num_heads),
            dropout=self.dropout,
        )
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        hidden = int(self.dim * self.mlp_ratio)
        self.mlp_dense1 = tf.keras.layers.Dense(hidden, activation='gelu')
        self.mlp_drop1 = tf.keras.layers.Dropout(self.dropout)
        self.mlp_dense2 = tf.keras.layers.Dense(self.dim)
        self.mlp_drop2 = tf.keras.layers.Dropout(self.dropout)
        super().build(input_shape)

    def call(self, x, training=None):
        # x: (B, H, W, C)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = tf.roll(x, shift=(-self.shift_size, -self.shift_size),
                        axis=(1, 2))

        x_windows = _window_partition(x, self.window_size)
        attn_out = self.attn(x_windows, x_windows, training=training)
        x = _window_reverse(attn_out, H, W, self.window_size, B)

        if self.shift_size > 0:
            x = tf.roll(x, shift=(self.shift_size, self.shift_size),
                        axis=(1, 2))

        x = shortcut + x

        # MLP
        h = self.norm2(x)
        h = self.mlp_dense1(h)
        h = self.mlp_drop1(h, training=training)
        h = self.mlp_dense2(h)
        h = self.mlp_drop2(h, training=training)
        return x + h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "dim":          self.dim,
            "num_heads":    self.num_heads,
            "window_size":  self.window_size,
            "shift_size":   self.shift_size,
            "mlp_ratio":    self.mlp_ratio,
            "dropout":      self.dropout,
        })
        return cfg


def build_swin_head(backbone_features, future_timesteps, num_outputs,
                    label_type, window_size=8, n_blocks=2, num_heads=4,
                    c_shared=64, dropout=0.0):
    """Swin head + 3 per-lead-time projections.

    backbone_features: (B, F, H, W, C_deep), output of the frozen backbone.
    Returns a Keras tensor with shape (B, F, H, W, num_outputs) matching
    the base model's output contract.
    """
    F = backbone_features.shape[1]
    H = backbone_features.shape[2]
    W = backbone_features.shape[3]
    C = backbone_features.shape[-1]

    # Step 1: collapse the F axis into channels and project to c_shared.
    # Permute non-batch axes (F, H, W, C) -> (H, W, F, C); 1-indexed in
    # Keras Permute means (2, 3, 1, 4).
    x = tf.keras.layers.Permute((2, 3, 1, 4),
                                 name='collapse_F_perm')(backbone_features)
    x = tf.keras.layers.Reshape((H, W, F * C),
                                 name='collapse_F_flat')(x)
    x = tf.keras.layers.Conv2D(
        c_shared, 1, activation='gelu',
        name='backbone_to_shared',
    )(x)

    # Step 2: stack of Swin blocks. Odd blocks shift to enable
    # cross-window flow. Even blocks use the regular partition.
    half_window = window_size // 2
    for i in range(n_blocks):
        shift = half_window if (i % 2 == 1) else 0
        x = SwinBlock(
            dim=c_shared,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift,
            dropout=dropout,
            name=f'swin_block_{i}',
        )(x)

    # Step 3: F independent projection heads (option (a)). Each head is
    # LayerNorm -> Conv 1x1 (gelu) -> Conv 1x1 (final activation, float32).
    final_activation = 'sigmoid' if label_type == 'lightning' else 'softmax'
    outs = []
    for t in range(future_timesteps):
        h = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name=f'head_norm_t{t}',
        )(x)
        h = tf.keras.layers.Conv2D(
            c_shared, 1, activation='gelu',
            name=f'head_hidden_t{t}',
        )(h)
        h = tf.keras.layers.Conv2D(
            num_outputs, 1, activation=final_activation,
            dtype='float32',          # numerical stability under mixed prec
            name=f'head_out_t{t}',
        )(h)
        outs.append(h)

    # Stack the F per-step predictions back onto axis=1 -> (B, F, H, W, num_outputs)
    seq_out = tf.keras.layers.Lambda(
        lambda lst: tf.stack(lst, axis=1),
        name='stack_lead_times',
    )(outs)

    return seq_out


# ============================================================================
# Fine-tune head v2: two-level Swin-UNet correction on the frozen backbone,
# with an optional conditional VAE (rain track)
# ============================================================================
#
# The head reads the frozen backbone's decoder features F (B, L, 256, 256,
# 32) and its logits L_b, and writes
#
#     logits_l = s_l * L_b[l] + head_l(x)          for every lead l
#
# with the four per-lead 1x1 heads initialised at zero and s_l at one, so
# at step 0 the model reproduces the frozen backbone exactly and training
# can only move away from it where the loss improves. The head learns a
# correction field, not the readout.
#
# Path (the shapes are for one batch element):
#   F collapsed over time     (256, 256, 128)
#   patch embed  k=2 s=2 + LN (128, 128, 64)   + MR stem (last 2 MR frames)
#   Swin stage 1, 2 blocks     (128, 128, 64)   8x8 windows, 4 heads
#   patch merge 2x2 -> LN -> linear  (64, 64, 128) = c
#   [cVAE] gated FiLM of c with a latent z (16, 16, 32)
#   Swin stage 2, 2 blocks     (64, 64, 128)    8x8 windows, 8 heads
#   linear 128->64, bilinear x2, + stage-1 skip, conv block   (128, 128, 64)
#   linear 64->128, pixel shuffle x2   (256, 256, 32)   + target stem
#   4 per-lead 1x1 heads, zero-init
#
# Two raw stems bypass the backbone: the last two frames of the target
# field at 1 km (lightning: the HR lightning channels; rain: OPERA rain
# rate and reflectivity upsampled 2 -> 1 km) join at the output
# resolution, so persistence is one linear step away, and the last two MR
# frames join at 2 km.
#
# The Swin blocks here carry the relative position bias and the shifted-
# window attention mask of the paper; the legacy `build_swin_head` keeps
# its lite variant so the models saved with it still load.
#
# The conditional VAE (rain track): a prior network p(z | c) and a
# posterior network q(z | c, y) both output a Gaussian over a (16, 16, 32)
# latent. Training samples z from q, inference from p (its mean by
# default; `set_member(k)` draws member k from a fixed seed). z is
# upsampled to c's grid and modulates it through a gated FiLM,
# c <- c + sigmoid(g(c)) * (gamma(z) * c + beta(z)), with gamma and beta
# zero-initialised so the latent starts inert. The loss adds
# beta_t * mean_i max(KL_i, free_bits) with KL_i the per-dimension
# KL(q || p) averaged over the batch, beta_t warmed up linearly.


class DropPath(tf.keras.layers.Layer):
    """Stochastic depth: drops a residual branch for a whole sample with
    probability `rate` in training, rescaling the kept ones."""

    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, x, training=None):
        if not training or self.rate <= 0.0:
            return x
        keep = 1.0 - self.rate
        shape = tf.concat([tf.shape(x)[:1],
                           tf.ones([tf.rank(x) - 1], dtype=tf.int32)], axis=0)
        mask = tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))
        return x / tf.cast(keep, x.dtype) * mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"rate": self.rate})
        return cfg


class WindowAttention(tf.keras.layers.Layer):
    """Multi-head self-attention inside (window_size x window_size)
    windows with the learned relative position bias of Swin and, for
    the shifted blocks, the additive mask that keeps tokens from
    attending across the cyclic wrap-around seam."""

    def __init__(self, dim, num_heads, window_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.dropout = float(dropout)
        self.head_dim = max(1, self.dim // self.num_heads)
        self.scale = self.head_dim ** -0.5

    def build(self, input_shape):
        ws = self.window_size
        self.qkv = tf.keras.layers.Dense(3 * self.num_heads * self.head_dim,
                                         name="qkv")
        self.proj = tf.keras.layers.Dense(self.dim, name="proj")
        self.attn_drop = tf.keras.layers.Dropout(self.dropout)
        # (2ws-1)^2 x heads table, indexed by the relative offset of every
        # (query, key) pair of the window.
        self.rel_pos_bias = self.add_weight(
            name="rel_pos_bias",
            shape=((2 * ws - 1) ** 2, self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True, dtype="float32")
        coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws),
                                      indexing="ij")).reshape(2, -1)     # (2, N)
        rel = coords[:, :, None] - coords[:, None, :]                     # (2, N, N)
        rel = rel.transpose(1, 2, 0) + (ws - 1)
        index = rel[:, :, 0] * (2 * ws - 1) + rel[:, :, 1]               # (N, N)
        self.rel_index = tf.constant(index.reshape(-1), dtype=tf.int32)
        super().build(input_shape)

    def call(self, x_windows, mask=None, training=None):
        # x_windows: (nW*B, N, C); mask: (nW, N, N) additive or None
        shape = tf.shape(x_windows)
        b_, n = shape[0], shape[1]
        qkv = self.qkv(x_windows)
        qkv = tf.reshape(qkv, [b_, n, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])           # (3, nW*B, h, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q * tf.cast(self.scale, q.dtype), k, transpose_b=True)
        n_static = self.window_size * self.window_size
        bias = tf.gather(self.rel_pos_bias, self.rel_index)
        bias = tf.transpose(tf.reshape(bias, [n_static, n_static, self.num_heads]),
                            [2, 0, 1])                       # (h, N, N)
        attn = attn + tf.cast(bias, attn.dtype)[None]
        if mask is not None:
            n_w = tf.shape(mask)[0]
            attn = tf.reshape(attn, [b_ // n_w, n_w, self.num_heads, n, n])
            attn = attn + tf.cast(mask, attn.dtype)[None, :, None]
            attn = tf.reshape(attn, [b_, self.num_heads, n, n])
        attn = tf.nn.softmax(tf.cast(attn, tf.float32), axis=-1)
        attn = tf.cast(attn, v.dtype)
        attn = self.attn_drop(attn, training=training)
        out = tf.matmul(attn, v)                            # (nW*B, h, N, d)
        out = tf.reshape(tf.transpose(out, [0, 2, 1, 3]),
                         [b_, n, self.num_heads * self.head_dim])
        return self.proj(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "num_heads": self.num_heads,
                    "window_size": self.window_size, "dropout": self.dropout})
        return cfg


def _shift_attention_mask(H, W, window_size, shift_size):
    """The (nW, N, N) additive mask of a shifted Swin block: -100 between
    tokens that come from different regions of the cyclic shift, 0
    otherwise, so a window never mixes the two sides of the seam."""
    ws, s = window_size, shift_size
    img = np.zeros((1, H, W, 1), dtype=np.float32)
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
        for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            img[:, hs, wsl, :] = cnt
            cnt += 1
    win = img.reshape(1, H // ws, ws, W // ws, ws, 1).transpose(0, 1, 3, 2, 4, 5)
    win = win.reshape(-1, ws * ws)
    mask = win[:, None, :] - win[:, :, None]
    return np.where(mask != 0, -100.0, 0.0).astype(np.float32)


class SwinBlockV2(tf.keras.layers.Layer):
    """Swin block with the relative position bias, the shift mask and
    stochastic depth: pre-norm windowed attention + pre-norm MLP, both
    residual."""

    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=2.0, dropout=0.0, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.drop_path_rate = float(drop_path)

    def build(self, input_shape):
        H, W = int(input_shape[1]), int(input_shape[2])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn = WindowAttention(self.dim, self.num_heads, self.window_size,
                                    dropout=self.dropout, name="attn")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        hidden = int(self.dim * self.mlp_ratio)
        self.mlp_dense1 = tf.keras.layers.Dense(hidden, activation="gelu", name="mlp1")
        self.mlp_dense2 = tf.keras.layers.Dense(self.dim, name="mlp2")
        self.drop_path = DropPath(self.drop_path_rate)
        self.mask = (tf.constant(_shift_attention_mask(H, W, self.window_size,
                                                       self.shift_size))
                     if self.shift_size > 0 else None)
        super().build(input_shape)

    def call(self, x, training=None):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        x_windows = _window_partition(x, self.window_size)
        x_windows = self.attn(x_windows, mask=self.mask, training=training)
        x = _window_reverse(x_windows, H, W, self.window_size, B)
        if self.shift_size > 0:
            x = tf.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        x = shortcut + self.drop_path(x, training=training)
        h = self.mlp_dense2(self.mlp_dense1(self.norm2(x)))
        return x + self.drop_path(h, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "num_heads": self.num_heads,
                    "window_size": self.window_size, "shift_size": self.shift_size,
                    "mlp_ratio": self.mlp_ratio, "dropout": self.dropout,
                    "drop_path": self.drop_path_rate})
        return cfg


def _conv_block(filters, name):
    """3x3 conv + LayerNorm + GELU, the small conv block of the stems and
    the decoder."""
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv"),
        tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln"),
        tf.keras.layers.Activation("gelu", name=f"{name}_act"),
    ], name=name)


class PosWeightedBCE(tf.keras.losses.Loss):
    """Binary cross-entropy with a positive-class weight, and an optional
    focal factor (gamma > 0) as the fallback when false alarms dominate.
    pos_weight = min(N_neg / N_pos, cap) from the training-set frequency."""

    def __init__(self, ones_fraction, cap=30.0, gamma=0.0,
                 name="pos_weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ones_fraction = float(ones_fraction)
        self.cap = float(cap)
        self.gamma = float(gamma)
        self.pos_weight = float(min((1.0 - self.ones_fraction)
                                    / max(self.ones_fraction, 1e-6), self.cap))

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        bce = -(self.pos_weight * y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        if self.gamma > 0:
            pt = tf.where(y_true > 0.5, y_pred, 1.0 - y_pred)
            bce = tf.pow(1.0 - pt, self.gamma) * bce
        return tf.reduce_mean(bce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ones_fraction": self.ones_fraction, "cap": self.cap,
                    "gamma": self.gamma})
        return cfg


FINETUNE_HEADS = ("swin_legacy", "swin_unet", "swin_unet_cvae")


def finetune_suffix(finetuned) -> str:
    """Artefact suffix of a fine-tuned variant: "_finetuned" for the
    deterministic heads (finetuned=True), "_finetuned_cvae" for the
    conditional VAE (finetuned="cvae"), "" for a base model. Both rain
    heads can then sit in one models folder."""
    if finetuned == "cvae":
        return "_finetuned_cvae"
    return "_finetuned" if finetuned else ""


def finetune_ckpt_stem(run_tag: str, finetuned) -> str:
    """Stem of the per-epoch checkpoint of a fine-tune run."""
    return f"{run_tag}_finetune{'_cvae' if finetuned == 'cvae' else ''}_latest"


def finetune_head_defaults() -> dict:
    """Hyperparameters of the v2 head, the values chosen inside the
    recommended ranges; [finetune] in training.config overrides them."""
    return {
        "head":                 "swin_unet",
        "stage_dims":           [64, 128],
        "blocks_per_stage":     2,
        "stage_heads":          [4, 8],
        "window_size":          8,
        "mlp_ratio":            2.0,
        "stem_width":           16,
        "expand_dim":           32,
        "drop_path":            0.1,
        "latent_res":           16,
        "latent_channels":      32,
        "beta":                 0.02,
        "beta_warmup_epochs":   2,
        "free_bits":            0.2,
        "pos_weight_cap":       30.0,
        "lightning_gamma":      0.0,
        "expected_class_weight": 0.2,
        "lead_weights":         None,
        "n_members":            3,
    }


class FinetuneHeadV2(tf.keras.layers.Layer):
    """The correction head. call(features, logits_b, inputs, y=None,
    training=None) -> probabilities (B, L, H, W, n_out). `y` is the
    future target, read by the posterior network of the cVAE variant
    in training only."""

    def __init__(self, hp: dict, label_type: str, num_outputs: int,
                 future_timesteps: int, hr_channels: int, mr_channels: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.hp = dict(hp)
        self.label_type = label_type
        self.num_outputs = int(num_outputs)
        self.L = int(future_timesteps)
        self.hr_channels = int(hr_channels)
        self.mr_channels = int(mr_channels)
        self.cvae = self.hp["head"] == "swin_unet_cvae"
        d1, d2 = [int(v) for v in self.hp["stage_dims"]]
        h1, h2 = [int(v) for v in self.hp["stage_heads"]]
        ws = int(self.hp["window_size"])
        nb = int(self.hp["blocks_per_stage"])
        stem = int(self.hp["stem_width"])
        ex = int(self.hp["expand_dim"])
        dp = float(self.hp["drop_path"])
        mlp = float(self.hp["mlp_ratio"])
        # --- level 1
        self.embed = tf.keras.layers.Conv2D(d1, 2, strides=2, name="patch_embed")
        self.embed_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="patch_embed_ln")
        self.mr_stem = _conv_block(stem, "mr_stem")
        self.level1_fuse = tf.keras.layers.Dense(d1, name="level1_fuse")
        self.stage1 = [SwinBlockV2(d1, h1, ws, ws // 2 if i % 2 else 0,
                                   mlp_ratio=mlp, drop_path=dp, name=f"stage1_block{i}")
                       for i in range(nb)]
        # --- merge to level 2
        self.merge_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="merge_ln")
        self.merge = tf.keras.layers.Dense(d2, name="merge_linear")
        # --- cVAE
        if self.cvae:
            lc = int(self.hp["latent_channels"])
            self.prior_net = tf.keras.Sequential([
                tf.keras.layers.Conv2D(d2, 3, strides=2, padding="same", activation="gelu"),
                tf.keras.layers.Conv2D(d2, 3, strides=2, padding="same", activation="gelu"),
                tf.keras.layers.Conv2D(2 * lc, 1, dtype="float32"),
            ], name="prior_net")
            self.post_y_stem = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="gelu"),
                tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="gelu"),
            ], name="posterior_y_stem")
            self.post_net = tf.keras.Sequential([
                tf.keras.layers.Conv2D(d2, 3, strides=2, padding="same", activation="gelu"),
                tf.keras.layers.Conv2D(d2, 3, strides=2, padding="same", activation="gelu"),
                tf.keras.layers.Conv2D(2 * lc, 1, dtype="float32"),
            ], name="posterior_net")
            self.film_gamma = tf.keras.layers.Conv2D(
                d2, 1, kernel_initializer="zeros", bias_initializer="zeros", name="film_gamma")
            self.film_beta = tf.keras.layers.Conv2D(
                d2, 1, kernel_initializer="zeros", bias_initializer="zeros", name="film_beta")
            # the gate starts mostly closed (sigmoid(-2) = 0.12) and opens
            # where the latent turns out to help
            self.film_gate = tf.keras.layers.Conv2D(
                1, 1, kernel_initializer="zeros",
                bias_initializer=tf.keras.initializers.Constant(-2.0), name="film_gate")
        # --- level 2
        self.stage2 = [SwinBlockV2(d2, h2, ws, ws // 2 if i % 2 else 0,
                                   mlp_ratio=mlp, drop_path=dp, name=f"stage2_block{i}")
                       for i in range(nb)]
        # --- decoder
        self.dec_proj = tf.keras.layers.Dense(d1, name="dec_proj")
        self.dec_up = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="dec_up")
        self.dec_block = _conv_block(d1, "dec_block")
        self.expand = tf.keras.layers.Dense(ex * 4, name="patch_expand")
        self.target_stem = _conv_block(stem, "target_stem")
        self.out_block = _conv_block(ex, "out_block")
        self.heads = [tf.keras.layers.Conv2D(
            self.num_outputs, 1, kernel_initializer="zeros",
            bias_initializer="zeros", dtype="float32", name=f"head_t{t}")
            for t in range(self.L)]
        self.lb_scale = self.add_weight(
            name="lb_scale", shape=(self.L,), initializer="ones",
            trainable=True, dtype="float32")
        self.last_stats: dict = {}

    # -- pieces -----------------------------------------------------------
    def _target_frames(self, inputs):
        """The last two raw frames of the target field at 1 km: the HR
        lightning channels (lightning), or OPERA rain rate and
        reflectivity from the MR group upsampled x2 (rain)."""
        if self.label_type == "lightning":
            x = inputs["past_hr"][:, -2:, :, :, :min(3, self.hr_channels)]
        else:
            x = inputs["past_mr"][:, -2:, :, :, :min(2, self.mr_channels)]
        shape = tf.shape(x)
        x = tf.transpose(x, [0, 2, 3, 1, 4])
        x = tf.reshape(x, [shape[0], shape[2], shape[3], -1])
        if self.label_type != "lightning":
            x = tf.image.resize(x, [shape[2] * 2, shape[3] * 2], method="bilinear")
        return x

    def _mr_frames(self, inputs):
        x = inputs["past_mr"][:, -2:]
        shape = tf.shape(x)
        x = tf.transpose(x, [0, 2, 3, 1, 4])
        return tf.reshape(x, [shape[0], shape[2], shape[3], -1])

    @staticmethod
    def _gaussian(params):
        mu, log_sigma = tf.split(tf.cast(params, tf.float32), 2, axis=-1)
        log_sigma = tf.clip_by_value(log_sigma, -7.0, 4.0)
        return mu, log_sigma

    def latent(self, c, y=None, member=None):
        """(z, stats): the prior and, when `y` is given, the posterior
        over the latent; z sampled from q when y is given (training),
        else the prior mean, or the prior's member `member` (>= 0) drawn
        from a fixed seed."""
        mu_p, ls_p = self._gaussian(self.prior_net(c))
        stats = {"mu_p": mu_p, "log_sigma_p": ls_p}
        if y is not None:
            shape = tf.shape(y)
            yy = tf.transpose(y, [0, 2, 3, 1, 4])
            yy = tf.reshape(yy, [shape[0], shape[2], shape[3], -1])
            yy = self.post_y_stem(tf.cast(yy, c.dtype))
            mu_q, ls_q = self._gaussian(self.post_net(tf.concat([c, yy], axis=-1)))
            eps = tf.random.normal(tf.shape(mu_q))
            z = mu_q + tf.exp(ls_q) * eps
            stats.update({"mu_q": mu_q, "log_sigma_q": ls_q})
        elif member is not None:
            def _draw():
                seed = tf.stack([tf.cast(member, tf.int32), tf.constant(7, tf.int32)])
                return mu_p + tf.exp(ls_p) * tf.random.stateless_normal(tf.shape(mu_p), seed=seed)
            z = tf.cond(tf.cast(member, tf.int32) >= 0, _draw, lambda: mu_p)
        else:
            z = mu_p
        return z, stats

    def film(self, c, z):
        target = tf.shape(c)[1:3]
        zz = tf.image.resize(z, target, method="bilinear")
        zz = tf.cast(zz, c.dtype)
        gate = tf.sigmoid(self.film_gate(c))
        mod = self.film_gamma(zz) * c + self.film_beta(zz)
        self.last_stats["gate_mean"] = tf.reduce_mean(tf.cast(gate, tf.float32))
        self.last_stats["gate_std"] = tf.math.reduce_std(tf.cast(gate, tf.float32))
        return c + gate * mod

    # -- forward ----------------------------------------------------------
    def call(self, features, logits_b, inputs, y=None, training=None, member=None):
        shape = tf.shape(features)
        x = tf.transpose(features, [0, 2, 3, 1, 4])
        x = tf.reshape(x, [shape[0], shape[2], shape[3], -1])
        x = self.embed_norm(self.embed(x))                            # (B,128,128,d1)
        mr = self.mr_stem(tf.cast(self._mr_frames(inputs), x.dtype))
        x = self.level1_fuse(tf.concat([x, mr], axis=-1))
        for blk in self.stage1:
            x = blk(x, training=training)
        s1 = x
        c = tf.nn.space_to_depth(x, 2)                                 # (B,64,64,4*d1)
        c = self.merge(self.merge_norm(c))
        self.last_stats = {}
        stats = {}
        if self.cvae:
            z, stats = self.latent(c, y=y, member=member)
            c = self.film(c, z)
        for blk in self.stage2:
            c = blk(c, training=training)
        d = tf.cast(self.dec_up(self.dec_proj(c)), s1.dtype)   # bilinear resize returns float32
        d = self.dec_block(d + s1)
        e = tf.nn.depth_to_space(self.expand(d), 2)                    # (B,256,256,ex)
        t = self.target_stem(tf.cast(self._target_frames(inputs), e.dtype))
        e = self.out_block(tf.concat([e, t], axis=-1))
        logits_b = tf.cast(logits_b, tf.float32)
        outs, ratios = [], []
        for l in range(self.L):
            corr = self.heads[l](e)
            lb = tf.cast(self.lb_scale[l], tf.float32) * logits_b[:, l]
            outs.append(lb + corr)
            ratios.append(tf.norm(corr) / (tf.norm(logits_b[:, l]) + 1e-6))
        logits = tf.stack(outs, axis=1)
        self.last_stats["correction_ratio"] = tf.stack(ratios)
        self.last_stats.update(stats)
        if self.label_type == "lightning":
            return tf.sigmoid(logits)
        return tf.nn.softmax(logits, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hp": self.hp, "label_type": self.label_type,
                    "num_outputs": self.num_outputs, "future_timesteps": self.L,
                    "hr_channels": self.hr_channels, "mr_channels": self.mr_channels})
        return cfg


class _Confusion(tf.keras.metrics.Metric):
    """Pooled TP / FP / FN of a binary event; result() is the CSI."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)

    def update_state(self, gt, pred, sample_weight=None):
        gt = tf.cast(gt, tf.bool)
        pred = tf.cast(pred, tf.bool)
        self.tp.assign_add(tf.reduce_sum(tf.cast(gt & pred, tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(~gt & pred, tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(gt & ~pred, tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-7)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)


class _Fss(tf.keras.metrics.Metric):
    """Fractions skill score at one neighbourhood: 1 - sum((Pf-Po)^2) /
    (sum(Pf^2) + sum(Po^2)) over the neighbourhood fractions."""

    def __init__(self, name, window, **kwargs):
        super().__init__(name=name, **kwargs)
        self.window = int(window)
        self.num = self.add_weight(name="num", initializer="zeros", dtype=tf.float32)
        self.den = self.add_weight(name="den", initializer="zeros", dtype=tf.float32)

    def update_state(self, gt, pred, sample_weight=None):
        g = tf.cast(gt, tf.float32)[..., None]
        p = tf.cast(pred, tf.float32)[..., None]
        fg = tf.nn.avg_pool2d(g, self.window, 1, "SAME")
        fp = tf.nn.avg_pool2d(p, self.window, 1, "SAME")
        self.num.assign_add(tf.reduce_sum(tf.square(fp - fg)))
        self.den.assign_add(tf.reduce_sum(tf.square(fp)) + tf.reduce_sum(tf.square(fg)))

    def result(self):
        return 1.0 - self.num / (self.den + 1e-7)

    def reset_state(self):
        self.num.assign(0.0)
        self.den.assign(0.0)


class _MeanOf(tf.keras.metrics.Metric):
    """The mean of other metrics' results (the early-stopping monitor)."""

    def __init__(self, name, parts, **kwargs):
        super().__init__(name=name, **kwargs)
        self.parts = list(parts)

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        return tf.add_n([p.result() for p in self.parts]) / float(len(self.parts))

    def reset_state(self):
        pass


class FinetuneModelV2(tf.keras.Model):
    """Frozen backbone + FinetuneHeadV2, with the losses, the beta
    schedule and the monitoring of the fine-tune stage.

    Logged per epoch (train and val): loss, l_forecast, l_kl,
    csi_t<k> per lead (lightning: p >= 0.5; rain: class >= 1),
    csi_mid (mean of the two middle leads, the early-stopping monitor),
    brier (lightning), fss3 / fss9 per lead, adjacent_frac (rain: the
    share of wrong pixels that are off by one class), persistence_csi
    per lead (lightning, the last occurrence frame as the forecast),
    corr_ratio_t<k> (|correction| / |L_b|), lb_scale_t<k>, grad_norm_stage2,
    and for the cVAE kl, active_units, gate_mean, gate_std and, on
    validation, l_forecast_q (z from the posterior, the ceiling)."""

    def __init__(self, feature_model, head, label_type, forecast_loss,
                 hp, **kwargs):
        super().__init__(**kwargs)
        self.feature_model = feature_model
        self.head = head
        self.label_type = label_type
        self.forecast_loss = forecast_loss
        self.hp = dict(hp)
        self.cvae = head.cvae
        L = head.L
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="beta")
        self.member = tf.Variable(-1, trainable=False, dtype=tf.int32, name="member")
        w = self.hp.get("lead_weights") or [1.0] * L
        self.lead_weights = tf.constant([float(v) for v in w], dtype=tf.float32)
        self.expected_class_weight = float(self.hp.get("expected_class_weight", 0.0)
                                           if label_type != "lightning" else 0.0)
        self.free_bits = float(self.hp.get("free_bits", 0.0))
        mid = [min(1, L - 1), min(2, L - 1)]
        self.mid_leads = sorted(set(mid))
        # trackers
        self.t_loss = tf.keras.metrics.Mean(name="loss")
        self.t_forecast = tf.keras.metrics.Mean(name="l_forecast")
        self.t_kl = tf.keras.metrics.Mean(name="l_kl")
        self.t_forecast_q = tf.keras.metrics.Mean(name="l_forecast_q")
        self.t_active = tf.keras.metrics.Mean(name="active_units")
        self.t_gate_mean = tf.keras.metrics.Mean(name="gate_mean")
        self.t_gate_std = tf.keras.metrics.Mean(name="gate_std")
        self.t_grad2 = tf.keras.metrics.Mean(name="grad_norm_stage2")
        self.t_brier = tf.keras.metrics.Mean(name="brier")
        self.t_adjacent = tf.keras.metrics.Mean(name="adjacent_frac")
        self.t_corr = [tf.keras.metrics.Mean(name=f"corr_ratio_t{l}") for l in range(L)]
        self.t_scale = [tf.keras.metrics.Mean(name=f"lb_scale_t{l}") for l in range(L)]
        self.m_csi = [_Confusion(f"csi_t{l}") for l in range(L)]
        self.m_fss3 = [_Fss(f"fss3_t{l}", 3) for l in range(L)]
        self.m_fss9 = [_Fss(f"fss9_t{l}", 9) for l in range(L)]
        self.m_persist = ([_Confusion(f"persistence_csi_t{l}") for l in range(L)]
                          if label_type == "lightning" else [])
        self.m_csi_mid = _MeanOf("csi_mid", [self.m_csi[i] for i in self.mid_leads])

    # -- API ----------------------------------------------------------------
    def set_member(self, k: int):
        """Member k >= 0 draws the latent from the prior with a fixed seed;
        -1 uses the prior mean (the deterministic forecast)."""
        self.member.assign(int(k))

    @property
    def metrics(self):
        base = [self.t_loss, self.t_forecast]
        if self.cvae:
            base += [self.t_kl, self.t_forecast_q, self.t_active,
                     self.t_gate_mean, self.t_gate_std]
        base += self.m_csi + [self.m_csi_mid] + self.m_fss3 + self.m_fss9
        if self.label_type == "lightning":
            base += [self.t_brier] + self.m_persist
        else:
            base += [self.t_adjacent]
        base += self.t_corr + self.t_scale + [self.t_grad2]
        return base

    def call(self, inputs, training=None, y=None):
        feats, logits_b = self.feature_model(inputs, training=False)
        return self.head(feats, logits_b, inputs, y=y, training=training,
                         member=self.member)

    # -- losses -------------------------------------------------------------
    def _forecast(self, y, probs):
        y = tf.cast(y, tf.float32)
        probs = tf.cast(probs, tf.float32)
        total = 0.0
        for l in range(self.head.L):
            term = self.forecast_loss(y[:, l], probs[:, l])
            if self.expected_class_weight > 0:
                classes = tf.range(self.head.num_outputs, dtype=tf.float32)
                exp_c = tf.reduce_sum(probs[:, l] * classes, axis=-1)
                true_c = tf.reduce_sum(y[:, l] * classes, axis=-1)
                term += self.expected_class_weight * tf.reduce_mean(tf.abs(exp_c - true_c))
            total += self.lead_weights[l] * term
        return total

    def _kl(self, stats):
        mu_q, ls_q = stats["mu_q"], stats["log_sigma_q"]
        mu_p, ls_p = stats["mu_p"], stats["log_sigma_p"]
        var_q, var_p = tf.exp(2.0 * ls_q), tf.exp(2.0 * ls_p)
        kl = 0.5 * (var_q / var_p + tf.square(mu_q - mu_p) / var_p
                    - 1.0 - 2.0 * (ls_q - ls_p))              # (B, h, w, c)
        kl_dim = tf.reduce_mean(kl, axis=0)                     # per latent dimension
        active = tf.reduce_sum(tf.cast(kl_dim > 0.02, tf.float32))
        kl_fb = tf.reduce_mean(tf.maximum(kl_dim, self.free_bits))
        return kl_fb, active

    # -- monitoring ---------------------------------------------------------
    def _event(self, t):
        t = tf.cast(t, tf.float32)
        if self.label_type == "lightning":
            return t[..., 0] > 0.5
        return tf.argmax(t, axis=-1) > 0

    def _update_skill(self, inputs, y, probs):
        y = tf.cast(y, tf.float32)
        probs = tf.cast(probs, tf.float32)
        for l in range(self.head.L):
            g = self._event(y[:, l])
            p = self._event(probs[:, l])
            self.m_csi[l].update_state(g, p)
            self.m_fss3[l].update_state(g, p)
            self.m_fss9[l].update_state(g, p)
            if self.label_type == "lightning" and self.head.hr_channels >= 3:
                last = tf.cast(inputs["past_hr"][:, -1, :, :, 2], tf.float32) > 0.5
                self.m_persist[l].update_state(g, last)
        if self.label_type == "lightning":
            self.t_brier.update_state(tf.reduce_mean(tf.square(probs[..., 0] - y[..., 0])))
        else:
            yc = tf.argmax(y, axis=-1)
            pc = tf.argmax(probs, axis=-1)
            wrong = tf.not_equal(yc, pc)
            adjacent = tf.abs(yc - pc) == 1
            self.t_adjacent.update_state(
                tf.reduce_sum(tf.cast(wrong & adjacent, tf.float32))
                / (tf.reduce_sum(tf.cast(wrong, tf.float32)) + 1e-7))
        st = self.head.last_stats
        for l in range(self.head.L):
            self.t_corr[l].update_state(st["correction_ratio"][l])
            self.t_scale[l].update_state(self.head.lb_scale[l])
        if self.cvae:
            self.t_gate_mean.update_state(st["gate_mean"])
            self.t_gate_std.update_state(st["gate_std"])

    # -- steps --------------------------------------------------------------
    def train_step(self, data):
        inputs, y = data
        with tf.GradientTape() as tape:
            probs = self(inputs, training=True, y=y if self.cvae else None)
            l_forecast = self._forecast(y, probs)
            if self.cvae:
                kl, active = self._kl(self.head.last_stats)
                loss = l_forecast + self.beta * kl
            else:
                kl, active = 0.0, 0.0
                loss = l_forecast
            scaled = (self.optimizer.get_scaled_loss(loss)
                      if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
                      else loss)
        variables = self.head.trainable_variables
        grads = tape.gradient(scaled, variables)
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = self.optimizer.get_unscaled_gradients(grads)
        stage2 = {id(v) for blk in self.head.stage2 for v in blk.trainable_variables}
        g2 = [g for g, v in zip(grads, variables) if id(v) in stage2 and g is not None]
        if g2:
            self.t_grad2.update_state(tf.linalg.global_norm(g2))
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, variables) if g is not None])
        self.t_loss.update_state(loss)
        self.t_forecast.update_state(l_forecast)
        if self.cvae:
            # the training forward already samples z from the posterior,
            # so the "q" value is the forecast loss itself here; only on
            # validation do the two differ (prior mean vs posterior)
            self.t_forecast_q.update_state(l_forecast)
            self.t_kl.update_state(kl)
            self.t_active.update_state(active)
        self._update_skill(inputs, y, probs)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, y = data
        # what you actually get: the prior mean (or the set member)
        probs = self(inputs, training=False)
        l_forecast = self._forecast(y, probs)
        loss = l_forecast
        if self.cvae:
            # the ceiling: z from the posterior, and the KL of the pair
            probs_q = self(inputs, training=False, y=y)
            l_q = self._forecast(y, probs_q)
            kl, active = self._kl(self.head.last_stats)
            self.t_forecast_q.update_state(l_q)
            self.t_kl.update_state(kl)
            self.t_active.update_state(active)
            loss = l_forecast + self.beta * kl
            probs = self(inputs, training=False)   # restore last_stats of the prior pass
        self.t_loss.update_state(loss)
        self.t_forecast.update_state(l_forecast)
        self._update_skill(inputs, y, probs)
        return {m.name: m.result() for m in self.metrics}


class _BetaWarmup(tf.keras.callbacks.Callback):
    """beta_t: linear from 0 to beta over `warmup_epochs`, then constant."""

    def __init__(self, beta, warmup_epochs):
        super().__init__()
        self.beta_final = float(beta)
        self.warmup = max(0, int(warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        frac = 1.0 if self.warmup == 0 else min(1.0, (epoch + 1) / self.warmup)
        self.model.beta.assign(self.beta_final * frac)
        print(f"  beta = {self.beta_final * frac:.4f}")


class _WeightsCheckpoint(tf.keras.callbacks.Callback):
    """Per-epoch weights of a subclassed model (HDF5 through the .keras
    name), plus the epoch sidecar; the optimizer state is not kept."""

    def __init__(self, filepath, epoch_meta_path):
        super().__init__()
        self.filepath = str(filepath)
        self.epoch_meta_path = str(epoch_meta_path)

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.model.save_weights(self.filepath)
            with open(self.epoch_meta_path, "w") as f:
                json.dump({"next_epoch": epoch + 1, "completed_epoch": epoch}, f, indent=2)
            print(f"  [ckpt] saved epoch {epoch + 1} -> {self.filepath}")
        except Exception as e:
            print(f"  [ckpt] WARNING: failed to save checkpoint: {e}")


def _ensemble_metrics(model, dataset, n_members: int, max_batches: int = 20) -> dict:
    """Ensemble quality of a cVAE model on a few validation batches:
    n_members draws from the prior. RPS (ordinal classes) or the Brier
    score of the ensemble mean (lightning), the ensemble-mean CSI of the
    event and of the top two classes, the mean pairwise member
    disagreement, the spread-skill ratio of the expected class, and a
    rank histogram of the truth among the members."""
    L = model.head.L
    n_out = model.head.num_outputs
    rps_num = 0.0
    n_pix = 0.0
    csi = {"event": [0.0, 0.0, 0.0], "top2": [0.0, 0.0, 0.0]}
    disagree = 0.0
    spread_sq = 0.0
    err_sq = 0.0
    ranks = np.zeros(n_members + 1, dtype=np.int64)
    classes = np.arange(n_out, dtype=np.float32)
    for b, (inputs, y) in enumerate(dataset):
        if b >= max_batches:
            break
        members = []
        for k in range(n_members):
            model.set_member(k)
            members.append(np.asarray(model(inputs, training=False), dtype=np.float32))
        model.set_member(-1)
        y = np.asarray(y, dtype=np.float32)
        stack = np.stack(members, axis=0)                     # (K, B, L, H, W, C)
        mean = stack.mean(axis=0)
        if n_out > 1:
            cum_f = np.cumsum(mean, axis=-1)
            cum_o = np.cumsum(y, axis=-1)
            rps_num += float(np.sum(np.square(cum_f - cum_o)[..., :-1]))
            n_pix += float(np.prod(y.shape[:-1]))
            yc = np.argmax(y, axis=-1)
            mc = np.argmax(mean, axis=-1)
            for name, thr in (("event", 1), ("top2", n_out - 2)):
                g, p = yc >= thr, mc >= thr
                csi[name][0] += float(np.sum(g & p))
                csi[name][1] += float(np.sum(~g & p))
                csi[name][2] += float(np.sum(g & ~p))
            exp_m = np.sum(stack * classes, axis=-1)          # (K, ...)
            exp_o = np.sum(y * classes, axis=-1)
            arg_m = np.argmax(stack, axis=-1)
            for i in range(n_members):
                for j in range(i + 1, n_members):
                    disagree += float(np.mean(arg_m[i] != arg_m[j]))
            spread_sq += float(np.mean(np.var(exp_m, axis=0)))
            err_sq += float(np.mean(np.square(exp_m.mean(axis=0) - exp_o)))
            rank = np.sum(exp_m < exp_o[None], axis=0)
            ranks += np.bincount(rank.ravel(), minlength=n_members + 1)[:n_members + 1]
        else:
            rps_num += float(np.sum(np.square(mean[..., 0] - y[..., 0])))
            n_pix += float(np.prod(y.shape[:-1]))
            g, p = y[..., 0] > 0.5, mean[..., 0] >= 0.5
            csi["event"][0] += float(np.sum(g & p))
            csi["event"][1] += float(np.sum(~g & p))
            csi["event"][2] += float(np.sum(g & ~p))
            bm = stack[..., 0] >= 0.5
            for i in range(n_members):
                for j in range(i + 1, n_members):
                    disagree += float(np.mean(bm[i] != bm[j]))
            spread_sq += float(np.mean(np.var(stack[..., 0], axis=0)))
            err_sq += float(np.mean(np.square(mean[..., 0] - y[..., 0])))
            rank = np.sum(stack[..., 0] < y[None, ..., 0], axis=0)
            ranks += np.bincount(rank.ravel(), minlength=n_members + 1)[:n_members + 1]
    n_b = min(max_batches, b + 1) if n_pix else 1
    n_pairs = max(1, n_members * (n_members - 1) // 2)
    out = {
        "n_members": n_members,
        "n_batches": n_b,
        ("rps" if n_out > 1 else "brier_ensemble_mean"): rps_num / max(n_pix, 1.0),
        "ensemble_mean_csi_event": csi["event"][0] / (sum(csi["event"]) + 1e-7),
        "member_disagreement": disagree / (n_b * n_pairs),
        "spread_skill_ratio": float(np.sqrt(spread_sq / n_b) / (np.sqrt(err_sq / n_b) + 1e-7)),
        "rank_histogram": (ranks / max(ranks.sum(), 1)).tolist(),
    }
    if n_out > 1:
        out["ensemble_mean_csi_top2"] = csi["top2"][0] / (sum(csi["top2"]) + 1e-7)
    return out


class _EnsembleMonitor(tf.keras.callbacks.Callback):
    """The ensemble metrics of a cVAE model at the end of every epoch, on
    a few validation batches, written into the logs (history)."""

    def __init__(self, val_ds, n_members, max_batches=20):
        super().__init__()
        self.val_ds = val_ds
        self.n_members = int(n_members)
        self.max_batches = int(max_batches)

    def on_epoch_end(self, epoch, logs=None):
        m = _ensemble_metrics(self.model, self.val_ds, self.n_members, self.max_batches)
        if logs is not None:
            for k, v in m.items():
                if k != "rank_histogram":
                    logs[f"val_ens_{k}"] = float(v)
            for i, v in enumerate(m["rank_histogram"]):
                logs[f"val_ens_rank_{i}"] = float(v)
        print(f"  ensemble ({self.n_members} members): "
              + "  ".join(f"{k}={v:.4f}" for k, v in m.items()
                          if isinstance(v, float)))


def build_finetune_model_v2(base_model_path, hp: dict, label_type_hint=None,
                            ones_fraction=None, class_fractions=None,
                            radar_loss_cfg=None):
    """Frozen backbone + FinetuneHeadV2 as a FinetuneModelV2 (built, not
    compiled). The backbone's final 1x1 conv is copied without its
    activation so the head sees the logits L_b."""
    base = tf.keras.models.load_model(
        str(base_model_path),
        custom_objects={
            "ConvBlock": ConvBlock, "ResBlock": ResBlock,
            "GRUResBlock": GRUResBlock, "ResGRU": ResGRU,
            "WeightedFocalLoss": WeightedFocalLoss,
            "WeightedFocalCategoricalCrossentropy": WeightedFocalCategoricalCrossentropy,
            "iou_metric": iou_metric, "true_pos": true_pos,
            "false_pos": false_pos, "false_neg": false_neg,
        },
        compile=False,
    )
    base.trainable = False
    out_shape = base.output_shape
    L = int(out_shape[1])
    num_outputs = int(out_shape[-1])
    label_type = "lightning" if num_outputs == 1 else "radar"
    features = base.get_layer("backbone_output").output
    final_td = base.layers[-1]
    final_conv = final_td.layer if hasattr(final_td, "layer") else final_td
    logit_conv = tf.keras.layers.Conv2D(num_outputs, 1, activation=None,
                                        dtype="float32", name="backbone_logits")
    logits = tf.keras.layers.TimeDistributed(logit_conv, name="backbone_logits_td")(features)
    feature_model = tf.keras.Model(inputs=base.inputs, outputs=[features, logits],
                                   name="frozen_backbone")
    logit_conv.set_weights(final_conv.get_weights())
    feature_model.trainable = False

    shapes = {inp.name.split(":")[0]: inp.shape for inp in base.inputs}
    hr_channels = int(shapes.get("past_hr", [None, 0, 0, 0, 0])[-1] or 0)
    mr_channels = int(shapes.get("past_mr", [None, 0, 0, 0, 0])[-1] or 0)

    head = FinetuneHeadV2(hp, label_type, num_outputs, L, hr_channels, mr_channels,
                          name="finetune_head_v2")
    if label_type == "lightning":
        forecast_loss = PosWeightedBCE(ones_fraction if ones_fraction else 0.0106,
                                       cap=float(hp.get("pos_weight_cap", 30.0)),
                                       gamma=float(hp.get("lightning_gamma", 0.0)))
    else:
        cfg = dict(radar_loss_cfg or {})
        if hp["head"] == "swin_unet_cvae":
            # the reconstruction term of the ELBO: class-weighted CE, no
            # focal modulation
            cfg["gamma"] = 0.0
        forecast_loss = _build_radar_loss(class_fractions, cfg)
    model = FinetuneModelV2(feature_model, head, label_type, forecast_loss, hp,
                            name="finetuned_v2")
    # build the variables with one symbolic pass
    dummy = {k: tf.zeros([1] + [int(d) for d in shape[1:]], dtype=tf.float32)
             for k, shape in shapes.items()}
    model(dummy, training=False)
    if head.cvae:
        # the posterior path exists only with a target; build it too, so
        # every weight exists before load_weights / the optimizer
        y0 = tf.zeros([1] + [int(d) for d in out_shape[1:]], dtype=tf.float32)
        model(dummy, training=False, y=y0)
    return model, label_type


def build_finetune_model(base_model_path, finetune_cfg, ones_fraction,
                         class_fractions=None, radar_loss_cfg=None):
    """Load a base model, freeze it, and graft on a Swin head.

    Args:
        base_model_path: path to the saved base model (`.keras`).
        finetune_cfg: dict from `load_training_config()["finetune"]`. We
            pull `window_size`, `n_swin_blocks`, `num_heads`, `c_shared`,
            `head_dropout` from it.
        ones_fraction: occurrence rate (lightning modes only) for the
            WeightedFocalLoss. Ignored when the base model targets the
            5-class radar/OPERA head.

    Returns: (model, loss, metrics) - the compiled finetune model, plus
    the loss/metrics it should be compiled with (the caller wires those
    into model.compile alongside the AdamW optimizer).

    With `finetune_cfg["head"]` other than "swin_legacy" the v2 head is
    built instead (see FinetuneModelV2); the loss lives inside the model
    and (None, None) comes back for loss and metrics.
    """
    if finetune_cfg.get("head", "swin_legacy") != "swin_legacy":
        model, _label_type = build_finetune_model_v2(
            base_model_path, finetune_cfg, ones_fraction=ones_fraction,
            class_fractions=class_fractions, radar_loss_cfg=radar_loss_cfg)
        return model, None, None
    # Custom layers + loss + metrics defined in this module need to be
    # registered when reloading a saved base model. Without this Keras
    # can't reconstruct ResBlock/ResGRU/ConvBlock instances and bails
    # with `ValueError: Unknown layer: ResBlock`.
    base = tf.keras.models.load_model(
        str(base_model_path),
        custom_objects={
            # Backbone layers from build_coalition_model:
            "ConvBlock":         ConvBlock,
            "ResBlock":          ResBlock,
            "GRUResBlock":       GRUResBlock,
            "ResGRU":            ResGRU,
            # Loss + metrics (only present if the base was saved with
            # compile=True, but cheap to include unconditionally):
            "WeightedFocalLoss": WeightedFocalLoss,
            "WeightedFocalCategoricalCrossentropy": WeightedFocalCategoricalCrossentropy,
            "iou_metric":        iou_metric,
            "true_pos":          true_pos,
            "false_pos":         false_pos,
            "false_neg":         false_neg,
        },
        compile=False,
    )
    base.trainable = False  # freeze every weight in the backbone

    # Build a sub-model that ends at backbone_output. Calling it with
    # training=False forces dropout / batch-norm into inference mode at
    # finetune training time - critical because layer.trainable=False
    # only freezes weights, not stochastic behaviour.
    backbone = tf.keras.Model(
        inputs=base.inputs,
        outputs=base.get_layer('backbone_output').output,
        name='frozen_backbone',
    )
    backbone.trainable = False

    features = backbone(base.inputs, training=False)

    # Infer head shape from the base model's final output.
    out_shape = base.output_shape   # (None, F, H, W, num_outputs)
    future_timesteps = int(out_shape[1])
    num_outputs = int(out_shape[-1])
    label_type = 'lightning' if num_outputs == 1 else 'radar'

    head_out = build_swin_head(
        features,
        future_timesteps=future_timesteps,
        num_outputs=num_outputs,
        label_type=label_type,
        window_size=finetune_cfg["window_size"],
        n_blocks=finetune_cfg["n_swin_blocks"],
        num_heads=finetune_cfg["num_heads"],
        c_shared=finetune_cfg["c_shared"],
        dropout=finetune_cfg["head_dropout"],
    )

    finetuned = tf.keras.Model(
        inputs=base.inputs, outputs=head_out, name='finetuned',
    )

    if label_type == 'lightning':
        loss = WeightedFocalLoss(ones_fraction=ones_fraction, gamma=2.0)
        metrics = [iou_metric, true_pos, false_pos, false_neg]
    else:
        # _build_radar_loss falls back to plain CategoricalCrossentropy
        # (label_smoothing=0.01) when radar_loss_cfg is None / 'none' so
        # existing finetune runs stay numerically identical. The label
        # smoothing default mirrors the historical behaviour - it prevents
        # log(0) when the Swin head's softmax saturates a class to exact
        # 0 under mixed_float16 (intermediate attention/MLP layers run in
        # fp16 and can produce extreme pre-softmax logits that round-trip
        # to inf).
        loss = _build_radar_loss(class_fractions, radar_loss_cfg)
        metrics = ['accuracy']

    return finetuned, loss, metrics


# ============================================================================
# Dataset loading
# ============================================================================

def load_dataset(dataset_dir, batch_size, shuffle=False, shuffle_buffer=256):
    """Load a saved dataset split and prepare it for training.

    Supports two on-disk formats — distinguished by the `format` field
    in `metadata.json`:

      - "tfrecord" (current): sharded `shard_*.tfrecord` files, parsed
        with the signature reconstructed from metadata. Streams from
        disk so the shuffle buffer doesn't have to hold the whole
        dataset in RAM. The default `shuffle_buffer=256` is sized for
        ~5-MB samples (well under 2 GB of host RAM).
      - "tf_dataset_save" (legacy): the old monolithic
        `tf.data.Dataset.save` snapshot. Kept for backward compatibility.
    """
    dataset_dir = Path(dataset_dir)

    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.is_file():
        with open(metadata_path) as f:
            meta = json.load(f)
        fmt = meta.get("format", "tf_dataset_save")
    else:
        # No metadata.json -> assume legacy format.
        meta = None
        fmt = "tf_dataset_save"

    if fmt == "tfrecord":
        ds = _load_tfrecord_split(dataset_dir, meta)
    else:
        ds = tf.data.Dataset.load(str(dataset_dir))

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer,
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _load_tfrecord_split(split_dir: Path, meta: dict) -> tf.data.Dataset:
    """Reconstruct the parse signature from metadata.json and read the
    `shard_*.tfrecord` files. Mirrors `create_datasets.load_tfrecord_dataset`
    but reads the shapes from metadata so train_models doesn't have to
    pull in the mode-config registry."""
    shard_paths = sorted(str(p) for p in split_dir.glob("shard_*.tfrecord"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No TFRecord shards in {split_dir} (expected "
            f"`shard_*.tfrecord`). Re-run create_datasets.py."
        )

    input_shapes: dict[str, list[int]] = meta["input_shapes"]
    label_shape: list[int] = meta["label_shape"]

    feature_description = {
        key: tf.io.FixedLenFeature([], tf.string) for key in input_shapes
    }
    feature_description["label"] = tf.io.FixedLenFeature([], tf.string)

    def parse(serialised):
        parsed = tf.io.parse_single_example(serialised, feature_description)
        inputs = {}
        for key, shape in input_shapes.items():
            t = tf.io.parse_tensor(parsed[key], out_type=tf.float32)
            t.set_shape(shape)
            inputs[key] = t
        label = tf.io.parse_tensor(parsed["label"], out_type=tf.float32)
        label.set_shape(label_shape)
        return inputs, label

    files_ds = tf.data.Dataset.from_tensor_slices(shard_paths)
    ds = files_ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    # TFRecordDataset advertises an unknown cardinality, which makes
    # Keras print "X/Unknown" with no ETA during model.fit. Stamp the
    # exact count from metadata so the progress bar shows X/Y and ETA.
    n_samples = int(meta.get("n_samples", 0))
    if n_samples > 0:
        ds = ds.apply(tf.data.experimental.assert_cardinality(n_samples))
    return ds


# ============================================================================
# ones_fraction from pre-computed JSON
# ============================================================================

def load_ones_fraction(data_root, source, period=None):
    """Load lightning occurrence fraction from the prior JSON for this tag.

    Reads `our_data/lightning_fraction_<source>[_<period>].json` (produced
    by `lightning_fraction.py [--period <label>]`) and returns the
    'occurrence' fraction.

    The tag has to match the split the model trains on, exactly as the
    radar prior does: a prior computed over a different window describes a
    class balance this model never sees, and the focal loss would then
    correct for an imbalance that is not the one present. Windows differ
    sharply here - how many "no lightning" timesteps a split contains is
    precisely what this measures.

    Args:
        data_root: path to our_data/ directory
        source:    sample-selection source (pipeline_config.SOURCE)
        period:    period label, or None for the whole-archive split

    Returns:
        float: occurrence fraction scoped to that split's train CSV

    Raises:
        FileNotFoundError: if the prior JSON does not exist
        KeyError: if 'occurrence' key is missing from the JSON
    """
    tag = data_tag(source, period)
    json_path = Path(data_root) / f"lightning_fraction_{tag}.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Lightning fraction file not found: {json_path}\n"
            f"Run `python lightning_fraction.py"
            + (f" --period {period}" if period else "") + "` first."
        )

    with open(json_path) as f:
        fractions = json.load(f)

    if "occurrence" not in fractions:
        raise KeyError(
            f"'occurrence' key not found in {json_path}. "
            f"Available keys: {list(fractions.keys())}"
        )

    ones_fraction = fractions["occurrence"]["fraction"]
    print(f"  Loaded ones_fraction from {json_path}")
    print(f"    occurrence fraction: {ones_fraction:.6f}")
    print(f"    class ratio: 1:{int(1/ones_fraction) if ones_fraction > 0 else 'inf'}")
    return ones_fraction


def load_class_fractions(data_root, source, period=None):
    """Load OPERA rainfall-rate per-class pixel fractions from the prior JSON.

    Reads `our_data/opera_rainfall_fraction_<source>[_<period>].json`
    (produced by `opera_rainfall_fraction.py [--period <label>]`) and
    returns the 5-element fractions list used by
    WeightedFocalCategoricalCrossentropy.

    Tagged per period for the same reason the lightning prior and the
    normalization statistics are: the prior has to describe the split the
    model actually trains on. One computed over a different window
    measures a class balance this model never sees, and the focal loss
    would then correct for an imbalance that is not present - silently,
    because a fractions list from any window is structurally valid.
    """
    tag = data_tag(source, period)
    json_path = Path(data_root) / f"opera_rainfall_fraction_{tag}.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"OPERA class-fraction file not found: {json_path}\n"
            f"Run `python opera_rainfall_fraction.py"
            + (f" --period {period}" if period else "") + "` first."
        )
    with open(json_path) as f:
        stats = json.load(f)
    if "fractions" not in stats or len(stats["fractions"]) != 5:
        raise KeyError(
            f"{json_path}: expected a 5-element 'fractions' list, "
            f"got {stats.get('fractions')!r}"
        )
    fractions = [float(x) for x in stats["fractions"]]
    labels = stats.get("classes", [f"c{i}" for i in range(5)])
    print(f"  Loaded class_fractions from {json_path}")
    for k, (lab, fr) in enumerate(zip(labels, fractions)):
        ratio = f"1:{int(1/fr):,}" if fr > 0 else "1:inf"
        print(f"    class {k} ({lab:>5}): {fr:.6f}  ({ratio})")
    return fractions


# ============================================================================
# Wall-time callback
# ============================================================================

class WallTimeCallback(tf.keras.callbacks.Callback):
    """Track wall time per epoch and cumulative."""

    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.epoch_times = []
        print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start
        cumulative = time.time() - self.train_start
        self.epoch_times.append(elapsed)
        print(f"  Epoch {epoch + 1} wall time: {elapsed:.1f}s "
              f"(cumulative: {cumulative:.1f}s)")

    def on_train_end(self, logs=None):
        total = time.time() - self.train_start
        print(f"\nTotal training wall time: {total:.1f}s "
              f"({total/60:.1f} min)")
        if self.epoch_times:
            print(f"  Average per epoch: {np.mean(self.epoch_times):.1f}s")


class HistoryWriter(tf.keras.callbacks.Callback):
    """Writes the history JSON after every epoch, so an interrupted run
    keeps its curves. `meta_fn()` returns the static fields of the
    document; `history` accumulates the per-epoch logs and, on a resume,
    continues the file's own history up to `initial_epoch`. The stage
    calls write(complete=True) at the end; until then the file carries
    "complete": false. Put it last in the callbacks so it sees what the
    other callbacks add to the logs."""

    def __init__(self, path, meta_fn, initial_epoch: int = 0):
        super().__init__()
        self.path = Path(path)
        self.meta_fn = meta_fn
        self.history: dict[str, list] = {}
        if initial_epoch > 0 and self.path.is_file():
            try:
                with open(self.path) as f:
                    prev = json.load(f).get("history") or {}
                self.history = {k: [float(x) for x in v[:initial_epoch]]
                                for k, v in prev.items()}
                print(f"  History: continuing {self.path.name} from epoch {initial_epoch}")
            except (OSError, ValueError):
                self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            try:
                self.history.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                continue
        self.write(complete=False)

    def write(self, complete: bool = True):
        doc = dict(self.meta_fn())
        doc["epochs_completed"] = len(self.history.get("loss", []))
        doc["history"] = self.history
        doc["complete"] = bool(complete)
        try:
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(doc, f, indent=2)
            tmp.replace(self.path)
        except OSError as e:
            print(f"  WARNING: could not write {self.path}: {e}")
        return self.path


class _ResumableCheckpoint(tf.keras.callbacks.Callback):
    """Per-epoch checkpoint that writes both the .keras model file and a
    small JSON sidecar with the next-epoch index, so a subsequent run can
    resume at the right position in the LR schedule.

    Distinct from ModelCheckpoint(save_best_only=...) — this one always
    saves the *latest* state, intentionally overwriting any previous
    checkpoint. That's what 'resume from where you were' needs.
    """

    def __init__(self, filepath: str, epoch_meta_path: str, verbose: int = 1):
        super().__init__()
        self.filepath = filepath
        self.epoch_meta_path = epoch_meta_path
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # Save the full model (architecture + weights + optimizer state).
        # Optimizer state matters: the LR schedule writes the current lr
        # into the optimizer at each epoch begin, but Adam's m / v
        # accumulators also persist here so resume picks up momentum.
        try:
            self.model.save(self.filepath)
            with open(self.epoch_meta_path, "w") as f:
                json.dump({"next_epoch": epoch + 1,
                           "completed_epoch": epoch}, f, indent=2)
            if self.verbose:
                print(f"  [ckpt] saved epoch {epoch + 1} -> "
                      f"{self.filepath}")
        except Exception as e:
            # Never let a checkpoint failure kill the training run.
            print(f"  [ckpt] WARNING: failed to save checkpoint: {e}")


# ============================================================================
# Training
# ============================================================================

def train(mode, data_root, epochs, batch_size, output_dir,
          dropout=0.1, norm=None, dataset_dir=None,
          source="dbscan",
          shuffle_buffer=256,
          mixed_precision=True,
          lr_schedule_cfg=None,
          early_stopping_cfg=None,
          checkpoint_cfg=None,
          radar_loss_cfg=None,
          resume=True,
          period=None, datasets_root=None):
    """Main training function (base stage).

    Args:
        mode: training-mode name (see TRAINING_MODES at the top of this
            file). Used to name the saved model / history files and to
            locate the default dataset directory.
        data_root: path to our_data/ containing datasets/{mode}_{source}/
            and lightning_fraction_<source>.json.
        epochs: number of training epochs.
        batch_size: training batch size.
        output_dir: where to save model + history.
        dropout: dropout rate.
        norm: normalization type ('batch', 'layer', or None).
        dataset_dir: explicit path to the dataset directory. When
            provided, overrides the default data_root/datasets/{mode}_{source}.
        source: extract_patch_seq source ('dbscan' = patch_index.csv,
            (always pipeline_config.SOURCE) the dataset was built
            from. Selects which datasets/<mode>_<source>/ directory to
            read and is appended to the checkpoint / model / history
            filenames so the two tracks don't clobber each other.
        shuffle_buffer: sample count for the training-time shuffle
            buffer. With ~5 MB samples, 256 ~= 1.3 GB host RAM.
        lr_schedule_cfg: dict with keys `type`, `initial_lr`,
            `warmup_epochs`, `min_lr`. None -> use Adam's default LR
            with no schedule (legacy behaviour).
        early_stopping_cfg: dict with keys `enabled`, `monitor`, `mode`,
            `patience`, `min_delta`, `restore_best_weights`. None ->
            disable early stopping.

    Returns: (model_path, history_path) - tuple of pathlib.Path values
    pointing at the saved base model and its history JSON. The
    finetune stage reads `model_path` to graft the Swin head onto.
    """
    data_root = resolve_data_root(data_root)
    datasets_root = resolve_datasets_root(data_root, datasets_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Mode + source together are the unique experiment identifier, and the
    # mode name already states its own track, so the tag names saved
    # artefacts (weights + history + checkpoints) AND dataset directories
    # self-describingly. See build_run_tag at the top of this module.
    run_tag = build_run_tag(mode, source, period)

    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)
    else:
        dataset_dir = datasets_root / run_tag
        # An archived dataset is extracted first. Blocking on purpose:
        # training cannot start without the bytes, so there is nothing
        # useful to overlap with. The archive is kept; reclaiming the
        # on-disk copy afterwards is a manual compress_datasets step.
        ensure_available(run_tag, datasets_root)

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"

    # Check datasets exist
    for d in [train_dir, val_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Dataset not found: {d}")

    # Load metadata from the training split (always present)
    meta_path = train_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"metadata.json not found in {train_dir}. "
            f"Regenerate datasets with create_datasets.py.")

    with open(meta_path) as f:
        meta = json.load(f)

    input_shapes = meta["input_shapes"]  # e.g. {"past_hr": [3,256,256,10], ...}
    label_type = meta.get("label_type", "lightning" if "lightning" in mode else "radar")
    past_timesteps = next(iter(input_shapes.values()))[0]
    # future timesteps = label timesteps
    label_shape = meta.get("label_shape", [3, 256, 256, 1])
    future_timesteps = label_shape[0]

    print("=" * 70)
    print(f"COALITION-4 Training (base) - Mode: {mode}  Source: {source}")
    print("=" * 70)
    print(f"  Dataset:       {dataset_dir}")
    print(f"  Label type:    {label_type}")
    print(f"  Inputs:        {list(input_shapes.keys())}")
    for name, shape in input_shapes.items():
        print(f"    {name}: {shape}")
    print(f"  Label shape:   {label_shape}")
    print(f"  Past steps:    {past_timesteps}")
    print(f"  Future steps:  {future_timesteps}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Dropout:       {dropout}")
    print(f"  Norm:          {norm}")
    print(f"  Mixed prec:    {'float16' if mixed_precision else 'float32'}")
    print()

    # Configure the TF runtime (memory growth + XLA + mixed precision).
    # This addresses the "Windows fatal exception: access violation" crash
    # signature we kept seeing on the heavier modes — see the docstring of
    # configure_tf_runtime for the per-flag rationale.
    print("Configuring TF runtime...")
    configure_tf_runtime(use_mixed_precision=mixed_precision)

    # Load ones_fraction for lightning modes from the per-source prior
    # JSON. Each track's class balance is computed from its own
    # train_data_<source>.csv, so the focal-loss prior matches what the
    # model actually sees during training.
    if label_type == "lightning":
        ones_fraction = load_ones_fraction(data_root, source, period)
        class_fractions = None
    else:
        ones_fraction = 0.0106  # unused for radar
        # Only load the radar prior when the configured loss actually
        # needs it - keeps plain-CCE runs from failing on a missing JSON.
        # The continuous head minimises weighted MSE and never touches
        # the per-class prior, so only the 5-class head requires it.
        needs_prior = (label_type == "radar" and
                       (radar_loss_cfg or {}).get("weighting", "none")
                       != "none")
        class_fractions = (
            load_class_fractions(data_root, source, period)
            if needs_prior else None
        )

    # Load datasets
    print("\nLoading datasets...")
    train_ds = load_dataset(train_dir, batch_size,
                             shuffle=True, shuffle_buffer=shuffle_buffer)
    val_ds = load_dataset(val_dir, batch_size,
                           shuffle=False)
    print("  Datasets loaded")

    # Build model dynamically from metadata
    print("\nBuilding model...")
    model = build_coalition_model(
        input_shapes=input_shapes,
        label_type=label_type,
        past_timesteps=past_timesteps,
        future_timesteps=future_timesteps,
        dropout=dropout,
        norm=norm,
        ones_fraction=ones_fraction,
        class_fractions=class_fractions,
        radar_loss_cfg=radar_loss_cfg,
    )
    model.summary(print_fn=lambda x: print(f"  {x}"))
    print()

    # ------------------------------------------------------------------
    # Resumable training
    # ------------------------------------------------------------------
    # Per-epoch checkpoint at `models/checkpoints/<mode>_<source>_latest.keras`.
    # On launch, if a checkpoint exists and `resume=True`, its weights
    # are loaded so the run picks up where the previous one stopped -
    # critical on Windows where the occasional driver-level CUDA crash
    # would otherwise lose hours of progress.
    ckpt_cfg = checkpoint_cfg or {}
    ckpt_enabled = ckpt_cfg.get("enabled", True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"{run_tag}_latest.keras"
    initial_epoch = 0
    if ckpt_enabled and resume and ckpt_path.is_file():
        try:
            print(f"Resuming from checkpoint: {ckpt_path}")
            model.load_weights(str(ckpt_path))
            # Also restore the epoch counter so the LR schedule and
            # callbacks see the right position. The companion JSON below
            # is written alongside the .keras file by the checkpoint
            # callback.
            ckpt_meta_path = ckpt_dir / f"{run_tag}_latest.json"
            if ckpt_meta_path.is_file():
                with open(ckpt_meta_path) as f:
                    initial_epoch = int(json.load(f).get("next_epoch", 0))
                print(f"  Resumed at epoch {initial_epoch}")
        except Exception as e:
            print(f"  WARNING: could not load {ckpt_path}: {e}")
            print(f"  Starting fresh.")
            initial_epoch = 0

    # Callbacks
    wall_time = WallTimeCallback()
    callbacks: list = [wall_time]

    # Learning-rate schedule. The cosine-with-warmup config drives a
    # standard LearningRateScheduler — we don't combine it with
    # ReduceLROnPlateau because the cosine decay is already an explicit
    # decay schedule; stacking both would fight each other.
    if lr_schedule_cfg is not None:
        sched_fn = cosine_warmup_schedule(
            initial_lr=lr_schedule_cfg["initial_lr"],
            warmup_epochs=lr_schedule_cfg["warmup_epochs"],
            total_epochs=epochs,
            min_lr=lr_schedule_cfg["min_lr"],
        )
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(sched_fn, verbose=1)
        )
        print(f"  LR schedule:  cosine_warmup "
              f"(initial={lr_schedule_cfg['initial_lr']:g}, "
              f"warmup={lr_schedule_cfg['warmup_epochs']} ep, "
              f"min={lr_schedule_cfg['min_lr']:g})")

    # Early stopping. When `restore_best_weights=True` the model held in
    # memory at the end of fit() is the best-epoch one, so the post-fit
    # save below captures that automatically whether ES fired or the run
    # went the full `epochs`.
    if early_stopping_cfg is not None and early_stopping_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_cfg["monitor"],
                mode=early_stopping_cfg["mode"],
                patience=early_stopping_cfg["patience"],
                min_delta=early_stopping_cfg["min_delta"],
                restore_best_weights=early_stopping_cfg["restore_best_weights"],
                verbose=1,
            )
        )
        print(f"  EarlyStop:    monitor={early_stopping_cfg['monitor']} "
              f"({early_stopping_cfg['mode']}), "
              f"patience={early_stopping_cfg['patience']}, "
              f"restore_best={early_stopping_cfg['restore_best_weights']}")

    # Per-epoch resumable checkpoint. Distinct from the final model save
    # below: this is the *latest* state used for resume, not the *best*
    # state used for inference. EarlyStopping(restore_best_weights=True)
    # still controls what ends up in the final `coalition_<mode>.keras`.
    if ckpt_enabled:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(_ResumableCheckpoint(
            filepath=str(ckpt_path),
            epoch_meta_path=str(ckpt_dir / f"{run_tag}_latest.json"),
            verbose=1,
        ))
        print(f"  Checkpoint:   per-epoch -> {ckpt_path}")
    print()

    # Train
    print("\nStarting training...")
    def _base_meta():
        return {
            "mode": mode, "source": source, "stage": "base",
            "label_type": label_type, "batch_size": batch_size,
            "dropout": dropout, "norm": norm,
            "ones_fraction": float(ones_fraction) if label_type == "lightning" else None,
            "wall_times": wall_time.epoch_times,
            "total_wall_time": sum(wall_time.epoch_times),
        }
    history_writer = HistoryWriter(output_dir / f"history_{run_tag}.json",
                                   _base_meta, initial_epoch)
    callbacks.append(history_writer)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    # Save model
    model_path = output_dir / f"coalition_{run_tag}.keras"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Provenance sidecar. Read back by the leakage gate whenever this
    # model is later frozen and reused as a feature extractor.
    ds_period = Period.from_dict(meta.get("period"))
    sidecar = save_model_period(model_path, ds_period, mode=mode,
                               source=source, stage="base",
                               dataset_dir=dataset_dir)
    print(f"Period sidecar  : {sidecar} "
          f"({ds_period or 'no period declared'})")

    # Save history (the writer has been saving it after every epoch;
    # this is the complete version)
    history_path = history_writer.write(complete=True)
    print(f"History saved to: {history_path}")

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)

    return model_path, history_path


def train_finetune(mode, data_root, base_model_path, output_dir,
                   source="dbscan", batch_size=4,
                   finetune_cfg=None,
                   shuffle_buffer=256,
                   mixed_precision=True,
                   early_stopping_cfg=None,
                   checkpoint_cfg=None,
                   radar_loss_cfg=None,
                   resume=True,
                   period=None,
                   allow_period_overlap=False, datasets_root=None):
    """Domain-adaptation fine-tune: freeze base, attach Swin head, train.

    Args:
        mode: training-mode name (drives label_type via the metadata).
        data_root: path to our_data/.
        base_model_path: path to the saved base model (`.keras`).
        output_dir: where to save the fine-tuned model + history.
        source: which extract_patch_seq source the dataset was built
            from. Used to locate datasets/<mode>_<source>/ and to suffix
            output filenames.
        batch_size: training batch size.
        finetune_cfg: dict from `load_training_config()["finetune"]`.
            Drives the Swin head hyperparameters + AdamW config.
        early_stopping_cfg / checkpoint_cfg / resume: same semantics as
            in `train()`.
    """
    if finetune_cfg is None:
        raise ValueError("finetune_cfg is required for train_finetune()")

    data_root = resolve_data_root(data_root)
    datasets_root = resolve_datasets_root(data_root, datasets_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = build_run_tag(mode, source, period)

    dataset_dir = datasets_root / run_tag
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    for d in [train_dir, val_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Dataset not found: {d}")

    meta_path = train_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    label_type = meta.get(
        "label_type", "lightning" if "lightning" in mode else "radar",
    )

    # ---- Feature-extractor leakage gate ------------------------------
    # The base model is about to be frozen and used as a feature
    # extractor. If it was trained on dates this dataset also covers, the
    # backbone has already seen the answers and every score measured here
    # is optimistic. Compare the recorded periods, not the filenames.
    fe_period = load_model_period(base_model_path)
    ds_period = Period.from_dict(meta.get("period"))
    print(f"Feature extractor : {base_model_path}")
    print(f"  FE period       : {fe_period or 'UNDECLARED (pre-period artefact)'}")
    print(f"  dataset period  : {ds_period or 'UNDECLARED (whole archive)'}")
    if fe_period is None or ds_period is None:
        print(
            "  WARNING: at least one side declares no period, so overlap "
            "cannot be checked. Rebuild with --period on both sides to "
            "make this verifiable."
        )
    require_no_overlap(fe_period, ds_period, allow=allow_period_overlap)

    epochs = finetune_cfg["epochs"]

    print("=" * 70)
    print(f"COALITION-4 Training (finetune) - Mode: {mode}  Source: {source}")
    print("=" * 70)
    print(f"  Dataset:        {dataset_dir}")
    print(f"  Base model:     {base_model_path}")
    print(f"  Label type:     {label_type}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Optimizer:      {finetune_cfg['optimizer']}")
    print(f"  Weight decay:   {finetune_cfg['weight_decay']}")
    print(f"  Initial LR:     {finetune_cfg['initial_lr']:g}")
    print(f"  Swin window:    {finetune_cfg['window_size']}")
    print(f"  Swin blocks:    {finetune_cfg['n_swin_blocks']}")
    print(f"  Swin heads:     {finetune_cfg['num_heads']}")
    print(f"  Swin c_shared:  {finetune_cfg['c_shared']}")
    print(f"  Head dropout:   {finetune_cfg['head_dropout']}")
    print(f"  Mixed prec:     {'float16' if mixed_precision else 'float32'}")
    print()

    configure_tf_runtime(use_mixed_precision=mixed_precision)

    if label_type == "lightning":
        ones_fraction = load_ones_fraction(data_root, source, period)
        class_fractions = None
    else:
        ones_fraction = 0.0106  # unused for radar
        # The continuous head minimises weighted MSE and never touches
        # the per-class prior, so only the 5-class head requires it.
        needs_prior = (label_type == "radar" and
                       (radar_loss_cfg or {}).get("weighting", "none")
                       != "none")
        class_fractions = (
            load_class_fractions(data_root, source, period)
            if needs_prior else None
        )

    print("\nLoading datasets...")
    train_ds = load_dataset(train_dir, batch_size,
                             shuffle=True, shuffle_buffer=shuffle_buffer)
    val_ds = load_dataset(val_dir, batch_size, shuffle=False)
    if finetune_cfg.get("max_batches"):
        n = int(finetune_cfg["max_batches"])
        train_ds, val_ds = train_ds.take(n), val_ds.take(n)
        print(f"  DRY RUN: {n} batch(es) per split")
    print("  Datasets loaded")

    if finetune_cfg.get("head", "swin_legacy") != "swin_legacy":
        return _train_finetune_v2(
            mode, source, period, run_tag, base_model_path, output_dir,
            train_ds, val_ds, finetune_cfg, label_type, ones_fraction,
            class_fractions, radar_loss_cfg, checkpoint_cfg, resume,
            batch_size, ds_period, fe_period, allow_period_overlap, dataset_dir)

    # Build the fine-tune model: frozen backbone + Swin head + per-step heads.
    print("\nBuilding fine-tune model (frozen backbone + Swin head)...")
    model, loss, metrics = build_finetune_model(
        base_model_path=base_model_path,
        finetune_cfg=finetune_cfg,
        ones_fraction=ones_fraction,
        class_fractions=class_fractions,
        radar_loss_cfg=radar_loss_cfg,
    )

    # Pick optimizer. AdamW is the default for fine-tune; falling back
    # to plain Adam is supported for parity experiments. The
    # construction order is fiddly because of two interacting issues
    # on the typical TF 2.10 install:
    #   1. tf.keras.optimizers.AdamW (non-experimental) only exists on
    #      TF >= 2.11; older versions raise AttributeError.
    #   2. tf.keras.optimizers.experimental.AdamW + mixed_float16 +
    #      its default XLA-compiled update step can't trace
    #      AutoCastVariable tensors, blowing up at the first
    #      train_step with `TypeError: Python object could not be
    #      represented through the generic tracing type`. Passing
    #      jit_compile=False to the experimental optimizer skips that
    #      XLA path and the update runs in eager-graph mode instead.
    # If both AdamW attempts fail we fall back to plain Adam (no
    # weight_decay) with a warning - that loses regularization but the
    # finetune head can still train.
    lr = finetune_cfg["initial_lr"]
    wd = finetune_cfg["weight_decay"]
    # global_clipnorm=1.0 caps the L2 norm of the full gradient at 1.0. This
    # is the standard transformer fine-tune recipe and the canonical defense
    # against NaN losses when fp16 attention products overflow inside the
    # Swin head (pre-softmax Q@K^T can exceed 65504) or when an extreme batch
    # produces a huge gradient that throws weights out of their stable regime.
    optimizer = None
    if finetune_cfg["optimizer"] == "adamw":
        # 1. Non-experimental AdamW (TF >= 2.11).
        if optimizer is None and hasattr(tf.keras.optimizers, "AdamW"):
            try:
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=lr, weight_decay=wd,
                    global_clipnorm=1.0,
                )
            except (TypeError, ValueError):
                optimizer = None
        # 2. Experimental AdamW with XLA disabled (TF 2.10 / 2.11 path).
        if (optimizer is None
                and hasattr(tf.keras.optimizers, "experimental")
                and hasattr(tf.keras.optimizers.experimental, "AdamW")):
            try:
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=lr,
                    weight_decay=wd,
                    jit_compile=False,
                    global_clipnorm=1.0,
                )
            except (TypeError, ValueError):
                optimizer = None
        # 3. Last resort.
        if optimizer is None:
            print("  WARNING: AdamW unavailable on this TF; falling back "
                  "to Adam without weight_decay. Install TF >= 2.11 or "
                  "`tensorflow_addons` to recover the decoupled "
                  "weight-decay term.")
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, global_clipnorm=1.0,
            )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, global_clipnorm=1.0,
        )

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary(print_fn=lambda x: print(f"  {x}"))
    print()

    # Resume from a finetune-stage checkpoint if it exists. Distinct
    # from the base checkpoint above so a partially-finished finetune
    # doesn't overwrite the base latest.
    ckpt_cfg = checkpoint_cfg or {}
    ckpt_enabled = ckpt_cfg.get("enabled", True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"{run_tag}_finetune_latest.keras"
    initial_epoch = 0
    if ckpt_enabled and resume and ckpt_path.is_file():
        try:
            print(f"Resuming fine-tune from checkpoint: {ckpt_path}")
            model.load_weights(str(ckpt_path))
            ckpt_meta_path = ckpt_dir / f"{run_tag}_finetune_latest.json"
            if ckpt_meta_path.is_file():
                with open(ckpt_meta_path) as f:
                    initial_epoch = int(json.load(f).get("next_epoch", 0))
                print(f"  Resumed at epoch {initial_epoch}")
        except Exception as e:
            print(f"  WARNING: could not load {ckpt_path}: {e}")
            print(f"  Starting fresh.")
            initial_epoch = 0

    wall_time = WallTimeCallback()
    callbacks: list = [wall_time]

    # Cosine warmup at the finetune LR.
    sched_fn = cosine_warmup_schedule(
        initial_lr=finetune_cfg["initial_lr"],
        warmup_epochs=finetune_cfg["warmup_epochs"],
        total_epochs=epochs,
        min_lr=finetune_cfg["min_lr"],
    )
    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(sched_fn, verbose=1)
    )
    print(f"  LR schedule:    cosine_warmup "
          f"(initial={finetune_cfg['initial_lr']:g}, "
          f"warmup={finetune_cfg['warmup_epochs']} ep, "
          f"min={finetune_cfg['min_lr']:g})")

    if (early_stopping_cfg is not None
            and early_stopping_cfg.get("enabled", True)):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_cfg["monitor"],
                mode=early_stopping_cfg["mode"],
                patience=early_stopping_cfg["patience"],
                min_delta=early_stopping_cfg["min_delta"],
                restore_best_weights=early_stopping_cfg["restore_best_weights"],
                verbose=1,
            )
        )

    if ckpt_enabled:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(_ResumableCheckpoint(
            filepath=str(ckpt_path),
            epoch_meta_path=str(ckpt_dir / f"{run_tag}_finetune_latest.json"),
            verbose=1,
        ))
        print(f"  Checkpoint:     per-epoch -> {ckpt_path}")
    print()

    def _legacy_meta():
        return {
            "mode": mode, "source": source, "stage": "finetune",
            "label_type": label_type, "base_model": str(base_model_path),
            "batch_size": batch_size,
            "optimizer": finetune_cfg["optimizer"],
            "weight_decay": finetune_cfg["weight_decay"],
            "initial_lr": finetune_cfg["initial_lr"],
            "swin": {
                "window_size":   finetune_cfg["window_size"],
                "n_blocks":      finetune_cfg["n_swin_blocks"],
                "num_heads":     finetune_cfg["num_heads"],
                "c_shared":      finetune_cfg["c_shared"],
                "head_dropout":  finetune_cfg["head_dropout"],
            },
            "ones_fraction": float(ones_fraction) if label_type == "lightning" else None,
            "wall_times": wall_time.epoch_times,
            "total_wall_time": sum(wall_time.epoch_times),
        }
    history_writer = HistoryWriter(output_dir / f"history_{run_tag}_finetuned.json",
                                   _legacy_meta, initial_epoch)
    callbacks.append(history_writer)

    print("\nStarting fine-tune training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    model_path = output_dir / f"coalition_{run_tag}_finetuned.keras"
    model.save(str(model_path))
    print(f"\nFine-tuned model saved to: {model_path}")

    # The fine-tuned model's period is the dataset it was tuned on, not
    # the feature extractor's. `feature_extractor` records the backbone's
    # provenance so the chain stays traceable if this model is itself
    # frozen and reused later.
    sidecar = save_model_period(model_path, ds_period, mode=mode,
                               source=source, stage="finetune",
                               dataset_dir=dataset_dir)
    with open(sidecar, encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["feature_extractor"] = {
        "model": str(base_model_path),
        "period": fe_period.to_dict() if fe_period else None,
        "overlap_allowed": bool(allow_period_overlap),
    }
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print(f"Period sidecar  : {sidecar} "
          f"({ds_period or 'no period declared'})")

    history_path = history_writer.write(complete=True)
    print(f"History saved to: {history_path}")

    print("\n" + "=" * 70)
    print("Fine-tune training complete.")
    print("=" * 70)

    return model_path, history_path


def _train_finetune_v2(mode, source, period, run_tag, base_model_path, output_dir,
                       train_ds, val_ds, finetune_cfg, label_type, ones_fraction,
                       class_fractions, radar_loss_cfg, checkpoint_cfg, resume,
                       batch_size, ds_period, fe_period, allow_period_overlap,
                       dataset_dir):
    """The fine-tune stage with the v2 head: AdamW (decay off the norms,
    biases, position tables and L_b scalars), cosine warm-up, beta
    warm-up for the cVAE, early stopping on val_csi_mid, per-epoch
    weights, the step-0 check against the frozen backbone, and the
    ensemble monitor for the cVAE."""
    hp = dict(finetune_cfg)
    head = hp["head"]
    variant = "cvae" if head == "swin_unet_cvae" else True
    art = finetune_suffix(variant)            # _finetuned or _finetuned_cvae
    epochs = int(hp["epochs"])
    print("=" * 70)
    print(f"COALITION-4 Training (finetune v2: {head}) - Mode: {mode}  Source: {source}")
    print("=" * 70)
    print(f"  Base model:     {base_model_path}")
    print(f"  Label type:     {label_type}")
    print(f"  Epochs:         {epochs}   batch {batch_size}")
    print(f"  Optimizer:      adamw lr={hp['initial_lr']:g} wd={hp['weight_decay']:g} "
          f"warmup={hp['warmup_epochs']} ep, clipnorm 1.0")
    print(f"  Head:           dims {hp['stage_dims']}, {hp['blocks_per_stage']} blocks/stage, "
          f"heads {hp['stage_heads']}, window {hp['window_size']}, stems {hp['stem_width']}, "
          f"expand {hp['expand_dim']}, drop_path {hp['drop_path']}")
    if head == "swin_unet_cvae":
        print(f"  cVAE:           latent {hp['latent_res']}^2 x {hp['latent_channels']}, "
              f"beta {hp['beta']} (warm-up {hp['beta_warmup_epochs']} ep), "
              f"free bits {hp['free_bits']}, {hp['n_members']} members monitored")
    if label_type == "lightning":
        print(f"  Loss:           BCE, pos_weight = min(N_neg/N_pos, {hp['pos_weight_cap']:g})"
              + (f", focal gamma {hp['lightning_gamma']:g}" if hp['lightning_gamma'] > 0 else ""))
    else:
        print(f"  Loss:           " + ("class-weighted CE (no focal) + beta * KL"
                                       if head == "swin_unet_cvae" else
                                       f"weighted focal CE + {hp['expected_class_weight']:g} * expected-class L1"))

    model, _lt = build_finetune_model_v2(
        base_model_path, hp, ones_fraction=ones_fraction,
        class_fractions=class_fractions, radar_loss_cfg=radar_loss_cfg)
    n_train = sum(int(np.prod(v.shape)) for v in model.head.trainable_variables)
    print(f"  Head parameters: {n_train:,} trainable")

    # ---- step-0 check: the head must reproduce the frozen backbone
    base = tf.keras.models.load_model(
        str(base_model_path), compile=False,
        custom_objects={"ConvBlock": ConvBlock, "ResBlock": ResBlock,
                        "GRUResBlock": GRUResBlock, "ResGRU": ResGRU})
    for inputs, _y in val_ds.take(1):
        ref = np.asarray(base(inputs, training=False), dtype=np.float32)
        got = np.asarray(model(inputs, training=False), dtype=np.float32)
        diff = float(np.max(np.abs(ref - got)))
        print(f"  Step-0 check:   max |head - backbone| = {diff:.2e} on one batch"
              + ("" if diff < 1e-3 else "  WARNING: the residual is miswired"))
    del base

    # ---- optimizer: AdamW, no decay on norms, biases, position tables, scalars
    lr, wd = float(hp["initial_lr"]), float(hp["weight_decay"])
    optimizer = None
    if hasattr(tf.keras.optimizers, "AdamW"):
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd,
                                                  global_clipnorm=1.0)
        except (TypeError, ValueError):
            optimizer = None
    if optimizer is None and hasattr(tf.keras.optimizers, "experimental"):
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr, weight_decay=wd, jit_compile=False, global_clipnorm=1.0)
    if optimizer is None:
        print("  WARNING: AdamW unavailable; Adam without weight decay")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, global_clipnorm=1.0)
    if hasattr(optimizer, "exclude_from_weight_decay"):
        optimizer.exclude_from_weight_decay(
            var_names=["layer_normalization", "_ln", "norm1", "norm2", "bias",
                       "rel_pos_bias", "lb_scale", "merge_ln", "patch_embed_ln"])
    model.compile(optimizer=optimizer)

    # ---- resume
    ckpt_cfg = checkpoint_cfg or {}
    ckpt_enabled = ckpt_cfg.get("enabled", True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"{finetune_ckpt_stem(run_tag, variant)}.keras"
    ckpt_meta = ckpt_dir / f"{finetune_ckpt_stem(run_tag, variant)}.json"
    initial_epoch = 0
    if ckpt_enabled and resume and ckpt_path.is_file():
        try:
            print(f"Resuming fine-tune from checkpoint: {ckpt_path}")
            model.load_weights(str(ckpt_path))
            if ckpt_meta.is_file():
                with open(ckpt_meta) as f:
                    initial_epoch = int(json.load(f).get("next_epoch", 0))
                print(f"  Resumed at epoch {initial_epoch}")
        except Exception as e:
            print(f"  WARNING: could not load {ckpt_path}: {e}; starting fresh")
            initial_epoch = 0

    wall_time = WallTimeCallback()
    callbacks: list = [wall_time]
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(
        cosine_warmup_schedule(initial_lr=lr, warmup_epochs=int(hp["warmup_epochs"]),
                               total_epochs=epochs, min_lr=float(hp["min_lr"])),
        verbose=1))
    if head == "swin_unet_cvae":
        callbacks.append(_BetaWarmup(hp["beta"], hp["beta_warmup_epochs"]))
        callbacks.append(_EnsembleMonitor(val_ds, hp["n_members"]))
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_csi_mid", mode="max", patience=int(hp["es_patience"]),
        restore_best_weights=True, verbose=1))
    if ckpt_enabled:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(_WeightsCheckpoint(ckpt_path, ckpt_meta))
        print(f"  Checkpoint:     per-epoch weights -> {ckpt_path}")

    head_keys = [k for k in finetune_head_defaults() if k in hp]

    def _v2_meta():
        return {
            "mode": mode, "source": source, "stage": "finetune",
            "label_type": label_type, "base_model": str(base_model_path),
            "batch_size": batch_size,
            "optimizer": "adamw", "weight_decay": wd, "initial_lr": lr,
            "warmup_epochs": int(hp["warmup_epochs"]), "min_lr": float(hp["min_lr"]),
            "head_variant": head,
            "head": {k: hp[k] for k in head_keys},
            "swin": {"window_size": hp["window_size"], "n_blocks": hp["blocks_per_stage"],
                     "num_heads": hp["stage_heads"][0], "c_shared": hp["stage_dims"][0],
                     "head_dropout": 0.0},
            "early_stopping": {"monitor": "val_csi_mid", "patience": int(hp["es_patience"])},
            "ones_fraction": float(ones_fraction) if label_type == "lightning" else None,
            "wall_times": wall_time.epoch_times,
            "total_wall_time": sum(wall_time.epoch_times),
        }
    history_writer = HistoryWriter(output_dir / f"history_{run_tag}{art}.json",
                                   _v2_meta, initial_epoch)
    callbacks.append(history_writer)      # last: it records what the others log

    print("\nStarting fine-tune training (v2)...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        initial_epoch=initial_epoch, callbacks=callbacks)

    model_path = output_dir / f"coalition_{run_tag}{art}.keras"
    model.save_weights(str(model_path))
    print(f"\nFine-tuned weights saved to: {model_path}")
    sidecar = save_model_period(model_path, ds_period, mode=mode, source=source,
                               stage="finetune", dataset_dir=dataset_dir)
    with open(sidecar, encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["feature_extractor"] = {
        "model": str(base_model_path),
        "period": fe_period.to_dict() if fe_period else None,
        "overlap_allowed": bool(allow_period_overlap),
    }
    blob["head_variant"] = head
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    history_path = history_writer.write(complete=True)
    print(f"History saved to: {history_path}")
    print("\n" + "=" * 70)
    print("Fine-tune training complete.")
    print("=" * 70)
    return model_path, history_path


# ============================================================================
# CLI
# ============================================================================

def _print_modes_and_exit() -> None:
    print("Available training modes (defined in TRAINING_MODES):\n")
    name_width = max(len(k) for k in TRAINING_MODES)
    for name, info in TRAINING_MODES.items():
        print(f"  {name:<{name_width}}  target: {info['target']}")
        print(f"  {'':<{name_width}}  {info['summary']}")
        print()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Train COALITION-4 model(s) from pre-built TF datasets. "
                    "All hyperparameters live in training.config — see the "
                    "docstring at the top of train_models.py for the list "
                    "of available modes."
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_TRAINING_CONFIG),
        help=f"Path to training.config (default: {DEFAULT_TRAINING_CONFIG}).",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        help="Train a single mode instead of the [modes].run list in the "
             "config. Hyperparameters are still read from the config "
             "(defaults + [mode.<name>] overrides). One of: "
             f"{sorted(TRAINING_MODES)}.",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Root holding patches/, split CSVs, statistics and "
             "lightning_fraction_<source>.json (default: the our_data/ "
             "beside this script, or $COALITION4_DATA_ROOT).",
    )
    parser.add_argument(
        "--datasets_root", type=str, default=None, metavar="PATH",
        help="Root holding the built TFRecord datasets (default: "
             "<data_root>/datasets, or $COALITION4_DATASETS_ROOT). Point "
             "it at another disk to keep datasets off the one holding "
             "the patch pool. Archive locks and in-use markers follow it, "
             "so restore and reclaim stay consistent with training.",
    )
    parser.add_argument(
        "--stage", type=str, default="base",
        choices=["base", "finetune", "both"],
        help="Training stage. 'base' (default) trains the standard "
             "encoder-forecaster from scratch. 'finetune' loads an "
             "existing base model and trains a Swin head on top with "
             "the backbone frozen (domain-adaptation flow). 'both' "
             "runs the two back-to-back in the same process.",
    )
    parser.add_argument(
        "--base_checkpoint", type=str, default=None,
        help="Path to the saved base model used as the frozen backbone "
             "for the finetune stage. Required when --stage=finetune; "
             "ignored when --stage=base. When --stage=both, the base "
             "model just produced is used and this flag is unused.",
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Explicit path to the dataset directory "
             "(overrides data_root/datasets/{mode}_{source}). Use with "
             "--mode only and only with --stage base.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(resolve_model_dir()),
        help="Directory to save trained model and history. Defaults to "
             "the same place the evaluators read from ($COALITION4_MODEL_DIR "
             "or the models/ beside this script), so training from a "
             "different working directory does not strand the checkpoints "
             "somewhere evaluation will not look.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any saved per-epoch checkpoint and start training "
             "from scratch. By default the run resumes from "
             "models/checkpoints/<mode>_<source>_latest.keras if it exists.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override [defaults].batch_size for this run.",
    )
    parser.add_argument(
        "--max_batches", type=int, default=None,
        help="Dry run: train and validate on the first N batches of each "
             "split, then save. Checks a configuration end to end without "
             "the cost of an epoch; combine with --epochs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the configured number of epochs for this run.",
    )
    parser.add_argument(
        "--list-modes", action="store_true",
        help="Print the available training modes with their descriptions "
             "and exit. No training is performed.",
    )
    parser.add_argument(
        "--period", type=str, default=None,
        help="Train one ensemble member, e.g. --period 2025warm. The label "
             "must appear in the registered plan; artefacts are suffixed "
             "with it. Omit to train the unscoped, whole-archive model.",
    )
    parser.add_argument(
        "--check-ensemble", action="store_true",
        help="Read the registry's most recent plan, report which member "
             "datasets exist for this mode and which are missing, then "
             "exit without training.",
    )
    parser.add_argument(
        "--allow_period_overlap", action="store_true",
        help="Proceed with the finetune stage even when the frozen feature "
             "extractor was trained on dates the dataset also covers. "
             "Scores measured under this flag are optimistic — the "
             "backbone has already seen those dates.",
    )
    parser.add_argument(
        "--head", type=str, default=None, choices=list(FINETUNE_HEADS),
        help="Fine-tune head, overriding [finetune].head: swin_unet (the "
             "deterministic two-level correction head), swin_unet_cvae "
             "(the same with the conditional VAE, rain track), or "
             "swin_legacy (the earlier single-level Swin head).",
    )

    args = parser.parse_args()

    # Resolve the roots ONCE, so every use below - including the plain
    # `Path(args.data_root)` ones - sees a real path rather than None.
    args.data_root = str(resolve_data_root(args.data_root))
    args.datasets_root = str(resolve_datasets_root(args.data_root,
                                                  args.datasets_root))

    # --- Ensemble availability check (no training) -----------------------
    # The registry says what the ensemble is supposed to contain; this
    # reports what has actually been built for the requested mode.
    if args.check_ensemble:
        if not args.mode:
            parser.error("--check-ensemble needs --mode to know which "
                         "dataset directories to look for.")
        state = require_last_state(args.data_root)
        print(f"Plan registered {state['registered_utc']} — "
              f"{state['n_members']} member(s), "
              f"{state['n_buildable']} buildable\n")
        check = check_member_datasets(
            state, args.mode, SOURCE,
            resolve_datasets_root(args.data_root, args.datasets_root))
        print(format_dataset_check(check, args.mode, SOURCE))
        sys.exit(1 if check["missing"] else 0)

    # A period must resolve to bounds somebody recorded, so a model can
    # never be trained under a label whose meaning is unknown. Two ways it
    # can, checked in the same order as create_datasets.py - the two must
    # agree, or a dataset would build under a label training then rejects.
    #
    #  1. Sequence metadata on disk. What extract_patch_seq_for_datasets.py
    #     actually wrote is the authority on what exists. This covers window
    #     tags such as `w48`, which name a sequence WINDOW (past=4/future=8)
    #     rather than an ensemble member, and so never appear in the registry.
    #  2. The registered ensemble plan, for genuine members.
    if args.period:
        seq_meta = Path(args.data_root) / sequence_meta_name(SOURCE,
                                                             args.period)
        if not seq_meta.is_file():
            state = require_last_state(args.data_root)
            if state_period(state, args.period) is None:
                known = [m["label"] for m in state.get("members", [])]
                parser.error(
                    f"Period {args.period!r} has neither sequence metadata "
                    f"at {seq_meta} nor an entry in the registered ensemble "
                    f"plan. Registered members: {known or '(none)'}. Build "
                    f"the window with `extract_patch_seq_for_datasets.py "
                    f"--period {args.period} --start ... --end ...`, or "
                    f"register a member with `create_datasets.py --mode "
                    f"<mode> --ensemble`."
                )

    if args.list_modes:
        _print_modes_and_exit()

    cfg = load_training_config(Path(args.config))

    # Decide what to run.
    if args.mode is not None:
        if args.mode not in TRAINING_MODES:
            sys.exit(
                f"ERROR: --mode {args.mode!r} is not a known training "
                f"mode. Run `python train_models.py --list-modes` for the "
                f"list."
            )
        modes_to_run = [args.mode]
    else:
        modes_to_run = cfg["modes"]
        if not modes_to_run:
            sys.exit(
                f"ERROR: [modes].run is empty in {args.config}, and no "
                f"--mode was given on the command line. Either populate "
                f"the config or pass --mode."
            )
        if args.dataset_dir is not None:
            sys.exit(
                "ERROR: --dataset_dir is only meaningful when training a "
                "single --mode. Drop --dataset_dir or pass --mode."
            )

    # Stage-specific argument validation.
    if args.stage == "finetune" and not args.base_checkpoint:
        sys.exit(
            "ERROR: --stage finetune requires --base_checkpoint pointing "
            "at the saved base `.keras` model to graft the Swin head onto."
        )
    if args.stage == "base" and args.base_checkpoint:
        print(
            "  NOTE: --base_checkpoint is ignored when --stage=base."
        )
    if args.stage == "both" and args.dataset_dir is not None:
        sys.exit(
            "ERROR: --dataset_dir is only meaningful with --stage base. "
            "Drop it when running --stage both."
        )

    print(f"Training {len(modes_to_run)} mode(s): {modes_to_run} "
          f"(source={SOURCE}, stage={args.stage})\n")

    for i, mode in enumerate(modes_to_run, start=1):
        print("#" * 70)
        print(f"# [{i}/{len(modes_to_run)}] mode: {mode}  "
              f"source: {SOURCE}  stage: {args.stage}")
        print("#" * 70)
        params = merge_for_mode(cfg, mode)
        if args.batch_size:
            params["batch_size"] = int(args.batch_size)
        if args.epochs:
            params["epochs"] = int(args.epochs)
        print(f"  Effective hyperparameters: {params}")

        base_model_path = None
        datasets_root = resolve_datasets_root(args.data_root,
                                              args.datasets_root)
        run_tag_for_mode = build_run_tag(mode, SOURCE, args.period)

        # Claim the dataset for the duration. A background archive job
        # started right after creation checks this marker before its
        # delete step, so "build a member then immediately train it" does
        # not pull the shards out from under the run.
        inuse = inuse_for(datasets_root, run_tag_for_mode)
        if not inuse.acquire():
            print(f"  NOTE: {run_tag_for_mode} is already marked in use by "
                  f"pid {inuse.held_by()}; proceeding without the marker.")
        try:

            # --- Base stage ---
            if args.stage in ("base", "both"):
                base_model_path, _ = train(
                    mode=mode,
                    data_root=args.data_root,
                    datasets_root=args.datasets_root,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    output_dir=args.output_dir,
                    dropout=params["dropout"],
                    norm=params["norm"],
                    dataset_dir=args.dataset_dir,
                    source=SOURCE,
                    shuffle_buffer=params["shuffle_buffer"],
                    mixed_precision=params["mixed_precision"],
                    lr_schedule_cfg=cfg["lr_schedule"],
                    early_stopping_cfg=cfg["early_stopping"],
                    checkpoint_cfg=cfg["checkpointing"],
                    radar_loss_cfg=cfg["radar_loss"],
                    resume=cfg["checkpointing"].get("resume", True)
                           and not args.fresh,
                    period=args.period,
                )

            # --- Finetune stage ---
            if args.stage in ("finetune", "both"):
                ft_base = (
                    base_model_path if args.stage == "both"
                    else Path(args.base_checkpoint)
                )
                ft_cfg = dict(cfg["finetune"])
                if args.head:
                    ft_cfg["head"] = args.head
                if args.max_batches:
                    ft_cfg["max_batches"] = int(args.max_batches)
                if args.epochs:
                    ft_cfg["epochs"] = int(args.epochs)
                train_finetune(
                    mode=mode,
                    data_root=args.data_root,
                    datasets_root=args.datasets_root,
                    base_model_path=ft_base,
                    output_dir=args.output_dir,
                    source=SOURCE,
                    batch_size=params["batch_size"],
                    finetune_cfg=ft_cfg,
                    shuffle_buffer=params["shuffle_buffer"],
                    mixed_precision=params["mixed_precision"],
                    early_stopping_cfg=cfg["early_stopping"],
                    checkpoint_cfg=cfg["checkpointing"],
                    radar_loss_cfg=cfg["radar_loss"],
                    resume=cfg["checkpointing"].get("resume", True)
                           and not args.fresh,
                    period=args.period,
                    allow_period_overlap=args.allow_period_overlap,
                )
        finally:
            inuse.release()

    print("\nAll requested training runs completed.")


if __name__ == "__main__":
    main()
