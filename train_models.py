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

       python train_models.py --config training.config --mode mtg_opera_full

Available training modes
------------------------
The mode determines which input groups feed the model and what the target
is. The dataset for each mode must already exist under
`our_data/datasets/<mode>/` (created by create_datasets.py — see its CLI
choices for the full list).

  - mtg_lightning
        Inputs:  radar + LINET lightning + MTG vis_06 (HR) + MTG IR/WV (MR).
        Target:  binary lightning occurrence.

  - mtg_radar
        Inputs:  same channel set as mtg_lightning.
        Target:  5-class precipitation (RZC bins).

  - mtg_radar_continuous
        Inputs:  same as mtg_radar.
        Target:  continuous RZC regression in [0, 1] (normalised).

  - mtg_opera_radar_only
        Inputs:  MTG vis_06 (HR) + OPERA reflectivity + rainfall_rate (MR).
                 No MTG IR/WV. No lightning.
        Target:  opera_rainfall_rate 5-class (same bin edges as RZC).

  - mtg_opera_mtgmr
        Inputs:  mtg_opera_radar_only + MTG IR/WV in MR.
        Target:  same as mtg_opera_radar_only.

  - mtg_lightning_opera
        Inputs:  lightning (density / current / occurrence) + MTG vis_06
                 in HR; OPERA reflectivity + rainfall_rate + MTG IR/WV
                 in MR. No ANM radar. No NWCSAF.
        Target:  opera_rainfall_rate 5-class.

  - mtg_lightning_opera_occurrence
        Inputs:  same as mtg_lightning_opera.
        Target:  lightning binary occurrence (focal loss). Pairs with
                 mtg_lightning_opera as the dual-target experiment on
                 OPERA-driven sample selection.

Hyperparameters, the run list, the LR schedule, and the early-stopping
configuration all live in `training.config`. See the docstring at the top
of that file for the editable fields.

Requires:
    - TensorFlow 2.x with GPU support
    - Pre-built TF datasets from create_datasets.py
    - our_data/lightning_fraction.json (for lightning modes only)
"""

import argparse
import configparser
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# =============================================================================
# Documented training modes
# =============================================================================
#
# Used to validate `--mode` / `[modes].run` against a known set and to make
# the registry of available modes discoverable. Kept in sync with
# create_datasets.get_mode_config() — when you add a new mode there, add
# a matching entry here so `--list-modes` and config validation keep working.

TRAINING_MODES: dict[str, dict[str, str]] = {
    "mtg_lightning": {
        "target":  "lightning (binary occurrence)",
        "summary": "Radar + lightning + MTG (vis_06 HR / IR/WV MR).",
    },
    "mtg_radar": {
        "target":  "RZC 5-class precipitation",
        "summary": "Same channel set as mtg_lightning, different label head.",
    },
    "mtg_radar_continuous": {
        "target":  "RZC continuous regression",
        "summary": "Same channels as mtg_radar with a continuous target.",
    },
    "mtg_opera_radar_only": {
        "target":  "opera_rainfall_rate 5-class",
        "summary": "Baseline: MTG vis_06 + OPERA. No MTG IR/WV, no lightning.",
    },
    "mtg_opera_mtgmr": {
        "target":  "opera_rainfall_rate 5-class",
        "summary": "Baseline + MTG IR/WV in MR.",
    },
    "mtg_lightning_opera": {
        "target":  "opera_rainfall_rate 5-class",
        "summary": "Lightning + MTG vis_06 in HR; OPERA + MTG IR/WV in MR; "
                   "OPERA rainfall as label.",
    },
    "mtg_lightning_opera_occurrence": {
        "target":  "lightning binary occurrence",
        "summary": "Same inputs as mtg_lightning_opera; target is lightning "
                   "occurrence instead of OPERA rainfall.",
    },
}


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

    # [checkpointing]
    c = parser["checkpointing"] if parser.has_section("checkpointing") else {}
    checkpointing = {
        "enabled": _coerce(c.get("enabled", "true"), bool),
        "resume":  _coerce(c.get("resume", "true"), bool),
    }

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

def build_coalition_model(input_shapes, label_type, past_timesteps=3,
                          future_timesteps=3, dropout=0, norm=None,
                          ones_fraction=0.0106):
    """Build the COALITION encoder-forecaster model dynamically.

    The model architecture adapts to whatever inputs the dataset provides.
    Input shapes are read from the dataset metadata so adding or removing
    input groups (e.g. dropping NWCSAF) requires no code changes here.

    Args:
        input_shapes: dict from metadata.json["input_shapes"], e.g.
            {"past_hr": [3, 256, 256, 10], "past_mr": [3, 128, 128, 4],
             "past_lr": [3, 64, 64, 8]}
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
        divisor = max_res // res  # 1 for HR, 2 for MR, 4 for LR

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
        loss = tf.keras.losses.CategoricalCrossentropy()
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


def build_finetune_model(base_model_path, finetune_cfg, ones_fraction):
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
    """
    base = tf.keras.models.load_model(
        str(base_model_path),
        custom_objects={
            "WeightedFocalLoss": WeightedFocalLoss,
            "iou_metric": iou_metric,
            "true_pos":   true_pos,
            "false_pos":  false_pos,
            "false_neg":  false_neg,
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
        loss = tf.keras.losses.CategoricalCrossentropy()
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

def load_ones_fraction(data_root):
    """Load lightning occurrence fraction from pre-computed JSON.

    Reads our_data/lightning_fraction.json and returns the 'occurrence'
    fraction, which is the correct value for the binary focal loss.

    Args:
        data_root: path to our_data/ directory

    Returns:
        float: occurrence fraction

    Raises:
        FileNotFoundError: if lightning_fraction.json does not exist
        KeyError: if 'occurrence' key is missing from JSON
    """
    json_path = Path(data_root) / "lightning_fraction.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Lightning fraction file not found: {json_path}\n"
            f"Run the lightning fraction computation script first to generate it."
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
          resume=True):
    """Main training function (base stage).

    Args:
        mode: training-mode name (see TRAINING_MODES at the top of this
            file). Used to name the saved model / history files and to
            locate the default dataset directory.
        data_root: path to our_data/ containing datasets/{mode}_{source}/
            and lightning_fraction.json.
        epochs: number of training epochs.
        batch_size: training batch size.
        output_dir: where to save model + history.
        dropout: dropout rate.
        norm: normalization type ('batch', 'layer', or None).
        dataset_dir: explicit path to the dataset directory. When
            provided, overrides the default data_root/datasets/{mode}_{source}.
        source: extract_patch_seq source ('dbscan' = patch_index.csv,
            or 'lightning' = lightning_patches.csv) the dataset was built
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
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Mode + source together are the unique experiment identifier - used
    # everywhere we'd previously used just `mode` so the DBSCAN-driven
    # and lightning-driven runs can coexist on disk.
    run_tag = f"{mode}_{source}"

    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)
    else:
        dataset_dir = data_root / "datasets" / run_tag

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

    # Load ones_fraction for lightning modes from pre-computed JSON
    if label_type == "lightning":
        ones_fraction = load_ones_fraction(data_root)
    else:
        ones_fraction = 0.0106  # unused for radar

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

    # Save history
    history_data = {
        "mode": mode,
        "source": source,
        "stage": "base",
        "label_type": label_type,
        "epochs_completed": len(history.history.get("loss", [])),
        "batch_size": batch_size,
        "dropout": dropout,
        "norm": norm,
        "ones_fraction": float(ones_fraction) if label_type == "lightning" else None,
        "wall_times": wall_time.epoch_times,
        "total_wall_time": sum(wall_time.epoch_times),
        "history": {k: [float(v) for v in vals]
                    for k, vals in history.history.items()},
    }
    history_path = output_dir / f"history_{run_tag}.json"
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
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
                   resume=True):
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

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{mode}_{source}"

    dataset_dir = data_root / "datasets" / run_tag
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
        ones_fraction = load_ones_fraction(data_root)
    else:
        ones_fraction = 0.0106  # unused for radar

    print("\nLoading datasets...")
    train_ds = load_dataset(train_dir, batch_size,
                             shuffle=True, shuffle_buffer=shuffle_buffer)
    val_ds = load_dataset(val_dir, batch_size, shuffle=False)
    print("  Datasets loaded")

    # Build the fine-tune model: frozen backbone + Swin head + per-step heads.
    print("\nBuilding fine-tune model (frozen backbone + Swin head)...")
    model, loss, metrics = build_finetune_model(
        base_model_path=base_model_path,
        finetune_cfg=finetune_cfg,
        ones_fraction=ones_fraction,
    )

    # Pick optimizer. AdamW is the default for fine-tune; falling back
    # to plain Adam is supported for parity experiments.
    if finetune_cfg["optimizer"] == "adamw":
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=finetune_cfg["initial_lr"],
                weight_decay=finetune_cfg["weight_decay"],
            )
        except AttributeError:
            # TF < 2.11 - fall back to the experimental package
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=finetune_cfg["initial_lr"],
                weight_decay=finetune_cfg["weight_decay"],
            )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=finetune_cfg["initial_lr"],
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

    history_data = {
        "mode": mode,
        "source": source,
        "stage": "finetune",
        "label_type": label_type,
        "base_model": str(base_model_path),
        "epochs_completed": len(history.history.get("loss", [])),
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
        "history": {k: [float(v) for v in vals]
                    for k, vals in history.history.items()},
    }
    history_path = output_dir / f"history_{run_tag}_finetuned.json"
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
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
        "--data_root", type=str, default="./our_data",
        help="Root directory containing datasets/ and lightning_fraction.json.",
    )
    parser.add_argument(
        "--source", type=str, default="dbscan",
        choices=["dbscan", "lightning"],
        help="Which extract_patch_seq source the training dataset was "
             "built from. 'dbscan' (default) = patch_index.csv from "
             "identify_patches (whichever sensor that script was run "
             "with). 'lightning' = lightning_patches.csv from "
             "identify_lightning_periods. Selects datasets/<mode>_<source>/ "
             "as the input and suffixes every saved checkpoint / model / "
             "history with `_<source>`.",
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
        "--output_dir", type=str, default="./models",
        help="Directory to save trained model and history.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any saved per-epoch checkpoint and start training "
             "from scratch. By default the run resumes from "
             "models/checkpoints/<mode>_<source>_latest.keras if it exists.",
    )
    parser.add_argument(
        "--list-modes", action="store_true",
        help="Print the available training modes with their descriptions "
             "and exit. No training is performed.",
    )

    args = parser.parse_args()

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
          f"(source={args.source}, stage={args.stage})\n")

    for i, mode in enumerate(modes_to_run, start=1):
        print("#" * 70)
        print(f"# [{i}/{len(modes_to_run)}] mode: {mode}  "
              f"source: {args.source}  stage: {args.stage}")
        print("#" * 70)
        params = merge_for_mode(cfg, mode)
        print(f"  Effective hyperparameters: {params}")

        base_model_path = None

        # --- Base stage ---
        if args.stage in ("base", "both"):
            base_model_path, _ = train(
                mode=mode,
                data_root=args.data_root,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                output_dir=args.output_dir,
                dropout=params["dropout"],
                norm=params["norm"],
                dataset_dir=args.dataset_dir,
                source=args.source,
                shuffle_buffer=params["shuffle_buffer"],
                mixed_precision=params["mixed_precision"],
                lr_schedule_cfg=cfg["lr_schedule"],
                early_stopping_cfg=cfg["early_stopping"],
                checkpoint_cfg=cfg["checkpointing"],
                resume=cfg["checkpointing"].get("resume", True)
                       and not args.fresh,
            )

        # --- Finetune stage ---
        if args.stage in ("finetune", "both"):
            ft_base = (
                base_model_path if args.stage == "both"
                else Path(args.base_checkpoint)
            )
            train_finetune(
                mode=mode,
                data_root=args.data_root,
                base_model_path=ft_base,
                output_dir=args.output_dir,
                source=args.source,
                batch_size=params["batch_size"],
                finetune_cfg=cfg["finetune"],
                shuffle_buffer=params["shuffle_buffer"],
                mixed_precision=params["mixed_precision"],
                early_stopping_cfg=cfg["early_stopping"],
                checkpoint_cfg=cfg["checkpointing"],
                resume=cfg["checkpointing"].get("resume", True)
                       and not args.fresh,
            )

    print("\nAll requested training runs completed.")


if __name__ == "__main__":
    main()
