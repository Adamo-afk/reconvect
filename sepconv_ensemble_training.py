"""
train_sepconv_ensemble.py — SepConv Ensemble (Regression) for COALITION-4
==========================================================================
Faithful ensemble: three base models (Bm1, Bm2, Bm3), each predicting a
single lead time as continuous rain rate via regression.

Output: sigmoid [0,1] (continuous), NOT softmax classification.
Loss: weighted MSE from Czibula et al. 2024 (upweights high precipitation).
Labels: one-hot from create_datasets.py auto-converted to continuous midpoints.
Classification: recovered at evaluation time via post-processing thresholds.

Radar-only by design, and compared against `opera_radar_only_rainfall`
on the identical input tensor. Both predictions are binned in mm/h at
create_datasets.RAINFALL_CLASS_EDGES, so the two cannot be told apart by
their thresholds.

Usage:
    python sepconv_ensemble_training.py --period w44
    python sepconv_ensemble_training.py --period w44 --lead 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, SeparableConv2D, Activation, Concatenate,
    UpSampling2D, Lambda
)
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback

from create_datasets import get_mode_config, load_tfrecord_dataset
from pipeline_config import (
    SOURCE,
    resolve_data_root,
    resolve_datasets_root,
    resolve_model_dir,
)
from train_models import (
    DEFAULT_TRAINING_CONFIG,
    build_run_tag,
    load_training_config,
    _ResumableCheckpoint,
)


# ============================================================================
# Mode configurations
# ============================================================================

# Branch shapes mirror create_datasets.get_mode_config for
# `opera_sepconv_logz`, so the baseline consumes byte-identical batches:
#   HR  = OPERA rainfall_rate at 256 px                     -> 1 channel
# The ablation it is compared against carries the same single field in
# the same tier, so a gap between them is architecture, not input.
# The baseline is radar-only by design — no MTG, no LINET. Modality
# enrichment is what RECONVECT is being credited for, so handing it to
# the baseline would erase the very difference under test.
SEPCONV_MODE = "opera_sepconv_logz"

# Architecture constants (from paper)
HR_SIZE = 256
KERNEL = (5, 5)
ACTIVATION = 'selu'
DEPTH_MULT = 1

# Base models, per the paper: Bm1, Bm3, Bm5 predict 1, 3 and 5 steps
# ahead. On our 15-minute grid that is 15, 45 and 75 minutes — the paper's
# 6-minute grid made them 6, 18 and 30.
SEPCONV_BASE_LEADS = (1, 3, 5)
STEP_MINUTES = 15
MAX_COMPOSED_STEPS = 8

LEAD_MINUTES = {k: k * STEP_MINUTES for k in range(1, MAX_COMPOSED_STEPS + 1)}
LEAD_NAMES = {k: f"t+{v}" for k, v in LEAD_MINUTES.items()}

# Which of the 5 available past frames (t-4 .. t0) feed the model during
# TRAINING. The paper trains every base model on the last four frames,
# Phi_i(M_t-3, M_t-2, M_t-1, M_t) = M_t+i; t-4 is used only by the
# composition scheme at inference. Index 0 is t-4.
TRAIN_FRAME_SLICE = slice(1, 5)

# Post-processing thresholds to recover 5 classes (used in evaluation)
# Raw rain rate normalized by /70: thresholds at 10, 20, 30, 40 mm/h
THRESHOLDS_NORM = [10.0 / 70.0, 20.0 / 70.0, 30.0 / 70.0, 40.0 / 70.0]


# ============================================================================
# Weighted MSE loss (from paper)
# ============================================================================

# ============================================================================
# Base model (regression: sigmoid output)
# ============================================================================

# Paper architecture constants (Czibula et al. 2024, Fig. 2).
SEPCONV_PAST_STEPS = 4        # l = 4 previous frames, one Input each
SEPCONV_EXPAND = 51           # per-input SeparableConv2D output width
SEPCONV_TRUNK_WIDE = 100      # first reduction from the 204-wide concat
SEPCONV_TRUNK_NARROW = 50     # second reduction
SEPCONV_REPEATS = 4           # the "X4" blocks in Fig. 2


def build_sepconv_base_model(lead_steps, input_size=HR_SIZE,
                             in_channels=1, out_channels=1):
    """One SepConv base model, faithful to Czibula et al. 2024 Fig. 2.

    Topology, per the paper:
        4 Inputs (one per past frame)
          -> SeparableConv2D  C_in -> 51   + SELU     (per input)
          -> Concatenate                    = 204
          -> SeparableConv2D  204 -> 100   + SELU
          -> SeparableConv2D  100 -> 100   + SELU     x4
          -> SeparableConv2D  100 -> 50    + SELU
          -> SeparableConv2D   50 -> 50    + SELU     x4
          -> SeparableConv2D   50 -> C_out            (LINEAR)
    ten hidden separable layers plus the output layer.

    Two documented deviations from the paper:

    1. `C_in = 1`, not 6. The paper stacks six reflectivity elevations
       (R01-R04, R06-R07); we have a single OPERA composite rainfall-rate
       field. The 51-wide expansion is kept as published, so the concat is
       still 204 and the trunk is untouched — only the first layer's input
       depth changes.

    2. The output layer is LINEAR, where the paper applies SELU after
       every separable layer. Our target is log_zscore rain rate, whose
       dry point mass sits at z = -0.291; SELU's negative saturation floor
       is -lambda*alpha = -1.758, so SELU would in fact reach it. Linear
       is kept anyway because the upper tail runs to z = +9.5 and there is
       no reason to put a saturating nonlinearity in front of a regression
       target we later exponentiate — an error at the top of the range
       becomes multiplicative in mm/h.

    Args:
        lead_steps: how many steps ahead this base model predicts (1, 3
            or 5), used only for naming.
        input_size: patch edge, 256.
        in_channels: channels per past frame (1 for OPERA rainfall).
        out_channels: predicted channels (1).

    Returns:
        keras Model with `SEPCONV_PAST_STEPS` named inputs.
    """
    inputs = [
        Input(shape=(input_size, input_size, in_channels), name=f"past_t{t}")
        for t in range(SEPCONV_PAST_STEPS)
    ]

    # --- per-input expansion -------------------------------------------
    expanded = []
    for t, inp in enumerate(inputs):
        x = SeparableConv2D(SEPCONV_EXPAND, KERNEL, padding='same',
                            depth_multiplier=DEPTH_MULT,
                            name=f"expand_t{t}")(inp)
        x = Activation(ACTIVATION, name=f"expand_selu_t{t}")(x)
        expanded.append(x)

    x = Concatenate(axis=-1, name="concat")(expanded)   # 4 x 51 = 204

    # --- trunk: 204 -> 100 (x5) -> 50 (x5) ------------------------------
    x = SeparableConv2D(SEPCONV_TRUNK_WIDE, KERNEL, padding='same',
                        name="trunk_100_0")(x)
    x = Activation(ACTIVATION, name="trunk_100_selu_0")(x)
    for i in range(1, SEPCONV_REPEATS + 1):
        x = SeparableConv2D(SEPCONV_TRUNK_WIDE, KERNEL, padding='same',
                            name=f"trunk_100_{i}")(x)
        x = Activation(ACTIVATION, name=f"trunk_100_selu_{i}")(x)

    x = SeparableConv2D(SEPCONV_TRUNK_NARROW, KERNEL, padding='same',
                        name="trunk_50_0")(x)
    x = Activation(ACTIVATION, name="trunk_50_selu_0")(x)
    for i in range(1, SEPCONV_REPEATS + 1):
        x = SeparableConv2D(SEPCONV_TRUNK_NARROW, KERNEL, padding='same',
                            name=f"trunk_50_{i}")(x)
        x = Activation(ACTIVATION, name=f"trunk_50_selu_{i}")(x)

    # Linear output — see deviation 2 above. Do NOT add an activation
    # here: the value is in log_zscore space and is exponentiated
    # downstream, so saturation is silently destructive.
    outputs = SeparableConv2D(out_channels, KERNEL, padding='same',
                              name="output_conv")(x)

    return Model(inputs=inputs, outputs=outputs,
                 name=f"sepconv_bm{lead_steps}")


# ============================================================================
# Dataset: extract single lead time (labels already continuous)
# ============================================================================

class WeightedMSELogZ(tf.keras.losses.Loss):
    """MSE re-weighted by rainfall class, evaluated in log_zscore space.

    Necessary, not cosmetic: 99.815% of training pixels are class 0, and
    in this space the dry point mass sits at z = -0.291 while 10-40 mm/h
    spans z = +5.55..+6.72. Plain MSE is minimised by emitting -0.291
    everywhere, which is a worse failure mode than the smoothing the
    paper reports.

    Weights come from measured class frequencies (see
    train_models.load_sepconv_class_weights) and the class of a pixel is
    decided from the TARGET, so the weighting is a property of the data
    rather than of the current prediction.

    The paper's own weighting is unpublished; this scheme is ours and is
    reported as ours.
    """

    def __init__(self, edges_z, weights, name="weighted_mse_logz", **kw):
        super().__init__(name=name, **kw)
        self.edges_z = [float(e) for e in edges_z]
        self.weights = [float(w) for w in weights]

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Class index from the target: 0 below the first edge, then one
        # step per edge crossed.
        cls = tf.zeros_like(y_true, dtype=tf.int32)
        for edge in self.edges_z:
            cls += tf.cast(y_true >= edge, tf.int32)
        w = tf.gather(tf.constant(self.weights, tf.float32), cls)
        return tf.reduce_mean(w * tf.square(y_true - y_pred))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"edges_z": self.edges_z, "weights": self.weights})
        return cfg


def build_sepconv_loss(data_root, source, weights_period=None,
                       stats_period=None):
    """Weighted MSE with class edges mapped into log_zscore space.

    The two scopes are deliberately opposite, and conflating them is a
    silent-corruption bug in both directions:

    * `stats_period` — the mean/std defining the space. Scoped to this
      model's own training split. A z-value only means something relative
      to the constants that produced it, so the invariant that matters is
      that TRAINING and INVERSION use the same ones: whatever is passed
      here must also be passed to `sepconv_predict.to_mmh`. Mixing them
      recovers the wrong mm/h, monotonically worse with intensity.

      Not shared with RECONVECT on purpose. Its training split contains
      36 of this model's test timestamps, so borrowing its statistics
      would let them into the constants defining this model's space.

    * `weights_period` — the class balance of the split this model
      actually trains on. Scoped for the same reason: the baseline uses a
      longer sequence window and therefore a different set of rows.
    """
    from create_datasets import (RAINFALL_CLASS_EDGES, mmh_to_logz,
                                 set_normalization_stats_path)
    from periods import normalization_stats_name, data_tag
    from train_models import load_sepconv_class_weights

    stats_file = normalization_stats_name(source, stats_period)
    set_normalization_stats_path(Path(data_root) / stats_file)

    weights = load_sepconv_class_weights(data_root, source, weights_period)
    edges_z = [float(mmh_to_logz(e)) for e in RAINFALL_CLASS_EDGES]

    print(f"  Loss: weighted MSE in log_zscore space")
    print(f"    stats (shared)     : {stats_file}")
    print(f"    weights scope      : "
          f"opera_rainfall_fraction_{data_tag(source, weights_period)}.json")
    print(f"    class edges (mm/h) : {list(RAINFALL_CLASS_EDGES)}")
    print(f"    class edges (z)    : {[round(e, 4) for e in edges_z]}")
    print(f"    weights            : {[round(w, 1) for w in weights]}")
    return WeightedMSELogZ(edges_z, weights)


def extract_lead_time(inputs, labels, lead_steps):
    """Reshape one sample for a SepConv base model.

    Inputs arrive as a dict with `past_hr` of shape (5, 256, 256, 1) —
    frames t-4 .. t0. Training uses the last four, matching the paper's
    Phi_i(M_t-3, M_t-2, M_t-1, M_t). Labels arrive as (8, 256, 256, 1);
    a base model targets exactly one of them.
    """
    frames = inputs["past_hr"][TRAIN_FRAME_SLICE]
    model_inputs = {f"past_t{i}": frames[i] for i in range(SEPCONV_PAST_STEPS)}
    return model_inputs, labels[lead_steps - 1]


def prepare_dataset(ds_path, lead_steps, batch_size, shuffle=False,
                    shuffle_buffer=1000):
    # TFRecord shards written by create_datasets.py; the mode config
    # supplies the parse signature.
    ds = load_tfrecord_dataset(Path(ds_path), get_mode_config(SEPCONV_MODE))
    ds = ds.map(lambda x, y: extract_lead_time(x, y, lead_steps),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================================
# Wall time callback
# ============================================================================

class WallTimeCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.train_start = None

    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        cumulative = time.time() - self.train_start
        print(f"  Epoch {epoch + 1} wall time: {elapsed:.1f}s "
              f"(cumulative: {cumulative:.1f}s)")


# ============================================================================
# Train one base model
# ============================================================================

def train_base_model(lead_steps, data_root, model_dir, epochs, batch_size,
                     ds_root, checkpoint_cfg=None, resume=True,
                     learning_rate=1e-3, lr_patience=5, es_patience=10,
                     period=None):
    """Train one base model Bm{lead_steps}.

    Optimiser parity with the paper: AMSGrad variant of Adam at lr 1e-3,
    with a callback that halves the rate when validation loss plateaus.
    The paper does not state its stopping criterion or epoch budget, so
    `--es_patience` / `--epochs` are ours and are recorded in the history
    JSON alongside the sweep that chose them.
    """
    data_root = Path(data_root)
    lead_name = LEAD_NAMES[lead_steps]

    print(f"\n{'-' * 60}")
    print(f"  Training Bm{lead_steps}: {lead_steps}-step model, "
          f"{LEAD_MINUTES[lead_steps]} min from its own window end")
    print(f"{'-' * 60}")

    # `ds_root` is passed in, never rebuilt here: train() already
    # resolved it against --datasets_root and checked the splits exist.
    # Deriving it a second time is how the check came to pass on one disk
    # while the loader looked on another.
    train_ds = prepare_dataset(ds_root / "train", lead_steps, batch_size, True)
    val_ds = prepare_dataset(ds_root / "validation", lead_steps, batch_size)

    model = build_sepconv_base_model(lead_steps)
    print(f"  Parameters: {model.count_params():,}")

    # AMSGrad-Adam, lr 1e-3 — as published.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         amsgrad=True)
    # Both scopes follow this run's own split. Statistics are NOT shared
    # with RECONVECT: its training split contains 36 of the baseline's
    # own test timestamps, which would put them inside the constants that
    # define the baseline's space. What actually has to hold is that
    # training and inversion agree — so sepconv_predict must denormalise
    # with this same period.
    loss = build_sepconv_loss(data_root, SOURCE,
                              weights_period=period, stats_period=period)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mse", "mae"])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=lr_patience,
        min_lr=1e-15, min_delta=0.0001, verbose=1)
    early_stop = EarlyStopping(
        monitor='val_loss', patience=es_patience,
        restore_best_weights=True, verbose=1)
    wall_timer = WallTimeCallback()
    callbacks = [reduce_lr, early_stop, wall_timer]

    run_tag = build_run_tag(SEPCONV_MODE, SOURCE, period)

    # Per-epoch rolling checkpoint, mirroring RECONVECT. Each base model
    # gets its own file: three are trained per invocation, and a shared
    # one would have Bm3 resume from Bm1's weights.
    ckpt_cfg = checkpoint_cfg or {}
    initial_epoch = 0
    if ckpt_cfg.get("enabled", True):
        ckpt_dir = Path(model_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"sepconv_{run_tag}_bm{lead_steps}_latest.keras"
        meta_path = ckpt_dir / f"sepconv_{run_tag}_bm{lead_steps}_latest.json"
        if resume and ckpt_path.is_file():
            try:
                print(f"  Resuming from checkpoint: {ckpt_path}")
                model.load_weights(str(ckpt_path))
                if meta_path.is_file():
                    with open(meta_path) as f:
                        initial_epoch = int(json.load(f).get("next_epoch", 0))
                    print(f"    Resumed at epoch {initial_epoch}")
            except Exception as e:
                print(f"  WARNING: could not load {ckpt_path}: {e}")
                print(f"  Starting fresh.")
                initial_epoch = 0
        callbacks.append(_ResumableCheckpoint(str(ckpt_path),
                                              str(meta_path)))

    if initial_epoch >= epochs:
        print(f"  Already trained to epoch {initial_epoch} of {epochs}; "
              f"nothing to do. Delete the checkpoint or raise epochs to "
              f"continue.")

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks, verbose=1)
    model_path = Path(model_dir) / f"sepconv_{run_tag}_bm{lead_steps}.keras"
    model.save(str(model_path))
    print(f"  Saved: {model_path}")

    return {
        "lead_idx": lead_steps, "lead_name": lead_name,
        "lead_minutes": LEAD_MINUTES[lead_steps],
        "hyperparameters": {
            "learning_rate": learning_rate,
            "optimizer": "adam_amsgrad",
            "lr_schedule": "ReduceLROnPlateau factor=0.5 "
                           f"patience={lr_patience}",
            "early_stopping_patience": es_patience,
            "epochs_budget": epochs,
            "batch_size": batch_size,
        },
        "history": {k: [float(v) for v in vals]
                    for k, vals in history.history.items()},
        "wall_times": wall_timer.epoch_times,
        "total_wall_time": sum(wall_timer.epoch_times),
        "epochs_completed": len(history.history.get("loss", [])),
        "model_params": model.count_params(),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }


# ============================================================================
# Main
# ============================================================================

def train(data_root, model_dir, epochs=50, batch_size=32, lead=None,
          learning_rate=1e-3, lr_patience=5, es_patience=6, period=None,
          datasets_root=None, checkpoint_cfg=None, resume=True):
    data_root = resolve_data_root(data_root)
    datasets_root = resolve_datasets_root(data_root, datasets_root)
    model_dir = resolve_model_dir(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    ds_root = datasets_root / build_run_tag(SEPCONV_MODE, SOURCE, period)
    for split in ["train", "validation"]:
        if not (ds_root / split).exists():
            raise FileNotFoundError(
                f"Dataset not found: {ds_root / split}\n"
                f"The SepConv baseline needs its own past=4/future=8 "
                f"sequence set. Build it with:\n"
                f"    python extract_patch_seq_for_datasets.py "
                f"--past 4 --future 8 --period <label> --start ... --end ...\n"
                f"    python create_datasets.py --mode {SEPCONV_MODE} "
                f"--period <label>")

    # NOTE: no mixed precision here. The target lives in log_zscore space
    # with a tail at z = +9.5 and a loss weight up to 1000x, so the
    # weighted squared error reaches ~1e5 — comfortably inside fp32 but
    # close enough to fp16's 65504 ceiling that overflow is a real risk.
    # The model is 100k parameters; there is nothing to gain by taking it.
    tf.keras.mixed_precision.set_global_policy('float32')

    mode_config = get_mode_config(SEPCONV_MODE)
    n_train = sum(1 for _ in load_tfrecord_dataset(ds_root / "train", mode_config))
    n_val = sum(1 for _ in load_tfrecord_dataset(ds_root / "validation", mode_config))

    print("=" * 70)
    print(f"SepConv-ens baseline (Czibula et al. 2024) — {SEPCONV_MODE}")
    print("=" * 70)
    print(f"  Inputs   : {SEPCONV_PAST_STEPS} x OPERA rainfall_rate, "
          f"radar-only by design")
    print(f"  Target   : log_zscore rain rate, LINEAR output head")
    print(f"  Optimizer: Adam (AMSGrad, lr={learning_rate}, halve on plateau)")
    print(f"  Precision: float32 (see note in train())")
    print(f"  Train: {n_train}, Val: {n_val}")

    leads_to_train = [lead] if lead is not None else list(SEPCONV_BASE_LEADS)

    # A base model is named for its OWN lead - how far ahead it predicts
    # from the end of its own input window - not for the horizon it
    # serves. Bm5 is a 5-step model, but it supplies t+4 because its
    # window is shifted back one step. Spelling both out, because
    # "Bm5 (t+75min)" reads as a 75-minute forecast and the ensemble
    # makes no such thing.
    from sepconv_compose import COMPOSITION, OBSERVED_ONLY_STEPS
    serves = {}
    for step, lead_k, _offsets in COMPOSITION:
        if step in OBSERVED_ONLY_STEPS:
            serves.setdefault(lead_k, []).append(step)
    print("  Base models (each named by its own lead, not the horizon):")
    for k in leads_to_train:
        steps = serves.get(k, [])
        supplies = (", ".join(f"t+{t} ({LEAD_MINUTES[t]} min)" for t in steps)
                    if steps else "not used by the composition")
        print(f"    Bm{k}: {k}-step model "
              f"({LEAD_MINUTES[k]} min from its window end) -> {supplies}")
    horizon = max(OBSERVED_ONLY_STEPS)
    print(f"  Forecast horizon: t+{horizon} = {LEAD_MINUTES[horizon]} min "
          f"(nothing autoregressive)")
    total_start = time.time()
    all_results = {}

    for lead_steps in leads_to_train:
        all_results[f"bm{lead_steps}"] = train_base_model(
            lead_steps, data_root, model_dir, epochs, batch_size,
            ds_root=ds_root,
            checkpoint_cfg=checkpoint_cfg, resume=resume,
            learning_rate=learning_rate, lr_patience=lr_patience,
            es_patience=es_patience, period=period)

    total_time = time.time() - total_start

    history_data = {
        "mode": SEPCONV_MODE, "architecture": "sepconv_ensemble",
        "output_type": "regression",
        "base_models": all_results,
        "total_wall_time": total_time,
        "batch_size": batch_size, "n_train": n_train, "n_val": n_val,
        "total_params": sum(r["model_params"] for r in all_results.values()),
        "label_info": {
            "type": "continuous",
            "normalization": "opera_rainfall_rate / 70 → [0, 1]",
            "thresholds_for_classification": THRESHOLDS_NORM,
        },
        "config": {
            "kernel": list(KERNEL), "activation": ACTIVATION,
            "loss": "weighted_MSE [15,1,2,7,15,30,1000]",
            "optimizer": "Adam(AMSGrad)",
        },
    }

    # Tagged like the weights beside it: several windows are trained
    # from the same mode, and one filename for all of them would keep
    # only the last run's history.
    history_path = (Path(model_dir)
                    / f"history_sepconv_{build_run_tag(SEPCONV_MODE, SOURCE, period)}.json")
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)

    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING COMPLETE")
    for key, r in all_results.items():
        print(f"  {key} ({r['lead_name']}): {r['epochs_completed']} epochs, "
              f"val_loss={r['final_val_loss']:.6f}, params={r['model_params']:,}")
    print(f"  Total params: {history_data['total_params']:,}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train the SepConv-ens baseline (Czibula et al. 2024). "
                    "Radar-only by design; consumes the "
                    f"{SEPCONV_MODE} dataset built on a past=4/future=8 "
                    "sequence window.")
    parser.add_argument("--data_root", type=str, default=str(resolve_data_root()))
    parser.add_argument("--model_dir", type=str, default=str(resolve_model_dir()))
    parser.add_argument("--period", type=str, default=None,
                        help="Ensemble member label, if the baseline is "
                             "being trained per period. Omit for the "
                             "whole-archive run used by the comparison.")
    parser.add_argument("--config", type=str,
                        default=str(DEFAULT_TRAINING_CONFIG), metavar="PATH",
                        help="Hyperparameters come from this file - the SAME "
                             "one RECONVECT uses - so both halves of the "
                             "comparison train under identical settings. "
                             "[defaults] supplies epochs and batch_size; the "
                             "optional [sepconv] section overrides them and "
                             "carries the ReduceLROnPlateau patience. Flags "
                             "below override the file.")
    parser.add_argument("--datasets_root", type=str, default=None,
                        metavar="PATH",
                        help="Root holding the built TFRecord datasets "
                             "(default: <data_root>/datasets, or "
                             "$COALITION4_DATASETS_ROOT).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override [defaults].epochs / [sepconv].epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override [defaults].batch_size / "
                             "[sepconv].batch_size.")
    parser.add_argument("--lead", type=int, default=None,
                        choices=list(SEPCONV_BASE_LEADS),
                        help="Train one base model instead of all three. "
                             f"{SEPCONV_BASE_LEADS} = "
                             f"{[LEAD_MINUTES[k] for k in SEPCONV_BASE_LEADS]} "
                             f"minutes ahead.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override [sepconv].learning_rate (which "
                             "defaults to [lr_schedule].initial_lr). The "
                             "published value is 1e-3.")
    parser.add_argument("--lr_patience", type=int, default=None,
                        help="Override [sepconv].lr_patience - epochs on a "
                             "val_loss plateau before the rate is halved.")
    parser.add_argument("--es_patience", type=int, default=None,
                        help="Override [sepconv].es_patience (which defaults "
                             "to [early_stopping].patience).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any per-epoch checkpoint and start "
                             "from scratch, regardless of "
                             "[checkpointing].resume.")
    args = parser.parse_args()

    full_cfg = load_training_config(Path(args.config))
    cfg = full_cfg["sepconv"]
    ckpt_cfg = full_cfg["checkpointing"]
    # Flag beats file, file beats nothing - the same precedence the roots
    # use, so `--epochs 80` still works for a one-off sweep.
    def pick(flag, key):
        return cfg[key] if flag is None else flag

    epochs = pick(args.epochs, "epochs")
    batch_size = pick(args.batch_size, "batch_size")
    learning_rate = pick(args.learning_rate, "learning_rate")
    lr_patience = pick(args.lr_patience, "lr_patience")
    es_patience = pick(args.es_patience, "es_patience")

    print(f"Config        : {args.config}")
    print(f"  epochs={epochs}  batch_size={batch_size}  "
          f"lr={learning_rate}  lr_patience={lr_patience}  "
          f"es_patience={es_patience}")

    train(data_root=args.data_root, model_dir=args.model_dir,
          epochs=epochs, batch_size=batch_size, lead=args.lead,
          learning_rate=learning_rate, lr_patience=lr_patience,
          es_patience=es_patience, period=args.period,
          datasets_root=args.datasets_root,
          checkpoint_cfg=ckpt_cfg,
          resume=ckpt_cfg.get("resume", True) and not args.fresh)


if __name__ == "__main__":
    main()