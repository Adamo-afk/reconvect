"""
train_sepconv_ensemble.py — SepConv Ensemble (Regression) for COALITION-4
==========================================================================
Faithful ensemble: three base models (Bm1, Bm2, Bm3), each predicting a
single lead time as continuous rain rate via regression.

Output: sigmoid [0,1] (continuous), NOT softmax classification.
Loss: weighted MSE from Czibula et al. 2024 (upweights high precipitation).
Labels: one-hot from create_datasets.py auto-converted to continuous midpoints.
Classification: recovered at evaluation time via post-processing thresholds.

Usage:
    python train_sepconv_ensemble.py --mode msg_radar --epochs 50 --batch_size 8
    python train_sepconv_ensemble.py --mode msg_radar --lead 1 --epochs 50
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


# ============================================================================
# Mode configurations
# ============================================================================

MODE_CONFIGS = {
    "msg_radar_continuous": {
        "branches": {
            "past_hr": {"shape": (3, 256, 256, 9), "resolution": 256},
            "past_lr": {"shape": (3, 64, 64, 13), "resolution": 64},
        },
    },
    "mtg_radar_continuous": {
        "branches": {
            "past_hr": {"shape": (3, 256, 256, 10), "resolution": 256},
            "past_mr": {"shape": (3, 128, 128, 4), "resolution": 128},
            "past_lr": {"shape": (3, 64, 64, 8), "resolution": 64},
        },
    },
}

# Architecture constants (from paper)
HR_SIZE = 256
KERNEL = (5, 5)
ACTIVATION = 'selu'
DEPTH_MULT = 1
TARGET_CHANNELS = 200

LEAD_NAMES = {1: "t+15", 2: "t+30", 3: "t+45"}
LEAD_MINUTES = {1: 15, 2: 30, 3: 45}

# Post-processing thresholds to recover 5 classes (used in evaluation)
# Raw RZC normalized by /70: thresholds at 10, 20, 30, 40 mm/h
THRESHOLDS_NORM = [10.0 / 70.0, 20.0 / 70.0, 30.0 / 70.0, 40.0 / 70.0]


# ============================================================================
# Weighted MSE loss (from paper)
# ============================================================================

def weighted_loss_multiple_thresholds(weights, max_value=1.0):
    """Weighted MSE that upweights high precipitation values.

    Splits [0, max_value] into len(weights) bins and applies different
    weights per bin. Paper used weights=[15, 1, 2, 7, 15, 30, 1000].
    """
    num_steps = len(weights)
    thresholds = [max_value * i / num_steps for i in range(1, num_steps)]

    def inner_weighted_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        diff = K.pow(y_true - y_pred, 2)

        masks_less = [K.cast(K.less(y_true, t), K.floatx())
                      for t in thresholds]
        masks_greater = [K.cast(K.greater_equal(y_true, t), K.floatx())
                         for t in [0.0] + thresholds]

        result = tf.constant(0.0, dtype=tf.float32)
        for i in range(len(masks_less)):
            result += weights[i] * K.mean(
                masks_less[i] * masks_greater[i] * diff)
        result += weights[-1] * K.mean(masks_greater[-1] * diff)
        return result

    return inner_weighted_loss


# ============================================================================
# Base model (regression: sigmoid output)
# ============================================================================

def build_sepconv_base_model(mode, lead_idx):
    """Build one SepConv base model.

    Output: (batch, 256, 256, 1) — continuous rain rate in [0, 1] via sigmoid.
    """
    config = MODE_CONFIGS[mode]
    branches = config["branches"]
    n_timesteps = 3
    n_total_inputs = len(branches) * n_timesteps
    expand_ch = TARGET_CHANNELS // n_total_inputs + 1

    inputs = {}
    for key, cfg in branches.items():
        inputs[key] = Input(shape=cfg["shape"], name=key)

    branch_features = []
    for key, cfg in branches.items():
        branch_input = inputs[key]
        resolution = cfg["resolution"]
        timestep_features = []
        for t in range(n_timesteps):
            xt = Lambda(lambda x, _t=t: x[:, _t],
                        name=f"{key}_t{t}")(branch_input)
            xt = SeparableConv2D(expand_ch, KERNEL, padding='same',
                                 depth_multiplier=DEPTH_MULT,
                                 name=f"{key}_expand_t{t}")(xt)
            xt = Activation(ACTIVATION)(xt)
            timestep_features.append(xt)

        branch_feat = Concatenate(axis=-1,
                                   name=f"{key}_time_cat")(timestep_features)
        if resolution < HR_SIZE:
            scale = HR_SIZE // resolution
            branch_feat = UpSampling2D(
                size=(scale, scale), interpolation='bilinear',
                name=f"{key}_upsample")(branch_feat)
        branch_features.append(branch_feat)

    if len(branch_features) > 1:
        x = Concatenate(axis=-1, name="branch_merge")(branch_features)
    else:
        x = branch_features[0]

    for i in range(5):
        x = SeparableConv2D(100, KERNEL, padding='same',
                            name=f"trunk_100_{i}")(x)
        x = Activation(ACTIVATION)(x)
    for i in range(5):
        x = SeparableConv2D(50, KERNEL, padding='same',
                            name=f"trunk_50_{i}")(x)
        x = Activation(ACTIVATION)(x)

    # Regression output: 1 channel, sigmoid bounds to [0, 1]
    x = SeparableConv2D(1, KERNEL, padding='same', name="output_conv")(x)
    outputs = Activation('sigmoid', name="output_sigmoid")(x)

    return Model(inputs=inputs, outputs=outputs,
                 name=f"sepconv_ensemble_{mode}_bm{lead_idx}")


# ============================================================================
# Dataset: extract single lead time (labels already continuous)
# ============================================================================

def extract_lead_time(inputs, labels, lead_idx):
    """(3, 256, 256, 1) → single lead → (256, 256, 1)"""
    return inputs, labels[lead_idx - 1]


def prepare_dataset(ds_path, lead_idx, batch_size, shuffle=False):
    ds = tf.data.Dataset.load(str(ds_path))
    ds = ds.map(lambda x, y: extract_lead_time(x, y, lead_idx),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
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

def train_base_model(mode, lead_idx, data_root, model_dir, epochs, batch_size):
    data_root = Path(data_root)
    lead_name = LEAD_NAMES[lead_idx]

    print(f"\n{'─' * 60}")
    print(f"  Training Bm{lead_idx} ({lead_name} = {LEAD_MINUTES[lead_idx]} min)")
    print(f"{'─' * 60}")

    train_ds = prepare_dataset(
        data_root / "datasets" / mode / "train", lead_idx, batch_size, True)
    val_ds = prepare_dataset(
        data_root / "datasets" / mode / "validation", lead_idx, batch_size, False)

    model = build_sepconv_base_model(mode, lead_idx)
    print(f"  Parameters: {model.count_params():,}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    loss = weighted_loss_multiple_thresholds([15, 1, 2, 7, 15, 30, 1000])
    model.compile(optimizer=optimizer, loss=loss)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5,
        min_lr=1e-15, min_delta=0.0001, verbose=1)
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1)
    wall_timer = WallTimeCallback()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=[reduce_lr, early_stop, wall_timer], verbose=1)

    model_path = Path(model_dir) / f"sepconv_ensemble_{mode}_bm{lead_idx}.keras"
    model.save(str(model_path))
    print(f"  Saved: {model_path}")

    return {
        "lead_idx": lead_idx, "lead_name": lead_name,
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

def train(mode, data_root, model_dir, epochs=50, batch_size=8, lead=None):
    data_root = Path(data_root)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation"]:
        if not (data_root / "datasets" / mode / split).exists():
            raise FileNotFoundError(f"Dataset not found: {data_root}/datasets/{mode}/{split}")

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    n_train = sum(1 for _ in tf.data.Dataset.load(str(data_root / "datasets" / mode / "train")))
    n_val = sum(1 for _ in tf.data.Dataset.load(str(data_root / "datasets" / mode / "validation")))

    print("=" * 70)
    print(f"SepConv ENSEMBLE Training (Regression) — Mode: {mode}")
    print("=" * 70)
    print(f"  Output: sigmoid [0,1] (continuous rain rate)")
    print(f"  Loss: weighted MSE [15, 1, 2, 7, 15, 30, 1000]")
    print(f"  Optimizer: Adam (AMSGrad, lr=0.001, halve on plateau)")
    print(f"  Train: {n_train}, Val: {n_val}")

    leads_to_train = [lead] if lead is not None else [1, 2, 3]
    total_start = time.time()
    all_results = {}

    for lead_idx in leads_to_train:
        all_results[f"bm{lead_idx}"] = train_base_model(
            mode, lead_idx, data_root, model_dir, epochs, batch_size)

    total_time = time.time() - total_start

    history_data = {
        "mode": mode, "architecture": "sepconv_ensemble",
        "output_type": "regression",
        "base_models": all_results,
        "total_wall_time": total_time,
        "batch_size": batch_size, "n_train": n_train, "n_val": n_val,
        "total_params": sum(r["model_params"] for r in all_results.values()),
        "label_info": {
            "type": "continuous",
            "normalization": "RZC / 70 → [0, 1]",
            "thresholds_for_classification": THRESHOLDS_NORM,
        },
        "config": {
            "kernel": list(KERNEL), "activation": ACTIVATION,
            "loss": "weighted_MSE [15,1,2,7,15,30,1000]",
            "optimizer": "Adam(AMSGrad)",
        },
    }

    history_path = model_dir / f"history_sepconv_ensemble_{mode}.json"
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
    parser = argparse.ArgumentParser(description="Train SepConv ensemble (regression).")
    parser.add_argument("--mode", type=str, required=True, choices=["msg_radar_continuous", "mtg_radar_continuous"])
    parser.add_argument("--data_root", type=str, default="./our_data")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lead", type=int, default=None, choices=[1, 2, 3])
    args = parser.parse_args()
    train(mode=args.mode, data_root=args.data_root, model_dir=args.model_dir,
          epochs=args.epochs, batch_size=args.batch_size, lead=args.lead)


if __name__ == "__main__":
    main()