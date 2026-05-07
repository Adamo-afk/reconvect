"""
train_coalition.py — COALITION-4 Romanian Adaptation: Training Script
=====================================================================
Builds the recurrent-convolutional architecture from scratch, loads the
pre-built TF datasets, trains, and saves the model + history.

Usage:
    python train_coalition.py --mode msg_lightning --epochs 10
    python train_coalition.py --mode mtg_radar --epochs 1 --batch_size 8

Requires:
    - TensorFlow 2.x with GPU support
    - Pre-built TF datasets from create_datasets.py
    - our_data/lightning_fraction.json (for lightning modes)
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

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

def setup_mixed_precision():
    """Enable mixed precision training for faster GPU execution."""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy: {policy.name}")
    print(f"  Compute dtype: {policy.compute_dtype}")
    print(f"  Variable dtype: {policy.variable_dtype}")


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
# Dataset loading
# ============================================================================

def load_dataset(dataset_dir, batch_size, shuffle=False, shuffle_buffer=2048):
    """Load a saved tf.data.Dataset and prepare it for training."""
    ds = tf.data.Dataset.load(str(dataset_dir))

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer,
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
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


# ============================================================================
# Training
# ============================================================================

def train(mode, data_root, epochs, batch_size, output_dir,
          dropout=0.1, norm=None, dataset_dir=None):
    """Main training function.

    Args:
        mode: msg_lightning, msg_radar, mtg_lightning, mtg_radar (used for
              naming only when dataset_dir is provided explicitly)
        data_root: path to our_data/ containing datasets/{mode}/ and
                   lightning_fraction.json
        epochs: number of training epochs
        batch_size: training batch size
        output_dir: where to save model + history
        dropout: dropout rate
        norm: normalization type
        dataset_dir: explicit path to the dataset directory. When provided,
                     overrides the default data_root/datasets/{mode} path.
                     This allows training on custom dataset variants (e.g.
                     datasets built without NWCSAF).
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)
    else:
        dataset_dir = data_root / "datasets" / mode

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
    print(f"COALITION-4 Training — Mode: {mode}")
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
    print(f"  Mixed prec:    float16")
    print()

    # Setup mixed precision
    setup_mixed_precision()

    # Load ones_fraction for lightning modes from pre-computed JSON
    if label_type == "lightning":
        ones_fraction = load_ones_fraction(data_root)
    else:
        ones_fraction = 0.0106  # unused for radar

    # Load datasets
    print("\nLoading datasets...")
    train_ds = load_dataset(train_dir, batch_size, shuffle=True)
    val_ds = load_dataset(val_dir, batch_size, shuffle=False)
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

    # Callbacks
    wall_time = WallTimeCallback()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        patience=3, mode="min", factor=0.2, monitor="val_loss",
        verbose=1, min_delta=0.0
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=6, mode="min", restore_best_weights=True,
        monitor="val_loss"
    )
    callbacks = [wall_time, reduce_lr, early_stop]

    # Train
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save model
    model_path = output_dir / f"coalition_{mode}.keras"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save history
    history_data = {
        "mode": mode,
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
    history_path = output_dir / f"history_{mode}.json"
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"History saved to: {history_path}")

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)

    return model, history


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train COALITION-4 model from pre-built TF datasets."
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        help="Model variant name (e.g. mtg_lightning, mtg_lightning_no_nwcsaf). "
             "Used for naming the saved model and history files."
    )
    parser.add_argument(
        "--data_root", type=str, default="./our_data",
        help="Root directory containing datasets/ and lightning_fraction.json"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Explicit path to the dataset directory (overrides data_root/datasets/{mode}). "
             "Use this to train on custom dataset variants, e.g. datasets without NWCSAF."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./models",
        help="Directory to save trained model and history"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--norm", type=str, default=None, choices=[None, "batch", "layer"],
        help="Normalization type (default: None)"
    )
    args = parser.parse_args()

    train(
        mode=args.mode,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        dropout=args.dropout,
        norm=args.norm,
        dataset_dir=args.dataset_dir,
    )


if __name__ == "__main__":
    main()
