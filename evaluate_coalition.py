"""
evaluate_coalition.py — COALITION-4 Romanian Adaptation: Evaluation Script
==========================================================================
Loads training history and trained model, runs evaluation on the test set,
computes all metrics from the original COALITION paper, and saves plots.

Usage:
    # Base model, OPERA-driven sample selection
    python evaluate_coalition.py --mode mtg_lightning_opera_occurrence \
        --source dbscan --data_root ./our_data

    # Swin-head fine-tuned model
    python evaluate_coalition.py --mode mtg_lightning_opera_occurrence \
        --source dbscan --finetuned

    # Lightning-driven sample selection, base model
    python evaluate_coalition.py --mode mtg_lightning \
        --source lightning

Outputs (saved to output_dir/eval_{mode}_{source}[_finetuned]/):
    - training_curves_{mode}.png       — loss and metrics vs epoch
    - metrics_per_leadtime_{mode}.png  — CSI, POD, FAR etc. vs lead time
    - calibration_{mode}.png           — reliability diagram
    - pr_curve_{mode}.png              — precision-recall curve
    - roc_curve_{mode}.png             — ROC curve
    - confusion_matrix_{mode}.png      — for radar multi-class
    - evaluation_results_{mode}.json   — all numerical results
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Add, Conv2D, Dropout, Activation, ELU, LeakyReLU, ReLU,
    AveragePooling2D, BatchNormalization, TimeDistributed,
    LayerNormalization
)


# ============================================================================
# Custom Layers — embedded for standalone model loading
# ============================================================================

def _load_split(split_dir: Path) -> tf.data.Dataset:
    """Auto-detect the on-disk format of a saved dataset split and load it.

    Supports two layouts, distinguished by `metadata.json["format"]`:
      - "tfrecord" (current): `shard_*.tfrecord` files, parsed using
        `input_shapes` + `label_shape` from metadata.
      - "tf_dataset_save" (legacy): monolithic `tf.data.Dataset.save`
        snapshot. Kept so older datasets keep working.
    """
    metadata_path = split_dir / "metadata.json"
    if metadata_path.is_file():
        with open(metadata_path) as f:
            meta = json.load(f)
        fmt = meta.get("format", "tf_dataset_save")
    else:
        meta = None
        fmt = "tf_dataset_save"

    if fmt != "tfrecord":
        return tf.data.Dataset.load(str(split_dir))

    shard_paths = sorted(str(p) for p in split_dir.glob("shard_*.tfrecord"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No TFRecord shards in {split_dir} (expected "
            f"`shard_*.tfrecord`). Re-run create_datasets.py."
        )
    input_shapes: dict = meta["input_shapes"]
    label_shape: list = meta["label_shape"]

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
    n_samples = int(meta.get("n_samples", 0))
    if n_samples > 0:
        ds = ds.apply(tf.data.experimental.assert_cardinality(n_samples))
    return ds


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
# Swin transformer head — embedded for standalone loading of fine-tuned models
# ============================================================================
#
# Mirror of the Swin layer defined in train_models.py so saved
# `*_finetuned.keras` files (Swin head grafted onto a frozen backbone)
# can be reloaded here without importing the training module. Same
# class name + same get_config so the serialised graph round-trips.


def _window_partition(x, window_size):
    """(B, H, W, C) -> (B*nW, ws*ws, C) where nW = (H/ws)*(W/ws)."""
    shape = tf.shape(x)
    B, H, W = shape[0], shape[1], shape[2]
    C = x.shape[-1]
    ws = window_size
    x = tf.reshape(x, [B, H // ws, ws, W // ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, ws * ws, C])
    return x


def _window_reverse(x_windows, H, W, window_size, B):
    """(B*nW, ws*ws, C) -> (B, H, W, C)."""
    ws = window_size
    C = x_windows.shape[-1]
    x = tf.reshape(x_windows, [B, H // ws, W // ws, ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x


class SwinBlock(tf.keras.layers.Layer):
    """Single Swin transformer block (W-MSA or SW-MSA + MLP).

    Re-implemented here for `tf.keras.models.load_model()` to find by
    name when deserialising `*_finetuned.keras` files written by
    train_models.train_finetune. Behaviour must stay in lockstep with
    the definition in train_models.py.
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


# ============================================================================
# Loss and metrics — embedded for standalone model loading
# ============================================================================

class WeightedFocalLoss(tf.keras.losses.Loss):
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
# Lead time labels
# ============================================================================

LEAD_TIMES = [15, 30, 45]  # minutes
LEAD_LABELS = ["t+15", "t+30", "t+45"]
N_LEAD = len(LEAD_TIMES)


# ============================================================================
# GPU-accelerated binary metrics (TensorFlow)
# ============================================================================

@tf.function
def tf_confusion_components(y_true, y_pred_binary):
    """Compute TP, FP, FN, TN fractions on GPU."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred_binary, tf.float64)
    N = tf.cast(tf.size(y_true), tf.float64)
    TP = tf.reduce_sum(y_pred * y_true) / N
    FP = tf.reduce_sum(y_pred * (1.0 - y_true)) / N
    FN = tf.reduce_sum((1.0 - y_pred) * y_true) / N
    TN = tf.reduce_sum((1.0 - y_pred) * (1.0 - y_true)) / N
    return TP, FP, FN, TN


@tf.function
def tf_metrics_at_threshold(y_true, y_pred, threshold):
    """Compute all binary metrics at a given threshold on GPU.
    Returns: (CSI, POD, FAR, FPR, ETS, HSS, PSS, TP, FP, FN, TN)
    """
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64)
    threshold = tf.cast(threshold, tf.float64)
    y_bin = tf.cast(y_pred >= threshold, tf.float64)
    N = tf.cast(tf.size(y_true), tf.float64)

    TP = tf.reduce_sum(y_bin * y_true) / N
    FP = tf.reduce_sum(y_bin * (1.0 - y_true)) / N
    FN = tf.reduce_sum((1.0 - y_bin) * y_true) / N
    TN = tf.reduce_sum((1.0 - y_bin) * (1.0 - y_true)) / N

    eps = 1e-10
    _pod = TP / (TP + FN + eps)
    _far = FP / (TP + FP + eps)
    _fpr = FP / (FP + TN + eps)
    _csi = TP / (TP + FP + FN + eps)

    R = (TP + FN) * (TP + FP) / (TP + FP + FN + TN + eps)
    _ets = (TP - R) / (TP + FP + FN - R + eps)

    hss_num = 2.0 * (TP * TN - FN * FP)
    hss_den = (TP + FN) * (FN + TN) + (TP + FP) * (FP + FN)
    _hss = hss_num / (hss_den + eps)

    _pss = _pod - _fpr

    return _csi, _pod, _far, _fpr, _ets, _hss, _pss, TP, FP, FN, TN


@tf.function
def tf_find_optimal_threshold(y_true, y_pred, thresholds):
    """Find threshold maximizing CSI on GPU (vectorized over thresholds)."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    thresholds = tf.cast(thresholds, tf.float64)
    N = tf.cast(tf.size(y_true), tf.float64)

    best_csi = tf.constant(-1.0, dtype=tf.float64)
    best_t = tf.constant(0.5, dtype=tf.float64)

    for i in tf.range(tf.shape(thresholds)[0]):
        t = tf.cast(thresholds[i], tf.float64)
        y_bin = tf.cast(y_pred >= t, tf.float64)
        TP = tf.reduce_sum(y_bin * y_true) / N
        FP = tf.reduce_sum(y_bin * (1.0 - y_true)) / N
        FN = tf.reduce_sum((1.0 - y_bin) * y_true) / N
        current_csi = TP / (TP + FP + FN + 1e-10)
        if current_csi > best_csi:
            best_csi = current_csi
            best_t = t

    return best_t, best_csi


@tf.function
def tf_pr_roc_curves(y_true, y_pred, thresholds):
    """Compute PR and ROC curves on GPU in a single pass over thresholds."""
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    thresholds = tf.cast(thresholds, tf.float64)
    N = tf.cast(tf.size(y_true), tf.float64)
    n_t = tf.shape(thresholds)[0]

    precisions = tf.TensorArray(dtype=tf.float64, size=n_t)
    recalls = tf.TensorArray(dtype=tf.float64, size=n_t)
    fprs_arr = tf.TensorArray(dtype=tf.float64, size=n_t)
    tprs_arr = tf.TensorArray(dtype=tf.float64, size=n_t)

    for i in tf.range(n_t):
        t = tf.cast(thresholds[i], tf.float64)
        y_bin = tf.cast(y_pred >= t, tf.float64)
        TP = tf.reduce_sum(y_bin * y_true) / N
        FP = tf.reduce_sum(y_bin * (1.0 - y_true)) / N
        FN = tf.reduce_sum((1.0 - y_bin) * y_true) / N
        TN = tf.reduce_sum((1.0 - y_bin) * (1.0 - y_true)) / N

        eps = 1e-10
        recall = TP / (TP + FN + eps)
        precision = 1.0 - FP / (TP + FP + eps)
        fpr_val = FP / (FP + TN + eps)

        recalls = recalls.write(i, recall)
        precisions = precisions.write(i, precision)
        fprs_arr = fprs_arr.write(i, fpr_val)
        tprs_arr = tprs_arr.write(i, recall)

    return (recalls.stack(), precisions.stack(),
            fprs_arr.stack(), tprs_arr.stack())


def compute_auc(x, y):
    """Trapezoidal AUC from sorted arrays."""
    sorted_idx = np.argsort(x)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return trapz_fn(y[sorted_idx], x[sorted_idx])


def compute_calibration_gpu(y_true_flat, y_pred_flat, n_bins=100):
    """Calibration on GPU."""
    y_true = tf.constant(y_true_flat, dtype=tf.float64)
    y_pred = tf.constant(y_pred_flat, dtype=tf.float64)
    bin_edges = tf.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = tf.cast(bin_edges, tf.float64)

    observed = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        c = tf.reduce_sum(tf.cast(mask, tf.float64))
        counts[i] = c.numpy()
        if counts[i] > 0:
            observed[i] = tf.reduce_mean(
                tf.boolean_mask(y_true, mask)).numpy()

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers.numpy(), observed, counts


# ============================================================================
# Evaluation runners
# ============================================================================

def collect_predictions_gpu(model, dataset):
    """Collect all predictions and labels per lead time using GPU.
    Returns dict of tf.Tensors per lead time + aggregated.
    """
    all_true = {t: [] for t in range(N_LEAD)}
    all_pred = {t: [] for t in range(N_LEAD)}

    for inputs, labels in dataset:
        preds = model(inputs, training=False)
        for t in range(N_LEAD):
            all_true[t].append(tf.reshape(labels[:, t, :, :, 0], [-1]))
            all_pred[t].append(tf.reshape(preds[:, t, :, :, 0], [-1]))

    # Concatenate per lead time
    for t in range(N_LEAD):
        all_true[t] = tf.concat(all_true[t], axis=0)
        all_pred[t] = tf.concat(all_pred[t], axis=0)

    # Aggregate across all lead times
    all_true_agg = tf.concat([all_true[t] for t in range(N_LEAD)], axis=0)
    all_pred_agg = tf.concat([all_pred[t] for t in range(N_LEAD)], axis=0)

    return all_true, all_pred, all_true_agg, all_pred_agg


def evaluate_lightning(model, test_ds, output_dir, threshold=None, val_ds=None):
    """Run full evaluation for lightning (binary) mode.

    Args:
        model: trained Keras model
        test_ds: batched test tf.data.Dataset
        output_dir: Path for saving plots
        threshold: float or None.
            - If float: use this fixed threshold (e.g. 0.5)
            - If None: optimize on val_ds (proper approach from the paper)
        val_ds: batched validation tf.data.Dataset (required if threshold is None)

    Returns dict with all metrics and curves.
    """
    thresholds_array = tf.constant(np.linspace(0.01, 0.99, 99), dtype=tf.float64)

    # --- Determine threshold ---
    if threshold is not None:
        opt_threshold = float(threshold)
        print(f"  Using fixed threshold: {opt_threshold:.3f}")
    else:
        if val_ds is None:
            raise ValueError("val_ds is required when threshold is None. "
                             "Either provide --threshold or ensure validation "
                             "dataset exists.")
        print("  Optimizing threshold on validation set (GPU)...")
        val_true, val_pred, val_true_agg, val_pred_agg = \
            collect_predictions_gpu(model, val_ds)
        opt_threshold, opt_csi = tf_find_optimal_threshold(
            val_true_agg, val_pred_agg, thresholds_array)
        opt_threshold = float(opt_threshold.numpy())
        opt_csi = float(opt_csi.numpy())
        print(f"    Optimal threshold (from val): {opt_threshold:.3f} "
              f"(val CSI={opt_csi:.4f})")
        # Free validation tensors
        del val_true, val_pred, val_true_agg, val_pred_agg

    # --- Collect test predictions on GPU ---
    print("  Collecting test predictions (GPU)...")
    all_true, all_pred, all_true_agg, all_pred_agg = \
        collect_predictions_gpu(model, test_ds)

    n_pixels = int(all_true_agg.shape[0])
    print(f"    Total test pixels: {n_pixels:,}")

    # --- Metrics per lead time (GPU) ---
    print("  Computing metrics per lead time (GPU)...")
    threshold_tf = tf.constant(opt_threshold, dtype=tf.float64)
    results = {
        "optimal_threshold": opt_threshold,
        "threshold_source": "fixed" if threshold is not None else "validation",
        "per_leadtime": {},
        "aggregate": {},
    }

    METRIC_NAMES = ["CSI", "POD", "FAR", "FPR", "ETS", "HSS", "PSS",
                    "TP", "FP", "FN", "TN"]

    for t in range(N_LEAD):
        vals = tf_metrics_at_threshold(all_true[t], all_pred[t], threshold_tf)
        lt_metrics = {name: float(v.numpy()) for name, v in
                      zip(METRIC_NAMES, vals)}
        results["per_leadtime"][LEAD_LABELS[t]] = lt_metrics
        print(f"    {LEAD_LABELS[t]}: CSI={lt_metrics['CSI']:.4f}, "
              f"POD={lt_metrics['POD']:.4f}, FAR={lt_metrics['FAR']:.4f}")

    # Aggregate metrics
    agg_vals = tf_metrics_at_threshold(all_true_agg, all_pred_agg, threshold_tf)
    for name, v in zip(METRIC_NAMES, agg_vals):
        results["aggregate"][name] = float(v.numpy())

    # --- PR and ROC curves (GPU) ---
    print("  Computing PR and ROC curves (GPU)...")
    curve_thresholds = tf.constant(np.linspace(0.01, 0.99, 200), dtype=tf.float64)
    recalls, precisions, fprs_arr, tprs_arr = tf_pr_roc_curves(
        all_true_agg, all_pred_agg, curve_thresholds)

    recalls_np = recalls.numpy()
    precisions_np = precisions.numpy()
    fprs_np = fprs_arr.numpy()
    tprs_np = tprs_arr.numpy()

    pr_auc = compute_auc(recalls_np, precisions_np)
    roc_auc = compute_auc(fprs_np, tprs_np)
    results["aggregate"]["PR_AUC"] = float(pr_auc)
    results["aggregate"]["ROC_AUC"] = float(roc_auc)

    # Per-leadtime PR AUC
    for t in range(N_LEAD):
        r, p, _, _ = tf_pr_roc_curves(all_true[t], all_pred[t], curve_thresholds)
        results["per_leadtime"][LEAD_LABELS[t]]["PR_AUC"] = float(
            compute_auc(r.numpy(), p.numpy()))

    # --- Calibration (GPU) ---
    print("  Computing calibration (GPU)...")
    cal_centers, cal_observed, cal_counts = compute_calibration_gpu(
        all_true_agg.numpy(), all_pred_agg.numpy())

    print(f"    Aggregate: CSI={results['aggregate']['CSI']:.4f}, "
          f"PR_AUC={pr_auc:.4f}, ROC_AUC={roc_auc:.4f}")

    # ==================== PLOTS ====================

    # 1. Metrics per lead time
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plot_metrics = ["CSI", "POD", "FAR", "ETS", "HSS", "PSS"]
    for idx, metric_name in enumerate(plot_metrics):
        ax = axes[idx // 3, idx % 3]
        vals = [results["per_leadtime"][lt][metric_name] for lt in LEAD_LABELS]
        ax.plot(LEAD_TIMES, vals, 'o-', linewidth=2, markersize=8,
                color='#1f77b4')
        ax.set_xlabel("Lead time (min)")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_xticks(LEAD_TIMES)
        ax.grid(True, alpha=0.3)
        # Add aggregate as horizontal dashed line
        agg_val = results["aggregate"][metric_name]
        ax.axhline(y=agg_val, color='gray', linestyle='--', alpha=0.5,
                   label=f"agg={agg_val:.3f}")
        ax.legend(fontsize=8)
    plt.suptitle("Lightning metrics per lead time", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_per_leadtime.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. PR curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls_np, precisions_np, linewidth=2, color='#1f77b4',
            label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall (POD)")
    ax.set_ylabel("Precision (1 - FAR)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs_np, tprs_np, linewidth=2, color='#ff7f0e',
            label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (POD)")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Calibration
    fig, ax = plt.subplots(figsize=(7, 6))
    valid_mask = cal_counts > 10  # only plot bins with enough samples
    ax.plot(cal_centers[valid_mask], cal_observed[valid_mask], 'o-',
            linewidth=2, markersize=4, color='#2ca02c', label="Model")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed occurrence rate")
    ax.set_title("Calibration (Reliability Diagram)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "calibration.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Store curves for JSON
    results["curves"] = {
        "pr": {"recalls": recalls_np.tolist(), "precisions": precisions_np.tolist()},
        "roc": {"fprs": fprs_np.tolist(), "tprs": tprs_np.tolist()},
        "calibration": {
            "bin_centers": cal_centers.tolist(),
            "observed": cal_observed.tolist(),
            "counts": cal_counts.tolist(),
        },
    }

    return results


def evaluate_radar(model, test_ds, output_dir):
    """Run evaluation for radar (multi-class) mode.

    Classes: 0=R<10, 1=10≤R<20, 2=20≤R<30, 3=30≤R<40, 4=R≥40 mm/h
    """
    CLASS_NAMES = ["R<10", "10≤R<20", "20≤R<30", "30≤R<40", "R≥40"]
    N_CLASSES = 5

    print("  Collecting predictions...")

    # Per-leadtime confusion matrices
    conf_matrices = {t: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
                     for t in range(N_LEAD)}

    for inputs, labels in test_ds:
        preds = model.predict(inputs, verbose=0)
        labels_np = labels.numpy()

        for t in range(N_LEAD):
            y_true_cls = np.argmax(labels_np[:, t, :, :, :], axis=-1).ravel()
            y_pred_cls = np.argmax(preds[:, t, :, :, :], axis=-1).ravel()
            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    conf_matrices[t][i, j] += np.sum(
                        (y_true_cls == i) & (y_pred_cls == j))

    # Compute per-class and overall metrics
    results = {"per_leadtime": {}, "aggregate": {}}
    conf_agg = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    for t in range(N_LEAD):
        cm = conf_matrices[t]
        conf_agg += cm
        total = cm.sum()
        accuracy = cm.trace() / (total + 1e-10)

        per_class = {}
        for c in range(N_CLASSES):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            per_class[CLASS_NAMES[c]] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(cm[c, :].sum()),
            }

        results["per_leadtime"][LEAD_LABELS[t]] = {
            "accuracy": float(accuracy),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }
        print(f"    {LEAD_LABELS[t]}: accuracy={accuracy:.4f}")

    # Aggregate
    total = conf_agg.sum()
    results["aggregate"]["accuracy"] = float(conf_agg.trace() / (total + 1e-10))
    results["aggregate"]["confusion_matrix"] = conf_agg.tolist()

    # ==================== PLOTS ====================

    # 1. Accuracy per lead time
    fig, ax = plt.subplots(figsize=(7, 5))
    accs = [results["per_leadtime"][lt]["accuracy"] for lt in LEAD_LABELS]
    ax.plot(LEAD_TIMES, accs, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax.axhline(y=results["aggregate"]["accuracy"], color='gray',
               linestyle='--', alpha=0.5,
               label=f"agg={results['aggregate']['accuracy']:.3f}")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Radar Classification Accuracy per Lead Time")
    ax.set_xticks(LEAD_TIMES)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_per_leadtime.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    # 2. Confusion matrices
    fig, axes = plt.subplots(1, N_LEAD + 1, figsize=(5 * (N_LEAD + 1), 5))
    all_cms = [conf_matrices[t] for t in range(N_LEAD)] + [conf_agg]
    all_titles = LEAD_LABELS + ["Aggregate"]

    for ax, cm, title in zip(axes, all_cms, all_titles):
        # Normalize per row
        cm_norm = cm.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        # Add text annotations
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                        fontsize=7,
                        color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.suptitle("Confusion matrices (row-normalized)", fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    # 3. Per-class F1 per lead time
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, name in enumerate(CLASS_NAMES):
        f1s = [results["per_leadtime"][lt]["per_class"][name]["f1"]
               for lt in LEAD_LABELS]
        ax.plot(LEAD_TIMES, f1s, 'o-', linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 Score vs Lead Time")
    ax.set_xticks(LEAD_TIMES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_per_class.png", dpi=150, bbox_inches='tight')
    plt.close()

    return results


def plot_predictions_for_date_hour(model, mode, data_root, output_dir,
                                    plot_date, plot_hour, plot_threshold=0.5,
                                    csv_name="test_data_dbscan.csv",
                                    label_type=None, source="dbscan"):
    """Plot all patches for all timesteps matching a given date and hour.

    Uses the already-loaded model for the current mode only.
    Lightning label_type → lightning plots. Radar label_type → radar plots.

    Args:
        model: trained Keras model (already loaded)
        mode: full mode string e.g. "mtg_lightning_opera_occurrence"
        data_root: path to our_data/
        output_dir: path for saving plots
        plot_date: date string e.g. "2025-05-26"
        plot_hour: integer hour 0-23
        plot_threshold: probability threshold for lightning display
        csv_name: per-source CSV filename (default: "test_data_dbscan.csv")
        label_type: "lightning" or "radar"; if None, inferred from mode
            (best-effort - the dataset's metadata.json is authoritative
            and used by the caller).
    """
    import ast
    import pandas as pd
    from datetime import datetime, timedelta

    try:
        from create_datasets import (
            get_mode_config, load_and_transform_group, load_label,
            set_normalization_stats_path, LABEL_CHANNELS,
        )
    except ImportError:
        print("  WARNING: Could not import from create_datasets.py. Skipping.")
        return

    data_root = Path(data_root)
    # The lazy normalization-stats loader in create_datasets defaults to
    # `normalization_stats.json` at the project root; the per-source
    # pipeline writes `normalization_stats_<source>.json`. Point the
    # loader at the right file before any input transform fires.
    set_normalization_stats_path(
        data_root / f"normalization_stats_{source}.json"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = data_root / "patches"

    if label_type is None:
        # Best-effort fallback. The caller normally passes label_type
        # straight through from the dataset's metadata.json so we don't
        # misclassify modes like `mtg_lightning_opera` (which has
        # 'lightning' in its name but actually uses the 5-class OPERA
        # rainfall target, i.e. label_type='radar').
        mode_cfg = get_mode_config(mode)
        label_type = mode_cfg["label_type"]
    mode_config = get_mode_config(mode)

    label_var = mode_config["label_var"]
    label_transform = mode_config["label_transform"]
    label_suffix = mode_config["label_suffix"]
    n_label_ch = LABEL_CHANNELS[label_type]

    input_groups = {}
    for key in ["past_hr", "past_lr", "past_mr"]:
        cfg = mode_config.get(key)
        if cfg is not None:
            input_groups[key] = cfg

    # Per-source split CSVs use step *indices* (idx_t-2, idx_t-1, idx_t0,
    # idx_t+1, ...) not minute offsets - the schema is stable across the
    # step_minutes choice. Read step_minutes from timestep_config.json
    # to recover the actual minute values used to build HHMM filenames.
    step_minutes_path = data_root / "timestep_config.json"
    if step_minutes_path.is_file():
        with open(step_minutes_path) as _f:
            step_minutes = int(json.load(_f)["step_minutes"])
    else:
        print(f"  WARNING: {step_minutes_path} not found, assuming "
              f"step_minutes=15.")
        step_minutes = 15
    INPUT_STEPS = [-2, -1, 0]
    LABEL_STEPS = [1, 2, 3]
    INPUT_COLS = [
        f"idx_t{s}" if s < 0 else ("idx_t0" if s == 0 else f"idx_t+{s}")
        for s in INPUT_STEPS
    ]
    LABEL_COLS_LOCAL = [f"idx_t+{s}" for s in LABEL_STEPS]
    T_OFFSETS = [s * step_minutes for s in INPUT_STEPS + LABEL_STEPS]
    N_INPUT = 3
    N_LABEL_STEPS = 3
    RADAR_CLASS_NAMES = ["R<10", "10≤R<20", "20≤R<30", "30≤R<40", "R≥40"]

    # --- Filter by date and hour ---
    csv_path = data_root / csv_name
    if not csv_path.is_file():
        print(f"  WARNING: {csv_path} not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    target_refs = [f"{plot_hour}:{m:02d}" for m in [0, 15, 30, 45]]
    mask = (df["date"] == plot_date)
    matching = df[mask].copy()
    matching["ref_clean"] = matching["reference_utc"].str.strip()
    matching = matching[matching["ref_clean"].isin(target_refs)]

    if len(matching) == 0:
        date_mask = df["date"] == plot_date
        if date_mask.sum() == 0:
            available_dates = sorted(df["date"].unique())
            print(f"  ERROR: Date '{plot_date}' not found in {csv_name}.")
            print(f"  Available dates: {available_dates}")
        else:
            available_refs = sorted(
                df[date_mask]["reference_utc"].str.strip().unique())
            available_hours = sorted(set(
                int(r.split(":")[0]) for r in available_refs))
            print(f"  ERROR: No rows for date={plot_date}, hour={plot_hour} "
                  f"in {csv_name}.")
            print(f"  Available hours for {plot_date}: {available_hours}")
            print(f"  Available refs: {available_refs}")
        return

    print(f"  Found {len(matching)} rows for {plot_date} hour {plot_hour}:00 "
          f"({label_type} plots)")
    n_plots = 0

    for _, row in matching.iterrows():
        date_str = row["date"]
        ref_utc = row["reference_utc"].strip()
        patch_numbers = ast.literal_eval(row["patch_numbers"])

        ref_parts = ref_utc.split(":")
        ref_dt = datetime(2000, 1, 1, int(ref_parts[0]), int(ref_parts[1]))
        hhmm_list = [(ref_dt + timedelta(minutes=off)).strftime("%H%M")
                     for off in T_OFFSETS]

        idx_lists = {}
        for col in INPUT_COLS + LABEL_COLS_LOCAL:
            idx_lists[col] = ast.literal_eval(row[col])

        future_hhmms = [hhmm_list[N_INPUT + t] for t in range(N_LABEL_STEPS)]

        for p_pos in range(len(patch_numbers)):
            patch_num = patch_numbers[p_pos]

            # --- Load inputs ---
            input_tensors = {key: [] for key in input_groups}
            ok = True
            for t_idx in range(N_INPUT):
                col = INPUT_COLS[t_idx]
                hhmm = hhmm_list[t_idx]
                npy_idx = idx_lists[col][p_pos]
                for gk, (vc, res, sfx) in input_groups.items():
                    ts = load_and_transform_group(
                        str(patches_dir), date_str, hhmm, sfx, vc, npy_idx, res)
                    if ts is None:
                        ok = False; break
                    input_tensors[gk].append(ts)
                if not ok:
                    break
            if not ok:
                continue

            # --- Load labels ---
            label_frames = []
            for t_idx in range(N_LABEL_STEPS):
                col = LABEL_COLS_LOCAL[t_idx]
                hhmm = hhmm_list[N_INPUT + t_idx]
                npy_idx = idx_lists[col][p_pos]
                label_frames.append(load_label(
                    str(patches_dir), date_str, hhmm,
                    label_var, label_transform, label_suffix,
                    npy_idx, n_label_channels=n_label_ch))

            model_inputs = {}
            for key in input_tensors:
                model_inputs[key] = np.expand_dims(
                    np.stack(input_tensors[key], axis=0), axis=0
                ).astype(np.float32)

            gt = np.stack(label_frames, axis=0)
            pred = model(model_inputs, training=False).numpy()[0]

            title_meta = (f"Date: {date_str}  |  Ref: {ref_utc} UTC  |  "
                          f"Patch #{patch_num}")
            fn_base = f"patch{patch_num}_{date_str}_{ref_utc.replace(':', '')}"

            if label_type == "lightning":
                # 2 rows: GT binary, Pred thresholded
                fig, axes = plt.subplots(2, N_LEAD, figsize=(16, 10))
                fig.subplots_adjust(left=0.03, right=0.88, top=0.88,
                                    bottom=0.03, hspace=0.25, wspace=0.08)

                for t in range(N_LEAD):
                    ax = axes[0, t]
                    gt_frame = gt[t, :, :, 0]
                    ax.imshow(gt_frame, cmap='Reds', vmin=0, vmax=1,
                              interpolation='nearest')
                    ax.set_title(f"GT — {LEAD_LABELS[t]} ({future_hhmms[t]} UTC)",
                                 fontsize=10)
                    ax.axis('off')
                    n_pos = int(np.sum(gt_frame >= 0.5))
                    ax.text(5, 245, f"pixels={n_pos}", fontsize=8, color='white',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

                    ax = axes[1, t]
                    pred_frame = pred[t, :, :, 0].copy()
                    pred_thresholded = np.where(pred_frame >= plot_threshold,
                                                pred_frame, 0.0)
                    im_pred = ax.imshow(pred_thresholded, cmap='RdYlBu_r',
                                        vmin=0, vmax=1, interpolation='nearest')
                    ax.set_title(f"Pred (≥{plot_threshold}) — {LEAD_LABELS[t]} ({future_hhmms[t]} UTC)",
                                 fontsize=10)
                    ax.axis('off')
                    n_pred = int(np.sum(pred_frame >= plot_threshold))
                    p_max = float(np.max(pred_frame))
                    ax.text(5, 245, f"pixels≥{plot_threshold}={n_pred}  max={p_max:.3f}",
                            fontsize=8, color='white',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

                cbar_ax1 = fig.add_axes([0.90, 0.52, 0.015, 0.35])
                fig.colorbar(axes[0, -1].images[0], cax=cbar_ax1, label="Occurrence")
                cbar_ax2 = fig.add_axes([0.90, 0.05, 0.015, 0.35])
                fig.colorbar(im_pred, cax=cbar_ax2, label="Probability")
                fig.suptitle(f"Lightning prediction  —  {title_meta}",
                             fontsize=13, fontweight='bold', y=0.95)

                plt.savefig(output_dir / f"lightning_{fn_base}.png", dpi=150)
                plt.close()

            else:  # radar
                gt_cls = np.argmax(gt, axis=-1)
                pred_cls = np.argmax(pred, axis=-1)
                pred_conf = np.max(pred, axis=-1)
                n_classes = len(RADAR_CLASS_NAMES)
                class_cmap = matplotlib.colormaps.get_cmap('RdYlGn_r').resampled(n_classes)

                fig, axes = plt.subplots(3, N_LEAD, figsize=(16, 14))
                fig.subplots_adjust(left=0.03, right=0.88, top=0.91,
                                    bottom=0.03, hspace=0.22, wspace=0.08)

                for t in range(N_LEAD):
                    ax = axes[0, t]
                    im_gt = ax.imshow(gt_cls[t], cmap=class_cmap,
                                      vmin=-0.5, vmax=n_classes - 0.5,
                                      interpolation='nearest', origin='lower')
                    ax.set_title(f"GT — {LEAD_LABELS[t]} ({future_hhmms[t]} UTC)",
                                 fontsize=10)
                    ax.axis('off')

                    ax = axes[1, t]
                    im_pred_r = ax.imshow(pred_cls[t], cmap=class_cmap,
                                          vmin=-0.5, vmax=n_classes - 0.5,
                                          interpolation='nearest', origin='lower')
                    ax.set_title(f"Pred — {LEAD_LABELS[t]} ({future_hhmms[t]} UTC)",
                                 fontsize=10)
                    ax.axis('off')
                    acc = float(np.mean(gt_cls[t] == pred_cls[t]))
                    ax.text(5, 10, f"acc={acc:.3f}", fontsize=8, color='white',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

                    ax = axes[2, t]
                    im_conf = ax.imshow(pred_conf[t], cmap='viridis',
                                        vmin=0, vmax=1, interpolation='nearest',
                                        origin='lower')
                    ax.set_title(f"Conf — {LEAD_LABELS[t]} ({future_hhmms[t]} UTC)",
                                 fontsize=10)
                    ax.axis('off')

                cbar_ax1 = fig.add_axes([0.90, 0.65, 0.015, 0.24])
                cbar1 = fig.colorbar(im_gt, cax=cbar_ax1)
                cbar1.set_ticks(range(n_classes))
                cbar1.set_ticklabels(RADAR_CLASS_NAMES)

                cbar_ax2 = fig.add_axes([0.90, 0.36, 0.015, 0.24])
                cbar2 = fig.colorbar(im_pred_r, cax=cbar_ax2)
                cbar2.set_ticks(range(n_classes))
                cbar2.set_ticklabels(RADAR_CLASS_NAMES)

                cbar_ax3 = fig.add_axes([0.90, 0.05, 0.015, 0.24])
                fig.colorbar(im_conf, cax=cbar_ax3, label="Max prob.")

                fig.suptitle(f"Radar prediction  —  {title_meta}",
                             fontsize=13, fontweight='bold', y=0.96)

                plt.savefig(output_dir / f"radar_{fn_base}.png", dpi=150)
                plt.close()

            n_plots += 1

    print(f"  Generated {n_plots} {label_type} plots")


# ============================================================================
# Training history plotting
# ============================================================================

def plot_training_history(history_path, output_dir):
    """Load and plot training history from JSON."""
    with open(history_path) as f:
        data = json.load(f)

    history = data["history"]
    mode = data["mode"]
    wall_times = data.get("wall_times", [])
    total_wall = data.get("total_wall_time", 0)

    # Determine which metrics to plot
    loss_keys = [k for k in history if "loss" in k]
    metric_keys = [k for k in history if "loss" not in k and not k.startswith("val_")]
    # Pair train/val metrics
    metric_pairs = []
    for k in metric_keys:
        val_k = f"val_{k}"
        if val_k in history:
            metric_pairs.append((k, val_k))
        else:
            metric_pairs.append((k, None))

    n_plots = 1 + len(metric_pairs) + (1 if wall_times else 0)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    epochs = range(1, len(history.get("loss", [])) + 1)
    plot_idx = 0

    # Plot 1: Loss
    ax = axes[plot_idx]
    if "loss" in history:
        ax.plot(epochs, history["loss"], 'b-', linewidth=2, label="Train loss")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], 'r-', linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Plot metrics
    for train_k, val_k in metric_pairs:
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]
        ax.plot(epochs, history[train_k], 'b-', linewidth=2,
                label=f"Train {train_k}")
        if val_k and val_k in history:
            ax.plot(epochs, history[val_k], 'r-', linewidth=2,
                    label=f"Val {train_k}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(train_k)
        ax.set_title(train_k)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot wall time per epoch
    if wall_times and plot_idx < len(axes):
        ax = axes[plot_idx]
        ax.bar(range(1, len(wall_times) + 1), wall_times, color='#2ca02c',
               alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"Wall Time per Epoch (total: {total_wall:.0f}s)")
        ax.grid(True, alpha=0.3, axis='y')
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Training history", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"training_curves.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves plot")


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate(mode, data_root, model_dir, output_dir, batch_size=32,
             threshold=None, plot_threshold=0.5,
             plot_date=None, plot_hour=None, split="test",
             source="dbscan", finetuned=False):
    """Run full evaluation pipeline.

    Args:
        mode: one of the entries in TRAINING_MODES (e.g.
            `mtg_lightning_opera_occurrence`, `mtg_opera_mtgmr`).
        data_root: path to our_data/ containing datasets/{mode}_{source}/
        model_dir: path to directory containing trained model and history
        output_dir: where to save evaluation results and plots
        threshold: float or None.
            - If float (e.g. 0.5): use this fixed threshold for lightning
            - If None: optimize threshold on validation set (paper's approach)
        source: which sample-selection track the artifacts were built
            from (`dbscan` for OPERA-driven, `lightning` for
            lightning-driven). Used to suffix every on-disk path.
        finetuned: if True, evaluate the Swin-head fine-tuned model
            (`coalition_<mode>_<source>_finetuned.keras` /
            `history_<mode>_<source>_finetuned.json`) instead of the
            base model. The dataset path is unchanged - the fine-tune
            stage trains against the same TFRecord shards.
    """
    data_root = Path(data_root)
    model_dir = Path(model_dir)
    run_tag = f"{mode}_{source}"
    artifact_tag = f"{run_tag}_finetuned" if finetuned else run_tag
    output_dir = Path(output_dir) / f"eval_{artifact_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Authoritative label_type comes from the dataset's metadata.json
    # (written by create_datasets.py). Mode-name heuristics misclassify
    # modes like `mtg_lightning_opera` which has 'lightning' in its name
    # but targets OPERA 5-class rainfall.
    meta_path = data_root / "datasets" / run_tag / split / "metadata.json"
    if meta_path.is_file():
        with open(meta_path) as f:
            label_type = json.load(f).get(
                "label_type",
                "lightning" if "lightning" in mode else "radar",
            )
    else:
        label_type = "lightning" if "lightning" in mode else "radar"

    print("=" * 70)
    print(f"COALITION-4 Evaluation - Mode: {mode}  Source: {source}  "
          f"{'(finetuned)' if finetuned else '(base)'}")
    print("=" * 70)
    print(f"  Label type:    {label_type}")
    print(f"  Split:         {split}")
    print(f"  Threshold:     {threshold if threshold is not None else 'optimize on validation'}")
    print(f"  Plot thresh:   {plot_threshold}")

    SPLIT_CSV = {
        "train":      f"train_data_{source}.csv",
        "validation": f"validation_data_{source}.csv",
        "test":       f"test_data_{source}.csv",
    }
    csv_name = SPLIT_CSV[split]

    # ---- 1. Plot training history ----
    history_path = model_dir / f"history_{artifact_tag}.json"
    if history_path.is_file():
        print(f"\n1. Plotting training history from {history_path}")
        plot_training_history(history_path, output_dir)
    else:
        print(f"\n1. WARNING: History file not found: {history_path}")

    # ---- 2. Load model (with mixed precision matching training) ----
    model_path = model_dir / f"coalition_{artifact_tag}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n2. Loading model from {model_path}")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Custom objects for deserialization (all defined above). SwinBlock
    # is only present in `*_finetuned.keras` files (the Swin head grafted
    # by train_models.train_finetune); it's harmless to include for the
    # base `coalition_*.keras` too because Keras only looks up names it
    # actually finds in the saved config.
    custom_objects = {
        'ReflectionPadding2D': ReflectionPadding2D,
        'ConvBlock': ConvBlock,
        'ResBlock': ResBlock,
        'GRUResBlock': GRUResBlock,
        'ResGRU': ResGRU,
        'SwinBlock': SwinBlock,
        'WeightedFocalLoss': WeightedFocalLoss,
        'iou_metric': iou_metric,
        'true_pos': true_pos,
        'false_pos': false_pos,
        'false_neg': false_neg,
    }

    if finetuned:
        # `tf.keras.models.load_model` of a fine-tuned `.keras` is fragile
        # under TF 2.10: the saved model contains a sub-Model (the frozen
        # backbone, wrapped by train_finetune) nested inside the outer
        # Model. Keras's deserializer rebuilds the variable list in a
        # different order than save_model wrote them, so weights line up
        # against the wrong layers and assigns end up with shape
        # mismatches like (1,1,64,32) -> (3,3,64,64).
        #
        # The robust workaround is to rebuild the architecture from
        # scratch using the exact same `build_finetune_model` train_models
        # uses, then call `model.load_weights(...)` which matches by name
        # (not by index). Requires the base checkpoint path because the
        # backbone is loaded fresh from there.
        base_ckpt = model_dir / f"coalition_{run_tag}.keras"
        if not base_ckpt.is_file():
            raise FileNotFoundError(
                f"Fine-tune evaluation needs the base checkpoint at "
                f"{base_ckpt} (the .keras file produced by `--stage base`). "
                f"Make sure both the base and the _finetuned files are in "
                f"--model_dir."
            )

        # Read the Swin hyperparameters used at training time from the
        # history JSON so we rebuild the SAME architecture.
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Fine-tune evaluation needs {history_path} to recover "
                f"the Swin hyperparameters used at training time."
            )
        with open(history_path) as f:
            hist_meta = json.load(f)
        swin_cfg = hist_meta.get("swin", {})
        ones_fraction = hist_meta.get("ones_fraction") or 0.0106

        finetune_cfg = {
            "optimizer":     hist_meta.get("optimizer", "adamw"),
            "weight_decay":  hist_meta.get("weight_decay", 0.01),
            "initial_lr":    hist_meta.get("initial_lr", 3e-4),
            "warmup_epochs": 1,
            "min_lr":        1e-6,
            "epochs":        hist_meta.get("epochs_completed", 1),
            "window_size":   swin_cfg.get("window_size", 8),
            "n_swin_blocks": swin_cfg.get("n_blocks", 2),
            "num_heads":     swin_cfg.get("num_heads", 4),
            "c_shared":      swin_cfg.get("c_shared", 64),
            "head_dropout":  swin_cfg.get("head_dropout", 0.1),
        }

        # Lazy import here so the base-model path doesn't pay the cost of
        # importing train_models when it isn't needed.
        from train_models import build_finetune_model
        model, _loss, _metrics = build_finetune_model(
            base_model_path=str(base_ckpt),
            finetune_cfg=finetune_cfg,
            ones_fraction=ones_fraction,
        )
        model.load_weights(str(model_path))
        print(f"  Fine-tune model rebuilt + weights loaded from "
              f"{model_path.name}")
    else:
        model = tf.keras.models.load_model(
            str(model_path), custom_objects=custom_objects,
        )

    print(f"  Model loaded: {model.count_params():,} parameters")

    # ---- 3. Load datasets ----
    eval_dir = data_root / "datasets" / run_tag / split
    if not eval_dir.exists():
        raise FileNotFoundError(f"{split.capitalize()} dataset not found: {eval_dir}")

    print(f"\n3. Loading {split} dataset from {eval_dir}")
    eval_ds = _load_split(eval_dir)
    eval_ds = eval_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Load validation dataset if threshold optimization needed
    val_ds = None
    if label_type == "lightning" and threshold is None:
        val_dir = data_root / "datasets" / run_tag / "validation"
        if val_dir.exists():
            print(f"  Loading validation dataset for threshold optimization...")
            val_ds = _load_split(val_dir)
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            print(f"  WARNING: Validation dataset not found at {val_dir}")
            print(f"  Falling back to threshold=0.5")
            threshold = 0.5

    # ---- 4. Run evaluation ----
    print(f"\n4. Running evaluation on {split} set...")
    if label_type == "lightning":
        results = evaluate_lightning(model, eval_ds, output_dir,
                                     threshold=threshold, val_ds=val_ds)
    elif label_type == "radar":
        results = evaluate_radar(model, eval_ds, output_dir)
    else:
        raise ValueError(f"Unknown label type: {label_type}")

    # ---- 5. Visualize predictions ----
    if plot_date and plot_hour is not None:
        print(f"\n5. Visualizing predictions for {plot_date} hour {plot_hour}:00 "
              f"(from {csv_name})...")
        try:
            plot_predictions_for_date_hour(model, mode, data_root,
                                            output_dir, plot_date, plot_hour,
                                            plot_threshold=plot_threshold,
                                            csv_name=csv_name,
                                            label_type=label_type,
                                            source=source)
        except Exception as e:
            print(f"  Skipping visualization: {e}")
    else:
        print(f"\n5. Skipping visualization (use --date and --hour to enable)")

    # ---- 6. Save results ----
    results["mode"] = mode
    results["source"] = source
    results["finetuned"] = finetuned
    results["label_type"] = label_type
    results["split"] = split
    results["model_params"] = model.count_params()

    # Load history for wall time info
    if history_path.is_file():
        with open(history_path) as f:
            hist_data = json.load(f)
        results["training_wall_time"] = hist_data.get("total_wall_time")
        results["epochs_completed"] = hist_data.get("epochs_completed")

    results_path = output_dir / f"evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n6. Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    if label_type == "lightning":
        print(f"  Threshold: {results['optimal_threshold']:.3f} "
              f"(source: {results['threshold_source']})")
        print(f"  {'Lead time':<12} {'CSI':>8} {'POD':>8} {'FAR':>8} "
              f"{'ETS':>8} {'HSS':>8} {'PSS':>8}")
        print(f"  {'-'*60}")
        for lt in LEAD_LABELS:
            m = results["per_leadtime"][lt]
            print(f"  {lt:<12} {m['CSI']:>8.4f} {m['POD']:>8.4f} "
                  f"{m['FAR']:>8.4f} {m['ETS']:>8.4f} {m['HSS']:>8.4f} "
                  f"{m['PSS']:>8.4f}")
        agg = results["aggregate"]
        print(f"  {'Aggregate':<12} {agg['CSI']:>8.4f} {agg['POD']:>8.4f} "
              f"{agg['FAR']:>8.4f} {agg['ETS']:>8.4f} {agg['HSS']:>8.4f} "
              f"{agg['PSS']:>8.4f}")
        print(f"\n  PR AUC:  {agg['PR_AUC']:.4f}")
        print(f"  ROC AUC: {agg['ROC_AUC']:.4f}")
    else:
        print(f"  {'Lead time':<12} {'Accuracy':>10}")
        print(f"  {'-'*25}")
        for lt in LEAD_LABELS:
            acc = results["per_leadtime"][lt]["accuracy"]
            print(f"  {lt:<12} {acc:>10.4f}")
        print(f"  {'Aggregate':<12} {results['aggregate']['accuracy']:>10.4f}")

    print("=" * 70)
    print(f"All results and plots saved to: {output_dir}")
    print("=" * 70)

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained COALITION-4 model on test set."
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["mtg_lightning", "mtg_radar", "mtg_radar_continuous",
                 "mtg_opera_radar_only", "mtg_opera_mtgmr",
                 "mtg_lightning_opera", "mtg_lightning_opera_occurrence"],
        help="Model variant to evaluate. Matches the modes registered "
             "in train_models.TRAINING_MODES and create_datasets."
    )
    parser.add_argument(
        "--source", type=str, default="dbscan",
        choices=["dbscan", "lightning"],
        help="Sample-selection track the artifacts were built from. "
             "`dbscan` = OPERA-driven (patch_index.csv from "
             "identify_patches --source opera); `lightning` = "
             "lightning-driven (lightning_patches.csv from "
             "identify_lightning_periods.py). Drives every on-disk "
             "path (model, history, datasets/, CSVs)."
    )
    parser.add_argument(
        "--finetuned", action="store_true",
        help="Evaluate the Swin-head fine-tuned model "
             "(coalition_<mode>_<source>_finetuned.keras) instead of "
             "the base coalition_<mode>_<source>.keras. Dataset path "
             "and label_type are unchanged."
    )
    parser.add_argument(
        "--data_root", type=str, default="./our_data",
        help="Root directory containing datasets/ subfolder"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./models",
        help="Directory containing trained model and history"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation",
        help="Directory to save evaluation results and plots"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed decision threshold for lightning metrics (e.g. 0.5). "
             "If not set, optimizes on validation set (paper's approach)."
    )
    parser.add_argument(
        "--plot_threshold", type=float, default=0.5,
        help="Probability threshold for lightning prediction visualization "
             "(default: 0.5). Only pixels above this value are shown."
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date for visualization, e.g. '2025-05-26'"
    )
    parser.add_argument(
        "--hour", type=int, default=None,
        help="Reference hour for visualization (0-23), e.g. 5 → 5:00..5:45"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Which dataset split to evaluate on (default: test)"
    )
    args = parser.parse_args()

    evaluate(
        mode=args.mode,
        data_root=args.data_root,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        plot_threshold=args.plot_threshold,
        plot_date=args.date,
        plot_hour=args.hour,
        split=args.split,
        source=args.source,
        finetuned=args.finetuned,
    )


if __name__ == "__main__":
    main()
    