"""
Feature Importance Analysis for COALITION-4 RCNN Models.

Combines three complementary methods:
  1. Grad-CAM + Xi correlation  (spatial attention, single model)
  2. SHAP (pixel-level importance, single model)
  3. Classical Shapley values   (source-level, multiple models)

Usage:
    python feature_importance_analysis.py --model models/coalition_mtg_lightning.keras \
        --data datasets/mtg_lightning/test --output results/feature_importance \
        --methods gradcam_xi shap
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import zoom
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from c4dl.ml.models.blocks import ConvBlock, ResBlock
from c4dl.ml.models.models import (
    CastLayer, RepeatLayer, iou_metric, dice_metric,
    true_pos, true_neg, false_pos, false_neg,
    make_rain_loss_hist, prob_binary_crossentropy,
    WeightedBinaryCrossentropy, WeightedFocalLoss,
)
from c4dl.ml.models.optimizers import AdaBeliefOptimizer
from c4dl.ml.models.rnn import ResGRU

CUSTOM_OBJECTS = {
    "CastLayer": CastLayer,
    "RepeatLayer": RepeatLayer,
    "ResGRU": ResGRU,
    "ConvBlock": ConvBlock,
    "ResBlock": ResBlock,
    "AdaBeliefOptimizer": AdaBeliefOptimizer,
    "iou_metric": iou_metric,
    "dice_metric": dice_metric,
    "true_pos": true_pos,
    "true_neg": true_neg,
    "false_pos": false_pos,
    "false_neg": false_neg,
    "make_rain_loss_hist": make_rain_loss_hist,
    "prob_binary_crossentropy": prob_binary_crossentropy,
    "WeightedFocalLoss": WeightedFocalLoss,
    "WeightedBinaryCrossentropy": WeightedBinaryCrossentropy,
}


# ============================================================================
# MODULE 1 — MODEL INTROSPECTION
# ============================================================================

def load_model(model_path):
    """Load a .keras model with all required custom objects."""
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    print(f"Loaded model: {model_path}  ({len(model.layers)} layers)")
    return model


def inspect_architecture(model):
    """Dynamically discover model architecture.

    Returns a dict with:
        inputs        – list of {name, shape, channels, resolution}
        resblocks     – list of {index, name, output_shape, phase}
        resgrus       – list of {index, name, output_shape}
        timedist      – list of {index, name, inner, output_shape}
        concatenates  – list of {index, name, output_shape}
        output_layer  – {index, name, output_shape, timesteps, spatial}
        first_resblock, last_resblock – layer names for GradCAM targeting
    """
    arch = {
        "inputs": [],
        "resblocks": [],
        "resgrus": [],
        "timedist": [],
        "concatenates": [],
        "output_layer": None,
        "first_resblock": None,
        "last_resblock": None,
    }

    first_rb_idx = None
    last_rb_idx = None

    for i, layer in enumerate(model.layers):
        out_shape = _safe_output_shape(layer)

        # --- Input layers ---
        if isinstance(layer, tf.keras.layers.InputLayer):
            # shape is (None, T, H, W, C)
            shape_tuple = out_shape[0] if isinstance(out_shape, list) else out_shape
            arch["inputs"].append({
                "name": layer.name,
                "shape": shape_tuple,
                "channels": shape_tuple[-1],
                "timesteps": shape_tuple[1],
                "resolution": (shape_tuple[2], shape_tuple[3]),
            })

        # --- ResBlock ---
        if isinstance(layer, ResBlock):
            phase = "encoder" if first_rb_idx is None or i < len(model.layers) // 2 else "decoder"
            arch["resblocks"].append({
                "index": i, "name": layer.name,
                "output_shape": out_shape, "phase": phase,
            })
            if first_rb_idx is None:
                first_rb_idx = i
            last_rb_idx = i

        # --- ResGRU ---
        if isinstance(layer, ResGRU):
            arch["resgrus"].append({
                "index": i, "name": layer.name, "output_shape": out_shape,
            })

        # --- TimeDistributed ---
        if isinstance(layer, tf.keras.layers.TimeDistributed):
            inner = type(layer.layer).__name__ if hasattr(layer, "layer") else "?"
            arch["timedist"].append({
                "index": i, "name": layer.name,
                "inner": inner, "output_shape": out_shape,
            })

        # --- Concatenate ---
        if isinstance(layer, tf.keras.layers.Concatenate):
            arch["concatenates"].append({
                "index": i, "name": layer.name, "output_shape": out_shape,
            })

    # Output layer
    last = model.layers[-1]
    out_shape = _safe_output_shape(last)
    # output shape is (None, T, H, W, C)
    arch["output_layer"] = {
        "index": len(model.layers) - 1,
        "name": last.name,
        "output_shape": out_shape,
        "timesteps": out_shape[1],
        "spatial": (out_shape[2], out_shape[3]),
    }

    if first_rb_idx is not None:
        arch["first_resblock"] = model.layers[first_rb_idx].name
    if last_rb_idx is not None:
        arch["last_resblock"] = model.layers[last_rb_idx].name

    return arch


def print_architecture_summary(arch):
    """Pretty-print the discovered architecture."""
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)

    print("\nInputs:")
    for inp in arch["inputs"]:
        print(f"  {inp['name']:<20} {inp['resolution'][0]}x{inp['resolution'][1]}"
              f"  channels={inp['channels']}  timesteps={inp['timesteps']}")

    print(f"\nResBlocks: {len(arch['resblocks'])}")
    for rb in arch["resblocks"]:
        print(f"  #{rb['index']:<4} {rb['name']:<30} {rb['phase']:<8} shape={rb['output_shape']}")

    print(f"\nResGRUs: {len(arch['resgrus'])}")
    for rg in arch["resgrus"]:
        print(f"  #{rg['index']:<4} {rg['name']:<30} shape={rg['output_shape']}")

    print(f"\nTimeDistributed: {len(arch['timedist'])}")
    for td in arch["timedist"]:
        print(f"  #{td['index']:<4} {td['name']:<30} inner={td['inner']:<15} shape={td['output_shape']}")

    print(f"\nConcatenate: {len(arch['concatenates'])}")
    for c in arch["concatenates"]:
        print(f"  #{c['index']:<4} {c['name']:<30} shape={c['output_shape']}")

    out = arch["output_layer"]
    print(f"\nOutput: #{out['index']} {out['name']}")
    print(f"  timesteps={out['timesteps']}  spatial={out['spatial']}")
    print(f"\nGradCAM targets: first_resblock={arch['first_resblock']}, "
          f"last_resblock={arch['last_resblock']}")
    print("=" * 70)


def _safe_output_shape(layer):
    try:
        return layer.output_shape
    except Exception:
        return "unknown"


# ============================================================================
# MODULE 2 — GRAD-CAM COMPUTATION
# ============================================================================

def compute_gradcam(model, input_data, target_layer_name, output_timestep=None):
    """Core Grad-CAM for a single target layer.

    Args:
        model: Keras model.
        input_data: Single sample (batch_size=1), as list of arrays.
        target_layer_name: Name of the convolutional layer to target.
        output_timestep: Which output timestep to maximise (None = average all).

    Returns:
        2-D numpy heatmap normalised to [0, 1].
    """
    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        tape.watch(conv_outputs)

        if output_timestep is not None and len(predictions.shape) == 5:
            # predictions: (batch, T, H, W, C) — target specific timestep
            target = predictions[0, output_timestep, :, :, :]
        else:
            target = predictions[0]
        loss = tf.reduce_mean(target)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        spatial = _safe_output_shape(target_layer)
        if isinstance(spatial, (list, tuple)) and len(spatial) >= 4:
            return np.zeros((spatial[2], spatial[3]))
        return np.zeros((64, 64))

    # Handle 5-D (batch, T, H, W, C) or 4-D (batch, H, W, C)
    if len(grads.shape) == 5:
        if output_timestep is not None and output_timestep < grads.shape[1]:
            grads_slice = grads[0, output_timestep]
            conv_slice = conv_outputs[0, output_timestep]
        else:
            grads_slice = tf.reduce_mean(grads[0], axis=0)
            conv_slice = tf.reduce_mean(conv_outputs[0], axis=0)
    else:
        grads_slice = grads[0]
        conv_slice = conv_outputs[0]

    # Channel importance weights via global average pooling
    weights = tf.reduce_mean(grads_slice, axis=(0, 1))  # (C,)

    # Weighted combination of feature maps
    cam = tf.reduce_sum(conv_slice * weights, axis=-1)  # (H, W)
    cam = tf.nn.relu(cam)

    cam_max = tf.reduce_max(cam)
    if cam_max > 1e-8:
        cam = cam / cam_max

    return cam.numpy()


def compute_gradcam_per_input(model, input_data, arch, target_layer_name=None):
    """Compute one Grad-CAM heatmap per model input.

    The first Concatenate layer merges inputs by resolution. We use the
    first ResBlock (or a user-specified layer) as the GradCAM target and
    derive channel ranges from the actual input channel counts.

    Returns:
        dict mapping input_name -> 2-D heatmap.
    """
    if target_layer_name is None:
        target_layer_name = arch["first_resblock"]

    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        tape.watch(conv_outputs)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return {inp["name"]: np.zeros(inp["resolution"]) for inp in arch["inputs"]}

    # Average over time if 5-D
    if len(grads.shape) == 5:
        grads_avg = tf.reduce_mean(grads[0], axis=0)   # (H, W, C)
        conv_avg = tf.reduce_mean(conv_outputs[0], axis=0)
    else:
        grads_avg = grads[0]
        conv_avg = conv_outputs[0]

    total_channels = int(conv_avg.shape[-1])

    # Build channel ranges: inputs that share the same resolution are
    # concatenated in order before the first ResBlock.
    # For the MTG architecture the first concat merges past_hr channels
    # that were downsampled to 128x128 with past_mr, so all feed into
    # the same ResBlock.  The ResBlock's output channels are *learned*
    # and don't map 1-to-1 to input channels. Therefore we compute a
    # *per-input* Grad-CAM by running separate forward passes where we
    # zero-out other inputs.
    heatmaps = {}
    for idx, inp_info in enumerate(arch["inputs"]):
        # Create a masked copy: zero everything except this input
        masked = []
        for j, inp in enumerate(arch["inputs"]):
            if isinstance(input_data, list):
                if j == idx:
                    masked.append(input_data[j])
                else:
                    masked.append(tf.zeros_like(input_data[j]))
            elif isinstance(input_data, dict):
                key = inp["name"]
                if j == idx:
                    masked.append(input_data[key])
                else:
                    masked.append(tf.zeros_like(input_data[key]))

        heatmap = compute_gradcam(model, masked, target_layer_name)
        heatmaps[inp_info["name"]] = heatmap

    return heatmaps


def compute_gradcam_per_output_timestep(model, input_data, arch,
                                        target_layer_name=None):
    """Compute Grad-CAM for each output timestep.

    Returns:
        list of 2-D heatmaps, length = number of output timesteps.
    """
    if target_layer_name is None:
        target_layer_name = arch["last_resblock"]

    n_timesteps = arch["output_layer"]["timesteps"]
    heatmaps = []
    for t in range(n_timesteps):
        hm = compute_gradcam(model, input_data, target_layer_name,
                             output_timestep=t)
        heatmaps.append(hm)
    return heatmaps


def batch_average_gradcam(model, dataset, arch, num_samples=4,
                          target_input_layer=None, target_output_layer=None):
    """Process multiple samples and average Grad-CAM results.

    Args:
        model: Keras model.
        dataset: tf.data.Dataset yielding (inputs_dict, label) tuples.
        arch: Architecture dict from inspect_architecture().
        num_samples: How many samples to average over.
        target_input_layer: Layer name for input GradCAM (default: first ResBlock).
        target_output_layer: Layer name for output GradCAM (default: last ResBlock).

    Returns:
        avg_input_gradcams  – dict {input_name: 2-D heatmap}
        avg_output_gradcams – list of 2-D heatmaps (one per output timestep)
        sample_data         – first sample's input data (for visualisation)
        sample_predictions  – first sample's predictions
    """
    all_input_cams = []   # list of dicts
    all_output_cams = []  # list of lists

    sample_data = None
    sample_predictions = None

    for i, (inputs, _label) in enumerate(dataset.take(num_samples)):
        # Add batch dimension
        input_list = [tf.expand_dims(inputs[name], 0)
                      for name in sorted(inputs.keys())]

        if i == 0:
            sample_data = input_list
            sample_predictions = model.predict(input_list, verbose=0)

        print(f"  GradCAM sample {i + 1}/{num_samples} ...", end="\r")

        inp_cams = compute_gradcam_per_input(
            model, input_list, arch, target_layer_name=target_input_layer)
        out_cams = compute_gradcam_per_output_timestep(
            model, input_list, arch, target_layer_name=target_output_layer)

        all_input_cams.append(inp_cams)
        all_output_cams.append(out_cams)

        tf.keras.backend.clear_session()
        gc.collect()

    print(f"  Completed {num_samples} samples.              ")

    # Average input Grad-CAMs
    input_names = list(all_input_cams[0].keys())
    avg_input = {}
    for name in input_names:
        stacked = np.stack([c[name] for c in all_input_cams])
        avg_input[name] = np.mean(stacked, axis=0)

    # Average output Grad-CAMs
    n_ts = len(all_output_cams[0])
    avg_output = []
    for t in range(n_ts):
        stacked = np.stack([c[t] for c in all_output_cams])
        avg_output.append(np.mean(stacked, axis=0))

    return avg_input, avg_output, sample_data, sample_predictions


# ============================================================================
# MODULE 3 — XI CORRELATION
# ============================================================================

def compute_xi_coefficient(x, y):
    """Compute Chatterjee's Xi coefficient between two 1-D arrays.

    A rank-based, non-parametric measure of dependence in [0, 1].
    """
    n = len(x)
    if n < 2:
        return 0.0

    sorted_indices = np.argsort(x)
    y_sorted = y[sorted_indices]
    y_ranks = np.argsort(np.argsort(y_sorted)).astype(np.float64)

    numerator = np.sum(np.abs(np.diff(y_ranks)))
    denominator = 2.0 * n ** 2 / 3.0

    if denominator == 0:
        return 0.0

    xi = 1.0 - (numerator / denominator)
    return float(np.clip(xi, 0.0, 1.0))


def compute_xi_matrix(input_gradcams, output_gradcams):
    """Compute Xi between every input Grad-CAM and every output timestep.

    Args:
        input_gradcams: dict {name: 2-D array}
        output_gradcams: list of 2-D arrays (one per output timestep)

    Returns:
        pd.DataFrame of shape (num_inputs, num_timesteps) with Xi values.
    """
    input_names = list(input_gradcams.keys())
    n_timesteps = len(output_gradcams)

    xi_data = np.full((len(input_names), n_timesteps), np.nan)

    for i, name in enumerate(input_names):
        inp_cam = input_gradcams[name]
        for t in range(n_timesteps):
            out_cam = output_gradcams[t]

            # Resize to common resolution
            if inp_cam.shape != out_cam.shape:
                scale = (out_cam.shape[0] / inp_cam.shape[0],
                         out_cam.shape[1] / inp_cam.shape[1])
                inp_resized = zoom(inp_cam, scale, order=1)
            else:
                inp_resized = inp_cam

            x = inp_resized.flatten()
            y = out_cam.flatten()

            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
                continue

            xi_data[i, t] = compute_xi_coefficient(x, y)

    columns = [f"T+{(t + 1) * 5}min" for t in range(n_timesteps)]
    return pd.DataFrame(xi_data, index=input_names, columns=columns)


# ============================================================================
# MODULE 4 — SHAP ANALYSIS
# ============================================================================

def compute_shap_values(model, background_data, test_data, input_names,
                        max_background=50, max_test=4):
    """Compute SHAP values per model input.

    Uses DeepExplainer (falls back to GradientExplainer).

    Returns:
        dict {input_name: {"spatial_importance": 2-D, "global_importance": float}}
    """
    import shap  # optional dependency

    # Subsample
    bg = [b[:max_background] for b in background_data]
    ts = [t[:max_test] for t in test_data]

    print("Initialising SHAP explainer ...")
    try:
        explainer = shap.DeepExplainer(model, bg)
    except Exception as e:
        print(f"  DeepExplainer failed ({e}), trying GradientExplainer ...")
        explainer = shap.GradientExplainer(model, bg)

    print("Computing SHAP values ...")
    shap_values = explainer.shap_values(ts)

    # shap_values is a list matching the input structure
    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    results = {}
    for idx, name in enumerate(input_names):
        if idx >= len(shap_values):
            break

        sv = shap_values[idx]
        # sv shape: (samples, T, H, W, C) or (samples, H, W, C)
        if sv.ndim == 5:
            spatial = np.mean(np.abs(sv), axis=(0, 1, 4))
        elif sv.ndim == 4:
            spatial = np.mean(np.abs(sv), axis=(0, 3))
        else:
            spatial = np.mean(np.abs(sv), axis=0)

        results[name] = {
            "spatial_importance": spatial,
            "global_importance": float(np.mean(np.abs(sv))),
        }
        print(f"  {name}: global SHAP = {results[name]['global_importance']:.6f}")

    return results


# ============================================================================
# MODULE 5 — CLASSICAL SHAPLEY VALUES
# ============================================================================

def load_evaluation_scores(results_dir, prefix="lightning", file_type="eval",
                           score_index=0):
    """Load evaluation CSVs for all source combinations.

    Wraps the logic from c4dl.analysis.shapley.load_scores.
    Expected file pattern: {file_type}-{prefix}-{sources}.csv
    """
    import re

    files = os.listdir(results_dir)
    pattern = re.compile(rf"{file_type}-{prefix}-(?P<sources>.+)\.csv")
    scores = {}

    for fn in files:
        m = pattern.match(fn)
        if m is None:
            continue
        sources = m.group("sources")
        if sources == "null":
            sources = ""
        path = os.path.join(results_dir, fn)
        vals = np.loadtxt(path)
        if np.ndim(vals) == 0:
            vals = [float(vals)]
        scores[sources] = vals[score_index]

    print(f"Loaded {len(scores)} source-combination scores from {results_dir}")
    return scores


def compute_classical_shapley(scores, source):
    """Compute the Shapley value for *source* given a coalition score dict.

    ``scores`` maps frozenset-compatible source strings to scalar scores.
    ``source`` is the single character/string whose contribution to measure.
    """
    scores = {frozenset(k): v for (k, v) in scores.items()}
    source_keys = [k for k in scores if source in k]
    N = max((len(k) for k in source_keys), default=0)
    source_set = set(source)

    s = []
    for num in range(1, N + 1):
        keys_num = [k for k in source_keys if len(k) == num]
        contribs = []
        for coalition in keys_num:
            without = coalition - source_set
            if without in scores:
                contribs.append(scores[coalition] - scores[without])
        if contribs:
            s.append(np.mean(contribs))

    return float(np.mean(s)) if s else 0.0


def compute_shapley_by_leadtime(results_dir, prefix="lightning",
                                file_type="eval_leadtime"):
    """Compute Shapley values per lead-time step.

    Expects files like eval_leadtime-lightning-{sources}.csv with one score
    per lead-time row.
    """
    import re

    files = os.listdir(results_dir)
    pattern = re.compile(rf"{file_type}-{prefix}-(?P<sources>.+)\.csv")
    scores_by_lt = {}

    for fn in files:
        m = pattern.match(fn)
        if m is None:
            continue
        sources = m.group("sources")
        if sources == "null":
            sources = ""
        path = os.path.join(results_dir, fn)
        vals = np.loadtxt(path)
        scores_by_lt[sources] = vals  # array of length num_leadtimes

    if not scores_by_lt:
        print("No lead-time score files found.")
        return {}

    n_lt = len(next(iter(scores_by_lt.values())))
    all_sources = set()
    for k in scores_by_lt:
        all_sources.update(k)

    shapley_lt = {src: np.zeros(n_lt) for src in all_sources}
    for src in all_sources:
        for t in range(n_lt):
            scores_t = {k: v[t] for k, v in scores_by_lt.items()}
            shapley_lt[src][t] = compute_classical_shapley(scores_t, src)

    return shapley_lt


# ============================================================================
# MODULE 6 — VISUALIZATION
# ============================================================================

def plot_gradcam_comparison(input_gradcam, output_gradcam, input_name,
                            timestep_idx, xi_value, raw_input=None,
                            raw_output=None):
    """5-panel figure: raw input, input GradCAM, raw output, output GradCAM, overlay."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 1 — Raw input
    if raw_input is not None:
        im0 = axes[0].imshow(raw_input, cmap="viridis")
        axes[0].set_title(f"Raw Input\n{input_name}", fontweight="bold")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
    else:
        axes[0].text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=axes[0].transAxes)
    axes[0].axis("off")

    # 2 — Input Grad-CAM
    im1 = axes[1].imshow(input_gradcam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Input Grad-CAM\n{input_name}", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="Attention")

    # 3 — Raw output
    if raw_output is not None:
        im2 = axes[2].imshow(raw_output, cmap="viridis")
        axes[2].set_title(f"Raw Output\nT+{(timestep_idx + 1) * 5}min",
                          fontweight="bold")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        axes[2].text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=axes[2].transAxes)
    axes[2].axis("off")

    # 4 — Output Grad-CAM
    im3 = axes[3].imshow(output_gradcam, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title(f"Output Grad-CAM\nT+{(timestep_idx + 1) * 5}min",
                      fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, label="Attention")

    # 5 — Overlay
    if input_gradcam.shape != output_gradcam.shape:
        scale = (output_gradcam.shape[0] / input_gradcam.shape[0],
                 output_gradcam.shape[1] / input_gradcam.shape[1])
        inp_resized = zoom(input_gradcam, scale, order=1)
    else:
        inp_resized = input_gradcam
    axes[4].imshow(inp_resized, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
    axes[4].imshow(output_gradcam, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    axes[4].set_title(f"Overlay (Blue=In, Red=Out)\nXi={xi_value:.4f}",
                      fontweight="bold")
    axes[4].axis("off")

    plt.suptitle(f"Grad-CAM: {input_name} vs Output T+{(timestep_idx + 1) * 5}min",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_xi_heatmap(xi_df):
    """Interactive Plotly heatmap of Xi matrix (inputs x timesteps)."""
    fig = go.Figure(data=go.Heatmap(
        z=xi_df.values,
        x=list(xi_df.columns),
        y=list(xi_df.index),
        colorscale="RdYlBu",
        zmid=0.5, zmin=0, zmax=1,
        colorbar=dict(title="Xi"),
        text=np.round(xi_df.values, 3),
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    fig.update_layout(
        title="Xi Correlation: Input Grad-CAM vs Output Grad-CAM",
        xaxis_title="Output Timestep",
        yaxis_title="Input",
        height=max(400, 80 * len(xi_df)),
        width=1000,
    )
    return fig


def plot_xi_bar_chart(xi_df, top_n=10):
    """Clustered bar chart of top-N inputs by average Xi across timesteps."""
    avg = xi_df.mean(axis=1).sort_values(ascending=False)
    top = avg.head(top_n).index.tolist()
    df_top = xi_df.loc[top]

    fig = go.Figure()
    for name in top:
        fig.add_trace(go.Bar(
            x=list(xi_df.columns),
            y=df_top.loc[name].values,
            name=name,
            text=np.round(df_top.loc[name].values, 3),
            textposition="auto",
        ))

    fig.update_layout(
        title=f"Top {top_n} Inputs by Xi Correlation",
        xaxis_title="Output Timestep",
        yaxis_title="Xi",
        barmode="group",
        height=600, width=1200,
        legend=dict(orientation="v", yanchor="top", y=1,
                    xanchor="left", x=1.02),
    )
    return fig


def plot_xi_boxplots(xi_df):
    """Box plots showing Xi distribution across timesteps per input."""
    fig = go.Figure()
    for name in xi_df.index:
        fig.add_trace(go.Box(
            y=xi_df.loc[name].values,
            name=name, boxmean="sd",
        ))
    fig.update_layout(
        title="Xi Distribution Across Output Timesteps",
        yaxis_title="Xi",
        xaxis_title="Input",
        height=600, width=max(800, 120 * len(xi_df)),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_shap_spatial_maps(shap_results, top_n=10):
    """Matplotlib grid of SHAP spatial importance maps."""
    sorted_inputs = sorted(shap_results.items(),
                           key=lambda x: x[1]["global_importance"],
                           reverse=True)[:top_n]

    n_cols = min(5, len(sorted_inputs))
    n_rows = (len(sorted_inputs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (name, data) in enumerate(sorted_inputs):
        ax = axes[i // n_cols, i % n_cols]
        im = ax.imshow(data["spatial_importance"], cmap="Reds")
        ax.set_title(f"{name}\nSHAP={data['global_importance']:.4f}",
                     fontsize=9, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide empty axes
    for j in range(len(sorted_inputs), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    plt.suptitle("SHAP Spatial Importance (Top Inputs)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_bar_chart(shap_results):
    """Interactive bar chart of global SHAP importance."""
    names = sorted(shap_results, key=lambda n: shap_results[n]["global_importance"],
                   reverse=True)
    vals = [shap_results[n]["global_importance"] for n in names]

    fig = go.Figure(data=[go.Bar(
        x=names, y=vals, marker_color="crimson",
        text=[f"{v:.4f}" for v in vals], textposition="auto",
    )])
    fig.update_layout(
        title="Global SHAP Importance",
        xaxis_title="Input", yaxis_title="Mean |SHAP|",
        height=500, width=max(600, 100 * len(names)),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_method_comparison(comparison_df):
    """Side-by-side normalised bar chart of all available methods."""
    methods = [c for c in comparison_df.columns if c != "Input"]
    fig = go.Figure()

    colors = {"Xi_mean": "steelblue", "SHAP": "crimson",
              "Classical_Shapley": "forestgreen"}

    for method in methods:
        vals = comparison_df[method].values
        vmax = np.nanmax(vals)
        normed = vals / vmax if vmax > 0 else vals
        fig.add_trace(go.Bar(
            name=method,
            x=comparison_df["Input"],
            y=normed,
            marker_color=colors.get(method, None),
        ))

    fig.update_layout(
        title="Normalised Feature Importance: Method Comparison",
        xaxis_title="Input", yaxis_title="Normalised Importance",
        barmode="group", height=600, width=1400,
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_shapley_by_leadtime(shapley_lt, interval_min=5):
    """Line plot of normalised Shapley values over lead time."""
    if not shapley_lt:
        return None

    n_lt = len(next(iter(shapley_lt.values())))
    leadtimes = np.arange(1, n_lt + 1) * interval_min

    # Normalise
    val_sum = None
    for v in shapley_lt.values():
        val_sum = v.copy() if val_sum is None else val_sum + v

    fig, ax = plt.subplots(figsize=(8, 4))
    for src, vals in shapley_lt.items():
        ax.plot(leadtimes, vals / val_sum, linewidth=1.5, label=src)

    ax.set_xlabel("Lead time [min]")
    ax.set_ylabel("Normalised Shapley value")
    ax.set_xlim(leadtimes[0], leadtimes[-1])
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ablation_impact(xi_df_full, xi_df_ablated):
    """Compare Xi matrices from a full model and an ablated one.

    Generic input-ablation comparison: train a second model with one
    input group removed, then diff the two Xi matrices to see how the
    remaining inputs take over the dropped group's role. The mode set
    is already an ablation ladder, so e.g.

        full    = mtg_opera_mtgmr_rainfall        (OPERA + MTG IR/WV)
        ablated = mtg_opera_radar_only_rainfall   (OPERA only)

    isolates MTG IR/WV, and

        full    = mtg_lightning_opera_rainfall    (+ LINET)
        ablated = mtg_opera_mtgmr_rainfall

    isolates lightning. Same comparison bundle_eval_scores.py encodes as
    its coalition letters.
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Full model", "Ablated model"])

    for col, df in enumerate([xi_df_full, xi_df_ablated], start=1):
        fig.add_trace(go.Heatmap(
            z=df.values, x=list(df.columns), y=list(df.index),
            colorscale="RdYlBu", zmid=0.5, zmin=0, zmax=1,
            text=np.round(df.values, 3), texttemplate="%{text}",
            textfont={"size": 8}, showscale=(col == 2),
        ), row=1, col=col)

    fig.update_layout(title="Input-ablation impact on Xi correlations",
                      height=600, width=1400)
    return fig


def plot_prediction_diagnostics(model, dataset, arch, num_samples=8,
                                label_name="Rainfall", interval_min=5):
    """Generate a 4-panel evaluation diagnostic figure.

    Panels:
        1. MAE over forecast time (with std band)
        2. RMSE over forecast time (with std band)
        3. Mean predicted vs target value over forecast time
        4. Per-sample MAE heatmap

    Args:
        model: Keras model.
        dataset: tf.data.Dataset yielding (inputs_dict, label).
        arch: Architecture dict from inspect_architecture().
        num_samples: Number of samples to evaluate.
        label_name: Variable name for axis labels (e.g. "Rainfall", "Lightning").
        interval_min: Minutes between output timesteps.

    Returns:
        matplotlib Figure.
    """
    input_names = sorted([inp["name"] for inp in arch["inputs"]])
    n_timesteps = arch["output_layer"]["timesteps"]
    leadtimes = np.arange(1, n_timesteps + 1) * interval_min

    all_preds = []
    all_targets = []

    for inputs, label in dataset.take(num_samples):
        input_list = [tf.expand_dims(inputs[name], 0) for name in input_names]
        pred = model.predict(input_list, verbose=0)

        # pred shape: (1, T, H, W, C) or (1, H, W, C)
        if isinstance(pred, tf.Tensor):
            pred = pred.numpy()
        label_np = label.numpy()

        # Ensure 4-D per sample: (T, H, W, C)
        if pred.ndim == 5:
            pred = pred[0]
        if label_np.ndim == 4:
            pass  # already (T, H, W, C)
        elif label_np.ndim == 3:
            label_np = label_np[np.newaxis, ...]

        all_preds.append(pred)
        all_targets.append(label_np)

    all_preds = np.array(all_preds)      # (S, T, H, W, C)
    all_targets = np.array(all_targets)  # (S, T, H, W, C)

    # --- Per-timestep metrics across all spatial pixels ---
    # MAE per sample per timestep: mean over (H, W, C)
    mae_per_sample = np.mean(np.abs(all_preds - all_targets), axis=(2, 3, 4))  # (S, T)
    mae_mean = np.mean(mae_per_sample, axis=0)  # (T,)
    mae_std = np.std(mae_per_sample, axis=0)

    # RMSE per sample per timestep
    mse_per_sample = np.mean((all_preds - all_targets) ** 2, axis=(2, 3, 4))  # (S, T)
    rmse_per_sample = np.sqrt(mse_per_sample)
    rmse_mean = np.mean(rmse_per_sample, axis=0)
    rmse_std = np.std(rmse_per_sample, axis=0)

    # Mean value per timestep
    mean_pred = np.mean(all_preds, axis=(0, 2, 3, 4))   # (T,)
    mean_target = np.mean(all_targets, axis=(0, 2, 3, 4))

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: MAE over time
    ax = axes[0, 0]
    ax.plot(leadtimes, mae_mean, "o-", color="steelblue", linewidth=2)
    ax.fill_between(leadtimes, mae_mean - mae_std, mae_mean + mae_std,
                    color="steelblue", alpha=0.2)
    ax.set_title("MAE Over Time", fontweight="bold")
    ax.set_xlabel("Forecast Time (minutes)")
    ax.set_ylabel("Mean Absolute Error")
    ax.grid(True, alpha=0.3)

    # Panel 2: RMSE over time
    ax = axes[0, 1]
    ax.plot(leadtimes, rmse_mean, "s-", color="forestgreen", linewidth=2)
    ax.fill_between(leadtimes, rmse_mean - rmse_std, rmse_mean + rmse_std,
                    color="forestgreen", alpha=0.2)
    ax.set_title("RMSE Over Time", fontweight="bold")
    ax.set_xlabel("Forecast Time (minutes)")
    ax.set_ylabel("Root Mean Square Error")
    ax.grid(True, alpha=0.3)

    # Panel 3: Mean value comparison
    ax = axes[1, 0]
    ax.plot(leadtimes, mean_pred, "^-", color="red", linewidth=1.5,
            label="Predicted")
    ax.plot(leadtimes, mean_target, "o--", color="orange", linewidth=1.5,
            label="Target")
    ax.set_title(f"Mean {label_name} Value Comparison", fontweight="bold")
    ax.set_xlabel("Forecast Time (minutes)")
    ax.set_ylabel(f"Mean {label_name} Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: MAE heatmap (samples x timesteps)
    ax = axes[1, 1]
    timestep_labels = [f"t+{t}min" for t in leadtimes]
    im = ax.imshow(mae_per_sample, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(n_timesteps))
    ax.set_xticklabels(timestep_labels, rotation=45, ha="right")
    ax.set_yticks(range(num_samples))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sample")
    ax.set_title("MAE Heatmap (Samples vs Time)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="MAE")

    plt.suptitle("Prediction Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return fig


# ============================================================================
# MODULE 7 — METHOD COMPARISON (Spearman + Pearson)
# ============================================================================

def compare_methods(xi_df, shap_results=None, classical_scores=None,
                    source_names=None):
    """Build a comparison DataFrame and compute correlations between methods.

    Args:
        xi_df: Xi matrix from Module 3.
        shap_results: dict from Module 4 (optional).
        classical_scores: dict of scalar scores from Module 5 (optional).
        source_names: mapping from input name to classical-Shapley source key.

    Returns:
        comparison_df: DataFrame with one row per input and one column per method.
        correlations: dict of pairwise {(methodA, methodB): {spearman, pearson}}.
    """
    input_names = list(xi_df.index)

    comp = pd.DataFrame({"Input": input_names})
    comp["Xi_mean"] = xi_df.mean(axis=1).values

    if shap_results:
        comp["SHAP"] = [
            shap_results[n]["global_importance"]
            if n in shap_results else np.nan
            for n in input_names
        ]

    if classical_scores and source_names:
        comp["Classical_Shapley"] = [
            classical_scores.get(source_names.get(n), np.nan)
            for n in input_names
        ]

    comp = comp.sort_values("Xi_mean", ascending=False).reset_index(drop=True)

    # Pairwise correlations
    method_cols = [c for c in comp.columns if c != "Input"]
    correlations = {}
    for i, m1 in enumerate(method_cols):
        for m2 in method_cols[i + 1:]:
            v1 = comp[m1].dropna()
            v2 = comp[m2].dropna()
            common = v1.index.intersection(v2.index)
            if len(common) < 3:
                continue
            a, b = v1.loc[common].values, v2.loc[common].values
            sp_r, sp_p = spearmanr(a, b)
            pe_r, pe_p = pearsonr(a, b)
            correlations[(m1, m2)] = {
                "spearman_r": sp_r, "spearman_p": sp_p,
                "pearson_r": pe_r, "pearson_p": pe_p,
            }

    return comp, correlations


def print_correlations(correlations):
    """Pretty-print pairwise method correlations."""
    print("\nMethod Correlations:")
    print("-" * 70)
    for (m1, m2), vals in correlations.items():
        print(f"  {m1} vs {m2}:")
        print(f"    Spearman r = {vals['spearman_r']:.4f}  (p = {vals['spearman_p']:.2e})")
        print(f"    Pearson  r = {vals['pearson_r']:.4f}  (p = {vals['pearson_p']:.2e})")


# ============================================================================
# MODULE 8 — MAIN PIPELINE + CLI
# ============================================================================

def load_test_dataset(data_dir):
    """Load a saved dataset split + its metadata.

    Supports both on-disk formats — TFRecord shards (current) and the
    legacy `tf.data.Dataset.save` snapshot — distinguished by the
    `format` field in `metadata.json`.
    """
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Dataset metadata: mode={meta.get('mode')}, "
              f"split={meta.get('split')}")
    else:
        meta = {}

    fmt = meta.get("format", "tf_dataset_save")
    if fmt != "tfrecord":
        ds = tf.data.Dataset.load(data_dir)
        return ds, meta

    from pathlib import Path
    split_dir = Path(data_dir)
    shard_paths = sorted(str(p) for p in split_dir.glob("shard_*.tfrecord"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No TFRecord shards in {data_dir} (expected "
            f"`shard_*.tfrecord`). Re-run create_datasets.py."
        )
    input_shapes = meta["input_shapes"]
    label_shape = meta["label_shape"]

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
    return ds, meta


def run_analysis(model_path, data_dir, output_dir, methods, num_samples=4,
                 scores_dir=None, model_path_ablated=None,
                 data_dir_ablated=None):
    """Orchestrate the full feature-importance analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model and inspect ---
    model = load_model(model_path)
    arch = inspect_architecture(model)
    print_architecture_summary(arch)

    input_names = [inp["name"] for inp in arch["inputs"]]
    n_timesteps = arch["output_layer"]["timesteps"]

    # --- Load data ---
    print("\nLoading test dataset ...")
    ds, meta = load_test_dataset(data_dir)

    # ===================== Prediction Diagnostics =====================
    print("\n--- Generating Prediction Diagnostics ---")
    diag_fig = plot_prediction_diagnostics(
        model, ds, arch, num_samples=num_samples,
        label_name=meta.get("label_var", "Value"))
    diag_fig.savefig(os.path.join(output_dir, "prediction_diagnostics.png"),
                     dpi=150, bbox_inches="tight")
    plt.close(diag_fig)
    print(f"Saved prediction_diagnostics.png")

    # ===================== Grad-CAM + Xi =====================
    xi_df = None
    avg_input_cams = None
    avg_output_cams = None
    sample_data = None
    sample_preds = None

    if "gradcam_xi" in methods:
        print("\n--- Running Grad-CAM + Xi Analysis ---")
        avg_input_cams, avg_output_cams, sample_data, sample_preds = \
            batch_average_gradcam(model, ds, arch, num_samples=num_samples)

        xi_df = compute_xi_matrix(avg_input_cams, avg_output_cams)
        xi_df.to_csv(os.path.join(output_dir, "xi_matrix.csv"))
        print("\nXi Matrix:")
        print(xi_df.to_string())

        # Save heatmap
        fig = plot_xi_heatmap(xi_df)
        fig.write_html(os.path.join(output_dir, "xi_heatmap.html"))

        # Save bar chart
        fig = plot_xi_bar_chart(xi_df, top_n=min(10, len(input_names)))
        fig.write_html(os.path.join(output_dir, "xi_bar_chart.html"))

        # Save box plots
        fig = plot_xi_boxplots(xi_df)
        fig.write_html(os.path.join(output_dir, "xi_boxplots.html"))

        # Save Grad-CAM comparison PNGs for top inputs at key timesteps
        avg_xi = xi_df.mean(axis=1).sort_values(ascending=False)
        timestep_indices = list(range(n_timesteps))
        for rank, name in enumerate(avg_xi.head(3).index):
            for t in timestep_indices:
                xi_val = xi_df.loc[name].iloc[t]
                raw_in = _extract_raw_input(sample_data, rank, arch)
                raw_out = _extract_raw_output(sample_preds, t)
                fig = plot_gradcam_comparison(
                    avg_input_cams[name], avg_output_cams[t],
                    name, t, xi_val, raw_in, raw_out)
                fig.savefig(os.path.join(
                    output_dir,
                    f"gradcam_rank{rank + 1}_{name}_t{t + 1}.png"),
                    dpi=150, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved Grad-CAM plots to {output_dir}/")

    # ===================== SHAP =====================
    shap_results = None
    if "shap" in methods:
        print("\n--- Running SHAP Analysis ---")
        # Collect background and test batches from dataset
        bg_list, test_list = _prepare_shap_data(ds, arch, max_bg=50,
                                                 max_test=4)
        shap_results = compute_shap_values(model, bg_list, test_list,
                                           input_names)

        fig = plot_shap_spatial_maps(shap_results)
        fig.savefig(os.path.join(output_dir, "shap_spatial_maps.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_shap_bar_chart(shap_results)
        fig.write_html(os.path.join(output_dir, "shap_bar_chart.html"))

        # Save global importance CSV
        shap_df = pd.DataFrame([
            {"Input": n, "SHAP": shap_results[n]["global_importance"]}
            for n in input_names if n in shap_results
        ]).sort_values("SHAP", ascending=False)
        shap_df.to_csv(os.path.join(output_dir, "shap_importance.csv"),
                       index=False)

    # ===================== Classical Shapley =====================
    classical_shapley_vals = None
    shapley_lt = None
    if "classical_shapley" in methods and scores_dir:
        print("\n--- Running Classical Shapley Analysis ---")
        scores = load_evaluation_scores(scores_dir)
        if scores:
            # Compute scalar Shapley per source key present in scores
            all_sources = set()
            for k in scores:
                all_sources.update(k)
            classical_shapley_vals = {
                src: compute_classical_shapley(scores, src)
                for src in all_sources
            }
            print("Classical Shapley values:")
            for src, val in sorted(classical_shapley_vals.items(),
                                   key=lambda x: -x[1]):
                print(f"  {src}: {val:.4f}")

            pd.DataFrame([
                {"Source": s, "Shapley": v}
                for s, v in classical_shapley_vals.items()
            ]).sort_values("Shapley", ascending=False).to_csv(
                os.path.join(output_dir, "classical_shapley.csv"), index=False)

            # Lead-time Shapley
            shapley_lt = compute_shapley_by_leadtime(scores_dir)
            if shapley_lt:
                fig = plot_shapley_by_leadtime(shapley_lt)
                if fig:
                    fig.savefig(os.path.join(output_dir,
                                             "shapley_by_leadtime.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

    # ===================== Method Comparison =====================
    if xi_df is not None:
        print("\n--- Method Comparison ---")
        comp_df, correlations = compare_methods(
            xi_df, shap_results=shap_results,
            classical_scores=classical_shapley_vals,
        )
        comp_df.to_csv(os.path.join(output_dir, "method_comparison.csv"),
                       index=False)
        print_correlations(correlations)

        # Save correlation summary
        corr_rows = []
        for (m1, m2), vals in correlations.items():
            corr_rows.append({
                "Method_A": m1, "Method_B": m2,
                "Spearman_r": vals["spearman_r"],
                "Spearman_p": vals["spearman_p"],
                "Pearson_r": vals["pearson_r"],
                "Pearson_p": vals["pearson_p"],
            })
        if corr_rows:
            pd.DataFrame(corr_rows).to_csv(
                os.path.join(output_dir, "method_correlations.csv"),
                index=False)

        fig = plot_method_comparison(comp_df)
        fig.write_html(os.path.join(output_dir, "method_comparison.html"))

    # ===================== Ablation impact =====================
    if model_path_ablated and data_dir_ablated and xi_df is not None:
        print("\n--- Input-ablation comparison ---")
        model_no = load_model(model_path_ablated)
        arch_no = inspect_architecture(model_no)
        ds_no, _ = load_test_dataset(data_dir_ablated)

        inp_cams_no, out_cams_no, _, _ = batch_average_gradcam(
            model_no, ds_no, arch_no, num_samples=num_samples)
        xi_df_no = compute_xi_matrix(inp_cams_no, out_cams_no)
        xi_df_no.to_csv(os.path.join(output_dir, "xi_matrix_ablated.csv"))

        fig = plot_ablation_impact(xi_df, xi_df_no)
        fig.write_html(os.path.join(output_dir, "ablation_impact.html"))
        del model_no
        tf.keras.backend.clear_session()

    print("\n" + "=" * 70)
    print(f"Analysis complete. Results saved to {output_dir}/")
    print("=" * 70)


# ---- Helpers ----

def _extract_raw_input(sample_data, input_idx, arch):
    """Extract a displayable 2-D slice from sample input data."""
    if sample_data is None or input_idx >= len(sample_data):
        return None
    arr = sample_data[input_idx]
    if isinstance(arr, tf.Tensor):
        arr = arr.numpy()
    # shape: (1, T, H, W, C) -> average time, take first channel
    if arr.ndim == 5:
        return np.mean(arr[0], axis=0)[:, :, 0]
    if arr.ndim == 4:
        return arr[0, :, :, 0]
    return None


def _extract_raw_output(predictions, timestep_idx):
    """Extract a displayable 2-D slice from model predictions."""
    if predictions is None:
        return None
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
    if predictions.ndim == 5:
        return predictions[0, timestep_idx, :, :, 0]
    if predictions.ndim == 4:
        return predictions[0, :, :, 0]
    return None


def _prepare_shap_data(dataset, arch, max_bg=50, max_test=4):
    """Collect samples from a tf.data.Dataset into lists for SHAP."""
    input_names = sorted([inp["name"] for inp in arch["inputs"]])
    bg_accum = {n: [] for n in input_names}
    test_accum = {n: [] for n in input_names}

    for i, (inputs, _) in enumerate(dataset.take(max_bg + max_test)):
        target = test_accum if i < max_test else bg_accum
        for name in input_names:
            target[name].append(inputs[name].numpy())

    bg_list = [np.stack(bg_accum[n]) for n in input_names]
    test_list = [np.stack(test_accum[n]) for n in input_names]
    return bg_list, test_list


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(
        description="Feature importance analysis for COALITION-4 models.")
    parser.add_argument("--model", required=True,
                        help="Path to .keras model file")
    parser.add_argument("--data", required=True,
                        help="Path to saved test dataset directory")
    parser.add_argument("--output", default="results/feature_importance",
                        help="Output directory")
    parser.add_argument("--methods", nargs="+",
                        default=["gradcam_xi", "shap"],
                        choices=["gradcam_xi", "shap", "classical_shapley"],
                        help="Which analyses to run")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to average GradCAM over")
    parser.add_argument("--scores-dir", default=None,
                        help="Directory with eval CSVs for classical Shapley")
    parser.add_argument("--model-ablated", default=None,
                        help="Path to a model trained with one input group "
                             "removed, for the ablation comparison")
    parser.add_argument("--data-ablated", default=None,
                        help="Path to the test dataset for --model-ablated")

    args = parser.parse_args()

    run_analysis(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        methods=args.methods,
        num_samples=args.num_samples,
        scores_dir=args.scores_dir,
        model_path_ablated=args.model_ablated,
        data_dir_ablated=args.data_ablated,
    )


if __name__ == "__main__":
    main()
