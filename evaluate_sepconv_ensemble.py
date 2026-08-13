"""
evaluate_sepconv_ensemble.py — SepConv Ensemble Evaluation (Regression)
========================================================================
Loads 3 base models (each outputting continuous [0,1]), stacks predictions,
applies post-processing thresholds to recover 5 rain rate classes, then
evaluates with the same metrics as COALITION.

Post-processing: continuous prediction → 5 classes via thresholds at
normalized rain rates [10/60, 20/60, 30/60, 40/60].

Usage:
    python evaluate_sepconv_ensemble.py --mode mtg_opera_mtgmr_continuous

Outputs: ./evaluation/eval_sepconv_ensemble_{mode}/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from create_datasets import get_mode_config, load_tfrecord_dataset
from pipeline_config import SOURCE
from train_models import build_run_tag


# ============================================================================
# Constants
# ============================================================================

LEAD_TIMES = [15, 30, 45]
LEAD_LABELS = ["t+15", "t+30", "t+45"]
N_LEAD = 3
CLASS_NAMES = ["R<10", "10≤R<20", "20≤R<30", "30≤R<40", "R≥40"]
N_CLASSES = 5

# Label info (must match create_datasets.py continuous mode)
# OPERA rainfall_rate normalized: clip(R, 0, 70) / 70
# Thresholds at physical 10, 20, 30, 40 mm/h → normalized
THRESHOLDS_NORM = [10.0 / 70.0, 20.0 / 70.0, 30.0 / 70.0, 40.0 / 70.0]


# ============================================================================
# Post-processing: continuous → class index
# ============================================================================

def continuous_to_class(values):
    """Convert continuous [0,1] values to 5-class indices via thresholds.

    Thresholds (normalized by /70): [0.143, 0.286, 0.429, 0.571]
      class 0: val < 10/70     (R < 10 mm/h)
      class 1: 10/70 ≤ val < 20/70  (10 ≤ R < 20)
      class 2: 20/70 ≤ val < 30/70  (20 ≤ R < 30)
      class 3: 30/70 ≤ val < 40/70  (30 ≤ R < 40)
      class 4: val ≥ 40/70     (R ≥ 40)
    """
    cls = np.zeros_like(values, dtype=np.int32)
    for i, t in enumerate(THRESHOLDS_NORM):
        cls[values >= t] = i + 1
    return cls


# ============================================================================
# Ensemble wrapper
# ============================================================================

class SepConvEnsemble:
    """Loads 3 base models (regression), stacks into (batch, 3, 256, 256, 1)."""

    def __init__(self, model_dir, mode):
        model_dir = Path(model_dir)
        self.models = []
        for i in [1, 2, 3]:
            path = model_dir / f"sepconv_ensemble_{mode}_bm{i}.keras"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Base model not found: {path}\n"
                    f"Train all 3 with: python train_sepconv_ensemble.py --mode {mode}")
            self.models.append(tf.keras.models.load_model(
                str(path), compile=False))
            print(f"    Loaded bm{i}: {self.models[-1].count_params():,} params")

    def predict(self, inputs):
        """Run all 3 models, return (batch, 3, 256, 256, 1)."""
        preds = []
        for model in self.models:
            p = model(inputs, training=False)  # (batch, 256, 256, 1)
            preds.append(p)
        return tf.stack(preds, axis=1)  # (batch, 3, 256, 256, 1)

    def count_params(self):
        return sum(m.count_params() for m in self.models)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_radar(ensemble, test_ds, output_dir):
    """Evaluate: continuous predictions → post-process → classification metrics."""
    print("  Collecting predictions and post-processing...")

    conf_matrices = {t: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
                     for t in range(N_LEAD)}

    for inputs, labels in test_ds:
        # predictions: (batch, 3, 256, 256, 1)
        preds_raw = ensemble.predict(inputs).numpy()
        labels_np = labels.numpy()  # (batch, 3, 256, 256, 1)

        for t in range(N_LEAD):
            # GT: continuous → class index via same thresholds
            y_true_cls = continuous_to_class(
                labels_np[:, t, :, :, 0]).ravel()
            # Pred: continuous → class index via thresholds
            y_pred_cls = continuous_to_class(
                preds_raw[:, t, :, :, 0]).ravel()

            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    conf_matrices[t][i, j] += np.sum(
                        (y_true_cls == i) & (y_pred_cls == j))

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
                "precision": float(precision), "recall": float(recall),
                "f1": float(f1), "support": int(cm[c, :].sum()),
            }

        results["per_leadtime"][LEAD_LABELS[t]] = {
            "accuracy": float(accuracy),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }
        print(f"    {LEAD_LABELS[t]}: accuracy={accuracy:.4f}")

    total = conf_agg.sum()
    results["aggregate"]["accuracy"] = float(conf_agg.trace() / (total + 1e-10))
    results["aggregate"]["confusion_matrix"] = conf_agg.tolist()

    # ==================== PLOTS ====================

    # 1. Accuracy per lead time
    fig, ax = plt.subplots(figsize=(7, 5))
    accs = [results["per_leadtime"][lt]["accuracy"] for lt in LEAD_LABELS]
    ax.plot(LEAD_TIMES, accs, 'o-', linewidth=2, markersize=8, color='#d62728')
    ax.axhline(y=results["aggregate"]["accuracy"], color='gray',
               linestyle='--', alpha=0.5,
               label=f"agg={results['aggregate']['accuracy']:.3f}")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Radar accuracy per lead time")
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
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center',
                        va='center', fontsize=7,
                        color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.suptitle("Confusion matrices (row-normalized)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    # 3. Per-class F1
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, name in enumerate(CLASS_NAMES):
        f1s = [results["per_leadtime"][lt]["per_class"][name]["f1"]
               for lt in LEAD_LABELS]
        ax.plot(LEAD_TIMES, f1s, 'o-', linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 score vs lead time")
    ax.set_xticks(LEAD_TIMES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_per_class.png", dpi=150, bbox_inches='tight')
    plt.close()

    return results


# ============================================================================
# Training history plot
# ============================================================================

def plot_training_history(history_path, output_dir):
    with open(history_path) as f:
        data = json.load(f)

    base_models = data.get("base_models", {})
    if not base_models:
        return

    colors = {'bm1': '#1f77b4', 'bm2': '#ff7f0e', 'bm3': '#2ca02c'}
    labels = {'bm1': 'Bm1 (t+15)', 'bm2': 'Bm2 (t+30)', 'bm3': 'Bm3 (t+45)'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, bm in sorted(base_models.items()):
        hist = bm["history"]
        epochs = range(1, len(hist["loss"]) + 1)
        axes[0].plot(epochs, hist["loss"], '-', linewidth=2,
                     color=colors.get(key), label=labels.get(key))
        if "val_loss" in hist:
            axes[1].plot(range(1, len(hist["val_loss"]) + 1),
                         hist["val_loss"], '-', linewidth=2,
                         color=colors.get(key), label=labels.get(key))

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training history", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves")


# ============================================================================
# Sample predictions
# ============================================================================

def compute_classification_confidence(pred_continuous):
    """Compute confidence of thresholded classification.

    For each pixel, measures how far the prediction is from the nearest
    class boundary. 1.0 = at class center, 0.0 = right at a boundary.

    Args:
        pred_continuous: array of shape (...) with values in [0, 1]
    Returns:
        confidence array of same shape in [0, 1]
    """
    boundaries = np.array([0.0] + THRESHOLDS_NORM + [1.0])
    cls = continuous_to_class(pred_continuous)

    lower = boundaries[cls]
    upper = boundaries[cls + 1]
    center = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0

    # Avoid division by zero for edge cases
    half_width = np.maximum(half_width, 1e-8)
    confidence = 1.0 - np.abs(pred_continuous - center) / half_width
    return np.clip(confidence, 0.0, 1.0)


def plot_predictions_for_date_hour(ensemble, data_root, mode, output_dir,
                                    plot_date, plot_hour,
                                    csv_name="test_data.csv"):
    """Plot all patches for all timesteps matching a given date and hour.

    Filters the specified CSV for rows where:
      - date matches plot_date
      - reference_utc hour matches plot_hour (e.g. hour=5 → refs 5:00..5:45)

    Generates one plot per (row, patch) combination.
    3 rows: GT class, Pred class, Classification confidence.

    Args:
        ensemble: SepConvEnsemble model
        data_root: path to our_data/
        mode: dataset mode string
        output_dir: where to save plots
        plot_date: date string e.g. "2025-05-26"
        plot_hour: integer hour 0-23
        csv_name: CSV filename to read (default: "test_data.csv")
    """
    import ast
    import pandas as pd
    from datetime import datetime, timedelta

    try:
        from create_datasets import (
            get_mode_config, load_and_transform_group, load_label,
            LABEL_CHANNELS,
        )
    except ImportError:
        print("  WARNING: Could not import create_datasets. Skipping.")
        return

    data_root = Path(data_root)
    csv_path = data_root / csv_name
    patches_dir = data_root / "patches"
    if not csv_path.is_file():
        print(f"  WARNING: {csv_path} not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    mode_config = get_mode_config(mode)
    label_var = mode_config["label_var"]
    label_transform_fn = mode_config["label_transform"]
    label_suffix = mode_config["label_suffix"]
    n_label_ch = LABEL_CHANNELS["radar_continuous"]

    input_groups = {}
    for key in ["past_hr", "past_mr"]:
        cfg = mode_config.get(key)
        if cfg is not None:
            input_groups[key] = cfg

    INPUT_COLS = ["idx_t-30", "idx_t-15", "idx_t0"]
    LABEL_COLS = ["idx_t+15", "idx_t+30", "idx_t+45"]
    T_OFFSETS = [-30, -15, 0, 15, 30, 45]

    # Filter rows by date and hour
    # Reference times in the target hour: H:00, H:15, H:30, H:45
    target_refs = [f"{plot_hour}:{m:02d}" for m in [0, 15, 30, 45]]
    mask = (df["date"] == plot_date)
    matching = df[mask].copy()
    matching["ref_clean"] = matching["reference_utc"].str.strip()
    matching = matching[matching["ref_clean"].isin(target_refs)]

    if len(matching) == 0:
        # Show available options to help the user
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

    print(f"  Found {len(matching)} rows for {plot_date} hour {plot_hour}:00")
    class_cmap = matplotlib.colormaps.get_cmap('RdYlGn_r').resampled(N_CLASSES)
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
        for col in INPUT_COLS + LABEL_COLS:
            idx_lists[col] = ast.literal_eval(row[col])

        future_hhmms = [hhmm_list[3 + t] for t in range(3)]

        for p_pos in range(len(patch_numbers)):
            patch_num = patch_numbers[p_pos]

            # Load inputs
            input_tensors = {key: [] for key in input_groups}
            ok = True
            for t_idx in range(3):
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

            # Load GT labels (continuous)
            rad_labels = []
            for t_idx in range(3):
                col = LABEL_COLS[t_idx]
                hhmm = hhmm_list[3 + t_idx]
                npy_idx = idx_lists[col][p_pos]
                rad_labels.append(load_label(
                    str(patches_dir), date_str, hhmm,
                    label_var, label_transform_fn, label_suffix,
                    npy_idx, n_label_channels=n_label_ch))

            model_inputs = {}
            for key in input_tensors:
                model_inputs[key] = np.expand_dims(
                    np.stack(input_tensors[key], axis=0), axis=0).astype(np.float32)

            rad_gt = np.stack(rad_labels, axis=0)  # (3, 256, 256, 1)
            gt_cls = continuous_to_class(rad_gt[:, :, :, 0])

            pred_raw = ensemble.predict(model_inputs).numpy()[0]  # (3, 256, 256, 1)
            pred_cls = continuous_to_class(pred_raw[:, :, :, 0])
            pred_conf = compute_classification_confidence(pred_raw[:, :, :, 0])

            title_meta = (f"Date: {date_str}  |  Ref: {ref_utc} UTC  |  "
                          f"Patch #{patch_num}")

            # 3 rows: GT class, Pred class, Confidence
            fig, axes = plt.subplots(3, N_LEAD, figsize=(16, 14))
            fig.subplots_adjust(left=0.03, right=0.88, top=0.91,
                                bottom=0.03, hspace=0.22, wspace=0.08)

            for t in range(N_LEAD):
                ax = axes[0, t]
                im_gt = ax.imshow(gt_cls[t], cmap=class_cmap,
                                  vmin=-0.5, vmax=N_CLASSES - 0.5,
                                  interpolation='nearest', origin='lower')
                ax.set_title(f"GT — {LEAD_LABELS[t]} ({future_hhmms[t]} UTC)",
                             fontsize=10)
                ax.axis('off')

                ax = axes[1, t]
                im_pred = ax.imshow(pred_cls[t], cmap=class_cmap,
                                    vmin=-0.5, vmax=N_CLASSES - 0.5,
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
            cb1 = fig.colorbar(im_gt, cax=cbar_ax1)
            cb1.set_ticks(range(N_CLASSES)); cb1.set_ticklabels(CLASS_NAMES)

            cbar_ax2 = fig.add_axes([0.90, 0.36, 0.015, 0.24])
            cb2 = fig.colorbar(im_pred, cax=cbar_ax2)
            cb2.set_ticks(range(N_CLASSES)); cb2.set_ticklabels(CLASS_NAMES)

            cbar_ax3 = fig.add_axes([0.90, 0.05, 0.015, 0.24])
            fig.colorbar(im_conf, cax=cbar_ax3, label="Confidence")

            fig.suptitle(f"{title_meta}",
                         fontsize=13, fontweight='bold', y=0.96)

            fn = (f"sepconv_patch{patch_num}_{date_str}_"
                  f"{ref_utc.replace(':', '')}.png")
            plt.savefig(output_dir / fn, dpi=150)
            plt.close()
            n_plots += 1

    print(f"  Generated {n_plots} plots")


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate(mode, data_root, model_dir, output_dir, batch_size=32,
             plot_date=None, plot_hour=None, split="test"):
    data_root = Path(data_root)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir) / f"eval_sepconv_ensemble_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    SPLIT_CSV = {"train": "train_data.csv", "validation": "validation_data.csv",
                 "test": "test_data.csv"}
    csv_name = SPLIT_CSV[split]

    print("=" * 70)
    print(f"SepConv Ensemble Evaluation (Regression) — Mode: {mode}")
    print("=" * 70)
    print(f"  Split: {split}")
    print(f"  Post-processing thresholds (normalized): {THRESHOLDS_NORM}")

    # 1. History
    history_path = model_dir / f"history_sepconv_ensemble_{mode}.json"
    if history_path.is_file():
        print(f"\n1. Plotting training history")
        plot_training_history(history_path, output_dir)
    else:
        print(f"\n1. WARNING: History not found: {history_path}")

    # 2. Load ensemble
    print(f"\n2. Loading ensemble models")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    ensemble = SepConvEnsemble(model_dir, mode)
    print(f"  Total params: {ensemble.count_params():,}")

    # 3. Dataset
    ds_dir = data_root / "datasets" / mode / split
    if not ds_dir.exists():
        raise FileNotFoundError(f"{split.capitalize()} dataset not found: {ds_dir}")
    print(f"\n3. Loading {split} dataset from {ds_dir}")
    eval_ds = tf.data.Dataset.load(str(ds_dir))
    eval_ds = eval_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 4. Evaluate
    print(f"\n4. Evaluating on {split} set (continuous → thresholded classification)...")
    results = evaluate_radar(ensemble, eval_ds, output_dir)

    # 5. Visualize
    if plot_date and plot_hour is not None:
        print(f"\n5. Visualizing predictions for {plot_date} hour {plot_hour}:00 "
              f"(from {csv_name})...")
        try:
            plot_predictions_for_date_hour(ensemble, data_root, mode, output_dir,
                                           plot_date, plot_hour, csv_name=csv_name)
        except Exception as e:
            print(f"  Skipping visualization: {e}")
    else:
        print(f"\n5. Skipping visualization (use --date and --hour to enable)")

    # 6. Save
    results["mode"] = mode
    results["architecture"] = "sepconv_ensemble"
    results["output_type"] = "regression"
    results["split"] = split
    results["model_params"] = ensemble.count_params()
    results["post_processing"] = {
        "method": "threshold_classification",
        "thresholds": THRESHOLDS_NORM,
    }

    if history_path.is_file():
        with open(history_path) as f:
            results["training_wall_time"] = json.load(f).get("total_wall_time")

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n6. Results saved to {results_path}")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY — SepConv Ensemble (Regression + Post-processing)")
    print("=" * 70)
    print(f"  {'Lead time':<12} {'Accuracy':>10} {'Model':>10}")
    print(f"  {'-'*35}")
    for i, lt in enumerate(LEAD_LABELS):
        acc = results["per_leadtime"][lt]["accuracy"]
        print(f"  {lt:<12} {acc:>10.4f} {'Bm' + str(i+1):>10}")
    print(f"  {'Aggregate':<12} {results['aggregate']['accuracy']:>10.4f}")
    print(f"\n  Total params: {ensemble.count_params():,}")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SepConv ensemble (regression).")
    parser.add_argument("--mode", type=str,
                        default="mtg_opera_mtgmr_continuous",
                        choices=["mtg_opera_mtgmr_continuous"],
                        help="Continuous-target COALITION-4 mode whose "
                             "dataset this baseline was trained on.")
    parser.add_argument("--data_root", type=str, default="./our_data")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--output_dir", type=str, default="./evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Which dataset split to evaluate on (default: test)")
    parser.add_argument("--date", type=str, default=None,
                        help="Date for visualization, e.g. '2025-05-26'")
    parser.add_argument("--hour", type=int, default=None,
                        help="Reference hour for visualization (0-23), e.g. 5 → 5:00..5:45")
    args = parser.parse_args()
    evaluate(mode=args.mode, data_root=args.data_root, model_dir=args.model_dir,
             output_dir=args.output_dir, batch_size=args.batch_size,
             plot_date=args.date, plot_hour=args.hour, split=args.split)


if __name__ == "__main__":
    main()