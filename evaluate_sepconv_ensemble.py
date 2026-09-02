"""
evaluate_sepconv_ensemble.py — SepConv-ens evaluation
========================================================================
Scores the composed SepConv-ens forecast against the same 5 rainfall
classes RECONVECT is scored on.

The prediction path is `sepconv_predict`, not a second copy of it:

    compose (log_zscore)  ->  10**(z*std + mean)  ->  bin at mm/h edges

Every part of that matters. `compose` applies the paper's scheme, which
is worth 3-4x in CSI over running one base model repeatedly. The
denormalisation uses the window's OWN statistics, named on the command
line, because several windows coexist on disk. And the binning happens in
mm/h against `RAINFALL_CLASS_EDGES` — the same edges the RECONVECT label
uses — so the two models' outputs are classified identically and the
comparison cannot be an artefact of the thresholds.

Ground truth is put through the same denormalise-then-bin path as the
prediction, so any error in the statistics moves both sides together
rather than showing up as skill.

Horizons are t+1..t+4 (15/30/45/60 min). Those are the steps the
composition builds from observations alone — nothing here is
autoregressive.

Usage:
    python evaluate_sepconv_ensemble.py --period w44
    python evaluate_sepconv_ensemble.py --period w44 --plot_samples 6

Outputs: ./evaluation/eval_sepconv_<run_tag>/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from create_datasets import (
    RAINFALL_CLASS_EDGES,
    get_mode_config,
    get_output_signature,
    load_tfrecord_dataset,
    load_tfrecord_with_key,
    require_key_matches,
)
from periods import data_tag
from pipeline_config import (
    SOURCE,
    resolve_data_root,
    resolve_datasets_root,
    resolve_model_dir,
)
from sepconv_compose import STEP_MINUTES
from sepconv_ensemble_training import SEPCONV_MODE
from sepconv_predict import load_base_models, predict_classes, to_mmh
from train_models import build_run_tag


# ============================================================================
# Constants
# ============================================================================

MAX_STEP = 4          # t+1..t+4; beyond this the composition goes AR
CLASS_NAMES = ["R<10", "10≤R<20", "20≤R<30", "30≤R<40", "R≥40"]
N_CLASSES = len(CLASS_NAMES)

LEAD_MINUTES = [k * STEP_MINUTES for k in range(1, MAX_STEP + 1)]
LEAD_LABELS = [f"t+{m}" for m in LEAD_MINUTES]


# ============================================================================
# Prediction
# ============================================================================

def frames_from_batch(inputs):
    """Split a dataset batch into the frame list `compose` expects.

    `past_hr` arrives as (N, T, H, W, 1) with T = past+1 frames oldest
    first. `compose` wants them as a list of (N, H, W), one per offset.
    """
    past = np.asarray(inputs["past_hr"])
    if past.ndim != 5:
        raise ValueError(
            f"expected past_hr with shape (N, T, H, W, C), got {past.shape}")
    return [past[:, t, :, :, 0] for t in range(past.shape[1])]


def labels_to_classes(labels, stats_period, data_root):
    """Ground truth: log_zscore -> mm/h -> class, exactly as predictions."""
    arr = np.asarray(labels)                       # (N, F, H, W, 1)
    out = {}
    for step in range(1, min(MAX_STEP, arr.shape[1]) + 1):
        z = arr[:, step - 1, :, :, 0]
        mmh = to_mmh(z, stats_period, data_root=data_root)
        cls = np.zeros(mmh.shape, dtype=np.int32)
        for edge in RAINFALL_CLASS_EDGES:
            cls += (mmh >= edge).astype(np.int32)
        out[step] = cls
    return out


# ============================================================================
# Metrics
# ============================================================================

def evaluate_radar(models, test_ds, stats_period, data_root, output_dir,
                   batch_size=8):
    """Composed forecast vs ground truth, per lead, as confusion matrices."""
    print("  Composing forecasts and binning to classes...")

    conf = {s: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
            for s in range(1, MAX_STEP + 1)}
    n_samples = 0

    for inputs, labels in test_ds:
        frames = frames_from_batch(inputs)
        pred_cls, _ = predict_classes(
            models, frames, stats_period, max_step=MAX_STEP,
            data_root=data_root, batch_size=batch_size, batched=True)
        true_cls = labels_to_classes(labels, stats_period, data_root)
        n_samples += frames[0].shape[0]

        for step in range(1, MAX_STEP + 1):
            if step not in true_cls:
                continue
            yt = true_cls[step].ravel()
            yp = np.asarray(pred_cls[step]).ravel()
            # One pass with bincount beats 25 boolean reductions per step.
            idx = yt * N_CLASSES + yp
            counts = np.bincount(idx, minlength=N_CLASSES * N_CLASSES)
            conf[step] += counts.reshape(N_CLASSES, N_CLASSES)

    print(f"  Scored {n_samples:,} sample(s)")

    results = {"per_leadtime": {}, "aggregate": {},
               "n_samples": int(n_samples),
               "class_edges_mmh": list(RAINFALL_CLASS_EDGES),
               "max_step": MAX_STEP, "autoregressive": False}
    agg = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    for step in range(1, MAX_STEP + 1):
        cm = conf[step]
        agg += cm
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
            # CSI is the metric the paper reports; it ignores true
            # negatives, which on a 99.8% dry field is the difference
            # between a meaningful number and a flattering one.
            csi = tp / (tp + fp + fn + 1e-10)
            per_class[CLASS_NAMES[c]] = {
                "precision": float(precision), "recall": float(recall),
                "f1": float(f1), "csi": float(csi),
                "support": int(cm[c, :].sum()),
            }

        label = LEAD_LABELS[step - 1]
        results["per_leadtime"][label] = {
            "lead_minutes": LEAD_MINUTES[step - 1],
            "accuracy": float(accuracy),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }
        wet = np.mean([per_class[c]["csi"] for c in CLASS_NAMES[1:]])
        print(f"    {label:>6} ({LEAD_MINUTES[step-1]:>3} min): "
              f"accuracy={accuracy:.4f}  mean wet-class CSI={wet:.4f}")

    total = agg.sum()
    results["aggregate"] = {
        "accuracy": float(agg.trace() / (total + 1e-10)),
        "confusion_matrix": agg.tolist(),
    }

    plot_metrics(results, agg, output_dir)
    return results


# ============================================================================
# Plots
# ============================================================================

def plot_metrics(results, agg, output_dir):
    """Accuracy and wet-class CSI per lead, plus the aggregate matrix."""
    accs = [results["per_leadtime"][lt]["accuracy"] for lt in LEAD_LABELS]
    csis = [np.mean([results["per_leadtime"][lt]["per_class"][c]["csi"]
                     for c in CLASS_NAMES[1:]]) for lt in LEAD_LABELS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(LEAD_MINUTES, accs, 'o-', lw=2, ms=8, color='#1f77b4',
                 label='Accuracy (all classes)')
    axes[0].plot(LEAD_MINUTES, csis, 's-', lw=2, ms=8, color='#d62728',
                 label='Mean CSI (wet classes)')
    axes[0].set_xlabel("Lead time (min)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("SepConv-ens skill by lead time")
    axes[0].set_xticks(LEAD_MINUTES)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Row-normalised: class 0 is ~99.8% of pixels, so raw counts show one
    # dark cell and nothing else.
    with np.errstate(invalid='ignore', divide='ignore'):
        norm = agg / agg.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)
    im = axes[1].imshow(norm, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=45, ha='right')
    axes[1].set_yticks(range(N_CLASSES), CLASS_NAMES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Observed")
    axes[1].set_title("Aggregate confusion (row-normalised)")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            axes[1].text(j, i, f"{norm[i, j]:.2f}", ha='center', va='center',
                         fontsize=8,
                         color='white' if norm[i, j] > 0.5 else '#333')
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    fig.savefig(output_dir / "metrics_per_leadtime.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'metrics_per_leadtime.png'}")


def plot_training_history(history_path, output_dir):
    """Loss curves for each base model, from the training history JSON."""
    with open(history_path) as f:
        hist = json.load(f)

    base = hist.get("base_models", {})
    if not base:
        print("  History has no per-base-model curves; skipping.")
        return

    fig, axes = plt.subplots(1, len(base), figsize=(5 * len(base), 4),
                            squeeze=False)
    for ax, (name, blk) in zip(axes[0], sorted(base.items())):
        h = blk.get("history", {})
        if "loss" in h:
            ax.plot(h["loss"], label="train")
        if "val_loss" in h:
            ax.plot(h["val_loss"], label="val")
        ax.set_title(f"{name} ({blk.get('lead_name', '')})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted MSE (log_zscore)")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'training_curves.png'}")


def plot_samples(models, test_ds, stats_period, data_root, output_dir,
                 n_samples, batch_size=8):
    """Observed vs predicted class, plus the predicted rain rate.

    Driven by the test dataset rather than by re-reading patches from a
    split CSV: the sample the metrics scored is the sample plotted, and
    the figure follows whatever window the dataset was built with instead
    of assuming one.
    """
    cmap = plt.get_cmap('viridis', N_CLASSES)
    written = 0

    for inputs, labels in test_ds:
        frames = frames_from_batch(inputs)
        pred_cls, pred_mmh = predict_classes(
            models, frames, stats_period, max_step=MAX_STEP,
            data_root=data_root, batch_size=batch_size, batched=True)
        true_cls = labels_to_classes(labels, stats_period, data_root)

        for n in range(frames[0].shape[0]):
            if written >= n_samples:
                return
            fig, axes = plt.subplots(3, MAX_STEP,
                                     figsize=(3.2 * MAX_STEP, 9.6),
                                     squeeze=False)
            for s in range(1, MAX_STEP + 1):
                col = s - 1
                axes[0][col].imshow(true_cls[s][n], cmap=cmap,
                                    vmin=0, vmax=N_CLASSES - 1)
                axes[0][col].set_title(f"{LEAD_LABELS[col]} observed",
                                       fontsize=10)
                axes[1][col].imshow(np.asarray(pred_cls[s])[n], cmap=cmap,
                                    vmin=0, vmax=N_CLASSES - 1)
                axes[1][col].set_title("predicted class", fontsize=10)
                im = axes[2][col].imshow(np.asarray(pred_mmh[s])[n],
                                         cmap='turbo', vmin=0, vmax=50)
                axes[2][col].set_title("predicted mm/h", fontsize=10)
                for r in range(3):
                    axes[r][col].set_xticks([])
                    axes[r][col].set_yticks([])
            fig.colorbar(im, ax=axes[2], fraction=0.025, label="mm/h")
            fig.suptitle(f"SepConv-ens sample {written}", fontweight='bold')
            fig.savefig(output_dir / f"sample_{written:03d}.png", dpi=140,
                        bbox_inches='tight')
            plt.close(fig)
            written += 1

    print(f"  Wrote {written} sample figure(s)")


# ============================================================================
# Main evaluation
# ============================================================================

def load_verification_keys(path):
    """Read the frozen leakage-free key set as {(date, ref, patch)}.

    Written by verification_keys.py. Restricting the score to it is what
    makes the baseline comparison legitimate: the two windows are split
    independently, so each model's own test split is a different
    population from the other's AND overlaps the other's training data.
    """
    blob = json.loads(Path(path).read_text())
    keys = blob.get("keys")
    if not keys:
        raise ValueError(
            f"{path} holds no 'keys' array. Rebuild it with:\n"
            f"    python verification_keys.py --write "
            f"--reconvect_tag <tag> --sepconv_tag <tag>")
    return {(str(d), str(r), int(p)) for d, r, p in keys}


def filter_to_keys(ds, keys):
    """Keep only samples whose (date, reference_utc, patch) is frozen.

    Filtering happens in Python rather than inside the tf.data graph:
    the key set is a few thousand tuples, and a tf.lookup table keyed on
    a string pair plus an int is far more machinery than a per-sample
    set membership test needs.
    """
    for inputs, label, date, ref, patch in ds:
        k = (date.numpy().decode(), ref.numpy().decode(), int(patch.numpy()))
        if k in keys:
            yield inputs, label


def evaluate(mode, data_root, model_dir, output_dir, batch_size=8,
             split="test", period=None, n_plot_samples=0,
             verification_keys=None, datasets_root=None,
             weights="best"):
    data_root = resolve_data_root(data_root)
    datasets_root = resolve_datasets_root(data_root, datasets_root)
    model_dir = Path(model_dir)
    # One tag resolves the weights, the history, the dataset and the
    # normalization statistics together, so an evaluation cannot read one
    # window's models against another window's test split.
    run_tag = build_run_tag(mode, SOURCE, period)
    output_dir = Path(output_dir) / f"eval_sepconv_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SepConv-ens evaluation — {run_tag}")
    print("=" * 70)
    print(f"  Split          : {split}")
    print(f"  Horizons       : {', '.join(LEAD_LABELS)} "
          f"({', '.join(str(m) + ' min' for m in LEAD_MINUTES)})")
    print(f"  Class edges    : {list(RAINFALL_CLASS_EDGES)} mm/h")
    print(f"  Statistics     : {data_tag(SOURCE, period)}")
    print(f"  Autoregressive : no (t+1..t+4 read observations only)")

    print("\n1. Training history")
    history_path = model_dir / f"history_sepconv_{run_tag}.json"
    if history_path.is_file():
        plot_training_history(history_path, output_dir)
    else:
        print(f"  WARNING: not found: {history_path}")

    print("\n2. Base models")
    print(f"  Weights        : {weights}"
          f"{'  (last epoch run, not the best)' if weights == 'latest' else ''}")
    models = load_base_models(model_dir, run_tag, weights=weights)

    print("\n3. Dataset")
    ds_dir = datasets_root / run_tag / split
    if not ds_dir.exists():
        raise FileNotFoundError(
            f"{split} dataset not found: {ds_dir}\n"
            f"Build it with: python create_datasets.py --mode {mode} "
            f"--period {period}")
    print(f"  {ds_dir}")
    if verification_keys:
        keys = load_verification_keys(verification_keys)
        print(f"  Scoring restricted to the frozen verification set: "
              f"{len(keys):,} key(s)")
        print(f"    {verification_keys}")
        kept, dropped = require_key_matches(ds_dir, keys)
        print(f"  verification filter: keeps {kept:,}, drops {dropped:,}")
        keyed = load_tfrecord_with_key(ds_dir, get_mode_config(mode))
        input_specs, label_spec = get_output_signature(get_mode_config(mode))
        test_ds = tf.data.Dataset.from_generator(
            lambda: filter_to_keys(keyed, keys),
            output_signature=(input_specs, label_spec),
        )
    else:
        print("  Scoring the FULL split. For the baseline-vs-ablation "
              "comparison pass --verification_keys: the two windows are "
              "split independently, so this split is a different "
              "population from the other model's and overlaps its "
              "training data.")
        test_ds = load_tfrecord_dataset(ds_dir, get_mode_config(mode))
    test_ds = test_ds.batch(batch_size)

    print("\n4. Scoring")
    results = evaluate_radar(models, test_ds, period, data_root, output_dir,
                             batch_size=batch_size)

    if n_plot_samples:
        print(f"\n5. Sample figures ({n_plot_samples})")
        plot_samples(models, test_ds, period, data_root, output_dir,
                     n_plot_samples, batch_size=batch_size)

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")

    print("\n" + "=" * 70)
    print(f"  {'lead':>8} {'accuracy':>10} {'wet CSI':>10}")
    for lt in LEAD_LABELS:
        blk = results["per_leadtime"][lt]
        wet = np.mean([blk["per_class"][c]["csi"] for c in CLASS_NAMES[1:]])
        print(f"  {lt:>8} {blk['accuracy']:>10.4f} {wet:>10.4f}")
    print(f"  {'aggregate':>8} {results['aggregate']['accuracy']:>10.4f}")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the composed SepConv-ens forecast.")
    # Must match sepconv_ensemble_training.SEPCONV_MODE: the evaluation
    # loads that run's weights and its test split, so a different mode
    # here would score one model against another's data.
    parser.add_argument("--mode", type=str, default=SEPCONV_MODE,
                        choices=[SEPCONV_MODE],
                        help=f"Mode the baseline was trained on "
                             f"(default: {SEPCONV_MODE}).")
    parser.add_argument("--period", type=str, default=None, metavar="LABEL",
                        help="Window tag identifying which trained ensemble, "
                             "dataset and normalization statistics to use, "
                             "e.g. w44. Omit only for an unsuffixed "
                             "whole-archive run.")
    parser.add_argument("--data_root", type=str, default=None,
                        metavar="PATH",
                        help="Root holding patches/, split CSVs and "
                             "statistics (default: the our_data/ beside "
                             "this script, or $COALITION4_DATA_ROOT).")
    parser.add_argument("--datasets_root", type=str, default=None,
                        metavar="PATH",
                        help="Root holding the built TFRecord datasets "
                             "(default: <data_root>/datasets, or "
                             "$COALITION4_DATASETS_ROOT).")
    parser.add_argument("--model_dir", type=str, default=str(resolve_model_dir()))
    parser.add_argument("--output_dir", type=str, default="./evaluation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--verification_keys", type=str, default=None,
                        metavar="PATH",
                        help="Frozen key set from verification_keys.py "
                             "--write. Restricts scoring to the "
                             "leakage-free intersection of the two test "
                             "splits. REQUIRED for a legitimate "
                             "baseline-vs-ablation comparison; without it "
                             "each model is scored on its own split, which "
                             "is a different population and overlaps the "
                             "other model's training data.")
    parser.add_argument("--weights", type=str, default="best",
                        choices=["best", "latest"],
                        help="Which of the two saved states to score. "
                             "'best' is the final save, whose weights early "
                             "stopping restored to the best epoch. 'latest' "
                             "is the rolling per-epoch checkpoint under "
                             "checkpoints/ - the last epoch actually run.")
    parser.add_argument("--plot_samples", type=int, default=0, metavar="N",
                        help="Render N test samples as observed vs predicted "
                             "class plus predicted mm/h.")
    args = parser.parse_args()

    evaluate(mode=args.mode, data_root=args.data_root,
             datasets_root=args.datasets_root,
             model_dir=args.model_dir, output_dir=args.output_dir,
             batch_size=args.batch_size, split=args.split,
             period=args.period, n_plot_samples=args.plot_samples,
             verification_keys=args.verification_keys,
             weights=args.weights)


if __name__ == "__main__":
    main()
