"""
train_lightning_kd.py
=====================
Knowledge distillation for the lightning-occurrence track.

  Teacher:  mtg_lightning_opera_occurrence
              HR = LINET (density, current, occurrence) + MTG vis_06
              MR = OPERA reflectivity + rainfall_rate + MTG IR/WV
              Label = binary lightning occurrence at t+15/+30/+45
              Loss (its own training): WeightedFocalLoss(gamma=2)
  Student:  mtg_opera_occurrence   (this repo's new mode; create_datasets.py)
              HR = MTG vis_06 ONLY (LINET dropped)
              MR = same as teacher
              Label = same binary occurrence
              Loss (this script): KD_soft * T^2 + (1-alpha) * WeightedFocalLoss

KD LOSS - standard Hinton distillation adapted for binary sigmoid outputs:

    logit(p) = log(p / (1 - p))                             # inverse sigmoid
    soft_x   = sigmoid(logit(p_x) / T)                      # temperature-softened
    L_soft   = BCE(soft_teacher, soft_student) * T ** 2     # gradient-scale fix
    L_hard   = WeightedFocalLoss(y_gt, student)             # supervised anchor
    L_total  = alpha * L_soft + (1 - alpha) * L_hard

Defaults are the canonical Hinton values: alpha=0.7, temperature=4.0.
Both are CLI-configurable (see --kd_alpha / --kd_temperature) and
persisted in the history JSON so a re-run against a checked-in log
reproduces the exact training.

WHY WE REUSE THE TEACHER'S DATASET:
  Teacher's past_hr = concatenation of {density, current, occurrence, vis_06}
  in that channel order (dict-declaration order preserved through Python
  3.7+ dict merge). Student's past_hr = vis_06 only, which is the LAST
  channel of the teacher's HR. So `student_past_hr = teacher_past_hr[..., -1:]`
  is a zero-copy slice per batch. No separate `datasets/mtg_opera_occurrence/`
  needs to exist, and both models see the identically-shuffled batches so
  the distillation signal is on the same references the teacher saw.

Usage:
    python train_lightning_kd.py \\
        --data_root ./our_data --model_dir ./models \\
        --epochs 50 --batch_size 8

    # non-default KD hyperparameters
    python train_lightning_kd.py --kd_alpha 0.5 --kd_temperature 6.0

    # distil from the fine-tuned teacher (Swin head) instead of the base
    python train_lightning_kd.py --teacher_finetuned

Outputs:
    models/coalition_mtg_opera_occurrence_dbscan_kd.keras
    models/history_mtg_opera_occurrence_dbscan_kd.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# All the training infrastructure lives in train_models.py - we reuse
# build_coalition_model, WeightedFocalLoss, iou_metric etc. instead of
# duplicating them.
from train_models import (
    build_coalition_model,
    WeightedFocalLoss,
    WeightedFocalCategoricalCrossentropy,
    ConvBlock, ResBlock, GRUResBlock, ResGRU,
    iou_metric, true_pos, false_pos, false_neg,
    load_ones_fraction,
    load_dataset,
    configure_tf_runtime,
    WallTimeCallback,
)

from pipeline_config import SOURCE


# ============================================================================
# Constants (Hinton et al. defaults; overridable via CLI)
# ============================================================================
DEFAULT_KD_ALPHA = 0.7            # weight on the soft-teacher loss
DEFAULT_KD_TEMPERATURE = 4.0      # temperature for both teacher and student

# The LAST channel of the teacher's past_hr is MTG vis_06 (the student's
# only HR input); the first three are LINET density/current/occurrence.
# This constant makes the slicing intent explicit; changing it requires
# a matching change in create_datasets.HR_LIGHTNING_CONFIG.
STUDENT_HR_CHANNELS = 1

TEACHER_MODE = "mtg_lightning_opera_occurrence"
STUDENT_MODE = "mtg_opera_occurrence"

# Ensure numerical stability of logit(p) = log(p / (1-p)) when p is at
# 0 or 1 (sigmoid outputs can hit exact boundaries under mixed_float16
# rounding). This is the same clip WeightedFocalLoss uses.
_EPS = 1e-7


# ============================================================================
# KD loss (numerically-stable, uses float32 throughout)
# ============================================================================
def _inverse_sigmoid(p: tf.Tensor) -> tf.Tensor:
    """logit(p) = log(p / (1 - p)), with an eps clip so values at exact 0
    or 1 don't produce -inf / +inf."""
    p = tf.clip_by_value(p, _EPS, 1.0 - _EPS)
    return tf.math.log(p) - tf.math.log1p(-p)


def _soft_bce(soft_teacher: tf.Tensor, soft_student: tf.Tensor) -> tf.Tensor:
    """Element-wise BCE with the temperature-softened teacher as target and
    softened student as prediction. Averages across the whole batch tensor.
    Matches WeightedFocalLoss's `tf.reduce_mean` reduction so the two loss
    terms are on the same scale before alpha weighting."""
    soft_teacher = tf.clip_by_value(soft_teacher, _EPS, 1.0 - _EPS)
    soft_student = tf.clip_by_value(soft_student, _EPS, 1.0 - _EPS)
    bce = -soft_teacher * tf.math.log(soft_student) \
          - (1.0 - soft_teacher) * tf.math.log(1.0 - soft_student)
    return tf.reduce_mean(bce)


def kd_loss(
    teacher_probs: tf.Tensor, student_probs: tf.Tensor,
    y_true: tf.Tensor,
    alpha: float, temperature: float,
    hard_loss_fn: tf.keras.losses.Loss,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute (total_loss, soft_component, hard_component). All three
    reduced to a scalar so the training loop can log each independently."""
    teacher_probs = tf.cast(teacher_probs, tf.float32)
    student_probs = tf.cast(student_probs, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    T = tf.constant(float(temperature), dtype=tf.float32)

    soft_teacher = tf.sigmoid(_inverse_sigmoid(teacher_probs) / T)
    soft_student = tf.sigmoid(_inverse_sigmoid(student_probs) / T)

    L_soft = _soft_bce(soft_teacher, soft_student) * (T * T)
    L_hard = hard_loss_fn(y_true, student_probs)

    L_total = alpha * L_soft + (1.0 - alpha) * L_hard
    return L_total, L_soft, L_hard


# ============================================================================
# KD training wrapper (subclasses tf.keras.Model so we get fit()/callbacks)
# ============================================================================
class KDModel(tf.keras.Model):
    """Wraps (student, frozen teacher) with a custom train_step / test_step
    that runs both models per batch and applies the KD loss ONLY to the
    student. The teacher is stored under a name Keras does not track so
    its weights aren't included in .save() / .get_weights()."""

    def __init__(
        self, student: tf.keras.Model, teacher: tf.keras.Model,
        alpha: float, temperature: float,
        hard_loss_fn: tf.keras.losses.Loss,
        student_hr_channels: int = STUDENT_HR_CHANNELS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.student = student
        # Underscore prefix + object.__setattr__ so tf.keras.Model doesn't
        # auto-register teacher as a sub-layer (which would save its
        # weights alongside the student's).
        object.__setattr__(self, "_teacher", teacher)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.hard_loss_fn = hard_loss_fn
        self.student_hr_channels = int(student_hr_channels)
        # Metric trackers for the three loss components.
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.soft_tracker = tf.keras.metrics.Mean(name="l_soft")
        self.hard_tracker = tf.keras.metrics.Mean(name="l_hard")

    @property
    def teacher(self) -> tf.keras.Model:
        return self._teacher

    def _student_inputs(self, inputs: dict) -> dict:
        """Derive the student's inputs from a teacher-format batch. Only
        past_hr changes (slice the LAST N channels = vis_06); past_mr
        passes through unchanged since the mode configs share it."""
        student_hr = inputs["past_hr"][..., -self.student_hr_channels:]
        return {
            "past_hr": student_hr,
            "past_mr": inputs["past_mr"],
        }

    def call(self, inputs, training=False):
        """Inference-time forward = the student's forward. Lets us call
        .predict(...) on this wrapper if we ever want to."""
        return self.student(self._student_inputs(inputs), training=training)

    def train_step(self, data):
        # Mixed-precision handling: when the global policy is
        # mixed_float16, Keras auto-wraps model.compile's optimizer in a
        # LossScaleOptimizer that requires get_scaled_loss +
        # get_unscaled_gradients around the tape/apply so gradients don't
        # underflow to zero in float16. Model.fit's default train_step
        # does this transparently; a custom train_step has to opt in
        # explicitly, otherwise TF emits the "You forgot to call ..."
        # warning and quality quietly degrades.
        is_lso = isinstance(
            self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer,
        )
        inputs, y_true = data
        student_inputs = self._student_inputs(inputs)
        teacher_probs = self._teacher(inputs, training=False)
        with tf.GradientTape() as tape:
            student_probs = self.student(student_inputs, training=True)
            total, soft, hard = kd_loss(
                teacher_probs, student_probs, y_true,
                self.alpha, self.temperature, self.hard_loss_fn,
            )
            loss_for_grad = (self.optimizer.get_scaled_loss(total)
                             if is_lso else total)
        raw_grads = tape.gradient(
            loss_for_grad, self.student.trainable_variables,
        )
        grads = (self.optimizer.get_unscaled_gradients(raw_grads)
                 if is_lso else raw_grads)
        self.optimizer.apply_gradients(
            zip(grads, self.student.trainable_variables)
        )
        self.loss_tracker.update_state(total)
        self.soft_tracker.update_state(soft)
        self.hard_tracker.update_state(hard)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, y_true = data
        student_inputs = self._student_inputs(inputs)
        teacher_probs = self._teacher(inputs, training=False)
        student_probs = self.student(student_inputs, training=False)
        total, soft, hard = kd_loss(
            teacher_probs, student_probs, y_true,
            self.alpha, self.temperature, self.hard_loss_fn,
        )
        self.loss_tracker.update_state(total)
        self.soft_tracker.update_state(soft)
        self.hard_tracker.update_state(hard)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.soft_tracker, self.hard_tracker]


# ============================================================================
# Teacher loading (mirrors train_models.build_finetune_model's custom_objects)
# ============================================================================
def load_teacher(model_dir: Path, source: str,
                 finetuned: bool = False) -> tf.keras.Model:
    """Load the frozen teacher. Uses the same custom_objects list as
    train_models.build_finetune_model so ResBlock / ResGRU / etc. round-
    trip cleanly."""
    from train_models import build_run_tag
    suffix = "_finetuned" if finetuned else ""
    path = model_dir / (
        f"coalition_{build_run_tag(TEACHER_MODE, source)}{suffix}.keras"
    )
    if not path.is_file():
        raise SystemExit(
            f"Teacher checkpoint not found: {path.name}. Train the base "
            f"teacher first (see train_models.py --mode {TEACHER_MODE})."
        )
    print(f"Loading teacher: {path.name}")
    teacher = tf.keras.models.load_model(
        str(path),
        custom_objects={
            "ConvBlock":         ConvBlock,
            "ResBlock":          ResBlock,
            "GRUResBlock":       GRUResBlock,
            "ResGRU":            ResGRU,
            "WeightedFocalLoss": WeightedFocalLoss,
            "WeightedFocalCategoricalCrossentropy": WeightedFocalCategoricalCrossentropy,
            "iou_metric":        iou_metric,
            "true_pos":          true_pos,
            "false_pos":         false_pos,
            "false_neg":         false_neg,
        },
        compile=False,
    )
    teacher.trainable = False
    print(f"  Teacher params: {teacher.count_params():,}  (frozen)")
    return teacher


# ============================================================================
# Student input-shape derivation
# ============================================================================
def student_input_shapes(teacher_input_shapes: dict,
                          student_hr_channels: int) -> dict:
    """Derive input_shapes for the student mode from the teacher's, by
    dropping all but the last `student_hr_channels` HR channels. past_mr
    passes through unchanged (shared between the two configs)."""
    out = dict(teacher_input_shapes)
    hr_shape = list(out["past_hr"])            # [T, H, W, C_teacher]
    hr_shape[-1] = student_hr_channels          # → [T, H, W, C_student]
    out["past_hr"] = hr_shape
    return out


# ============================================================================
# Main training entry point
# ============================================================================
def train_kd(
    data_root: Path, model_dir: Path,
    epochs: int, batch_size: int,
    alpha: float, temperature: float,
    source: str = "dbscan",
    teacher_finetuned: bool = False,
    learning_rate: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
    shuffle_buffer: int = 256,
    mixed_precision: bool = True,
):
    """Full KD training run. Loads the teacher's dataset, wraps
    (student, teacher) in KDModel, fits, saves the student only."""
    tf.keras.utils.set_random_seed(seed)

    data_root = Path(data_root)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # We consume the TEACHER's dataset, not a separate student one - see
    # the module docstring for why this is correct given the HR channel
    # ordering.
    from train_models import build_run_tag
    dataset_dir = data_root / "datasets" / build_run_tag(TEACHER_MODE, source)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise SystemExit(
                f"Teacher dataset split missing: {d}\n"
                f"Run create_datasets.py --mode {TEACHER_MODE} first."
            )

    # Grab input_shapes from the teacher metadata; derive the student's.
    with open(train_dir / "metadata.json") as f:
        meta = json.load(f)
    teacher_input_shapes = meta["input_shapes"]
    label_type = meta.get("label_type", "lightning")
    past_timesteps = next(iter(teacher_input_shapes.values()))[0]
    label_shape = meta.get("label_shape", [3, 256, 256, 1])
    future_timesteps = label_shape[0]

    student_shapes = student_input_shapes(teacher_input_shapes,
                                           STUDENT_HR_CHANNELS)
    ones_fraction = load_ones_fraction(data_root, source)

    print("=" * 70)
    print("Knowledge distillation (lightning occurrence)")
    print("=" * 70)
    print(f"  Teacher mode:  {TEACHER_MODE}"
          f"{' (finetuned)' if teacher_finetuned else ''}")
    print(f"  Student mode:  {STUDENT_MODE}")
    print(f"  Source:        {source}")
    print(f"  Dataset:       {dataset_dir}")
    print(f"  Batch size:    {batch_size}   Epochs: {epochs}   LR: {learning_rate}")
    print(f"  KD alpha:      {alpha}")
    print(f"  KD temperature:{temperature}")
    print(f"  Teacher past_hr channels: {teacher_input_shapes['past_hr'][-1]}"
          f"   (dropping first {teacher_input_shapes['past_hr'][-1] - STUDENT_HR_CHANNELS}"
          f" LINET channels for the student)")
    print(f"  Student past_hr channels: {STUDENT_HR_CHANNELS}")
    print(f"  ones_fraction (focal-loss prior): {ones_fraction}")
    print()

    print("Configuring TF runtime ...")
    configure_tf_runtime(use_mixed_precision=mixed_precision)

    print("Loading teacher ...")
    teacher = load_teacher(model_dir, source, finetuned=teacher_finetuned)

    print("\nBuilding student ...")
    student = build_coalition_model(
        input_shapes=student_shapes,
        label_type=label_type,
        past_timesteps=past_timesteps,
        future_timesteps=future_timesteps,
        ones_fraction=ones_fraction,
        # We do NOT compile the student here; KDModel handles the loss.
    )
    print(f"  Student params: {student.count_params():,}")

    print("\nLoading datasets ...")
    train_ds = load_dataset(train_dir, batch_size, shuffle=True,
                             shuffle_buffer=shuffle_buffer)
    val_ds = load_dataset(val_dir, batch_size, shuffle=False)
    print("  Datasets loaded")

    hard_loss_fn = WeightedFocalLoss(ones_fraction=ones_fraction, gamma=2.0)
    kd_model = KDModel(
        student=student, teacher=teacher,
        alpha=alpha, temperature=temperature,
        hard_loss_fn=hard_loss_fn,
        student_hr_channels=STUDENT_HR_CHANNELS,
    )
    kd_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience,
        restore_best_weights=True, verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(3, patience // 2),
        min_lr=1e-8, min_delta=1e-4, verbose=1,
    )
    wall_timer = WallTimeCallback()

    print("\n" + "-" * 70)
    print("Starting KD training")
    print("-" * 70)
    t0 = time.time()
    history = kd_model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=[early_stop, reduce_lr, wall_timer],
        verbose=1,
    )
    total_time = time.time() - t0

    student_run_tag = build_run_tag(STUDENT_MODE, source)
    student_path = model_dir / f"coalition_{student_run_tag}_kd.keras"
    student.save(str(student_path))
    print(f"\nSaved student: {student_path}")

    hist_path = model_dir / f"history_{student_run_tag}_kd.json"
    hist_data = {
        "teacher_mode": TEACHER_MODE,
        "student_mode": STUDENT_MODE,
        "source": source,
        "teacher_finetuned": teacher_finetuned,
        "kd_alpha": alpha,
        "kd_temperature": temperature,
        "epochs_configured": epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "seed": seed,
        "student_params": int(student.count_params()),
        "teacher_params": int(teacher.count_params()),
        "student_hr_channels": STUDENT_HR_CHANNELS,
        "teacher_hr_channels": int(teacher_input_shapes["past_hr"][-1]),
        "wall_time_sec": total_time,
        "epoch_wall_times": wall_timer.epoch_times,
        "history": {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        },
    }
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"Saved history: {hist_path}")
    print(f"\nTotal wall time: {total_time:.1f}s "
          f"({total_time / 60.0:.1f} min)")


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Knowledge distillation for lightning occurrence. Teacher is the "
            "mtg_lightning_opera_occurrence base or finetuned model; student "
            "is a fresh mtg_opera_occurrence trained to mimic the teacher "
            "from satellite-only inputs (no LINET at student inference time)."
        ),
    )
    p.add_argument("--data_root", type=str, default="./our_data")
    p.add_argument("--model_dir", type=str, default="./models")
    p.add_argument("--teacher_finetuned", action="store_true",
                   help="Distil from coalition_..._finetuned.keras instead "
                        "of the base teacher.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Both teacher and student forward per step, so 8 is "
                        "safer than 16 GPU-memory-wise. Halve if OOM.")
    p.add_argument("--kd_alpha", type=float, default=DEFAULT_KD_ALPHA,
                   help=f"Weight on the soft-teacher loss. Default "
                        f"{DEFAULT_KD_ALPHA} (Hinton canonical). Higher -> "
                        f"more mimicry of teacher, less GT anchoring.")
    p.add_argument("--kd_temperature", type=float,
                   default=DEFAULT_KD_TEMPERATURE,
                   help=f"Temperature for softening both teacher and student. "
                        f"Default {DEFAULT_KD_TEMPERATURE} (Hinton canonical). "
                        f"Higher -> softer distribution, more subtle knowledge "
                        f"transfer.")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10,
                   help="EarlyStopping patience on val_loss.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle_buffer", type=int, default=256)
    p.add_argument("--no_mixed_precision", action="store_true",
                   help="Disable mixed_float16 (use float32 everywhere). "
                        "Slower but sometimes helpful for debugging.")
    args = p.parse_args()

    if not (0.0 <= args.kd_alpha <= 1.0):
        raise SystemExit(f"--kd_alpha must be in [0, 1]; got {args.kd_alpha}")
    if args.kd_temperature <= 0:
        raise SystemExit(
            f"--kd_temperature must be > 0; got {args.kd_temperature}"
        )

    train_kd(
        data_root=Path(args.data_root), model_dir=Path(args.model_dir),
        epochs=args.epochs, batch_size=args.batch_size,
        alpha=args.kd_alpha, temperature=args.kd_temperature,
        source=SOURCE, teacher_finetuned=args.teacher_finetuned,
        learning_rate=args.learning_rate, patience=args.patience,
        seed=args.seed, shuffle_buffer=args.shuffle_buffer,
        mixed_precision=not args.no_mixed_precision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
