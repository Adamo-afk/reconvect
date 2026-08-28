"""
sepconv_predict.py — SepConv-ens inference, denormalization and binning
=======================================================================
The bridge that makes the baseline comparable to RECONVECT. SepConv is a
regressor in `log_zscore` space; RECONVECT emits a 5-class field. This
module runs the composition, returns to physical units, and bins — so
that from the verification code's point of view both models produce the
same thing.

Order of operations is not negotiable
-------------------------------------
    compose (log_zscore)  ->  10**(z*std + mean)  ->  bin in mm/h

Thresholding in z-space would be a bug even though it "works": z is a
coordinate whose meaning depends on the mean/std that produced it, so a
threshold chosen there is not a rain rate and cannot be compared across
models, calibrated against observations, or reported. Every threshold in
this project lives in mm/h. `bin_to_classes` therefore takes mm/h and
there is deliberately no z-space variant.

Autoregression stays in z-space
-------------------------------
The composition feeds predictions back as inputs, and the model consumes
`log_zscore`. So the rollout is done entirely in z and denormalised once
at the end — never round-tripped per step, which would accumulate
floating-point error through eight exponentiations and logs for no
reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline_config import SOURCE
from sepconv_compose import (
    BASE_LEADS,
    LEAD_MINUTES,
    MAX_STEP,
    REAL_FRAME_OFFSETS,
    compose,
)


# The period tag the baseline is trained and normalised under. It names a
# sequence WINDOW (past=4/future=8), not a date range. Every consumer must
# denormalise with these statistics: the invariant is that training and
# inversion agree, not that the baseline matches RECONVECT.
SEPCONV_STATS_PERIOD = "w48"


def load_base_models(model_dir, run_tag, leads=BASE_LEADS):
    """Load Bm1 / Bm3 / Bm5 for one run tag.

    Fails loudly on a missing member: a composition silently short of a
    base model would fall back on nothing and produce forecasts for the
    wrong lead times.
    """
    import tensorflow as tf
    from sepconv_ensemble_training import WeightedMSELogZ

    models = {}
    missing = []
    for lead in leads:
        path = Path(model_dir) / f"sepconv_{run_tag}_bm{lead}.keras"
        if not path.is_file():
            missing.append(str(path))
            continue
        models[lead] = tf.keras.models.load_model(
            str(path),
            custom_objects={"WeightedMSELogZ": WeightedMSELogZ},
            compile=False,
        )
    if missing:
        raise SystemExit(
            "Missing SepConv base model(s):\n  " + "\n  ".join(missing) +
            "\nTrain them with:\n"
            "    python sepconv_ensemble_training.py --period <tag>"
        )
    return models


def make_predict_fn(models, batch_size: int = 8):
    """Adapter from `sepconv_compose.compose` to the loaded Keras models.

    `compose` hands over (lead, four frames oldest-first); the models take
    four named inputs. Frames stay in log_zscore throughout.
    """
    def predict_fn(lead, frames):
        model = models[lead]
        batched = {}
        for i, f in enumerate(frames):
            arr = np.asarray(f, dtype=np.float32)
            if arr.ndim == 2:                      # (H, W) -> (1, H, W, 1)
                arr = arr[None, ..., None]
            elif arr.ndim == 3:                    # (H, W, 1) -> (1, H, W, 1)
                arr = arr[None, ...]
            batched[f"past_t{i}"] = arr
        out = model.predict(batched, batch_size=batch_size, verbose=0)
        return np.asarray(out)[0, ..., 0]          # back to (H, W)

    return predict_fn


def predict_sequence(models, past_frames, max_step: int = MAX_STEP,
                     batch_size: int = 8) -> dict[int, np.ndarray]:
    """Compose t+1..t+max_step. Output stays in log_zscore space."""
    return compose(make_predict_fn(models, batch_size), past_frames,
                   max_step=max_step)


def to_mmh(z, data_root="./our_data", source=SOURCE,
           stats_period=SEPCONV_STATS_PERIOD):
    """Denormalise log_zscore -> mm/h.

    `stats_period` MUST be the period the models were trained under —
    the same value `build_sepconv_loss` received. It defaults to
    `SEPCONV_STATS_PERIOD` for that reason.

    Getting this wrong is the silent-corruption case: nothing raises, the
    numbers stay plausible, and the recovered rain rates are biased
    monotonically with intensity (about -6% at 30 mm/h for a 1%
    difference in std). Calibration then absorbs the bias into its
    thresholds, so it does not even surface as bad calibration — it
    surfaces as a skill difference that is not real.
    """
    from create_datasets import logz_to_mmh, set_normalization_stats_path
    from periods import normalization_stats_name

    stats_file = normalization_stats_name(source, stats_period)
    path = Path(data_root) / stats_file
    if not path.is_file():
        raise SystemExit(
            f"Normalization statistics not found: {path}\n"
            f"The baseline is normalised per training split. Build them "
            f"with:\n    python compute_normalization_stats.py "
            f"--period {stats_period}"
        )
    set_normalization_stats_path(path)
    return logz_to_mmh(z)


def bin_to_classes(mmh, edges=None) -> np.ndarray:
    """Bin a physical rain-rate field into the shared 5 classes.

    Takes mm/h, never z. The edges are the same
    `RAINFALL_CLASS_EDGES` the RECONVECT label uses, so the two models'
    outputs are binned identically and the verification code cannot tell
    them apart by construction.

    Returns int32 class indices 0..4, matching
    `paste_predictions_to_canvas`'s radar convention.
    """
    from create_datasets import RAINFALL_CLASS_EDGES

    edges = RAINFALL_CLASS_EDGES if edges is None else edges
    mmh = np.asarray(mmh)
    cls = np.zeros(mmh.shape, dtype=np.int32)
    for edge in edges:
        cls += (mmh >= edge).astype(np.int32)
    return cls


def predict_classes(models, past_frames, max_step: int = MAX_STEP,
                    data_root="./our_data", source=SOURCE,
                    stats_period=SEPCONV_STATS_PERIOD, batch_size: int = 8):
    """Full path: compose -> denormalise -> bin.

    Returns:
        (classes, mmh) — each a dict {step: array}. The continuous mm/h
        field is returned alongside because calibration needs it: a
        threshold sweep operates on rain rate, not on class indices.
    """
    z = predict_sequence(models, past_frames, max_step=max_step,
                         batch_size=batch_size)
    mmh = {k: to_mmh(v, data_root, source, stats_period)
           for k, v in z.items()}
    classes = {k: bin_to_classes(v) for k, v in mmh.items()}
    return classes, mmh


def describe_output(mmh: dict[int, np.ndarray]) -> str:
    """Per-step summary, for eyeballing a rollout before trusting it."""
    lines = [f"  {'step':>4} {'lead':>8} {'min':>8} {'mean':>9} "
             f"{'max':>9} {'>=10mm/h':>10}"]
    for step in sorted(mmh):
        f = mmh[step]
        frac = float(np.count_nonzero(f >= 10.0)) / f.size * 100
        lines.append(
            f"  {step:>4} {LEAD_MINUTES[step]:>5} min {f.min():>8.3f} "
            f"{f.mean():>9.3f} {f.max():>9.2f} {frac:>9.3f}%")
    return "\n".join(lines)
