"""
sepconv_compose.py — the SepConv-ens composition scheme (Czibula et al. 2024)
=============================================================================
The three base models Bm1, Bm3, Bm5 each predict a fixed number of steps
ahead. Reaching t+1 … t+8 is done by *composing* them, and the paper is
explicit that the composition is part of the architecture rather than a
post-hoc convenience: its Table 2 shows the alternative — repeatedly
applying a single base model — losing 3-4x in CSI. Dropping it would make
the baseline a strawman.

The rule
--------
Every base model consumes four consecutive frames and predicts `i` steps
past the LAST of them:

    Phi_i(M_a, M_b, M_c, M_tau)  ->  M_{tau + i}

So the target step of an entry is (last frame offset) + (base model lead).
`_validate_scheme` re-derives that for all eight entries at import time,
which is what stops a mis-transcribed row from silently producing a
forecast for the wrong lead time.

Steps 1-4 are computed from real observations only. Steps 5-8 are
autoregressive: they consume the predictions of steps 1-4. That is the
paper's design ("the estimations up to 24 minutes ahead is based only on
real radar data"), and it is why skill drops sharply past step 4.

Cadence
-------
The paper's step is 6 minutes, so its t+8 is 48 min. Ours is 15 minutes,
so the same eight steps reach **120 min**. Steps beyond t+4 therefore run
well past anything the paper validated, and past the horizon RECONVECT
produces at all. Only t+1..t+3 (15/30/45 min) overlap RECONVECT; the rest
are reported separately and must not anchor a comparative claim.

One transcription note: the paper prints M_{t+4} = Phi_5(M_{t-4}, M_{t-3},
M_{t-3}, M_{t-1}), repeating t-3 and omitting t-2. Read literally the
frames are non-consecutive and the model would receive a duplicate, so it
is taken here as the obvious typo for (t-4, t-3, t-2, t-1) — which is also
what makes the target step arithmetic come out at t+4.
"""

from __future__ import annotations

import numpy as np

# Offsets of the real frames a sample provides, oldest first. Matches the
# past=4 sequence window: idx_t-4 .. idx_t0.
REAL_FRAME_OFFSETS = (-4, -3, -2, -1, 0)

# (target_step, base_model_lead, four consecutive frame offsets)
# Negative/zero offsets are observations; positive ones are predictions
# made earlier in this same table.
COMPOSITION: tuple[tuple[int, int, tuple[int, int, int, int]], ...] = (
    # --- from observations only ---
    (1, 1, (-3, -2, -1, 0)),
    (2, 3, (-4, -3, -2, -1)),
    (3, 3, (-3, -2, -1, 0)),
    (4, 5, (-4, -3, -2, -1)),
    # --- autoregressive ---
    (5, 1, (1, 2, 3, 4)),
    (6, 3, (0, 1, 2, 3)),
    (7, 3, (1, 2, 3, 4)),
    (8, 5, (0, 1, 2, 3)),
)

BASE_LEADS = (1, 3, 5)
STEP_MINUTES = 15


def _validate_scheme() -> None:
    """Re-derive every target step from the rule, at import time.

    A composition table is exactly the kind of thing that gets edited
    once and silently mis-attributes a forecast to the wrong lead. This
    makes that a startup failure instead.
    """
    seen = set()
    for step, lead, frames in COMPOSITION:
        if lead not in BASE_LEADS:
            raise ValueError(f"step {step}: Bm{lead} is not a base model")
        if list(frames) != list(range(frames[0], frames[0] + 4)):
            raise ValueError(
                f"step {step}: frames {frames} are not four consecutive")
        derived = frames[-1] + lead
        if derived != step:
            raise ValueError(
                f"step {step}: Bm{lead} on frames ending {frames[-1]} "
                f"predicts t+{derived}, not t+{step}")
        for f in frames:
            if f > 0 and f >= step:
                raise ValueError(
                    f"step {step} consumes t+{f}, which is not available yet")
            if f <= 0 and f not in REAL_FRAME_OFFSETS:
                raise ValueError(f"step {step}: no observation at offset {f}")
        seen.add(step)
    if seen != set(range(1, len(COMPOSITION) + 1)):
        raise ValueError(f"composition does not cover t+1..t+8: {sorted(seen)}")


_validate_scheme()

MAX_STEP = len(COMPOSITION)
LEAD_MINUTES = {k: k * STEP_MINUTES for k in range(1, MAX_STEP + 1)}

# Steps computed purely from observations, versus those that consume
# earlier predictions. Reported alongside results because the distinction
# explains the shape of the skill curve.
OBSERVED_ONLY_STEPS = tuple(
    step for step, _, frames in COMPOSITION if all(f <= 0 for f in frames))
AUTOREGRESSIVE_STEPS = tuple(
    step for step in range(1, MAX_STEP + 1) if step not in OBSERVED_ONLY_STEPS)


def describe() -> str:
    """The explicit lead-time table, for logs and documentation."""
    lines = [
        f"SepConv-ens composition ({STEP_MINUTES}-min steps)",
        f"  {'step':>4} {'lead':>8}  {'model':<6} frames                source",
        "  " + "-" * 62,
    ]
    for step, lead, frames in COMPOSITION:
        src = "observed" if all(f <= 0 for f in frames) else "autoregressive"
        fs = ", ".join(f"t{f:+d}" if f else "t" for f in frames)
        lines.append(f"  {step:>4} {LEAD_MINUTES[step]:>6} min  "
                     f"Bm{lead:<4} ({fs})  {src}")
    lines.append("")
    lines.append(f"  observed-only : t+{OBSERVED_ONLY_STEPS} "
                 f"(to {LEAD_MINUTES[max(OBSERVED_ONLY_STEPS)]} min)")
    lines.append(f"  autoregressive: t+{AUTOREGRESSIVE_STEPS} "
                 f"(to {LEAD_MINUTES[MAX_STEP]} min)")
    return "\n".join(lines)


def compose(predict_fn, past_frames, max_step: int = MAX_STEP):
    """Run the full composition and return predictions for t+1..t+max_step.

    Args:
        predict_fn: callable (lead, list_of_4_arrays) -> array. Takes the
            base-model lead (1, 3 or 5) and four frames oldest-first,
            returns one predicted frame. Injected rather than taking the
            models directly so the scheme is testable without weights.
        past_frames: the five real frames t-4 .. t0, oldest first. Each
            (H, W) or (H, W, 1), in log_zscore space.
        max_step: stop early; useful when only the horizons that overlap
            RECONVECT are wanted.

    Returns:
        dict {step: predicted frame}, steps 1..max_step.
    """
    if len(past_frames) != len(REAL_FRAME_OFFSETS):
        raise ValueError(
            f"expected {len(REAL_FRAME_OFFSETS)} real frames "
            f"(t{REAL_FRAME_OFFSETS[0]}..t0), got {len(past_frames)}")

    # One dict keyed by offset, so observations and predictions are looked
    # up identically and an autoregressive step cannot accidentally reach
    # for an observation that does not exist.
    frames = {off: np.asarray(f)
              for off, f in zip(REAL_FRAME_OFFSETS, past_frames)}

    out: dict[int, np.ndarray] = {}
    for step, lead, offsets in COMPOSITION:
        if step > max_step:
            break
        stack = [frames[o] for o in offsets]
        pred = np.asarray(predict_fn(lead, stack))
        frames[step] = pred
        out[step] = pred
    return out


def frames_used(step: int) -> tuple[int, ...]:
    for s, _lead, offsets in COMPOSITION:
        if s == step:
            return offsets
    raise KeyError(f"no composition entry for t+{step}")


def base_model_for(step: int) -> int:
    for s, lead, _offsets in COMPOSITION:
        if s == step:
            return lead
    raise KeyError(f"no composition entry for t+{step}")


if __name__ == "__main__":
    print(describe())
