"""
build_patch_ensemble.py — per-patch member selection for the seasonal ensemble
=============================================================================
Each ensemble member is a model trained on one season of one year (see
ensemble_plan.py). They are not uniformly better or worse than each other:
a member can be strong over the Carpathians and weak over the plains, or
the reverse. This script decides, patch by patch, which member to trust.

Selection metric
----------------
Per-patch CSI = TP / (TP + FP + FN), pooled over lead times, read from
the `per_patch` block that validate_predictions.py writes.

Scoring deliberately does NOT happen here. It happens inside
validate_predictions.py, at the point where post-processing already
runs — lightning is Hann-blended at stride 128 then hysteresis-
thresholded at the tuned per-lead HIGH; rainfall goes through the swept
p(argmax) hysteresis. Both tracks are therefore scored on the product
that actually ships, not on raw model output, and both are scored on the
full 768x1536 canvas, which is the only place a Hann blend spanning
patch boundaries is even defined.

This script is the selector: it reads one summary per member, compares
their per-patch CSI, and writes the manifest. The highest score wins the
patch; ties break on the member label, so the outcome never depends on
dict ordering or on which member was validated first.

Minimum sample count (--min_samples, required)
----------------------------------------------
A pooled CSI over two samples is noise, so a patch only chooses for
itself when it was validated on at least --min_samples samples. Below
that it takes the *overall best* member, the one with the highest CSI
pooled over every patch. When no patch at all reaches the minimum the
month is too thin for the rule to mean anything, so every patch with
samples keeps its own best member and only patches with zero samples go
to the overall best. Either way every patch ends up with a member; the
manifest records which rule chose it.

Determinism and the knowledge cutoff
------------------------------------
The result is written to a manifest that names every member, its training
period, the winning member per patch and the full score table. That
manifest is the ensemble: inference reads it rather than re-scoring, so
repeated validations resolve the same members and reproduce the same
numbers. `knowledge_cutoff` records the latest training-period end date
across all members - the point in time this ensemble's knowledge stops.
It only moves when a member is retrained and the manifest rebuilt.

Usage
-----
Validate each member first — that is what produces the per-patch tables:

    python validate_predictions.py --track rainfall --year 2025 --month 07 \\
        --mode mtg_opera_mtgmr_rainfall --period 2025warm

Then select:

    python build_patch_ensemble.py --mode mtg_opera_mtgmr_rainfall \\
        --track rainfall --year 2025 --month 07 --min_samples 20

    # a chosen subset, or a dry run that only reports what it would read
    python build_patch_ensemble.py --mode mtg_opera_mtgmr_rainfall \\
        --track rainfall --year 2025 --month 07 --min_samples 20 \\
        --members 2025warm 2025cold --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_config import SOURCE, resolve_data_root, resolve_model_dir
from periods import Period
from ensemble_plan import require_last_state, state_period

# Patch grid: 6 x 3 = 18 slots over the Romania canvas.
N_PATCHES = 18

MANIFEST_NAME_FMT = "ensemble_manifest_{mode}_{source}.json"


# =============================================================================
# Scoring
# =============================================================================


# =============================================================================
# Member evaluation
# =============================================================================

def load_member_scores(summary_path: Path) -> dict:
    """Read one member's per-patch table out of a validation summary.

    Scoring lives in validate_predictions.py, where post-processing
    already happens: lightning is Hann-blended then hysteresis-thresholded,
    rainfall goes through the swept p(argmax) hysteresis. Scoring here
    instead would judge raw model output, which is not what either track
    actually emits.
    """
    if not summary_path.is_file():
        raise SystemExit(f"Validation summary not found: {summary_path}")

    blob = json.loads(summary_path.read_text())
    per_patch = blob.get("per_patch")
    if not per_patch:
        raise SystemExit(
            f"{summary_path} has no `per_patch` block.\n"
            f"It predates per-patch scoring — re-run validate_predictions.py "
            f"in extraction mode for this member."
        )

    pp = blob.get("post_processing", {})
    return {
        "scores": {int(k): float(v["csi"]) for k, v in per_patch.items()},
        "detail": per_patch,
        "n_samples": blob.get("total_selected_samples", 0),
        "post_processing": {
            "low_threshold": (pp.get("low_threshold_per_lead")
                              or pp.get("low_threshold")),
            "high_threshold_per_lead": pp.get("high_threshold_per_lead"),
            "method": pp.get("method", "hysteresis"),
        },
        "summary": str(summary_path),
    }


def _best_of(candidates: list[tuple[float, str]]) -> tuple[float, str, bool]:
    """Highest score; ties break on the member label. Returns (score,
    label, tie_broken)."""
    best_score = max(score for score, _ in candidates)
    tied = sorted(lbl for score, lbl in candidates if score == best_score)
    return best_score, tied[0], len(tied) > 1


def overall_best(member_results: dict[str, dict]) -> dict:
    """The member with the highest CSI pooled over every patch it scored.

    Pooled from the per-patch TP/FP/FN, so a member is judged on the
    whole month, not on an average of per-patch ratios.
    """
    eps = 1e-7
    pooled: list[tuple[float, str]] = []
    for label, res in member_results.items():
        tp = sum(int(v.get("TP", 0)) for v in res["detail"].values())
        fp = sum(int(v.get("FP", 0)) for v in res["detail"].values())
        fn = sum(int(v.get("FN", 0)) for v in res["detail"].values())
        pooled.append((tp / (tp + fp + fn + eps), label))
    score, label, tie = _best_of(pooled)
    return {
        "member": label,
        "csi": round(score, 6),
        "tie_broken": tie,
        "all_scores": {lbl: round(sc, 6) for sc, lbl in
                       sorted(pooled, reverse=True)},
    }


def patch_sample_counts(member_results: dict[str, dict]) -> dict[int, int]:
    """Validated samples per patch: the largest count any member reports
    (members validated on the same month report the same count)."""
    counts = {patch: 0 for patch in range(1, N_PATCHES + 1)}
    for res in member_results.values():
        for key, v in res["detail"].items():
            patch = int(key)
            counts[patch] = max(counts[patch], int(v.get("n_samples", 0)))
    return counts


def select_per_patch(member_results: dict[str, dict],
                     min_samples: int) -> tuple[dict[int, dict], dict]:
    """Assign every patch to a member.

    A patch validated on at least `min_samples` samples takes its own
    best-scoring member. Below the minimum it takes the overall best
    member. When no patch reaches the minimum, patches with samples still
    take their own best member and only patches with zero samples go to
    the overall best. Ties break on the member label so the assignment
    is reproducible regardless of evaluation order.

    Returns (assignment, rule_info).
    """
    counts = patch_sample_counts(member_results)
    overall = overall_best(member_results)
    any_qualifies = any(n >= min_samples for n in counts.values())
    threshold = min_samples if any_qualifies else 1

    assignment: dict[int, dict] = {}
    for patch in range(1, N_PATCHES + 1):
        n = counts[patch]
        candidates = [
            (res["scores"][patch], label)
            for label, res in member_results.items()
            if patch in res["scores"]
        ]
        if n >= threshold and candidates:
            best_score, best_label, tie = _best_of(candidates)
            rule = "own_best"
        else:
            best_label, best_score, tie = (
                overall["member"], overall["csi"], overall["tie_broken"])
            rule = "no_samples" if n == 0 else "below_min_samples"
        assignment[patch] = {
            "member": best_label,
            "csi": round(best_score, 6),
            "rule": rule,
            "n_samples": n,
            "runner_up": (
                sorted(((sc, l) for sc, l in candidates if l != best_label),
                       reverse=True)[0][1]
                if rule == "own_best" and len(candidates) > 1 else None
            ),
            "all_scores": {lbl: round(sc, 6) for sc, lbl in
                           sorted(candidates, reverse=True)},
            "tie_broken": tie,
        }
    rule_info = {
        "min_samples": min_samples,
        "any_patch_meets_minimum": any_qualifies,
        "own_best_threshold": threshold,
        "overall_best": overall,
    }
    return assignment, rule_info


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Select the best ensemble member per patch from the "
                    "per-patch CSI tables validate_predictions.py wrote."
    )
    parser.add_argument("--mode", required=True,
                        help="Training mode shared by every member.")
    parser.add_argument("--track", default="rainfall",
                        choices=["rainfall", "lightning"],
                        help="Which validation summaries to read.")
    parser.add_argument("--split", default=None,
                        choices=["train", "validation", "test"],
                        help="Read the summaries of the members' validation "
                             "on this dataset split (validate_predictions "
                             "--split). With --year/--month, the split "
                             "restricted to that month.")
    parser.add_argument("--year", type=int, default=None,
                        help="Validation year the summaries cover.")
    parser.add_argument("--month", type=int, default=None,
                        help="Validation month the summaries cover. "
                             "Required without --split.")
    parser.add_argument("--data_root", default=str(resolve_data_root()))
    parser.add_argument("--model_dir", default=str(resolve_model_dir()))
    parser.add_argument("--validation_dir", default="./validation",
                        help="Where validate_predictions.py wrote its "
                             "summaries (default: ./validation).")
    parser.add_argument("--rainfall_threshold_mmh", type=float, default=10.0,
                        help="Selection threshold the members were validated "
                             "at (the thr<T>mmh piece of the summary names; "
                             "default 10).")
    parser.add_argument("--min_samples", type=int, required=True,
                        help="Validated samples a patch needs before its own "
                             "per-patch CSI chooses the member; below it the "
                             "overall best member is used. If no patch "
                             "reaches it, only patches with zero samples "
                             "fall back.")
    parser.add_argument("--members", nargs="+", default=None,
                        help="Member labels to consider (default: every "
                             "registered member with a summary on disk).")
    parser.add_argument("--finetuned", action="store_true",
                        help="Read the _finetuned summaries.")
    parser.add_argument("--output", default=None,
                        help="Manifest path (default: <model_dir>/"
                             "ensemble_manifest_<mode>_<source>.json).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be read and exit.")
    args = parser.parse_args()
    if args.min_samples < 1:
        parser.error("--min_samples must be at least 1")
    if (args.year is None) != (args.month is None):
        parser.error("--year and --month go together")
    if args.split is None and args.year is None:
        parser.error("give --split, or --year and --month, or both")
    # The scope piece of the validation stems, as validate_predictions
    # names them: 2026_06, test, test_2026_06.
    scope = "_".join(
        ([args.split] if args.split else [])
        + ([f"{args.year:04d}_{args.month:02d}"] if args.year is not None else [])
        + [f"thr{args.rainfall_threshold_mmh:g}mmh"])
    scope_label = scope.replace("_", "-") if not args.split else scope

    from train_models import build_run_tag, load_model_period

    state = require_last_state(args.data_root)
    registered = [m["label"] for m in state.get("members", [])
                  if m.get("status") != "no-data"]
    wanted = args.members or registered

    unknown = [m for m in wanted if m not in registered]
    if unknown:
        sys.exit(
            f"ERROR: {unknown} are not buildable members of the registered "
            f"plan. Registered: {registered}"
        )

    model_dir = Path(args.model_dir)
    validation_dir = Path(args.validation_dir)
    suffix = "_finetuned" if args.finetuned else ""

    # Summaries are written per (track, year, month); members are told
    # apart by the run tag baked into the filename by the per-period
    # validation run.
    resolved: dict[str, dict] = {}
    problems: list[str] = []
    for label in wanted:
        run_tag = build_run_tag(args.mode, SOURCE, label)
        model_path = model_dir / f"coalition_{run_tag}{suffix}.keras"
        stem = f"{args.track}_{scope}_{run_tag}{suffix}"
        summary = validation_dir / f"{stem}_summary.json"
        if not summary.is_file():
            problems.append(
                f"  {label}: no summary at {summary}\n"
                f"           run: python validate_predictions.py --track "
                f"{args.track}"
                + (f" --split {args.split}" if args.split else "")
                + (f" --year {args.year} --month {args.month}"
                   if args.year is not None else "")
                + f" --rainfall_threshold_mmh {args.rainfall_threshold_mmh:g}"
                + f" --mode {args.mode} --period {label}"
            )
            continue
        resolved[label] = {
            "summary": summary,
            "model_path": model_path,
            "period": state_period(state, label),
            "declared_period": (load_model_period(model_path)
                                if model_path.is_file() else None),
        }

    print("=" * 70)
    print(f"Patch ensemble - mode {args.mode}, track {args.track}, "
          f"{scope_label}")
    print("=" * 70)
    print(f"Registered members : {registered}")
    print(f"Summaries found    : {sorted(resolved) or 'none'}")
    if problems:
        print("Unavailable:")
        for line in problems:
            print(line)

    if len(resolved) < 2:
        print("\nNOTE: fewer than two members have summaries, so there is "
              "nothing to choose between. Every patch would go to the only "
              "candidate.")

    if args.dry_run:
        print("\n--dry-run: nothing read, nothing written.")
        return
    if not resolved:
        sys.exit("ERROR: no member has a validation summary - nothing to do.")

    # -- read -------------------------------------------------------------
    member_results: dict[str, dict] = {}
    for label, info in sorted(resolved.items()):
        result = load_member_scores(info["summary"])
        member_results[label] = result
        pp = result["post_processing"]
        print(f"\n  {label}: {len(result['scores'])} patches, "
              f"{result['n_samples']} samples")
        print(f"    post-processing: {pp['method']} "
              f"low={pp['low_threshold']} high={pp['high_threshold_per_lead']}")

    # Members scored at different operating points are not comparable: a
    # higher CSI could just mean a better-tuned threshold.
    lows = {json.dumps(m["post_processing"]["low_threshold"], sort_keys=True)
            for m in member_results.values()}
    if len(lows) > 1:
        print(f"\n  WARNING: members were scored at different LOW thresholds "
              f"{sorted(lows)}. Their CSI values are not strictly "
              f"comparable.")

    assignment, rule_info = select_per_patch(member_results,
                                             args.min_samples)

    # -- knowledge cutoff --------------------------------------------------
    ends = [info["declared_period"].end for info in resolved.values()
            if info["declared_period"]]
    ends += [info["period"].end for info in resolved.values()
             if info["period"] and not info["declared_period"]]
    cutoff = max(ends).strftime("%Y-%m-%d") if ends else None

    manifest = {
        "mode": args.mode,
        "source": SOURCE,
        "track": args.track,
        "validation_period": (
            f"{args.year:04d}-{args.month:02d}" if args.year is not None else None),
        "validation_split": args.split,
        "validation_scope": scope,
        "stage": "finetuned" if args.finetuned else "base",
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "knowledge_cutoff": cutoff,
        "selection_metric": (
            "per-patch CSI on post-processed canvases, pooled over lead "
            "times; read from validate_predictions per_patch blocks"
        ),
        "selection_rule": rule_info,
        "registry_state_utc": state.get("registered_utc"),
        "seasons": state.get("seasons"),
        "members": {
            label: {
                "model": str(info["model_path"]),
                "summary": str(info["summary"]),
                "period": (info["declared_period"] or info["period"]).to_dict()
                          if (info["declared_period"] or info["period"])
                          else None,
                "post_processing": member_results[label]["post_processing"],
                "n_samples": member_results[label]["n_samples"],
                "patch_scores": {
                    str(p): s for p, s in
                    sorted(member_results[label]["scores"].items())
                },
            }
            for label, info in resolved.items()
        },
        "patch_assignment": {str(p): a for p, a in sorted(assignment.items())},
        "unassigned_patches": [p for p in range(1, N_PATCHES + 1)
                               if p not in assignment],
    }

    out_path = Path(args.output) if args.output else (
        model_dir / MANIFEST_NAME_FMT.format(mode=args.mode, source=SOURCE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # -- report ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Per-patch assignment")
    print("=" * 70)
    ob = rule_info["overall_best"]
    print(f"  Overall best member: {ob['member']} (pooled CSI {ob['csi']:.4f})"
          f"{'  (tie)' if ob['tie_broken'] else ''}")
    if rule_info["any_patch_meets_minimum"]:
        print(f"  Patches with fewer than {args.min_samples} samples take "
              f"the overall best member.")
    else:
        print(f"  No patch reaches {args.min_samples} samples: every patch "
              f"with samples keeps its own best member; only patches with "
              f"none take the overall best.")
    print(f"\n  {'patch':>5}  {'samples':>7}  {'member':<14} {'CSI':>8}   "
          f"{'rule':<18} runner-up")
    print("  " + "-" * 70)
    # Most-validated patches first, the order validation works through them.
    for patch, a in sorted(assignment.items(),
                           key=lambda kv: (-kv[1]["n_samples"], kv[0])):
        flag = "  (tie)" if a["tie_broken"] else ""
        print(f"  {patch:>5}  {a['n_samples']:>7}  {a['member']:<14} "
              f"{a['csi']:>8.4f}   {a['rule']:<18} "
              f"{a['runner_up'] or '-'}{flag}")

    won: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    for a in assignment.values():
        won[a["member"]] = won.get(a["member"], 0) + 1
        by_rule[a["rule"]] = by_rule.get(a["rule"], 0) + 1
    print("\n  Patches won: " + ", ".join(
        f"{lbl} {n}" for lbl, n in sorted(won.items())))
    print("  By rule:     " + ", ".join(
        f"{r} {n}" for r, n in sorted(by_rule.items())))
    print(f"  Knowledge cutoff: {cutoff}")
    print(f"\nManifest written -> {out_path}")


if __name__ == "__main__":
    main()
