"""
bundle_eval_scores.py — Step 11 of the Shapley walkthrough.

Read each trained mode's `evaluation/eval_<mode>/evaluation_results.json`
and emit the per-leadtime CSVs that `feature_importance_analysis.py
--methods classical_shapley` expects:

    results/eval_scores/
        eval_leadtime-<prefix>-o.csv      <- mode with OPERA only
        eval_leadtime-<prefix>-om.csv     <- OPERA + MTG IR/WV
        eval_leadtime-<prefix>-on.csv     <- OPERA + NWCSAF
        eval_leadtime-<prefix>-omn.csv    <- OPERA + both

Each CSV has one numeric value per row, no header, in lead-time order
(t+15, t+30, t+45). The coalition letter on the right of the filename
encodes which input groups the model was trained with — see the
docstring of train_models.py for the registry, or
README.md (Step 11 of the Shapley runbook).

The classical Shapley formula then computes per-source contributions:

    phi(n) = 1/2 * [ (v_omn - v_om) + (v_on - v_o) ]    (NWCSAF)
    phi(m) = 1/2 * [ (v_omn - v_on) + (v_om - v_o) ]    (MTG IR/WV)

where v_<coalition> is the metric value for that model.

Usage
-----
Default — four OPERA Shapley modes, CSI (lightning) or accuracy (radar)
auto-detected from each JSON:

    python bundle_eval_scores.py

Custom metric — e.g. take HSS instead of accuracy:

    python bundle_eval_scores.py --metric HSS

Custom output directory:

    python bundle_eval_scores.py --output_dir results/eval_scores_hss

Override the coalition mapping (e.g. for a different Shapley study):

    python bundle_eval_scores.py \\
        --mode "mtg_lightning_no_nwcsaf=o" \\
        --mode "mtg_lightning=on"          \\
        --prefix lightning
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# Default 2-mode OPERA coalition (NWCSAF was dropped from the active
# build). Each (mode, letters) pair maps the trained-mode name to its
# source-letter combination. 'o' = OPERA-only baseline, 'om' = baseline
# + MTG IR/WV. Pass repeated `--mode MODE=LETTERS` to override the
# pairing (e.g. when including the lightning-as-input modes).
DEFAULT_COALITION = [
    ("mtg_opera_radar_only", "o"),
    ("mtg_opera_mtgmr",      "om"),
]

# Lead-time keys exactly as written into evaluation_results.json by
# evaluate_coalition.py (see LEAD_LABELS there).
LEAD_LABELS = ["t+15", "t+30", "t+45"]


def _default_metric(label_type: str) -> str:
    """Pick a sensible default metric for each label_type.

    evaluate_coalition.py writes different shapes depending on
    label_type:
      - 'lightning' (binary occurrence)  -> CSI, POD, FAR, ETS, HSS, PSS, ...
      - 'radar'     (multi-class bins)   -> accuracy, per_class.<name>.f1, ...
    """
    if label_type == "lightning":
        return "CSI"
    if label_type == "radar":
        return "accuracy"
    # Anything else (e.g. radar_continuous) falls back to MAE if present.
    return "MAE"


def _extract_metric(per_leadtime_block: dict, lead: str, metric: str,
                    mode: str) -> float:
    """Pull `metric` for a single lead time, returning a float.

    Supports nested keys via `per_class.<name>.<sub_metric>` so callers
    can reach the multi-class radar entries (e.g. `per_class.R<10.f1`).
    """
    if lead not in per_leadtime_block:
        raise KeyError(
            f"[{mode}] per_leadtime is missing {lead!r}. Available: "
            f"{sorted(per_leadtime_block.keys())}"
        )
    block = per_leadtime_block[lead]
    # Nested key lookup
    parts = metric.split(".")
    value = block
    for p in parts:
        if not isinstance(value, dict) or p not in value:
            available = (sorted(block.keys()) if isinstance(block, dict)
                         else type(block).__name__)
            raise KeyError(
                f"[{mode}] per_leadtime[{lead!r}] does not contain "
                f"{metric!r}. Available at this level: {available}"
            )
        value = value[p]
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"[{mode}] per_leadtime[{lead!r}][{metric!r}] is not numeric "
            f"(got {value!r}): {e}"
        )


def bundle_eval_scores(coalition: list[tuple[str, str]],
                       prefix: str,
                       eval_root: Path,
                       output_dir: Path,
                       metric: str | None = None,
                       source: str = "dbscan",
                       finetuned: bool = False,
                       ) -> dict[str, dict]:
    """Read per-mode `evaluation_results.json` files and write the
    per-leadtime CSVs that classical Shapley reads.

    Returns a summary dict keyed by coalition letter that records the
    file written, the metric used, and the per-leadtime values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for mode, letters in coalition:
        # Match the directory naming convention used by
        # evaluate_coalition.py: eval_{mode}_{source}[_finetuned].
        run_tag = f"{mode}_{source}"
        if finetuned:
            run_tag = f"{run_tag}_finetuned"
        results_path = eval_root / f"eval_{run_tag}" / "evaluation_results.json"
        if not results_path.is_file():
            raise FileNotFoundError(
                f"[{mode}] no evaluation_results.json at {results_path}. "
                f"Run `evaluate_coalition.py --mode {mode} --source "
                f"{source}{' --finetuned' if finetuned else ''}` first."
            )
        with open(results_path) as f:
            data = json.load(f)

        per_leadtime = data.get("per_leadtime")
        if not isinstance(per_leadtime, dict):
            raise ValueError(
                f"[{mode}] {results_path} is missing a `per_leadtime` "
                f"block — was this run produced by evaluate_coalition.py?"
            )

        label_type = data.get("label_type", "")
        chosen_metric = metric or _default_metric(label_type)

        values: list[float] = []
        for lead in LEAD_LABELS:
            values.append(_extract_metric(per_leadtime, lead,
                                          chosen_metric, mode))

        out_path = output_dir / f"eval_leadtime-{prefix}-{letters}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            for v in values:
                w.writerow([v])

        per_leadtime_pretty = ", ".join(
            f"{lead}={v:.4f}" for lead, v in zip(LEAD_LABELS, values)
        )
        print(f"  {mode:<25s} -> {out_path.name}   "
              f"metric={chosen_metric} ({per_leadtime_pretty})")

        summary[letters] = {
            "mode":          mode,
            "label_type":    label_type,
            "metric":        chosen_metric,
            "csv":           str(out_path),
            "values":        values,
        }

    return summary


def _parse_mode_arg(raw: str) -> tuple[str, str]:
    """Parse `MODE=LETTERS` (e.g. `mtg_opera_full=omn`)."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--mode expects MODE=LETTERS, got {raw!r}"
        )
    mode, _, letters = raw.partition("=")
    mode = mode.strip()
    letters = letters.strip().lower()
    if not mode or not letters:
        raise argparse.ArgumentTypeError(
            f"--mode {raw!r}: both MODE and LETTERS must be non-empty"
        )
    return mode, letters


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bundle per-leadtime CSVs from each mode's "
                    "evaluation_results.json into the format expected by "
                    "feature_importance_analysis.py "
                    "--methods classical_shapley.",
    )
    parser.add_argument(
        "--mode", action="append", type=_parse_mode_arg, default=None,
        metavar="MODE=LETTERS",
        help="Repeatable mapping of trained-mode name to its "
             "coalition-letter string. e.g. `mtg_opera_full=omn`. If "
             "omitted, the four-mode OPERA Shapley study is used "
             f"({DEFAULT_COALITION}).",
    )
    parser.add_argument(
        "--prefix", type=str, default="radar",
        help="Filename middle token: eval_leadtime-<prefix>-<letters>.csv. "
             "`radar` for the precipitation Shapley study (default), "
             "`lightning` for the lightning study.",
    )
    parser.add_argument(
        "--metric", type=str, default=None,
        help="Metric to extract per lead time. Defaults to CSI for "
             "label_type=lightning, accuracy for label_type=radar. "
             "Supports nested keys via dots, e.g. "
             "`per_class.R≥40.f1` for per-class F1 of the heaviest "
             "rain bin.",
    )
    parser.add_argument(
        "--eval_root", type=str, default="evaluation",
        help="Directory containing per-mode `eval_<mode>_<source>"
             "[_finetuned]/evaluation_results.json` files "
             "(default: evaluation/).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/eval_scores",
        help="Directory for the bundled per-leadtime CSVs "
             "(default: results/eval_scores/).",
    )
    parser.add_argument(
        "--source", type=str, default="dbscan",
        choices=["dbscan", "lightning"],
        help="Sample-selection track. Must match the --source flag "
             "the evaluation run used. Drives the eval_<mode>_<source> "
             "directory lookup (default: dbscan).",
    )
    parser.add_argument(
        "--finetuned", action="store_true",
        help="Read from the Swin-head fine-tuned model's evaluation "
             "directory (eval_<mode>_<source>_finetuned) instead of "
             "the base model's.",
    )
    args = parser.parse_args()

    coalition = args.mode if args.mode else DEFAULT_COALITION
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Bundling per-leadtime CSVs for classical Shapley")
    print("=" * 70)
    print(f"  eval_root:  {eval_root}")
    print(f"  output_dir: {output_dir}")
    print(f"  prefix:     {args.prefix}")
    print(f"  metric:     {args.metric or '(auto, per label_type)'}")
    print(f"  coalition:  {[m for m, _ in coalition]}")
    print()

    try:
        summary = bundle_eval_scores(
            coalition,
            prefix=args.prefix,
            eval_root=eval_root,
            output_dir=output_dir,
            metric=args.metric,
            source=args.source,
            finetuned=args.finetuned,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print()
    print(f"Wrote {len(summary)} CSV(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
