"""
compare_models.py - every evaluated model of one track, side by side.

Two sources, two outputs each.

1. evaluation/eval_<tag>/evaluation_results.json, written by
   evaluate_coalition.py and evaluate_sepconv_ensemble.py, become
   one figure per metric with every model as a line over lead time, and
   one table - rows models, columns metric x lead - as CSV, Markdown and
   LaTeX, the shape a paper prints.

2. validation/<track>_<yyyy>_<mm>_<tag>_samples.csv, written by
   validate_predictions.py, become the hit-level figures: for each lead
   and each season, the share of samples whose post-processed map hit at
   least L % of the ground-truth pixels, for L = 50 .. 90. The rainfall
   track carries a threshold sweep (hit_pct_t+k_ge<T>), so it gets one
   figure per T; lightning carries per-sample POD, one figure.

The baseline joins only with --include_baseline: it is a different
architecture scored on its own class map, and the comparison should
say so on purpose rather than by default. A lightning baseline does
not exist yet; the flag is accepted there and reports nothing.

Usage:
    python compare_models.py --track rainfall --include_baseline
    python compare_models.py --track lightning
    python compare_models.py --track rainfall --models opera_radar_only_rainfall_dbscan_w34 \\
        mtg_lightning_opera_rainfall_dbscan_f34 --label opera_radar_only_rainfall_dbscan_w34=ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAINFALL_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro F1"),
    ("macro_csi", "Macro CSI"),
    ("wet_csi", "Mean CSI, wet classes"),
]
LIGHTNING_METRICS = [
    ("CSI", "CSI"), ("POD", "POD"), ("FAR", "FAR"),
    ("ETS", "ETS"), ("HSS", "HSS"), ("PSS", "PSS"),
]
DEFAULT_SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
                   "JJA": [6, 7, 8], "SON": [9, 10, 11]}
DEFAULT_LEVELS = [50, 60, 70, 80, 90]
PALETTE = plt.get_cmap("tab10")
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]


# ============================================================================
# Part 1 - evaluation results
# ============================================================================

def _lead_minutes(label: str) -> int:
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else 0


def track_of(results: dict) -> str | None:
    """Which track a results file describes, from its per-lead keys."""
    per = results.get("per_leadtime") or {}
    if not per:
        return None
    first = next(iter(per.values()))
    if "CSI" in first and "POD" in first:
        return "lightning"
    if "per_class" in first or "macro_csi" in first:
        return "rainfall"
    return None


def rainfall_metrics(block: dict) -> dict:
    """The five rainfall numbers, derived from per_class when a file
    predates the macro keys."""
    per_class = block.get("per_class") or {}
    names = list(per_class)
    out = {"accuracy": block.get("accuracy")}
    if "balanced_accuracy" in block:
        out["balanced_accuracy"] = block["balanced_accuracy"]
    elif per_class:
        out["balanced_accuracy"] = float(np.mean(
            [per_class[c]["recall"] for c in names]))
    if "macro_f1" in block:
        out["macro_f1"] = block["macro_f1"]
    elif per_class:
        out["macro_f1"] = float(np.mean([per_class[c]["f1"] for c in names]))
    if "macro_csi" in block:
        out["macro_csi"] = block["macro_csi"]
    elif per_class:
        out["macro_csi"] = float(np.mean([per_class[c]["csi"] for c in names]))
    if per_class and len(names) > 1:
        out["wet_csi"] = float(np.mean([per_class[c]["csi"] for c in names[1:]]))
    return out


def load_evaluations(eval_root: Path, track: str, include_baseline: bool,
                     only: set[str] | None) -> list[dict]:
    """One record per evaluated model of `track`:
    {tag, baseline, leads: [min], per_lead: {min: {metric: v}}, aggregate}."""
    found = []
    for results_path in sorted(eval_root.glob("eval_*/evaluation_results.json")):
        tag = results_path.parent.name[len("eval_"):]
        baseline = tag.startswith("sepconv_")
        if baseline and not include_baseline:
            continue
        if only and tag not in only:
            continue
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"  skip {results_path}: {exc}", file=sys.stderr)
            continue
        if track_of(results) != track:
            continue
        per_lead = {}
        for label, block in results["per_leadtime"].items():
            minutes = block.get("lead_minutes") or _lead_minutes(label)
            per_lead[int(minutes)] = (rainfall_metrics(block) if track == "rainfall"
                                      else {k: block.get(k) for k, _ in LIGHTNING_METRICS})
        agg_block = results.get("aggregate") or {}
        aggregate = (rainfall_metrics(agg_block) if track == "rainfall"
                     else {k: agg_block.get(k) for k, _ in LIGHTNING_METRICS})
        found.append({"tag": tag, "baseline": baseline,
                      "leads": sorted(per_lead), "per_lead": per_lead,
                      "aggregate": aggregate, "path": str(results_path),
                      "n_samples": results.get("n_samples"),
                      "split": results.get("split")})
    return found


def plot_metrics_per_lead(models: list[dict], metrics: list[tuple[str, str]],
                          labels: dict[str, str], out_dir: Path, track: str):
    """One figure per metric, every model a line; plus a grid of all."""
    if not models:
        return
    n = len(metrics)
    cols = 3 if n > 4 else 2
    rows = (n + cols - 1) // cols
    grid, gaxes = plt.subplots(rows, cols, figsize=(6 * cols, 4.4 * rows),
                               squeeze=False)
    for idx, (key, title) in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for target in (ax, gaxes[idx // cols][idx % cols]):
            for j, m in enumerate(models):
                xs = m["leads"]
                ys = [m["per_lead"][x].get(key) for x in xs]
                if all(v is None for v in ys):
                    continue
                target.plot(xs, ys, marker=MARKERS[j % len(MARKERS)],
                            color=PALETTE(j % 10), linewidth=2, markersize=6,
                            linestyle="--" if m["baseline"] else "-",
                            label=labels.get(m["tag"], m["tag"]))
            target.set_xlabel("Lead time (min)")
            target.set_ylabel(title)
            target.set_title(title)
            leads = sorted({x for m in models for x in m["leads"]})
            target.set_xticks(leads)
            target.grid(True, alpha=0.3)
            target.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"metrics_per_leadtime_{key}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    for idx in range(n, rows * cols):
        gaxes[idx // cols][idx % cols].set_visible(False)
    grid.suptitle(f"{track} - every evaluated model, per lead time",
                  fontsize=14, fontweight="bold")
    grid.tight_layout()
    grid.savefig(out_dir / "metrics_per_leadtime.png", dpi=150,
                 bbox_inches="tight")
    plt.close(grid)
    print(f"  Wrote metrics_per_leadtime.png + {n} per-metric figures")


def write_table(models: list[dict], metrics: list[tuple[str, str]],
                labels: dict[str, str], out_dir: Path, track: str):
    """Rows: models. Columns: metric x lead, plus the aggregate.
    CSV for machines, Markdown for the README, LaTeX for the paper."""
    if not models:
        return
    leads = sorted({x for m in models for x in m["leads"]})
    header = ["Model"]
    for key, title in metrics:
        header += [f"{title} t+{x}" for x in leads] + [f"{title} agg."]
    rows = []
    for m in models:
        row = [labels.get(m["tag"], m["tag"]) + (" (baseline)" if m["baseline"] else "")]
        for key, _ in metrics:
            for x in leads:
                v = m["per_lead"].get(x, {}).get(key)
                row.append("" if v is None else f"{v:.3f}")
            v = m["aggregate"].get(key)
            row.append("" if v is None else f"{v:.3f}")
        rows.append(row)

    with open(out_dir / "comparison_table.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # Markdown: one block per metric so the table fits a page width.
    md = [f"# {track} - model comparison", ""]
    for key, title in metrics:
        md.append(f"## {title}")
        md.append("")
        md.append("| Model | " + " | ".join(f"t+{x}" for x in leads) + " | aggregate |")
        md.append("|---|" + "---|" * (len(leads) + 1))
        for m in models:
            cells = []
            for x in leads:
                v = m["per_lead"].get(x, {}).get(key)
                cells.append("" if v is None else f"{v:.3f}")
            v = m["aggregate"].get(key)
            cells.append("" if v is None else f"{v:.3f}")
            name = labels.get(m["tag"], m["tag"]) + (" (baseline)" if m["baseline"] else "")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
    (out_dir / "comparison_table.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX: booktabs, the best value per column in bold. FAR is lower-is-better.
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{" + f"{track.capitalize()} models on the held-out split, per lead time." + "}",
           "\\label{tab:" + f"{track}-comparison" + "}"]
    for key, title in metrics:
        col_spec = "l" + "r" * (len(leads) + 1)
        tex += ["\\begin{tabular}{" + col_spec + "}", "\\toprule",
                "\\textbf{" + title + "} & " + " & ".join(f"t+{x}" for x in leads)
                + " & agg. \\\\", "\\midrule"]
        columns = [[m["per_lead"].get(x, {}).get(key) for m in models] for x in leads]
        columns.append([m["aggregate"].get(key) for m in models])
        best = []
        for col in columns:
            vals = [v for v in col if v is not None]
            if not vals:
                best.append(None)
            else:
                best.append(min(vals) if key == "FAR" else max(vals))
        for i, m in enumerate(models):
            cells = []
            for j, col in enumerate(columns):
                v = col[i]
                if v is None:
                    cells.append("--")
                elif best[j] is not None and abs(v - best[j]) < 1e-9:
                    cells.append("\\textbf{" + f"{v:.3f}" + "}")
                else:
                    cells.append(f"{v:.3f}")
            name = labels.get(m["tag"], m["tag"]).replace("_", "\\_")
            if m["baseline"]:
                name += " (baseline)"
            tex.append(name + " & " + " & ".join(cells) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}", "\\vspace{0.6em}", ""]
    tex += ["\\end{table}"]
    (out_dir / "comparison_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print("  Wrote comparison_table.csv / .md / .tex")


# ============================================================================
# Part 2 - validation samples, hit levels
# ============================================================================

def parse_seasons(specs: list[str] | None) -> dict[str, list[int]]:
    if not specs:
        return dict(DEFAULT_SEASONS)
    out = {}
    for spec in specs:
        name, _, months = spec.partition("=")
        out[name.strip()] = [int(m) for m in months.split(",") if m.strip()]
    return out


def load_samples(validation_dir: Path, track: str, include_baseline: bool,
                 only: set[str] | None) -> dict[str, list[dict]]:
    """{tag: rows}, every month's per-sample CSV of every model, each row
    carrying its month. Legacy untagged files are reported and skipped."""
    pattern = re.compile(rf"^{track}_(\d{{4}})_(\d{{2}})_(.+)_samples\.csv$")
    per_model: dict[str, list[dict]] = defaultdict(list)
    legacy = 0
    for path in sorted(validation_dir.glob(f"{track}_*_samples.csv")):
        m = pattern.match(path.name)
        if not m:
            legacy += 1
            continue
        year, month, tag = int(m.group(1)), int(m.group(2)), m.group(3)
        if tag in ("finetuned", "kd"):
            # The pre-tag naming put only the variant in the name.
            legacy += 1
            continue
        if tag.startswith("sepconv_") and not include_baseline:
            continue
        if only and tag not in only:
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_month"] = month
                row["_year"] = year
                per_model[tag].append(row)
    if legacy:
        print(f"  NOTE: {legacy} untagged {track}_*_samples.csv file(s) "
              f"(written before per-model naming) were skipped.")
    return dict(per_model)


def hit_columns(rows: list[dict], track: str) -> dict:
    """{threshold_or_None: {lead_key: column}} from the CSV header.
    Rainfall: hit_pct_t+<k>_ge<T>, in percent. Lightning: pod_t+<min>,
    a ratio, scaled to percent when read."""
    if not rows:
        return {}
    cols = rows[0].keys()
    out: dict = defaultdict(dict)
    if track == "rainfall":
        rx = re.compile(r"^hit_pct_t\+(\d+)_ge([\d.]+)$")
        for c in cols:
            m = rx.match(c)
            if m:
                out[float(m.group(2))][int(m.group(1))] = c
    else:
        rx = re.compile(r"^pod_t\+(\d+)$")
        for c in cols:
            m = rx.match(c)
            if m:
                out[None][int(m.group(1))] = c
    return dict(out)


def share_above(rows: list[dict], column: str, level: float,
                scale: float) -> float | None:
    vals = []
    for r in rows:
        v = r.get(column)
        if v in (None, ""):
            continue
        try:
            vals.append(float(v) * scale)
        except ValueError:
            continue
    if not vals:
        return None
    return 100.0 * float(np.mean(np.asarray(vals) >= level))


def master_step_minutes(default: int = 15) -> int:
    """The master step from our_data/timestep_config.json, for lead labels."""
    try:
        from pipeline_config import resolve_data_root
        cfg = Path(resolve_data_root()) / "timestep_config.json"
        return int(json.loads(cfg.read_text(encoding="utf-8"))["step_minutes"])
    except Exception:
        return default


def plot_hit_levels(per_model: dict[str, list[dict]], track: str,
                    seasons: dict[str, list[int]], levels: list[int],
                    labels: dict[str, str], out_dir: Path,
                    step_minutes: int = 15):
    """For each threshold (rainfall) or once (lightning): a grid with one
    row per season plus "all", one column per lead; each panel is the
    share of samples whose hit rate reached at least L, over L, one line
    per model. Also a long-form CSV of every number plotted."""
    if not per_model:
        print("  No per-sample validation files for this track.")
        return
    tags = sorted(per_model)
    scale = 1.0 if track == "rainfall" else 100.0
    columns = {t: hit_columns(per_model[t], track) for t in tags}
    thresholds = sorted({T for t in tags for T in columns[t]}, key=lambda x: (x is None, x))
    groups = list(seasons.items()) + [("all", None)]
    records = []

    for T in thresholds:
        lead_keys = sorted({k for t in tags for k in columns[t].get(T, {})})
        if not lead_keys:
            continue
        fig, axes = plt.subplots(len(groups), len(lead_keys),
                                 figsize=(4.6 * len(lead_keys), 3.6 * len(groups)),
                                 squeeze=False)
        for gi, (gname, months) in enumerate(groups):
            for li, lead in enumerate(lead_keys):
                ax = axes[gi][li]
                for j, tag in enumerate(tags):
                    col = columns[tag].get(T, {}).get(lead)
                    if col is None:
                        continue
                    rows = [r for r in per_model[tag]
                            if months is None or r["_month"] in months]
                    ys = [share_above(rows, col, L, scale) for L in levels]
                    if all(v is None for v in ys):
                        continue
                    ax.plot(levels, ys, marker=MARKERS[j % len(MARKERS)],
                            color=PALETTE(j % 10), linewidth=2, markersize=5,
                            linestyle="--" if tag.startswith("sepconv_") else "-",
                            label=f"{labels.get(tag, tag)} (n={len(rows)})")
                    for L, y in zip(levels, ys):
                        records.append({"threshold_mmh": "" if T is None else f"{T:g}",
                                        "season": gname, "lead": lead,
                                        "model": tag, "level_pct": L,
                                        "share_of_samples_pct": "" if y is None else f"{y:.2f}",
                                        "n_samples": len(rows)})
                # Rainfall columns are keyed by step (t+k), lightning by
                # minutes (t+15); both are shown in minutes.
                lead_label = (f"t+{lead * step_minutes} min" if track == "rainfall"
                              else f"t+{lead} min")
                ax.set_title(f"{gname} - {lead_label}", fontsize=10)
                ax.set_xlabel("hit rate at least L (%)")
                ax.set_ylabel("share of samples (%)")
                ax.set_xticks(levels)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
        # One legend for the figure, taken from the first panel that drew
        # anything: the models are the same in every panel.
        for row_axes in axes:
            handles, names = None, None
            for ax in row_axes:
                handles, names = ax.get_legend_handles_labels()
                if handles:
                    break
            if handles:
                fig.legend(handles, [n.split(" (n=")[0] for n in names],
                           loc="upper right", fontsize=9, ncol=min(3, len(handles)))
                break
        what = ("post-processed rainfall map, GT at or above "
                f"{T:g} mm/h" if T is not None else "post-processed lightning map")
        fig.suptitle(f"{track}: samples whose {what} was hit at least L %",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        name = f"hit_levels_ge{T:g}.png" if T is not None else "hit_levels.png"
        fig.savefig(out_dir / name, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {name}")

    if records:
        with open(out_dir / "hit_levels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        print(f"  Wrote hit_levels.csv ({len(records)} rows)")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Every evaluated model of one track, side by side: "
                    "metrics per lead time, a paper-style table, and the "
                    "hit-level figures per season.")
    parser.add_argument("--track", required=True, choices=["rainfall", "lightning"])
    parser.add_argument("--eval_root", default="./evaluation",
                        help="Root holding eval_<tag>/evaluation_results.json.")
    parser.add_argument("--validation_dir", default="./validation",
                        help="Root holding <track>_<yyyy>_<mm>_<tag>_samples.csv.")
    parser.add_argument("--output_dir", default="./comparison")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Add the SepConv-ens baseline (tags starting "
                             "with sepconv_). Rainfall only: no lightning "
                             "baseline exists yet.")
    parser.add_argument("--models", nargs="+", default=None, metavar="TAG",
                        help="Restrict to these artifact tags.")
    parser.add_argument("--label", action="append", default=[], metavar="TAG=NAME",
                        help="Display name for a tag, repeatable.")
    parser.add_argument("--hit_levels", nargs="+", type=int, default=DEFAULT_LEVELS,
                        help="Hit-rate levels L, in percent (default 50 60 70 80 90).")
    parser.add_argument("--seasons", action="append", default=None,
                        metavar="NAME=M,M,...",
                        help="Season definition, repeatable (default DJF, "
                             "MAM, JJA, SON).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.track
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = dict(spec.split("=", 1) for spec in args.label)
    only = set(args.models) if args.models else None
    if args.track == "lightning" and args.include_baseline:
        print("NOTE: no lightning baseline exists; --include_baseline adds nothing.")

    print("=" * 70)
    print(f"Model comparison - {args.track}")
    print("=" * 70)

    print(f"\n1. Evaluation results under {args.eval_root}")
    models = load_evaluations(Path(args.eval_root), args.track,
                              args.include_baseline, only)
    for m in models:
        print(f"   {m['tag']:55s} leads {m['leads']}"
              f"{'  (baseline)' if m['baseline'] else ''}")
    if not models:
        print("   none found")
    metrics = RAINFALL_METRICS if args.track == "rainfall" else LIGHTNING_METRICS
    plot_metrics_per_lead(models, metrics, labels, out_dir, args.track)
    write_table(models, metrics, labels, out_dir, args.track)

    print(f"\n2. Validation samples under {args.validation_dir}")
    per_model = load_samples(Path(args.validation_dir), args.track,
                             args.include_baseline, only)
    for tag, rows in sorted(per_model.items()):
        months = sorted({(r["_year"], r["_month"]) for r in rows})
        print(f"   {tag:55s} {len(rows):6d} samples over {len(months)} month(s)")
    plot_hit_levels(per_model, args.track, parse_seasons(args.seasons),
                    args.hit_levels, labels, out_dir,
                    step_minutes=master_step_minutes())

    summary = {"track": args.track, "include_baseline": args.include_baseline,
               "evaluated": [{"tag": m["tag"], "baseline": m["baseline"],
                              "leads": m["leads"], "path": m["path"]} for m in models],
               "validated": {t: len(r) for t, r in per_model.items()},
               "hit_levels": args.hit_levels,
               "seasons": parse_seasons(args.seasons)}
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nOutputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
