"""
compare_models.py - every evaluated model of one track, side by side.

Two sources, two outputs each.

1. evaluation/eval_<tag>/evaluation_results.json, written by
   evaluate_coalition.py and evaluate_sepconv_ensemble.py, become
   one figure per metric with every model as a line over lead time, and
   one table - rows models, columns metric x lead - as CSV, Markdown and
   LaTeX, the shape a paper prints.

2. validation/<stem>_samples.csv (stem = <track>_<scope>_thr<T>mmh_<tag>), written by
   validate_predictions.py, become the hit-level figures: for each lead
   and each season, the share of samples whose post-processed map hit at
   least L % of the ground-truth pixels, for L = 50 .. 90. Rainfall
   reads hits_pct_t+<k> (share of GT-active pixels detected on the
   post-processed map at the tuned HIGH, the >= 10 mm/h event);
   lightning reads pod_t+<min>. One figure per scope.

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
LOWER_IS_BETTER = {"FAR"}


def metric_title(key: str, title: str) -> str:
    """The metric's name with the direction that is better."""
    if key in LOWER_IS_BETTER:
        return f"{title} \u2193 (lower is better)"
    return f"{title} \u2191 (higher is better)"
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


def canonical_tag(tag: str) -> tuple[str, str]:
    """(model tag without the weights suffix, weights). The `_latest`
    suffix names which saved state a run scored, not another model, so
    a model's `best` and `latest` artefacts are matched as one model
    here; when both are on disk the `best` one is used and the other
    noted."""
    if tag.endswith("_latest"):
        return tag[:-len("_latest")], "latest"
    return tag, "best"


def load_evaluations(eval_root: Path, track: str, include_baseline: bool,
                     only: set[str] | None, weights: str = "any") -> list[dict]:
    """One record per evaluated model of `track`:
    {tag, baseline, leads: [min], per_lead: {min: {metric: v}}, aggregate}."""
    by_tag: dict[str, dict] = {}
    wanted = dict(canonical_tag(t) for t in (only or []))
    for results_path in sorted(eval_root.glob("eval_*/evaluation_results.json")):
        raw_tag = results_path.parent.name[len("eval_"):]
        tag, state = canonical_tag(raw_tag)
        baseline = tag.startswith("sepconv_")
        if baseline and not include_baseline:
            continue
        if only:
            # --models names the exact artefact: a plain tag is the final
            # save, a _latest tag the checkpoint run
            if tag not in wanted or wanted[tag] != state:
                continue
        elif weights != "any" and state != weights:
            continue
        if tag in by_tag and by_tag[tag]["weights"] == "best":
            print(f"  NOTE: {tag} evaluated with both weights; using best, "
                  f"skipping eval_{raw_tag}")
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
        if tag in by_tag:
            print(f"  NOTE: {tag} evaluated with both weights; using best, "
                  f"skipping eval_{by_tag[tag]['raw_tag']}")
        by_tag[tag] = {"tag": tag, "raw_tag": raw_tag, "weights": state,
                       "baseline": baseline,
                       "leads": sorted(per_lead), "per_lead": per_lead,
                       "aggregate": aggregate, "path": str(results_path),
                       "n_samples": results.get("n_samples"),
                       "split": results.get("split")}
    return list(by_tag.values())


def plot_metrics_per_lead(models: list[dict], metrics: list[tuple[str, str]],
                          labels: dict[str, str], out_dir: Path, track: str):
    """One grid figure: a panel per metric, every model a line, the
    title of each panel saying which direction is better."""
    if not models:
        return
    n = len(metrics)
    cols = 3 if n > 4 else 2
    rows = (n + cols - 1) // cols
    grid, gaxes = plt.subplots(rows, cols, figsize=(6 * cols, 4.4 * rows),
                               squeeze=False)
    for idx, (key, title) in enumerate(metrics):
        target = gaxes[idx // cols][idx % cols]
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
        target.set_title(metric_title(key, title))
        leads = sorted({x for m in models for x in m["leads"]})
        target.set_xticks(leads)
        target.grid(True, alpha=0.3)
        target.legend(fontsize=7)
    for idx in range(n, rows * cols):
        gaxes[idx // cols][idx % cols].set_visible(False)
    grid.suptitle(f"{track} - every evaluated model, per lead time",
                  fontsize=14, fontweight="bold")
    grid.tight_layout()
    grid.savefig(out_dir / "metrics_per_leadtime.png", dpi=150,
                 bbox_inches="tight")
    plt.close(grid)
    print("  Wrote metrics_per_leadtime.png")


def write_table(models: list[dict], metrics: list[tuple[str, str]],
                labels: dict[str, str], out_dir: Path, track: str):
    """Rows: models. Columns: metric x lead, plus the aggregate.
    CSV for machines, LaTeX for the paper, PNG as the asset of the
    README: one block per metric, the best value per column in bold
    (FAR lower is better)."""
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

    # LaTeX: booktabs, the best value per column in bold. FAR is lower-is-better.
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{" + f"{track.capitalize()} models on the held-out split, per lead time." + "}",
           "\\label{tab:" + f"{track}-comparison" + "}"]
    for key, title in metrics:
        col_spec = "l" + "r" * (len(leads) + 1)
        tex += ["\\begin{tabular}{" + col_spec + "}", "\\toprule",
                "\\textbf{" + title + "} "
                + ("($\\downarrow$ lower is better)" if key in LOWER_IS_BETTER
                   else "($\\uparrow$ higher is better)")
                + " & " + " & ".join(f"t+{x}" for x in leads)
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

    # PNG: the same blocks rendered as a figure, the best per column bold.
    n_rows = len(models)
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(9, 1.4 * (len(leads) + 2) + 4),
                                      0.42 * (n_rows + 1.4) * len(metrics) + 0.8),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, metrics):
        ax.axis("off")
        columns = [[m["per_lead"].get(x, {}).get(key) for m in models] for x in leads]
        columns.append([m["aggregate"].get(key) for m in models])
        best = []
        for col in columns:
            vals = [v for v in col if v is not None]
            best.append(None if not vals else (min(vals) if key == "FAR" else max(vals)))
        cells, row_names = [], []
        for i, m in enumerate(models):
            row_names.append(labels.get(m["tag"], m["tag"])
                             + (" (baseline)" if m["baseline"] else ""))
            cells.append(["--" if col[i] is None else f"{col[i]:.3f}" for col in columns])
        table = ax.table(cellText=cells, rowLabels=row_names,
                         colLabels=[f"t+{x}" for x in leads] + ["aggregate"],
                         loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)
        for (r, c), cell in table.get_celld().items():
            if r == 0 or c == -1:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#e8e8e8")
            elif best[c] is not None and columns[c][r - 1] is not None \
                    and abs(columns[c][r - 1] - best[c]) < 1e-9:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#dff0d8")
        ax.set_title(metric_title(key, title), fontsize=11, fontweight="bold", loc="left")
    fig.suptitle(f"{track} - model comparison (best per column in bold)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_dir / "comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Wrote comparison_table.csv / .tex / .png")


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


SPLITS = ("train", "validation", "test")


def scope_name(split: str | None, year: int | None, month: int | None,
               threshold_mmh: float | None = None) -> str:
    """`test_thr8mmh`, `2026_06_thr8mmh`, `all_months_thr8mmh`: names the
    hit-level output folder after the validation files it was built from."""
    parts = []
    if split:
        parts.append(split)
    if year is not None and month is not None:
        parts.append(f"{year:04d}_{month:02d}")
    name = "_".join(parts) or "all_months"
    if threshold_mmh is not None:
        name += f"_thr{threshold_mmh:g}mmh"
    return name


def load_samples(validation_dir: Path, track: str, include_baseline: bool,
                 only: set[str] | None, split: str | None = None,
                 year: int | None = None, month: int | None = None,
                 threshold_mmh: float | None = None,
                 weights: str = "any",
                 ) -> dict[str, list[dict]]:
    """{tag: rows} from the per-sample CSVs that match the scope.

    File names are <track>_[<split>_][<yyyy>_<mm>_]thr<T>mmh_<tag>_samples.csv.
    With `split`, only that split's files (of the given month when one
    is named); without it, only whole-month files (of the given month,
    or every month). `threshold_mmh` picks the runs at that selection
    threshold; when several thresholds are on disk and none is named,
    the call refuses rather than mixing populations. Each row carries
    its own year and month, read from its date. Files without the
    threshold piece (or untagged) are legacy and skipped."""
    pattern = re.compile(
        rf"^{track}_(?:(train|validation|test)_)?"
        rf"(?:(\d{{4}})_(\d{{2}})_)?thr([\d.]+)mmh_(.+)_samples\.csv$")
    per_model: dict[str, list[dict]] = defaultdict(list)
    val_weights: dict[str, str] = {}
    wanted = dict(canonical_tag(t) for t in (only or []))
    legacy = 0
    used: list[str] = []
    seen_thresholds: set[float] = set()
    for path in sorted(validation_dir.glob(f"{track}_*_samples.csv")):
        m = pattern.match(path.name)
        if not m:
            legacy += 1
            continue
        f_split, f_year, f_month, f_thr, raw_tag = m.groups()
        tag, state = canonical_tag(raw_tag)
        if tag in ("finetuned", "kd"):
            # The pre-tag naming put only the variant in the name.
            legacy += 1
            continue
        if f_split != split:
            continue
        if year is not None and month is not None:
            if f_year is None or (int(f_year), int(f_month)) != (year, month):
                continue
        elif f_year is not None and split:
            # A whole-split run asked for; month-restricted split files
            # would double count its samples.
            continue
        if tag.startswith("sepconv_") and not include_baseline:
            continue
        if only:
            if tag not in wanted or wanted[tag] != state:
                continue
        elif weights != "any" and state != weights:
            continue
        if tag in val_weights and val_weights[tag] != state:
            if val_weights[tag] == "best":
                print(f"  NOTE: {tag} validated with both weights; using best, "
                      f"skipping {path.name}")
                continue
            print(f"  NOTE: {tag} validated with both weights; using best, "
                  f"dropping the latest rows")
            per_model[tag] = []
        val_weights[tag] = state
        seen_thresholds.add(float(f_thr))
        if threshold_mmh is not None and float(f_thr) != float(threshold_mmh):
            continue
        used.append(path.name)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                date = (row.get("date") or "").strip()
                if len(date) >= 7 and date[4] == "-":
                    row["_year"], row["_month"] = int(date[:4]), int(date[5:7])
                else:
                    row["_year"] = int(f_year) if f_year else 0
                    row["_month"] = int(f_month) if f_month else 0
                per_model[tag].append(row)
    if threshold_mmh is None and len(seen_thresholds) > 1:
        raise SystemExit(
            f"validation runs at several selection thresholds are on disk "
            f"for this scope ({sorted(seen_thresholds)} mm/h); name one "
            f"with --rainfall_threshold_mmh.")
    for name in used:
        print(f"   read {name}")
    if legacy:
        print(f"  NOTE: {legacy} {track}_*_samples.csv file(s) without the "
              f"per-model and threshold naming were skipped.")
    return dict(per_model)


def hit_columns(rows: list[dict], track: str) -> dict:
    """{lead_key: column} from the CSV header. Rainfall: hits_pct_t+<k>
    (percent of GT-active pixels detected at the tuned HIGH). Lightning:
    pod_t+<min>, a ratio, scaled to percent when read."""
    if not rows:
        return {}
    cols = rows[0].keys()
    rx = re.compile(r"^hits_pct_t\+(\d+)$" if track == "rainfall"
                    else r"^pod_t\+(\d+)$")
    out: dict = {}
    for c in cols:
        m = rx.match(c)
        if m:
            out[int(m.group(1))] = c
    return out


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
    """A grid with one row per season plus "all", one column per lead;
    each panel is the share of samples whose hit rate reached at least
    L, over L, one line per model. Also a long-form CSV of every number
    plotted."""
    if not per_model:
        print("  No per-sample validation files for this track.")
        return
    tags = sorted(per_model)
    scale = 1.0 if track == "rainfall" else 100.0
    columns = {t: hit_columns(per_model[t], track) for t in tags}
    groups = list(seasons.items()) + [("all", None)]
    records = []

    lead_keys = sorted({k for t in tags for k in columns[t]})
    if not lead_keys:
        print("  No hit columns in the validation files.")
        return
    if True:
        fig, axes = plt.subplots(len(groups), len(lead_keys),
                                 figsize=(4.6 * len(lead_keys), 3.6 * len(groups)),
                                 squeeze=False)
        for gi, (gname, months) in enumerate(groups):
            for li, lead in enumerate(lead_keys):
                ax = axes[gi][li]
                for j, tag in enumerate(tags):
                    col = columns[tag].get(lead)
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
                        records.append({"season": gname, "lead": lead,
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
        what = ("post-processed rainfall map (>= 10 mm/h event, tuned HIGH)"
                if track == "rainfall" else "post-processed lightning map")
        fig.suptitle(f"{track}: samples whose {what} hit at least L % "
                     f"of the GT-active pixels",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        name = "hit_levels.png"
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
                        help="Root holding <stem>_samples.csv per run.")
    parser.add_argument("--output_dir", default="./comparison")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Add the SepConv-ens baseline (tags starting "
                             "with sepconv_). Rainfall only: no lightning "
                             "baseline exists yet.")
    parser.add_argument("--weights", default="any", choices=["best", "latest", "any"],
                        help="Which saved state of each model to read: best "
                             "(the final saves only), latest (the per-epoch "
                             "checkpoint runs only, the _latest artefacts), or "
                             "any (default: best when it exists, else latest; "
                             "the _latest suffix never splits a model in two).")
    parser.add_argument("--models", nargs="+", default=None, metavar="TAG",
                        help="Restrict to these artefact tags, each naming the "
                             "exact saved state: a plain tag is the final "
                             "save, a tag ending in _latest the checkpoint "
                             "run. Overrides --weights for the listed models.")
    parser.add_argument("--label", action="append", default=[], metavar="TAG=NAME",
                        help="Display name for a tag, repeatable.")
    parser.add_argument("--hit_levels", nargs="+", type=int, default=DEFAULT_LEVELS,
                        help="Hit-rate levels L, in percent (default 50 60 70 80 90).")
    parser.add_argument("--seasons", action="append", default=None,
                        metavar="NAME=M,M,...",
                        help="Season definition, repeatable (default DJF, "
                             "MAM, JJA, SON).")
    parser.add_argument("--split", default=None, choices=SPLITS,
                        help="Build the hit-level figures from the "
                             "validation runs on this dataset split "
                             "(<track>_<split>_<tag>_samples.csv).")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None,
                        help="With --year: use the validation runs on this "
                             "month (of the split with --split). Without "
                             "either, every whole-month run is pooled.")
    parser.add_argument("--rainfall_threshold_mmh", type=float, default=None,
                        help="Use the validation runs made at this selection "
                             "threshold (the thr<T>mmh piece of their names). "
                             "Needed only when runs at several thresholds "
                             "exist for the scope.")
    args = parser.parse_args()
    if (args.year is None) != (args.month is None):
        parser.error("--year and --month go together")

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
                              args.include_baseline, only, weights=args.weights)
    for m in models:
        print(f"   {m['tag']:55s} leads {m['leads']}"
              f"{'  (baseline)' if m['baseline'] else ''}"
              f"{'  [latest weights]' if m['weights'] == 'latest' else ''}")
    if not models:
        print("   none found")
    metrics = RAINFALL_METRICS if args.track == "rainfall" else LIGHTNING_METRICS
    plot_metrics_per_lead(models, metrics, labels, out_dir, args.track)
    write_table(models, metrics, labels, out_dir, args.track)

    print(f"\n2. Validation samples under {args.validation_dir}")
    per_model = load_samples(Path(args.validation_dir), args.track,
                             args.include_baseline, only,
                             split=args.split, year=args.year, month=args.month,
                             threshold_mmh=args.rainfall_threshold_mmh,
                             weights=args.weights)
    thr_used = args.rainfall_threshold_mmh
    if thr_used is None:
        # One threshold on disk (load_samples refused otherwise): read it
        # back from any file name so the folder says which.
        rx = re.compile(r"_thr([\d.]+)mmh_")
        found = {float(m.group(1)) for p in Path(args.validation_dir).glob(
            f"{args.track}_*_samples.csv") for m in [rx.search(p.name)] if m}
        thr_used = found.pop() if len(found) == 1 else None
    scope = scope_name(args.split, args.year, args.month, thr_used)
    print(f"   scope: {scope}")
    for tag, rows in sorted(per_model.items()):
        months = sorted({(r["_year"], r["_month"]) for r in rows})
        print(f"   {tag:55s} {len(rows):6d} samples over {len(months)} month(s)")
    hit_dir = out_dir / f"hit_levels_{scope}"
    hit_dir.mkdir(parents=True, exist_ok=True)
    plot_hit_levels(per_model, args.track, parse_seasons(args.seasons),
                    args.hit_levels, labels, hit_dir,
                    step_minutes=master_step_minutes())

    summary = {"track": args.track, "include_baseline": args.include_baseline,
               "evaluated": [{"tag": m["tag"], "baseline": m["baseline"],
                              "leads": m["leads"], "path": m["path"]} for m in models],
               "validated": {t: len(r) for t, r in per_model.items()},
               "hit_levels": args.hit_levels,
               "scope": {"split": args.split, "year": args.year,
                         "month": args.month, "threshold_mmh": thr_used,
                         "name": scope},
               "seasons": parse_seasons(args.seasons)}
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nOutputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
