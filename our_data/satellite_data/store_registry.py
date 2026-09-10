"""
store_registry.py — which disk holds which MTG date
========================================================================
The MTG store is the one product large enough to outgrow a disk: ~47 MB
of .npy per repeat cycle, so a year and a half runs past a terabyte. It
therefore lives across several roots, and something has to record which
date went where — otherwise every reader has to stat every root, and a
date that is simply on another drive reads as missing.

This is that record. `pipeline_msg_mtg.py` writes to it as it extracts;
`reproject.py` reads it to know which stores to walk.

Schema (our_data/satellite_data/mtg_store_index.json):

    {
      "updated_utc": "2026-08-31T09:12:00Z",
      "roots": ["E:\\\\...\\\\MTG", "G:\\\\nowcasting\\\\mtg_store"],
      "dates": {"2025-01-01": "E:\\\\...\\\\MTG", ...}
    }

A date maps to ONE root: the pipeline spills between drives at a window
boundary, so a date is never split. `resolve()` still verifies the
directory is really there before returning it, because an index is a
claim about the disk and the disk is the authority.

Usage:
    python our_data/satellite_data/store_registry.py
    python our_data/satellite_data/store_registry.py --scan E:\\...\\MTG G:\\nowcasting\\mtg_store
    python our_data/satellite_data/store_registry.py --verify
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A store may be zstd-compressed in place; count and measure both forms
# so the index and the per-disk chart stay honest either way.
sys.path.insert(0, str(PROJECT_ROOT))
from compress_datasets import list_arrays  # noqa: E402
INDEX_NAME = "mtg_store_index.json"
DEFAULT_INDEX = Path(__file__).resolve().parent / INDEX_NAME
DEFAULT_CHART = Path(__file__).resolve().parent / "mtg_store_distribution.png"

# All five channels are written per cycle, so a cycle's footprint is the
# sum across them - and they differ, vis_06 being 1 km where the IR/WV
# channels are 2 km. Sampling one day per channel is enough to size it.
CHANNELS = ("vis_06", "ir_38", "ir_105", "wv_63", "wv_73")

# The channel whose day-folders are taken as evidence a date is present.
# Any of the five would do; one is enough, and scanning all five to answer
# "which disk" would be four times the directory reads for no more
# certainty.
PROBE_CHANNEL = "ir_105"


# ============================================================================
# Load / save
# ============================================================================

def load(index_path=None) -> dict:
    """Read the index. Returns an empty skeleton when absent."""
    path = Path(index_path or DEFAULT_INDEX)
    if not path.is_file():
        return {"updated_utc": None, "roots": [], "dates": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: cannot read {path}: {exc}")
    blob.setdefault("roots", [])
    blob.setdefault("dates", {})
    return blob


def save(blob: dict, index_path=None) -> Path:
    """Write the index atomically.

    Temp-then-replace because a run interrupted mid-write would otherwise
    leave a truncated index, and a reader cannot tell a truncated index
    from a store that genuinely holds fewer dates.
    """
    path = Path(index_path or DEFAULT_INDEX)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob["updated_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    blob["roots"] = sorted({str(r) for r in blob.get("roots", [])})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


# ============================================================================
# Registration
# ============================================================================

def register(root, dates, index_path=None) -> Path:
    """Record that `dates` now live under `root`.

    Re-registering a date moves it: the last writer is the one that
    actually holds the arrays, and a spill re-extracting a date onto a
    second drive should update the answer rather than keep the old one.
    """
    root = str(Path(root).resolve())
    blob = load(index_path)
    if root not in blob["roots"]:
        blob["roots"].append(root)
    for d in dates:
        blob["dates"][str(d)] = root
    return save(blob, index_path)


def roots(index_path=None, include_default=None) -> list[str]:
    """Every store root known to the index, existing ones first.

    `include_default` is added when it is a real directory even if the
    index has never heard of it — the common case being a single-store
    setup that predates the index entirely.
    """
    blob = load(index_path)
    out = [r for r in blob["roots"] if Path(r).is_dir()]
    if include_default:
        d = str(Path(include_default).resolve())
        if d not in out and Path(d).is_dir():
            out.append(d)
    return out


def resolve(date_str, index_path=None, fallback_roots=()) -> str | None:
    """Which root holds `date_str`, or None.

    The index is a claim; the disk is the authority. A registered root
    that no longer has the date falls through to the scan, so an index
    left behind by a move does not silently hide data.
    """
    blob = load(index_path)
    candidates = []
    claimed = blob["dates"].get(str(date_str))
    if claimed:
        candidates.append(claimed)
    candidates.extend(str(r) for r in blob["roots"] if r != claimed)
    candidates.extend(str(r) for r in fallback_roots if r not in candidates)

    for root in candidates:
        if (Path(root) / PROBE_CHANNEL
                / f"nc4_{date_str}-Romania_{PROBE_CHANNEL}").is_dir():
            return root
    return None


# ============================================================================
# Scanning
# ============================================================================

def dates_in(root) -> list[str]:
    """Dates a store actually holds, read from its probe channel."""
    probe = Path(root) / PROBE_CHANNEL
    if not probe.is_dir():
        return []
    out = []
    for entry in os.scandir(probe):
        # nc4_2025-04-01-Romania_ir_105 -> 2025-04-01
        name = entry.name
        if entry.is_dir() and name.startswith("nc4_") and "-Romania_" in name:
            out.append(name[4:name.index("-Romania_")])
    return sorted(out)


def cycle_megabytes(root) -> float:
    """Average MB one repeat cycle occupies in `root`, across all channels.

    Measured rather than assumed: the figure is a capacity-planning tool,
    and a per-cycle constant carried over from another store would make
    it wrong in exactly the situation it exists for.
    """
    total = 0.0
    for channel in CHANNELS:
        ch_root = Path(root) / channel
        if not ch_root.is_dir():
            continue
        day = next((d for d in sorted(os.scandir(ch_root),
                                      key=lambda e: e.name) if d.is_dir()),
                   None)
        if day is None:
            continue
        sizes = [f.stat().st_size for f in os.scandir(day.path)
                 if f.name.endswith((".npy", ".npy.zst"))]
        if sizes:
            total += sum(sizes) / len(sizes) / (1024 ** 2)
    return total


def monthly_volume(index_path=None) -> tuple[list[str], dict[str, dict]]:
    """(months, {root: {month: {'dates', 'cycles', 'gb'}}}).

    Cycles are counted from the probe channel - one file per cycle - and
    scaled by the measured per-cycle footprint, which avoids walking
    123k files to answer a question about months.
    """
    blob = load(index_path)
    per_root: dict[str, dict] = {}
    months: set[str] = set()

    for root in sorted({str(r) for r in blob["dates"].values()}):
        if not Path(root).is_dir():
            continue
        mb = cycle_megabytes(root)
        buckets: dict[str, dict] = {}
        for date_str, claimed in blob["dates"].items():
            if claimed != root:
                continue
            probe = (Path(root) / PROBE_CHANNEL
                     / f"nc4_{date_str}-Romania_{PROBE_CHANNEL}")
            if not probe.is_dir():
                continue
            n_cycles = len(list_arrays(probe))
            month = date_str[:7]
            months.add(month)
            b = buckets.setdefault(month, {"dates": 0, "cycles": 0, "gb": 0.0})
            b["dates"] += 1
            b["cycles"] += n_cycles
            b["gb"] += n_cycles * mb / 1024
        if buckets:
            per_root[root] = buckets

    return sorted(months), per_root


def _fmt_gb(v: float) -> str:
    """GB with precision that suits the magnitude.

    A fixed format would print "0" for every bar on a small archive and
    a wall of decimals on a large one.
    """
    if v >= 100:
        return f"{v:,.0f} GB"
    if v >= 10:
        return f"{v:.1f} GB"
    if v >= 1:
        return f"{v:.2f} GB"
    if v >= 0.001:
        return f"{v * 1024:.0f} MB"
    return f"{v * 1024 * 1024:.0f} KB"


def render_chart(index_path=None, out_path=None) -> Path | None:
    """Stacked bars: how much of each month sits on which disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chart skipped: matplotlib is not installed.", file=sys.stderr)
        return None

    months, per_root = monthly_volume(index_path)
    if not months:
        print("Chart skipped: the index holds no dates yet.", file=sys.stderr)
        return None

    out_path = Path(out_path or DEFAULT_CHART)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One colour per disk, in a fixed order so a re-render keeps the same
    # disk the same colour.
    palette = ["#2b7bba", "#e8833a", "#4f9d69", "#b5495b", "#7a5195"]
    roots = sorted(per_root)

    # Label by drive when the drives differ, which is the whole point of
    # the chart; fall back to the folder name when two stores share one.
    drives = [Path(r).drive for r in roots]
    use_drive = len(set(drives)) == len(drives) and all(drives)
    labels = {r: (Path(r).drive if use_drive else Path(r).name)
              for r in roots}

    fig, ax = plt.subplots(figsize=(max(10, len(months) * 0.55), 5.5))
    bottom = [0.0] * len(months)
    for i, root in enumerate(roots):
        vals = [per_root[root].get(m, {}).get("gb", 0.0) for m in months]
        total = sum(vals)
        ax.bar(months, vals, bottom=bottom, width=0.72,
               color=palette[i % len(palette)],
               label=f"{labels[root]}  ({_fmt_gb(total)})   {root}")
        bottom = [b + v for b, v in zip(bottom, vals)]

    for x, tot in enumerate(bottom):
        if tot > 0:
            ax.text(x, tot, _fmt_gb(tot), ha="center", va="bottom",
                    fontsize=7, color="#555")

    ax.set_ylabel("Stored volume (GB)")
    ax.set_xlabel("Month")
    ax.set_title("MTG store — monthly volume by disk", fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart: {out_path}")
    return out_path


def scan(root_list, index_path=None) -> dict:
    """Rebuild the index from what is on disk.

    Later roots win a contested date, so pass them in the order the data
    was written. Use this after moving a store, or to adopt stores that
    were filled before the index existed.
    """
    blob = {"updated_utc": None, "roots": [], "dates": {}}
    for root in root_list:
        root = str(Path(root).resolve())
        if not Path(root).is_dir():
            print(f"  SKIP (not a directory): {root}", file=sys.stderr)
            continue
        found = dates_in(root)
        blob["roots"].append(root)
        for d in found:
            blob["dates"][d] = root
        print(f"  {len(found):>5} date(s)  {root}")
    save(blob, index_path)
    return blob


def verify(index_path=None) -> tuple[int, list[str]]:
    """Check every registered date is where the index says. Returns
    (n_ok, list_of_problems)."""
    blob = load(index_path)
    ok, bad = 0, []
    for date_str, root in sorted(blob["dates"].items()):
        p = (Path(root) / PROBE_CHANNEL
             / f"nc4_{date_str}-Romania_{PROBE_CHANNEL}")
        if p.is_dir():
            ok += 1
        else:
            bad.append(f"{date_str} claimed at {root}, not found")
    return ok, bad


# ============================================================================
# Reporting
# ============================================================================

def describe(index_path=None) -> str:
    blob = load(index_path)
    if not blob["dates"]:
        return ("MTG store index is empty. Populate it with:\n"
                "    python our_data/satellite_data/store_registry.py "
                "--scan <root> [<root> ...]")

    by_root: dict[str, list[str]] = {}
    for date_str, root in blob["dates"].items():
        by_root.setdefault(root, []).append(date_str)

    lines = [f"MTG store index  ({blob.get('updated_utc') or 'never'})", ""]
    for root in sorted(by_root):
        ds = sorted(by_root[root])
        here = "" if Path(root).is_dir() else "   [MISSING]"
        lines.append(f"  {len(ds):>5} date(s)  {ds[0]} .. {ds[-1]}   "
                     f"{root}{here}")
    lines.append("")
    lines.append(f"  {len(blob['dates']):>5} date(s) total across "
                 f"{len(by_root)} store(s)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record and query which disk holds which MTG date.")
    parser.add_argument("--index", default=None,
                        help=f"Index path (default: {DEFAULT_INDEX}).")
    parser.add_argument("--scan", nargs="+", metavar="ROOT", default=None,
                        help="Rebuild the index from these stores, in the "
                             "order the data was written (later roots win "
                             "a contested date).")
    parser.add_argument("--verify", action="store_true",
                        help="Check every registered date is where the "
                             "index claims.")
    parser.add_argument("--chart", nargs="?", const=str(DEFAULT_CHART),
                        default=None, metavar="PATH",
                        help=f"Render monthly stored volume, one colour per "
                             f"disk (default: {DEFAULT_CHART}).")
    args = parser.parse_args()

    if args.scan:
        print("Scanning stores:")
        scan(args.scan, args.index)
        print()
        if args.chart:
            render_chart(args.index, args.chart)
            print()

    if args.verify:
        ok, bad = verify(args.index)
        print(f"Verified {ok} date(s).")
        for line in bad:
            print(f"  MISSING: {line}")
        if bad:
            print(f"\n{len(bad)} problem(s). Re-scan with --scan to rebuild "
                  f"from disk.")
            return 1
        return 0

    print(describe(args.index))
    if args.chart:
        print()
        render_chart(args.index, args.chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
