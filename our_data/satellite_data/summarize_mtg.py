"""
summarize_mtg.py - Summarise downloaded MTG FCI chunks by date.

Scans the _raw_chunks directory, parses FCI filenames, and produces a
per-date CSV + missing-timesteps JSON. Coverage is measured against the
configured MTG minute filter (`products.mtg.filter` from
timestep_config.json) - the same set pipeline_msg_mtg.py uses to filter
downloads before transfer - so a fully-covered day reports 100%, not
"66.7%" against the unfiltered 144-cycle native cadence.

Usage:
    python summarize_mtg.py
    python summarize_mtg.py --raw_dir path/to/_raw_chunks
    python summarize_mtg.py --output summary.csv
"""

import os
import re
import sys
import csv
import json
import argparse
import datetime
from collections import defaultdict
from pathlib import Path


# Default path: anchor to the project root (two levels up from this script,
# which lives at our_data/satellite_data/) so the same default works whether
# you run from the project root or from the script's own directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = str(PROJECT_ROOT / 'our_data' / 'satellite_data'
                      / 'MTG' / '_raw_chunks')
DEFAULT_REPROJ_DIR = (PROJECT_ROOT / 'our_data' / 'reprojected_data'
                      / 'satellite_data' / 'MTG')
TIMESTEP_CONFIG_PATH = PROJECT_ROOT / 'our_data' / 'timestep_config.json'
# Summaries, the missing-timestep JSON and the coverage chart belong with
# the product they describe, not at the repository root. This file already
# lives in that folder, so anchor to it - and to the file, not the working
# directory, so the defaults hold from anywhere.
PRODUCT_DIR = Path(__file__).resolve().parent


# The .npy stores may be zstd-compressed in place (see
# compress_datasets.py --compress-npy); list_arrays yields logical
# .npy names either way, so the filename patterns below still match.
sys.path.insert(0, str(PROJECT_ROOT))
from compress_datasets import list_arrays  # noqa: E402


# MTG native cadence (repeat cycle = 10 min); the file has 144 cycles
# per 24h day. The filter from timestep_config.json picks a subset of
# the cycles that align with the master grid.
NATIVE_CADENCE_MINUTES = 10
NATIVE_CYCLES_PER_DAY = 1440 // NATIVE_CADENCE_MINUTES  # 144


# =============================================================================
# Cadence config
# =============================================================================

def load_minute_filter():
    """Read the MTG minute filter + step from timestep_config.json.

    Returns (filter set, step_minutes). The filter is the minute-of-hour
    set the master grid snaps to for MTG - exactly the set
    pipeline_msg_mtg.py pre-filters on before downloading.
    """
    if not TIMESTEP_CONFIG_PATH.exists():
        print(
            f"ERROR: timestep config not found at {TIMESTEP_CONFIG_PATH}.\n"
            f"Run from the project root:\n"
            f"    python validate_timestep.py --step_minutes <N>",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = json.loads(TIMESTEP_CONFIG_PATH.read_text())
    flt = cfg.get('products', {}).get('mtg', {}).get('filter')
    if flt is None:
        print(
            f"ERROR: product 'mtg' has no minute filter in "
            f"{TIMESTEP_CONFIG_PATH}. Re-run validate_timestep.py with a "
            f"non-continuous mtg cadence in product_cadences.config.",
            file=sys.stderr,
        )
        sys.exit(2)
    return set(int(m) for m in flt), int(cfg['step_minutes'])


def rc_minute_of_hour(rc):
    """Minute-of-hour for a 1-indexed MTG repeat cycle (10-min native)."""
    return ((rc - 1) * NATIVE_CADENCE_MINUTES) % 60


def expected_rc_set(filter_minutes):
    """Set of MTG repeat-cycle numbers that fall on the configured filter.

    For a 10-min cadence and a filter of e.g. {0, 10, 30, 40}, this picks
    cycles whose minute-of-hour is in the filter - typically 4 of every
    6 cycles per hour, so 96 per day at step=15.
    """
    return {
        rc for rc in range(1, NATIVE_CYCLES_PER_DAY + 1)
        if rc_minute_of_hour(rc) in filter_minutes
    }


# =============================================================================
# Filename parsing
# =============================================================================

def parse_fci_filename(filename):
    """Extract metadata from an FCI L1C filename."""
    info = {
        'chunk_number': None,
        'repeat_cycle_in_day': None,
        'sensing_start': None,
        'is_body': 'CHK-BODY' in filename or 'CHK_BODY' in filename,
        'is_trailer': 'TRAIL' in filename,
    }

    match = re.search(r'_(\d{4})_(\d{4})\.nc$', filename)
    if match:
        info['repeat_cycle_in_day'] = int(match.group(1))
        info['chunk_number'] = int(match.group(2))

    match = re.search(r'_(?:OPE|DEV)_(\d{14})_(\d{14})_', filename)
    if match:
        try:
            info['sensing_start'] = datetime.datetime.strptime(
                match.group(1), '%Y%m%d%H%M%S'
            )
        except ValueError:
            pass

    return info


def nominal_time(date_ref, repeat_cycle_in_day, period_min=NATIVE_CADENCE_MINUTES):
    """Compute nominal start time from repeat cycle number."""
    base = date_ref.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + datetime.timedelta(
        minutes=(repeat_cycle_in_day - 1) * period_min
    )


def rc_to_hhmm(rc):
    """1-indexed repeat cycle -> 'HH:MM' string."""
    total_min = (rc - 1) * NATIVE_CADENCE_MINUTES
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}"


# =============================================================================
# Summarise
# =============================================================================

def date_range(start: str | None, end: str | None) -> list[str]:
    """Every YYYY-MM-DD from start to end inclusive, or [] if unbounded.

    This is what turns "no files for this date" into a reported gap. Without
    an explicit range the scan can only describe dates it found files for,
    so a date absent from disk entirely is silently not missing — it simply
    does not exist as far as the summary is concerned, and the Data Store
    backfill therefore never asks for it.
    """
    if not start or not end:
        return []
    from datetime import datetime, timedelta
    try:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        print(f"ERROR: --start/--end must be YYYY-MM-DD "
              f"(got {start!r}, {end!r})", file=sys.stderr)
        sys.exit(2)
    if d1 < d0:
        print(f"ERROR: --end {end} precedes --start {start}", file=sys.stderr)
        sys.exit(2)
    out, cur = [], d0
    while cur <= d1:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def summarize(raw_dir, filter_minutes, start=None, end=None,
              required_parts=2):
    """
    Scan _raw_chunks and build a per-date summary.

    `start`/`end` (YYYY-MM-DD, inclusive) declare the range the archive is
    EXPECTED to cover. Dates in that range with no files on disk are
    reported as fully missing rather than omitted. Without them the range
    is inferred from what was found, which cannot see an absent date.

    Returns (rows, dates) where:
        rows: list of per-date dicts ready for CSV / printing
        dates: raw per-date data passed to build_missing_timesteps
    """
    if not os.path.isdir(raw_dir):
        print(f"ERROR: directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    expected_rcs = expected_rc_set(filter_minutes)
    expected_per_day = len(expected_rcs)

    nc_files = [f for f in os.listdir(raw_dir) if f.endswith('.nc')]
    print(f"Found {len(nc_files)} .nc files in {raw_dir}")

    dates = defaultdict(lambda: {
        'body_files':     0,
        'trailer_files':  0,
        'repeat_cycles':  set(),       # all rc numbers present
        'chunks_seen':    defaultdict(set),  # rc_num -> set of chunk_numbers
    })

    for filename in nc_files:
        info = parse_fci_filename(filename)
        if info['sensing_start'] is None:
            continue
        date_str = info['sensing_start'].strftime('%Y-%m-%d')

        if info['is_trailer']:
            dates[date_str]['trailer_files'] += 1
            continue

        if info['is_body']:
            dates[date_str]['body_files'] += 1
            if info['repeat_cycle_in_day'] is not None:
                rc = info['repeat_cycle_in_day']
                dates[date_str]['repeat_cycles'].add(rc)
                if info['chunk_number'] is not None:
                    dates[date_str]['chunks_seen'][rc].add(
                        info['chunk_number']
                    )

    # Seed every expected date so one with no files on disk still produces
    # a row — the defaultdict gives it zero counts and an empty rc set, so
    # the whole day comes out as missing rather than absent.
    expected_dates = date_range(start, end)
    if expected_dates:
        found = set(dates.keys())
        for date_str in expected_dates:
            dates[date_str]  # touch: instantiates the defaultdict entry
        empty = [d for d in expected_dates if d not in found]
        outside = sorted(found - set(expected_dates))
        print(f"Expected range : {expected_dates[0]} .. {expected_dates[-1]} "
              f"({len(expected_dates)} dates)")
        if empty:
            print(f"  {len(empty)} date(s) in range have NO files at all "
                  f"— reported as fully missing")
        if outside:
            print(f"  WARNING: {len(outside)} date(s) on disk fall OUTSIDE "
                  f"the range and are still reported: {outside[:5]}"
                  + (" ..." if len(outside) > 5 else ""))

    rows = []
    for date_str in sorted(dates.keys()):
        d = dates[date_str]
        complete = 0
        incomplete = 0
        for rc, chunks in d['chunks_seen'].items():
            if len(chunks) >= required_parts:
                complete += 1
            else:
                incomplete += 1

        present_rcs = d['repeat_cycles']
        on_grid_present = present_rcs & expected_rcs
        off_grid_present = present_rcs - expected_rcs

        coverage_pct = (
            len(on_grid_present) / expected_per_day * 100
            if expected_per_day else 0
        )

        rows.append({
            'date':              date_str,
            'body_files':        d['body_files'],
            'trailer_files':     d['trailer_files'],
            'repeat_cycles':     len(present_rcs),       # total (audit)
            'on_grid':           len(on_grid_present),   # match the filter
            'off_grid':          len(off_grid_present),  # outside the filter
            'expected':          expected_per_day,
            'complete_2chunk':   complete,
            'incomplete_1chunk': incomplete,
            'coverage_pct':      round(coverage_pct, 1),
        })

    return rows, dates


def build_missing_timesteps(dates, filter_minutes, required_parts=2):
    """
    Compute the exact missing on-grid timesteps for each date.

    Only on-grid cycles count - intersect_product_coverage.py snaps the
    master HHMM to the MTG filter, so off-grid native cycles are
    unreachable. They're tracked separately for diagnostics but not as
    "missing".
    """
    expected_rcs = expected_rc_set(filter_minutes)

    result = {
        'config': {'minute_filter': sorted(filter_minutes)},
        'dates': {},
        'summary': {},
    }
    total_present = 0
    total_missing = 0

    for date_str in sorted(dates.keys()):
        d = dates[date_str]
        present_rcs = d['repeat_cycles']
        on_grid_present = present_rcs & expected_rcs
        on_grid_missing = expected_rcs - present_rcs
        off_grid_present = present_rcs - expected_rcs

        missing_times = sorted(rc_to_hhmm(rc) for rc in on_grid_missing)
        off_grid_times = sorted(rc_to_hhmm(rc) for rc in off_grid_present)

        incomplete_times = []
        for rc in sorted(on_grid_present):
            chunks = d['chunks_seen'].get(rc, set())
            if len(chunks) < required_parts:
                incomplete_times.append(rc_to_hhmm(rc))

        n_present = len(on_grid_present)
        n_missing = len(on_grid_missing)
        coverage = (
            n_present / len(expected_rcs) * 100 if expected_rcs else 0
        )

        total_present += n_present
        total_missing += n_missing

        result['dates'][date_str] = {
            'present':          n_present,
            'missing':          n_missing,
            'off_grid_present': len(off_grid_present),
            'expected':         len(expected_rcs),
            'coverage_pct':     round(coverage, 1),
            'missing_times':    missing_times,
            'off_grid_times':   off_grid_times,
            'incomplete_times': incomplete_times,
        }

    total_expected = len(dates) * len(expected_rcs)
    result['summary'] = {
        'total_dates':           len(dates),
        'expected_per_day':      len(expected_rcs),
        'total_present':         total_present,
        'total_missing':         total_missing,
        'total_expected':        total_expected,
        'overall_coverage_pct':  round(
            total_present / total_expected * 100, 1
        ) if total_expected > 0 else 0,
    }

    return result


def save_missing_json(dates, filter_minutes, output_path,
                      required_parts=2):
    """Build and save the missing timesteps JSON."""
    missing = build_missing_timesteps(dates, filter_minutes, required_parts)
    with open(output_path, 'w') as f:
        json.dump(missing, f, indent=2)

    s = missing['summary']
    print(f"Saved missing timesteps: {output_path}")
    print(f"  {s['total_present']}/{s['total_expected']} present "
          f"({s['overall_coverage_pct']}%), "
          f"{s['total_missing']} missing")


# =============================================================================
# Output
# =============================================================================

# =============================================================================
# Scanning the processed .npy output
# =============================================================================
# Once --delete_raw removes the chunks, the extracted arrays are the only
# evidence a cycle exists, so coverage has to be measured from them. The
# per-date structure produced here is deliberately identical to the raw
# scan's, with one substitution: `chunks_seen[rc]` holds the set of
# CHANNELS extracted rather than the set of chunk numbers present.
#
# That changes what "incomplete" means. On raw it was "one of the two
# Romania chunks arrived". On .npy it is "some channels extracted but not
# all" — the same shape of failure one stage later, which is why
# `required_parts` is a parameter rather than a hard-coded 2.

NPY_FILE_PATTERN = re.compile(
    r'^nc4_(?P<date>\d{4}-\d{2}-\d{2})-Romania_(?P<hhmm>\d{4})_'
    r'(?P<channel>[a-z]+_\d+)\.npy$'
)


def expected_channels(products_file=None):
    """Channels a complete cycle must have, from satellite_products.json.

    Read from the products file rather than discovered from the directory
    tree on purpose: a channel that was never downloaded has no directory,
    so discovery would quietly redefine "complete" as "whatever is here".
    """
    path = Path(products_file) if products_file else (
        Path(__file__).resolve().parent / 'satellite_products.json')
    if not path.is_file():
        print(f"ERROR: products file not found: {path}", file=sys.stderr)
        sys.exit(2)
    blob = json.loads(path.read_text())
    channels = blob.get('mtg') or []
    if not channels:
        print(f"ERROR: no 'mtg' channels listed in {path}", file=sys.stderr)
        sys.exit(2)
    return sorted(set(channels))


def hhmm_to_rc(hhmm):
    """'HHMM' -> 1-indexed repeat cycle. Inverse of rc_to_hhmm."""
    h, m = int(hhmm[:2]), int(hhmm[2:])
    return (h * 60 + m) // NATIVE_CADENCE_MINUTES + 1


def summarize_npy(base_dirs, filter_minutes, channels,
                  start=None, end=None):
    """Scan MTG/<channel>/nc4_<date>-Romania_<channel>/*.npy.

    `base_dirs` is one path or several. The MTG store is the one product
    large enough to outgrow a disk - ~47 MB of .npy per repeat cycle - so
    it can legitimately be split across drives. Several roots are scanned
    as ONE archive: coverage is a property of the data, not of where it
    happens to sit, and reporting per drive would call a date missing
    that is simply on the other one.

    Returns (rows, dates) in the same shape as summarize(), so every
    downstream consumer — the table, the CSV, the missing JSON, the chart —
    works unchanged.
    """
    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]
    bases = [Path(b) for b in base_dirs]

    missing_roots = [b for b in bases if not b.is_dir()]
    if missing_roots:
        for b in missing_roots:
            print(f"ERROR: directory not found: {b}", file=sys.stderr)
        sys.exit(1)

    expected_rcs = expected_rc_set(filter_minutes)
    expected_per_day = len(expected_rcs)

    dates = defaultdict(lambda: {
        'body_files':     0,
        'trailer_files':  0,
        'repeat_cycles':  set(),
        'chunks_seen':    defaultdict(set),   # rc -> set of CHANNELS
    })

    n_files = 0
    per_root: dict[str, int] = {}
    # A (date, rc, channel) seen in two roots is one cycle, not two, so
    # the file tally is deduplicated as well as the cycle set.
    seen: set[tuple[str, int, str]] = set()
    for base in bases:
        root_files = 0
        for channel in channels:
            ch_root = base / channel
            if not ch_root.is_dir():
                print(f"  NOTE: no directory for channel {channel} in "
                      f"{base} — cycles held only there count as "
                      f"incomplete")
                continue
            for day_dir in sorted(ch_root.iterdir()):
                if not day_dir.is_dir():
                    continue
                for filename in list_arrays(day_dir):
                    m = NPY_FILE_PATTERN.match(filename)
                    if not m or m.group('channel') != channel:
                        continue
                    date_str = m.group('date')
                    rc = hhmm_to_rc(m.group('hhmm'))
                    # root_files counts what this store physically
                    # holds; n_files counts the archive once, so the two
                    # differ exactly by what is duplicated across stores.
                    root_files += 1
                    if (date_str, rc, channel) in seen:
                        continue
                    seen.add((date_str, rc, channel))
                    d = dates[date_str]
                    d['body_files'] += 1
                    d['repeat_cycles'].add(rc)
                    d['chunks_seen'][rc].add(channel)
                    n_files += 1
        per_root[str(base)] = root_files

    if len(bases) > 1:
        print(f"Found {n_files} .npy files across {len(channels)} "
              f"channel(s) in {len(bases)} store(s):")
        for root, count in per_root.items():
            print(f"  {count:>9,}  {root}")
        dup = sum(per_root.values()) - n_files
        if dup:
            print(f"  {dup:>9,}  duplicated across stores (counted once)")
    else:
        print(f"Found {n_files} .npy files across {len(channels)} "
              f"channel(s) in {bases[0]}")

    expected_dates = date_range(start, end)
    if expected_dates:
        found = set(dates.keys())
        for date_str in expected_dates:
            dates[date_str]
        empty = [d for d in expected_dates if d not in found]
        outside = sorted(found - set(expected_dates))
        print(f"Expected range : {expected_dates[0]} .. {expected_dates[-1]} "
              f"({len(expected_dates)} dates)")
        if empty:
            print(f"  {len(empty)} date(s) in range have NO files at all "
                  f"— reported as fully missing")
        if outside:
            print(f"  WARNING: {len(outside)} date(s) on disk fall OUTSIDE "
                  f"the range and are still reported: {outside[:5]}"
                  + (" ..." if len(outside) > 5 else ""))

    rows = []
    for date_str in sorted(dates.keys()):
        d = dates[date_str]
        complete = incomplete = 0
        for rc, chans in d['chunks_seen'].items():
            if len(chans) >= len(channels):
                complete += 1
            else:
                incomplete += 1
        present_rcs = d['repeat_cycles']
        on_grid_present = present_rcs & expected_rcs
        off_grid_present = present_rcs - expected_rcs
        coverage_pct = (len(on_grid_present) / expected_per_day * 100
                        if expected_per_day else 0)
        rows.append({
            'date':          date_str,
            'body_files':    d['body_files'],
            'trailer_files': 0,
            'repeat_cycles': len(present_rcs),
            'on_grid':       len(on_grid_present),
            'off_grid':      len(off_grid_present),
            'expected':      expected_per_day,
            # Same keys as the raw scan so the table, CSV and chart are
            # shared. On .npy "complete" means every channel extracted,
            # not two chunks present — see summarize_npy's header note.
            'complete_2chunk':   complete,
            'incomplete_1chunk': incomplete,
            'coverage_pct':      round(coverage_pct, 1),
        })
    return rows, dates


def report_provenance(dates, base_dirs):
    """Print the NMA / Data Store split across the scanned cycles.

    Each store carries its own ledger, so they are merged the same way
    the scan is. Where two stores record the same cycle the first root
    wins - they should agree, and disagreeing would mean the cycle was
    fetched twice from different sources.
    """
    from datastore_fill import load_provenance, provenance_of
    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    blob = {"cycles": {}}
    for base in base_dirs:
        part = load_provenance(base)
        for date_str, cycles in (part.get("cycles") or {}).items():
            merged = blob["cycles"].setdefault(date_str, {})
            for hhmm, src in cycles.items():
                merged.setdefault(hhmm, src)

    if not blob.get("cycles"):
        print("\nProvenance     : no ledger yet - origin is unrecorded for "
              "every cycle.\n                 It is written from now on by "
              "pipeline_msg_mtg.py, whichever --source is used.")
        return {}

    tally = defaultdict(int)
    for date_str, d in dates.items():
        for rc in d["repeat_cycles"]:
            src = provenance_of(blob, date_str, rc_to_hhmm(rc))
            tally[src or "unrecorded"] += 1

    total = sum(tally.values())
    if total:
        parts = ", ".join(f"{k}={v:,} ({v / total * 100:.1f}%)"
                          for k, v in sorted(tally.items()))
        print(f"\nProvenance     : {parts}")
    if tally.get("unrecorded"):
        print("                 'unrecorded' predates the ledger - the two "
              "sources share filenames, so origin is unrecoverable for "
              "those.")
    return dict(tally)


# =============================================================================
# Monthly coverage chart
# =============================================================================
# Lives here rather than in a module of its own: summarize_mtg is the
# canonical summariser the OPERA and lightning reports already mirror, and
# they import `plot_monthly` from it so all three charts stay identical.
# Three copies of this would drift, and the point of the chart is that the
# products can be read side by side.

BAR_COLOR = "#9db8d6"
BAR_ALPHA = 0.45
LINE_COLOR = "#1f4e79"
EXPECTED_COLOR = "#b0b0b0"
SHORTFALL_COLOR = "#c25450"


def monthly_counts(per_date):
    """Collapse {'YYYY-MM-DD': n} into {'YYYY-MM': total}."""
    out = defaultdict(int)
    for date_str, n in per_date.items():
        out[date_str[:7]] += int(n)
    return dict(out)


def plot_monthly(per_date_found, output_path, title,
                 per_date_expected=None, ylabel="files"):
    """Faded bars carrying magnitude, a line through their tops carrying shape.

    `per_date_expected` is drawn as a dashed reference. Without it a month
    that is uniformly half-empty looks identical to a complete one, which
    is the failure this chart exists to make visible.
    """
    if not per_date_found:
        print("  (no dates - chart skipped)")
        return None

    # Agg: these run headless, and on Windows a missing display would
    # otherwise kill the whole summary at its very last step.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    found = monthly_counts(per_date_found)
    expected = monthly_counts(per_date_expected) if per_date_expected else {}
    months = sorted(set(found) | set(expected))
    y_found = [found.get(m, 0) for m in months]
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(max(8, len(months) * 0.62), 4.6))
    if expected:
        ax.bar(x, [expected.get(m, 0) for m in months], color="none",
               edgecolor=EXPECTED_COLOR, linewidth=1.0, linestyle="--",
               zorder=1, label="expected")
    ax.bar(x, y_found, color=BAR_COLOR, alpha=BAR_ALPHA, zorder=2,
           label=ylabel)
    ax.plot(x, y_found, color=LINE_COLOR, linewidth=1.8, marker="o",
            markersize=4, zorder=3)

    if expected:
        for xi, m in enumerate(months):
            exp, got = expected.get(m, 0), found.get(m, 0)
            if exp and got < exp * 0.99:
                ax.annotate(f"-{(1 - got / exp) * 100:.0f}%", (xi, got),
                            textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=7, color=SHORTFALL_COLOR)

    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if expected:
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    total = sum(y_found)
    msg = f"  Chart saved: {out}  ({len(months)} months, {total:,} {ylabel})"
    if expected:
        total_exp = sum(expected.get(m, 0) for m in months)
        if total_exp:
            msg += f", {total / total_exp * 100:.1f}% of expected"
    print(msg)
    return out


def render_chart(rows, output_path):
    """Monthly coverage chart from the per-date rows."""
    found = {r['date']: r['on_grid'] for r in rows}
    expected = {r['date']: r['expected'] for r in rows}
    span = f"{min(found)} .. {max(found)}" if found else ""
    return plot_monthly(found, output_path,
                        title=f"MTG FCI on-grid coverage  {span}",
                        per_date_expected=expected,
                        ylabel="repeat cycles")


def print_table(rows):
    """Print a formatted table to stdout."""
    if not rows:
        print("No data found.")
        return

    header = (
        f"{'Date':<12} {'Body':>6} {'Trail':>6} {'Cycles':>7} "
        f"{'OnGrid':>7} {'OffGrid':>8} {'Exp':>5} "
        f"{'OK(all)':>8} {'Partial':>8} {'Coverage':>9}"
    )
    print(header)
    print("-" * len(header))

    total_body = 0
    total_cycles = 0
    total_on_grid = 0

    for r in rows:
        total_body += r['body_files']
        total_cycles += r['repeat_cycles']
        total_on_grid += r['on_grid']
        print(
            f"{r['date']:<12} {r['body_files']:>6} {r['trailer_files']:>6} "
            f"{r['repeat_cycles']:>7} {r['on_grid']:>7} {r['off_grid']:>8} "
            f"{r['expected']:>5} "
            f"{r['complete_2chunk']:>8} {r['incomplete_1chunk']:>8} "
            f"{r['coverage_pct']:>8.1f}%"
        )

    print("-" * len(header))
    overall_pct = (
        total_on_grid / (rows[0]['expected'] * len(rows)) * 100
        if rows and rows[0]['expected'] else 0
    )
    print(
        f"{'TOTAL':<12} {total_body:>6} {'':>6} "
        f"{total_cycles:>7} {total_on_grid:>7}"
        + ' ' * (8 + 1 + 5 + 1 + 8 + 1 + 8 + 1)
        + f"{overall_pct:>8.1f}%"
    )
    print(f"\n{len(rows)} dates, {total_body} body files, "
          f"{total_cycles} repeat cycles total "
          f"({total_on_grid} on grid)")


def save_csv(rows, output_path):
    """Save summary as CSV."""
    if not rows:
        print("No data to save.")
        return

    fieldnames = [
        'date', 'body_files', 'trailer_files', 'repeat_cycles',
        'on_grid', 'off_grid', 'expected',
        'complete_2chunk', 'incomplete_1chunk', 'coverage_pct',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Summarise downloaded FCI chunk files by date.'
    )
    parser.add_argument(
        '--raw_dir', type=str, default=DEFAULT_RAW_DIR,
        help=f'Path to _raw_chunks directory (default: {DEFAULT_RAW_DIR})',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default=str(PRODUCT_DIR / 'mtg_summary.csv'),
        help=(
            f'Output CSV path. Default lands at the project root '
            f'({PRODUCT_DIR / "mtg_summary.csv"}) so '
            f'intersect_product_coverage.py finds it regardless of '
            f'where the script is invoked from.'
        ),
    )
    parser.add_argument(
        '--missing', '-m', type=str,
        default=str(PRODUCT_DIR / 'mtg_missing_timesteps.json'),
        help=(
            f'Output JSON with missing timesteps '
            f'(default: {PRODUCT_DIR / "mtg_missing_timesteps.json"} - '
            f'anchored to the project root, matching the nwcsaf / opera '
            f'naming convention).'
        ),
    )
    parser.add_argument(
        '--timesteps', type=str, nargs='+', default=None,
        help="Override the cadence minute filter (e.g. 00 10 30 40). "
             "Default: read from timestep_config.json.",
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help="First date the archive is EXPECTED to cover (YYYY-MM-DD). "
             "Dates in range with no files on disk are reported as fully "
             "missing instead of being omitted — which is the only way "
             "--fill-from-datastore can be asked for them. Without this "
             "the range is inferred from the files found, so an entirely "
             "absent date is invisible.",
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help="Last expected date, inclusive (YYYY-MM-DD). See --start.",
    )
    parser.add_argument(
        '--scan', type=str, default='npy',
        choices=['npy', 'raw', 'reprojected'],
        help="Which question to answer. The three views disagree by "
             "design. 'npy' (default) reads the extracted MTG store - "
             "what the Data Store backfill needs, since a cycle absent "
             "there has to be re-fetched. 'reprojected' reads "
             "reprojected_data/satellite_data/MTG - what extract_patches "
             "actually loads, and therefore what the coverage manifest "
             "must gate on: a cycle extracted but never reprojected is "
             "invisible to the store view and would be dropped at build "
             "time without an error. 'raw' reads _raw_chunks/*.nc, the "
             "pre-extraction view: use it to see what arrived but failed "
             "to extract.",
    )
    parser.add_argument(
        '--npy_dir', type=str, nargs='+', default=None, metavar='PATH',
        help="MTG root(s) holding the per-channel .npy directories "
             "(default: the parent of --raw_dir). Several may be given: "
             "the store can span drives, and they are scanned as one "
             "archive so a date held on another drive is not reported "
             "missing. Provenance ledgers are merged the same way.",
    )
    parser.add_argument(
        '--chart', type=str, nargs='?', const=str(PRODUCT_DIR / 'mtg_coverage.png'), default=None,
        help="Render a monthly coverage chart (faded bars + line through "
             "the bar tops, with the cadence expectation as a dashed "
             "reference). Optional PNG path; defaults to "
             "mtg_coverage.png at the project root.",
    )
    args = parser.parse_args()

    if args.timesteps is not None:
        filter_minutes = {int(m) for m in args.timesteps}
        step_src = 'CLI override'
    else:
        filter_minutes, step = load_minute_filter()
        step_src = f'timestep_config.json (step={step} min)'

    print("=" * 70)
    print("MTG Cache Summary")
    print("=" * 70)
    print(f"Raw dir        : {args.raw_dir}")
    print(f"Minute filter  : {sorted(filter_minutes)} ({step_src})")
    print(f"Expected/day   : {len(expected_rc_set(filter_minutes))} "
          f"(of {NATIVE_CYCLES_PER_DAY} native cycles)")
    print()

    if args.npy_dir:
        npy_dirs = args.npy_dir
    elif args.scan == 'reprojected':
        npy_dirs = [str(DEFAULT_REPROJ_DIR)]
    else:
        npy_dirs = [str(Path(args.raw_dir).parent)]
    scans_npy = args.scan in ('npy', 'reprojected')
    channels = expected_channels() if scans_npy else []
    required_parts = len(channels) if scans_npy else 2

    # Say which question is being answered, because the three views
    # disagree by design and the missing-JSON they write means different
    # things.
    print(f"Scanning       : {args.scan}")
    if args.scan == 'reprojected':
        print("                 reprojected arrays - what extract_patches "
              "reads")
        print("                 NOTE: the missing list this writes "
              "describes reprojection")
        print("                 gaps, NOT downloads. Do not feed it to "
              "--source datastore.")
    elif args.scan == 'npy':
        print("                 extracted MTG store - what the Data Store "
              "backfill needs")
    else:
        print("                 raw chunks - what arrived before "
              "extraction")

    def run_scan():
        if scans_npy:
            return summarize_npy(npy_dirs, filter_minutes, channels,
                                 start=args.start, end=args.end)
        return summarize(args.raw_dir, filter_minutes,
                         start=args.start, end=args.end,
                         required_parts=required_parts)

    rows, dates = run_scan()
    print_table(rows)
    save_csv(rows, args.output)
    save_missing_json(dates, filter_minutes, args.missing,
                      required_parts=required_parts)
    report_provenance(dates, npy_dirs)
    if args.chart:
        render_chart(rows, args.chart)
