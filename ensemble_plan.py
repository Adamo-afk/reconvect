"""
ensemble_plan.py — declare, register and verify the seasonal ensemble
=====================================================================
The ensemble is a set of models, one per season per year, each trained on
its own disjoint slice of the archive. Which members exist is not a free
choice: it falls out of the `[seasons]` block in training.config crossed
with the years actually present in `patch_index.csv`.

That plan is declared once, at dataset-creation time:

    python create_datasets.py --mode <mode> --ensemble

which prints the member count, each member's period and how much data
backs it, flags any overlap between members, and appends the result to
`our_data/ensemble_registry.json`. Nothing is built by that call.

Members are then built one at a time:

    python create_datasets.py --mode <mode> --period 2025warm

and training reads the registry's most recent state to learn what the
ensemble is supposed to contain, then checks which of those datasets
actually exist on disk before it starts.

Why a registry rather than re-deriving the plan each time: the plan
depends on the season config AND on how much data was downloaded at the
moment it was drawn. Both drift. Recording states makes the ensemble
reproducible - a validation run months later resolves the same members it
did the day the plan was registered, unless a new state was deliberately
appended.

Stdlib only, like periods.py.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from periods import (
    Period,
    load_seasons,
    season_period,
    validate_seasons,
)

REGISTRY_NAME = "ensemble_registry.json"
REGISTRY_VERSION = 1

# A member backed by fewer than this fraction of its calendar days is
# reported as `partial`. It is still buildable - the threshold only drives
# how loudly the plan talks about it.
PARTIAL_THRESHOLD = 0.90


# =============================================================================
# Members
# =============================================================================

@dataclass
class Member:
    """One ensemble member: a season of a year, plus how much data backs it."""

    period: Period
    season: str
    year: int
    dates_available: int = 0
    timesteps_available: int = 0

    @property
    def label(self) -> str:
        return self.period.label

    @property
    def expected_days(self) -> int:
        return self.period.days

    @property
    def coverage(self) -> float:
        """Fraction of the period's calendar days that have any data."""
        if not self.expected_days:
            return 0.0
        return self.dates_available / self.expected_days

    @property
    def status(self) -> str:
        if self.dates_available == 0:
            return "no-data"
        return "partial" if self.coverage < PARTIAL_THRESHOLD else "ok"

    @property
    def buildable(self) -> bool:
        return self.dates_available > 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "season": self.season,
            "year": self.year,
            "start": self.period.start.strftime("%Y-%m-%d"),
            "end": self.period.end.strftime("%Y-%m-%d"),
            "dates_available": self.dates_available,
            "timesteps_available": self.timesteps_available,
            "expected_days": self.expected_days,
            "coverage_pct": round(self.coverage * 100, 1),
            "status": self.status,
        }


# =============================================================================
# Available data
# =============================================================================

def load_index_dates(patch_index_csv: str | Path) -> dict[str, int]:
    """Map each date in patch_index.csv to its number of timesteps.

    This is the ground truth for what the ensemble can be built over -
    identify_patches.py only emits rows for timesteps that survived the
    DBSCAN activity gate, so a date absent here cannot back a member.
    """
    path = Path(patch_index_csv)
    if not path.is_file():
        raise SystemExit(
            f"Patch index not found: {path}\n"
            f"Run identify_patches.py first - the ensemble plan is derived "
            f"from the dates it emits."
        )

    counts: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            raise SystemExit(f"{path} has no `date` column.")
        for row in reader:
            day = (row.get("date") or "").strip()
            if day:
                counts[day] = counts.get(day, 0) + 1
    return counts


# =============================================================================
# Plan
# =============================================================================

@dataclass
class Plan:
    """A full ensemble plan: its members plus everything worth reporting."""

    members: list[Member]
    seasons: dict[str, list[int]]
    uncovered_months: list[int] = field(default_factory=list)
    overlaps: list[dict] = field(default_factory=list)
    dates_outside: int = 0

    @property
    def buildable(self) -> list[Member]:
        return [m for m in self.members if m.buildable]

    def member(self, label: str) -> Member | None:
        for m in self.members:
            if m.label == label:
                return m
        return None


def enumerate_members(date_counts: dict[str, int],
                      seasons: dict[str, list[int]]) -> Plan:
    """Cross the season definitions with the years present in the data.

    A member is emitted for every (season, year) whose period intersects
    the archive's span, including ones with no data - a member that cannot
    be built is worth seeing in the plan rather than silently absent.
    """
    uncovered = validate_seasons(seasons)

    if not date_counts:
        return Plan(members=[], seasons=seasons, uncovered_months=uncovered)

    days = sorted(date_counts)
    first = datetime.strptime(days[0], "%Y-%m-%d").date()
    last = datetime.strptime(days[-1], "%Y-%m-%d").date()

    # A season anchored at year Y can run into Y+1, so start one year back
    # to catch a cold season that began before the archive does.
    members: list[Member] = []
    for year in range(first.year - 1, last.year + 2):
        for season in seasons:
            period = season_period(season, year, seasons)
            if period.end < first or period.start > last:
                continue
            members.append(Member(period=period, season=season, year=year))

    # Attribute each date to its member.
    claimed = 0
    for day_str, n_steps in date_counts.items():
        day = datetime.strptime(day_str, "%Y-%m-%d").date()
        for m in members:
            if m.period.contains(day):
                m.dates_available += 1
                m.timesteps_available += n_steps
                claimed += 1
                break

    members.sort(key=lambda m: (m.period.start, m.label))

    return Plan(
        members=members,
        seasons=seasons,
        uncovered_months=uncovered,
        overlaps=find_overlaps(members),
        dates_outside=len(date_counts) - claimed,
    )


def find_overlaps(members: list[Member]) -> list[dict]:
    """Every pair of members sharing a date.

    Disjoint seasons should make this impossible, but the definitions come
    from a hand-edited config and a member built over dates another member
    also trained on is exactly the leak the period machinery exists to
    prevent. Cheap to check, so it is checked.
    """
    from periods import overlap  # local import keeps the module import flat

    found: list[dict] = []
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            hit = overlap(a.period, b.period)
            if hit is None:
                continue
            found.append({
                "a": a.label,
                "b": b.label,
                "start": hit[0].strftime("%Y-%m-%d"),
                "end": hit[1].strftime("%Y-%m-%d"),
                "days": (hit[1] - hit[0]).days + 1,
            })
    return found


# =============================================================================
# Reporting
# =============================================================================

def format_plan(plan: Plan, mode: str | None = None,
                source: str | None = None) -> str:
    """The human-facing plan report printed by --ensemble."""
    lines: list[str] = []
    lines.append("=" * 70)
    title = "Ensemble plan"
    if mode:
        title += f" — mode {mode}"
    if source:
        title += f", source {source}"
    lines.append(title)
    lines.append("=" * 70)

    seasons_desc = ", ".join(
        f"{name} [{','.join(str(m) for m in months)}]"
        for name, months in plan.seasons.items()
    )
    lines.append(f"Seasons        : {seasons_desc}")

    buildable = plan.buildable
    lines.append(
        f"Members        : {len(plan.members)} planned, "
        f"{len(buildable)} buildable"
    )
    lines.append("")

    if not plan.members:
        lines.append("  (no members — patch_index.csv is empty)")
        return "\n".join(lines)

    # Season names are user-defined, so the label column is sized to the
    # widest one rather than a fixed guess.
    width = max(12, max(len(m.label) for m in plan.members))
    header = (f"  {'label':<{width}} {'period':<25} {'dates':>9} "
              f"{'coverage':>9}  status")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for m in plan.members:
        span = f"{m.period.start} .. {m.period.end}"
        cov = f"{m.coverage * 100:.1f}%" if m.expected_days else "n/a"
        flag = {"ok": "ok", "partial": "PARTIAL", "no-data": "NO DATA"}[m.status]
        counts = f"{m.dates_available}/{m.expected_days}"
        lines.append(
            f"  {m.label:<{width}} {span:<25} {counts:>9} {cov:>9}  {flag}"
        )
    lines.append("")

    # -- discrepancies -----------------------------------------------------
    problems = False

    empty = [m.label for m in plan.members if m.status == "no-data"]
    if empty:
        problems = True
        lines.append(
            f"  NO DATA        : {', '.join(empty)} — planned but not "
            f"buildable; download the dates or ignore the member."
        )

    partial = [m for m in plan.members if m.status == "partial"]
    if partial:
        problems = True
        for m in partial:
            missing = m.expected_days - m.dates_available
            lines.append(
                f"  PARTIAL        : {m.label} is missing {missing} of "
                f"{m.expected_days} days ({m.coverage * 100:.1f}% present)."
            )

    if plan.uncovered_months:
        problems = True
        lines.append(
            f"  UNCLAIMED      : months {plan.uncovered_months} belong to no "
            f"season, so their dates join no member."
        )

    if plan.dates_outside:
        problems = True
        lines.append(
            f"  EXCLUDED       : {plan.dates_outside} date(s) in the index "
            f"fall outside every member period."
        )

    if plan.overlaps:
        problems = True
        for o in plan.overlaps:
            lines.append(
                f"  OVERLAP        : {o['a']} and {o['b']} share "
                f"{o['days']} day(s) ({o['start']} .. {o['end']}) — members "
                f"must be disjoint; fix [seasons]."
            )
    else:
        lines.append("  Overlaps       : none")

    if not problems:
        lines.append("  No discrepancies.")

    return "\n".join(lines)


# =============================================================================
# Registry
# =============================================================================

def registry_path(data_root: str | Path) -> Path:
    return Path(data_root) / REGISTRY_NAME


def load_registry(data_root: str | Path) -> dict:
    path = registry_path(data_root)
    if not path.is_file():
        return {"registry_version": REGISTRY_VERSION, "states": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_state(data_root: str | Path, plan: Plan,
                 mode: str | None = None, source: str | None = None) -> dict:
    """Append this plan as the registry's newest state and return it.

    Append-only on purpose: an earlier ensemble stays reconstructable, so
    a result produced against a previous plan can still be explained.
    """
    registry = load_registry(data_root)
    state = {
        "registered_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "registered_by": {"mode": mode, "source": source},
        "seasons": {k: list(v) for k, v in plan.seasons.items()},
        "uncovered_months": plan.uncovered_months,
        "n_members": len(plan.members),
        "n_buildable": len(plan.buildable),
        "members": [m.to_dict() for m in plan.members],
        "overlaps": plan.overlaps,
        "dates_outside": plan.dates_outside,
    }
    registry.setdefault("states", []).append(state)
    registry["registry_version"] = REGISTRY_VERSION

    path = registry_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    return state


def load_last_state(data_root: str | Path) -> dict | None:
    """The most recently registered plan, or None when none exists."""
    states = load_registry(data_root).get("states", [])
    return states[-1] if states else None


def require_last_state(data_root: str | Path) -> dict:
    """Like load_last_state, but abort with instructions when unregistered."""
    state = load_last_state(data_root)
    if state is None:
        raise SystemExit(
            f"No ensemble plan registered in "
            f"{registry_path(data_root)}.\n"
            f"Register one first:\n"
            f"    python create_datasets.py --mode <mode> --ensemble"
        )
    return state


def state_member(state: dict, label: str) -> dict | None:
    for m in state.get("members", []):
        if m.get("label") == label:
            return m
    return None


def state_period(state: dict, label: str) -> Period | None:
    """Rebuild a member's Period from a registered state."""
    blob = state_member(state, label)
    if blob is None:
        return None
    return Period.parse(blob["label"], blob["start"], blob["end"])


def check_member_datasets(state: dict, mode: str, source: str,
                          datasets_root: str | Path) -> dict:
    """Which of the registered members have a dataset built for this mode.

    Called at training time: the registry says what the ensemble should
    contain, this says what exists.

    Returns a dict with `present` / `missing` / `not_buildable` lists of
    member labels, plus the resolved directory for each present member.
    """
    from train_models import build_run_tag  # late import: pulls TensorFlow

    root = Path(datasets_root)
    present: list[str] = []
    missing: list[str] = []
    not_buildable: list[str] = []
    dirs: dict[str, str] = {}

    for blob in state.get("members", []):
        label = blob["label"]
        if blob.get("status") == "no-data":
            not_buildable.append(label)
            continue
        run_tag = build_run_tag(mode, source, label)
        candidate = root / run_tag
        if (candidate / "train").is_dir():
            present.append(label)
            dirs[label] = str(candidate)
        else:
            missing.append(label)

    return {
        "present": present,
        "missing": missing,
        "not_buildable": not_buildable,
        "dirs": dirs,
    }


def format_dataset_check(check: dict, mode: str, source: str) -> str:
    """Report for the training-time availability check."""
    lines = [f"Ensemble datasets for mode {mode} (source {source}):"]
    if check["present"]:
        lines.append(f"  built       : {', '.join(check['present'])}")
    else:
        lines.append("  built       : none")
    if check["missing"]:
        lines.append(f"  MISSING     : {', '.join(check['missing'])}")
        lines.append("                build them with "
                     "`create_datasets.py --mode <mode> --period <label>`")
    if check["not_buildable"]:
        lines.append(
            f"  no data     : {', '.join(check['not_buildable'])} "
            f"(registered but unbuildable)"
        )
    return "\n".join(lines)
