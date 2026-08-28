"""
periods.py — training-period identity for datasets and model artefacts
======================================================================
Two datasets built over different date ranges must not collide on disk,
and a model used as a frozen feature extractor must not have been trained
on the dates it is about to be fine-tuned over. Both needs reduce to the
same thing: every dataset and every model carries an explicit period, and
periods can be compared.

A period is a label plus inclusive date bounds:

    2025h1  2025-01-01 .. 2025-06-30

The label is what lands in filenames (`build_run_tag` appends it); the
bounds are what gets compared. Bounds live in the sidecar metadata so a
mislabelled run is still detectable — the label is a convenience, the
dates are the truth.

Seasons
-------
Ensemble members are keyed by season, defined in the `[seasons]` block of
training.config as ordered month lists:

    [seasons]
    warm = 4,5,6,7,8,9
    cold = 10,11,12,1,2,3

Order matters and encodes the year wrap: the year advances whenever the
month number decreases, so `10,11,12,1,2,3` anchored at 2025 resolves to
2025-10-01 .. 2026-03-31. Shifting the boundary is a config edit — set
`warm = 5,6,7,8,9,10` / `cold = 11,12,1,2,3,4` and nothing else changes.

Dependency-free (stdlib only) so it can be imported from any script,
including ones that must not pull in TensorFlow.
"""

from __future__ import annotations

import calendar
import configparser
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

DATE_FMT = "%Y-%m-%d"

# Filenames are built from the label, so keep it to characters that are
# safe on NTFS and unambiguous in a tag: lowercase alphanumerics and
# hyphens. Underscores are excluded on purpose - `build_run_tag` joins
# with underscores, and a label containing one makes the resulting tag
# impossible to read back apart by eye.
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

DEFAULT_SEASONS: dict[str, list[int]] = {
    "warm": [4, 5, 6, 7, 8, 9],
    "cold": [10, 11, 12, 1, 2, 3],
}


# =============================================================================
# Period
# =============================================================================

@dataclass(frozen=True)
class Period:
    """A labelled, inclusive date range."""

    label: str
    start: date
    end: date

    def __post_init__(self):
        if not LABEL_RE.match(self.label):
            raise ValueError(
                f"Invalid period label {self.label!r}. Use lowercase "
                f"letters, digits and hyphens (e.g. '2025h1', "
                f"'2025-warm'); no underscores, they collide with the "
                f"artefact tag separator."
            )
        if self.end < self.start:
            raise ValueError(
                f"Period {self.label!r} ends before it starts: "
                f"{self.start} .. {self.end}"
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def parse(cls, label: str, start: str, end: str) -> "Period":
        """Build from CLI strings (`YYYY-MM-DD` bounds, both inclusive)."""
        return cls(label, parse_date(start), parse_date(end))

    @classmethod
    def from_dict(cls, blob: dict | None) -> "Period | None":
        """Rebuild from the `period` block of a metadata/sidecar file.

        Returns None for a missing or empty block, which is how artefacts
        predating period support identify themselves.
        """
        if not blob:
            return None
        return cls(blob["label"], parse_date(blob["start"]),
                   parse_date(blob["end"]))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start.strftime(DATE_FMT),
            "end": self.end.strftime(DATE_FMT),
        }

    # -- queries -----------------------------------------------------------

    @property
    def days(self) -> int:
        """Length in days, both bounds inclusive."""
        return (self.end - self.start).days + 1

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def __str__(self) -> str:
        return f"{self.start} .. {self.end} ({self.label})"


def parse_date(value: str | date) -> date:
    """Parse `YYYY-MM-DD`, raising with the expected format on failure."""
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip(), DATE_FMT).date()
    except ValueError:
        raise ValueError(
            f"Invalid date {value!r}. Expected {DATE_FMT} (e.g. 2025-04-01)."
        )


# =============================================================================
# Artefact tags
# =============================================================================

def data_tag(source: str, period: Period | str | None = None) -> str:
    """Suffix for per-source data files: CSVs, stats, sequence metadata.

    Without a period this returns `source` unchanged, so every artefact
    produced before period support keeps its existing filename and stays
    readable. That backwards compatibility is deliberate - there are
    trained models on disk under the two-part convention.
    """
    if period is None:
        return source
    label = period.label if isinstance(period, Period) else str(period)
    return f"{source}_{label}"


def split_csv_name(split: str, source: str,
                   period: Period | str | None = None) -> str:
    """`train_data_dbscan.csv` / `train_data_dbscan_2025h1.csv`."""
    return f"{split}_data_{data_tag(source, period)}.csv"


def sequence_meta_name(source: str,
                       period: Period | str | None = None) -> str:
    return f"sequence_meta_{data_tag(source, period)}.json"


def normalization_stats_name(source: str,
                             period: Period | str | None = None) -> str:
    return f"normalization_stats_{data_tag(source, period)}.json"


# =============================================================================
# Overlap
# =============================================================================

def overlap(a: Period, b: Period) -> tuple[date, date] | None:
    """Intersection of two periods, or None when they are disjoint."""
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    return (lo, hi) if lo <= hi else None


def overlap_days(a: Period, b: Period) -> int:
    hit = overlap(a, b)
    return 0 if hit is None else (hit[1] - hit[0]).days + 1


def format_overlap_error(feature_extractor: Period, dataset: Period,
                         allow_flag: str = "--allow_period_overlap") -> str:
    """The message raised when a feature extractor has seen these dates."""
    hit = overlap(feature_extractor, dataset)
    n = 0 if hit is None else (hit[1] - hit[0]).days + 1
    return (
        "Feature-extractor period overlaps the dataset period.\n"
        f"  FE trained on : {feature_extractor}\n"
        f"  dataset period: {dataset}\n"
        f"  overlap       : {hit[0]} .. {hit[1]} ({n} days)\n"
        "The frozen backbone has already seen these dates, so any score "
        "measured on them is optimistic.\n"
        f"Rebuild the dataset over a disjoint range, or pass {allow_flag} "
        "to proceed anyway."
    )


def require_no_overlap(feature_extractor: Period | None,
                       dataset: Period | None,
                       allow: bool = False,
                       allow_flag: str = "--allow_period_overlap") -> None:
    """Abort when a feature extractor was trained on the dataset's dates.

    Silent when either side has no declared period - a legacy artefact
    cannot be checked, and failing closed there would lock out every model
    trained before period support existed. The caller is expected to warn
    about the unknown instead.
    """
    if feature_extractor is None or dataset is None:
        return
    if overlap(feature_extractor, dataset) is None:
        return
    msg = format_overlap_error(feature_extractor, dataset, allow_flag)
    if allow:
        print("WARNING: " + msg.replace(
            f"Rebuild the dataset over a disjoint range, or pass "
            f"{allow_flag} to proceed anyway.",
            f"Proceeding anyway because {allow_flag} was passed.",
        ))
        return
    raise SystemExit("ERROR: " + msg)


# =============================================================================
# Seasons
# =============================================================================

def load_seasons(config_path: str | Path | None = None) -> dict[str, list[int]]:
    """Read the `[seasons]` block, falling back to the built-in halves.

    Values are ordered, comma-separated month numbers. Order encodes the
    year wrap - see the module docstring.
    """
    if config_path is None:
        return {k: list(v) for k, v in DEFAULT_SEASONS.items()}

    path = Path(config_path)
    if not path.is_file():
        return {k: list(v) for k, v in DEFAULT_SEASONS.items()}

    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    parser.read(path, encoding="utf-8")
    if not parser.has_section("seasons"):
        return {k: list(v) for k, v in DEFAULT_SEASONS.items()}

    seasons: dict[str, list[int]] = {}
    for label, raw in parser["seasons"].items():
        months = [int(tok) for tok in raw.split(",") if tok.strip()]
        bad = [m for m in months if not 1 <= m <= 12]
        if bad:
            raise ValueError(
                f"[seasons] {label}: month numbers out of range: {bad}"
            )
        if len(set(months)) != len(months):
            raise ValueError(
                f"[seasons] {label}: repeated month(s) in {months}"
            )
        seasons[label] = months

    validate_seasons(seasons)
    return seasons


def validate_seasons(seasons: dict[str, list[int]]) -> list[int]:
    """Check a season definition and report the months it leaves out.

    Overlap is fatal: a month claimed by two seasons makes
    `season_for_date` ambiguous and there is no principled way to choose.

    Incomplete coverage is NOT fatal. Any number of seasons, with any
    names and any month grouping, is a valid configuration - a
    convective-only setup might define a single May-Aug season and intend
    the rest of the year to be excluded. Uncovered months simply map to no
    member, which callers see as a None from `season_for_date`. They are
    returned rather than raised so the caller can report them instead of
    dropping those dates silently.

    Returns:
        Sorted month numbers belonging to no season; empty when the
        definition covers the whole year.
    """
    seen: dict[int, str] = {}
    for label, months in seasons.items():
        for m in months:
            if m in seen:
                raise ValueError(
                    f"[seasons] month {m} claimed by both {seen[m]!r} and "
                    f"{label!r} - seasons must be disjoint."
                )
            seen[m] = label
    return sorted(set(range(1, 13)) - set(seen))


def season_bounds(months: list[int], year: int) -> tuple[date, date]:
    """Resolve an ordered month list anchored at `year` to date bounds.

    The year advances each time the month number decreases, so
    `[10, 11, 12, 1, 2, 3]` at 2025 gives 2025-10-01 .. 2026-03-31.
    """
    if not months:
        raise ValueError("Empty month list has no bounds.")

    resolved: list[tuple[int, int]] = []
    current_year = year
    previous = None
    for month in months:
        if previous is not None and month < previous:
            current_year += 1
        resolved.append((current_year, month))
        previous = month

    first_year, first_month = resolved[0]
    last_year, last_month = resolved[-1]
    return (
        date(first_year, first_month, 1),
        date(last_year, last_month,
             calendar.monthrange(last_year, last_month)[1]),
    )


def season_period(label: str, year: int,
                  seasons: dict[str, list[int]] | None = None) -> Period:
    """Period for one season of one year, labelled `<year><season>`."""
    seasons = seasons or DEFAULT_SEASONS
    if label not in seasons:
        raise ValueError(
            f"Unknown season {label!r}. Known: {sorted(seasons)}"
        )
    start, end = season_bounds(seasons[label], year)
    return Period(f"{year}{label}", start, end)


def season_for_date(day: date | str,
                    seasons: dict[str, list[int]] | None = None) -> str | None:
    """Which season a date belongs to, or None when no season claims it."""
    seasons = seasons or DEFAULT_SEASONS
    month = parse_date(day).month if not isinstance(day, date) else day.month
    for label, months in seasons.items():
        if month in months:
            return label
    return None
