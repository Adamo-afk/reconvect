"""
datastore_fill.py — backfill MTG FCI gaps from the EUMETSAT Data Store
=====================================================================
The primary MTG source is the NMA internal server (see pipeline_msg_mtg.py).
When a repeat cycle never arrived there — or arrived with only one of the two
Romania chunks — this module fetches the shortfall from the EUMETSAT Data
Store over EUMDAC and drops it into the same `_raw_chunks/` directory.

Why this works with no translation layer: the Data Store delivers products
under the native FCI naming convention

    ..._OPE_<sensing_start>_<sensing_end>_..._<repeat_cycle>_<chunk>.nc

which is exactly what `summarize_mtg.parse_fci_filename` already reads, and
`_raw_chunks/` is a flat directory. So a re-run of the summary picks up the
backfilled files automatically.

Scope
-----
Only FDHSI (EO:EUM:DAT:0662) is fetched. It carries all five channels this
project uses at the resolutions it uses them (vis_06 at 1 km; ir_38, ir_105,
wv_63, wv_73 at 2 km). HRFI would only add vis_06 at 500 m, which the
pipeline pools away.

Both kinds of shortfall are filled:
  * `missing_times`     — repeat cycle absent entirely (both chunks)
  * `incomplete_times`  — only one of the two Romania chunks present

Credentials
-----------
EUMDAC needs a key and a secret (https://api.eumetsat.int/api-key/). Supply
either of:
  * `--eumdac_credentials PATH` — two-line text file: key on line 1,
    secret on line 2. Mirrors the `--password_file` convention used by
    pipeline_msg_mtg.py / pipeline_opera.py.
  * environment variables `EUMDAC_KEY` and `EUMDAC_SECRET`.

`eumdac` is an optional dependency; it is imported lazily so the summary
script runs normally when the backfill is not requested.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# FDHSI — full-disk high spectral resolution imagery. See module docstring
# for why HRFI is deliberately excluded.
FDHSI_COLLECTION_ID = "EO:EUM:DAT:0662"

# The two FCI chunks covering the Romania grid. Kept in sync with
# pipeline_msg_mtg.ROMANIA_CHUNKS.
ROMANIA_CHUNKS = (35, 36)

# MTG native repeat cycle, in minutes.
NATIVE_CADENCE_MINUTES = 10


# =============================================================================
# Provenance ledger
# =============================================================================
# The NMA server and the Data Store deliver products under the SAME native
# FCI naming convention — deliberately, it is why no renaming step exists
# above. The consequence is that once a cycle is processed to .npy, nothing
# on disk says where it came from.
#
# So origin is recorded at write time by whoever wrote it:
#     pipeline_msg_mtg.py  -> record_provenance(..., "nma")
#     fill_gaps() below    -> record_provenance(..., "datastore")
#
# This matters far more once raw chunks are deleted after processing: the
# .npy files then become the only evidence a cycle exists, and without the
# ledger a coverage report can say WHAT is present but never HOW it got
# there — which is the question you ask when deciding whether a gap is a
# transfer problem or a source problem.
#
# Keyed by date and nominal HH:MM (how summarize_mtg reports), not by
# filename — the filename is precisely the thing carrying no origin.

PROVENANCE_NAME = "provenance.json"
PROVENANCE_SOURCES = ("nma", "datastore")


def provenance_path(base_dir) -> Path:
    """Ledger lives beside the data it describes, in the MTG root."""
    return Path(base_dir) / PROVENANCE_NAME


def load_provenance(base_dir) -> dict:
    path = provenance_path(base_dir)
    if not path.is_file():
        return {"updated_utc": None, "sources": {}, "cycles": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: provenance ledger unreadable ({exc}); starting fresh")
        return {"updated_utc": None, "sources": {}, "cycles": {}}
    blob.setdefault("cycles", {})
    blob.setdefault("sources", {})
    return blob


def record_provenance(base_dir, entries, source: str) -> int:
    """Record (date, 'HH:MM') pairs as having come from `source`.

    Existing entries are overwritten: a cycle backfilled from the Data
    Store after an incomplete NMA transfer genuinely originates from the
    Data Store now, and the ledger should say so.

    Written via a temp file and os.replace so an interrupted run can never
    leave a half-written ledger — the ledger is the only copy of this
    information.
    """
    if source not in PROVENANCE_SOURCES:
        raise ValueError(
            f"unknown source {source!r}; expected {PROVENANCE_SOURCES}")

    entries = list(entries)
    if not entries:
        return 0

    blob = load_provenance(base_dir)
    for date_str, hhmm in entries:
        blob["cycles"].setdefault(date_str, {})[hhmm] = source

    tally: dict[str, int] = {}
    for per_date in blob["cycles"].values():
        for src in per_date.values():
            tally[src] = tally.get(src, 0) + 1
    blob["sources"] = dict(sorted(tally.items()))
    blob["updated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    path = provenance_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, path)
    return len(entries)


def provenance_of(blob: dict, date_str: str, hhmm: str):
    """Origin of one cycle, or None when it predates the ledger."""
    return blob.get("cycles", {}).get(date_str, {}).get(hhmm)


# =============================================================================
# Credentials
# =============================================================================

def load_credentials(credentials_file: str | None = None) -> tuple[str, str]:
    """Return (key, secret) from a two-line file or the environment.

    Raises SystemExit with actionable guidance when neither is available,
    rather than letting EUMDAC fail later with an opaque auth error.
    """
    if credentials_file:
        path = Path(credentials_file)
        if not path.is_file():
            raise SystemExit(f"EUMDAC credentials file not found: {path}")
        lines = [ln.strip() for ln in path.read_text().splitlines()
                 if ln.strip()]
        if len(lines) < 2:
            raise SystemExit(
                f"{path} must hold two non-empty lines: key on line 1, "
                f"secret on line 2."
            )
        return lines[0], lines[1]

    key = os.environ.get("EUMDAC_KEY")
    secret = os.environ.get("EUMDAC_SECRET")
    if key and secret:
        return key, secret

    raise SystemExit(
        "No EUMDAC credentials. Pass --eumdac_credentials PATH (two-line "
        "file: key, then secret) or set EUMDAC_KEY and EUMDAC_SECRET. "
        "Get a key at https://api.eumetsat.int/api-key/"
    )


# =============================================================================
# Gap -> request translation
# =============================================================================

def rc_from_hhmm(hhmm: str) -> int:
    """'HH:MM' -> 1-indexed repeat cycle. Inverse of summarize_mtg.rc_to_hhmm."""
    hours, minutes = (int(p) for p in hhmm.split(":"))
    return (hours * 60 + minutes) // NATIVE_CADENCE_MINUTES + 1


def collect_gaps(missing_json: dict,
                 include_incomplete: bool = True) -> dict[str, list[str]]:
    """Map each date to the sorted list of 'HH:MM' cycles needing a fetch.

    `missing_times` are absent entirely; `incomplete_times` have one of the
    two Romania chunks. Both are unusable downstream, so both are filled by
    default — the extra chunk on an incomplete cycle is skipped at download
    time because the file already exists on disk.
    """
    gaps: dict[str, list[str]] = {}
    for date_str, block in missing_json.get("dates", {}).items():
        wanted = set(block.get("missing_times", []))
        if include_incomplete:
            wanted |= set(block.get("incomplete_times", []))
        if wanted:
            gaps[date_str] = sorted(wanted)
    return gaps


def _chunk_patterns() -> list[str]:
    """fnmatch patterns selecting the Romania chunks of any repeat cycle."""
    return [f"*_????_{chunk:04d}.nc" for chunk in ROMANIA_CHUNKS]


def _rc_of(entry: str) -> int | None:
    """Repeat cycle encoded in an FCI entry filename, or None."""
    m = re.search(r"_(\d{4})_(\d{4})\.nc$", entry)
    return int(m.group(1)) if m else None


# =============================================================================
# Download
# =============================================================================

def fill_gaps(gaps: dict[str, list[str]],
              raw_dir: str,
              credentials_file: str | None = None,
              dry_run: bool = False,
              verbose: bool = True) -> dict:
    """Fetch the chunks backing `gaps` into `raw_dir`.

    Returns a report dict recording what was fetched, suitable for
    embedding in mtg_missing_timesteps.json.
    """
    report = {
        "collection_id": FDHSI_COLLECTION_ID,
        "chunks": list(ROMANIA_CHUNKS),
        "dates_with_gaps": len(gaps),
        "timesteps_requested": sum(len(v) for v in gaps),
        "files_downloaded": 0,
        "files": [],
        "timesteps_filled": {},
        "skipped_already_present": 0,
        "errors": [],
        "dry_run": dry_run,
    }
    if not gaps:
        if verbose:
            print("No gaps to fill — MTG coverage is already complete.")
        return report

    try:
        import eumdac
    except ImportError:
        raise SystemExit(
            "The `eumdac` package is required for --fill-from-datastore.\n"
            "    pip install eumdac"
        )

    key, secret = load_credentials(credentials_file)

    # Exchange key/secret for a bearer token up front. This is the auth
    # smoke test: bad credentials fail here with a clear message instead
    # of surfacing later as an opaque search error. The token is
    # short-lived and eumdac refreshes it automatically mid-run.
    try:
        token = eumdac.AccessToken((key, secret))
        if verbose:
            print(f"EUMDAC token acquired, expires {token.expiration}")
    except Exception as exc:                         # noqa: BLE001
        raise SystemExit(
            f"EUMDAC authentication failed: {exc}\n"
            f"Check the consumer key and secret "
            f"(https://api.eumetsat.int/api-key/)."
        )

    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(FDHSI_COLLECTION_ID)

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    patterns = _chunk_patterns()

    for date_str in sorted(gaps):
        wanted_rcs = {rc_from_hhmm(t) for t in gaps[date_str]}
        day = datetime.strptime(date_str, "%Y-%m-%d")
        if verbose:
            print(f"\n[{date_str}] {len(wanted_rcs)} cycle(s) to recover")

        try:
            products = collection.search(dtstart=day,
                                         dtend=day + timedelta(days=1))
        except Exception as exc:                     # noqa: BLE001
            msg = f"{date_str}: search failed: {exc}"
            report["errors"].append(msg)
            print(f"  ERROR {msg}", file=sys.stderr)
            continue

        filled_here: list[str] = []
        for product in products:
            try:
                entries = list(product.entries)
            except Exception as exc:                 # noqa: BLE001
                report["errors"].append(f"{date_str}: entries failed: {exc}")
                continue

            for entry in entries:
                if not any(fnmatch.fnmatch(entry, p) for p in patterns):
                    continue
                rc = _rc_of(entry)
                if rc is None or rc not in wanted_rcs:
                    continue

                out_file = raw_path / Path(entry).name
                if out_file.exists():
                    report["skipped_already_present"] += 1
                    continue
                if dry_run:
                    report["files"].append(out_file.name)
                    report["files_downloaded"] += 1
                    filled_here.append(out_file.name)
                    if verbose:
                        print(f"  [dry-run] would fetch {out_file.name}")
                    continue

                try:
                    with product.open(entry=entry) as src, \
                            open(out_file, "wb") as dst:
                        dst.write(src.read())
                except Exception as exc:             # noqa: BLE001
                    msg = f"{entry}: {exc}"
                    report["errors"].append(msg)
                    print(f"  ERROR {msg}", file=sys.stderr)
                    if out_file.exists():
                        out_file.unlink()            # no truncated files
                    continue

                report["files"].append(out_file.name)
                report["files_downloaded"] += 1
                filled_here.append(out_file.name)
                if verbose:
                    print(f"  fetched {out_file.name}")

        if filled_here:
            report["timesteps_filled"][date_str] = sorted(
                {t for t in gaps[date_str]
                 if any(_rc_of(f) == rc_from_hhmm(t) for f in filled_here)}
            )

    # Record origin while it is still known. After this returns, nothing on
    # disk distinguishes these files from NMA ones.
    if not dry_run and report["timesteps_filled"]:
        try:
            pairs = [(d, t) for d, times in report["timesteps_filled"].items()
                     for t in times]
            n = record_provenance(raw_path.parent, pairs, "datastore")
            report["provenance_recorded"] = n
            if verbose:
                print(f"\nProvenance: {n} cycle(s) recorded as 'datastore'")
        except Exception as exc:                         # noqa: BLE001
            # A ledger failure must not discard a download that succeeded.
            report["errors"].append(f"provenance ledger: {exc}")
            print(f"WARNING: could not record provenance: {exc}",
                  file=sys.stderr)

    return report


def print_report(report: dict) -> None:
    """Console summary of a fill run."""
    print("\n" + "=" * 70)
    print("EUMETSAT Data Store backfill")
    print("=" * 70)
    print(f"Collection        : {report['collection_id']} (FDHSI)")
    print(f"Chunks            : {report['chunks']}")
    print(f"Dates with gaps   : {report['dates_with_gaps']}")
    print(f"Cycles requested  : {report['timesteps_requested']}")
    print(f"Files downloaded  : {report['files_downloaded']}"
          + ("  [DRY RUN]" if report["dry_run"] else ""))
    if report["skipped_already_present"]:
        print(f"Already on disk   : {report['skipped_already_present']}")
    if report["errors"]:
        print(f"Errors            : {len(report['errors'])}")
    if report["files"]:
        print("\nFiles obtained from the Data Store:")
        for name in report["files"]:
            print(f"  {name}")
