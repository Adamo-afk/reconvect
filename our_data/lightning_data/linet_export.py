"""Bulk LINET stroke data export via the linetview HTTP export endpoint.

Replaces the manual UI workflow (Historic -> date -> Statistics and Data Export
-> Export Stroke data of rectangle -> draw box) with direct GET requests to
/export/lightning.export, discovered in linetview.controller.DataExport (app.js).

Auth is handled automatically: a POST to /functions/doLogin.function with
LVN/LVP form fields yields an LVSESSION cookie, which requests.Session keeps.

Credentials come from either
  * --password_file <path>  (RECOMMENDED - matches the pattern used by
    pipeline_opera.py / pipeline_msg_mtg.py / pipeline_nwcsaf.py) - a
    plain-text file with the username on line 1 and the password on
    line 2, OR
  * the LINET_USER / LINET_PASS environment variables (fallback).

Usage:
    # via credentials file (recommended)
    python our_data/lightning_data/linet_export.py \\
        --start 2024-06-01 --end 2024-07-01 \\
        --password_file linet_credentials.txt

    # via env vars (Windows: `setx` affects only NEW shells, not the shell
    # that ran it - use `$env:LINET_USER=...` in PowerShell for immediate
    # effect in the current session)
    set LINET_USER=...
    set LINET_PASS=...
    python our_data/lightning_data/linet_export.py --start 2024-06-01 --end 2024-07-01

    python our_data/lightning_data/linet_export.py --start 2024-06-15 --end 2024-06-16 \\
        --format kml -pw linet_credentials.txt
    python our_data/lightning_data/linet_export.py --start 2024-06-01 --end 2024-09-01 ^
        --bbox 20.0 43.5 30.0 48.5 --out linet_summer2024 -pw linet_credentials.txt
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Make the project root importable when the script is launched with its
# full path from anywhere (e.g. `python our_data/lightning_data/linet_export.py`
# from the project root, or via an absolute path from a sibling directory).
# We need `c4dl.projection` reachable so DEFAULT_BBOX can auto-track the
# Romania grid extent rather than hardcoding lat/lon numbers.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from c4dl.projection import ROMANIA_GRID_LONLAT_BBOX

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("linet_export")

# ----------------------------- configuration -----------------------------

BASE_URL = "http://192.168.16.211:8080"
LOGIN_PATH = "/functions/doLogin.function"
EXPORT_PATH = "/export/lightning.export"

# Prefer env vars so credentials never end up in the repo/committed file.
USERNAME = os.environ.get("LINET_USER", "")
PASSWORD = os.environ.get("LINET_PASS", "")

# Default bbox auto-derived from the Romania grid: the densified WGS84
# envelope of romania_grid_area["area_extent"] (see c4dl.projection.
# grid_extent_lonlat_bbox). Any future change to the grid definition
# propagates to the LINET download automatically, and the bbox always
# covers the full grid — no strip along any edge gets silently cropped
# at the server side.
DEFAULT_BBOX = ROMANIA_GRID_LONLAT_BBOX  # (min_lon, min_lat, max_lon, max_lat)
DEFAULT_FORMAT = "kml"
# For --format kml, files are written to `{DEFAULT_OUT}/kml_data/{date}/{date}.kml`
# so read_kml_version2.py (which defaults to reading from
# our_data/lightning_data) finds them directly. For txt/asc, files land
# flat under {DEFAULT_OUT}.
DEFAULT_OUT = str(_PROJECT_ROOT / "our_data" / "lightning_data")
REQUEST_TIMEOUT = 300          # export of an active day can be slow
PAUSE_BETWEEN_REQUESTS = 2.0   # be polite to the internal server

# --------------------------------------------------------------------------


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (linet-export-script)",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/linetview/pages/index.html",
        "Origin": BASE_URL,
    })
    return s


def read_credentials(password_file: str) -> tuple[str, str]:
    """Return (username, password) read from a two-line credentials file.

    Same spirit as read_password() in pipeline_opera.py /
    pipeline_msg_mtg.py / pipeline_nwcsaf.py, extended to two lines
    because the LinetView login needs BOTH an LVN (username) and an
    LVP (password) - there is no separate --remote_user CLI flag as in
    the OPERA script.

    File format:
        line 1: username (LVN)
        line 2: password (LVP)
    Blank lines and surrounding whitespace are stripped.
    """
    path = Path(password_file)
    if not path.exists():
        print(f"ERROR: Credentials file not found: {password_file}",
              file=sys.stderr)
        sys.exit(1)
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        print(
            f"ERROR: Credentials file {password_file} must have two non-empty "
            f"lines (username on line 1, password on line 2). "
            f"Got {len(lines)} non-empty line(s).",
            file=sys.stderr,
        )
        sys.exit(1)
    return lines[0], lines[1]


def login(session: requests.Session, username: str, password: str) -> None:
    """POST credentials; server sets the LVSESSION cookie on the session."""
    if not username or not password:
        raise SystemExit(
            "No LINET credentials available.\n"
            "  Either pass --password_file <path> (two lines: username, "
            "password),\n"
            "  or set LINET_USER and LINET_PASS in the CURRENT shell.\n"
            "  On Windows, `setx` only affects NEW shells - use\n"
            "  `$env:LINET_USER = ...` in PowerShell for the current one."
        )
    resp = session.post(
        BASE_URL + LOGIN_PATH,
        data={"LVN": username, "LVP": password},
        timeout=30,
    )
    resp.raise_for_status()
    if "LVSESSION" not in session.cookies.get_dict():
        raise RuntimeError(
            f"Login did not yield an LVSESSION cookie. Response ({resp.status_code}): "
            f"{resp.text[:300]!r}"
        )
    log.info("logged in, LVSESSION acquired")


def build_params(cfg: argparse.Namespace, t0: datetime, t1: datetime) -> dict:
    """Reproduce exactly what linetview.controller.DataExport sends."""
    if cfg.format == "asc":
        layer, type_, filename = "stroke_density", "raster", "lightning.asc"
    else:
        layer, type_, filename = "strokes", cfg.format, f"lightning.{cfg.format}"
    return {
        "bbox": ",".join(f"{v:.4f}" for v in cfg.bbox),
        "from": t0.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": t1.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "unit": "KM",
        "layer": layer,
        "type": type_,
        "filename": filename,
        "lightningType": cfg.lightning_type,
        "ampThreshold": cfg.amp_threshold,
    }


def looks_like_login_page(first_bytes: bytes, content_type: str) -> bool:
    return "text/html" in content_type and b"login" in first_bytes[:2000].lower()


def _target_path_for(cfg: argparse.Namespace, t0: datetime, t1: datetime) -> Path:
    """Where the window's output goes on disk, by --format.

    For --format kml we write to the per-date layout
    `{out}/kml_data/{YYYY-MM-DD}/{YYYY-MM-DD}.kml` — exactly what
    read_kml_version2.py::discover_dates walks — so the arrange step is
    unnecessary for LINET-sourced KMLs. This requires the window to be a
    full UTC calendar day (enforced by make_windows / calendar-day
    iterator when --format kml).

    For --format txt|asc we keep the flat timestamped layout — these are
    analysis exports, not pipeline input, and users often want custom
    time ranges for them.
    """
    if cfg.format == "kml":
        date_str = t0.strftime("%Y-%m-%d")
        return Path(cfg.out) / "kml_data" / date_str / f"{date_str}.kml"
    return Path(cfg.out) / f"linet_{t0:%Y%m%dT%H%M}_{t1:%Y%m%dT%H%M}.{cfg.format}"


def export_window(cfg: argparse.Namespace, session, t0: datetime, t1: datetime) -> str:
    """Fetch (or plan the fetch of) one time window.

    Returns the action taken as a string:
        "downloaded" — file was fetched and saved
        "skipped"    — file already existed non-empty and --force not set
        "planned"    — --dry-run; nothing touched
        "empty"      — server returned zero bytes (no strokes)
    """
    out_file = _target_path_for(cfg, t0, t1)
    already = out_file.exists() and out_file.stat().st_size > 0

    if cfg.dry_run:
        marker = "SKIP (exists)" if already and not cfg.force else "DOWNLOAD"
        log.info("[DRY RUN] %-14s %s <- %s .. %s",
                 marker, out_file, t0.isoformat(), t1.isoformat())
        return "planned"

    if already and not cfg.force:
        log.info("skip (exists): %s", out_file)
        return "skipped"

    params = build_params(cfg, t0, t1)
    resp = session.get(BASE_URL + EXPORT_PATH, params=params,
                       timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    if looks_like_login_page(resp.content, resp.headers.get("Content-Type", "")):
        raise SessionExpired("server returned the login page")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(resp.content)

    size = out_file.stat().st_size
    log.info("saved %s (%.1f KB)", out_file, size / 1024)
    if size == 0:
        log.warning("empty file for %s - %s (no strokes in window, or check params)",
                    t0.isoformat(), t1.isoformat())
        return "empty"
    return "downloaded"


class SessionExpired(RuntimeError):
    pass


def daterange_24h(start: datetime, end: datetime):
    """Continuous <=24h chunks covering [start, end)."""
    t = start
    while t < end:
        yield t, min(t + timedelta(hours=24), end)
        t += timedelta(hours=24)


def daterange_calendar_days(start: datetime, end: datetime):
    """Iterate full UTC calendar days covering [start, end).

    Each yielded (t0, t1) is a full day: t0 = day 00:00 UTC, t1 = next
    day 00:00 UTC. Mid-day start/end are snapped OUTWARD so the range
    covers at least everything the user asked for — necessary because
    the per-date output layout must represent complete calendar days
    (a `kml_data/2026-08-01/2026-08-01.kml` file that only holds half
    a day's strokes would silently mislead downstream tooling).
    """
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_snapped = end
    end_midnight = end.replace(hour=0, minute=0, second=0, microsecond=0)
    if end != end_midnight:
        end_snapped = end_midnight + timedelta(days=1)
    while day < end_snapped:
        yield day, day + timedelta(days=1)
        day += timedelta(days=1)


def daterange_daily_window(start: datetime, end: datetime, window):
    """One window per calendar day: [day HH0:MM0, day HH1:MM1], clipped to [start, end)."""
    (h0, m0), (h1, m1) = window
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        t0 = day.replace(hour=h0, minute=m0)
        t1 = day.replace(hour=h1, minute=m1)
        t0, t1 = max(t0, start), min(t1, end)
        if t0 < t1:
            yield t0, t1
        day += timedelta(days=1)


def make_windows(cfg):
    """Dispatch the (start, end) range into (t0, t1) chunks.

    kml + no --daily-window -> full UTC calendar days (matches the
        per-date output layout expected by read_kml_version2.py). Mid-day
        --start/--end are snapped outward to full-day boundaries.
    kml + --daily-window     -> rejected upstream in parse_args (they
        cannot both be set; a per-date file must be a full day).
    txt/asc + --daily-window -> per-day HH:MM slice, one file per day.
    txt/asc + no --daily-window -> rolling 24h chunks anchored to start.
    """
    if cfg.daily_window is not None:
        return daterange_daily_window(cfg.start, cfg.end, cfg.daily_window)
    if cfg.format == "kml":
        return daterange_calendar_days(cfg.start, cfg.end)
    return daterange_24h(cfg.start, cfg.end)


def parse_utc_date(s: str) -> datetime:
    """Accept YYYY-MM-DD or full ISO (2024-06-01T12:00); interpreted as UTC."""
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk LINET stroke export via the linetview HTTP endpoint.")
    p.add_argument("--start", required=True, type=parse_utc_date,
                   help="start of period, UTC (YYYY-MM-DD or ISO datetime), inclusive")
    p.add_argument("--end", required=True, type=parse_utc_date,
                   help="end of period, UTC, exclusive")
    p.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX),
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                   help="rectangle in EPSG:4326 (default: densified WGS84 "
                        "envelope of the Romania grid, computed from "
                        "c4dl.projection.romania_grid_area)")
    p.add_argument("--format", choices=["txt", "kml", "asc"], default=DEFAULT_FORMAT,
                   help="asc = stroke density raster instead of point list")
    p.add_argument("--lightning-type", type=int, choices=[0, 1, 2], default=0,
                   help="0=all, 1/2=CG/IC only (matches UI checkboxes)")
    p.add_argument("--amp-threshold", type=int, default=0,
                   help="minimum amplitude in kA, 0=disabled")
    p.add_argument("--daily-window", nargs=2, metavar=("HH:MM", "HH:MM"), default=None,
                   help="restrict each day to this UTC time-of-day window, e.g. "
                        "--daily-window 00:00 23:55; without it, days are chunked "
                        "continuously (full 24h)")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"output root (default: {DEFAULT_OUT}). For --format kml, "
                        f"files land at {{out}}/kml_data/YYYY-MM-DD/YYYY-MM-DD.kml "
                        f"— the layout read_kml_version2.py reads directly. "
                        f"For txt/asc, files land flat at "
                        f"{{out}}/linet_*.{{ext}}.")
    p.add_argument("--password_file", "-pw", type=str, default=None,
                   help="Path to a two-line text file with LinetView credentials "
                        "(line 1: username, line 2: password). If omitted, falls "
                        "back to the LINET_USER / LINET_PASS environment variables. "
                        "Matches the --password_file pattern used by pipeline_opera.py "
                        "and pipeline_msg_mtg.py.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-window plan (DOWNLOAD vs SKIP marker + target "
                        "path) without contacting the server. Credentials are not "
                        "needed. Useful before a big multi-month run.")
    p.add_argument("--force", action="store_true",
                   help="Re-download and overwrite even if the destination file "
                        "already exists and is non-empty. Default behaviour is "
                        "resume-safe skip so an interrupted batch can be re-launched.")
    p.add_argument("--pause", type=float, default=PAUSE_BETWEEN_REQUESTS,
                   help="seconds between requests")
    args = p.parse_args()
    if args.end <= args.start:
        p.error("--end must be after --start")
    if args.daily_window is not None:
        try:
            h0, m0 = map(int, args.daily_window[0].split(":"))
            h1, m1 = map(int, args.daily_window[1].split(":"))
            args.daily_window = ((h0, m0), (h1, m1))
        except ValueError:
            p.error("--daily-window expects two HH:MM values")
        if (h1, m1) <= (h0, m0):
            p.error("--daily-window end must be after start (windows may not cross midnight)")
        if args.format == "kml":
            p.error("--daily-window is incompatible with --format kml: the per-date "
                    "output layout requires each file to hold a full UTC calendar day. "
                    "Use --format txt/asc for partial-day slices.")
    # Warn when --format kml gets mid-day boundaries — the range will be
    # snapped outward by daterange_calendar_days to cover full days.
    if args.format == "kml":
        midnight_start = args.start.replace(hour=0, minute=0, second=0, microsecond=0)
        midnight_end = args.end.replace(hour=0, minute=0, second=0, microsecond=0)
        if args.start != midnight_start or args.end != midnight_end:
            snapped_start = midnight_start
            snapped_end = midnight_end if args.end == midnight_end \
                else midnight_end + timedelta(days=1)
            log.warning(
                "--format kml requires full UTC calendar days; snapping range "
                "%s .. %s outward to %s .. %s",
                args.start.isoformat(), args.end.isoformat(),
                snapped_start.isoformat(), snapped_end.isoformat(),
            )
    return args


def main():
    cfg = parse_args()

    # Credentials: --password_file wins if given, otherwise env vars.
    # Skipped entirely under --dry-run (we don't hit the server).
    session = None
    username = password = ""
    if not cfg.dry_run:
        if cfg.password_file:
            username, password = read_credentials(cfg.password_file)
            log.info("loaded credentials from %s", cfg.password_file)
        else:
            username, password = USERNAME, PASSWORD
        session = make_session()
        login(session, username, password)

    tally = {"downloaded": 0, "skipped": 0, "empty": 0, "planned": 0}
    failed: list[tuple[datetime, datetime]] = []

    for t0, t1 in make_windows(cfg):
        try:
            action = export_window(cfg, session, t0, t1)
            tally[action] = tally.get(action, 0) + 1
        except SessionExpired:
            log.info("session expired mid-batch, re-authenticating")
            login(session, username, password)
            try:
                action = export_window(cfg, session, t0, t1)
                tally[action] = tally.get(action, 0) + 1
            except Exception as exc:  # noqa: BLE001
                log.error("FAILED after re-login %s -> %s: %s",
                          t0.isoformat(), t1.isoformat(), exc)
                failed.append((t0, t1))
        except Exception as exc:  # noqa: BLE001 - log and continue the batch
            log.error("FAILED %s -> %s: %s", t0.isoformat(), t1.isoformat(), exc)
            failed.append((t0, t1))
        # No point pausing in dry-run (no server contact).
        if not cfg.dry_run:
            time.sleep(cfg.pause)

    log.info("=" * 70)
    log.info("Summary%s:", " (DRY RUN)" if cfg.dry_run else "")
    if cfg.dry_run:
        log.info("  Planned    : %d", tally["planned"])
    else:
        log.info("  Downloaded : %d", tally["downloaded"])
        log.info("  Skipped    : %d (already present; use --force to overwrite)",
                 tally["skipped"])
        log.info("  Empty      : %d (server returned 0 bytes)", tally["empty"])
    log.info("  Failed     : %d", len(failed))
    for t0, t1 in failed:
        log.info("    retry manually: %s -> %s", t0.isoformat(), t1.isoformat())


if __name__ == "__main__":
    main()