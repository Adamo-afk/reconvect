"""Bulk LINET stroke data export via the linetview HTTP export endpoint.

Replaces the manual UI workflow (Historic -> date -> Statistics and Data Export
-> Export Stroke data of rectangle -> draw box) with direct GET requests to
/export/lightning.export, discovered in linetview.controller.DataExport (app.js).

Auth is handled automatically: a POST to /functions/doLogin.function with
LVN/LVP form fields yields an LVSESSION cookie, which requests.Session keeps.

Usage:
    set LINET_USER=...
    set LINET_PASS=...
    python linet_export.py --start 2024-06-01 --end 2024-07-01
    python linet_export.py --start 2024-06-15 --end 2024-06-16 --format kml
    python linet_export.py --start 2024-06-01 --end 2024-09-01 ^
        --bbox 20.0 43.5 30.0 48.5 --out linet_summer2024
"""

import argparse
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("linet_export")

# ----------------------------- configuration -----------------------------

BASE_URL = "http://192.168.16.211:8080"
LOGIN_PATH = "/functions/doLogin.function"
EXPORT_PATH = "/export/lightning.export"

# Prefer env vars so credentials never end up in the repo/committed file.
USERNAME = os.environ.get("LINET_USER", "")
PASSWORD = os.environ.get("LINET_PASS", "")

# Defaults (all overridable from the CLI). Bbox is EPSG:4326: Romania + margin.
DEFAULT_BBOX = (20.0, 43.5, 30.0, 48.5)  # min_lon min_lat max_lon max_lat
DEFAULT_FORMAT = "txt"
DEFAULT_OUT = "linet_exports"
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


def login(session: requests.Session) -> None:
    """POST credentials; server sets the LVSESSION cookie on the session."""
    if not USERNAME or not PASSWORD:
        raise SystemExit("Set LINET_USER and LINET_PASS environment variables first.")
    resp = session.post(
        BASE_URL + LOGIN_PATH,
        data={"LVN": USERNAME, "LVP": PASSWORD},
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


def export_window(cfg: argparse.Namespace, session: requests.Session,
                  t0: datetime, t1: datetime) -> Path:
    params = build_params(cfg, t0, t1)
    out_file = Path(cfg.out) / f"linet_{t0:%Y%m%dT%H%M}_{t1:%Y%m%dT%H%M}.{cfg.format}"
    if out_file.exists() and out_file.stat().st_size > 0:
        log.info("skip (exists): %s", out_file.name)
        return out_file

    resp = session.get(BASE_URL + EXPORT_PATH, params=params,
                       timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    if looks_like_login_page(resp.content, resp.headers.get("Content-Type", "")):
        raise SessionExpired("server returned the login page")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(resp.content)

    size = out_file.stat().st_size
    log.info("saved %s (%.1f KB)", out_file.name, size / 1024)
    if size == 0:
        log.warning("empty file for %s - %s (no strokes in window, or check params)",
                    t0.isoformat(), t1.isoformat())
    return out_file


class SessionExpired(RuntimeError):
    pass


def daterange_24h(start: datetime, end: datetime):
    """Continuous <=24h chunks covering [start, end)."""
    t = start
    while t < end:
        yield t, min(t + timedelta(hours=24), end)
        t += timedelta(hours=24)


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
    if cfg.daily_window is not None:
        return daterange_daily_window(cfg.start, cfg.end, cfg.daily_window)
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
                   help="rectangle in EPSG:4326 (default: Romania + margin)")
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
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
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
    return args


def main():
    cfg = parse_args()
    session = make_session()
    login(session)

    ok, failed = 0, []
    for t0, t1 in make_windows(cfg):
        try:
            export_window(cfg, session, t0, t1)
            ok += 1
        except SessionExpired:
            log.info("session expired mid-batch, re-authenticating")
            login(session)
            try:
                export_window(cfg, session, t0, t1)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                log.error("FAILED after re-login %s -> %s: %s",
                          t0.isoformat(), t1.isoformat(), exc)
                failed.append((t0, t1))
        except Exception as exc:  # noqa: BLE001 - log and continue the batch
            log.error("FAILED %s -> %s: %s", t0.isoformat(), t1.isoformat(), exc)
            failed.append((t0, t1))
        time.sleep(cfg.pause)

    log.info("done: %d ok, %d failed", ok, len(failed))
    for t0, t1 in failed:
        log.info("  retry manually: %s -> %s", t0.isoformat(), t1.isoformat())


if __name__ == "__main__":
    main()