"""
Validate a chosen training cadence against native data product cadences and
write a timestep configuration file consumed by all downstream pipeline scripts.

The COALITION-4 pipeline previously assumed a fixed 15-minute cadence. This
script makes the cadence explicit and validated up front so that arrange,
download, regrid, patch-extract, sequence-extract, dataset-build and training
scripts all consume the same configuration instead of duplicating constants.

Native cadences (highest-frequency = lowest minutes between samples):
    radar     : 10 min  (raw composites at :00 :10 :20 :30 :40 :50)
    MTG FCI   : 10 min  (FDHSI / HRFI native)
    NWCSAF    : 10 min  (tied to underlying MTG cadence in EURMLAND-NR)
    lightning : continuous (KML strokes; aggregated into the chosen window)

For each non-continuous product, the chosen step must be >= the product's
native cadence — otherwise we'd need data we don't have. The step does not
need to be a multiple of the cadence (e.g. step = 15 min with 10-min radar is
allowed); when it is not a multiple, the script picks the native minutes
closest to each grid slot ("nearest neighbour" selection, no interpolation).

Output (default: our_data/timestep_config.json):
    {
        "step_minutes": 15,
        "steps_per_day": 96,
        "max_native_cadence_minutes": 10,
        "minute_filter": [0, 10, 30, 50],   # union across products
        "products": {
            "radar":     {"cadence_minutes": 10, "filter": [0, 10, 30, 50]},
            "mtg":       {"cadence_minutes": 10, "filter": [0, 10, 30, 50]},
            "nwcsaf":    {"cadence_minutes": 10, "filter": [0, 10, 30, 50]},
            "lightning": {"cadence_minutes": null, "filter": null}
        },
        "created_utc": "..."
    }

Usage:
    python validate_timestep.py --step_minutes 15
    python validate_timestep.py --step_minutes 10
    python validate_timestep.py --step_minutes 5    # ERROR
    python validate_timestep.py --step_minutes 15 --output_path some/path.json
    python validate_timestep.py --print              # show current config
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# Native cadences in minutes. None = continuous (no minute filter required).
PRODUCT_CADENCES = {
    "radar":     10,
    "mtg":       10,
    "nwcsaf":    10,
    "lightning": None,
}

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "our_data" / "timestep_config.json"


def compute_minute_filter(step_minutes: int, cadence_minutes: int) -> list[int]:
    """
    Pick the native minute slots (0..59 within each hour) closest to the
    requested step grid.

    For each grid slot k * step_minutes within an hour, find the native minute
    n * cadence_minutes that minimises |n*cadence - k*step|. Ties are broken
    by preferring earlier minutes (consistent with `round-half-down`).

    Returns the sorted union of selected native minutes across one full day,
    de-duplicated and constrained to [0, 60).
    """
    if cadence_minutes is None:
        return []

    selected = set()
    minutes_per_day = 24 * 60
    n_steps = minutes_per_day // step_minutes
    native_minutes_in_hour = list(range(0, 60, cadence_minutes))

    for k in range(n_steps):
        target_total = k * step_minutes
        target_in_hour = target_total % 60
        # Find native minute in current hour closest to target_in_hour
        best_minute = min(
            native_minutes_in_hour,
            key=lambda m: (abs(m - target_in_hour), m),
        )
        selected.add(best_minute)

    return sorted(selected)


def validate(step_minutes: int) -> dict:
    """
    Validate the requested step against all product cadences and return a
    structured config dict ready for serialisation.

    Raises ValueError with a descriptive message if step_minutes is below the
    highest available native cadence — listing the limiting products.
    """
    if step_minutes <= 0:
        raise ValueError(f"step_minutes must be positive, got {step_minutes}")
    if 1440 % step_minutes != 0:
        raise ValueError(
            f"step_minutes={step_minutes} does not divide 1440 (a day) evenly. "
            f"Use a step that fits cleanly into 24h, e.g. 5, 10, 15, 20, 30, 60."
        )

    discrete = {p: c for p, c in PRODUCT_CADENCES.items() if c is not None}
    max_cadence = max(discrete.values())

    if step_minutes < max_cadence:
        limiting = sorted(p for p, c in discrete.items() if c == max_cadence)
        raise ValueError(
            f"Requested step_minutes={step_minutes} is smaller than the "
            f"highest native cadence ({max_cadence} min) of products: "
            f"{', '.join(limiting)}. The lowest valid step is {max_cadence} min."
        )

    products = {}
    union_filter = set()
    for name, cadence in PRODUCT_CADENCES.items():
        if cadence is None:
            products[name] = {"cadence_minutes": None, "filter": None}
            continue
        flt = compute_minute_filter(step_minutes, cadence)
        products[name] = {"cadence_minutes": cadence, "filter": flt}
        union_filter.update(flt)

    return {
        "step_minutes": step_minutes,
        "steps_per_day": 1440 // step_minutes,
        "max_native_cadence_minutes": max_cadence,
        "minute_filter": sorted(union_filter),
        "products": products,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_config(config: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)


def print_config(config: dict) -> None:
    print(json.dumps(config, indent=2))


def load_existing(output_path: Path) -> dict | None:
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate training cadence and write timestep_config.json"
    )
    parser.add_argument(
        "--step_minutes", type=int, default=None,
        help="Desired training step in minutes (e.g. 10, 15, 20, 30)."
    )
    parser.add_argument(
        "--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH),
        help=f"Where to write the config (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--print", dest="print_only", action="store_true",
        help="Print the existing config (if any) and exit without writing."
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)

    if args.print_only:
        existing = load_existing(output_path)
        if existing is None:
            print(f"No config found at {output_path}.")
            return 1
        print_config(existing)
        return 0

    if args.step_minutes is None:
        parser.error("--step_minutes is required (or use --print)")

    try:
        config = validate(args.step_minutes)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    write_config(config, output_path)
    print(f"Wrote timestep config to {output_path}")
    print_config(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
