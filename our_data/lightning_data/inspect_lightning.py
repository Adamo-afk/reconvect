"""
inspect_lightning.py — Reconstruct CF-compliant `.nc` from a reprojected
lightning `.npy`.

Combines a reprojected lightning `.npy` file (`density` / `current` /
`occurrence`) with the Romania grid lat/lon arrays to produce a
CF-compliant NetCDF that opens cleanly in QGIS, Panoply or any
xarray-based workflow. Mirrors the
`identify_patches.write_diagnostic_nc` pattern.

Reconstruction only — no plotting. The point of the `.nc` is to feed it
to GIS software for visual inspection on a basemap.

Usage:
    # Single file
    python our_data/lightning_data/inspect_lightning.py \\
        --npy our_data/reprojected_data/lightning_data/occurrence/\\
nc4_2026-03-14-Romania_occurrence/lightning_occurrence_20260314_1200.npy

    # Specify the output path explicitly
    python our_data/lightning_data/inspect_lightning.py \\
        --npy <input.npy> --output my_lightning.nc

    # Override the auto-discovered grid coordinates directory
    python our_data/lightning_data/inspect_lightning.py \\
        --npy <input.npy> --grid_dir /path/to/reprojected_data
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray is required. Install with: pip install xarray",
          file=sys.stderr)
    sys.exit(1)


LIGHTNING_PRODUCTS = {"density", "current", "occurrence"}

# Filename patterns we accept. The reproject pipeline writes
# `lightning_<product>_YYYYMMDD_HHMM.npy`; older snapshots used
# `nc4_YYYY-MM-DD-Romania_<HHMM>_<product>.npy`. Both are supported so the
# script can read either.
_NAME_PATTERNS = (
    re.compile(
        r"^lightning_(?P<product>density|current|occurrence)_"
        r"(?P<date>\d{8})_(?P<hhmm>\d{4})\.npy$"
    ),
    re.compile(
        r"^nc4_(?P<date>\d{4}-\d{2}-\d{2})-Romania_(?P<hhmm>\d{4})_"
        r"(?P<product>density|current|occurrence)\.npy$"
    ),
)


def parse_lightning_filename(npy_path: Path) -> dict:
    """Return {product, date_iso, hhmm} parsed from the `.npy` filename."""
    name = npy_path.name
    for pat in _NAME_PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        product = m.group("product")
        raw_date = m.group("date")
        if "-" in raw_date:
            date_iso = raw_date
        else:
            date_iso = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        return {
            "product":  product,
            "date_iso": date_iso,
            "hhmm":     m.group("hhmm"),
        }
    raise ValueError(
        f"Cannot parse lightning filename: {name}. Expected either "
        f"`lightning_<product>_YYYYMMDD_HHMM.npy` or "
        f"`nc4_YYYY-MM-DD-Romania_HHMM_<product>.npy`."
    )


def discover_grid_dir(npy_path: Path) -> Path:
    """Walk up from `npy_path` to locate `romania_grid_lats.npy`.

    The reproject pipeline writes the shared lat/lon arrays once at the
    `reprojected_data/` root, so we walk up the tree from the npy until
    we find them.
    """
    for parent in npy_path.resolve().parents:
        if (parent / "romania_grid_lats.npy").is_file():
            return parent
    raise FileNotFoundError(
        "Cannot find romania_grid_lats.npy by walking up from "
        f"{npy_path}. Pass --grid_dir explicitly."
    )


def build_reprojected_nc(npy_path: Path,
                          grid_dir: Path | None = None) -> xr.Dataset:
    """Load a reprojected lightning `.npy` and assemble a CF dataset."""
    if not npy_path.is_file():
        raise FileNotFoundError(npy_path)

    info = parse_lightning_filename(npy_path)
    product  = info["product"]
    date_iso = info["date_iso"]
    hhmm     = info["hhmm"]

    if grid_dir is None:
        grid_dir = discover_grid_dir(npy_path)
    lats = np.load(grid_dir / "romania_grid_lats.npy")
    lons = np.load(grid_dir / "romania_grid_lons.npy")

    data = np.load(npy_path).astype(np.float32)
    if data.shape != lats.shape:
        raise ValueError(
            f"Shape mismatch: data {data.shape} vs grid {lats.shape}. "
            f"Are the grid coordinates from the same reproject run?"
        )

    # Per-product long_name + units so the .nc is self-describing in
    # GIS software's metadata panel.
    var_attrs = {
        "density":    {
            "long_name": "Lightning stroke density per bin",
            "units":     "strokes / pixel",
        },
        "current":    {
            "long_name": "Peak lightning current per bin (signed)",
            "units":     "kA",
        },
        "occurrence": {
            "long_name": "Lightning occurrence flag per bin (binary)",
            "units":     "1",
            "flag_values": np.array([0, 1], dtype=np.int8),
            "flag_meanings": "no_lightning lightning",
        },
    }[product]
    var_attrs.update({
        "coordinates":  "latitude longitude",
        "grid_mapping": "crs",
    })

    ds = xr.Dataset(
        {product: (["y", "x"], data)},
        coords={
            "latitude":  (["y", "x"], lats),
            "longitude": (["y", "x"], lons),
        },
    )
    ds[product].attrs = var_attrs

    ds["crs"] = xr.DataArray(np.int32(0))
    ds["crs"].attrs = {
        "grid_mapping_name": "oblique_stereographic",
        "EPSG":              31700,
        "comment":           "Romania Stereo70 / Dealul Piscului 1970",
    }

    ds.attrs = {
        "title":       f"LINET lightning {product} — "
                       f"{date_iso} {hhmm[:2]}:{hhmm[2:]} UTC",
        "source":      "inspect_lightning.py (COALITION-4 reproject pipeline)",
        "Conventions": "CF-1.8",
        "date":        date_iso,
        "time_utc":    f"{hhmm[:2]}:{hhmm[2:]}",
        "product":     product,
    }
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct a CF-compliant .nc from a reprojected "
                    "lightning .npy.",
    )
    parser.add_argument(
        "--npy", required=True, type=str,
        help="Reprojected lightning .npy file path.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .nc path. Defaults to the input .npy with a .nc "
             "extension in the same directory.",
    )
    parser.add_argument(
        "--grid_dir", type=str, default=None,
        help="Directory containing romania_grid_{lats,lons}.npy. "
             "Auto-discovered by walking up from --npy if omitted.",
    )
    args = parser.parse_args()

    npy_path = Path(args.npy)
    grid_dir = Path(args.grid_dir) if args.grid_dir else None
    out_path = (Path(args.output) if args.output
                else npy_path.with_suffix(".nc"))

    print(f"Reconstructing {npy_path.name} -> {out_path}")
    ds = build_reprojected_nc(npy_path, grid_dir=grid_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    ds.close()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
