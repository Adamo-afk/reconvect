"""
inspect_nwcsaf.py — Reconstruct CF-compliant `.nc` from a reprojected
NWCSAF `.npy`.

Combines a reprojected NWCSAF `.npy` (`ctth_alti`, `ctth_tempe`,
`cmic_phase`, `cmic_cot`) with the Romania grid lat/lon arrays to
produce a CF-compliant NetCDF that opens cleanly in QGIS, Panoply or
any xarray-based workflow. Mirrors the `inspect_mtg.py --reprojected`
and `identify_patches.write_diagnostic_nc` pattern.

Reconstruction only — no plotting. The point of the `.nc` is to feed it
to GIS software for visual inspection on a basemap.

Usage:
    python our_data/nwcsaf_data/inspect_nwcsaf.py \\
        --npy our_data/reprojected_data/nwcsaf_data/ctth_alti/\\
nc4_2026-03-14-Romania_ctth_alti/nc4_2026-03-14-Romania_1200_ctth_alti.npy

    # Specify the output path explicitly
    python our_data/nwcsaf_data/inspect_nwcsaf.py \\
        --npy <input.npy> --output my_nwcsaf.nc

    # Override the auto-discovered grid coordinates directory
    python our_data/nwcsaf_data/inspect_nwcsaf.py \\
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


# Per-variable metadata. `cmic_phase` is the only categorical variable
# (int8 on disk, with 0..4 / 255 cluster codes); the rest are continuous
# float32. The `flag_*` attributes follow CF-1.8 conventions and let
# GIS software show category labels in the value lookup table.
NWCSAF_VARS = {
    "ctth_alti": {
        "long_name": "Cloud top height",
        "units":     "m",
        "dtype":     np.float32,
    },
    "ctth_tempe": {
        "long_name": "Cloud top temperature",
        "units":     "K",
        "dtype":     np.float32,
    },
    "cmic_phase": {
        "long_name":     "Cloud microphysics phase classification",
        "units":         "1",
        "dtype":         np.int8,
        "flag_values":   np.array([0, 1, 2, 3, 4], dtype=np.int8),
        "flag_meanings": "no_cloud water_cloud "
                         "ice_cloud_supercooled_water "
                         "ice_cloud_no_supercooled_water mixed_phase",
    },
    "cmic_cot": {
        "long_name": "Cloud optical thickness",
        "units":     "1",
        "dtype":     np.float32,
    },
}


_NAME_PATTERN = re.compile(
    r"^nc4_(?P<date>\d{4}-\d{2}-\d{2})-Romania_(?P<hhmm>\d{4})_"
    r"(?P<var>ctth_alti|ctth_tempe|cmic_phase|cmic_cot)\.npy$"
)


def parse_nwcsaf_filename(npy_path: Path) -> dict:
    """Return {variable, date_iso, hhmm} parsed from the `.npy` filename."""
    name = npy_path.name
    m = _NAME_PATTERN.match(name)
    if not m:
        raise ValueError(
            f"Cannot parse NWCSAF filename: {name}. Expected "
            f"`nc4_YYYY-MM-DD-Romania_HHMM_<variable>.npy` with variable "
            f"in {sorted(NWCSAF_VARS)}."
        )
    return {
        "variable": m.group("var"),
        "date_iso": m.group("date"),
        "hhmm":     m.group("hhmm"),
    }


def discover_grid_dir(npy_path: Path) -> Path:
    """Walk up from `npy_path` to locate `romania_grid_lats.npy`."""
    for parent in npy_path.resolve().parents:
        if (parent / "romania_grid_lats.npy").is_file():
            return parent
    raise FileNotFoundError(
        "Cannot find romania_grid_lats.npy by walking up from "
        f"{npy_path}. Pass --grid_dir explicitly."
    )


def build_reprojected_nc(npy_path: Path,
                          grid_dir: Path | None = None) -> xr.Dataset:
    """Load a reprojected NWCSAF `.npy` and assemble a CF dataset."""
    if not npy_path.is_file():
        raise FileNotFoundError(npy_path)

    info = parse_nwcsaf_filename(npy_path)
    variable = info["variable"]
    date_iso = info["date_iso"]
    hhmm     = info["hhmm"]

    if grid_dir is None:
        grid_dir = discover_grid_dir(npy_path)
    lats = np.load(grid_dir / "romania_grid_lats.npy")
    lons = np.load(grid_dir / "romania_grid_lons.npy")

    spec = NWCSAF_VARS[variable]
    data = np.load(npy_path).astype(spec["dtype"])
    if data.shape != lats.shape:
        raise ValueError(
            f"Shape mismatch: data {data.shape} vs grid {lats.shape}. "
            f"Are the grid coordinates from the same reproject run?"
        )

    var_attrs = {
        "long_name":    spec["long_name"],
        "units":        spec["units"],
        "coordinates":  "latitude longitude",
        "grid_mapping": "crs",
    }
    if "flag_values" in spec:
        var_attrs["flag_values"]   = spec["flag_values"]
        var_attrs["flag_meanings"] = spec["flag_meanings"]

    ds = xr.Dataset(
        {variable: (["y", "x"], data)},
        coords={
            "latitude":  (["y", "x"], lats),
            "longitude": (["y", "x"], lons),
        },
    )
    ds[variable].attrs = var_attrs

    ds["crs"] = xr.DataArray(np.int32(0))
    ds["crs"].attrs = {
        "grid_mapping_name": "oblique_stereographic",
        "EPSG":              31700,
        "comment":           "Romania Stereo70 / Dealul Piscului 1970",
    }

    ds.attrs = {
        "title":       f"NWCSAF {variable} — "
                       f"{date_iso} {hhmm[:2]}:{hhmm[2:]} UTC",
        "source":      "inspect_nwcsaf.py (COALITION-4 reproject pipeline)",
        "Conventions": "CF-1.8",
        "date":        date_iso,
        "time_utc":    f"{hhmm[:2]}:{hhmm[2:]}",
        "variable":    variable,
    }
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct a CF-compliant .nc from a reprojected "
                    "NWCSAF .npy.",
    )
    parser.add_argument(
        "--npy", required=True, type=str,
        help="Reprojected NWCSAF .npy file path.",
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
