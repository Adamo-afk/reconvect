"""
regrid_opera.py — Reproject OPERA radar data onto the Romania EPSG:31700 grid.

Reads OPERA ODC HDF5 files (downloaded by `pipeline_opera.py`), extracts the
data array + source projection from the embedded `/where` and `/dataset1`
metadata, reprojects onto the 1536×768 EPSG:31700 Stereo70 Romania grid, and
writes one CF-compliant NetCDF (`.nc`) per source file under
`our_data/regridded_data/opera_data/{product}/{date}-Romania/`.

OPERA HDF5 layout (ODYSSEY composite convention):

    /where  attributes:
        projdef    : str, proj4 string (typically Lambert Azimuthal Equal Area)
        xsize, ysize : grid dimensions in pixels
        xscale, yscale : pixel size in metres
        LL_lat, LL_lon : lower-left corner (degrees)
        UR_lat, UR_lon : upper-right corner (degrees)
    /dataset1/data1/data : the 2-D radar field (integer, scale_factor + offset)
    /dataset1/data1/what attributes:
        gain, offset : physical = gain * raw + offset
        nodata, undetect : sentinel values

The source projection parameters are read from the first file at each
resolution (1 km for reflectivity, 2 km for rainfall_rate); the KD-tree
mapping is built once per resolution and reused for every file at that
resolution. Same caching strategy as `regrid_satellite_mtg()` in regrid.py.

Output NetCDF schema (per file):

    dimensions : (y=768, x=1536)
    variables  :
        <product> (y, x)            -- physical units (dBZ or mm/h)
        latitude  (y, x)            -- degrees_north
        longitude (y, x)            -- degrees_east
        crs        (scalar)         -- grid_mapping with EPSG:31700 metadata

Usage:
    python our_data/opera_data/regrid_opera.py
    python our_data/opera_data/regrid_opera.py --products opera_reflectivity
    python our_data/opera_data/regrid_opera.py --date 2025-06-15
    python our_data/opera_data/regrid_opera.py --workers 4
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print(
        "ERROR: h5py is required to read OPERA HDF5 files.\n"
        "Install with: pip install h5py",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import pyproj
    from pyresample import geometry, kd_tree
except ImportError:
    print(
        "ERROR: pyresample and pyproj are required for regridding.\n"
        "Install with: pip install pyresample pyproj",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from netCDF4 import Dataset
except ImportError:
    print(
        "ERROR: netCDF4 is required to write the regridded output.\n"
        "Install with: pip install netCDF4",
        file=sys.stderr,
    )
    sys.exit(1)


# c4dl.projection holds the Romania grid definition (EPSG:31700, 1536×768).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from c4dl.projection import GridProjection, romania_grid_area  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_ROOT = PROJECT_ROOT / "our_data"
DEFAULT_OPERA_ROOT = DEFAULT_DATA_ROOT / "opera_data"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "regridded_data" / "opera_data"

PRODUCTS = {
    "opera_reflectivity":  {
        "subdir":     "reflectivity",
        "var_name":   "max_reflectivity",
        "units":      "dBZ",
        "long_name":  "Maximum reflectivity",
        "resolution": "1km",
    },
    "opera_rainfall_rate": {
        "subdir":     "rainfall_rate",
        "var_name":   "rainfall_rate",
        "units":      "mm h-1",
        "long_name":  "Instantaneous rainfall rate",
        "resolution": "2km",
    },
}

DEFAULT_WORKERS = 4

# Filename timestamp parsers — see pipeline_opera.py for the conventions.
TIMESTAMP_PATTERN_ISO = re.compile(
    r'(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})(\d{2})Z?'
)
TIMESTAMP_PATTERN_COMPACT = re.compile(r'(\d{12,14})')

# h5py is not always thread-safe across all builds; serialise reads.
_h5_lock = threading.Lock()


# =============================================================================
# Romania target grid
# =============================================================================

def init_romania_grid():
    print("Initialising Romania grid projection...")
    gp = GridProjection(romania_grid_area)
    y, x = np.mgrid[:gp.area.height, :gp.area.width]
    target_lons, target_lats = gp.inverse(y, x)
    print(f"  Target grid: {target_lats.shape}")
    return target_lats.astype(np.float64), target_lons.astype(np.float64)


# =============================================================================
# Read OPERA HDF5
# =============================================================================

def parse_opera_filename(path: Path):
    """
    Return (date_str 'YYYY-MM-DD', hhmm 'HHMM') from an OPERA filename.

    Supports ISO (`2026-05-11T000500Z-...`) and compact
    (`...20260511000500.h5`) conventions; ISO is tried first.
    """
    name = path.name
    m = TIMESTAMP_PATTERN_ISO.search(name)
    if m:
        try:
            dt = datetime.datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            )
            return dt.strftime('%Y-%m-%d'), dt.strftime('%H%M')
        except ValueError:
            pass
    m = TIMESTAMP_PATTERN_COMPACT.search(name)
    if m:
        ts = m.group(1)[:12]
        try:
            dt = datetime.datetime.strptime(ts, '%Y%m%d%H%M')
            return dt.strftime('%Y-%m-%d'), dt.strftime('%H%M')
        except ValueError:
            pass
    return None, None


def _decode_attr(value):
    """Decode an HDF5 attribute that may be bytes or a 0-d array."""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return value


def read_opera_source_grid(h5_path: Path):
    """
    Build a 2-D source lat/lon grid from an OPERA HDF5 file's /where
    metadata.

    Returns (lats, lons, projdef) where projdef is the proj4 string.
    """
    with h5py.File(h5_path, 'r') as f:
        where = f['/where'].attrs
        projdef = _decode_attr(where['projdef'])
        xsize = int(_decode_attr(where['xsize']))
        ysize = int(_decode_attr(where['ysize']))
        xscale = float(_decode_attr(where['xscale']))
        yscale = float(_decode_attr(where['yscale']))
        ll_lon = float(_decode_attr(where['LL_lon']))
        ll_lat = float(_decode_attr(where['LL_lat']))
        ur_lon = float(_decode_attr(where['UR_lon']))
        ur_lat = float(_decode_attr(where['UR_lat']))

    src_proj = pyproj.Proj(projdef)
    geographic = pyproj.Proj('epsg:4326')
    to_xy = pyproj.Transformer.from_proj(
        geographic, src_proj, always_xy=True
    )

    # The /where corners are geographic; convert to projection metres
    ll_x, ll_y = to_xy.transform(ll_lon, ll_lat)
    ur_x, ur_y = to_xy.transform(ur_lon, ur_lat)

    # Pixel centres on the projection plane (rows count from top in OPERA HDF5)
    # x increases east; y decreases as row index increases.
    x_centres = ll_x + (np.arange(xsize) + 0.5) * xscale
    y_centres = ur_y - (np.arange(ysize) + 0.5) * yscale
    xx, yy = np.meshgrid(x_centres, y_centres)

    to_geo = pyproj.Transformer.from_proj(src_proj, geographic, always_xy=True)
    lons, lats = to_geo.transform(xx, yy)

    # Off-disk / off-grid points may come back as inf or NaN
    invalid = np.isinf(lons) | np.isinf(lats) | np.isnan(lons) | np.isnan(lats)
    lons = np.where(invalid, 0.0, lons).astype(np.float64)
    lats = np.where(invalid, 0.0, lats).astype(np.float64)

    return lats, lons, projdef


def read_opera_data(h5_path: Path):
    """
    Return (data float32, gain, offset, nodata, undetect, where_attrs).
    Physical = gain * raw + offset, with raw==nodata|undetect masked to NaN.
    """
    with _h5_lock, h5py.File(h5_path, 'r') as f:
        ds = f['/dataset1/data1']
        what = ds['what'].attrs if 'what' in ds else None
        gain = float(_decode_attr(what['gain'])) if what is not None else 1.0
        offset = float(_decode_attr(what['offset'])) if what is not None else 0.0
        nodata = (float(_decode_attr(what['nodata']))
                  if what is not None and 'nodata' in what else None)
        undetect = (float(_decode_attr(what['undetect']))
                    if what is not None and 'undetect' in what else None)

        raw = np.asarray(ds['data'], dtype=np.float32)

    physical = gain * raw + offset
    mask = np.zeros_like(physical, dtype=bool)
    if nodata is not None:
        mask |= (raw == nodata)
    if undetect is not None:
        # 'undetect' typically means "no precipitation detected" — convert to
        # zero for downstream consumption rather than NaN, since 0 is the
        # meaningful physical value.
        physical = np.where(raw == undetect, 0.0, physical)
    if mask.any():
        physical = np.where(mask, np.nan, physical)

    return physical


# =============================================================================
# Precomputed mapping
# =============================================================================

class PrecomputedMapping:
    """KD-tree mapping from a source (lat, lon) grid to the Romania grid."""

    def __init__(self, src_lats, src_lons, target_lats, target_lons,
                 radius_km=10.0):
        src_lats = np.asarray(src_lats, dtype=np.float64)
        src_lons = np.asarray(src_lons, dtype=np.float64)
        src_lats = np.clip(src_lats, -90.0, 90.0)
        src_lons = np.clip(src_lons, -180.0, 180.0)

        src_area = geometry.SwathDefinition(lons=src_lons, lats=src_lats)
        tgt_area = geometry.SwathDefinition(lons=target_lons, lats=target_lats)

        info = kd_tree.get_neighbour_info(
            src_area, tgt_area,
            radius_of_influence=radius_km * 1000.0,
            neighbours=1,
        )
        # get_neighbour_info returns (valid_input_index, valid_output_index,
        # index_array, distance_array)
        self.valid_in, self.valid_out, self.index_array, _ = info
        self.tgt_shape = target_lats.shape

    def apply(self, src_data, fill_value=np.nan):
        return kd_tree.get_sample_from_neighbour_info(
            'nn',
            self.tgt_shape,
            src_data,
            self.valid_in, self.valid_out, self.index_array,
            fill_value=fill_value,
        )


# =============================================================================
# Output: CF-compliant NetCDF
# =============================================================================

def write_regridded_nc(out_path: Path, var_name: str, data: np.ndarray,
                       lats: np.ndarray, lons: np.ndarray,
                       units: str, long_name: str,
                       source_filename: str, source_projdef: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with Dataset(out_path, 'w', format='NETCDF4') as dst:
        ny, nx = data.shape
        dst.createDimension('y', ny)
        dst.createDimension('x', nx)

        v_lat = dst.createVariable('latitude', 'f4', ('y', 'x'),
                                   zlib=True, complevel=4)
        v_lat.units = 'degrees_north'
        v_lat.long_name = 'latitude'
        v_lat[:] = lats.astype(np.float32)

        v_lon = dst.createVariable('longitude', 'f4', ('y', 'x'),
                                   zlib=True, complevel=4)
        v_lon.units = 'degrees_east'
        v_lon.long_name = 'longitude'
        v_lon[:] = lons.astype(np.float32)

        v = dst.createVariable(var_name, 'f4', ('y', 'x'),
                               zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))
        v.units = units
        v.long_name = long_name
        v.coordinates = 'latitude longitude'
        v.grid_mapping = 'crs'
        v[:] = data.astype(np.float32)

        crs = dst.createVariable('crs', 'i4')
        crs.grid_mapping_name = 'oblique_stereographic'
        crs.EPSG = 31700
        crs.proj4 = (
            '+proj=sterea +lat_0=46 +lon_0=25 +k=0.99975 '
            '+x_0=500000 +y_0=500000 +ellps=krass +units=m +no_defs'
        )
        crs.comment = 'Romania Stereo70 / Dealul Piscului 1970'

        dst.title = 'OPERA radar regridded onto the Romania EPSG:31700 grid'
        dst.source_file = source_filename
        dst.source_projdef = source_projdef
        dst.history = (f"Created {datetime.datetime.utcnow().isoformat()}Z "
                       f"by regrid_opera.py")


# =============================================================================
# Per-product worker
# =============================================================================

def discover_h5_files(opera_root: Path, subdir: str,
                      date_filter: str | None = None):
    """
    Walk `{opera_root}/{subdir}/{YYYY}/{MM}/{DD}/*.h5` and return a list of
    Path objects.
    """
    root = opera_root / subdir
    if not root.is_dir():
        return []
    out: list[Path] = []
    for year in sorted(root.iterdir()):
        if not year.is_dir() or not year.name.isdigit():
            continue
        for month in sorted(year.iterdir()):
            if not month.is_dir() or not month.name.isdigit():
                continue
            for day in sorted(month.iterdir()):
                if not day.is_dir() or not day.name.isdigit():
                    continue
                day_str = f"{year.name}-{month.name}-{day.name}"
                if date_filter and date_filter != day_str:
                    continue
                for f in sorted(os.listdir(day)):
                    if f.endswith('.h5'):
                        out.append(day / f)
    return out


def regrid_product(product: str,
                   opera_root: Path,
                   output_root: Path,
                   target_lats: np.ndarray,
                   target_lons: np.ndarray,
                   date_filter: str | None,
                   workers: int) -> dict:
    cfg = PRODUCTS[product]
    files = discover_h5_files(opera_root, cfg["subdir"], date_filter)
    print(f"\n[{product}] {len(files)} .h5 files")
    if not files:
        return {'new': 0, 'skipped': 0, 'errors': 0}

    # Build the source-grid mapping from the first file
    first_file = files[0]
    print(f"[{product}] Reading source grid from {first_file.name} ...")
    try:
        src_lats, src_lons, projdef = read_opera_source_grid(first_file)
    except Exception as e:
        print(f"[{product}] ERROR reading source grid: {e}", file=sys.stderr)
        return {'new': 0, 'skipped': 0, 'errors': len(files)}

    print(f"[{product}] Source grid shape: {src_lats.shape}, projdef={projdef!r}")
    print(f"[{product}] Building KD-tree mapping...")
    mapping = PrecomputedMapping(src_lats, src_lons, target_lats, target_lons)
    print(f"[{product}] Mapping ready.")

    new = skipped = errors = 0

    def process_one(h5_path: Path):
        try:
            date_str, hhmm = parse_opera_filename(h5_path)
            if date_str is None:
                return ('error', f"unparsable filename: {h5_path.name}")

            out_dir = (output_root / cfg["subdir"]
                       / f"nc4_{date_str}-Romania_{cfg['subdir']}")
            out_name = f"nc4_{date_str}-Romania_{hhmm}_{cfg['subdir']}.nc"
            out_path = out_dir / out_name

            if out_path.exists():
                return ('skipped', None)

            data = read_opera_data(h5_path)
            regridded = mapping.apply(data, fill_value=np.nan)
            write_regridded_nc(
                out_path,
                var_name=cfg["var_name"],
                data=regridded,
                lats=target_lats,
                lons=target_lons,
                units=cfg["units"],
                long_name=cfg["long_name"],
                source_filename=h5_path.name,
                source_projdef=projdef,
            )
            return ('new', None)
        except Exception as e:
            return ('error', f"{h5_path.name}: {e}")

    print(f"[{product}] Reprojecting with {workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, p): p for p in files}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            status, msg = fut.result()
            if status == 'new':
                new += 1
            elif status == 'skipped':
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR: {msg}", file=sys.stderr)
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] new={new}, skipped={skipped}, errors={errors}")

    return {'new': new, 'skipped': skipped, 'errors': errors}


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproject downloaded OPERA HDF5 files onto the Romania "
                    "EPSG:31700 grid and save as CF-compliant NetCDF."
    )
    parser.add_argument('--opera_root', type=str,
                        default=str(DEFAULT_OPERA_ROOT),
                        help=f'Local OPERA cache root '
                             f'(default: {DEFAULT_OPERA_ROOT})')
    parser.add_argument('--output_root', type=str,
                        default=str(DEFAULT_OUTPUT_ROOT),
                        help=f'Output root for regridded NetCDFs '
                             f'(default: {DEFAULT_OUTPUT_ROOT})')
    parser.add_argument('--products', type=str, nargs='+',
                        default=list(PRODUCTS.keys()),
                        choices=list(PRODUCTS.keys()),
                        help='OPERA products to regrid (default: both)')
    parser.add_argument('--date', type=str, default=None,
                        help='Restrict to a single date YYYY-MM-DD (optional)')
    parser.add_argument('--workers', '-w', type=int, default=DEFAULT_WORKERS,
                        help=f'Parallel worker threads per product '
                             f'(default: {DEFAULT_WORKERS})')

    args = parser.parse_args()

    opera_root = Path(args.opera_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPERA → Romania Regridder")
    print("=" * 70)
    print(f"Opera root      : {opera_root}")
    print(f"Output root     : {output_root}")
    print(f"Products        : {args.products}")
    print(f"Date filter     : {args.date or '(all)'}")
    print(f"Workers         : {args.workers}")

    target_lats, target_lons = init_romania_grid()

    overall = {'new': 0, 'skipped': 0, 'errors': 0}
    for product in args.products:
        stats = regrid_product(
            product, opera_root, output_root,
            target_lats, target_lons,
            date_filter=args.date,
            workers=args.workers,
        )
        for k in overall:
            overall[k] += stats[k]

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  New        : {overall['new']}")
    print(f"  Skipped    : {overall['skipped']} (cached)")
    print(f"  Errors     : {overall['errors']}")

    return 0 if overall['errors'] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
