"""
Compute the fraction of non-zero (ones) pixels across all lightning maps.

Formula: fraction = ones_pixels / total_pixels  (0.0 if total_pixels == 0)

Scans all lightning NetCDF files (density, current, occurrence) and
computes the fraction globally and per product.

Output: lightning_fraction.json

Usage:
    python lightning_fraction.py
    python lightning_fraction.py -d F:/nowcasting/coalition4-rcnn/our_data
"""

import numpy as np
import os
import json
import argparse
from netCDF4 import Dataset


DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'our_data'
)

PRODUCTS = ['density', 'current', 'occurrence']


def compute_fraction(data_root):
    """
    Compute fraction of non-zero pixels across all lightning maps.

    Returns:
        dict: per-product and global fractions
    """
    lightning_dir = os.path.join(data_root, 'lightning_data')

    stats = {}
    global_ones = 0
    global_total = 0

    for product in PRODUCTS:
        product_dir = os.path.join(lightning_dir, product)
        if not os.path.isdir(product_dir):
            print(f"  {product}: NOT FOUND")
            continue

        ones = 0
        total = 0
        n_files = 0

        for day_folder in sorted(os.listdir(product_dir)):
            day_path = os.path.join(product_dir, day_folder)
            if not os.path.isdir(day_path):
                continue

            for nc_file in sorted(os.listdir(day_path)):
                if not nc_file.endswith('.nc'):
                    continue

                filepath = os.path.join(day_path, nc_file)
                try:
                    with Dataset(filepath, 'r') as ds:
                        data = ds.variables['datamap'][:]

                    if isinstance(data, np.ma.MaskedArray):
                        data = data.filled(0.0)
                    if data.ndim == 3:
                        data = np.squeeze(data, axis=0)

                    ones += int(np.count_nonzero(data))
                    total += int(data.size)
                    n_files += 1

                except Exception as e:
                    print(f"    ERROR {nc_file}: {e}")

        fraction = ones / total if total > 0 else 0.0

        stats[product] = {
            'ones_pixels': ones,
            'total_pixels': total,
            'fraction': fraction,
            'n_files': n_files,
        }

        global_ones += ones
        global_total += total

        print(f"  {product}: {fraction:.6f} "
              f"({ones:,} / {total:,}, {n_files} files)")

    global_fraction = global_ones / global_total if global_total > 0 else 0.0
    stats['global'] = {
        'ones_pixels': global_ones,
        'total_pixels': global_total,
        'fraction': global_fraction,
    }

    print(f"\n  Global: {global_fraction:.6f} "
          f"({global_ones:,} / {global_total:,})")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute fraction of non-zero pixels in lightning maps."
    )
    parser.add_argument(
        "--data_root", "-d", type=str, default=DEFAULT_DATA_ROOT,
        help="Path to our_data directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path (default: our_data/lightning_fraction.json)"
    )

    args = parser.parse_args()
    output_path = args.output or os.path.join(
        args.data_root, 'lightning_fraction.json'
    )

    print("=" * 50)
    print("Lightning pixel fraction")
    print("=" * 50)

    stats = compute_fraction(args.data_root)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
    