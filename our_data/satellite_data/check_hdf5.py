"""
Quick check: can we read FCI chunk files without hdf5plugin?
Run from tfenv:  python check_hdf5.py
"""

import glob
import sys

# Find an FCI chunk file
patterns = [
    r"F:\nowcasting\coalition4-rcnn\our_data\satellite_data\MTG\_raw_chunks\*BODY*.nc",
    r"F:\nowcasting\coalition4-rcnn\our_data\satellite_data\MTG\_raw_chunks\*.nc",
]

nc_file = None
for pat in patterns:
    hits = glob.glob(pat)
    if hits:
        nc_file = hits[0]
        break

if nc_file is None:
    print("ERROR: No .nc files found in _raw_chunks/")
    sys.exit(1)

print(f"Test file: {nc_file}\n")

# Test 1: netCDF4 alone
print("=" * 50)
print("Test 1: netCDF4 WITHOUT hdf5plugin")
print("=" * 50)
try:
    from netCDF4 import Dataset
    ds = Dataset(nc_file, 'r')
    ch = list(ds['data'].groups.keys())[0]
    v = ds['data'][ch]['measured'].variables['effective_radiance']
    print(f"  Channel: {ch}")
    print(f"  Shape:   {v.shape}")
    print(f"  Sample:  {v[0, 0]}")
    ds.close()
    print("  RESULT:  OK — hdf5plugin is NOT needed")
    needs_plugin = False
except Exception as e:
    print(f"  RESULT:  FAILED — {e}")
    needs_plugin = True

# Test 2: with hdf5plugin
print()
print("=" * 50)
print("Test 2: netCDF4 WITH hdf5plugin")
print("=" * 50)
try:
    import hdf5plugin
    from netCDF4 import Dataset
    ds = Dataset(nc_file, 'r')
    ch = list(ds['data'].groups.keys())[0]
    v = ds['data'][ch]['measured'].variables['effective_radiance']
    print(f"  Channel: {ch}")
    print(f"  Shape:   {v.shape}")
    print(f"  Sample:  {v[0, 0]}")
    ds.close()
    print("  RESULT:  OK")
except Exception as e:
    print(f"  RESULT:  FAILED — {e}")

# Summary
print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
if not needs_plugin:
    print("Your files are readable without hdf5plugin.")
    print("You can safely uninstall it:  pip uninstall hdf5plugin -y")
else:
    print("hdf5plugin IS required (files are CharLS-compressed).")
    print("Fix the DLL conflict with:")
    print("  conda remove --force netcdf4 h5py hdf5 hdf5plugin -y")
    print("  conda install -c conda-forge h5py netcdf4 hdf5plugin -y")