from netCDF4 import Dataset
import hdf5plugin
import numpy as np

ds = Dataset(r"temp_output_mtg_20260316111146661255\\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY---NC4E_C_EUMT_20250515103058_IDPFI_OPE_20250515102834_20250515102917_N__O_0063_0037.nc", 'r')

x = ds['data']['ir_105']['measured'].variables['x'][:]
y = ds['data']['ir_105']['measured'].variables['y'][:]

print(f"x: min={x.min():.6f}, max={x.max():.6f}, shape={x.shape}")
print(f"y: min={y.min():.6f}, max={y.max():.6f}, shape={y.shape}")
print(f"First 5 x: {x[:5]}")
print(f"First 5 y: {y[:5]}")

ds.close()