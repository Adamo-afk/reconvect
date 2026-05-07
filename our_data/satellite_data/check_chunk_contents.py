from netCDF4 import Dataset
import hdf5plugin  # make sure this is imported

chunk_file = r"temp_output_mtg_20260316111146661255\\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY---NC4E_C_EUMT_20250515103058_IDPFI_OPE_20250515102834_20250515102917_N__O_0063_0037.nc"

ds = Dataset(chunk_file, 'r')
measured = ds['data']['vis_04']['measured']
print("Variables:", list(measured.variables.keys()))
print("effective_radiance shape:", measured.variables['effective_radiance'].shape)
data = measured.variables['effective_radiance'][:]
print("Data read OK, shape:", data.shape, "min:", data.min(), "max:", data.max())
ds.close()