from netCDF4 import Dataset
import os

# Point to one of the body chunk files in the latest temp dir
# or a file you saved earlier
chunk_file = r"temp_output_mtg_20260316111146661255\\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY---NC4E_C_EUMT_20250515103058_IDPFI_OPE_20250515102834_20250515102917_N__O_0063_0037.nc"

ds = Dataset(chunk_file, 'r')
print("Root groups:", list(ds.groups.keys()))

if 'data' in ds.groups:
    print("\ndata/ subgroups:", list(ds.groups['data'].groups.keys()))
    
    # Print first 5 channel group names and their contents
    for i, ch_name in enumerate(ds.groups['data'].groups.keys()):
        ch = ds.groups['data'].groups[ch_name]
        print(f"\n  data/{ch_name}/")
        print(f"    subgroups: {list(ch.groups.keys())}")
        if 'measured' in ch.groups:
            print(f"    measured vars: {list(ch.groups['measured'].variables.keys())}")
        if i >= 5:
            print("  ... (truncated)")
            break

ds.close()