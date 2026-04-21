import netCDF4
import numpy as np

from .datasetreader import DatasetReader
from .. import projection

class SwissDEMReader(DatasetReader):
    name = "swissdem"

    fields = [
        "EW_deriv", "NS_deriv", "Altitude",
        # "TPI_1", "TPI_2", "TPI_3",
        # "morpho_1", "morpho_2", "morpho_3",
        # "ifac_1", "ifac_2", "ifac_3",
        # "Aspect", "Slope"
    ]

    def __init__(self, *, dem_file, variables=None):
        if variables is None:
            variables = [
                "EW_deriv", "NS_deriv", "Altitude",
                # "TPI_1", "TPI_2", "TPI_3",
                # "morpho_1", "morpho_2", "morpho_3",
                # "ifac_1", "ifac_2", "ifac_3",
                # "Sin_Aspect", "Cos_Aspect", 
                # "Slope"
            ]
        
        grid_projection = projection.romania_grid_area
        super().__init__(grid_projection, variables=variables)

        fields = {}
        with netCDF4.Dataset(dem_file, 'r') as ds:
            for field in SwissDEMReader.fields:
                fields[field] = ds[field][:]
                fields[field] = fields[field][::-1,:]

        self.var_data = {}
        for var in variables:
            if var in fields:
                self.var_data[var] = fields[var]
                if var.startswith("ifac"):
                    self.var_data[var][np.isnan(self.var_data[var])] = 1.0
            elif var == "Sin_Aspect":
                self.var_data[var] = np.sin(fields["Aspect"])
            elif var == "Cos_Aspect":
                self.var_data[var] = np.cos(fields["Aspect"])
            else:
                raise ValueError("Unknown DEM product.")
            
    def variable_for_time(self, time, variable):
        return self.var_data[variable]
