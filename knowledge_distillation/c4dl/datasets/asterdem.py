import netCDF4
import numpy as np
from pyresample.geometry import AreaDefinition
from satpy.resample import prepare_resampler, add_crs_xy_coords
from xarray import DataArray

from .datasetreader import DatasetReader
from ..diskcache import get_cache


class ASTERDEMReader(DatasetReader):
    name = "aster"
    fields = ["mean_elevation", "roughness", "gradient"]

    def __init__(self, grid_projection, *, dem_file, resampler='bilinear',
        variables=None):

        if variables is None:
            variables = ["mean_elevation", "roughness",
                "gradient_x", "gradient_y", "gradient_abs"]
        super().__init__(grid_projection, variables=variables)

        self.var_data = {}
        cache = get_cache(self.name)
        for var in variables:
            if var in cache:
                self.var_data[var] = cache[var]

        if set(self.var_data.keys()) == set(variables):
            return # everything was cached, no need to do anything else

        with netCDF4.Dataset(dem_file, 'r') as ds:
            latlon_data = {
                var: np.array(ds[var][:], copy=False) for var in self.fields
            }
            source_area_params = {
                "description": "latlong_aster",
                "proj_id": "latlong",
                "projection": {"proj": "latlong"},
                "width": len(ds["lon"]),
                "height": len(ds["lat"]),
                "area_id": "latlong_box",
                "area_extent": (
                    float(ds["lon"][0]),
                    float(ds["lat"][-1]),
                    float(ds["lon"][-1]),
                    float(ds["lat"][0])
                )
            }
            source_area = AreaDefinition(**source_area_params)

        (_, resampler) = prepare_resampler(
            source_area, grid_projection.area, resampler
        )
        
        latlon_data["gradient_x"] = latlon_data["gradient"][:,:,0]
        latlon_data["gradient_y"] = latlon_data["gradient"][:,:,1]
        latlon_data["gradient_abs"] = np.sqrt(
            latlon_data["gradient_x"]**2 + latlon_data["gradient_y"]**2
        )

        def resample(var):            
            v = latlon_data[var]
            v = DataArray(v, dims=('y','x'))
            return np.array(resampler.resample(v))

        for var in variables:            
            self.var_data[var] = resample(var)
            cache[var] = self.var_data[var] # cache for later use
       
    def variable_for_time(self, time, variable):
        return self.var_data[variable]
