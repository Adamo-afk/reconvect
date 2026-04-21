from datetime import datetime, timedelta
import os
import zipfile

from netCDF4 import Dataset
import numpy as np
from time import sleep

from ..utils import CacheDict
from .datasetreader import DatasetReader
from .. import projection


class COSMOCCS4Reader(DatasetReader):
    name = "cosmoccs4"

    def __init__(self, *, archive_path, 
            variables=[
                "CAPE_MU", "CIN_MU", "SLI", 
                "HZEROCL", "LCL_ML", "MCONV", "OMEGA",
                "T_2M", "T_SO", "SOILTYP"
            ],
            fc_interval_hr=3, model="cosmo-1",
            var_products=None, default_product="convection",
            min_lead_time_hr=1, cache_size=40
        ):

        grid_projection = projection.GridProjection(
            projection.romania_grid_area)
        super().__init__(grid_projection, variables=variables)

        self.archive_path = archive_path
        self.variables = variables
        self.cache = CacheDict(cache_size=cache_size)
        self.fc_interval_hr = fc_interval_hr
        self.model = model
        self.min_lead_time_hr = min_lead_time_hr
        if var_products is None:
            var_products = {}
        self.products = set(
            var_products.get(v, default_product) for v in variables
        )

    def read_fields_from_archive(self, time, variable, filename):
        # # find appropriate forecast time
        # fc_time = datetime(time.year, time.month, time.day, time.hour)
        # if time.minute >= 30: # round to nearest hour
        #     fc_time += timedelta(hours=1)
        # lead_time = fc_time.hour % self.fc_interval_hr
        # if lead_time < self.min_lead_time_hr:
        #     lead_time += self.fc_interval_hr
        # fc_time -= timedelta(hours=lead_time)

        # zip_filenames = {}
        # for product in self.products:
        #     zip_filenames[product] = os.path.join(
        #         self.archive_path,
        #         self.model,
        #         product,
        #         fc_time.strftime("%Y"),
        #         fc_time.strftime("%m"),
        #         "{}_{}_ccs4.zip".format(
        #             fc_time.strftime("%Y%m%d"),
        #             product
        #         ) 
        #     )

        # var_data = {}
        # for product in self.products:
        #     with zipfile.ZipFile(zip_filenames[product], 'r') as zip_file:                
        #         file_to_read = "{}_{:02d}_{}_{}_swiss.nc".format(
        #             fc_time.strftime("%Y%m%d%H"),
        #             lead_time,
        #             self.model,
        #             product,
        #         )
        #         file_to_read = "/".join([
        #             fc_time.strftime("%Y"),
        #             fc_time.strftime("%m"),
        #             fc_time.strftime("%d"),
        #             file_to_read
        #         ])
        #         with netCDF4.Dataset(None, 'r', memory=zip_file.read(file_to_read)) as ds:
        #             for var in (v for v in ds.variables if (v in self.variables)):
        #                 data = np.array(ds[var][:], copy=False)
        #                 if data.ndim > 2:
        #                     data = data[0,...] # remove time dimension
        #                 assert(data.ndim in (2,3))
        #                 if data.ndim == 2:
        #                     data = data.reshape(data.shape+(1,))
        #                 elif data.ndim == 3:
        #                     data = data.transpose((1,2,0))
        #                 data = data[::-1,...]
        #                 if var in ["CAPE_MU", "CIN_MU"]:
        #                     data.clip(min=0.0, out=data)
        #                 if var == "MCONV":
        #                     data = data[...,-1:]
        #                 var_data[var] = data

        # Read the NetCDF file with netCDF4
        ds = Dataset(filename, 'r')

        # Access the datamap data - navigate through the groups
        data = ds.variables[variable.lower()][:]
        # print(f"Variable {variable.lower()}")
        # sleep(10)

        return data
        # return np.concatenate([var_data[v] for v in self.variables], axis=-1)

    # def data_for_time(self, time):
    def variable_for_time(self, time, variable, filename):
        # if time not in self.cache:
        #     self.cache[time] = self.read_fields_from_archive(time, variable, filename)
        self.cache[time] = self.read_fields_from_archive(time, variable, filename)
        try:
            return self.cache[time]
        except KeyError:
            raise ValueError("No COSMO data found for {}.".format(
                time.strftime("%Y-%m-%d %H:%M%")))
    