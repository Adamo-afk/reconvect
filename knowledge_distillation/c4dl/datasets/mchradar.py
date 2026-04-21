from datetime import datetime, timedelta
import os
from tempfile import TemporaryDirectory
from zipfile import ZipFile
# import metranet
import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr
from netCDF4 import Dataset

from ..utils import CacheDict
from .datasetreader import DatasetReader
from .. import projection

file_dir = os.path.dirname(os.path.abspath(__file__))


# def read_metranet_zip_archive(archive_name, file_name):
#     """Extract desired file from the daily zip archive.
#     """

#     # We must extract the files to a temporary directory first because
#     # the metranet library cannot read from a file object.
#     with TemporaryDirectory() as temp_dir:
#         with ZipFile(archive_name, 'r') as zip_file:
#             temp_path = os.path.join(temp_dir, file_name)
#             with open(temp_path, 'wb') as temp_file:                
#                 temp_file.write(zip_file.read(file_name))

#         mn_data = metranet.read_file(temp_path)
#         data = mn_data.data
#         scale = mn_data.scale

#     return (data, scale)         


class MCHRadarReader(DatasetReader):
    name = "mchradar"

    def __init__(self, *,
            archive_path, 
            variables=["RZC", "CZC", "EZC-20",
                "EZC-45", "BZC", "HZC", "LZC", "CPCH"],
            cache_size=40,
            phys_values=True,
            min_value=None
        ):

        grid_projection = projection.GridProjection(
            projection.romania_grid_area)
        super().__init__(grid_projection, variables=variables)
        self.archive_path = archive_path
        self.cache = {v: CacheDict(cache_size=cache_size) for v in variables}
        self.phys_values = phys_values
        self.min_value = min_value
        self.cpc_lookup = None

    def read_data_and_scale(self, time, variable, filename):
        # print("Reading data and scale")
        # print(self.cache)
        if time not in self.cache[variable]:
            # # not necessary (for reading gif zip archives)
            # datestamp = pd.to_datetime(time).strftime("%y%j") # not necessary
            # timestamp = pd.to_datetime(time).strftime("%y%j%H%M") # not necessary
            # day_dir = os.path.join(
            #     self.archive_path,
            #     pd.to_datetime(time).strftime("%Y"),
            #     datestamp
            # ) # not necessary       

            # if variable in ["RZC", "CZC", "HZC", "LZC", "CPC", "CPCH"]:
                # level = "01"
                # var_code = variable
            # elif variable == "BZC":
                # level = "45"
                # var_code = variable
            # elif variable.startswith("EZC"): 
                # level = variable.split("-")[1]
                # var_code = "EZC"
            
            # if variable in ["CPC", "CPCH"]:
            #     timestamp += "X_00005"
            # else:
            #     timestamp += "VL"
            
            # zip_filename = "{}{}.8{}".format(var_code[:3], timestamp, level) # not necessary
            # zip_name = "{}{}.zip".format(var_code, datestamp) # not necessary
            # zip_path = os.path.join(day_dir,zip_name) # not necessary

            try:
                # if variable in ["CPC", "CPCH"]:
                    # (data, _) = self.read_gif_zip_archive(zip_path, zip_filename)
                # else:
                    # (data, scale) = read_metranet_zip_archive(zip_path, zip_filename)
                    
                    # # Read the NetCDF file
                    # ds = xr.open_dataset(os.path.join(self.archive_path, os.listdir(self.archive_path)[0]))

                    # # Look at the contents
                    # print(ds)

                    # # Access specific variables
                    # data = ds[variable].values

                    ################################ OLD CODE ################################
                    # # Read the NetCDF file with netCDF4
                    # ds = Dataset(filename, 'r')

                    # # Access the datamap data - navigate through the groups
                    # data = ds.groups['data'].groups['radarpicture'].groups['datamap'].variables['datamap'][:]
                    
                data = np.load(filename)

                # self.cache[variable][time] = (data, scale)
                self.cache[variable][time] = data
            except (KeyError, FileNotFoundError):
                # zip archive not found, or file not found in archive 
                # this will trigger a cache miss below, handled there
                pass 

        try:
            # (data, scale) = self.cache[variable][time]
            data = self.cache[variable][time]
        except KeyError:
            # raise ValueError("No radar data found for {}.".format(
            #     time.strftime("%Y-%m-%d %H:%M")))
            raise ValueError("No radar data found for {}.".format(time))

        # return (data, scale)
        return data


    def variable_for_time(self, time, variable, filename):
        """Read the radar data for a given time.

        Args:
            Time: Python datetime object giving the desired time.
        """
        data = self.read_data_and_scale(time, variable, filename)
        
        # if self.phys_values:
        #     data = scale[data]
        if self.min_value is not None:
            data[data < self.min_value] = np.nan

        return data

    # def get_scale(self, time, variable):
    #     (_, scale) = self.read_data_and_scale(time, variable, filename)
    #     return scale


    # def read_gif_zip_archive(self, archive_name, file_name):
    #     file_name += ".gif"
    #     with ZipFile(archive_name, 'r') as zip_file:
    #         names = set(zip_file.namelist())
    #         for quality_flag in range(9,-1,-1):
    #             fn = file_name.replace("X", str(quality_flag))
    #             if fn in names:
    #                 break
    #         else:
    #             raise FileNotFoundError()

    #         with zip_file.open(fn, 'r') as f:
    #             with Image.open(f, 'r', formats=["GIF"]) as im:
    #                 data = np.array(im)

    #     if self.cpc_lookup is None:
    #         lookup_fn = os.path.join(file_dir, "cpc_lookup.txt")
    #         self.cpc_lookup = np.loadtxt(lookup_fn, skiprows=2, usecols=2)
    #     scale = self.cpc_lookup.copy()

    #     return (data, scale)
