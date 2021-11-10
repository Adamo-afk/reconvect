from datetime import datetime, timedelta
import os
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import metranet
import numpy as np

from ..utils import CacheDict
from .datasetreader import DatasetReader
from .. import projection


def read_metranet_zip_archive(archive_name, file_name):
    """Extract desired file from the daily zip archive.
    """

    # We must extract the files to a temporary directory first because
    # the metranet library cannot read from a file object.
    with TemporaryDirectory() as temp_dir:
        with ZipFile(archive_name, 'r') as zip_file:
            temp_path = os.path.join(temp_dir, file_name)
            with open(temp_path, 'wb') as temp_file:                
                temp_file.write(zip_file.read(file_name))

        mn_data = metranet.read_file(temp_path)
        data = mn_data.data
        scale = mn_data.scale

    return (data, scale)


class MCHRadarReader(DatasetReader):
    name = "mchradar"

    def __init__(self, *,
            archive_path, 
            variables=["RZC", "CZC", "EZC-15", "EZC-20",
                "EZC-45", "EZC-50", "BZC", "HZC", "LZC"],
            cache_size=40,
            phys_values=True,
            min_value=None
        ):

        grid_projection = projection.GridProjection(
            projection.ccs4_swiss_grid_area)
        super().__init__(grid_projection, variables=variables)
        self.archive_path = archive_path
        self.cache = {v: CacheDict(cache_size=cache_size) for v in variables}
        self.phys_values = phys_values
        self.min_value = min_value

    def read_data_and_scale(self, time, variable):
        if time not in self.cache[variable]:
            datestamp = time.strftime("%y%j")
            timestamp = time.strftime("%y%j%H%M")
            day_dir = os.path.join(
                self.archive_path,
                time.strftime("%Y"),
                datestamp
            )        

            if variable in ["RZC", "CZC", "HZC", "LZC"]:
                level = "01"
                var_code = variable
            elif variable == "BZC":
                level = "45"
                var_code = variable
            elif variable.startswith("EZC"):                
                level = variable.split("-")[1]
                var_code = "EZC"
            zip_filename = "{}{}VL.8{}".format(var_code, timestamp, level)

            zip_name = "{}{}.zip".format(var_code, datestamp)
            zip_path = os.path.join(day_dir,zip_name)

            try:
                (data, scale) = read_metranet_zip_archive(zip_path, zip_filename)                
                self.cache[variable][time] = (data, scale)
            except (KeyError, FileNotFoundError):
                # zip archive not found, or file not found in archive 
                # this will trigger a cache miss below, handled there
                pass 

        try:
            (data, scale) = self.cache[variable][time]
        except KeyError:
            raise ValueError("No radar data found for {}.".format(
                time.strftime("%Y-%m-%d %H:%M")))

        return (data, scale)


    def variable_for_time(self, time, variable):
        """Read the radar data for a given time.

        Args:
            Time: Python datetime object giving the desired time.
        """
        (data, scale) = self.read_data_and_scale(time, variable)
        
        if self.phys_values:
            data = scale[data]
            if self.min_value is not None:
                data[data < self.min_value] = np.nan

        return data

    def get_scale(self, time, variable):
        (data, scale) = self.read_data_and_scale(time, variable)
        return scale
