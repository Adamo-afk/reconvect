from datetime import datetime, timedelta
import os
import warnings

import numpy as np
import satpy

from .datasetreader import DatasetReader
from ..diskcache import get_cache
from .parallax import ParallaxCorrection


def channel_from_filename(fn):
    return fn.split("_")[1].split("-")[3][-2:]


goes_time_format = "%Y%j%H%M%S%f"
def time_from_filename(fn):
    start_timestamp = fn.split("_")[3][1:]
    start_time = datetime.strptime(start_timestamp, goes_time_format)
    return start_time


L2_var_names = {
    "HT": "ACHA",
    "PRES": "CTP",
    "CAPE": "DSI",
    "KI": "DSI",
    "LI": "DSI",
    "SI": "DSI",
    "TT": "DSI",
    "COD": "COD"
}
def product_name_for_var(var, level):
    if level == "L1b":
        return var[-2:]
    if level == "L2":
        return L2_var_names[var]


region_codes = {
    "fulldisk": "F",
    "conus": "C"
}


def filter_datasets(dataset_names):
    # remove datasets not supported in satpy
    return list(set(dataset_names)-set(["CODN", "CODD"]))


class GOESABIReader(DatasetReader):
    name = "goesabi"

    def __init__(self, grid_projection, *, archive_path,
        region='fulldisk', variables=None, max_time_offset=timedelta(minutes=8),
        resampler='nearest', parallax_correct=True, cache_size=40,
        ):

        self.archive_path = archive_path
        self.region = region_codes[region.lower()]
        self.max_time_offset = max_time_offset

        if variables is None:
            variables = ["ABIC{:02d}".format(channel)
                for channel in range(1,17)]
            variables += [
                "HT", "PRES", "CAPE", "KI", "LI", "SI", "TT", "COD"
            ]
        super().__init__(grid_projection, variables=variables)
        
        self.resampler = resampler
        self.parallax_correct = parallax_correct
        self.parallax_correction = None
        if parallax_correct and resampler != 'nearest':
            raise ValueError(
                "Parallax correction only supported with the NN resampler.")
        if parallax_correct and ("HT" not in variables):
            raise ValueError(
                "Parallax correction requires a cloud top height variable.")
        
        self.cache = get_cache(self.name+"_"+"_".join(variables))

    def nearest_file(self, files, time):
        delta_t = [
            abs((time_from_filename(fn)-time).total_seconds())
            for fn in files
        ]
        nearest = np.array(delta_t).argmin()
        if (delta_t[nearest] > self.max_time_offset.total_seconds()):
            raise FileNotFoundError("File not found within time range.")
        return files[nearest]

    def files_for_time(self, time):
        found_products = []
        for variable in self.variables:
            level = "L1b" if variable.startswith("ABIC") else "L2"
            product = product_name_for_var(variable, level)
            if (level,product) in found_products:
                continue
            else:
                found_products.append((level,product))

            if level == "L1b":
                data_dir = os.path.join(
                    self.archive_path,
                    "ABI-L1b-Rad{}".format(self.region),
                    time.strftime("%Y"),
                    time.strftime("%j"),
                    time.strftime("%H")
                )
                files = os.listdir(data_dir)         
                files = [fn for fn in files if 
                    channel_from_filename(fn) == product]
            elif level == "L2":
                data_dir = os.path.join(
                    self.archive_path,
                    "ABI-L2-{}{}".format(product, self.region),
                    time.strftime("%Y"),
                    time.strftime("%j"),
                    time.strftime("%H")
                )
                files = os.listdir(data_dir)
            
            nearest_fn = self.nearest_file(files, time)
            yield (level, os.path.join(data_dir,nearest_fn))

    def data_for_files(self, files):
        files_by_level = {"L1b": [], "L2": []}
        for (level, fn) in files:
            files_by_level[level].append(fn)
        
        var_data = {}
        readers = {"L1b": "abi_l1b", "L2": "abi_l2_nc"}
        scenes = {}
        for level in readers:
            if files_by_level[level]:
                try:
                    scenes[level] = satpy.Scene(reader=readers[level],
                        filenames=files_by_level[level])
                except ValueError as e:
                    if str(e).startswith("unable to decode time units"):
                        raise ValueError(str(e) + " " + str(files_by_level[level]))
                    else:
                        raise

                scenes[level].load(filter_datasets(
                    scenes[level].available_dataset_names()))
        
        if self.parallax_correct:
            if (not "L2" in scenes) or (not "HT" in scenes["L2"]):
                raise ValueError("Parallax correction wanted but no "
                    "cloud top height found.")
            if self.parallax_correction is None:
                self.parallax_correction = ParallaxCorrection(
                    self.grid_projection.area)
            area = self.parallax_correction(scenes["L2"]["HT"])
        else:
            area = self.grid_projection.area

        for level in scenes:
            local_scene = scenes[level].resample(
                area, resampler=self.resampler
            )
            for product in filter_datasets(scenes[level].available_dataset_names()):
                if product not in L2_var_names:
                    var_name = "ABI" + product
                else:
                    var_name = product
                var_data[var_name] = local_scene[product]

        data = np.ma.empty((
            self.grid_projection.area.height,
            self.grid_projection.area.width,
            len(self.variables)
        ))
        data[:] = np.nan

        for (i,var) in enumerate(self.variables):
            if var in var_data:
                data[:,:,i] = var_data[var]

        return data

    def data_for_time(self, time):
        if time not in self.cache:
            files = self.files_for_time(time)
            data = self.data_for_files(files)
            self.cache[time] = data
        return self.cache[time]
