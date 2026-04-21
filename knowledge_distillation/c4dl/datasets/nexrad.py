import contextlib
from datetime import datetime, timedelta
from io import BytesIO
import os
import warnings
import zipfile

import numpy as np
with contextlib.redirect_stdout(None): # prevent pyart from spamming    
    import pyart

from .datasetreader import DatasetReader
from ..diskcache import get_cache
from ..motion.motion import MotionVectors
from ..utils import CacheDict


def product_from_filename(fn):
    return fn.split("_")[2][:3]


def time_from_filename(fn):
    return datetime.strptime(fn.split("_")[-1], "%Y%m%d%H%M")


class NEXRADReader(DatasetReader):
    name = "nexrad"
    nexrad_products = ["N0Q", "NAQ", "N1Q", "NBQ", "N2Q", "N3Q"]

    def __init__(self, grid_projection, *, archive_path, radars, 
        min_z=0, max_z=17e3, num_levels=35, interval=timedelta(minutes=5),
        variables=["ECHOTOP-25", "ECHOTOP-35", "ECHOTOP-45", "VIL", "MAXZ", "FLOW-U", "FLOW-V"],
        motion_var="MAXZ", min_reflectivity=20,
        cache_size=40):

        super().__init__(grid_projection, variables=variables)

        self.archive_path = archive_path
        self.radars = radars
        self.min_z = min_z
        self.max_z = max_z
        self.num_levels = num_levels
        self.levels = np.linspace(min_z, max_z, num_levels)
        self.interval = interval
        self.min_reflectivity = min_reflectivity
        if ("FLOW-U" in variables) and ("FLOW-V" in variables):
            self.motion_vectors = MotionVectors(self, motion_var=motion_var)
        else:
            self.motion_vectors = None
        self.cache = get_cache(self.name+"_"+"_".join(variables))
        self.motion_cache = get_cache(self.name+"_motion")
        self.grid_cache = get_cache(self.name+"_grid")
        
    def files_for_time(self, time):
        all_files = []
        time_range = (time-self.interval, time)
        for radar in self.radars:
            data_dirs = set(os.path.join(
                self.archive_path,
                t.strftime("%Y"),
                t.strftime("%m"),
                t.strftime("%d"),
                radar
            ) for t in time_range)

            for data_dir in data_dirs:
                try:
                    archive = os.path.join(data_dir, os.listdir(data_dir)[0])
                    with zipfile.ZipFile(archive, 'r') as zip_file:
                        files = zip_file.namelist()
                        files = (fn for fn in files if 
                            product_from_filename(fn) in self.nexrad_products)
                        files = (fn for fn in files if
                            time_range[0] <= time_from_filename(fn) < time_range[1])
                        for fn in files:
                            f = BytesIO(zip_file.read(fn))
                            f.seek(0)
                            yield f
                except FileNotFoundError:
                    pass

    def grid_from_files(self, files):
        radars = [pyart.io.read_nexrad_level3(fn) for fn in files]
        area = self.grid_projection.area
        grid = pyart.map.grid_from_radars(radars, 
            (self.num_levels, area.height, area.width),
            (
                (self.min_z, self.max_z),
                (area.area_extent[1], area.area_extent[3]),
                (area.area_extent[0], area.area_extent[2])
            ),
            grid_projection=area.proj_dict,
            #weighting_function='barnes2',
            roi_func='constant', constant_roi=4000.0,
            #roi_func='dist_beam', nb=2.5,
            weighting_function='cressman'
        )
        return grid

    def variable_from_grid(self, grid, time, variable):
        if variable.startswith("ECHOTOP"):
            echotop_threshold = float(variable.split("-")[-1])
            return self.echo_top(grid, echotop_threshold)
        elif variable.startswith("FLOW"):
            component = 0 if (variable[-1] == "U") else 1
            return self.motion(time, component)
        elif variable == "VIL":
            return self.vertical_integrated_liquid(grid)
        elif variable == "MAXZ":
            return self.max_reflectivity(grid)

    def grid_for_time(self, time):
        if time not in self.grid_cache:
            files = list(self.files_for_time(time))
            if not files:
                raise FileNotFoundError("No files found for {}.".format(
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ))
            self.grid_cache[time] = self.grid_from_files(files)
        return self.grid_cache[time]

    def data_for_time(self, time):
        if time not in self.cache:
            grid = self.grid_for_time(time)
            variables = [self.variable_from_grid(grid, time, v) for v in self.variables]
            variables = np.stack(variables, axis=-1)
            self.cache[time] = variables[::-1,:,:] # pyart flips this axis compared to satpy
        return self.cache[time]

    def variable_for_time(self, time, variable):
        motion_var = self.motion_vectors.motion_var \
            if self.motion_vectors is not None \
            else None
        if (variable == motion_var) and (time not in self.cache):
            # this needs to be available for the motion vector calculation
            grid = self.grid_for_time(time)
            var_data = self.variable_from_grid(grid, time, variable)
            var_data = var_data[::-1,:] # pyart flips this axis compared to satpy
            return var_data
        else:
            return super().variable_for_time(time, variable)

    def echo_top(self, grid, echotop_threshold):
        Z = np.array(grid.fields["reflectivity"]["data"], copy=False).copy()
        mask = grid.fields["reflectivity"]["data"].mask
        Z[mask] = np.nan
        above = (Z >= echotop_threshold)
        inds = above.shape[0]-1-above[::-1,:,:].argmax(axis=0)
        levels = self.levels[inds]
        levels[~above.any(axis=0)] = np.nan
        return levels

    def vertical_integrated_liquid(self, grid):
        Z = np.array(grid.fields["reflectivity"]["data"], copy=False)
        valid = ~grid.fields["reflectivity"]["data"].mask
        M = np.zeros_like(Z)
        Z_lin = 10**(Z[valid]/10)
        M[valid] = 3.44e-6 * Z_lin**(4.0/7.0)
        dz = self.levels[1]-self.levels[0]
        return M.sum(axis=0) * dz

    def max_reflectivity(self, grid):
        Z = np.array(grid.fields["reflectivity"]["data"], copy=False).copy()
        with warnings.catch_warnings(): # suppress warnings from nanmax
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_Z = np.nanmax(Z, axis=0)
        max_Z[max_Z < self.min_reflectivity] = np.nan
        return max_Z

    def motion(self, time, component):
        if time not in self.motion_cache:
            V = self.motion_vectors(time, self.grid_projection.area.shape,
                sparse=False)
            t = self.interval.total_seconds()
            V[:,:,0] *= (self.grid_projection.area.pixel_size_x / t)
            V[:,:,1] *= -(self.grid_projection.area.pixel_size_y / t)
            self.motion_cache[time] = V
        
        return self.motion_cache[time][:,:,component]
