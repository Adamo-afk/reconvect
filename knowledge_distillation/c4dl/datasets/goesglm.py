from datetime import datetime, timedelta
import os
from zipfile import ZipFile

import numpy as np
import netCDF4
import satpy

from .datasetreader import DatasetReader
from ..diskcache import get_cache
from .goesabi import time_from_filename, region_codes
from .gridding import grid_accumulate
from .parallax import ParallaxCorrection


class GOESGLMReader(DatasetReader):
    name = "goesglm"

    def __init__(self, grid_projection, *, archive_path,
        variables=None, 
        parallax_correct=False, cth_archive_path=None, cth_region='fulldisk',
        interval=timedelta(minutes=5), cache_size=40):

        if variables is None:
            variables = ["flash_density", "flash_energy_density",
                "event_density", "event_energy_density"]
        super().__init__(grid_projection, variables=variables)

        self.archive_path = archive_path
        self.interval = interval
        
        self.parallax_correct = parallax_correct
        if parallax_correct:
            if cth_archive_path is None:
                cth_archive_path = archive_path
            self.parallax_correction = ParallaxCorrection(
                grid_projection.area)
        self.cth_archive_path = cth_archive_path
        self.cth_region = region_codes[cth_region]
        
        self.cache = get_cache(self.name+"_"+"_".join(variables))

    def cth_file_for_time(self, time):
        t0 = time-self.interval
        t1 = time

        data_dir = os.path.join(
            self.cth_archive_path,
            "ABI-L2-ACHA{}".format(self.cth_region),
            t0.strftime("%Y"),
            t0.strftime("%j"),
            t0.strftime("%H")
        )
        files = os.listdir(data_dir)
        files = (
            os.path.join(data_dir,fn) 
            for fn in files 
            if t0 <= time_from_filename(fn) < t1
        )
        return sorted(files, key=time_from_filename)[-1]

    def files_for_time(self, time):
        t0 = time-self.interval
        t1 = time

        data_dir = os.path.join(
            self.archive_path,
            "GLM-L2-LCFA",
            t0.strftime("%Y"),
            t0.strftime("%j"),
            t0.strftime("%H")
        )

        files = os.listdir(data_dir)
        zip_files = [fn for fn in files if fn.endswith(".zip")]
        if zip_files:
            with ZipFile(os.path.join(data_dir,zip_files[0])) as zf:
                files = zf.namelist()

        files = [
            os.path.join(data_dir,fn) 
            for fn in files 
            if t0 <= time_from_filename(fn) < t1
        ]
        return files

    def accumulate_data_for_file(self, fn, grid_data, cth=None):
        glm_var_names = ["event_lat", "event_lon", "event_energy",
            "flash_lat", "flash_lon", "flash_energy"]

        if os.path.isfile(fn):
            with open(fn, 'rb') as f:
                data = f.read()
        else:
            (data_dir, data_file) = os.path.split(fn)
            files = os.listdir(data_dir)
            zip_fn = [fn for fn in files if fn.endswith(".zip")][0]            
            with ZipFile(os.path.join(data_dir,zip_fn)) as zf:
                data = zf.read(data_file)

        with netCDF4.Dataset(None, 'r', memory=data) as ds:
            glm_data = {
                v: np.array(ds[v][:], copy=False)
                for v in glm_var_names
            }

        (event_lon, event_lat, event_energy) = (
            glm_data["event_lon"],
            glm_data["event_lat"],
            glm_data["event_energy"]
        )
        (flash_lon, flash_lat, flash_energy) = (
            glm_data["flash_lon"],
            glm_data["flash_lat"],
            glm_data["flash_energy"]
        )

        if cth is not None:
            n_events = len(event_lon)
            lons = np.concatenate((event_lon, flash_lon))
            lats = np.concatenate((event_lat, flash_lat))
            (lons, lats) = self.parallax_correction.correct_points(cth, lons, lats)
            event_lon = lons[:n_events]
            event_lat = lats[:n_events]
            flash_lon = lons[n_events:]
            flash_lat = lats[n_events:]
            event_energy = event_energy[~event_lon.mask]
            event_lon = event_lon.data[~event_lon.mask]
            event_lat = event_lat.data[~event_lat.mask]
            flash_energy = flash_energy[~flash_lon.mask]
            flash_lon = flash_lon.data[~flash_lon.mask]
            flash_lat = flash_lat.data[~flash_lat.mask]            

        (event_i, event_j) = self.grid_projection(event_lon, event_lat)
        (flash_i, flash_j) = self.grid_projection(flash_lon, flash_lat)

        grid_accumulate(event_i, event_j, grid_data["event_density"],
            weights=event_energy,
            weighted_grid=grid_data["event_energy_density"])
        grid_accumulate(flash_i, flash_j, grid_data["flash_density"],
            weights=flash_energy,
            weighted_grid=grid_data["flash_energy_density"])

        return grid_data

    def data_for_time(self, time):
        if time not in self.cache:
            files = self.files_for_time(time)
            if not files:
                raise FileNotFoundError("No GLM files found for {}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ))

            shape = (
                self.grid_projection.area.height,
                self.grid_projection.area.width
            )
            grid_data = {var: np.zeros(shape) for var in self.variables}

            if self.parallax_correct:
                cth_file = self.cth_file_for_time(time)
                cth_scene = satpy.Scene(reader="abi_l2_nc",
                    filenames=[cth_file])
                cth_scene.load(["HT"])
                cth = cth_scene["HT"]
            else:
                cth = None

            for fn in files:
                self.accumulate_data_for_file(fn, grid_data, cth=cth)

            pixel_km2 = self.grid_projection.area.pixel_size_x * \
                self.grid_projection.area.pixel_size_y * 1e-6
            time_hr = 1 / (self.interval.total_seconds() / 3600)
            density_weight = 1 / (time_hr * pixel_km2)
            for v in self.variables:
                grid_data[v] *= density_weight

            self.cache[time] = np.stack(
                [grid_data[v] for v in self.variables],
                axis=-1
            )

        return self.cache[time]
