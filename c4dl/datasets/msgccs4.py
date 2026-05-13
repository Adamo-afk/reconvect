from datetime import datetime
from io import BytesIO
import os
import zipfile

import netCDF4
import numpy as np

from ..utils import CacheDict
from .datasetreader import DatasetReader

from .. import projection
from . import solar
from netCDF4 import Dataset

class MSGRadianceCCS4Reader(DatasetReader):
    name = "msg"

    visible_thresholds = {
        "HRV": 20,
        "VIS006": 20,
        "VIS008": 47,
        "IR_016": 35
    }

    def __init__(
            self, *, archive_path,
            variables=[
                "HRV",
                "VIS006", "VIS008",
                "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
                "WV_062", "WV_073"
            ],
            normalize_visible=False,
            min_solar_angle=10,
            threshold_visible=False
        ):

        grid_projection = projection.GridProjection(
            projection.romania_grid_area)
        super().__init__(grid_projection, variables=variables)
        
        self.archive_path = archive_path      
        self.normalize_visible = normalize_visible
        self.min_solar_angle = min_solar_angle
        if normalize_visible:
            self.solar_reader = solar.SolarReader(grid_projection=grid_projection)
        self.threshold_visible = True
        self.cache = CacheDict(cache_size=600)

    def read_fields_from_archive(self, time, filename):
        # filename = os.path.join(
        #     self.archive_path,
        #     time.strftime("%Y"),
        #     time.strftime("%m"),
        #     time.strftime("%Y%m%d_MSG-nc-alps-PLAX-RSS_ccs4.zip")
        # )
        # fn_in_archive = "/".join([
        #     time.strftime("%Y"),
        #     time.strftime("%m"),
        #     time.strftime("%d"),
        #     time.strftime("MSG3_ccs4_%Y%m%d%H%M_rad_PLAX.nc")
        # ])

        # with zipfile.ZipFile(filename, 'r') as zip_file:
        #     nc_data = zip_file.read(fn_in_archive)

        # with netCDF4.Dataset(None, 'r', memory=nc_data) as ds:
        
        ################################ OLD CODE ################################
        # # Read the NetCDF file with netCDF4
        # ds = Dataset(filename, 'r')

        # for var in self.variables:
        #     if var in ds.variables:
        #         fields = np.array(ds[var][:], copy=False)

        # # Fill NaN values with 0.0
        # np.nan_to_num(fields, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        # fields = [
        #     np.array(ds[var][:], copy=False)
        #     for var in self.variables if var in ds.variables
        # ]

        fields = np.load(filename)

        if self.normalize_visible:
            z_threshold = np.sin(self.min_solar_angle*np.pi/180.0)
            z = self.solar_reader.variable_for_time(time, "sun_z")
            z = z.reshape(z.shape+(1,))
            valid = (z > z_threshold)
            # for (i,var) in enumerate(self.variables):
            for var in self.variables:
                if (var in MSGRadianceCCS4Reader.visible_thresholds):
                    # fields[i][~valid] = 0
                    # fields[i][valid] /= z[valid]
                    fields[~valid] = 0
                    fields[valid] /= z[valid]

        # if self.threshold_visible:
        #     for (i,var) in enumerate(self.variables):
        #         if (var in MSGRadianceCCS4Reader.visible_thresholds):
        #             threshold = MSGRadianceCCS4Reader.visible_thresholds[var]
        #             # fields[i][fields[i] < threshold] = 0
        #             fields[fields < threshold] = 0

        # return np.concatenate(fields, axis=-1).astype(np.float32)
        # print(fields)
        # print(fields.shape)
        # print(f"Min value: {np.min(fields)}, Max value: {np.max(fields)}")
        # exit(0)
        return fields.astype(np.float32)

    def variable_for_time(self, time, variable, filename): # given 3 arguments as radar data
        # DOES NOT need all variables, only time and filename
        if time not in self.cache:
            self.cache[time] = self.read_fields_from_archive(time, filename)
        
        try:
            # print(self.cache[time])
            # print(self.cache[time].shape)
            # print(f"Min value: {np.min(self.cache[time])}, Max value: {np.max(self.cache[time])}")
            # exit(0)
            return self.cache[time]
        except KeyError:
            raise ValueError("No MSG radiance data found for {}.".format(
                time.strftime("%Y-%m-%d %H:%M%")))
    

class NWCSAFCCS4Reader(DatasetReader):
    # name = "nwcsaf"

    # path_CTTH = "CTTH/S_NWC_CTTH_MSG3_alps-VISIR_{}Z_PLAX.nc"
    # path_CMIC = "CMIC/S_NWC_CMIC_MSG3_alps-VISIR_{}Z_PLAX.nc"
    # path_CT = "CT/S_NWC_CT_MSG3_alps-VISIR_{}Z_PLAX.nc"

    # var_paths = {
    #     "ctth_pres": path_CTTH,
    #     "ctth_alti": path_CTTH,
    #     "ctth_tempe": path_CTTH,
    #     "cmic_phase": path_CMIC,
    #     "cmic_reff": path_CMIC,
    #     "cmic_cot": path_CMIC,
    #     "cmic_lwp": path_CMIC,
    #     "cmic_iwp": path_CMIC,
    #     "cmic_status_flag": path_CMIC,
    #     "cmic_conditions": path_CMIC,
    #     "cmic_quality": path_CMIC,
    #     "ct": path_CT,
    #     "ct_cumuliform": path_CT,
    #     "ct_multilayer": path_CT,
    #     "ct_status_flag": path_CT,
    #     "ct_conditions": path_CT,
    #     "ct_quality": path_CT,
    # }

    def __init__(self, grid_projection, *, archive_path, 
            variables=[
                "ctth_alti", "ctth_tempe",
                "cmic_cot", "cmic_phase", "cmic_reff", "cmic_lwp", "cmic_iwp",
                "ct"
            ],
            phys_values=True
        ):
        super().__init__(grid_projection, variables=variables)
        
        self.source_area = projection.geostationary_area_alps()
        self.mapper = projection.ImageMapper(
            self.source_area, self.grid_projection.area)
        self.archive_path = archive_path
        self.phys_values = phys_values
        self.cache = CacheDict(cache_size=600)

    def read_fields_from_archive(self, time, variable, filename):

        from legacy.process_nwcsaf import read_and_scale_nwcsaf_variable

        # if variables is None:
        #     variables = self.variables
        # var_paths = {v: self.var_paths[v] for v in variables}

        # # path to zip file
        # zip_filename = os.path.join(
        #     self.archive_path,
        #     time.strftime("%Y"),
        #     time.strftime("%m"),
        #     time.strftime("%Y%m%d_NWCSAF-v2016-alps_alps.zip")
        # )

        # # determine which files to extract from zip
        # timestamp = time.strftime("%Y%m%dT%H%M%S")
        # files_to_read = set(var_paths[var] for var in variables)
        # fns_in_archive = {
        #     fn:
        #     "/".join([
        #         time.strftime("%Y"),            
        #         time.strftime("%m"),
        #         time.strftime("%d"),
        #         fn.format(timestamp)
        #     ])
        #     for fn in files_to_read
        # }

        # # read netcdf files from zip file to memory
        # with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        #     nc_data = {
        #         fn: 
        #         BytesIO(zip_file.read(fns_in_archive[fn]))
        #         for fn in fns_in_archive
        #     }

        # scale_factor = []
        # add_offset = []
        # valid_range = []

        # var_data = {}
        # for fn in files_to_read:
        #     fn_vars = (v for v in var_paths if var_paths[v]==fn)
        #     with netCDF4.Dataset(None, 'r', memory=nc_data[fn].read()) as ds:
        #         if not self.phys_values:
        #             ds.set_auto_maskandscale(False)
        #         for var in fn_vars:
        #             source_data = np.array(ds[var][:], copy=False)
        #             proj_data = self.mapper(source_data, order=0)
        #             var_data[var] = proj_data
        #             if hasattr(ds[var], "scale_factor"):
        #                 scale_factor.append(ds[var].scale_factor)
        #                 add_offset.append(ds[var].add_offset)
        #                 valid_range.append(ds[var].valid_range)
        #             else:
        #                 scale_factor.append(None)
        #                 add_offset.append(None)
        #                 valid_range.append(None)

        # var_data = np.stack([var_data[v] for v in variables], axis=-1)

        if 'cmic' in variable:
            # result = read_and_scale_nwcsaf_variable(filename[0], variable, resampling_method="near")
            result = read_and_scale_nwcsaf_variable(filename[0], variable)
        elif 'ctth' in variable:
            # result = read_and_scale_nwcsaf_variable(filename[1], variable, resampling_method="near")
            result = read_and_scale_nwcsaf_variable(filename[1], variable)
        return result
        
        # return (var_data, scale_factor, add_offset, valid_range)

    # def data_for_time(self, time):
    #     if time not in self.cache:
    #         self.cache[time] = self.read_fields_from_archive(time)[0]
    #     try:
    #         return self.cache[time]
    #     except KeyError:
    #         raise ValueError("No NWCSAF data found for {}.".format(
    #             time.strftime("%Y-%m-%d %H:%M%")))

    def variable_for_time(self, time, variable, filename):
        # d = self.read_fields_from_archive(time, variables=[variable])[0]
        data = self.read_fields_from_archive(time, variable, filename)
        return data['data']
        # return d[...,0]

    # def get_scale(self, time, variable):
    #     (scale_factor, add_offset, valid_range) = \
    #         self.read_fields_from_archive(time, variables=[variable])[1:]

    #     if variable in ["ct", "cmic_phase"]:
    #         scale = None
    #     else:
    #         scale = np.arange(65536, dtype=np.float32)
    #         scale.clip(min=valid_range[0][0], 
    #             max=valid_range[0][1], out=scale)
    #         scale *= scale_factor[0]
    #         scale += add_offset[0]

    #     return scale
