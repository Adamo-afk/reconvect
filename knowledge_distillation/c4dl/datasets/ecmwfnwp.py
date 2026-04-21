from datetime import datetime, timedelta
import os

import numpy as np
import pygrib
from pyresample.geometry import AreaDefinition
from satpy.resample import add_crs_xy_coords, prepare_resampler
from xarray import DataArray

from .datasetreader import DatasetReader
from ..diskcache import get_cache


def round_to_nearest_hour(time):
    rounded_time = datetime(time.year,time.month,time.day,time.hour)
    if time.minute >= 30:
        rounded_time += timedelta(hours=1)
    return rounded_time


class ECMWFNWPReader(DatasetReader):
    name = "ecmwf"
    fields = [
        '100u', '100v', '10u', '10v', '200u', '200v', '2d', '2t', 'bld',
        'blh', 'cape', 'capes', 'cbh', 'cin', 'cp', 'crr', 'csfr', 'deg0l',
        'e', 'hcc', 'hcct', 'hwbt0', 'hwbt1', 'kx', 'lcc', 'lsp', 'lspf',
        'lsrr', 'lssfr', 'mcc', 'msl', 'pev', 'ptype', 'sf', 'skt', 'slhf',
        'sp', 'src', 'sshf', 'ssr', 'ssrc', 'str', 'strc', 'tcc', 'tciw',
        'tclw', 'tcrw', 'tcslw', 'tcsw', 'tcw', 'tcwv', 'totalx', 'tp',
        'tprate', 'vimd', 'viwve', 'viwvn', 'z', 'zust'
    ]

    def __init__(self, grid_projection, *, archive_path, 
        variables=None, cache_size=40, resampler_type='bilinear'):

        if variables is None:
            variables = self.fields
        super().__init__(grid_projection, variables=variables)

        self.archive_path = archive_path
        self.resampler = None
        self.resampler_source_area = None
        self.resampler_type = resampler_type
        self.cache = get_cache(self.name)

    def file_for_time(self, time, max_search_days=2):
        day = datetime(time.year, time.month, time.day)

        available_files = {}
        for i in range(max_search_days):
            file_dir = os.path.join(
                self.archive_path,
                day.strftime("%Y"),
                day.strftime("%m"),
                day.strftime("%d"),
            )
            files = os.listdir(file_dir)
            for fn in files:
                parts = fn.split(".")[0].split("-")
                init_time = datetime.strptime(parts[2], "%Y%m%d%H")
                start_time = init_time + timedelta(hours=int(parts[3]))
                end_time = init_time + timedelta(hours=int(parts[4]))
                if start_time <= time <= end_time:
                    lead_time = time-init_time
                    available_files[lead_time] = os.path.join(file_dir,fn)
        
        min_lead_time = min(available_files) # finds the smallest key
        return available_files[min_lead_time]

    def get_resampler(self, source_area):
        if (self.resampler_source_area != source_area):
            (_, self.resampler) = prepare_resampler(
                source_area, self.grid_projection.area, self.resampler_type
            )
            self.resampler_source_area = source_area
        
        return self.resampler

    def area_from_lonlat(self, lon, lat):
        # the data are in a regular lat-lon grid
        lon = lon[0,:]
        lat = lat[:,0]
        
        area_params = {
            "description": "latlong_ecmwf",
            "proj_id": "latlong",
            "projection": {"proj": "latlong"},
            "width": len(lon),
            "height": len(lat),
            "area_id": "latlong_box",
            "area_extent": (
                float(lon[0]),
                float(lat[-1]),
                float(lon[-1]),
                float(lat[0])
            )
        }
        return AreaDefinition(**area_params)
        
    def data_for_file(self, fn, time):
        (lons, lats) = (None, None)
        latlon_data = {}
        # yes, these values are stored as int in the grib files!
        date_int = int(time.strftime("%Y%m%d"))
        time_int = int(time.strftime("%H%M"))

        with pygrib.open(fn) as grbs:            
            selected = grbs(
                shortName=self.variables,
                validityDate=date_int,
                validityTime=time_int,
            )
            for grb in selected:
                var = grb.shortName
                if lats is None:
                    (lats, lons) = grb.latlons()
                # for each variable,
                # select the forecast with the shortest lead time
                if (var not in latlon_data) or \
                    (grb.step < latlon_data[var][0]):

                    latlon_data[var] = (grb.step, grb.values)
        
        # check that all desired variables were found
        var_set = set(self.variables)
        if var_set & set(latlon_data) != var_set:
            raise KeyError(
                "Not all variables were found in the ECMWF GRIB file."
            )
        
        # resample data
        source_area = self.area_from_lonlat(lons, lats)
        resampler = self.get_resampler(source_area)

        def resample(var):
            v = latlon_data[var][1]
            v = DataArray(v, dims=('y','x'))
            return np.array(resampler.resample(v))
        data = {v: resample(v) for v in self.variables}

        return data

    def data_for_time(self, time):
        fc_time = round_to_nearest_hour(time)
        
        if fc_time not in self.cache:
            fn = self.file_for_time(fc_time)
            data = self.data_for_file(fn, fc_time)
            data_arr = np.stack([data[v] for v in self.variables], axis=-1)
            self.cache[fc_time] = data_arr

        return self.cache[fc_time]
