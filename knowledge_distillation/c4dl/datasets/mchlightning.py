from datetime import datetime, timedelta
from io import BytesIO
import os

import numpy as np
import pandas as pd
from scipy.signal import convolve

from .datasetreader import DatasetReader
from .. import projection
from . import smoothing
from ..utils import CacheDict
from .gridding import grid_accumulate

from netCDF4 import Dataset


# def read_lightning_archive_file(file, discharge_type):
#     data = pd.read_csv(
#         file, 
#         names=('date','lon','lat','current','nS','mode','intra',
#             'Ax','Ki2','Exc','Incl','Arc','d1','d2','d3','d4','d5'),
#         delimiter="|",
#         parse_dates=['date'], 
#         date_parser=lambda x: pd.to_datetime(x, format="%d.%m.%Y %H:%M:%S.%f UTC"),
#         index_col=0,
#         memory_map=True
#     )
#     if "CG" in discharge_type and "IC" in discharge_type:
#         pass
#     elif "CG" in discharge_type:
#         data = data.loc[data.intra == 0]
#     elif "IC" in discharge_type:
#         data = data.loc[data.intra == 1]
#     else:
#         raise ValueError(f"Unsupported discharge type argument {discharge_type}")

#     return data[["lon","lat","current","intra"]]


# def read_lightning_archive(archive_path, day, discharge_type):
#     filename = os.path.join(
#         archive_path,
#         day.strftime("THX%y%j0000.prd")
#     )
#     all_lightning = read_lightning_archive_file(filename, discharge_type)
#     all_lightning.sort_index(inplace=True)
    
#     return all_lightning


class MCHLightningReader(DatasetReader):
    name = "mchlightning"

    var_params = {
        "occurrence": {
            "radius_mul": 1,
            "smoothing_func": smoothing.tophat,
            "normalize_filter": False,
            "data_map": "lightning"
        },
        "density": {
            "radius_mul": 4,
            "smoothing_func": smoothing.gaussian,
            "normalize_filter": True,
            "data_map": "lightning"
        },
        "current": {
            "radius_mul": 4,
            "smoothing_func": smoothing.gaussian,
            "normalize_filter": True,
            "data_map": "current"
        },
    }


    def __init__(self, grid_projection, *, archive_path,
        interval=timedelta(minutes=5), mode="archive",
        variables=["density-10", "occurrence-10", "current-10"], discharge_type="CG"):

        super().__init__(grid_projection, variables=variables)

        self.archive_path = archive_path
        self.interval = interval
        self.mode = mode
        self.days = CacheDict(cache_size=3)
        self.lightning_cache = CacheDict(cache_size=128)
        self.density_cache = CacheDict(cache_size=128)
        self.discharge_type = discharge_type


    # def lightning_for_time(self, time, interval=None):
    #     if interval is None:
    #         interval = self.interval
    #     if time not in self.lightning_cache:
    #         if self.mode == "archive":
    #             t = time-interval
    #             day = datetime(t.year,t.month,t.day)
    #             if not day in self.days:
    #                 self.days[day] = read_lightning_archive(self.archive_path, day, self.discharge_type)
    #             day = self.days[day]
    #             (i0,i1) = day.index.searchsorted([t, time])
    #             self.lightning_cache[time] = day.iloc[i0:i1]

    #     return self.lightning_cache[time]


    # def lightning_maps(self, time, variable):
    #     var_parts = variable.split("-")
    #     var_type = var_parts[0]
    #     var_p = MCHLightningReader.var_params[var_type]

    #     if len(var_parts) > 2:
    #         interval_mins = float(var_parts[2])
    #         interval = timedelta(minutes=interval_mins)
    #     else:
    #         interval = None # use default

    #     lightning = self.lightning_for_time(time, interval=interval)        
    #     lon = lightning["lon"].values
    #     lat = lightning["lat"].values        
    #     (i,j) = self.grid_projection(lon,lat)
        
    #     lmap = np.zeros((
    #         self.grid_projection.area.height,
    #         self.grid_projection.area.width
    #     ))
    #     if var_p["data_map"] == "current":
    #         current = abs(lightning["current"].values)
    #         grid_accumulate(i, j, weights=current, weighted_grid=lmap)
    #     else:
    #         grid_accumulate(i, j, grid=lmap)
        
    #     if len(var_parts) > 1:
    #         smoothing_scale = float(var_parts[1])
    #         smoothing_rad = int(np.ceil(smoothing_scale * var_p["radius_mul"]))
    #         smoothing_func = var_p["smoothing_func"](smoothing_scale)

    #         (x,y) = np.mgrid[
    #             -smoothing_rad:smoothing_rad+1,
    #             -smoothing_rad:smoothing_rad+1, 
    #         ]
    #         d = np.sqrt(x**2 + y**2)
    #         k = smoothing_func(d)
    #         if var_p["normalize_filter"]:
    #             k /= k.sum()

    #         lmap = convolve(lmap, k, mode='same')
        
    #     if var_type == "occurrence":
    #         lmap.clip(min=0, max=1, out=lmap)
    #         lmap.round(out=lmap)
    #     else:
    #         lmap.clip(min=0, out=lmap)
        
    #     return lmap

    def variable_for_time(self, time, variable, filename):
        # if (time, variable) not in self.density_cache:            
        #     density = self.lightning_maps(time, variable)
        #     self.density_cache[(time, variable)] = density

        ################ OLD CODE ################
        # # Read the NetCDF file with netCDF4
        # ds = Dataset(filename, 'r')

        # # Access the datamap data - navigate through the groups
        # data = ds.variables['datamap'][:]

        data = np.load(filename)

        # return self.density_cache[(time, variable)]
        return data
