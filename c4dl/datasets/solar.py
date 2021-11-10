from .datasetreader import DatasetReader

import numpy as np
from pyorbital.astronomy import get_alt_az

class SolarReader(DatasetReader):
    def __init__(self, grid_projection, 
        variables=["sun_x", "sun_y", "sun_z"]
    ):
        super().__init__(grid_projection, variables=variables)
        (y,x) = np.mgrid[
            :grid_projection.area.height,
            :grid_projection.area.width
        ]
        (self.lon, self.lat) = grid_projection.inverse(y,x)

    def data_for_time(self, time):
        (alt, az) = get_alt_az(time, self.lon, self.lat)
        cos_alt = np.cos(alt)
        x = np.sin(az) * cos_alt
        y = np.cos(az) * cos_alt
        z = np.sin(alt)

        return np.stack([x,y,z], axis=-1)
