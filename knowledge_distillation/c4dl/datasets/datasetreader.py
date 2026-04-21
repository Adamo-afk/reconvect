from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np

from ..utils import time_range


class DatasetReader(ABC):
    name: str

    def __init__(self, grid_projection, *, variables):
        self.grid_projection = grid_projection
        if not variables:
            raise ValueError("Must specify at least one variable.")
        self.variables = variables

    def data_for_time(self, time):
        """ Note that this is circular with variable_for_time.
        Either data_for_time or variable_for_time must be
        implemented by the subclass.
        """
        data = [self.variable_for_time(time, var) for var in self.variables]
        return np.stack(data, axis=-1)

    def data_for_times(self, times):
        data = [self.data_for_time(t) for t in times]
        return np.stack(data)

    def data_for_time_range(self, time_limits, interval=timedelta(minutes=5)):
        times = time_range(time_limits[0], time_limits[1], interval)
        return (times, self.data_for_times(times))

    def variable_for_time(self, time, variable):
        ind = self.variables.index(variable)
        return self.data_for_time(time)[...,ind]

    def variable_for_times(self, times, variable):
        data = [self.variable_for_time(t, variable) for t in times]
        return np.stack(data)

    def variable_for_time_range(self, time_range, variable,
        interval=timedelta(minutes=5)):
        times = time_range(time_limits[0], time_limits[1], interval)
        return (times, self.variable_for_times(times, variable))
