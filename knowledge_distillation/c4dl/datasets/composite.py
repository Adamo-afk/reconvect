
from abc import ABC, abstractmethod

import numpy as np

from .datasetreader import DatasetReader


class CompositeReader(DatasetReader):
    name = "composite"

    def __init__(self, spec):
        self.composites = {}
        for name in spec:
            comp_type = spec[name]["type"]
            comp_cls = composite_classes[comp_type]
            reader_vars = spec[name]["reader_vars"]
            self.composites[name] = comp_cls(reader_vars)
            
        variables=list(self.composites.keys())
        grid_projection = self.composites[variables[0]]. \
            reader_vars[0][0].grid_projection

        super().__init__(grid_projection, variables=variables)
            

    def variable_for_time(self, time, variable):
        return self.composites[variable](time)


class Composite(ABC):
    def __init__(self, reader_vars):
        self.reader_vars = reader_vars

    def inputs_for_time(self, time):
        return (
            r.variable_for_time(time, v) 
            for (r,v) in self.reader_vars
        )

    @abstractmethod
    def __call__(self, time):
        pass


class Sum(Composite):
    def __call__(self, time):
        inputs = self.inputs_for_time(time)
        return sum(inputs)


class Difference(Composite):
    def __call__(self, time):
        (x1,x2) = self.inputs_for_time(time)
        return x1 - x2


class Dot(Composite):
    def __call__(self, time):
        (u1,v1,u2,v2) = self.inputs_for_time(time)
        return u1*u2 + v1*v2


composite_classes = {
    "sum": Sum,
    "diff": Difference,
    "dot": Dot
}