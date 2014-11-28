#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure.system import TightBindingSystem
from bandstructure.lattice import Lattice
from bandstructure.plot import Plot


class Chain(Lattice):
    def getDistances(self, _):
        return np.array([[[
            [-2, 0],
            [-1, 0],
            [1, 0],
            [2, 0]
        ]]])

params = {
    'cutoff': 2.1,
    't': 1,
    't2': 0
}

l = Chain(params)
s = TightBindingSystem(l, params)

print("Parameters:")
s.params.showParams()

p = Plot(s)
p.plotDispersionPath()
