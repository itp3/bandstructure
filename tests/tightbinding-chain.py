#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure.system import TightBindingSystem
from bandstructure.lattice import Lattice
from bandstructure.plot import Plot


class Chain(Lattice):
    def getDistances(self, _):
        return np.array([[[[1, 0], [-1, 0]]]])


l = Chain()
s = TightBindingSystem(l)
s.set("cutoff", 1)
s.set("t", 1)

print("Parameters:")
s.showParams()

p = Plot(s)
p.plotDispersion()
