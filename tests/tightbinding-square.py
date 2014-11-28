#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure.system import TightBindingSystem
from bandstructure.lattice import SquareLattice
from bandstructure.plot import Plot


l = SquareLattice()
s = TightBindingSystem(l)
s.set("cutoff", 1)
s.set("t", 1)
# s.set("t2", 0.5)

print("Parameters:")
s.showParams()

p = Plot(s)
p.plotDispersionPath()
