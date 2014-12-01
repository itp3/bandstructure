#!/usr/bin/python3

import sys
sys.path.append("../")

from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice
from bandstructure.plot import Plot

params = Parameters({
    'cutoff': 300.1,
    't': 1,
})

l = SquareLattice(params)
s = DipolarSystem(l, params)

print("Parameters:")
s.params.showParams()

p = Plot(s)
p.plotDispersionPath()

# params["cutoff"] = 800
# s.initialize()
# print(s.solve([[0, 0]]))
