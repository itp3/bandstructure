#!/usr/bin/python3

import sys
sys.path.append("../")

from bandstructure import Parameters
from bandstructure.system import TightBindingSystem
from bandstructure.lattice import SquareLattice
from bandstructure.plot import Plot

params = Parameters({
    'cutoff': 2.1,
    't': 1,
    't2': 0
})

l = SquareLattice(params)
# l.makeFiniteAlongdirection(1, 20)
# l.makeFiniteAlongdirection(0, 20)

s = TightBindingSystem(l, params)

# print("Parameters:")
# s.params.showParams()

p = Plot(s)
#p.plotDispersionPath()
p.plotDispersion(resolution=100)
