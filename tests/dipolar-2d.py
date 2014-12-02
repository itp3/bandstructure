#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice, HoneycombLattice
from bandstructure.plot import Plot

# l = SquareLattice()
l = HoneycombLattice()

params = Parameters({
    'cutoff': 20,
    'tbar': 1,
    'mu': -4.54,
    't': 0.54,
    'w': 1.97,
    'lattice': l
})

params.showParams()

s = DipolarSystem(params)

p = Plot(s)

k = [0, 4/(np.sqrt(3)*3) * np.pi]
gam = [0, 0]
p.plotDispersionPath([k, gam, k])

# p.plotDispersion()
