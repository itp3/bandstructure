#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice, HoneycombLattice

# l = SquareLattice()
l = HoneycombLattice()

params = Parameters({
    'lattice': l,
    'cutoff': 10,
    'tbar': 1,
    'mu': -4.54,
    't': 0.54,
    'w': 1.97
})

s = DipolarSystem(params)

k = [0, 4/(np.sqrt(3)*3) * np.pi]
gam = [0, 0]
path, length = l.getKvectorsPath(300, points=[k, gam, k])
bs = s.solve(path)

# bz = l.getKvectorsZone(30)
# bs = s.solve(bz)

bs.plot()
