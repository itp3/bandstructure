#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice, HoneycombLattice

# === lattice ===
l = SquareLattice()
l = HoneycombLattice()
#l.makeFiniteAlongdirection(0,20)
l.plot(filename="",show=True,cutoff=5)


# === system ===
params = Parameters({
    'lattice': l,
    'cutoff': 30,
    'tbar': 1,
    'mu': -4.54,
    't': 0.54,
    'w': 1.97
})

s = DipolarSystem(params)

# === bandstructure ===
k = [0, 4/(np.sqrt(3)*3) * np.pi]
gam = [0, 0]

kvecs = l.getKvectorsPath(100, points=[k, gam, k])
#kvecs = l.getKvectorsZone(30, points=[k, gam, k])

b = s.solve(kvecs)
b.plot(filename="",show=True)