#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice, HoneycombLattice

# === lattice ===
#l = SquareLattice()
l = HoneycombLattice()
#l.makeFiniteAlongdirection(0,1)
#l.plot(show=True,cutoff=5)

# === system ===
params = Parameters({
    'lattice': l,
    'cutoff': 60,
    'tbar': 1,
    'mu': -4.54,
    't': 0.54,
    'w': 1.97
})

s = DipolarSystem(params)

#d = l.getDistances(cutoff=20)
#d.plot()

# === bandstructure ===
#kvecs = l.getKvectorsPath(100, pointlabels=['-X', 'G', 'X'])
kvecs = l.getKvectorsBox(30)
#kvecs = l.getKvectorsZone(30)

b = s.solve(kvecs)
b.plot(show=True)

#print(b.getBerryPhase(2))

# === chern numbers ===
cherns = b.getChernNumbers()
print(np.round(cherns,3),np.sum(cherns))