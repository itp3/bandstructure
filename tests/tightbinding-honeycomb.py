#!/usr/bin/python3

import sys
sys.path.append("../")

from bandstructure.lattice import HoneycombLattice
from bandstructure.system import TightBindingSystem
from bandstructure.plot import Plot

lattice = HoneycombLattice()
system = TightBindingSystem(lattice, {'t': 1})
#plot = Plot(system)
#plot.plotDispersion()


# === tests ===
import matplotlib.pyplot as plt
import numpy as np

# --- distance matrix ---

distances = lattice.getDistances(20)

nSubs = distances.shape[0]
nLinks = distances.shape[2]

distances = np.sqrt(np.sum(distances**2,axis=-1)).reshape(nSubs,nLinks*nSubs).T
plt.imshow(distances, aspect='auto',interpolation='nearest')
plt.show()

# --- positions ---

positions = lattice.getPositions(20)

plt.plot(positions[:,0],positions[:,1], 'ko',ms=3)
plt.axes().set_aspect('equal', 'datalim')
plt.show()
