#!/usr/bin/python3

import sys
sys.path.append("../")

from bandstructure.lattice import HoneycombLattice
from bandstructure.system import TightBindingSystem
from bandstructure.plot import Plot
from bandstructure import Parameters

params = Parameters()
lattice = HoneycombLattice(params)
#lattice.makeFiniteCircle(2)
#lattice.makeFiniteRectangle(8,10,center=[0.5,0])
#lattice.makeFiniteAlongdirection(1, 30)
#lattice.makeFiniteAlongdirection(0, 30)


system = TightBindingSystem(lattice, {'t': 1})
#plot = Plot(system)
#plot.plotDispersion()

# === tests ===
import matplotlib.pyplot as plt
import numpy as np

cutoff = 25

# --- distance matrix ---
print(1)
distances = lattice.getDistances(cutoff)
print(2)

#distances = distances.getNN(1)

nSubs = distances.shape[0]
nLinks = distances.shape[2]


distances = np.sqrt(np.sum(distances**2,axis=-1)).reshape(nSubs,nLinks*nSubs).T
plt.imshow(distances, aspect='auto',interpolation='nearest')
plt.show()

'''# --- positions ---
positions = lattice.getPositions(cutoff)

fig = plt.gcf()
fig.gca().add_artist(plt.Circle((0,0),cutoff,fc='0.9',ec='k'))

plt.plot(positions[:,0],positions[:,1], 'ko',ms=5)
plt.axes().set_aspect('equal')
plt.xlim(-1.3*cutoff,1.3*cutoff)
plt.ylim(-1.3*cutoff,1.3*cutoff)
plt.show()
# TODO lattice.getVecsBasis,lattice.deleteVecsBasis,lattice.getVecsLattice,lattice.deleteVecsLattice
# TODO plotter
# TODO periodic boundary conditions
# TODO initialize method for system (sets the things that need no recalculation for every k)
# TODO cick out unneeded cols of the Hamiltonian'''


# --- geometry ---

geometry = lattice.getGeometry(cutoff)
basis = lattice.getVecsBasis()


fig = plt.gcf()


for p,b in zip(geometry,basis):
    line, = plt.plot(p[:,0],p[:,1], 'o',ms=5)
    fig.gca().add_artist(plt.Circle(b,cutoff, fill = False ,ec=line.get_color(),alpha=0.5,lw=2))
    plt.plot(b[0],b[1], 'kx',ms=7,mew=1)
plt.axes().set_aspect('equal')
plt.xlim(-1.3*cutoff,1.3*cutoff)
plt.ylim(-1.3*cutoff,1.3*cutoff)
plt.show()

'''# --- Brillouin zone ---
kvectors = lattice.getKvectorsZone(100)
kvectorslength = np.sqrt(np.sum(kvectors**2,axis=-1))

plt.imshow(kvectorslength, aspect='equal',interpolation='nearest',extent=(np.min(kvectors[:,:,1]),np.max(kvectors[:,:,1]),np.min(kvectors[:,:,0]),np.max(kvectors[:,:,0])))

pathVecs, pathLength = lattice.getKvectorsPath(10)
plt.plot(pathVecs[:,1],pathVecs[:,0],'k-',lw=2)

plt.show()'''

