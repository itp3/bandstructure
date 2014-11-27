#!/usr/bin/python3

import sys
sys.path.append("../")

from bandstructure.lattice import HoneycombLattice
from bandstructure.system import TightBindingSystem
from bandstructure.plot import Plot

lattice = HoneycombLattice()
system = TightBindingSystem(lattice, {'t': 1})
plot = Plot(system)

print(lattice.getKvectorsZone(20).shape)
print(lattice.getDistances(100, 5).shape)

plot.plotDispersion()




'''# parallel
myfirstlattices = []
for angle in np.linspace(0,np.pi,5):
    lattice = Honeycomb()
    lattice.setAngle(angle)
    lattice.

    for w in ...:
        dictSystemSettings = {..., 'w':w}
        mysystems = System(lattice, dictSystemSettings)
        mysystems.calculateChernnumbers()

        Analyzer

        Plotter(mysystems)

lattice.set


Plotter(fileSystem)

pep8
'''
