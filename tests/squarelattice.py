#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
sys.path.append("../")

from bandstructure.lattice import Lattice, Square
from bandstructure.system import System
from bandstructure.plot import Plot

lattice = Square()
system = System(lattice, {'t': 1})
plot = Plot(system)

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
