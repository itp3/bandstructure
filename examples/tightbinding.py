#!/usr/bin/env python3

import sys
sys.path.append("../")

from bandstructure import Parameters
from bandstructure.system import TightBindingSystem
from bandstructure.lattice import HoneycombLattice

lattice = HoneycombLattice()

params = Parameters({
    'lattice': lattice,
    't': 1
})

s = TightBindingSystem(params)

# Solve system on high-symmetry path through Brillouin zone
path = lattice.getKvectorsPath(resolution=300, pointlabels=['A', 'G', 'X', 'A'])

bandstructure = s.solve(path)
bandstructure.plot("dispersion.pdf")
