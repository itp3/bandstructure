#!/usr/bin/python3

import sys
sys.path.append("../")

from bandstructure.system import TightBindingSystem
from bandstructure.lattice import SquareLattice

l = SquareLattice()
s = TightBindingSystem(l)
s.setParams({'t': 3})

print("Parameters:")
s.showParams()
