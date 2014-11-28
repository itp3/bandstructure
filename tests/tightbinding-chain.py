#!/usr/bin/python3

import sys
sys.path.append("../")
import numpy as np

from bandstructure import Parameters
from bandstructure.system import TightBindingSystem
from bandstructure.lattice import RegularChain
from bandstructure.plot import Plot


params = Parameters({
    'cutoff': 10.1,
    't': 1,
    't2': 1
})

l = RegularChain(params)
s = TightBindingSystem(l, params)

print("Parameters:")
s.params.showParams()

p = Plot(s)
p.plotDispersionPath()
