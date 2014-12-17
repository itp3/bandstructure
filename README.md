[![Build Status](https://travis-ci.org/sharkdp/bandstructure.svg?branch=master)](https://travis-ci.org/sharkdp/bandstructure) [![Coverage Status](https://img.shields.io/coveralls/sharkdp/bandstructure.svg)](https://coveralls.io/r/sharkdp/bandstructure)


Introduction
------------
*bandstructure* is a python module for solving tight-binding(-like) models. It has a modular structure allowing for easy customization of the underlying lattice structure as well as the specific system (defined by its tunneling rates).

Features
--------
- NumPy vectorization is used whenever possible. Parallelization is supported through the python multiprocessing module
- One- and two-dimensional lattices are supported as well as different kinds of (semi-)finite systems
- Predefined lattices: Chains, Square lattice, Honeycomb lattice, Kagome lattice, Ruby lattice
- Calculate topological properties: Chern numbers, Berry phases

Requirements
------------
* Tested with Python >= 3.2
* numpy
* scipy
* matplotlib (optional)

Example: graphene
-----------------
The following short programm solves the nearest-neighbor tight-binding model on the two-dimensional honeycomb lattice (graphene).

```python
from bandstructure import Parameters
from bandstructure.system import TightBindingSystem
from bandstructure.lattice import HoneycombLattice

lattice = HoneycombLattice()

params = Parameters({
    'lattice': lattice,
    't': 1
})

s = TightBindingSystem(params)

# Solve on high-symmetry path through Brillouin zone
path = lattice.getKvectorsPath(resolution=300, pointlabels=['A', 'G', 'X', 'A'])
bandstructure = s.solve(path)
bandstructure.plot("dispersion.pdf")
```
