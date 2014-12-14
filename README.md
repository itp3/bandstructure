Introduction
------------
*bandstructure* is a python module for solving tight-binding(-like) models. It has a modular structure allowing for easy customization of the underlying lattice structure as well as the specific system (specified by its tunneling rates).

Features
--------
- *Speed*: NumPy vectorization is used whenever possible. Parallelization is supported through the python *multiprocessing* module.
- *Dimensionality*: Two- and one-dimensional lattices are supported as well as different kinds of (semi-)finite systems.
- *Predefined lattices*: Square lattice, Honeycomb lattice, Kagome lattice, Ruby lattice.
- *Topological properties*: Calculation of Chern numbers and Berry phases.

Requirements
------------
* python 3
* numpy
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
