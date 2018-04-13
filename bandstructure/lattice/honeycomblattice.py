import numpy as np
from .lattice import Lattice


class HoneycombLattice(Lattice):
    def initialize(self):
        l1 = np.array([-np.sqrt(3), 0])
        l2 = np.array([-np.sqrt(3), 3])/2

        b1 = np.array([0, 0])
        b2 = np.array([-np.sqrt(3), 1])/2

        self.addLatticevector(l1)
        self.addLatticevector(l2)
        self.addBasisvector(b1)
        self.addBasisvector(b2)
