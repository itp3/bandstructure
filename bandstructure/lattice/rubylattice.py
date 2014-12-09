import numpy as np
from .lattice import Lattice


class RubyLattice(Lattice):
    def initialize(self):
        sqrt3 = np.sqrt(3)

        l = 1 + sqrt3

        self.addLatticevector([l, 0])
        self.addLatticevector([l / 2, sqrt3 * l / 2])

        self.addBasisvector([0, 0])
        self.addBasisvector([1, 0])
        self.addBasisvector([1, 1])
        self.addBasisvector([1 + sqrt3/2, 3/2])
        self.addBasisvector([l + 1/2, 1 + sqrt3/2])
        self.addBasisvector([1 + sqrt3, 1])
