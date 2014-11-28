from .lattice import Lattice


class RegularChain(Lattice):
    """Simple one-dimensional lattice"""

    def initialize(self):
        self.addLatticevector([1, 0])
        self.addBasisvector([0, 0])
