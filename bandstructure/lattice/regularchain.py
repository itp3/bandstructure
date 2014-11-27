from .lattice import Lattice


class RegularChain(Lattice):
    """Simple one-dimensional lattice"""

    def __init__(self):
        self.addLatticevector([1, 0])
        self.addBasisvector([0, 0])
