from .lattice import Lattice


class RegularChain(Lattice):
    """Simple one-dimensional lattice"""

    def __init__(self,params):
        super().__init__(params)

        self.addLatticevector([1, 0])
        self.addBasisvector([0, 0])
