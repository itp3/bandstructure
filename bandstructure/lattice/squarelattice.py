from .lattice import Lattice

class SquareLattice(Lattice):
    def __init__(self,params):
        super().__init__(params)

        self.addLatticevector([1,0])
        self.addLatticevector([0,1])
        self.addBasisvector([0,0])
