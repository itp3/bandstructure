from .lattice import Lattice

class SquareLattice(Lattice):
    def initialize(self):
        self.addLatticevector([1,0])
        self.addLatticevector([0,1])
        self.addBasisvector([0,0])
