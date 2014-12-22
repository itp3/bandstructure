from .lattice import Lattice


class LiebLattice(Lattice):
    def initialize(self):
        self.addLatticevector([2, 0])
        self.addLatticevector([0, 2])
        self.addBasisvector([0, 0])
        self.addBasisvector([1, 0])
        self.addBasisvector([0, 1])
