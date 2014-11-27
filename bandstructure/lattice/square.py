from .lattice import Lattice

class Square(Lattice):
    def __init__(self):
        self.addLatticevector([1,0,0])
        self.addLatticevector([0,1,0])
        self.addBasisvector([0,0,0])
