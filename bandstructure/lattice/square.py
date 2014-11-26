import numpy as np
from lattice import Lattice

class Square(Lattice):
    def __init__(self):
        Lattice.addLatticevector(self,[1,0,0])
        Lattice.addLatticevector(self,[0,1,0])
        Lattice.addBasisvector(self,[0,1,0])