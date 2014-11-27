import numpy as np
from .lattice import Lattice

class Honeycomb(Lattice):
    def __init__(self):
        l1 = np.array([np.sqrt(3),1])/2
        l2 = np.array([np.sqrt(3),-1])/2

        b1 = np.array([0,0])
        b2 = np.array([np.sqrt(3),0])/3

        a = np.linalg.norm(b2)

        Lattice.addLatticevector(self,l1/a)
        Lattice.addLatticevector(self,l2/a)
        Lattice.addBasisvector(self,b1/a)
        Lattice.addBasisvector(self,b2/a)