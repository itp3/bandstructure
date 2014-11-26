import numpy as np

class Lattice():
    __latticevectors = [None]
    __basisvectors = [None]

    def __init__(self):
        pass

    def addLatticevector(self,vector):
        """ adds a lattice vector"""

        vector = np.array(vector)
        if vector.shape != (3,):
            raise Exception("Lattice vectors have to be 3D vectors.")

        self.__latticevectors.append(vector)

    def addBasisvector(self,vector):
        """ adds a basis vector"""

        vector = np.array(vector)
        if vector.shape != (3,):
            raise Exception("Basis vectors have to be 3D vectors.")

        self.__basisvectors.append(vector)

    def getLattice(self):
        pass