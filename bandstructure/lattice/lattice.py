import numpy as np

class Lattice():
    __latticevectors = []
    __latticevectorsReciprocal = []
    __basisvectors = []
    __posBrillouinZone = None
    __posBrillouinPath = None

    def __init__(self):
        pass

    '''def _calcCircumcenter(vectorB, vectorC):
        """see http://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates """

        D = 2*(vectorC[1]*vectorB[0]-vectorB[1]*vectorC[0])
        x = (vectorC[1]*np.vdot(vectorB,vectorB)-vectorB[1]*np.vdot(vectorC,vectorC))/D
        y = (vectorB[0]*np.vdot(vectorC,vectorC)-vectorC[0]*np.vdot(vectorB,vectorB))/D
        return np.array([x,y])'''
    '''def _calcReciprocalVectors(self,vector):
        lattice_vector1reciprocal       = np.dot(np.array([[0,1],[-1,0]]),lattice_vector2)
    lattice_vector1reciprocal       = 2*np.pi*lattice_vector1reciprocal/(np.vdot(lattice_vector1,\
        lattice_vector1reciprocal))
    lattice_vector2reciprocal       = np.dot(np.array([[0,-1],[1,0]]),lattice_vector1)
    lattice_vector2reciprocal       = 2*np.pi*lattice_vector2reciprocal/(np.vdot(lattice_vector2,\
        lattice_vector2reciprocal))
    lattice_matreciprocal           = np.array([lattice_vector1reciprocal, lattice_vector2reciprocal]).T'''

    def addLatticevector(self,vector):
        """adds a lattice vector and calculate the reciprocal vectors"""

        # --- add lattice vector ---
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Lattice vectors have to be 2D vectors.")

        self.__latticevectors.append(vector)

        if len(self.__latticevectors) > 2:
            raise Exception("There must be at most 2 lattice vectors.")

        # --- calculate the reciprocal vectors ---
        if len(self.__latticevectors) == 1:
            self.__latticevectorsReciprocal = [
                2*np.pi*self.__latticevectors[0]/np.linalg.norm(self.__latticevectors[0])**2
                ]

        elif len(self.__latticevectors) == 2: #TODO
            self.__latticevectorsReciprocal = [
                np.dot(np.array([[0,1],[-1,0]]),self.__latticevectors[1]),
                np.dot(np.array([[0,-1],[1,0]]),self.__latticevectors[0])
                ]
            self.__latticevectorsReciprocal[0] = 2*np.pi*self.__latticevectorsReciprocal[0]/\
                (np.vdot(self.__latticevectors[0], self.__latticevectorsReciprocal[0]))
            self.__latticevectorsReciprocal[1] = 2*np.pi*self.__latticevectorsReciprocal[1]/\
                (np.vdot(self.__latticevectors[1], self.__latticevectorsReciprocal[1]))

    def addBasisvector(self,vector):
        """adds a basis vector"""

        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Basis vectors have to be 2D vectors.")

        self.__basisvectors.append(vector)

    def getKvectorsZone(self, resolution):
        #return self.__posBrillouinZone
        return np.random.rand(int(0.86*resolution**2),2) # idxK, idxCoord

    def getKvectorsPath(self, resolution):
        #return self.__posBrillouinPath
        return np.random.rand(int(1.1*resolution*3),2) # idxK, idxCoord

    def getDistances(self, resolution, cutoff):
        #return self.__posBrillouinZone
        numSubs = len(self.__basisvectors)
        return np.random.rand(numSubs,numSubs,10*cutoff,2) # idxSub1, idxSub2, idxLink, idxCoord


