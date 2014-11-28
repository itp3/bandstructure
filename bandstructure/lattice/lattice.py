import numpy as np

# TODOs
# * get nearest neighbor cutoff
# * docstrings
# * periodic boundary conditions
# * generation of k vectors



class Lattice():
    __latticevectors = np.array([])
    __latticevectorsReciprocal = np.array([])
    __basisvectors = np.array([])
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
        """Add a lattice vector and calculate the reciprocal vectors."""

        # === add lattice vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Lattice vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__latticevectors.shape[0] == 0:
            self.__latticevectors = np.array([vector])
        else:
            self.__latticevectors = np.append(self.__latticevectors,[vector], axis=0)

        # validation of lattice vector number
        if self.__latticevectors.shape[0] > 2:
            raise Exception("There must be at most 2 lattice vectors.")

        # === calculate the reciprocal vectors ===
        if self.__latticevectors.shape[0] == 1:
            self.__latticevectorsReciprocal = [
                2*np.pi*self.__latticevectors[0]/np.linalg.norm(self.__latticevectors[0])**2
                ]
        else:
            self.__latticevectorsReciprocal = [
                np.dot(np.array([[0,1],[-1,0]]),self.__latticevectors[1]),
                np.dot(np.array([[0,-1],[1,0]]),self.__latticevectors[0])
                ]
            self.__latticevectorsReciprocal[0] = 2*np.pi*self.__latticevectorsReciprocal[0]/\
                (np.vdot(self.__latticevectors[0], self.__latticevectorsReciprocal[0]))
            self.__latticevectorsReciprocal[1] = 2*np.pi*self.__latticevectorsReciprocal[1]/\
                (np.vdot(self.__latticevectors[1], self.__latticevectorsReciprocal[1]))

    def addBasisvector(self,vector):
        """Add a basis vector."""

        # === add basis vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Basis vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__basisvectors.shape[0] == 0:
            self.__basisvectors = np.array([vector])
        else:
            self.__basisvectors = np.append(self.__basisvectors,[vector], axis=0)

    def getKvectorsZone(self, resolution):
        #return self.__posBrillouinZone
        return np.random.rand(int(0.86*resolution**2),2) # idxK, idxCoord

    def getKvectorsPath(self, resolution):
        #return self.__posBrillouinPath
        return np.random.rand(int(1.1*resolution*3),2) # idxK, idxCoord

    def getPositions(self, cutoff):
        """Generate all positions from the lattice vectors using [0,0] as the basis vector.

        positions = getPositions(cutoff)
        positions[idxPosition, idxCoordinate]"""

        # value that is added to the cutoff to be on the save side
        safetyregion = 1*np.max(np.sqrt(np.sum(self.__latticevectors**2,axis=-1)))

        # array that will contain all positions
        positions = []

        # --- first shift (do it only if two lattice vectors exist) ---
        shiftidx1 = 0
        boolLoop1 = True
        while boolLoop1:
            if self.__latticevectors.shape[0] >= 2:
                shiftedpos1 = shiftidx1*self.__latticevectors[1]

                # substract the other lattice vector to be as central as possible inside the cutoff region
                shiftedpos1 -= np.round(np.vdot(shiftedpos1,self.__latticevectors[0])/\
                    np.linalg.norm(self.__latticevectors[0])**2 )*self.__latticevectors[0]

                if shiftidx1 < 0: shiftidx1 -= 1
                else: shiftidx1 += 1

                # change looping direction / break if the shift is larger than a cutoff
                if np.linalg.norm(shiftedpos1) > cutoff+safetyregion:
                    if shiftidx1 > 0:
                        shiftidx1 = -1
                        continue
                    else:
                        break
            else:
                shiftedpos1 = np.array([0,0])
                boolLoop1 = False

            # --- second shift (do it only if at least one lattice vector exists) ---
            shiftidx0 = 0
            boolLoop0 = True
            while boolLoop0:
                if self.__latticevectors.shape[0] >= 1:
                    shiftedpos0 = shiftidx0*self.__latticevectors[0]

                    # add together all shifts
                    shiftedpos = shiftedpos1+shiftedpos0

                    if shiftidx0 < 0: shiftidx0 -= 1
                    else: shiftidx0 += 1

                    # change looping direction / break if the sum of shifts is larger than a cutoff
                    if np.linalg.norm(shiftedpos) > cutoff+safetyregion:
                        if shiftidx0 > 0:
                            shiftidx0 = -1
                            continue
                        else:
                            break
                else:
                    shiftedpos = np.array([0,0])
                    boolLoop0 = False

                # append the sum of shifts to the array of positions
                positions.append(shiftedpos)

        return np.array(positions)

    def getGeometry(self, cutoff):
        """Generate all positions from the lattice vectors using all the basis vectors.

        geometry = getGeometry(cutoff)
        geometry[idxSublattice, idxPosition, idxCoordinate]"""

        # number of sublattices
        numSubs = self.__basisvectors.shape[0]

        # === creation of all positions ===
        positions = self.getPositions(cutoff)
        positionsAll = np.tile(positions, (numSubs,1,1)) + self.__basisvectors[:,None]

        return positionsAll

    def makeFiniteCircle(self, cutoff):
        """Generate a finite circular lattice.

        makeFiniteCircle(radius)"""

        positionsAll = self.getGeometry(cutoff).reshape(-1,2)

        # masking
        positionsAllAbs = np.sqrt(np.sum(positionsAll**2,axis=-1))
        positionsAllMask = (positionsAllAbs > cutoff)
        positionsAll = positionsAll[~positionsAllMask]

        # save the finite system as basisvectors
        self.__latticevectors = np.array([])
        self.__latticevectorsReciprocal = np.array([])
        self.__basisvectors = positionsAll

    def makeFiniteRectangle(self, cutoffX, cutoffY):
        """Generate a finite rectangular lattice.

        makeFiniteCircle(2*width,2*height)"""

        positionsAll = self.getGeometry(np.sqrt(2)*max(cutoffX,cutoffY)).reshape(-1,2)

        # masking
        positionsAllMask = (np.abs(positionsAll[:,0]) > cutoffX) | \
            (np.abs(positionsAll[:,1]) > cutoffY)
        positionsAll = positionsAll[~positionsAllMask]

        # save the finite system as basisvectors
        self.__latticevectors = np.array([])
        self.__latticevectorsReciprocal = np.array([])
        self.__basisvectors = positionsAll

    def getDistances(self, cutoff):
        """Create a matrix that contains all distances from the central position of a
        sublattice to all positions of another one.

        distances = getDistances(cutoff)
        distances[idxSublatticeTo, idxSublatticeFrom, idxLink, idxCoordinate]"""

        # positions generated from the lattice vectors
        positions = self.getPositions(cutoff)
        sorter = np.argsort(np.sum(positions**2,axis=-1))
        positions = positions[sorter]

        # shifts given by the basisvectors
        shifts = self.__basisvectors

        # === numbers ===
        # maximal number of links between the central position of a sublattice and all positions of another one
        numLinks = positions.shape[0]

        # number of sublattices
        numSubs = shifts.shape[0]

        # === creation of the distance matrix ===
        # array of central positions [Sub, Coord] that will be repeated to create the matrix matDeltaR
        positionsCentral = shifts

        # array of all positions [Sub, Link, Coord] that will be repeated to create the matrix matDeltaR
        positionsAll = np.tile(positions, (numSubs,1,1)) + positionsCentral[:,None]

        # creation of the matrix matDeltaR [Sub1, Sub2, Link, Coord]
        matPositionsCentral = np.tile(positionsCentral, (numLinks,numSubs, 1,1)).transpose(2,1,0,3)
        matPositionsAll = np.tile(positionsAll, (numSubs,1,1,1))
        matDeltaR = matPositionsAll-matPositionsCentral

        # masking of the matrix matDeltaR [Sub1, Sub2, Link, Coord]
        matDeltaRAbs = np.sqrt(np.sum(matDeltaR**2,axis=-1))
        matDeltaRMask = (matDeltaRAbs > cutoff) | (matDeltaRAbs < 1e-32) | \
            ~np.tri(numSubs,numSubs,dtype=bool)[:,:,None]
        matDeltaRMask = np.array([matDeltaRMask, matDeltaRMask]).transpose(1,2,3,0)
        matDeltaR = np.ma.array(matDeltaR, mask = matDeltaRMask)

        return matDeltaR
