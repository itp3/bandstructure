import numpy as np

# TODOs
# * periodic boundary conditions
# * generation of k vectors
# * makeFiniteAlongdirection



class Lattice():
    __params = None
    __vecsReciprocal = np.array([])
    __posBrillouinZone = np.array([])
    __posBrillouinPath = np.array([])

    def __init__(self, params):
        self.__params = params
        self.__params['vecsLattice'] = np.array([])
        self.__params['vecsBasis'] = np.array([])

    def addLatticevector(self,vector):
        """Add a lattice vector and calculate the reciprocal vectors."""

        # === add lattice vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Lattice vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__params['vecsLattice'].shape[0] == 0:
            self.__params['vecsLattice'] = np.array([vector])
        else:
            self.__params['vecsLattice'] = np.append(self.__params['vecsLattice'],[vector], axis=0)

        # validation of lattice vector number
        if self.__params['vecsLattice'].shape[0] > 2:
            raise Exception("There must be at most 2 lattice vectors.")

        # === calculate the reciprocal vectors ===
        if self.__params['vecsLattice'].shape[0] == 1:
            self.__vecsReciprocal = [
                2*np.pi*self.__params['vecsLattice'][0]/np.linalg.norm(self.__params['vecsLattice'][0])**2
                ]
        else:
            self.__vecsReciprocal = [
                np.dot(np.array([[0,1],[-1,0]]),self.__params['vecsLattice'][1]),
                np.dot(np.array([[0,-1],[1,0]]),self.__params['vecsLattice'][0])
                ]
            self.__vecsReciprocal[0] = 2*np.pi*self.__vecsReciprocal[0]/\
                (np.vdot(self.__params['vecsLattice'][0], self.__vecsReciprocal[0]))
            self.__vecsReciprocal[1] = 2*np.pi*self.__vecsReciprocal[1]/\
                (np.vdot(self.__params['vecsLattice'][1], self.__vecsReciprocal[1]))

    def addBasisvector(self,vector):
        """Add a basis vector."""

        # === add basis vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Basis vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__params['vecsBasis'].shape[0] == 0:
            self.__params['vecsBasis'] = np.array([vector])
        else:
            self.__params['vecsBasis'] = np.append(self.__params['vecsBasis'],[vector], axis=0)

    def _calcCircumcenter(vectorB, vectorC):
        """See http://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates."""

        D = 2*(vectorC[1]*vectorB[0]-vectorB[1]*vectorC[0])
        x = (vectorC[1]*np.vdot(vectorB,vectorB)-vectorB[1]*np.vdot(vectorC,vectorC))/D
        y = (vectorB[0]*np.vdot(vectorC,vectorC)-vectorC[0]*np.vdot(vectorB,vectorB))/D
        return np.array([x,y])

    def getKvectorsZone(self, resolution):
        if self.__vecsReciprocal.shape[0] == 1:
            return None

        else:
            # reciprocal positions
            matTrafo = np.array([self.__vecsReciprocal[0], self.__vecsReciprocal[1]]).T

            reciprocalpositions     = np.empty((3*3,2))
            for n,[x,y] in enumerate(itertools.product([0,-1,1],[0,-1,1])):
                reciprocalpositions[n] = np.dot(matTrafo, [x,y])


            # calculate "radius" of the Brillouizone
            '''cornerOfBrillouizone = calcCircumcenter(self.__vecsReciprocal[0],\
                self.__vecsReciprocal[1]+__vecsReciprocal[0])
            radius = np.linalg.norm(cornerOfBrillouizone)

            safetyregion = 2*radius/resoultion
            radius += safetyregion'''

            # generate a matrix [IdxX, IdxY, Coord] that stores the positions inside the brillouinzone
            positions=np.mgrid[-radius:radius:2j*resoultion,\
                -radius:radius:2j*resoultion,].transpose(1,2,0)

            '''# mask regions inside the matrix that do not belong to the brillouinzone
            distancecache = np.ones_like(brillouinzone_positions[:,:,0])*100/1
            for n, pos in enumerate(lattice_positionsreciprocal):
                distances = np.sum((brillouinzone_positions-pos)**2,axis=-1)
                smaller =  distances < distancecache
                distancecache[smaller] = distances[smaller]

                if n != 0: brillouinzone_mask[smaller] = False

            positions = np.ma.array(positions, mask = positionsMask)

        return positions


        brillouinzone_mask = np.ones_like(brillouinzone_positions[:,:,0],dtype=np.bool)




        r = np.linalg.norm(point_M)

        dr =
        brillouinzone_maxX = r+dr
        brillouinzone_maxY = r+dr
        brillouinzone_minX = -r-dr
        brillouinzone_minY = -r-dr
        resfactor = (brillouinzone_maxX-brillouinzone_minX)/(brillouinzone_maxY-brillouinzone_minY)
        if resfactor >= 1:
            brillouinzone_resolutionX = brillouinzone_resolution
            brillouinzone_resolutionY = brillouinzone_resolution/resfactor
        else:
            brillouinzone_resolutionX = brillouinzone_resolution*resfactor
            brillouinzone_resolutionY = brillouinzone_resolution

        brillouinzone_positions=np.mgrid[brillouinzone_minX:brillouinzone_maxX:1j*brillouinzone_resolutionX,\
            brillouinzone_minY:brillouinzone_maxY:1j*brillouinzone_resolutionY].transpose(1,2,0)
        brillouinzone_mask = np.ones_like(brillouinzone_positions[:,:,0],dtype=np.bool)

        distancecache = np.ones_like(brillouinzone_positions[:,:,0])*100/1
        for n, pos in enumerate(lattice_positionsreciprocal):
            distances = np.sum((brillouinzone_positions-pos)**2,axis=-1)
            smaller =  distances < distancecache
            distancecache[smaller] = distances[smaller]

            if n != 0: brillouinzone_mask[smaller] = False













        #return self.__posBrillouinZone
        return np.random.rand(int(0.86*resolution**2),2) # idxK, idxCoord'''

    def getKvectorsPath(self, resolution):
        #return self.__posBrillouinPath
        return np.random.rand(int(1.1*resolution*3),2) # idxK, idxCoord

    def getPositions(self, cutoff):
        """Generate all positions from the lattice vectors using [0,0] as the basis vector.

        positions = getPositions(cutoff)
        positions[idxPosition, idxCoordinate]"""

        # value that is added to the cutoff to be on the save side
        safetyregion = 1*np.max(np.sqrt(np.sum(self.__params['vecsLattice']**2,axis=-1)))

        # array that will contain all positions
        positions = []

        # --- first shift (do it only if two lattice vectors exist) ---
        shiftidx1 = 0
        boolLoop1 = True
        while boolLoop1:
            if self.__params['vecsLattice'].shape[0] >= 2:
                shiftedpos1 = shiftidx1*self.__params['vecsLattice'][1]

                # substract the other lattice vector to be as central as possible inside the cutoff region
                shiftedpos1 -= np.round(np.vdot(shiftedpos1,self.__params['vecsLattice'][0])/\
                    np.linalg.norm(self.__params['vecsLattice'][0])**2 )*self.__params['vecsLattice'][0]

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
                if self.__params['vecsLattice'].shape[0] >= 1:
                    shiftedpos0 = shiftidx0*self.__params['vecsLattice'][0]

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
        numSubs = self.__params['vecsBasis'].shape[0]

        # === creation of all positions ===
        positions = self.getPositions(cutoff)
        positionsAll = np.tile(positions, (numSubs,1,1)) + self.__params['vecsBasis'][:,None]

        return positionsAll

    def getNNCutoff(self):
        """Return the smallest distance that occures in the lattice (nearest neighbor cutoff)."""

        nearpositions = self.getGeometry(0).reshape(-1,2)

        # subtract central position
        nearpositions -= nearpositions[0]

        # return smallest distance to central position
        neardistances = np.sqrt(np.sum(nearpositions[1:]**2,axis=-1))
        return np.min(neardistances)

    def makeFiniteCircle(self, cutoff, center=[0,0]):
        """Generate a finite circular lattice.

        makeFiniteCircle(radius, center=[x,Y])"""

        positionsAll = self.getGeometry(cutoff+np.linalg.norm(center)).reshape(-1,2)

        # masking
        positionsAllAbs = np.sqrt(np.sum((positionsAll-center)**2,axis=-1))
        positionsAllMask = (positionsAllAbs > cutoff)
        positionsAll = positionsAll[~positionsAllMask]

        # save the finite system as basisvectors
        self.__params['vecsLattice'] = np.array([])
        self.__vecsReciprocal = np.array([])
        self.__params['vecsBasis'] = positionsAll

    def makeFiniteRectangle(self, cutoffX, cutoffY, center=[0,0]):
        """Generate a finite rectangular lattice.

        makeFiniteCircle(2*width,2*height, center=[x,Y])"""

        positionsAll = self.getGeometry(np.sqrt(2)*max(cutoffX,cutoffY)+np.linalg.norm(center)).reshape(-1,2)

        # masking
        positionsAllMask = (np.abs(positionsAll[:,0]-center[0]) > cutoffX) | \
            (np.abs(positionsAll[:,1]-center[1]) > cutoffY)
        positionsAll = positionsAll[~positionsAllMask]

        # save the finite system as basisvectors
        self.__params['vecsLattice'] = np.array([])
        self.__vecsReciprocal = np.array([])
        self.__params['vecsBasis'] = positionsAll

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
        shifts = self.__params['vecsBasis']

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
