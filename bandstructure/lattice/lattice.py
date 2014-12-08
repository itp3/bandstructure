import numpy as np
import itertools
from .. import Distances
from .. import Kpoints


class Lattice():
    """Class to generate the lattice."""

    __vecsLattice = np.array([])
    __vecsBasis = np.array([])
    __vecsReciprocal = np.array([])
    __posBrillouinZone = np.array([])
    __posBrillouinPath = np.array([])
    __specialPoints = { }

    __tol = 1e-16

    def __init__(self):
        self.initialize()

    def initialize(self):
        pass

    def getSpecialPoints(self, reciprocalBasis = False):
        """Return the list of userdefined and automatically generated special points that can be
        used to describe a path through the Brillouin zone ( e.g. 'G' stands for automatically
        generated gamma point)."""

        userdefinedSpecialPoints = self.__specialPoints.copy()

        automaticSpecialPoints = { }

        # === standardize the lattice vectors ===
        if self.getDimensionality() >= 1:
            vec1 = self.__vecsReciprocal[0]
            if np.vdot(vec1,[1,0]) < 0: vec1 *= -1

        if self.getDimensionality() >= 2:
            vec2 = self.__vecsReciprocal[1]
            if np.vdot(vec1,vec2) < 0: vec2 *= -1
            if np.arctan2(vec1[1],vec1[0]) < np.arctan2(vec2[1],vec2[0]): vec1, vec2 = vec2, vec1

        # === calculate special points ===
        # --- special points for 0D and higher dimensions ---
        automaticSpecialPoints['G'] = [0, 0]

        # --- special points for 1D and higher dimensions ---
        if self.getDimensionality() >= 1:
            automaticSpecialPoints['X']      = vec1/2
            automaticSpecialPoints['-X']     = -automaticSpecialPoints['X']

        # --- special points for 2D ---
        if self.getDimensionality() >= 2:
            automaticSpecialPoints['Y']      = vec2/2
            automaticSpecialPoints['-Y']     = -automaticSpecialPoints['Y']

            automaticSpecialPoints['Z']      = (vec2-vec1)/2
            automaticSpecialPoints['-Z']     = -automaticSpecialPoints['Z']

            automaticSpecialPoints['A']      = self._calcCircumcenter(2*automaticSpecialPoints['X'],2*automaticSpecialPoints['Y'])
            automaticSpecialPoints['-A']     = -automaticSpecialPoints['A']

            automaticSpecialPoints['B']      = self._calcCircumcenter(2*automaticSpecialPoints['Y'],2*automaticSpecialPoints['Z'])
            automaticSpecialPoints['-B']     = -automaticSpecialPoints['B']

            automaticSpecialPoints['C']      = self._calcCircumcenter(2*automaticSpecialPoints['Z'],2*automaticSpecialPoints['-X'])
            automaticSpecialPoints['-C']     = -automaticSpecialPoints['C']

        # === explicit lattice vector dependency? ===
        if self.getDimensionality() != 0:

            if self.getDimensionality() == 1:
                normal = self.__vecsReciprocal[0].copy()[::-1]
                normal[1] *= -1
                trafo = np.array([self.__vecsReciprocal[0],normal]).T

            if self.getDimensionality() == 2:
                trafo = np.array([self.__vecsReciprocal[0],self.__vecsReciprocal[1]]).T

            # get rid of the explicit lattice vector dependency
            if reciprocalBasis:
                trafo = np.linalg.inv(trafo)
                for k in iter(automaticSpecialPoints.keys()):
                    automaticSpecialPoints[k] = np.dot(trafo,automaticSpecialPoints[k])

            # introduce the explicit lattice vector dependency
            if not reciprocalBasis:
                for k in iter(userdefinedSpecialPoints.keys()):
                    userdefinedSpecialPoints[k] = np.dot(trafo,userdefinedSpecialPoints[k])

        for k in iter(userdefinedSpecialPoints.keys()):
            automaticSpecialPoints[k] = userdefinedSpecialPoints[k]

        return automaticSpecialPoints

    def addSpecialPoint(self,label,pos):
        """Add a special point."""

        self.__specialPoints[label] = pos

    def addLatticevector(self,vector):
        """Add a lattice vector and calculate the reciprocal vectors."""

        # === add lattice vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Lattice vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__vecsLattice.shape[0] == 0:
            self.__vecsLattice = np.array([vector])
        else:
            self.__vecsLattice = np.append(self.__vecsLattice,[vector], axis=0)

        # validation of lattice vector number
        if self.__vecsLattice.shape[0] > 2:
            raise Exception("There must be at most 2 lattice vectors.")

        self.__vecsReciprocal = self.getReciprocalVectors()

    def addBasisvector(self,vector):
        """Add a basis vector."""

        # === add basis vector ===
        # validation of vector shape
        vector = np.array(vector)
        if vector.shape != (2,):
            raise Exception("Basis vectors have to be 2D vectors.")

        # append vector to the array of lattice vectors
        if self.__vecsBasis.shape[0] == 0:
            self.__vecsBasis = np.array([vector])
        else:
            self.__vecsBasis = np.append(self.__vecsBasis,[vector], axis=0)

    def _calcCircumcenter(self,vectorB, vectorC):
        """See http://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates."""

        D = 2*(vectorC[1]*vectorB[0]-vectorB[1]*vectorC[0])
        x = (vectorC[1]*np.vdot(vectorB,vectorB)-vectorB[1]*np.vdot(vectorC,vectorC))/D
        y = (vectorB[0]*np.vdot(vectorC,vectorC)-vectorC[0]*np.vdot(vectorB,vectorB))/D
        return np.array([x,y])

    def getKvectorsZone(self, resolution, dilation = True):
        """Calculate a matrix that contains all the kvectors of the Brillouin zone.

        kvectors = getKvectorsZone(resolution, dilation = True)
        kvectors[idxX, idxY, idxCoordinate]"""

        if self.__vecsReciprocal.shape[0] == 0:
            # === 0D Brillouin zone ===
            positions = Kpoints([[[0,0]]])

        elif self.__vecsReciprocal.shape[0] == 1:
            # === 1D Brillouin zone ===
            pos = self.__vecsReciprocal[0]/2
            positions = np.transpose([np.linspace(-pos[0],pos[0],resolution,endpoint=False),
                np.linspace(-pos[1],pos[1],resolution,endpoint=False)])

            step = positions[1]-positions[0]
            positions = np.array([positions[0]-step]+positions.tolist()+[positions[-1]+step])

            positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
            positionsMask[1:-1] = False

            positions = Kpoints(positions, mask = positionsMask)

        else:
            # === 2D Brillouin zone ===
            # reciprocal positions
            matTrafo = np.array([self.__vecsReciprocal[0], self.__vecsReciprocal[1]]).T

            reciprocalpositions     = np.empty((3*3,2))
            for n,[x,y] in enumerate(itertools.product([0,-1,1],[0,-1,1])):
                reciprocalpositions[n] = np.dot(matTrafo, [x,y])

            # calculate "radius" of the Brillouin zone
            radius = np.max(np.sqrt(np.sum(self.__vecsReciprocal**2,axis=-1)))

            # generate a matrix [IdxX, IdxY, Coord] that stores the positions inside the brillouinzone
            positions=np.mgrid[-radius:radius:2j*resolution,
                -radius:radius:2j*resolution,].transpose(1,2,0)

            # calculate the distances of the matrix points from the reciprocal positions
            distances = np.tile(positions, (reciprocalpositions.shape[0],1,1,1))
            distances -= reciprocalpositions[:,None,None,:]
            distances = np.sqrt(np.sum(distances**2,axis=-1))

            # --- mask all points that are not close to the central position ---
            positionsMask = np.argmin(distances,axis=0) > 0

            # slice the matrices
            si, se = np.where(~positionsMask)
            slice = np.s_[si.min()-1:si.max() + 2, se.min()-1:se.max() + 2]

            positions = Kpoints(positions[slice], mask = positionsMask[slice])

        return positions

    def getKvectorsBox(self, resolution):

        if self.__vecsReciprocal.shape[0] == 0:
            # === 0D Brillouin box ===
            positions = Kpoints([[[0,0]]])

        elif self.__vecsReciprocal.shape[0] == 1:
            # === 1D Brillouin box ===
            l1 = np.linalg.norm(self.__vecsReciprocal[0])

            x,step = np.linspace(0, l1, resolution,endpoint=False,retstep=True)
            x = np.array([x[0]-step]+x.tolist()+[x[-1]+step])
            y = np.zeros_like(x)

            positions=np.transpose([x,y],(1,0))

            a = -np.arctan2(self.__vecsReciprocal[0,1],self.__vecsReciprocal[0,0])
            matRotate = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]).T
            positions = np.dot(positions,matRotate)

            positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
            positionsMask[1:-1] = False

            positions = Kpoints(positions, mask = positionsMask)

        else:
            # === 2D Brillouin box ===
            l1 = np.linalg.norm(self.__vecsReciprocal[0])
            l2 = np.linalg.norm(self.__vecsReciprocal[1])

            angle = np.abs(np.arccos(np.dot(self.__vecsReciprocal[0],self.__vecsReciprocal[1])/(l1*l2)))

            l2*=np.sin(angle)

            x,step = np.linspace(0, l1, resolution,endpoint=False,retstep=True)
            x = np.array([x[0]-step]+x.tolist()+[x[-1]+step])
            y,step = np.linspace(0, l2, resolution,endpoint=False,retstep=True)
            y = np.array([y[0]-step]+y.tolist()+[y[-1]+step])

            positions=np.transpose(np.meshgrid(x, y),(2,1,0))

            a = -np.arctan2(self.__vecsReciprocal[0,1],self.__vecsReciprocal[0,0])
            matRotate = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]).T
            positions = np.dot(positions,matRotate)

            positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
            positionsMask[1:-1,1:-1] = False

            positions = Kpoints(positions, mask = positionsMask)

        return positions

    def getKvectorsPath(self, resolution, pointlabels=None, points=None):
        """Calculate an array that contains the kvectors of a path through the Brillouin zone

        kvectors, length = getKvectorsPath(resolution, pointlabels=["G","X"])
        kvectors[idxPosition, idxCoordinate]"""

        if pointlabels is not None:
            specialPoints = self.getSpecialPoints()
            points = np.array([specialPoints[p] for p in pointlabels])
        elif points is not None:
            points = np.array(points)
        else:
            points = np.array(["G","G"])

        numPoints = points.shape[0]

        # path through the points
        stepsize = np.sum(np.sqrt(np.sum(np.diff(points,axis=0)**2,axis=-1)))/resolution

        positions = [None]*(numPoints-1)
        for n in range(1,numPoints):
            start = points[n-1]
            end = points[n]

            if stepsize == 0: steps = 1
            else: steps = np.max([np.round(np.linalg.norm(end-start)/stepsize),1])

            newpos = np.transpose([np.linspace(start[0],end[0],steps,endpoint=False),
                np.linspace(start[1],end[1],steps,endpoint=False)])

            if n == 1: # first round
                step = newpos[1]-newpos[0]
                positions[n-1] = np.array([newpos[0]-step]+newpos.tolist())
            elif n == numPoints-1: # last round
                step = newpos[1]-newpos[0]
                positions[n-1] = np.array(newpos.tolist()+[newpos[-1]+step])
            else:
                positions[n-1] = newpos

        positions = np.vstack(positions)


        # save the labels and positions of special points
        pos = positions.copy()
        specialpoints_idx = []
        for p in points:
            idx = np.nanargmin(np.sum((pos-p)**2,axis=-1))
            specialpoints_idx.append(idx)
            pos[:,0][idx] = np.nan
            pos[:,1][idx] = np.nan

        specialpoints_labels = pointlabels

        # mask
        positionsMask = np.ones(positions.shape[:-1],dtype=np.bool)
        positionsMask[1:-1] = False

        return Kpoints(positions, positionsMask, specialpoints_idx, specialpoints_labels)

    def getPositions(self, cutoff):
        """Generate all positions from the lattice vectors using [0,0] as the basis vector.

        positions = getPositions(cutoff)
        positions[idxPosition, idxCoordinate]"""

        # value that is added to the cutoff to be on the save side
        numSubs = self.numSublattices()
        pos = np.tile(self.__vecsBasis, (numSubs,1,1))
        pos -= pos.transpose(1,0,2)
        safetyregion = np.max(np.sqrt(np.sum(pos**2,axis=-1)))

        # array that will contain all positions
        positions = []

        # --- first shift (do it only if two lattice vectors exist) ---
        shiftidx1 = 0
        boolLoop1 = True
        while boolLoop1:
            if self.__vecsLattice.shape[0] >= 2:
                shiftedpos1 = shiftidx1*self.__vecsLattice[1]

                # substract the other lattice vector to be as central as possible inside the cutoff region
                shiftedpos1 -= np.round(np.vdot(shiftedpos1,self.__vecsLattice[0])/
                    np.linalg.norm(self.__vecsLattice[0])**2 )*self.__vecsLattice[0]

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
                if self.__vecsLattice.shape[0] >= 1:
                    shiftedpos0 = shiftidx0*self.__vecsLattice[0]

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

        numSubs = self.numSublattices()

        # === creation of all positions ===
        positions = self.getPositions(cutoff)
        positionsAll = np.tile(positions, (numSubs,1,1)) + self.__vecsBasis[:,None]

        return positionsAll

    def getVecsLattice(self):
        """Get array of lattice vectors"""

        return self.__vecsLattice

    def getVecsBasis(self):
        """Get array of basis vectors"""

        return self.__vecsBasis

    def makeFiniteCircle(self, cutoff, center=[0,0]):
        """Generate a finite circular lattice.

        makeFiniteCircle(radius, center=[x,Y])"""

        positionsAll = self.getGeometry(cutoff+np.linalg.norm(center)).reshape(-1,2)

        # masking
        positionsAllAbs = np.sqrt(np.sum((positionsAll-center)**2,axis=-1))
        positionsAllMask = (positionsAllAbs > cutoff)
        positionsAll = positionsAll[~positionsAllMask]

        # save the finite system as basisvectors
        self.__vecsLattice = np.array([])
        self.__vecsReciprocal = np.array([])
        self.__vecsBasis = positionsAll

    def makeFiniteRectangle(self, cutoffX, cutoffY, center=[0,0]):
        """Generate a finite rectangular lattice.

        makeFiniteCircle(2*width,2*height, center=[x,Y])"""

        positionsAll = self.getGeometry(np.sqrt(2)*max(cutoffX,cutoffY)+np.linalg.norm(center)).reshape(-1,2)

        # masking
        positionsAllMask = (np.abs(positionsAll[:,0]-center[0]) > cutoffX) | \
            (np.abs(positionsAll[:,1]-center[1]) > cutoffY)
        positionsAll = positionsAll[~positionsAllMask]

        # save the finite system as basisvectors
        self.__vecsLattice = np.array([])
        self.__vecsReciprocal = np.array([])
        self.__vecsBasis = positionsAll

    def makeFiniteAlongdirection(self,idxVecLattice, repetitions):
        numSubs = self.numSublattices()

        # === creation of all positions ===
        positions = np.arange(repetitions)[:,None]*self.__vecsLattice[idxVecLattice][None,:]
        positionsAll = (np.tile(positions, (numSubs,1,1)) + self.__vecsBasis[:,None]).reshape(-1,2)

        # save the "more finite" system
        boolarr = np.ones(self.__vecsLattice.shape[0],dtype=bool)
        boolarr[idxVecLattice] = False

        self.__vecsLattice = self.__vecsLattice[boolarr]
        self.__vecsBasis = positionsAll

        self.__vecsReciprocal = self.getReciprocalVectors()

    def numSublattices(self):
        """Returns the number of sublattices"""

        return self.__vecsBasis.shape[0]

    def getDimensionality(self):
        """Returns the number of lattice vectors (number of periodic directions)"""

        return self.__vecsLattice.shape[0]

    def getReciprocalVectors(self):
        """Returns the reciprocal lattice vectors (and saves them internally)."""

        dim = self.getDimensionality()

        if dim == 0:
            return np.array([[0,0]])
        elif dim == 1:
            return np.array([
                2*np.pi*self.__vecsLattice[0]/np.linalg.norm(self.__vecsLattice[0])**2
                ])
        elif dim == 2:
            vecs = np.array([
                np.dot(np.array([[0,1],[-1,0]]),self.__vecsLattice[1]),
                np.dot(np.array([[0,-1],[1,0]]),self.__vecsLattice[0])
                ],dtype=np.float)
            vecs[0] = 2*np.pi*vecs[0]/ (np.vdot(self.__vecsLattice[0], vecs[0]))
            vecs[1] = 2*np.pi*vecs[1]/ (np.vdot(self.__vecsLattice[1], vecs[1]))
            return vecs
        else:
            raise Exception("Lattices with more than 2 lattice vectors are not supported")

    def plot(self, filename=None,show=True,cutoff=10):
        """Plot the lattice."""

        import matplotlib.pyplot as plt

        fig = plt.gcf()

        for p,b in zip(self.getGeometry(cutoff),self.__vecsBasis):
            line, = plt.plot(p[:,0],p[:,1], 'o',ms=4)
            fig.gca().add_artist(plt.Circle(b,cutoff, fill = False ,
                ec=line.get_color(),alpha=0.5,lw=1))
            plt.plot(b[0],b[1], 'kx',ms=7,mew=1)
        plt.axes().set_aspect('equal')
        plt.xlim(-1.5*cutoff,1.5*cutoff)
        plt.ylim(-1.5*cutoff,1.5*cutoff)

        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()

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
        shifts = self.__vecsBasis

        # === numbers ===
        # maximal number of links between the central position of a sublattice and all positions of another one
        numLinks = positions.shape[0]

        # number of sublattices
        numSubs = shifts.shape[0]

        # === creation of the distance matrix ===
        # array of central positions [Sub, Coord] that will be repeated to create the matrix matDeltaR
        positionsCentral = shifts

        # array of the positions with no shifts
        positionsNoshifts = np.tile(positions, (numSubs,1,1))
        matPositionsNoshifts = np.tile(positionsNoshifts, (numSubs,1,1,1))

        # array of all positions [Sub, Link, Coord] that will be repeated to create the matrix matDeltaR
        positionsAll = positionsNoshifts + positionsCentral[:,None]

        # creation of the matrix matDeltaR [Sub1, Sub2, Link, Coord]
        matPositionsCentral = np.tile(positionsCentral, (numLinks,numSubs, 1,1)).transpose(2,1,0,3)
        matPositionsAll = np.tile(positionsAll, (numSubs,1,1,1))
        matDeltaR = matPositionsAll-matPositionsCentral

        # masking of the matrix matDeltaR [Sub1, Sub2, Link, Coord]
        matDeltaRAbs = np.sqrt(np.sum(matDeltaR**2,axis=-1))
        matDeltaRMask = (matDeltaRAbs > cutoff) | (matDeltaRAbs < self.__tol)
        unnecessaryLinks = np.all(matDeltaRMask,axis=(0,1))

        return Distances(matDeltaR[:,:,~unnecessaryLinks],
            matPositionsNoshifts[:,:,~unnecessaryLinks],
            matDeltaRMask[:,:,~unnecessaryLinks])

    def __str__(self):
        return str({'vecsLattice': self.__vecsLattice, 'vecsBasis': self.__vecsBasis})
