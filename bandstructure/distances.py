import numpy as np

class Distances(np.ma.MaskedArray):
    """Array to store the distances. This is just a numpy mask array with additional functionality."""

    __tol = 1e-16
    __differentdistances = None
    __absdistances = None

    @property
    def absdistances(self):
        if self.__absdistances is None:
            self.__absdistances = np.sqrt(np.sum(self**2,axis=-1))
            self.flags.writeable = False
        return self.__absdistances

    @property
    def differentdistances(self):
        if self.__differentdistances is None:
            sorteddist = np.sort(self.absdistances,axis=None)
            self.__differentdistances = sorteddist[np.append(True,np.diff(sorteddist) > self.__tol)]
            self.flags.writeable = False
        return self.__differentdistances

    def getNeighborsCutoff(self,layer = 1):
        """Return the smallest distance that occures in the lattice (if layer = 1) or
        the next smallest distance (if layer = 2).

        cutoff = getNNCutoff(layer)"""

        return self.differentdistances[layer-1]

    def getNeighbors(self,layer = 1):
        """Return a matrix of nearest neighbor distances (if layer = 1) or
        of next nearest neighbor distances (if layer = 2).

        matrices = getNNs(layer)"""

        if layer == 1:
            c = self.differentdistances[layer-1]
            matDeltaRMask = (self.absdistances > c + self.__tol)
        else:
            c = self.differentdistances[[layer-1,layer-2]]
            matDeltaRMask = (self.absdistances > c[0] + self.__tol) | (self.absdistances <= c[1] + self.__tol)

        matDeltaRMask = np.array([matDeltaRMask, matDeltaRMask]).transpose(1,2,3,0)
        return Distances(self, mask = matDeltaRMask)

    def getNeighborsMask(self,layer = 1):
        """Returns a boolean array with the corresponding layer of neighbors
        set to true."""

        return ~self.getNeighbors(layer).mask[:, :, :, 0]
