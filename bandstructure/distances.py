import numpy as np

class Distances():
    """Array to store the distances."""

    __tol = 1e-16
    __differentdistances = None
    __absdistances = None
    __mask = None

    __withShifts = None
    __noShifts = None

    def __init__(self, withShifts, noShifts, mask):
        self.__withShifts = withShifts
        self.__withShifts[mask] = np.nan
        self.__noShifts = noShifts
        self.__noShifts[mask] = np.nan

        self.__mask = mask

    @property
    def withShifts(self):
        return self.__withShifts

    @property
    def noShifts(self):
        return self.__noShifts

    @property
    def mask(self):
        return self.__mask

    @property
    def absdistances(self):
        if self.__absdistances is None:
            self.__absdistances = np.sqrt(np.sum(self.__withShifts**2,axis=-1))
            self.setWriteprotection()
        return self.__absdistances

    @property
    def differentdistances(self):
        if self.__differentdistances is None:
            sorteddist = np.sort(self.absdistances,axis=None)
            self.__differentdistances = sorteddist[np.append(True,np.diff(sorteddist) > self.__tol)]
            self.setWriteprotection()
        return self.__differentdistances

    def setWriteprotection():
        self.__withShifts.flags.writeable = False
        self.__noShifts.flags.writeable = False

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
        return Distances(self.__withShifts, self.__noShifts, mask = matDeltaRMask)

    def getNeighborsMask(self,layer = 1):
        """Returns a boolean array with the corresponding layer of neighbors
        set to true."""

        return ~self.getNeighbors(layer).mask[:, :, :, 0]

    def plot(self, filename=None,show=True,cutoff=10):
        """Plot the distances."""

        import matplotlib.pyplot as plt

        nSubs = self.__withShifts.shape[0]
        nLinks = self.__withShifts.shape[2]

        abs = np.sqrt(np.sum(self.__withShifts**2,axis=-1)).reshape(nSubs,nLinks*nSubs).T
        plt.imshow(abs, aspect='auto',interpolation='nearest')

        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()
