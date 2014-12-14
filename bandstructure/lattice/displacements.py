import numpy as np


class Displacements:
    """Container to store displacement vectors on a lattice."""

    __tol = 1e-16
    __differentdistances = None
    __absdistances = None

    vectors = None
    withoutShifts = None
    mask = None

    def __init__(self, vectors, withoutShifts, mask):
        self.vectors = vectors
        self.vectors[mask] = np.nan
        self.withoutShifts = withoutShifts
        self.mask = mask

    @property
    def absdistances(self):
        if self.__absdistances is None:
            self.__absdistances = np.sqrt(np.sum(self.vectors ** 2, axis=-1))
            self.setWriteProtection()
        return self.__absdistances

    @property
    def differentdistances(self):
        if self.__differentdistances is None:
            sorteddist = np.sort(self.absdistances, axis=None)
            self.__differentdistances = sorteddist[np.append(True, np.diff(sorteddist) > self.__tol)]
            self.setWriteProtection()
        return self.__differentdistances

    def setWriteProtection(self):
        self.vectors.flags.writeable = False
        self.withoutShifts.flags.writeable = False

    def getNeighborsCutoff(self, layer=1):
        """Return the smallest distance that occures on the lattice (layer = 1), the second
        smallest distance (layer = 2), etc."""

        return self.differentdistances[layer - 1]

    def getNeighbors(self, layer=1):
        """Return a new Displacements object with vectors to the nearest neighbors (layer = 1),
        the next-nearest neighbors (layer = 2), etc."""

        if layer == 1:
            c = self.differentdistances[layer - 1]
            matDeltaRMask = (self.absdistances > c + self.__tol)
        else:
            c = self.differentdistances[[layer - 1, layer - 2]]
            matDeltaRMask = (
                self.absdistances > c[0] + self.__tol) | (self.absdistances <= c[1] + self.__tol)

        matDeltaRMask = np.array(
            [matDeltaRMask, matDeltaRMask]).transpose(1, 2, 3, 0)
        return Displacements(self.vectors.copy(), self.withoutShifts.copy(), mask=matDeltaRMask)

    def getNeighborsMask(self, layer=1):
        """Returns a boolean array with the corresponding layer of neighbors set to true."""

        return ~self.getNeighbors(layer).mask[..., 0]

    def plot(self, filename=None, show=True, cutoff=10):
        """Plot the distances."""

        import matplotlib.pyplot as plt

        nSubs = self.vectors.shape[0]
        nLinks = self.vectors.shape[2]

        abs = np.sqrt(np.sum(self.vectors ** 2, axis=-1)).reshape(nSubs, nLinks * nSubs).T
        plt.imshow(abs, aspect='auto', interpolation='nearest')

        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()
