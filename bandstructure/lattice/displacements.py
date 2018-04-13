import numpy as np


class Displacements:
    """Container to store displacement vectors on a lattice."""

    __tol = 1e-10

    vectors = None
    central = None
    mask = None
    sub = None

    def __init__(self, vectors, central, mask, sub):
        self.vectors = vectors
        self.vectors[mask] = np.nan
        self.central = central
        self.mask = mask
        self.sub = sub

    def getNeighborsMask(self, layer=0):
        """Returns a boolean array with the (layer + 1)-th shell of neighbors set to true.
        layer=0 corresponds to nearest neighbors, layer=1 to next-to-nearest neighbors, etc."""

        distances = np.sqrt(np.sum(self.vectors ** 2, axis=-1))

        # Set NaNs to the largest float:
        distances[np.isnan(distances)] = np.nan_to_num(np.inf)

        uniqueDistances = np.sort(distances, axis=None)
        uniqueDistances = uniqueDistances[np.append(True, np.diff(uniqueDistances) > self.__tol)]

        cutoffHigh = uniqueDistances[layer]

        if layer == 0:
            mask = (distances < cutoffHigh + self.__tol)
        else:
            cutoffLow = uniqueDistances[layer - 1]
            mask = (distances < cutoffHigh + self.__tol) & (distances > cutoffLow + self.__tol)

        return mask
