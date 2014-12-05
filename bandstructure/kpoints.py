import numpy as np
from scipy.ndimage import binary_dilation

class Kpoints():
    """Array to store the kpoints."""

    __points = None
    __points_masked = None
    __points_maskedsmall = None
    __mask = None
    __masksmall = None

    __specialpoints_idx = None
    __specialpoints_labels = None

    def __init__(self, points, mask = None, specialpoints_idx = None, specialpoints_labels = None):
        self.points = points
        self.mask = mask

        self.specialpoints_idx = specialpoints_idx
        self.specialpoints_labels = specialpoints_labels

        # TODO np.squeeze inside the constructor, profile all the changes, check whether the dimensions of the input are compatible
        # TODO maskForCoordinates vs. maskForPoints (maybe maskForPoints is enough)

    def _resetPoints(self):
        self.__points_masked = None
        self.__points_maskedsmall = None

    @property
    def dim(self): # TODO
        pass

    @property
    def valid(self): # TODO
        pass

    @property
    def validsmall(self): # TODO
        pass

    @property
    def points(self):
        return self.__points

    @property
    def points_masked(self):
        if self.__points_masked is None:
            self.__points_masked = self.__points.copy()
            self.__points_masked[self.__mask] = np.nan
        return self.__points_masked

    @property
    def points_maskedsmall(self):
        if self.__points_maskedsmall is None:
            self.__points_maskedsmall = self.__points.copy()
            self.__points_maskedsmall[self.__masksmall] = np.nan
        return self.__points_maskedsmall

    @property
    def mask(self):
        return self.__mask

    @property
    def masksmall(self):
        return self.__masksmall

    @property
    def shape(self):
        return self.__points.shape

    @property
    def specialpoints_idx(self):
        return self.__specialpoints_idx

    @property
    def specialpoints_labels(self):
        return self.__specialpoints_labels

    @points.setter
    def points(self, points):
        self.__points = np.array(points)

        self._resetPoints()

    @mask.setter
    def mask(self, mask):
        if mask is None: self.__mask = np.zeros_like(self.__points,dtype=np.bool)
        else: self.__mask = np.array(mask)
        self.__masksmall = ~binary_dilation(~self.__mask.copy())

        self._resetPoints()

    @specialpoints_idx.setter
    def specialpoints_idx(self, specialpoints_idx):
        self.__specialpoints_idx = specialpoints_idx

    @specialpoints_labels.setter
    def specialpoints_labels(self,specialpoints_labels):
        self.__specialpoints_labels = specialpoints_labels