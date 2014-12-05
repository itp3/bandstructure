import numpy as np
from scipy.ndimage import binary_dilation

class Kpoints():
    """Array to store the kpoints."""

    __points = None
    __points_masked = None
    __points_maskedsmall = None
    __mask = None
    __masksmall = None

    def __init__(self, points, mask = None):
        self.points = points
        self.mask = mask

    def _resetPoints(self):
        self.__points_masked = None
        self.__points_maskedsmall = None

    @property
    def points(self):
        return self.__points

    @property
    def points_masked(self):
        if self.__points_masked == None:
            self.__points_masked = self.__points.copy()
            self.__points_masked[self.__mask] = np.nan
        return self.__points_masked

    @property
    def points_maskedsmall(self):
        if self.__points_maskedsmall == None:
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

    @points.setter
    def points(self, points):
        self.__points = points

        self._resetPoints()

    @mask.setter
    def mask(self, mask):
        if mask is None: self.__mask = np.zeros_like(self.__points,dtype=np.bool)
        else: self.__mask = np.array(mask)
        self.__masksmall = ~binary_dilation(~self.__mask.copy())

        self._resetPoints()