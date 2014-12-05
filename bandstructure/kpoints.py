import numpy as np

class Kpoints():
    """Array to store the kpoints."""

    __points = None
    __pointsMasked = None
    __mask = None

    def __init__(self, points, mask = None):
        self.__points = np.array(points)
        if mask == None: self.__mask = np.zeros_like(self.__points,dtype=np.bool)
        else: self.__mask = np.array(mask)
        self.__pointsMasked = None

    @property
    def pointsmasked(self):
        if self.__pointsMasked == None:
            self.__pointsMasked = self.__points.copy()
            self.__pointsMasked[self.__mask] = np.nan
        return self.__pointsMasked

    @property
    def points(self):
        return self.__points

    @property
    def mask(self):
        return self.__mask

    @property
    def shape(self):
        return self.__points.shape

    @points.setter
    def points(self, value):
        self._points = np.array(value)
        self.__pointsMasked = None

    @mask.setter
    def mask(self, value):
        self._points = np.array(value)
        self.__pointsMasked = None