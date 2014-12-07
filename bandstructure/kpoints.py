import numpy as np
from scipy.ndimage import binary_dilation

class Kpoints():
    """Array to store the kpoints."""

    __points = None
    __points_masked = None
    __points_maskedsmall = None

    __length = None
    __length_masked = None
    __length_maskedsmall = None

    __mask = None
    __masksmall = None

    __specialpoints_idx = None
    __specialpoints_labels = None

    __tol = 1e-12

    def __init__(self, points, mask = None, specialpoints_idx = None, specialpoints_labels = None):
        # === Process and save input ===
        self.points = points
        self.mask = mask
        self.specialpoints_idx = specialpoints_idx
        self.specialpoints_labels = specialpoints_labels

        # === Validate input ===
        if self.points.shape[-1] != 2:
            raise Exception("Points have invalid shape.")

        if (not self.mask is None) and np.any(self.points.shape[:-1] != self.mask.shape):
            raise Exception("The shape of mask is not compatible to points.")

        if (not self.specialpoints_idx is None):
            try:
                tmp = self.points[...,0].flat[specialpoints_idx]
                error = False
            except Exception as err:
                err.args = ("The array specialpoints_idx is not compatible to points.",)
                raise

        if ((not self.specialpoints_labels is None) and (self.specialpoints_idx is None)) or \
            ((self.specialpoints_labels is None) and (not self.specialpoints_idx is None)):
            raise Exception("The specialpoints are not fully defined.")

        if (not self.specialpoints_labels is None) and (not self.specialpoints_idx is None) and \
            np.any(self.specialpoints_idx.shape != self.specialpoints_labels.shape):
            raise Exception("The specialpoints_labels and specialpoints_idx differ in shape.")

    def _resetPoints(self):
        self.__points_masked = None
        self.__points_maskedsmall = None
        self.__length = None
        self.__length_masked = None
        self.__length_maskedsmall = None

    @property
    def dim(self):
        return len(self.__points.shape) - 1

    @property
    def length(self):
        if self.__length is None:
            dk = np.append([[0, 0]], np.diff(self.points.reshape(-1,2), axis=0), axis=0)
            self.__length = np.cumsum(np.sqrt(np.sum(dk**2, axis=1)))
        return self.__length

    @property
    def length_masked(self):
        if self.__length_masked is None:
            self.__length_masked = self.length.copy()
            self.__length_masked[self.__mask.ravel()] = np.nan
        return self.__length_masked

    @property
    def length_maskedsmall(self):
        if self.__length_maskedsmall is None:
            self.__length_maskedsmall = self.length.copy()
            self.__length_maskedsmall[self.__masksmall.ravel()] = np.nan
        return self.__length_maskedsmall

    @property
    def dx(self):
        diffs = np.sqrt(np.sum(np.diff(self.points_maskedsmall,axis=0)**2,axis=-1))
        diffs = diffs[~np.isnan(diffs)]
        if np.any(np.abs(diffs -diffs[0]) > self.__tol):
            raise Exception("No unique dx")
        return diffs[0]

    @property
    def dy(self):
        diffs = np.sqrt(np.sum(np.diff(self.points_maskedsmall,axis=1)**2,axis=-1))
        diffs = diffs[~np.isnan(diffs)]
        if np.any(np.abs(diffs -diffs[0]) > self.__tol):
            raise Exception("No unique dy")
        return diffs[0]

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
        self.__points = np.squeeze(points)

        self._resetPoints()

    @mask.setter
    def mask(self, mask):
        if mask is None: self.__mask = np.zeros(self.__points.shape[:-1],dtype=np.bool)
        else: self.__mask = np.squeeze(mask)
        self.__masksmall = ~binary_dilation(~self.__mask.copy())

        self._resetPoints()

    @specialpoints_idx.setter
    def specialpoints_idx(self, specialpoints_idx):
        if specialpoints_idx is None: self.__specialpoints_idx = None
        else: self.__specialpoints_idx = np.array(specialpoints_idx)

    @specialpoints_labels.setter
    def specialpoints_labels(self,specialpoints_labels):
        if specialpoints_labels is None: specialpoints_labels = None
        else: self.__specialpoints_labels = np.array(specialpoints_labels)