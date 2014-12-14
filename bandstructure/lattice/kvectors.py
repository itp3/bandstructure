import numpy as np
from scipy.ndimage import binary_dilation


class Kvectors:
    """Container to store a set of reciprocal vectors."""

    __points = None
    __points_masked = None
    __points_maskedsmall = None

    __pathlength = None

    __mask = None
    __masksmall = None

    __specialpoints_idx = None
    __specialpoints_labels = None

    __tol = 1e-12

    def __init__(self, points, mask=None, specialpoints_idx=None, specialpoints_labels=None):
        self.points = points
        self.mask = mask
        self.specialpoints_idx = specialpoints_idx
        self.specialpoints_labels = specialpoints_labels

        # Validate input
        if self.points.shape[-1] != 2:
            raise Exception("Points have invalid shape.")

        if (self.mask is not None) and np.any(self.points.shape[:-1] != self.mask.shape):
            raise Exception("The shape of the mask is not compatible to the shape of the points.")

        if self.specialpoints_idx is not None:
            try:
                self.points[..., 0].flat[specialpoints_idx]
            except Exception as err:
                err.args = ("The array specialpoints_idx is not compatible with points.",)
                raise

        if ((self.specialpoints_labels is not None) and (self.specialpoints_idx is None)) or \
                ((self.specialpoints_labels is None) and (self.specialpoints_idx is not None)):
            raise Exception("The special points are not fully defined.")

        if (self.specialpoints_labels is not None) and (self.specialpoints_idx is not None) and \
                np.any(self.specialpoints_idx.shape != self.specialpoints_labels.shape):
            raise Exception("specialpoints_labels and specialpoints_idx differ in shape.")

    def _resetPoints(self):
        self.__points_masked = None
        self.__points_maskedsmall = None
        self.__pathlength = None

    @property
    def dim(self):
        """Get the dimension of the kvectors array (1D path, 2D Brillouin zone, ..)"""

        return len(self.__points.shape) - 1

    @property
    def pathLength(self):
        """For a one-dimensional set of kvectors, return an array which gives the accumulated
        path length up to a specific point."""

        if self.__pathlength is None:
            dk = np.append([[0, 0]], np.diff(self.points.reshape(-1, 2), axis=0), axis=0)
            self.__pathlength = np.cumsum(np.sqrt(np.sum(dk ** 2, axis=1)))
        return self.__pathlength

    @property
    def dx(self):
        diffs = np.sqrt(np.sum(np.diff(self.points_maskedsmall, axis=0) ** 2, axis=-1))
        diffs = diffs[~np.isnan(diffs)]
        if np.any(np.abs(diffs - diffs[0]) > self.__tol):
            raise Exception("No unique dx")
        return diffs[0]

    @property
    def dy(self):
        diffs = np.sqrt(np.sum(np.diff(self.points_maskedsmall, axis=1) ** 2, axis=-1))
        diffs = diffs[~np.isnan(diffs)]
        if np.any(np.abs(diffs - diffs[0]) > self.__tol):
            raise Exception("No unique dy")
        return diffs[0]

    @property
    def points(self):
        return self.__points

    @points.setter
    def points(self, points):
        self.__points = np.squeeze(points)
        self._resetPoints()

    @property
    def shape(self):
        return self.__points.shape

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

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self.__mask = np.zeros(self.__points.shape[:-1], dtype=np.bool)
        else:
            self.__mask = np.squeeze(mask)
        self.__masksmall = ~binary_dilation(~self.__mask.copy())

        self._resetPoints()

    @property
    def masksmall(self):
        return self.__masksmall

    @property
    def specialpoints_idx(self):
        return self.__specialpoints_idx

    @specialpoints_idx.setter
    def specialpoints_idx(self, specialpoints_idx):
        if specialpoints_idx is None:
            self.__specialpoints_idx = None
        else:
            self.__specialpoints_idx = np.array(specialpoints_idx)

    @property
    def specialpoints_labels(self):
        return self.__specialpoints_labels

    @specialpoints_labels.setter
    def specialpoints_labels(self, specialpoints_labels):
        if specialpoints_labels is None:
            specialpoints_labels = None
        else:
            self.__specialpoints_labels = np.array(specialpoints_labels)
