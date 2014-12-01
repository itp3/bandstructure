import numpy as np

from .system import System


class DipolarSystem(System):
    def setDefaultParams(self):
        self.params.setdefault("t", 1)

    def tunnelingRate(self, dr):
        t = self.get("t")

        dist3 = np.sum(dr ** 2, axis=3) ** (-3/2)

        # Orbital matrix
        # m = np.array([[1, 0], [0, -1]])
        m = np.array([-t])

        return m * dist3[:, :, :, None, None]
