import numpy as np

from .system import System


class TightBindingSystem(System):
    def setDefaultParams(self):
        self.params.setdefault('t', 1)   # nearest neighbor tunneling strength
        self.params.setdefault('t2', 0)  # next-nearest neighbor ..

    def tunnelingRate(self, dr):
        t = self.get("t")
        t2 = self.get("t2")

        # Orbital matrix
        m = np.array([[1, 0], [0, -1]])
        # m = np.array([-t])

        nn = dr.getNeighborsMask(1)
        nnn = dr.getNeighborsMask(2)
        return t * m * nn[:, :, :, None, None] + t2 * m * nnn[:, :, :, None, None]
