import numpy as np

from .system import System


class TightBindingSystem(System):
    def setDefaultParams(self):
        self.params.update({
            "t": 1,  # nearest neighbor tunneling strength
            "t2": 0  # next-nearest neighbor ..
        })

    def tunnelingRate(self, dr):
        t = self.get("t")
        t2 = self.get("t2")

        # Nearest neighbors:

        # Only with newest numpy version:
        # nn = np.linalg.norm(dr, axis=3) == 1   # TODO! get the real nearest neighbor distance
        # nnn = np.linalg.norm(dr, axis=3) == 2  # TODO!

        nn = np.sqrt(np.sum(dr ** 2, axis=3)) == 1   # TODO! get the real nearest neighbor distance
        nnn = np.sqrt(np.sum(dr ** 2, axis=3)) == 2  # TODO

        # Orbital matrix
        m = np.array([[1, 0], [0, -1]])
        # m = np.array([-t])

        return t * m * nn[:, :, :, None, None] + t2 * m * nnn[:, :, :, None, None]
