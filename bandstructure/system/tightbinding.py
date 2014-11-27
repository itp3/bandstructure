import numpy as np

from .system import System


class TightBinding(System):
    def setSystemParams(self):
        self.setParams({'t': 1})

    def tunnelingRate(self, f, t, dr):
        if np.linalg.norm(dr) == 1:
            return -self.get("t")
        return 0
