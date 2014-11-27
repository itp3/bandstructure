import numpy as np

from .system import System


class TightBindingSystem(System):
    def setDefaultParams(self):
        self.setParams({'t': 1})

    def tunnelingRate(self, dr):
        # TODO: this is cheating, for now. We assume that dr only includes the nearest neighbors
        t = self.get("t")

        return -t
