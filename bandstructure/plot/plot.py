import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, system):
        self.system = system

    def plotDispersionPath(self, processes=None):
        """Plot the dispersion relation of the bands along a path in the Brillouin zone."""

        self.system.initialize()

        # TODO: this is just a hack for 1D right now
        kvals = np.linspace(-np.pi, np.pi, 300)
        kvecs = list(map(lambda x: [x, 0], kvals))
        energies = np.array(self.system.solve(kvecs, processes))

        plt.plot(kvals, energies)
        plt.savefig('dispersion.pdf')

    def plotDispersion2D(self):
        """Plot the dispersion relation of a 2D system over the full Brillouin zone."""

        pass
