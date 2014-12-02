import numpy as np
import matplotlib.pyplot as plt


class Bandstructure:
    def __init__(self, params, kvecs, energies, states):
        self.params = params
        self.kvecs = kvecs
        self.energies = energies
        self.states = states

    def numBands(self):
        """Get the number of bands"""

        pass

    def kSpaceDimension(self):
        return len(self.kvecs.shape) - 1

    def getFlatness(self, band=None, local=False):
        """Returns the flatness ratio (bandgap / bandwidth) for all bands, unless a specific band
        index is given."""

        pass

    def getChernNumbers(self, band=None):
        """Returns the Chern numbers for all bands, unless a specific band index is given."""

        pass

    def plot(self, filename="dispersion.pdf", resolution=50, processes=None):
        """Plot the band structure."""

        # Fill with NaN for 2D plotting
        energies = self.energies.filled(np.nan)

        if self.kSpaceDimension() == 1:
            plt.plot(energies)
        else:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for band in range(energies.shape[-1]):
                ax.plot_surface(self.kvecs[:, :, 0],
                                self.kvecs[:, :, 1],
                                energies[:, :, band],
                                cstride=1,
                                rstride=1,
                                cmap=cm.coolwarm,
                                linewidth=0.03,
                                antialiased=False
                                )

        filename = filename.format(**self.params)
        plt.savefig(filename)
