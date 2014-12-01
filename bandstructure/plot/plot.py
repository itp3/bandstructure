import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, system):
        self.system = system

    def plotDispersionPath(self,
                           points=[[np.pi, np.pi], [0, 0], [np.pi, 0], [np.pi, np.pi]],
                           resolution=300,
                           processes=None):
        """Plot the dispersion relation of the bands along a path in the Brillouin zone."""

        self.system.initialize()

        kvecs, length = self.system.lattice.getKvectorsPath(resolution, points)
        energies = np.array(self.system.solve(kvecs, processes))

        plt.plot(length, energies)
        plt.savefig('dispersion.pdf')

    def plotDispersion(self, resolution=50, processes=None):
        """Plot the dispersion relation of the bands along a path in the Brillouin zone."""

        self.system.initialize()

        kvecs = self.system.lattice.getKvectorsZone(resolution)
        energies = self.system.solve(kvecs, processes)

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for band in range(energies.shape[-1]):
            ax.plot_surface(kvecs[:, :, 0],
                            kvecs[:, :, 1],
                            energies[:, :, band],
                            cstride=1,
                            rstride=1,
                            cmap=cm.coolwarm,
                            linewidth=0.03,
                            antialiased=False
                            )
        plt.savefig('dispersion.pdf')

    def plotDispersion2D(self):
        """Plot the dispersion relation of a 2D system over the full Brillouin zone."""

        pass
