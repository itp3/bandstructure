import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self, system):
        self.system = system

    def plotDispersion(self):
        """Plot the dispersion relation of the bands"""

        # TODO: this just works for 1D at the moment
        kvecs = np.linspace(-np.pi, np.pi, 100)
        kvecs = list(map(lambda x: [x, 0], kvecs))
        energies = self.system.solve(kvecs)
        energies = list(map(lambda v: v[0], energies))

        plt.plot(kvecs, energies)
        plt.savefig('dispersion.png')
