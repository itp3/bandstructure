import numpy as np


class Bandstructure:
    def __init__(self, params, kvectors, energies, states, hamiltonians):
        self.params = params
        self.kvectors = kvectors
        self.energies = energies
        self.states = states
        self.hamiltonians = hamiltonians

    def numBands(self):
        """Get the number of bands"""

        return self.energies.shape[-1]

    def getFlatness(self, band=None, local=False):
        """Returns the flatness ratio (bandgap / bandwidth) for all bands, unless a specific band
        index is given. If local is set to true, the flatness is calculated with the value for the
        gap replaced by a local definition for the minimal gap: min_k(E_2 - E_1), instead of
        min_k(E_2) - max_k(E_1)."""

        nb = self.numBands()

        if nb == 1:
            raise Exception("The flatness ratio is not defined for a single band.")

        if band is None:
            bands = range(nb)
        else:
            bands = [band]

        ratios = []
        for b in bands:
            gaps = []
            enThis = self.energies[..., b]

            if b >= 1:  # not the lowest band
                enBottom = self.energies[..., b - 1]
                if local:
                    gaps.append(np.nanmin(enThis - enBottom))
                else:
                    gaps.append(np.nanmin(enThis) - np.nanmax(enBottom))

            if b < nb - 1:  # not the highest band
                enTop = self.energies[..., b + 1]
                if local:
                    gaps.append(np.nanmin(enTop - enThis))
                else:
                    gaps.append(np.nanmin(enTop) - np.nanmax(enThis))

            minGap = np.nanmin(gaps)
            bandwidth = np.nanmax(self.energies[..., b]) - np.nanmin(self.energies[..., b])
            ratios.append(minGap / bandwidth)

        return np.squeeze(ratios)

    def getBerryFlux(self, band=None):
        """Returns the total Berry flux for all bands, unless a specific band index is given."""

        if self.kvectors is None or self.kvectors.dim != 2:
            raise Exception("Only supports 2D k-space arrays")

        # Derivatives of the Hamiltonian (multiplied by dx, dy)
        Dx = np.empty_like(self.hamiltonians)
        Dx[1:-1, :] = (self.hamiltonians[2:, :] - self.hamiltonians[:-2, :])/2

        Dy = np.empty_like(self.hamiltonians)
        Dy[:, 1:-1] = (self.hamiltonians[:, 2:] - self.hamiltonians[:, :-2])/2

        nb = self.numBands()

        if band is None:
            bands = range(nb)
        else:
            bands = [band]

        fluxes = []
        for n in bands:
            # nth eigenvector
            vecn = self.states[..., n]
            # other eigenvectors
            vecm = self.states[..., np.arange(nb)[np.arange(nb) != n]]

            # nth eigenenergy
            en = self.energies[..., n]
            # other eigenenergies
            em = self.energies[..., np.arange(nb)[np.arange(nb) != n]]
            ediff = (em - en[:, :, None])**2

            # put everything together
            vecnDx = np.sum(vecn.conj()[:, :, :, None] * Dx, axis=-2)
            vecnDxvexm = np.sum(vecnDx[:, :, :, None] * vecm, axis=-2)

            vecnDy = np.sum(vecn.conj()[:, :, :, None] * Dy, axis=-2)
            vecnDyvexm = np.sum(vecnDy[:, :, :, None] * vecm, axis=-2)

            # calculate Berry curvature
            gamma = -2 * np.imag(np.sum((vecnDxvexm/ediff) * vecnDyvexm.conj(), axis=-1))
            gamma[self.kvectors.mask] = 0

            # calculate Berry flux
            fluxes.append(np.sum(gamma))

        return np.squeeze(fluxes)

    def getBerryPhase(self, band=None):
        """Returns the Berry phase along the underlying 1D path for all bands, unless a specific
        band index is given."""

        if self.kvectors is None or self.kvectors.dim != 1:
            raise Exception("Only supports 1D k-space arrays")

        if band is None:
            bands = range(self.numBands())
        else:
            bands = [band]

        phases = []
        for n in bands:
            psi = self.states[..., n]

            # Use a smooth gauge for psi=|u_k> by choosing the first entry of |u_k> to be real
            gauge = np.exp(-1j * np.angle(psi[:, 0]))
            psi = psi * gauge[:, None]

            # Calculate numerical derivative d/dk |u_k> dk
            deriv = np.empty_like(psi)
            deriv[1:-1, :] = (psi[2:]-psi[:-2])/2

            # Compute <u_k| i * d/dk |u_k> dk
            berry = 1j * np.sum(psi.conj() * deriv, axis=1)
            berry[self.kvectors.mask] = 0

            # Integrate over path and save the result
            phases.append(np.sum(berry).real)

        return np.squeeze(phases)

    def plot(self, filename=None, show=True, legend=False, elim=None):
        """Plot the band structure.

        :param filename: Filename of the plot (if it is None, plot will not be saved).
        :param show:     Show the plot in a matplotlib frontend.
        :param legend:   Show a plot legend describing the bands.
        :param elim:     Limits on the "energy" axis.
        """

        import matplotlib.pyplot as plt

        if self.kvectors is None:
            # Zero-dimensional system
            plt.plot(self.energies[0], linewidth=0, marker='+')

        elif self.kvectors.dim == 1:
            for b, energy in enumerate(self.energies.T):
                plt.plot(self.kvectors.pathLength, energy, label=str(b))

            if self.kvectors.specialpoints_idx is not None:
                specialpoints = self.kvectors.pathLength[self.kvectors.specialpoints_idx]
                plt.xticks(specialpoints, self.kvectors.specialpoints_labels)
                plt.xlim(min(specialpoints), max(specialpoints))
                if elim is not None:
                    plt.ylim(elim)

            if legend:
                plt.legend()

        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            from matplotlib import cm
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            eMin = np.nanmin(self.energies)
            eMax = np.nanmax(self.energies)

            for band in range(self.energies.shape[-1]):
                energy = self.energies[..., band].copy()
                energy[np.isnan(energy)] = np.nanmin(energy)

                ax.plot_surface(self.kvectors.points_masked[..., 0],
                                self.kvectors.points_masked[..., 1],
                                energy,
                                cstride=1,
                                rstride=1,
                                cmap=cm.cool,
                                vmin=eMin,
                                vmax=eMax,
                                linewidth=0.001,
                                antialiased=True
                                )

                if elim is not None:
                    plt.zlim(elim)

        if filename is not None:
            plt.savefig(filename.format(**self.params))

        if show:
            plt.show()

    def plotState(self, kIndex=0, band=0, orbital=None):
        """Plot the probability density corresponding to the eigenstate |u_{k,nu}> where the lattice
        momentum k = kvecs[kIndex] and the band index nu = band. A specific orbital can be given.
        If orbital is set to None, a sum over all orbitals is implied."""

        pass

    def plotBerryCurvature(self, band=0):
        """Plot the Berry curvature of a specific band in the 2D Brillouin zone."""

        pass
