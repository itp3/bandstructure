import numpy as np


class Bandstructure:
    def __init__(self, params, kvecs, energies, states, hamiltonian):
        self.params = params
        self.kvecs = kvecs
        self.energies = energies
        self.states = states
        self.hamiltonian = hamiltonian

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

        if self.kvecs.dim != 2:
            raise Exception("Only supports 2D k-space arrays")

        # === derivatives of the hamiltonians ===
        # determine derivatives *dx resp. *dy
        Dx = np.empty_like(self.hamiltonian)
        Dx[1:-1,:] = (self.hamiltonian[2:,:]-self.hamiltonian[:-2,:])/2

        Dy = np.empty_like(self.hamiltonian)
        Dy[:,1:-1] = (self.hamiltonian[:,2:]-self.hamiltonian[:,:-2])/2

        nb = self.numBands()

        if band is None:
            bands = range(nb)
        else:
            bands = [band]

        fluxes = []
        for n in bands:
            #nth eigenvector
            vecn = self.states[:,:,:,n]
            #other eigenvectors
            vecm = self.states[:,:,:,np.arange(nb)[np.arange(nb) != n]]

            #nth eigenenergy
            en = self.energies[:,:,n]
            #other eigenenergies
            em = self.energies[:,:,np.arange(nb)[np.arange(nb) != n]]
            ediff = (em[:,:,:] - en[:,:,None])**2

            # put everything together
            vecnDx = np.sum(vecn.conj()[:,:,:,None]*Dx[:,:,:,:],axis=-2)
            vecnDxvexm = np.sum(vecnDx[:,:,:,None]*vecm[:,:,:,:],axis=-2)

            vecnDy = np.sum(vecn.conj()[:,:,:,None]*Dy[:,:,:,:],axis=-2)
            vecnDyvexm = np.sum(vecnDy[:,:,:,None]*vecm[:,:,:,:],axis=-2)

            # calculate Berry flux
            gamma = 2*np.imag(np.sum((vecnDxvexm/ediff)*vecnDyvexm.conj(),axis=-1))
            gamma[self.kvecs.mask] = 0

            # calculate total Berry flux and save the result
            fluxes.append(np.sum(gamma))

        return np.squeeze(fluxes)

    def getBerryPhase(self, band=None):
        """Returns the Berry phase along the underlying 1D path for all bands, unless a specific
        band index is given."""

        if self.kvecs.dim != 1:
            raise Exception("Only supports 1D k-space arrays")

        nb = self.numBands()

        if band is None:
            bands = range(nb)
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
            deriv[1:-1,:] = (psi[2:]-psi[:-2])/2

            # Compute <u_k| i * d/dk |u_k> dk
            berry = 1j * np.sum(psi.conj() * deriv, axis=1)
            berry[self.kvecs.mask] = 0

            # Integrate over path and save the result
            phases.append(np.sum(berry).real)

        return np.squeeze(phases)

    def plot(self, filename=None, show=True):
        """Plot the band structure."""

        import matplotlib.pyplot as plt

        if self.kvecs.dim == 1:
            plt.plot(self.kvecs.length, self.energies)

            if self.kvecs.specialpoints_idx is not None:
                specialpoints = self.kvecs.length[self.kvecs.specialpoints_idx]
                plt.xticks(specialpoints, self.kvecs.specialpoints_labels)
                plt.xlim(min(specialpoints), max(specialpoints))
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            from matplotlib import cm
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for band in range(self.energies.shape[-1]):
                energy = self.energies[..., band].copy()
                energy[np.isnan(energy)] = np.nanmin(energy)

                ax.plot_surface(self.kvecs.points_masked[..., 0],
                                self.kvecs.points_masked[..., 1],
                                energy,
                                cstride=1,
                                rstride=1,
                                cmap=cm.coolwarm,
                                linewidth=0.03,
                                antialiased=False
                                )

        if filename is not None:
            plt.savefig(filename.format(**self.params))

        if show:
            plt.show()
