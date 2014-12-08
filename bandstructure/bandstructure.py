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
        index is given."""

        pass

    def getBerryFlux(self, band=None):
        """Returns the total Berry flux for all bands, unless a specific band index is given."""

        if self.kvecs.dim != 2:
            raise Exception("Only supports 2D k-space arrays")

        # === derivatives of the hamiltonians ===
        # determine derivatives
        Dx = np.empty_like(self.hamiltonian)
        Dx[1:-1,:] = (self.hamiltonian[2:,:]-self.hamiltonian[:-2,:])/(2*self.kvecs.dx)

        Dy = np.empty_like(self.hamiltonian)
        Dy[:,1:-1] = (self.hamiltonian[:,2:]-self.hamiltonian[:,:-2])/(2*self.kvecs.dy)

        # === loop over the bands ===
        d = self.numBands()
        fluxes = []

        if band is None: bands = range(d)
        else: bands = [band]

        for n in bands:
            #nth eigenvector
            vecn = self.states[:,:,:,n]
            #other eigenvectors
            vecm = self.states[:,:,:,np.arange(d)[np.arange(d) != n]]

            #nth eigenenergy
            en = self.energies[:,:,n]
            #other eigenenergies
            em = self.energies[:,:,np.arange(d)[np.arange(d) != n]]
            ediff = (em[:,:,:] - en[:,:,None])**2

            # put everything together
            vecnDx = np.sum(vecn.conj()[:,:,:,None]*Dx[:,:,:,:],axis=-2)
            vecnDxvexm = np.sum(vecnDx[:,:,:,None]*vecm[:,:,:,:],axis=-2)

            vecnDy = np.sum(vecn.conj()[:,:,:,None]*Dy[:,:,:,:],axis=-2)
            vecnDyvexm = np.sum(vecnDy[:,:,:,None]*vecm[:,:,:,:],axis=-2)

            # calculate Berry flux
            gamma = 2*np.imag(np.sum((vecnDxvexm/ediff)*vecnDyvexm.conj(),axis=-1))
            gamma[self.kvecs.mask] = 0

            # calculate total Berry flux
            pointsize = self.kvecs.dx*self.kvecs.dy
            flux = np.sum(gamma)*pointsize

            fluxes.append(flux)

        return np.array(fluxes)

    def getBerryPhase(self, band=0):
        """Returns the Berry phase along the underlying 1D path for all bands, unless a specific
        band index is given."""

        if self.kvecs.dim != 1:
            raise Exception("Only supports 1D k-space arrays")

        kvecs = self.kvecs.points_maskedsmall
        psi = self.states[..., band]

        # Use a smooth gauge for psi=|u_k> by choosing the first entry of |u_k> to be real
        gauge = np.exp(-1j * np.angle(psi[:, 0]))
        psi = psi * gauge[:, None]

        # Calculate numerical derivative d/dk |u_k>
        dk = np.gradient(self.kvecs.length)
        dpsi = np.zeros(psi.shape, dtype=np.complex)
        for k in range(psi.shape[1]):
            dpsi[:, k] = np.gradient(psi[:, k])
        deriv = dpsi / dk[:, None]

        # Compute <u_k| i * d/dk |u_k>
        berry = 1j * np.sum(psi.conj() * deriv, axis=1)
        berry[self.kvecs.mask] = 0

        # Integrate over path
        return np.sum(berry * dk).real

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
