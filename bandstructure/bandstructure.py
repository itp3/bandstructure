import numpy as np
from scipy.ndimage import binary_dilation

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

    def kSpaceDimension(self):
        """Returns the dimensionality of the underlying k-space array (1 or 2)."""

        return 2-np.sum(np.array(self.kvecs.shape[:2]) == 1)

    def getFlatness(self, band=None, local=False):
        """Returns the flatness ratio (bandgap / bandwidth) for all bands, unless a specific band
        index is given."""

        pass

    def getChernNumbers(self, band=None):
        """Returns the Chern numbers for all bands, unless a specific band index is given."""

        from scipy.integrate import simps

        # remove mask from kvecs
        kvecsNomask = self.kvecs.points

        # === derivatives of the hamiltonians ===
        # determine step size
        hx = np.linalg.norm(kvecsNomask[1,0,:]-kvecsNomask[0,0,:])
        hy = np.linalg.norm(kvecsNomask[0,1,:]-kvecsNomask[0,0,:])

        # determine derivatives
        Dx = np.empty_like(self.hamiltonian)
        Dx[1:-1,:] = (self.hamiltonian[2:,:]-self.hamiltonian[:-2,:])/(2*hx)

        Dy = np.empty_like(self.hamiltonian)
        Dy[:,1:-1] = (self.hamiltonian[:,2:]-self.hamiltonian[:,:-2])/(2*hy)

        # mask that can be applied to the berry flux
        mask = self.kvecs.mask[:,:,0]

        # === loop over the bands ===
        d = self.numBands()
        cherns = []

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
            gamma[mask] = 0

            # calculate Chern number
            pointsize = hx*hy
            chern = np.sum(gamma)*pointsize/(2*np.pi)

            cherns.append(chern)

        return np.array(cherns)

    def getBerryPhase(self, band=0):
        """Returns the Berry phase along the underlying 1D path for all bands, unless a specific
        band index is given."""

        if self.kSpaceDimension() != 1:
            raise Exception("Only supports 1D k-space arrays")

        states = np.squeeze(self.states)
        kvecs = np.squeeze(self.kvecs.points_maskedsmall)

        psi = states[:, :, band]

        # Use a smooth gauge for psi=|u_k> by choosing the first entry of |u_k> to be real
        gauge = np.exp(-1j * np.angle(psi[:, 0]))
        psi = psi * gauge[:, None]

        # Calculate numerical derivative d/dk |u_k>
        dk = np.gradient(kvecs[:, 0])  # TODO: calculate along any path, not just k_x
        dpsi = np.zeros(psi.shape, dtype=np.complex)
        for k in range(psi.shape[1]):
            dpsi[:, k] = np.gradient(psi[:, k])
        deriv = dpsi / dk[:, None]

        # Compute <u_k| i * d/dk |u_k>
        berry = 1j * np.sum(psi.conj() * deriv, axis=1)

        # Integrate over path
        return np.sum(berry * dk).real

    def plot(self, filename=None, show=True):
        """Plot the band structure."""

        import matplotlib.pyplot as plt

        if self.kSpaceDimension() == 1:
            points = np.squeeze(self.kvecs.points_masked)
            specialpoints_idx = np.squeeze(self.kvecs.specialpoints_idx)
            energies = np.squeeze(self.energies)

            # length of the path
            dk = np.append([[0, 0]], np.diff(points, axis=0), axis=0)
            length = np.cumsum(np.sqrt(np.sum(dk**2, axis=1)))

            plt.plot(length, energies)
            specialpoints = length[specialpoints_idx]
            plt.xticks(specialpoints, self.kvecs.specialpoints_labels)
            plt.xlim(min(specialpoints),max(specialpoints))
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            from matplotlib import cm
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for band in range(self.energies.shape[-1]):
                energy = self.energies[:, :, band].copy()
                energy[np.isnan(energy)] = np.nanmin(energy)

                ax.plot_surface(self.kvecs.points_masked[:, :, 0],
                                self.kvecs.points_masked[:, :, 1],
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
