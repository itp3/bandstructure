import numpy as np
import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from warnings import warn

from .. import Bandstructure
from .. import Parameters


class System(metaclass=ABCMeta):
    """Abstract class for the implementation of a specific model system. Child classes need to
    implement ``tunnelingRate()`` (and ``onSite()``)."""

    def __init__(self, params=Parameters()):
        self.setParams(params)
        self.setDefaultParams()

    def get(self, paramName):
        """Shortcut to get a certain parameter."""

        return self.params.get(paramName)

    def setDefaultParams(self):
        """This method can be implemented by child classes to set default system parameters."""

        pass

    def setParams(self, params):
        """Sets new parameters."""

        self.params = params
        self.hashOnInit = None  # this tells the solve method that initialization is needed

    @abstractmethod
    def tunnelingRate(self, displacements):
        """
        Specifies the tunneling rates for this system.

        :param displacements: ``Displacements`` object
        :returns: Returns a numpy array of tunneling rates of shape
                  ``(nSublattices, nSublattices, nPositions, nOrbitals, nOrbitals)`` where
                  ``nSublattices`` and ``nPositions`` are the number of sublattices and the
                  number of vectors connecting the central site to other lattice sites
                  (given by the ``displacements`` object). The number of orbitals
                  (internal degree of freedom) is specified by the system itself.
        """

        raise NotImplementedError("This method has to be implemented by a child class")

    def onSite(self):
        """
        Specifies the onsite Hamiltonian for this sytem.

        :returns: a numpy array of shape ``(nOrbitals, nOrbitals)`` which can include a chemical
                  potential in the form of a diagonal matrix. If ``None`` is returned, the onsite
                  Hamiltonian is assumed to be zero.
        """

        return None

    def initialize(self):
        """This needs to be run before doing any calculations on the lattice. The displacement
        vectors and all tunneling elements are calculated once."""

        # Get displacements within a certain cutoff radius
        cutoff = self.get("cutoff")
        self.displacements = self.get("lattice").getDisplacements(cutoff)

        # Get the tunneling rates for each displacement vector
        self.rates = self.tunnelingRate(self.displacements)

        # Check the dimension of the returned tensor
        rs = self.rates.shape
        if len(rs) != 5:
            raise Exception("tunnelingRate() needs to return a 5-tensor")

        self.diag = self.onSite()

        nSublattices = self.rates.shape[0]
        nOrbitals = self.rates.shape[4]

        self.dimH = nOrbitals * nSublattices

        # Save parameter hash to compare when solving
        self.hashOnInit = self.params.getHash()

    def getHamiltonian(self, kvec):
        """Constructs the (Bloch) Hamiltonian on the specified lattice.

        :param kvec: momentum for the Bloch Hamiltonian H(k).
        :returns:    A numpy array of shape ``(dimH, dimH)`` where dimH = nSublattices * nOrbitals
                     is the dimension of the Hilbert space.
        """

        # Compute the exp(i r k) factor
        expf = np.exp(1j * np.dot(self.displacements.central, kvec))

        # The Hamiltonian is given by the sum over all positions:
        product = expf[None, None, :, None, None] * self.rates
        product[self.displacements.mask] = 0
        h = (product).sum(axis=2)

        # Reshape Hamiltonian
        h = h.transpose((0, 2, 1, 3)).reshape((self.dimH, self.dimH))

        # Add onsite Hamiltonian:
        if self.diag is not None:
            h += self.diag

        return h

    def solve(self, kvecs=None, processes=None):
        """Solve the system for a given set of vectors in the Brillouin zone.

        :param kvecs: a Kvectors object or ``None``. In the latter case, solve for k=[0, 0].
        :param processes: The number of processes/threads for parallel computing. If set to
                          ``None``, all available CPUs will be used.
        """

        if self.hashOnInit is None:
            # initialization needed
            self.initialize()
        else:
            if self.params.getHash() != self.hashOnInit:
                warn("Parameters have changed since last call of System.initialize")

        if kvecs is None:
            kvecsR = [[0, 0]]
        else:
            # Mask that yields non-masked values
            nomask = ~kvecs.masksmall

            # Reshape the (possibly 2D array) of vectors to a one-dimensional list, use only
            # the non-masked values
            kvecsR = kvecs.points[nomask]

        if processes == 1:
            # Use a straight map in the single-process case to allow for cleaner profiling
            results = list(map(self._solveSingle, kvecsR))
        else:
            pool = mp.Pool(processes)
            results = pool.map(workerSolveSingle, zip([self] * len(kvecsR), kvecsR))
            pool.close()

        if kvecs is None:
            energies = np.array([r[0] for r in results])
            states = np.array([r[1] for r in results])
            hamiltonian = np.array([r[2] for r in results])
        else:
            # Wrap back to a masked array
            energies = np.ones(nomask.shape + (self.dimH,), dtype=np.float)*np.nan
            states = np.ones(nomask.shape + (self.dimH, self.dimH), dtype=np.complex)*np.nan
            hamiltonian = np.ones(nomask.shape + (self.dimH, self.dimH), dtype=np.complex)*np.nan

            energies[nomask] = [r[0] for r in results]
            states[nomask] = [r[1] for r in results]
            hamiltonian[nomask] = [r[2] for r in results]

        return Bandstructure(self.params, kvecs, energies, states, hamiltonian)

    def _solveSingle(self, kvec):
        """Helper function used by ``solve``."""

        # Diagonalize Hamiltonian
        h = self.getHamiltonian(kvec)
        return np.linalg.eigh(h) + (h,)

    def solveSweep(self, kvecs, param, pi, pf, steps, processes=None):
        """An iterator function which solves the system for a whole parameter range.

        :param kvecs: ``Kvectors`` object to solve for (see ``solve``).
        :param param: name of the parameter to loop over
        :param pi:    initial parameter value
        :param pf:    final parameter value
        :param steps: the number of sampling points.
        :yields:      the current parameter value and the corresponding ``Bandstructure`` object.

        Example:
        --------
        >>> for mu, bs in system.solveSweep(kvecs, 'mu', 0, 10, steps=20):
        >>>     print("Flatness for mu = {mu}: {flatness}".format(mu=mu, flatness=bs.getFlatness())
        """

        for val in np.linspace(pi, pf, steps):
            self.params[param] = val
            self.initialize()
            bandstructure = self.solve(kvecs, processes)

            yield val, bandstructure

    def optimizeFlatness(self, kvecs, params, band=0, monitor=False, processes=None, maxiter=None):
        """Maximize the flatness of a certain band with respect to the given parameters.

        :param kvecs: Kvectors object used as a basis for computing the bandstructure.
        :param params:    list of strings with the parameter names which may be varied.
        :param band:      Index of the band to be optimized (0: lowest band).
        :param monitor:   Print monitoring messages?
        :param processes: See ``solve``.
        :param maxiter:   Maximum number of iterations of the optimizer.
        """

        # initial parameter values
        x0 = [self.get(p) for p in params]

        def helpFlatness(x):
            for param, val in zip(params, x):
                self.params[param] = val

            self.initialize()
            bs = self.solve(kvecs, processes=processes)

            return -bs.getFlatness(band)

        def showFlatness(x):
            flatness = -helpFlatness(x)
            paramStr = ", ".join("{p} = {v}".format(p=p, v=v) for p, v in zip(params, x))
            print("flatness = {f} for {p}".format(f=flatness, p=paramStr))

        from scipy.optimize import minimize

        cb = None
        if monitor:
            cb = showFlatness

        res = minimize(helpFlatness, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter}, tol=0.01, callback=cb)

        # Set optimized parameters
        for param, val in zip(params, res.x):
            self.params[param] = val
        self.initialize()

        return res.x, -res.fun


def workerSolveSingle(args):
    return args[0]._solveSingle(args[1])
