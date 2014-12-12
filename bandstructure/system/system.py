import numpy as np
import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from warnings import warn

from .. import Bandstructure
from .. import Kpoints
from .. import Parameters


class System(metaclass=ABCMeta):
    """Abstract class for the implementation of a specific model system. Child classes need to
    implement tunnelingRate (and onSite)."""

    def __init__(self, params=Parameters()):
        # set passed params
        self.setParams(params)

        # set default params
        self.setDefaultParams()
        self.params.setdefault('cutoff', 1.1) # TODO: get 'default cutoff' from Lattice class

    def get(self, paramName):
        """Shortcut to a certain parameter"""

        return self.params.get(paramName)

    def setDefaultParams(self):
        """This method can be implemented by child classes to set default system parameters."""

        pass

    def setParams(self, params):
        """Sets new parameters."""

        self.params = params
        self.hashOnInit = None # this tells the solve method that initialization is needed

    @abstractmethod
    def tunnelingRate(self, dr):
        """Returns the tunneling rate for the given tunneling process. orbFrom is the orbital on
        the initial site, orbTo is the orbital on the final site and dr is the vector connecting
        the two sites (points from initial to final site).

        This method is used is independent from the sublattice."""

        raise NotImplementedError("This method has to be implemented by a child class")

    def onSite(self):
        """Returns the onsite Hamiltonian which can include a chemical potential in the form
        of a diagonal matrix."""

        return None

    def initialize(self):
        """This needs to be run before doing any calculations on the lattice. The
        displacement vectors and all tunneling elements are calculated once."""

        # Get distances within a certain cutoff radius
        cutoff = self.get("cutoff")
        self.distances = self.get("lattice").getDistances(cutoff)

        # Get the tunneling rates for each displacement vector
        self.rates = self.tunnelingRate(self.distances.withShifts)

        # Check the dimension of the returned tensor
        rs = self.rates.shape
        if len(rs) != 5:
            raise Exception("tunnelingRate() needs to return a 5-tensor")

        # TODO perform more checks, like: rs[4]==rs[3] ?

        self.diag = self.onSite()

        nSublattices = self.rates.shape[0]
        nOrbitals = self.rates.shape[4]

        self.dimH = nOrbitals * nSublattices

        # Save parameter hash to compare when solving
        self.hashOnInit = self.params.getHash()

    def getHamiltonian(self, kvec):
        """Constructs the (Bloch) Hamiltonian on the specified lattice from tunnelingRate and
        onSite energies."""

        # Compute the exp(i r k) factor
        expf = np.exp(1j * np.dot(self.distances.noShifts, kvec))

        # The Hamiltonian is given by the sum over all positions:
        product = expf[None, None, :, None, None] * self.rates
        product[self.distances.mask] = 0
        h = (product).sum(axis=2)

        # Reshape Hamiltonian
        h = h.transpose((0, 2, 1, 3)).reshape((self.dimH, self.dimH))

        # Add onsite Hamiltonian:
        if self.diag is not None:
            h += self.diag

        return h

    def solve(self, kvecs=None, processes=None):
        """Solve the system for a given set of vectors in the Brillouin zone. kvecs can be a
        list of vectors or None. In the first case, the number of processes/threads for
        parallel computing can be specified. If processes is set to None, all available CPUs
        will be used. If kvecs is set to None, solve for k=[0, 0]."""

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
            results = list(map(self.solveSingle, kvecsR))
        else:
            pool = mp.Pool(processes)
            results = pool.map(workerSolveSingle, zip([self] * len(kvecsR), kvecsR))
            pool.close()

        if kvecs is None:
            energies = [r[0] for r in results]
            states = [r[1] for r in results]
            hamiltonian = [r[2] for r in results]
        else:
            # Wrap back to a masked array
            energies = np.ones(nomask.shape + (self.dimH,), dtype=np.float)*np.nan
            states = np.ones(nomask.shape + (self.dimH, self.dimH), dtype=np.complex)*np.nan
            hamiltonian = np.ones(nomask.shape + (self.dimH, self.dimH), dtype=np.complex)*np.nan

            energies[nomask] = [r[0] for r in results]
            states[nomask] = [r[1] for r in results]
            hamiltonian[nomask] = [r[2] for r in results]

        return Bandstructure(self.params, kvecs, energies, states, hamiltonian)

    def solveSingle(self, kvec):
        """Helper function used by solve"""

        # Diagonalize Hamiltonian
        h = self.getHamiltonian(kvec)
        return np.linalg.eigh(h) + (h,)

    def solveSweep(self, kvecs, param, pi, pf, steps, processes=None):
        """This is a helper function to solve a system for a parameter range. 'kvec' is the
        array of k-vectors to solve for (see solve). 'param' is the name of the parameter to
        loop over. 'pi' and 'pf' are the initial and final values of the parameter. 'steps' is
        the number of sampling points.

        Usage:
        >>> for mu, bs in system.solveSweep(kvecs, 'mu', 0, 10, steps=20):
        >>>     print("Flatness for mu = {mu}: {flatness}".format(mu=mu, flatness=bs.getFlatness())
        """

        for val in np.linspace(pi, pf, steps):
            self.params[param] = val
            self.initialize()
            bandstructure = self.solve(kvecs, processes)

            yield val, bandstructure

    def optimizeFlatness(self, kvecs, params, band=0, monitor=False, processes=None, maxiter=None):
        """Maximize the flatness of a certain band with respect to the given parameters."""

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
    return args[0].solveSingle(args[1])
