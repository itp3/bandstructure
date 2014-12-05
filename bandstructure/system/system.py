import numpy as np
import multiprocessing as mp
from abc import ABCMeta, abstractmethod

from .. import Bandstructure


class System(metaclass=ABCMeta):
    """Abstract class for the implementation of a specific model system. Child classes need to
    implement tunnelingRate (and onSite)."""

    def __init__(self, params):
        self.params = params

        # TODO: get 'default cutoff' from Lattice class
        # lattice.getNearestNeighborCutoff()
        self.params.setdefault('cutoff', 1.1)

        self.setDefaultParams()

    def get(self, paramName):
        """Shortcut to a certain parameter"""
        return self.params.get(paramName)

    def setDefaultParams(self):
        """This method can be implemented by child classes to set default system parameters."""

        pass

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
        distances = self.get("lattice").getDistances(cutoff)

        self.delta = distances.noShifts

        # Get the tunneling rates for each displacement vector
        self.rates = self.tunnelingRate(distances.withShifts)

        # Check the dimension of the returned tensor
        rs = self.rates.shape
        if len(rs) != 5:
            raise Exception("tunnelingRate() needs to return a 5-tensor")

        # TODO perform more checks, like: rs[4]==rs[3] ?

        self.diag = self.onSite()

        nSublattices = self.delta.shape[0]
        nOrbitals = self.rates.shape[4]

        self.dimH = nOrbitals * nSublattices

    def getHamiltonian(self, kvec):
        """Constructs the (Bloch) Hamiltonian on the specified lattice from tunnelingRate and
        onSite energies."""

        # Compute the exp(i r k) factor
        expf = np.exp(1j * np.ma.dot(self.delta, kvec, strict=True))

        # The Hamiltonian is given by the sum over all positions:
        h = (expf[:, :, :, None, None] * self.rates).sum(2)

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

        if self.delta is None:
            self.initialize()

        if kvecs is None:
            kvecs = np.array([[0, 0]])

        # Reshape the (possibly 2D array) of vectors to a one-dimensional list
        kvecsR = kvecs.reshape((-1, 2))

        if processes == 1:
            # Use a straight map in the single-process case to allow for cleaner profiling
            results = list(map(self.solveSingle, kvecsR))
        else:
            pool = mp.Pool(processes)
            results = pool.map(workerSolveSingle, zip([self] * len(kvecsR), kvecsR))

        # Wrap back to a masked array
        energies = np.ma.array([r[0] for r in results])
        states = np.ma.array([r[1] for r in results])
        hamiltonian = np.ma.array([r[2] for r in results])

        # Reshape to the original form given by kvecs
        energies = energies.reshape(kvecs.shape[:-1] + (self.dimH,))
        states = states.reshape(kvecs.shape[:-1] + (self.dimH, self.dimH))
        hamiltonian = hamiltonian.reshape(kvecs.shape[:-1] + (self.dimH, self.dimH))

        return Bandstructure(self.params, kvecs, energies, states, hamiltonian)

    def solveSingle(self, kvec):
        """Helper function used by solve"""

        if hasattr(kvec, 'mask') and kvec.mask[0]:
            # This kvector is masked (is outside of the first Brillouin zone).
            # We return masked arrays of the correct size.

            return np.ma.masked_all((self.dimH)), \
                np.ma.masked_all((self.dimH, self.dimH)), \
                np.ma.masked_all((self.dimH, self.dimH))

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


def workerSolveSingle(args):
    return args[0].solveSingle(args[1])
