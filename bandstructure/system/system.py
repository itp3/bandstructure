import numpy as np
import multiprocessing as mp
from abc import ABCMeta, abstractmethod


class System(metaclass=ABCMeta):
    """Abstract class for the implementation of a specific model system. Child classes need to
    implement tunnelingRate (and onSite)."""

    def __init__(self, lattice, params={}):
        self.lattice = lattice
        self.params = {}

        # TODO: get 'default cutoff' from Lattice class
        # self.set("cutoff", lattice.getNearestNeighborCutoff())
        self.set("cutoff", 1.1)

        self.setDefaultParams()
        self.setParams(params)

    def setDefaultParams(self):
        """This method can be implemented by child classes to set default system parameters."""

        pass

    def setParams(self, newParams):
        """Sets multiple parameters at once. Parameters which are already set are overwritten."""

        # Standard parameters can be overwriten by new params
        self.params.update(newParams)

    def set(self, paramName, value):
        """Set a single parameter"""

        self.params[paramName] = value

    def get(self, paramName, default=None):
        """Returns a parameter specified by its name. If the parameter does not exist and 'default'
        is given, the default value is returned."""

        try:
            return self.params[paramName]
        except KeyError:
            if default is not None:
                return default

            raise Exception("Missing parameter '{}'".format(paramName))

    def showParams(self):
        """Print a list of all parameters in this system"""

        for name, value in self.params.items():
            print("{name} = {value}".format(name=name, value=value))

    @abstractmethod
    def tunnelingRate(self, dr):
        """Returns the tunneling rate for the given tunneling process. orbFrom is the orbital on
        the initial site, orbTo is the orbital on the final site and dr is the vector connecting
        the two sites (points from initial to final site).

        This method is used is independent from the sublattice."""

        raise NotImplementedError("This method has to be implemented by a child class")

    def tunnelingRateSublattice(self):
        pass

    def onSite(self, orb):
        """Returns the energy offset of the given site. If no onsite energy is specified,
        it is assumed to be zero."""

        return 0

    def getHamiltonian(self, kvec):
        """Constructs the (Bloch) Hamiltonian on the specified lattice from tunnelingRate and
        onSite energies."""

        cutoff = self.get("cutoff")

        dr = self.lattice.getDistances(cutoff)

        expf = np.exp(1j * np.tensordot(kvec, dr, axes=(0, 3)))

        rates = self.tunnelingRate(dr)  # TODO: we don't have to compute this every time

        # TODO: include onsite energies

        return np.sum(expf * rates, 2)

    def solve(self, kvecs, processes=None):
        """Solve the system for a given (set of) vectors in the Brillouin zone. kvecs can be a
        single vector or a list of vectors. In the latter case, the number of processes/threads for
        parallel computing can be specified. If processes is set to None, all available CPUs
        will be used."""

        if type(kvecs) is list:
            pool = mp.Pool(processes)
            return pool.map(workerSolveSingle, zip([self] * len(kvecs), kvecs))
        else:
            return self.solveSingle(kvecs)

    def solveSingle(self, kvec):
        """Helper function used by solve"""

        # TODO!
        (energies, _) = np.linalg.eigh(self.getHamiltonian(kvec))
        return energies

    def getFlatness(self, band=None):
        """Returns the flatness ratio (bandgap / bandwidth) for all bands, unless a specific band
        index is given."""

        pass

    def getChernNumbers(self, band=None):
        """Returns the Chern numbers for all bands, unless a specific band index is given."""

        pass


def workerSolveSingle(args):
    return args[0].solveSingle(args[1])
