import numpy as np
import multiprocessing as mp
from abc import ABCMeta, abstractmethod

from .. import Parameters


class System(metaclass=ABCMeta):
    """Abstract class for the implementation of a specific model system. Child classes need to
    implement tunnelingRate (and onSite)."""

    def __init__(self, lattice, params):
        self.lattice = lattice

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

    def tunnelingRateSublattice(self):
        pass

    def onSite(self, orb):
        """Returns the energy offset of the given site. If no onsite energy is specified,
        it is assumed to be zero."""

        return 0

    def getHamiltonian(self, kvec):
        """Constructs the (Bloch) Hamiltonian on the specified lattice from tunnelingRate and
        onSite energies."""

        # Get distances according within a certain cutoff radius
        cutoff = self.get("cutoff")
        dr = self.lattice.getDistances(cutoff)

        nSublattices = dr.shape[0]

        # Compute the exp(i k r) factor
        expf = np.exp(1j * np.tensordot(kvec, dr, axes=(0, 3)))

        # Get the tunneling rates for each displacement vector
        rates = self.tunnelingRate(dr)  # TODO: we don't have to compute 'rates' every time

        # Check the dimension of 'rates'-tensor
        rs = rates.shape
        if len(rs) != 5:
            raise Exception("tunnelingRate needs to return a 5-tensor")
        else:
            nOrbitals = rs[4]
            # TODO perform more checks, like rs[4]==rs[3]

        # TODO: include onsite energies


        # print("dr ~ {}, expf ~ {}, rates ~ {}, h ~ {}".format(dr.shape, expf.shape, rates.shape, h.shape))

        # The Hamiltonian is the sum over all positions:
        h = (expf[:, :, :, None, None] * rates).sum(2)

        # Reshape Hamiltonian
        dimH = nOrbitals * nSublattices
        h = h.transpose((0, 2, 1, 3)).reshape((dimH, dimH))

        return h

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
        h = self.getHamiltonian(kvec)
        # print(h.shape)
        (energies, _) = np.linalg.eigh(h)
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
