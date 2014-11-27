import numpy as np
import multiprocessing as mp
from abc import ABCMeta, abstractmethod


class System(metaclass=ABCMeta):
    """Abstract class for the implementation of a specific model system. Child classes need to
    implement tunnelingRate (and onSite)."""

    def __init__(self, lattice, params={}):
        self.lattice = lattice
        self.params = {}
        self.setSystemParams()
        self.setParams(params)

    def setSystemParams(self):
        """This method can be implemented by child classes to set system default parameters."""

        pass

    def setParams(self, newParams):
        """Sets multiple parameters at once. Parameters which are already set are overwritten."""

        # Standard parameters can be overwriten by new params
        self.params.update(newParams)

    def get(self, paramName, default=None):
        """Returns a parameter specified by its name. If the parameter does not exist and 'default'
        is given, the default value is returned."""

        try:
            return self.params[paramName]
        except KeyError:
            if default is not None:
                return default

            raise Exception("Wrong parameter name '{}'".format(paramName))

    def showParams(self):
        """Print a list of all parameters in this system"""

        for name, value in self.params.items():
            print("{name} = {value}".format(name=name, value=value))

    @abstractmethod
    def tunnelingRate(self, orbFrom, orbTo, dr):
        """Returns the tunneling rate for the given tunneling process. orbFrom is the orbital on
        the initial site, orbTo is the orbital on the final site and dr is the vector connecting
        the two sites (points from initial to final site)."""

        raise NotImplementedError("This method has to be implemented by a child class")

    def onSite(self, orb):
        """Returns the energy offset of the given site. If no onsite energy is specified,
        it is assumed to be zero."""

        return 0

    def initializeHamiltonian(self):
        """Constructs the Hamiltonian on the speficied lattice from tunnelingRate and onSite
        energies."""

        pass

    def solve(self, kvals, threads=1):
        """Solve the system for a given (set of) vectors in the Brillouin zone. kvals can be a
        single vector or a list of vectors. In the latter case, the number of threads for
        parallel computing can be specified."""

        if type(kvals) == 'list':
            pool = mp.Pool(threads)
            pass
        else:
            pass

    def getFlatness(self, band=None):
        """Returns the flatness ratio (bandgap / bandwidth) for all bands, unless a specific band
        index is given."""

        pass

    def getChernNumbers(self, band=None):
        """Returns the Chern numbers for all bands, unless a specific band index is given."""

        pass
