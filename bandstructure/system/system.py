"""Represents a specific model system on a given lattice."""

import numpy as np


class System:
    def __init__(self, lattice, params={}):
        self.lattice = lattice
        self.params = params

    def setParams(self, newParams):
        """Sets multiple parameters at once. Parameters which are already
        set are overwritten."""

        # Standard parameters can be overwriten by new params
        self.params = dict(self.params.items() + newParams.items())

    def get(self, paramName, default=None):
        """Returns a parameter specified by its name. If the parameter
        does not exist and 'default' is given, the default value is
        returned."""

        if paramName not in self.params:
            if default is None:
                raise Exception("Unknown parameter name '" + paramName + "'")
            else:
                return default
        return self.params[paramName]

    def tunnelingRate(self, orbFrom, orbTo):
        """Returns the tunneling rate for the given process."""

        raise NotImplementedError("This method has to be implemented" +
                                  "by a child class")

    def onSite(self, orb):
        """Returns the energy offset of the given site."""

        raise NotImplementedError("This method has to be implemented" +
                                  "by a child class")

    def getFlatness(self, band=None):
        """Returns the flatness ratio (bandgap / bandwidth) for all bands,
        unless a specifig band index is given."""

        pass

    def getChernNumbers(self, band=None):
        """Returns the Chern numbers for all bands, unless a specifig band
        index is given."""

        pass
