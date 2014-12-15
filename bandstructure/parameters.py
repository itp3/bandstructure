class Parameters(dict):
    """Parameter management for band structure calculations. This is just a python dictionary
    with additional functionality."""

    def __init__(self, params={}):
        super().__init__(params)

    def get(self, paramName, default=None):
        """Returns a parameter specified by its name. If the parameter does not exist and 'default'
        is given, the default value is returned."""

        try:
            return self[paramName]
        except KeyError:
            if default is not None:
                return default

            raise KeyError("Missing parameter '{}'".format(paramName)) from None

    def showParams(self):
        """Print a list of all parameters in this system"""

        for name, value in sorted(self.items()):
            print("{name} = {value}".format(name=name, value=value))

    def getJSON(self):
        """Serialize the Parameter object to JSON."""

        import json
        from .lattice import Lattice

        class LatticeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Lattice):
                    # A lattice is represented by its lattice vectors and
                    # its basis vectors alone
                    return {'vecsLattice': obj.getVecsLattice().tolist(),
                            'vecsBasis': obj.getVecsBasis().tolist()}

                return super().default(obj)

        return json.dumps(self, cls=LatticeEncoder, indent=4, sort_keys=True)

    def saveJSON(self, filename):
        """Save all parameters to a file in JSON format."""

        # Allow for filenames like parameters_{param1}_{param2}.json",
        # where param1 and param2 will be replaced by the parameter
        # values
        filename = filename.format(**self)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.getJSON())
            f.write("\n")

    def getHash(self):
        """Return an md5 hash for this set of parameters"""

        json = self.getJSON()

        import hashlib
        return hashlib.md5(json.encode()).hexdigest()
