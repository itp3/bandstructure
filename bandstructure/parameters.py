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

            raise Exception("Missing parameter '{}'".format(paramName))

    def showParams(self):
        """Print a list of all parameters in this system"""

        for name, value in self.items():
            print("{name} = {value}".format(name=name, value=value))
