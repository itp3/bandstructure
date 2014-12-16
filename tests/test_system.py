import numpy as np

from bandstructure import Parameters
from bandstructure.system import TightBindingSystem
from bandstructure.lattice import RegularChain


def test_paramter_change(recwarn):
    lattice = RegularChain()

    params = Parameters({
        'lattice': lattice,
        't': 1
    })

    s = TightBindingSystem(params)

    path = lattice.getKvectorsPath(resolution=2, pointlabels=['G', 'X'])

    # Solve for t = 1
    bs = s.solve(path)
    en = bs.energies

    # Change the parameter of the model
    s.params["t"] = 2

    # This is supposed to print a warning
    # because we did not call initialize()
    bs = s.solve(path)
    assert issubclass(recwarn.pop().category, UserWarning)

    # Solve again
    s.initialize()
    bs = s.solve(path)
    en2 = bs.energies

    # All energies should be twice as high
    assert np.all(2 * en == en2)
