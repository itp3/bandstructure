import numpy as np

from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import SquareLattice


def test_dipolar_square():
    lattice = SquareLattice()

    params = Parameters({
        'lattice': lattice,
        'cutoff': 10.001,
        'tbar': 1,
        't': 0,
        'w': 0
    })

    s = DipolarSystem(params)

    path = lattice.getKvectorsPath(resolution=4, pointlabels=['A', 'G', 'X'])

    bandstructure = s.solve(path)

    # Energy at high symmetry points:
    g = bandstructure.energies[3][0]
    m = bandstructure.energies[1][0]
    x = bandstructure.energies[5][0]

    # The following values are calculated analytically (for cutoff=10.001) by Mathematica
    np.testing.assert_almost_equal(g, -8.407854761)
    np.testing.assert_almost_equal(m, +2.641343369)
    np.testing.assert_almost_equal(x, +0.935057004)

    s.params["tbar"] = 0
    s.params["w"] = 1
    s.initialize()
    bandstructure = s.solve(path)

    # Energy at high symmetry points:
    g = bandstructure.energies[3][0]
    m = bandstructure.energies[1][0]
    x = bandstructure.energies[5][0]

    np.testing.assert_almost_equal(g, 0.0)
    np.testing.assert_almost_equal(m, 0.0)
    np.testing.assert_almost_equal(x, -3.719360597)
