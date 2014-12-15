import numpy as np

from bandstructure import Parameters
from bandstructure.system import TightBindingSystem
from bandstructure.lattice import HoneycombLattice


def test_tightbinding_honeycomb():
    lattice = HoneycombLattice()

    params = Parameters({
        'lattice': lattice,
        't': 1
    })

    s = TightBindingSystem(params)

    path = lattice.getKvectorsPath(resolution=4, pointlabels=['A', 'G', 'X'])

    assert len(path.points) == 6

    bandstructure = s.solve(path)

    np.testing.assert_almost_equal(bandstructure.energies[0][0], -1)
    np.testing.assert_almost_equal(bandstructure.energies[1][0], 0)
    np.testing.assert_almost_equal(bandstructure.energies[2][0], -2)
    np.testing.assert_almost_equal(bandstructure.energies[3][0], -3)
    np.testing.assert_almost_equal(bandstructure.energies[5][0], -1)
