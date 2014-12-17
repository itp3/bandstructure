import numpy as np

from bandstructure import Parameters
from bandstructure.system import DipolarSystem
from bandstructure.lattice import HoneycombLattice


def test_chern_number_dipolar_honeyomb():
    lattice = HoneycombLattice()

    params = Parameters({
        'lattice': lattice,
        'cutoff': 3,
        'tbar': 1,
        't': 0.54,
        'w': 3,
        'mu': 2.5
    })

    s = DipolarSystem(params)

    bzone = lattice.getKvectorsZone(50)

    bandstructure = s.solve(bzone)
    chernNumbers = bandstructure.getBerryFlux() / (2 * np.pi)

    np.testing.assert_almost_equal(chernNumbers, [1, 0, -3, 2], decimal=1)
