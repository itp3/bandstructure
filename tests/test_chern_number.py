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

    bbox = lattice.getKvectorsRhomboid(50)

    bandstructure = s.solve(bbox)

    chernNumbers = [bandstructure.getBerryFlux(n) / (2 * np.pi) for n in range(4)]
    np.testing.assert_almost_equal(chernNumbers, [1, 0, -3, 2], decimal=1)

    chernNumbers = [bandstructure.getBerryFlux(n, alternative_algorithm=True) / (2 * np.pi) for n in range(4)]
    np.testing.assert_almost_equal(chernNumbers, [1, 0, -3, 2], decimal=1)
