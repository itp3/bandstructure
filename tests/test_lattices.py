import numpy as np

from bandstructure import Parameters
from bandstructure.system import TightBindingSystem
from bandstructure.lattice import RegularChain, SquareLattice, HoneycombLattice, KagomeLattice, RubyLattice, LiebLattice


def test_tightbinding_honeycomb():
    nearestNeighbors = {
          RegularChain: 2
        , SquareLattice: 4
        , HoneycombLattice: 3
        , KagomeLattice: 4
        , RubyLattice: 4
        , LiebLattice: 2 * np.sqrt(2)
    }

    for latticeT, nn in nearestNeighbors.items():
        lattice = latticeT()
        params = Parameters({
            'lattice': lattice,
            't': 1
        })

        s = TightBindingSystem(params)

        path = lattice.getKvectorsPath(resolution=3, pointlabels=['G', 'X'])
        bandstructure = s.solve(path)

        assert len(path.points) == 4

        # Energy at the Gamma point
        g = bandstructure.energies[1][0]

        print("Lattice type: {}".format(latticeT))
        np.testing.assert_almost_equal(g, - params['t'] * nn)
