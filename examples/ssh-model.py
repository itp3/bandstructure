import numpy as np
import matplotlib.pyplot as plt

from bandstructure import Parameters
from bandstructure.system import System
from bandstructure.lattice import RegularChain


class SSHModel(System):
    def setDefaultParams(self):
        self.params.setdefault('cutoff', 0.9)

    def tunnelingRate(self, dr):
        v = self.get("v")
        w = self.get("w")

        res = v * np.all(dr.vectors == [-0.5, 0.2], axis=3) \
            + v * np.all(dr.vectors == [0.5, -0.2], axis=3) \
            + w * np.all(dr.vectors == [0.5, 0.2], axis=3) \
            + w * np.all(dr.vectors == [-0.5, -0.2], axis=3)

        return np.array([1]) * res[:, :, :, None, None]

lattice = RegularChain()
lattice.addBasisvector([0.5, -0.2])

params = Parameters({
    'lattice': lattice,

    # tunneling strength NW, SE (inter-cell)
    'v': -0.3,

    # tunneling strength NE, SW (intra-cell)
    'w': +.5
})

s = SSHModel(params)

# Infinite chain: calculate Berry phase
bs = s.solve(lattice.getKvectorsZone(300))
print(np.round(bs.getBerryPhase(), 2))

# Finite chain: spectrum and edge states
nSites = 40
lattice.makeFiniteAlongdirection(0, 40)
s.initialize()
res = s.solve()

res.plotState(stateInd=nSites, filename="state.pdf")
plt.clf()
res.plot("spectrum.pdf")
