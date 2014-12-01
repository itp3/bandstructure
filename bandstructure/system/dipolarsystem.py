import numpy as np

from .system import System


class DipolarSystem(System):
    def setDefaultParams(self):
        self.params.setdefault("tbar", 1)
        self.params.setdefault("t", 0)
        self.params.setdefault("w", 3)
        self.params.setdefault("mu", 0)

    def tunnelingRate(self, dr):
        tbar = self.get("tbar")
        t = self.get("t")
        w = self.get("w")

        # 1/R^3
        dist3 = np.sum(dr ** 2, axis=3) ** (-3/2)

        # Diagonal part
        mt = np.array([[-tbar + t, 0],
                      [0, -tbar - t]], dtype=np.complex)

        # Offdiagonal part: w * exp(2 i phi)
        x = dr[:, :, :, 0]
        y = dr[:, :, :, 1]
        anglef = np.exp(2j * np.angle(x + 1j * y))

        mw = np.array([[0, 0],
                      [w, 0]])

        return dist3[:, :, :, None, None] * (
            mt +
            mw * anglef[:, :, :, None, None] +
            mw.transpose() * anglef[:, :, :, None, None].conj())

    def onSite(self):
        mu = self.get("mu")
        # return np.diag([mu, -mu])
        return np.diag([mu, -mu, mu, -mu])
