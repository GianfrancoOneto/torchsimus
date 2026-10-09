import unittest
import numpy as np
import torch
from torchsimus.geometry.custom import CustomArray
from torchsimus.elements.rectangular import RectangularElement
from torchsimus.transducer import Transducer
from torchsimus.field import pfield3
from torchsimus.simulation import simus3

def planar_array(xe, ye, fc, bw, width, height):
    N = xe.size
    centers = torch.tensor(np.stack([xe, ye, np.zeros(N)], -1), dtype=torch.float64)
    u = torch.zeros(N, 3, dtype=torch.float64); u[:, 0] = 1
    v = torch.zeros(N, 3, dtype=torch.float64); v[:, 1] = 1
    n = torch.zeros(N, 3, dtype=torch.float64); n[:, 2] = 1
    el = RectangularElement(width); el.height = height
    return Transducer(CustomArray(centers, u, v, n), el, center_frequency=fc, bandwidth=bw)

class TestM35Pfield3(unittest.TestCase):
    def setUp(self):
        g = (np.arange(1, 17) - 8.5) * 300e-6
        xe, ye = np.meshgrid(g, g, indexing="ij")
        self.xe, self.ye = xe.flatten(order="F"), ye.flatten(order="F")
        self.tr = planar_array(self.xe, self.ye, 3e6, 0.7, 250e-6, 250e-6)

    def delays(self, x0, y0, z0, c=1540.0):
        t = np.sqrt((self.xe - x0) ** 2 + (self.ye - y0) ** 2 + z0 ** 2) / c
        return (t.max() - t)[None, :]

    def test_focus_symmetry(self):
        d = self.delays(0, 0, 15e-3)
        y, x, z = np.meshgrid(np.linspace(-3e-3, 3e-3, 9), np.linspace(-3e-3, 3e-3, 9), np.linspace(2e-3, 30e-3, 30))
        P = pfield3(self.tr, x, y, z, d).numpy()
        np.testing.assert_allclose(P, P[::-1], rtol=2e-3, atol=1e-6 * P.max())       # simetría en x
        np.testing.assert_allclose(P, P[:, ::-1], rtol=2e-3, atol=1e-6 * P.max())    # simetría en y
        print("✅ pfield3: campo simétrico para foco en el eje.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        p = pymust.utils.Param()
        p.fc, p.bandwidth, p.width, p.height, p.radius, p.pitch = 3e6, 70, 250e-6, 250e-6, np.inf, 300e-6
        p.elements = np.array([self.xe, self.ye]); p.Nelements = self.xe.size
        d = pymust.txdelay3(0, -1e-3, 15e-3, p)
        y, x, z = np.meshgrid(np.linspace(-3e-3, 3e-3, 7), np.linspace(-3e-3, 3e-3, 7), np.linspace(0, 30e-3, 20))
        Pm, _, _ = pymust.pfield3(x, y, z, d, p)
        Pt = pfield3(self.tr, x, y, z, d).numpy()
        self.assertLess(np.abs(Pm / Pm.max() - Pt / Pt.max()).max(), 1e-3)
        print("✅ pfield3 coincide con pymust (arreglo matricial).")


class TestM35Simus3(unittest.TestCase):
    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        g = (np.arange(1, 9) - 4.5) * 300e-6
        xe, ye = np.meshgrid(g, g, indexing="ij"); xe, ye = xe.flatten(order="F"), ye.flatten(order="F")
        tr = planar_array(xe, ye, 3e6, 0.7, 250e-6, 250e-6)
        pm = pymust.utils.Param(); pm.fc = 3e6; pm.bandwidth = 70; pm.width = pm.height = 250e-6
        pm.radius = np.inf; pm.pitch = 300e-6; pm.fs = 12e6; pm.elements = np.array([xe, ye]); pm.Nelements = 64
        xs, ys, zs = np.array([0, 1e-3, -2e-3]), np.array([0, -1e-3, 2e-3]), np.array([1e-2, 1.5e-2, 2e-2])
        d = pymust.txdelay3(0, -1e-3, 15e-3, pm.copy())
        Ra = pymust.simus3(xs, ys, zs, np.ones(3), d, pm.copy())[0]
        Rb = simus3(tr, xs, ys, zs, np.ones(3), d, fs=pm.fs).numpy()
        n = min(len(Ra), len(Rb))
        self.assertLess(np.linalg.norm(Ra[:n] - Rb[:n]) / np.linalg.norm(Ra), 1e-3)
        print("✅ simus3 coincide con pymust.")


if __name__ == "__main__":
    unittest.main()
