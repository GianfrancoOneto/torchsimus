import unittest
import math
import torch
from probes import get_probe
from processing.grids import impolgrid

class TestM41Grids(unittest.TestCase):
    def test_convex_sector(self):
        tr = get_probe("C5-2v")
        x, z = impolgrid(tr, (64, 48), 15e-2)
        self.assertEqual(tuple(x.shape), (64, 48))
        c0 = tr.geometry.curvature_center()
        r = torch.sqrt(x ** 2 + (z - c0[2]) ** 2)
        R, p = float(tr.geometry.radius), float(tr.geometry.pitch)
        torch.testing.assert_close(r[0], torch.full_like(r[0], R + p))       # primera fila: justo delante de la sonda
        self.assertAlmostEqual(float(z.max()), 15e-2, delta=1e-4)             # llega a zmax
        torch.testing.assert_close(x, -torch.flip(x, [1]))                   # simétrica
        print("✅ impolgrid (convexo) correcta.")

    def test_linear_sector(self):
        x, z = impolgrid(get_probe("P4-2v"), (50, 40), 10e-2, math.pi / 3)
        ang = torch.atan2(x, z)
        self.assertAlmostEqual(float(ang.max()), math.pi / 6, places=6)
        print("✅ impolgrid (lineal, sector de 60°) correcta.")

if __name__ == "__main__":
    unittest.main()
