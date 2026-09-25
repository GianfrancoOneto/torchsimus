import unittest
import math
import torch
from geometry.convex import ConvexArray
from geometry.base import validate_pose

class TestM33ConvexMUST(unittest.TestCase):
    def setUp(self):
        self.N, self.pitch, self.R = 128, 0.508e-3, 49.57e-3          # C5-2v
        self.arr = ConvexArray(self.N, self.pitch, self.R)
        self.pose = self.arr.pose()

    def test_geometry(self):
        validate_pose(self.pose)
        z = self.pose.centers[:, 2]
        self.assertAlmostEqual(float(z[0]), 0.0, places=12)          # bordes en z = 0
        self.assertAlmostEqual(float(z[-1]), 0.0, places=12)
        self.assertGreater(float(z[self.N // 2]), 0.0)                 # el centro sobresale
        d = torch.linalg.vector_norm(self.pose.centers - self.arr.curvature_center(), dim=-1)
        torch.testing.assert_close(d, torch.full_like(d, self.R))      # todos a distancia R del centro
        chord = torch.linalg.vector_norm(self.pose.centers[1:] - self.pose.centers[:-1], dim=-1)
        torch.testing.assert_close(chord, torch.full_like(chord, self.pitch))   # pitch = cuerda
        self.assertGreater(float(self.pose.n[self.N // 2, 2]), 0.999)  # normal hacia +z
        print("✅ ConvexArray en convención MUST (Milestone 33) superado con éxito.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        p = pymust.getparam("C5-2v")
        xe, ze, _, _ = p.getElementPositions()
        torch.testing.assert_close(self.pose.centers[:, 0], torch.as_tensor(xe[0], dtype=torch.float64))
        torch.testing.assert_close(self.pose.centers[:, 2], torch.as_tensor(ze[0], dtype=torch.float64))
        print("✅ Posiciones idénticas a pymust (C5-2v).")

if __name__ == "__main__":
    unittest.main()
