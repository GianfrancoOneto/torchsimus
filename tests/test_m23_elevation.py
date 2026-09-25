import unittest
import torch
from physics.propagation import elevation_factor

class TestM23Elevation(unittest.TestCase):
    def test_elevation_symmetry(self):
        F, S, N = 16, 5, 4
        frequencies = torch.linspace(2e6, 8e6, F, dtype=torch.float64)
        k = 2 * torch.pi * frequencies / 1540.0
        r = torch.full((S, N), 0.03, dtype=torch.float64)

        y_pos = torch.full((S, N), 0.002, dtype=torch.float64)
        y_neg = torch.full((S, N), -0.002, dtype=torch.float64)

        h = 0.01
        Rf = 0.05

        A = torch.tensor([0.25+0j, 0.25+0j, 0.25+0j, 0.25+0j], dtype=torch.complex128)
        B = torch.tensor([1.0+0.5j, 1.0-0.5j, 0.5+1.0j, 0.5-1.0j], dtype=torch.complex128)

        delta_pos = elevation_factor(y_pos, r, k, h, Rf, A, B)
        delta_neg = elevation_factor(y_neg, r, k, h, Rf, A, B)

        torch.testing.assert_close(torch.abs(delta_pos), torch.abs(delta_neg), rtol=1e-5, atol=1e-5)
        print("✅ Test de simetría en elevación (Milestone 23) superado con éxito.")

    def test_elevation_focusing(self):
        F, S, N = 16, 10, 4
        frequencies = torch.linspace(2e6, 8e6, F, dtype=torch.float64)
        k = 2 * torch.pi * frequencies / 1540.0
        r = torch.full((S, N), 0.03, dtype=torch.float64)

        y_center = torch.zeros((S, N), dtype=torch.float64)
        y_off = torch.full((S, N), 0.005, dtype=torch.float64)

        h = 0.01
        Rf = 0.03

        A = torch.tensor([0.25+0j, 0.25+0j, 0.25+0j, 0.25+0j], dtype=torch.complex128)
        B = torch.tensor([1.0+0.5j, 1.0-0.5j, 0.5+1.0j, 0.5-1.0j], dtype=torch.complex128)

        delta_center = elevation_factor(y_center, r, k, h, Rf, A, B)
        delta_off = elevation_factor(y_off, r, k, h, Rf, A, B)

        self.assertTrue(torch.all(torch.abs(delta_center) > torch.abs(delta_off)))
        print("✅ Test cualitativo de focusing en elevación (Milestone 23) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
