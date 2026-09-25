import unittest
import torch
from torchsimus.scattering import single_scatter

class TestM6SingleScatter(unittest.TestCase):
    def test_reflectivity_scaling(self):
        S, N, E = 4, 3, 2
        G = torch.randn(S, N, dtype=torch.complex128)
        A = torch.randn(N, E, dtype=torch.complex128)
        R1 = torch.ones(S, dtype=torch.float64)
        rx1 = single_scatter(G, R1, A)
        rx2 = single_scatter(G, 2.0 * R1, A)
        torch.testing.assert_close(rx2, 2.0 * rx1)

if __name__ == "__main__":
    unittest.main()
