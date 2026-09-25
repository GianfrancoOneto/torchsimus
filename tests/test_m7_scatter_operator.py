import unittest
import torch
from torchsimus.scattering import scatter_operator

class TestM7ScatterOperator(unittest.TestCase):
    def test_superposition(self):
        G1 = torch.randn(1, 4, dtype=torch.complex128)
        G2 = torch.randn(1, 4, dtype=torch.complex128)
        G12 = torch.cat([G1, G2], dim=0)
        R1 = torch.tensor([1.5], dtype=torch.float64)
        R2 = torch.tensor([-0.8], dtype=torch.float64)
        R12 = torch.cat([R1, R2], dim=0)
        A = torch.randn(4, 3, dtype=torch.complex128)

        Y1 = scatter_operator(G1, R1, A)
        Y2 = scatter_operator(G2, R2, A)
        Y12 = scatter_operator(G12, R12, A)
        torch.testing.assert_close(Y12, Y1 + Y2)

if __name__ == "__main__":
    unittest.main()
