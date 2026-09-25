import unittest
import torch
from torchsimus.physics.propagation import propagation_2d

class TestM4Propagation(unittest.TestCase):
    def test_2d_spreading(self):
        r = torch.tensor([0.01, 0.04], dtype=torch.float64)
        f = torch.tensor(5e6)
        G = propagation_2d(r, f)
        expected = 1 / torch.sqrt(r)
        torch.testing.assert_close(torch.abs(G), expected)

if __name__ == "__main__":
    unittest.main()
