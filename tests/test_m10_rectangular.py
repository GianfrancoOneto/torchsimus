import unittest
import torch
from torchsimus.elements.rectangular import RectangularElement

class TestM10Rectangular(unittest.TestCase):
    def test_on_axis(self):
        x = torch.tensor([[0.0]], dtype=torch.float64)
        z = torch.tensor([[0.02]], dtype=torch.float64)
        f = torch.tensor([5e6], dtype=torch.float64)
        elem = RectangularElement(0.3e-3)
        D = elem.directivity_2d(x, z, f)
        torch.testing.assert_close(D, torch.tensor([[[1.0]]], dtype=torch.float64))

if __name__ == "__main__":
    unittest.main()
