import unittest
import torch
from torchsimus.elements.point import PointElement

class TestM9PointElement(unittest.TestCase):
    def test_point_element(self):
        elem = PointElement()
        r = torch.ones(2, 3, dtype=torch.float64)
        f = torch.tensor(5e6, dtype=torch.float64)
        D = elem.directivity(None, None, None, r, f)
        torch.testing.assert_close(D, torch.ones_like(r))

if __name__ == "__main__":
    unittest.main()
