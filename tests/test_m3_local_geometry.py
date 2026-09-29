import unittest
import torch
from torchsimus.geometry.base import ElementPose
from torchsimus.physics.geometry import local_geometry

class TestM3LocalGeometry(unittest.TestCase):
    def test_local_geometry(self):
        scatterer = torch.tensor([[3.0, 0.0, 4.0]])
        centers = torch.zeros((1, 3))
        u = torch.tensor([[1.0, 0.0, 0.0]])
        v = torch.tensor([[0.0, 1.0, 0.0]])
        n = torch.tensor([[0.0, 0.0, 1.0]])
        pose = ElementPose(centers, u, v, n)

        x, y, z, r = local_geometry(scatterer, pose)

        torch.testing.assert_close(r, torch.tensor([[5.0]]))
        torch.testing.assert_close(x, torch.tensor([[3.0]]))
        torch.testing.assert_close(z, torch.tensor([[4.0]]))

if __name__ == "__main__":
    unittest.main()
