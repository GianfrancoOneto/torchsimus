import unittest
import torch
from torchsimus.geometry.base import ElementPose, validate_pose

class TestM1Pose(unittest.TestCase):
    def test_element_pose_orthogonality(self):
        centers = torch.zeros(2, 3, dtype=torch.float64)
        u = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        n = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

        pose = ElementPose(centers=centers, u=u, v=v, n=n)
        validate_pose(pose)

if __name__ == "__main__":
    unittest.main()
