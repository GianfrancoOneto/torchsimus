import unittest
import torch
from torchsimus.geometry.linear import LinearArray
from torchsimus.geometry.base import validate_pose

class TestM2LinearArray(unittest.TestCase):
    def test_linear_array_positions(self):
        array = LinearArray(3, 1e-3)
        pose = array.pose()
        expected = torch.tensor([-1e-3, 0, 1e-3], dtype=torch.float64)

        torch.testing.assert_close(pose.centers[:, 0], expected)
        validate_pose(pose)

if __name__ == "__main__":
    unittest.main()
