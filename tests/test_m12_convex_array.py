import unittest
import torch
from torchsimus.geometry.convex import ConvexArray
from torchsimus.geometry.base import validate_pose

class TestM12ConvexArray(unittest.TestCase):
    def test_convex_pose(self):
        array = ConvexArray(64, 0.3e-3, 0.04)
        validate_pose(array.pose())
        print("✅ Tests de ConvexArray (Milestone 12) superados con éxito.")

if __name__ == "__main__":
    unittest.main()
