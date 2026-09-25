import unittest
import torch
from geometry.ring import RingArray
from geometry.base import validate_pose

class TestM25RingArray(unittest.TestCase):
    def test_ring_array_properties(self):
        N = 16
        R = 0.02
        array = RingArray(num_elements=N, radius=R)
        pose = array.pose()

        # Validar ortonormalidad de frames y pose
        validate_pose(pose)

        # Verificar condición de radio x_n^2 + y_n^2 = R^2
        x = pose.centers[:, 0]
        y = pose.centers[:, 1]
        r_squared = x**2 + y**2
        expected_r_squared = torch.full_like(r_squared, R**2)
        torch.testing.assert_close(r_squared, expected_r_squared)
        print("✅ Test de RingArray (Milestone 25) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
