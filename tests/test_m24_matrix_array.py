import unittest
import torch
from geometry.linear import LinearArray
from geometry.matrix import MatrixArray
from geometry.base import validate_pose

class TestM24MatrixArray(unittest.TestCase):
    def test_matrix_linear_equivalence(self):
        pitch = 0.3e-3
        linear_array = LinearArray(num_elements=4, pitch=pitch)
        matrix_array = MatrixArray(Nx=4, Ny=1, pitch_x=pitch, pitch_y=pitch)

        pose_linear = linear_array.pose()
        pose_matrix = matrix_array.pose()

        validate_pose(pose_matrix)

        torch.testing.assert_close(pose_matrix.centers, pose_linear.centers)
        torch.testing.assert_close(pose_matrix.u, pose_linear.u)
        torch.testing.assert_close(pose_matrix.v, pose_linear.v)
        torch.testing.assert_close(pose_matrix.n, pose_linear.n)
        print("✅ Test de equivalencia MatrixArray(Nx=4, Ny=1) vs LinearArray(4) (Milestone 24) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
