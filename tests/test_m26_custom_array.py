import unittest
import torch
from geometry.linear import LinearArray
from geometry.custom import CustomArray
from geometry.base import validate_pose
from elements.point import PointElement
from transducer import Transducer
from simulator import compute_rf_signal

class TestM26CustomArray(unittest.TestCase):
    def test_custom_linear_equivalence(self):
        fc = 5e6
        fs = 40e6
        n_fft = 512

        linear_geom = LinearArray(num_elements=6, pitch=0.3e-3)
        pose_lin = linear_geom.pose()

        # Construir CustomArray manualmente con los datos exactos del LinearArray
        custom_geom = CustomArray(pose_lin.centers, pose_lin.u, pose_lin.v, pose_lin.n)
        pose_cust = custom_geom.pose()

        validate_pose(pose_cust)

        element_shape = PointElement()
        trans_lin = Transducer(linear_geom, element_shape, center_frequency=fc, bandwidth=0.7)
        trans_cust = Transducer(custom_geom, element_shape, center_frequency=fc, bandwidth=0.7)

        scatterers = torch.tensor([[0.0, 0.0, 0.03], [0.001, 0.0, 0.035]], dtype=torch.float64)
        reflectivity = torch.tensor([1.0, 0.5], dtype=torch.float64)
        delays = torch.zeros(6, 1, dtype=torch.float64)
        apodization = torch.ones(6, 1, dtype=torch.float64)

        rf_lin, _ = compute_rf_signal(trans_lin, scatterers, reflectivity, delays, apodization, fs, n_fft)
        rf_cust, _ = compute_rf_signal(trans_cust, scatterers, reflectivity, delays, apodization, fs, n_fft)

        torch.testing.assert_close(rf_cust, rf_lin)
        print("✅ Test de equivalencia RF CustomArray vs LinearArray (Milestone 26) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
