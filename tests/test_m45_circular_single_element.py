import math
import unittest
import numpy as np
import torch
from torchsimus.geometry.circular import CircularSingleElement
from torchsimus.geometry.custom import CustomArray
from torchsimus.geometry.base import validate_pose
from torchsimus.elements.rectangular import RectangularElement
from torchsimus.transducer import Transducer
from torchsimus.field import pfield3


class TestM45CircularSingleElement(unittest.TestCase):
    def test_un_solo_elemento(self):
        e = CircularSingleElement(3e-3)
        pose = e.pose()
        validate_pose(pose)
        self.assertEqual(pose.centers.shape, (1, 3))
        self.assertEqual(e.num_elements, 1)

    def test_parches_cubren_el_disco(self):
        R, paso = 3e-3, 0.1e-3
        p = CircularSingleElement(R).patches(paso)
        validate_pose(p)
        r = torch.linalg.vector_norm(p.centers[:, :2], dim=-1)
        self.assertTrue(bool((r < R).all()))
        self.assertLess(abs(p.centers.shape[0] * paso ** 2 / (math.pi * R ** 2) - 1), 0.02)   # área ≈ pi R^2
        torch.testing.assert_close(p.centers.mean(0), torch.zeros(3, dtype=p.centers.dtype), atol=1e-12, rtol=0)

    def test_campo_simetrico_y_haz_en_el_eje(self):
        R, paso = 2e-3, 0.2e-3
        p = CircularSingleElement(R).patches(paso)
        el = RectangularElement(paso); el.height = paso
        tr = Transducer(CustomArray(p.centers, p.u, p.v, p.n), el, center_frequency=3e6, bandwidth=0.7)
        tr.width = tr.height = paso
        d = torch.zeros(1, p.centers.shape[0], dtype=torch.float64)          # todos con la misma señal
        x, z = np.meshgrid(np.linspace(-4e-3, 4e-3, 9), np.linspace(5e-3, 30e-3, 6))
        P = pfield3(tr, x, 0 * x, z, d, dtype=torch.float64).numpy()
        np.testing.assert_allclose(P, P[:, ::-1], rtol=1e-4, atol=1e-9 * P.max())   # simétrico en x
        self.assertTrue(np.all(P.argmax(axis=1) == 4))                              # máximo en el eje
        print("✅ CircularSingleElement: 1 elemento, disco bien discretizado y campo simétrico.")


if __name__ == "__main__":
    unittest.main()
