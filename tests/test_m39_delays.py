import unittest
import math
import torch
from probes import get_probe
from delays import txdelay_focus, txdelay_plane, txdelay_diverging, txdelay3_focus, embed_subaperture, virtual_source

class TestM39Delays(unittest.TestCase):
    def setUp(self):
        self.tr = get_probe("P4-2v")
        self.xe = self.tr.pose().centers[:, 0]

    def test_focus(self):
        d = txdelay_focus(self.tr, 0.0, 4e-2)
        self.assertEqual(tuple(d.shape), (1, 64))
        self.assertAlmostEqual(float(d.min()), 0.0)
        torch.testing.assert_close(d, torch.flip(d, [1]))                     # foco en el eje -> simétrico
        self.assertGreater(float(d[0, 31]), float(d[0, 0]))                    # el centro dispara al final
        # todas las ondas llegan juntas al foco: retardo + tiempo de vuelo = constante
        tof = torch.sqrt(self.xe ** 2 + 4e-2 ** 2) / 1540
        s = d[0] + tof
        self.assertLess(float(s.max() - s.min()), 1e-15)
        self.assertEqual(tuple(txdelay_focus(self.tr, [-1e-2, 0, 1e-2], [4e-2] * 3).shape), (3, 64))   # MLT
        print("✅ txdelay_focus correcto.")

    def test_plane_and_diverging(self):
        d = txdelay_plane(self.tr, 10 * math.pi / 180)
        dd = torch.diff(d[0])
        torch.testing.assert_close(dd, torch.full_like(dd, 0.3e-3 * math.sin(10 * math.pi / 180) / 1540))
        L = float(self.xe.max() - self.xe.min())
        x0, z0 = virtual_source(L, 10 * math.pi / 180, 60 * math.pi / 180)
        torch.testing.assert_close(txdelay_diverging(self.tr, 10 * math.pi / 180, 60 * math.pi / 180), txdelay_focus(self.tr, x0, z0))
        self.assertLess(z0, 0)                                                 # fuente virtual detrás del arreglo
        print("✅ onda plana y divergente (= fuente virtual) correctas.")

    def test_subaperture_and_3d(self):
        d = embed_subaperture(txdelay_focus(get_probe("L11-5v", 24), 0, 2.5e-2), 128, 90)
        self.assertEqual(int(torch.isnan(d).sum()), 128 - 24)
        d3 = txdelay3_focus(self.tr, 0.0, 0.0, 4e-2)
        torch.testing.assert_close(d3, txdelay_focus(self.tr, 0.0, 4e-2))     # arreglo en y = 0: coinciden
        print("✅ subapertura y txdelay3 correctos.")

if __name__ == "__main__":
    unittest.main()
