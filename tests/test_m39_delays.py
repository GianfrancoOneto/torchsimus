import unittest
import math
import torch
from torchsimus.probes import get_probe
from torchsimus.delays import txdelay_focus, txdelay_plane, txdelay_diverging, txdelay3_focus, embed_subaperture, virtual_source
import numpy as np
from torchsimus.probes import PROBES, matrix_probe
from torchsimus.delays import txdelay3_plane, txdelay3_diverging, txdelay3_focus_line

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


NUEVAS = ["PA4-2/20", "L9-4/38", "LA530", "L14-5/38", "L14-5W/60", "P6-3"]


class TestM39Delays3(unittest.TestCase):
    def setUp(self):
        self.mp = matrix_probe(16, 16, 300e-6, 250e-6, 250e-6, fc=3e6, bandwidth=0.7)
        c = self.mp.geometry.pose().centers
        self.xe, self.ye = c[:, 0].numpy(), c[:, 1].numpy()

    def test_un_ancho_por_angulo(self):
        tr = get_probe("P4-2v")
        tilts, widths = [-0.2, 0.0, 0.2], [0.5, 1.0, 1.5]
        D = txdelay_diverging(tr, tilts, widths)
        self.assertEqual(D.shape, (3, 64))
        for k in range(3):
            torch.testing.assert_close(D[k], txdelay_diverging(tr, tilts[k], widths[k])[0])
        print("✅ txdelay_diverging acepta un ancho por ángulo.")

    def test_foco_en_linea(self):
        # pymust 0.1.9 falla con el foco en línea (concatena xe, ye en un vector en vez de una matriz 3 x N),
        # así que se compara con la distancia de cada elemento a la recta calculada directamente.
        p1, p2 = np.array([1e-3, -2e-3, 2e-2]), np.array([-1e-3, 2e-3, 2e-2])
        X = np.stack([self.xe, self.ye, np.zeros_like(self.xe)], 1)
        dist = np.linalg.norm(np.cross(X - p1, X - p2), axis=1) / np.linalg.norm(p2 - p1)
        ref = -dist / 1540.0; ref -= ref.min()
        np.testing.assert_allclose(txdelay3_focus_line(self.mp, p1, p2)[0].numpy(), ref, atol=1e-12)
        print("✅ txdelay3_focus_line: retardos = distancia a la línea focal.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        p = pymust.utils.Param(); p.fc = 3e6; p.bandwidth = 70; p.width = p.height = 250e-6
        p.radius = np.inf; p.pitch = 300e-6; p.elements = np.array([self.xe, self.ye]); p.Nelements = self.xe.size
        np.testing.assert_allclose(txdelay3_plane(self.mp, 0.1, -0.2)[0].numpy(),
                                   np.ravel(pymust.txdelay3(p.copy(), 0.1, -0.2)), atol=1e-12)
        np.testing.assert_allclose(txdelay3_diverging(self.mp, 0.1, 0.05, 0.5)[0].numpy(),
                                   np.ravel(pymust.txdelay3(p.copy(), 0.1, 0.05, 0.5)), rtol=1e-4, atol=1e-10)
        print("✅ txdelay3 (onda plana y divergente) coincide con pymust.")


class TestM40Probes(unittest.TestCase):
    def test_estan_las_10_sondas(self):
        self.assertEqual(len(PROBES), 10)
        for n in NUEVAS:
            self.assertEqual(get_probe(n).geometry.pose().centers.shape[0], PROBES[n]["num_elements"])
        print("✅ las 10 sondas de MUST están en PROBES.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        for n in NUEVAS:
            p = pymust.getparam(n)
            p.radius = np.inf                       # pymust no lo guarda para estas sondas
            q = PROBES[n]
            self.assertAlmostEqual(q["fc"], p.fc)
            self.assertAlmostEqual(q["pitch"], p.pitch)
            self.assertAlmostEqual(q["width"], p.pitch - p.kerf)
            self.assertEqual(q["num_elements"], p.Nelements)
            self.assertAlmostEqual(q["bandwidth"], (p.bandwidth if p.bandwidth is not None else 75) / 100)
            xe = get_probe(n).geometry.pose().centers[:, 0].numpy()
            np.testing.assert_allclose(xe, np.ravel(p.getElementPositions()[0]), atol=1e-12)
        print("✅ las sondas nuevas coinciden con la base de datos de pymust.")


if __name__ == "__main__":
    unittest.main()
