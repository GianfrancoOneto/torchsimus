import unittest
import numpy as np
from geometry.linear import LinearArray
from elements.rectangular import RectangularElement
from transducer import Transducer
from movie import mkmovie

class TestM36Mkmovie(unittest.TestCase):
    def setUp(self):
        self.tr = Transducer(LinearArray(64, 0.3e-3), RectangularElement(0.25e-3), center_frequency=2.72e6, bandwidth=0.74)
        xe = self.tr.pose().centers[:, 0].numpy()
        t = np.hypot(xe, 2e-2) / 1540
        self.d = (t.max() - t)[None, :]

    def test_frames_and_wavefront(self):
        F, info = mkmovie(self.tr, self.d, movie=[2, 4, 20])
        self.assertEqual(F.dtype, np.uint8)
        self.assertEqual(F.shape[:2], (len(info["Zgrid"]), len(info["Xgrid"])))
        # el frente de onda (máximo |F-127| en la columna central) baja con el tiempo
        col = np.abs(F[:, F.shape[1] // 2, :].astype(float) - 127.5)
        depth = [info["Zgrid"][col[:, k].argmax()] for k in (F.shape[2] // 5, F.shape[2] // 2)]
        self.assertGreater(depth[1], depth[0])
        print("✅ mkmovie: cuadros uint8 y frente de onda que avanza.")

    def test_attenuation(self):
        F0, _ = mkmovie(self.tr, self.d, movie=[2, 4, 20])
        F1, _ = mkmovie(self.tr, self.d, movie=[2, 4, 20], attenuation=0.5)
        k = int(F0.shape[2] * 0.8)                                    # cuadro tardío (frente profundo)
        e0 = np.abs(F0[:, :, k].astype(float) - 127.5).max()
        e1 = np.abs(F1[:, :, k].astype(float) - 127.5).max()
        self.assertLess(e1, e0)
        print("✅ mkmovie: la atenuación debilita el frente en profundidad.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        p = pymust.getparam("P4-2v"); p.movie = np.array([3, 3, 20])
        Fm, _, _ = pymust.mkmovie(self.d, p)
        Ft, _ = mkmovie(self.tr, self.d, movie=[3, 3, 20])
        # pymust reordena el espectro con orden de Fortran y lo reinterpreta en orden C:
        # con una grilla cuadrada eso equivale a transponer cada cuadro.
        self.assertLessEqual(np.abs(Fm.astype(int) - Ft.transpose(1, 0, 2).astype(int)).max(), 2)
        print("✅ mkmovie coincide con pymust (salvo el reordenamiento de pymust).")

if __name__ == "__main__":
    unittest.main()
