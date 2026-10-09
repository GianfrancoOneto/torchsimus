import unittest
import numpy as np
from torchsimus.phantoms import genscat


class TestM47Phantoms(unittest.TestCase):
    def test_genscat_densidad_y_3d(self):
        np.random.seed(0)
        x, y, z, RC = genscat([2e-2, 3e-2], 0.5e-3)
        self.assertTrue((x.abs() < 1e-2).all() and (z > 0).all() and (z < 3e-2).all() and (y == 0).all())
        self.assertLess(abs(len(x) / (2e-2 * 3e-2) * (0.5e-3) ** 2 - 2 / 5), 0.05)  # grilla de paso meandist/sqrt(2/5)
        x3, y3, z3, RC3 = genscat([1e-2, 1e-2, 1e-2], 1e-3)                         # en pymust 0.1.9 el 3-D falla
        self.assertTrue((y3.abs() < 5e-3).all() and len(x3) > 0)
        print("✅ genscat: dentro de la ROI, densidad correcta y 3-D funcionando.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        I = np.outer(np.hanning(40), np.hanning(30)) + 0.1
        np.random.seed(1); xa, _, za, RCa = pymust.genscat([np.nan, 4e-2], 0.4e-3, I)
        np.random.seed(1); xb, _, zb, RCb = genscat([np.nan, 4e-2], 0.4e-3, I, pymust_compat=True)
        np.testing.assert_allclose(xb.numpy(), xa); np.testing.assert_allclose(zb.numpy(), za)
        np.testing.assert_allclose(RCb.numpy(), RCa, rtol=1e-10, atol=1e-12)
        print("✅ genscat coincide con pymust (misma semilla).")


if __name__ == "__main__":
    unittest.main()
