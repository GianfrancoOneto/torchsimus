import unittest
import math
import numpy as np
import torch
from processing.doppler import iq2doppler, nyquist_velocity, bmode

class TestM37Doppler(unittest.TestCase):
    def test_known_velocity(self):
        fc, prf, c = 5e6, 4000.0, 1540.0
        VN = nyquist_velocity(fc, prf, c)
        v_true = 0.3 * VN
        phase = math.pi * v_true / VN                                  # fase por emisión que produce v_true
        k = torch.arange(8, dtype=torch.float64)
        IQ = torch.exp(1j * phase * k)[None, None, :].expand(10, 12, 8)
        vel, var = iq2doppler(IQ, fc, prf, c)
        torch.testing.assert_close(vel, torch.full_like(vel, v_true))
        # señal pura: |AC| = 7 (pares con lag 1) y P = 8  ->  var = 2 (VN/pi)^2 (1 - 7/8)
        torch.testing.assert_close(var, torch.full_like(var, 2 * (VN / math.pi) ** 2 * (1 - 7 / 8)))
        print("✅ iq2doppler recupera la velocidad exacta.")

    def test_bmode(self):
        IQ = torch.tensor([[1.0 + 0j, 0.1, 0.001]])
        B = bmode(IQ, 40)
        self.assertEqual(B.dtype, torch.uint8)
        self.assertEqual(B.tolist(), [[255, 127, 0]])
        print("✅ bmode: compresión logarítmica de 8 bits.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        rng = np.random.default_rng(0)
        IQ = rng.standard_normal((20, 15, 5)) + 1j * rng.standard_normal((20, 15, 5))
        p = pymust.utils.Param(); p.fc = 5e6; p.PRF = 4000.0
        vm, varm = pymust.iq2doppler(IQ, p, np.array([5, 5]))
        vt, vart = iq2doppler(IQ, 5e6, 4000.0, M=(5, 5))
        np.testing.assert_allclose(vt.numpy(), vm, atol=1e-12)
        np.testing.assert_allclose(vart.numpy(), varm, atol=1e-12)
        np.testing.assert_array_equal(bmode(IQ[:, :, 0], 30).numpy(), pymust.bmode(IQ[:, :, 0], 30))
        print("✅ iq2doppler y bmode coinciden con pymust.")

if __name__ == "__main__":
    unittest.main()
