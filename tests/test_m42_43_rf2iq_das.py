import unittest
import math
import numpy as np
import torch
from probes import get_probe
from delays import txdelay_plane
from processing.rf2iq import rf2iq
from processing.das import das, auto_fnumber
from processing.doppler import bmode

def point_echo_rf(tr, xs, zs, fs, n_t=1500, c=1540.0):
    """RF sintético de un punto con una onda plana a 0°: eco en t = (zs + |s - e_n|)/c, pulso gaussiano a fc."""
    fc = tr.fc
    xe = tr.pose().centers[:, 0].double()
    t = torch.arange(n_t, dtype=torch.float64)[:, None] / fs
    tau = (zs + torch.sqrt((xe - xs) ** 2 + zs ** 2)) / c
    s = 1 / (fc * 0.6)
    return torch.exp(-((t - tau) / s) ** 2) * torch.cos(2 * math.pi * fc * (t - tau))

class TestM42M43(unittest.TestCase):
    def setUp(self):
        self.tr = get_probe("L11-5v"); self.fs = 4 * self.tr.fc
        self.RF = point_echo_rf(self.tr, 3e-3, 2e-2, self.fs)

    def test_rf2iq_envelope(self):
        IQ = rf2iq(self.RF, self.fs, self.tr.fc)
        env = IQ.abs()[:, 64]
        self.assertAlmostEqual(float(env.max()), 1.0, delta=0.02)            # la envolvente del pulso vale 1
        self.assertLess(abs(int(env.argmax()) - int(self.RF[:, 64].abs().argmax())), 3)
        print("✅ rf2iq: envolvente correcta.")

    def test_das_point(self):
        x, z = np.meshgrid(np.linspace(-6e-3, 6e-3, 121), np.linspace(1.5e-2, 2.5e-2, 101))
        d = txdelay_plane(self.tr, 0.0)[0]
        for SIG, m in [(self.RF, "linear"), (rf2iq(self.RF, self.fs, self.tr.fc), "nearest")]:
            img = das(SIG, x, z, d, self.tr, self.fs, self.tr.fc, fnumber=1.5, method=m)
            env = img.abs() if torch.is_complex(img) else torch.as_tensor(np.abs(img.numpy()))
            iz, ix = np.unravel_index(int(env.argmax()), x.shape)
            self.assertLess(abs(x[iz, ix] - 3e-3), 3e-4); self.assertLess(abs(z[iz, ix] - 2e-2), 3e-4)
        self.assertEqual(bmode(img, 40).dtype, torch.uint8)
        f = auto_fnumber(0.27e-3, 7.6e6, 0.77)
        self.assertTrue(1.5 < f < 3.0)
        print("✅ DAS: el punto aparece en su posición (RF e I/Q).")

if __name__ == "__main__":
    unittest.main()
