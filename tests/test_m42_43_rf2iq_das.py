import unittest
import math
import numpy as np
import torch
from torchsimus.probes import get_probe
from torchsimus.delays import txdelay_plane
from torchsimus.processing.rf2iq import rf2iq
from torchsimus.processing.das import das, auto_fnumber
from torchsimus.processing.doppler import bmode
from torchsimus.probes import matrix_probe
from torchsimus.delays import txdelay_diverging, txdelay3_focus
from torchsimus.simulation import simus, simus3
from torchsimus.processing.das import das3
from torchsimus.processing.tgc import tgc

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


class TestM43DasCompleto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = get_probe("P4-2v")
        cls.fs = 4 * cls.tr.fc
        cls.d = txdelay_diverging(cls.tr, -20 * math.pi / 180, 60 * math.pi / 180).numpy()[0]
        x, z = np.array([-1e-2, 0.0, 1e-2]), np.array([3e-2, 4e-2, 5e-2])
        RF = simus(cls.tr, x, None, z, np.ones(3), cls.d, fs=cls.fs).numpy()
        cls.IQ = rf2iq(torch.as_tensor(RF), cls.fs, cls.tr.fc).numpy()
        cls.xi, cls.zi = np.meshgrid(np.linspace(-2e-2, 2e-2, 41), np.linspace(2e-2, 6e-2, 41))

    def test_todos_los_metodos(self):
        ref = np.abs(das(self.IQ, self.xi, self.zi, self.d, self.tr, self.fs, method="linear").numpy())
        for m in ["nearest", "quadratic", "lanczos3", "5points", "lanczos5"]:
            img = np.abs(das(self.IQ, self.xi, self.zi, self.d, self.tr, self.fs, method=m).numpy())
            self.assertGreater(np.corrcoef(img.ravel(), ref.ravel())[0, 1], 0.95, m)
        print("✅ das: los 6 métodos de interpolación dan la misma imagen (corr > 0.95).")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        p = pymust.getparam("P4-2v"); p.fs = self.fs
        for method, fnum, passive in [("linear", 0, False), ("lanczos3", None, False), ("quadratic", 1.5, True)]:
            q = p.copy(); q.fnumber = fnum
            if passive:
                q.passive = True
            M = pymust.dasmtx(self.IQ, self.xi, self.zi, self.d, q, method)
            a = (M @ self.IQ.reshape(-1, order="F")).reshape(self.xi.shape, order="F")
            b = das(self.IQ, self.xi, self.zi, self.d, self.tr, self.fs, method=method, fnumber=fnum, passive=passive).numpy()
            self.assertLess(np.linalg.norm(a - b) / np.linalg.norm(a), 1e-6, method)
        print("✅ das coincide con pymust.dasmtx (linear, lanczos3 + f-number automático, pasivo).")

    def test_das3_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        mp = matrix_probe(8, 8, 300e-6, 250e-6, 250e-6, fc=3e6, bandwidth=0.7)
        c8 = mp.geometry.pose().centers
        pm = pymust.utils.Param(); pm.fc = 3e6; pm.bandwidth = 70; pm.width = pm.height = 250e-6
        pm.radius = np.inf; pm.pitch = 300e-6; pm.fs = 12e6
        pm.elements = np.array([c8[:, 0].numpy(), c8[:, 1].numpy()]); pm.Nelements = 64
        d = txdelay3_focus(mp, 0, 0, 1.5e-2).numpy()[0]
        RF = simus3(mp, [0.0], [0.0], [1.5e-2], [1.0], d, fs=pm.fs).numpy()
        IQ = rf2iq(torch.as_tensor(RF), pm.fs, pm.fc).numpy()
        xi, yi, zi = np.meshgrid(np.linspace(-2e-3, 2e-3, 5), np.linspace(-2e-3, 2e-3, 5), np.linspace(1e-2, 2e-2, 9), indexing="ij")
        q = pm.copy(); q.fnumber = np.array([0, 0])
        M = pymust.dasmtx3(IQ, xi, yi, zi, d[None, :], q, "linear")
        a = (M @ IQ.reshape(-1, order="F")).reshape(xi.shape, order="F")
        b = das3(IQ, xi, yi, zi, d, mp, pm.fs).numpy()
        self.assertLess(np.linalg.norm(a - b) / np.linalg.norm(a), 1e-6)
        print("✅ das3 coincide con pymust.dasmtx3.")


class TestM43TGC(unittest.TestCase):
    def test_tgc_compensa_decaimiento(self):
        t = np.arange(2000)[:, None]
        S = np.exp(-t / 400) * np.cos(0.3 * t) * np.ones((1, 8))
        np.random.seed(0)
        St, C = tgc(torch.as_tensor(S))
        env = St.abs().numpy()
        self.assertLess(env[1500:1800].max() / env[200:500].max(), 3)
        print("✅ tgc compensa el decaimiento con la profundidad.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        rng = np.random.default_rng(3)
        S = rng.standard_normal((1500, 16)) * np.exp(-np.arange(1500) / 300)[:, None]
        np.random.seed(2); Ta = pymust.tgc(S)[0]
        np.random.seed(2); Tb = tgc(torch.as_tensor(S))[0].numpy()
        np.testing.assert_allclose(Tb, Ta, rtol=1e-8, atol=1e-12)
        print("✅ tgc coincide con pymust (misma semilla).")


if __name__ == "__main__":
    unittest.main()
