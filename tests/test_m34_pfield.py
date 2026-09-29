import unittest
import numpy as np
import torch
from geometry.linear import LinearArray
from elements.rectangular import RectangularElement
from transducer import Transducer
from field import pfield

def p4_2v():
    tr = Transducer(LinearArray(64, 0.3e-3), RectangularElement(0.25e-3), center_frequency=2.72e6, bandwidth=0.74)
    tr.height, tr.elevation_focus = 14e-3, 60e-3
    return tr

def focus_delays(tr, xf, zf, c=1540.0):
    xe = tr.pose().centers[:, 0].numpy()
    t = np.hypot(xe - xf, zf) / c
    return (t.max() - t)[None, :]

class TestM34Pfield(unittest.TestCase):
    def setUp(self):
        self.tr = p4_2v()
        self.x, self.z = np.meshgrid(np.linspace(-3e-2, 3e-2, 61), np.linspace(0.2e-2, 8e-2, 60))

    def test_focus_and_symmetry(self):
        d = focus_delays(self.tr, 0.0, 4e-2)
        P = pfield(self.tr, self.x, None, self.z, d).numpy()
        torch.testing.assert_close(torch.tensor(P), torch.tensor(P[:, ::-1].copy()), rtol=1e-3, atol=1e-6 * P.max())
        iz, ix = np.unravel_index(P.argmax(), P.shape)
        self.assertLess(abs(self.x[iz, ix]), 2e-3)
        self.assertLess(abs(self.z[iz, ix] - 4e-2), 1.5e-2)
        print("✅ pfield: simetría y foco correctos.")

    def test_nan_equals_zero_apod_and_mlt(self):
        d = focus_delays(self.tr, 1e-2, 4e-2)
        dn = d.copy(); dn[0, :20] = np.nan
        apod = np.ones(64); apod[:20] = 0
        P1 = pfield(self.tr, self.x, None, self.z, dn)
        P2 = pfield(self.tr, self.x, None, self.z, d, apod)
        torch.testing.assert_close(P1, P2, rtol=1e-4, atol=1e-6 * float(P2.max()))
        # MLT: con la misma grilla de frecuencias, el espectro de 2 emisiones simultáneas = suma de espectros
        d2 = focus_delays(self.tr, -1e-2, 4e-2)
        kw = dict(return_spectrum=True, dtype=torch.float64, df=20e3)
        _, S12, _, _ = pfield(self.tr, self.x, None, self.z, np.vstack([d, d2]), **kw)
        _, S1, _, _ = pfield(self.tr, self.x, None, self.z, d, **kw)
        _, S2, _, _ = pfield(self.tr, self.x, None, self.z, d2, **kw)
        torch.testing.assert_close(S12, S1 + S2)
        print("✅ pfield: NaN = elemento apagado y MLT = superposición.")

    def test_attenuation(self):
        d = focus_delays(self.tr, 0.0, 4e-2)
        P0 = pfield(self.tr, self.x, None, self.z, d)
        P1 = pfield(self.tr, self.x, None, self.z, d, attenuation=0.5)
        ratio = (P1 / P0)[:, 30].numpy()
        self.assertTrue(np.all(ratio <= 1.0 + 1e-6))                  # la atenuación nunca amplifica
        self.assertLess(ratio[-10:].mean(), 0.8 * ratio[:10].mean())  # y pesa más en profundidad
        print("✅ pfield: atenuación reduce el campo con la profundidad.")

    def test_gradient_wrt_delays(self):
        d = torch.tensor(focus_delays(self.tr, 0.0, 4e-2), dtype=torch.float32, requires_grad=True)
        P = pfield(self.tr, np.array([0.0]), None, np.array([4e-2]), d)
        P.sum().backward()
        self.assertTrue(torch.isfinite(d.grad).all() and d.grad.abs().sum() > 0)
        print("✅ pfield: diferenciable respecto de los retardos.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        p = pymust.getparam("P4-2v")
        d = pymust.txdelay(2e-2, 5e-2, p)
        x, z = np.meshgrid(np.linspace(-4e-2, 4e-2, 60), np.linspace(0, 10e-2, 60))
        Pm, _, _ = pymust.pfield(x, 0 * x, z, d, p.copy())
        Pt = pfield(self.tr, x, None, z, d).numpy()
        self.assertLess(np.abs(Pm / Pm.max() - Pt / Pt.max()).max(), 1e-3)
        y, z2 = np.meshgrid(np.linspace(-.75, .75, 20) * p.height, np.linspace(0, 10e-2, 40))
        x2 = np.full_like(y, 2e-2)
        Pm, _, _ = pymust.pfield(x2, y, z2, d, p.copy())
        Pt = pfield(self.tr, x2, y, z2, d).numpy()
        self.assertLess(np.abs(Pm / Pm.max() - Pt / Pt.max()).max(), 1e-3)
        print("✅ pfield coincide con pymust (2-D y 3-D con elevación).")

if __name__ == "__main__":
    unittest.main()
