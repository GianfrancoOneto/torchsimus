import unittest
import numpy as np
import torch
from torchsimus.geometry.linear import LinearArray
from torchsimus.elements.rectangular import RectangularElement
from torchsimus.transducer import Transducer
from torchsimus.field import pfield
from torchsimus.simulation import simus

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


def _pymust_fresnel_corregido(pymust):
    """pymust 0.1.9 falla en utils.fresnelint (usa `x[not issmall]`), así que su chirp nunca funciona.
    Para comparar contra la fórmula de MUST se reemplaza por la integral de Fresnel exacta de scipy."""
    from scipy.special import fresnel
    def fresnelint(x):
        S, C = fresnel(np.asarray(x, dtype=float))
        return C + 1j * S
    pymust.utils.fresnelint = fresnelint


class TestM34Simus(unittest.TestCase):
    """simus (RF con la física de MUST): reutiliza el motor de pfield y le agrega la recepción."""
    def setUp(self):
        self.tr = p4_2v()
        self.x = np.array([-1e-2, 0.0, 1.5e-2]); self.z = np.array([3e-2, 5e-2, 7e-2]); self.RC = np.array([1.0, 0.5, 0.8])
        self.d = focus_delays(self.tr, 0.0, 5e-2)

    def test_lineal_en_RC(self):
        a = simus(self.tr, self.x, None, self.z, self.RC, self.d, fs=4 * self.tr.fc)
        b = simus(self.tr, self.x, None, self.z, 2 * self.RC, self.d, fs=4 * self.tr.fc)
        torch.testing.assert_close(b, 2 * a)
        print("✅ simus es lineal en la reflectividad.")

    def test_gradiente_RC(self):
        RC = torch.tensor(self.RC, requires_grad=True)
        rf = simus(self.tr, self.x, None, self.z, RC, self.d, fs=4 * self.tr.fc)
        (rf ** 2).sum().backward()
        self.assertTrue(torch.isfinite(RC.grad).all() and RC.grad.abs().sum() > 0)
        print("✅ simus es diferenciable respecto de RC.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        p = pymust.getparam("P4-2v"); p.fs = 4 * p.fc
        d = pymust.txdelay(p, np.deg2rad(10), np.deg2rad(60))                 # onda divergente
        for att in (0.0, 0.5):
            p.attenuation = att
            RFa = pymust.simus(self.x, self.z, self.RC, d, p)[0]
            RFb = simus(self.tr, self.x, None, self.z, self.RC, d, fs=p.fs, attenuation=att).numpy()
            n = min(len(RFa), len(RFb))
            self.assertLess(np.linalg.norm(RFa[:n] - RFb[:n]) / np.linalg.norm(RFa), 1e-3)
        print("✅ simus coincide con pymust (sin y con atenuación), sin factor de escala.")

    def test_pfield_chirp_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        _pymust_fresnel_corregido(pymust)
        x, z = np.meshgrid(np.linspace(-1e-2, 1e-2, 21), np.linspace(5e-3, 4e-2, 25))
        d = focus_delays(self.tr, 0.0, 2e-2)
        p = pymust.getparam("P4-2v"); p.TXnow = 4; p.TXfreqsweep = 1e6
        Pa = pymust.pfield(x, None, z, d, p)[0]
        Pb = pfield(self.tr, x, None, z, d, n_cycles=4, freq_sweep=1e6, dtype=torch.float64).numpy()
        self.assertLess(np.linalg.norm(Pa - Pb) / np.linalg.norm(Pa), 1e-3)
        print("✅ pfield con chirp coincide con pymust.")


if __name__ == "__main__":
    unittest.main()
