import unittest
import torch
from torchsimus.spectra import simus_spectrum, pulse_spectrum, probe_spectrum, oneway_spectrum
import numpy as np
from torchsimus.spectra import pulse_spectrum_full
from torchsimus.pulse import getpulse
from torchsimus.geometry.linear import LinearArray
from torchsimus.elements.rectangular import RectangularElement
from torchsimus.transducer import Transducer

class TestM32OnewaySpectrum(unittest.TestCase):
    def test_oneway_vs_pulse_echo(self):
        fc, bw = 3e6, 0.7
        f = torch.linspace(0, 2 * fc, 1001, dtype=torch.float64)
        # pulso-eco (simus_spectrum) = pulso x sonda^2  ->  una vía = pulso x sonda
        torch.testing.assert_close(pulse_spectrum(f, fc) * probe_spectrum(f, fc, bw) ** 2, simus_spectrum(f, fc, bw))
        torch.testing.assert_close(oneway_spectrum(f, fc, bw), pulse_spectrum(f, fc) * probe_spectrum(f, fc, bw))
        # la sonda vale 1 en fc y su respuesta pulso-eco cae 6 dB en fc*(1 +- bw/2)
        self.assertAlmostEqual(float(probe_spectrum(torch.tensor(fc, dtype=torch.float64), fc, bw)), 1.0, places=12)
        edge = torch.tensor([fc * (1 - bw / 2), fc * (1 + bw / 2)], dtype=torch.float64)
        torch.testing.assert_close(probe_spectrum(edge, fc, bw) ** 2, torch.full((2,), 0.5, dtype=torch.float64))
        print("✅ Espectros de una vía (Milestone 32) superados con éxito.")


def _pymust_fresnel_corregido(pymust):
    """pymust 0.1.9 falla en utils.fresnelint (usa `x[not issmall]`), así que su chirp nunca funciona.
    Para comparar contra la fórmula de MUST se reemplaza por la integral de Fresnel exacta de scipy."""
    from scipy.special import fresnel
    def fresnelint(x):
        S, C = fresnel(np.asarray(x, dtype=float))
        return C + 1j * S
    pymust.utils.fresnelint = fresnelint


class TestM32Pulso(unittest.TestCase):
    def setUp(self):
        self.tr = Transducer(LinearArray(64, 0.3e-3), RectangularElement(0.25e-3), center_frequency=2.72e6, bandwidth=0.74)

    def test_sin_chirp_es_el_pulso_de_siempre(self):
        f = torch.linspace(0, 6e6, 301, dtype=torch.float64)
        torch.testing.assert_close(pulse_spectrum_full(f, 2.72e6, 2), pulse_spectrum(f, 2.72e6, 2))
        print("✅ pulse_spectrum_full sin freq_sweep = pulse_spectrum.")

    def test_pulso_normalizado(self):
        p, t = getpulse(self.tr, 2)
        self.assertAlmostEqual(float(p.abs().max()), 1.0, places=6)
        self.assertEqual(p.shape, t.shape)
        print("✅ getpulse: pulso normalizado y eje de tiempo del mismo largo.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        for way in (1, 2):
            pa, _ = pymust.getpulse(pymust.getparam("P4-2v"), way)
            pb, _ = getpulse(self.tr, way)
            self.assertLess(np.linalg.norm(pa - pb.numpy()) / np.linalg.norm(pa), 1e-6)
        _pymust_fresnel_corregido(pymust)
        p = pymust.getparam("P4-2v"); p.TXnow = 3
        f = np.linspace(1e5, 6e6, 200)
        Sa = p.getPulseSpectrumFunction(1e6)(2 * np.pi * f)
        Sb = pulse_spectrum_full(torch.as_tensor(f), p.fc, 3, 1e6).numpy()
        self.assertLess(np.linalg.norm(Sa - Sb) / np.linalg.norm(Sa), 1e-6)
        print("✅ getpulse y el espectro chirp coinciden con pymust.")


if __name__ == "__main__":
    unittest.main()
