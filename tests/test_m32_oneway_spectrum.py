import unittest
import torch
from spectra import simus_spectrum, pulse_spectrum, probe_spectrum, oneway_spectrum

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

if __name__ == "__main__":
    unittest.main()
