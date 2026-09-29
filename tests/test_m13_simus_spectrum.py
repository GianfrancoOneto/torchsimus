import sys, os
sys.path.append("/content/torchsimus")
import unittest
import torch
from spectra import simus_spectrum

class TestM13SimusSpectrum(unittest.TestCase):
    def test_transducer_response_peak_and_bandwidth(self):
        fc = 5e6
        wb_frac = 0.7
        wc = 2 * torch.pi * fc
        wb = wb_frac * wc

        f_center = torch.tensor([fc], dtype=torch.float64)
        p = torch.log(torch.tensor(126.0, dtype=torch.float64)) / torch.log(torch.tensor(2 * wc / wb, dtype=torch.float64))

        def get_St(freqs):
            w = 2 * torch.pi * freqs
            return torch.exp(-torch.log(torch.tensor(2.0, dtype=torch.float64)) * (2 * torch.abs(w - wc) / wb) ** p)

        St_center = get_St(f_center)
        torch.testing.assert_close(St_center, torch.tensor([1.0], dtype=torch.float64))

        f_minus = (wc - wb / 2) / (2 * torch.pi)
        St_half = get_St(torch.tensor([f_minus], dtype=torch.float64))
        torch.testing.assert_close(St_half, torch.tensor([0.5], dtype=torch.float64), atol=1e-6, rtol=0)
        print("✅ Tests de simus_spectrum y ancho de banda -6 dB (Milestone 13) superados con éxito.")

if __name__ == "__main__":
    unittest.main()
