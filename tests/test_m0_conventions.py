import unittest
import torch
from torchsimus.spectra import physical_to_torch_spectrum

class TestM0Conventions(unittest.TestCase):
    def test_tensor_conventions(self):
        S, N, E, F = 10, 8, 3, 32
        positions = torch.zeros(S, 3)
        R = torch.ones(S)
        delays = torch.zeros(E, N)
        self.assertEqual(positions.shape, (S, 3))
        self.assertEqual(R.shape, (S,))
        self.assertEqual(delays.shape, (E, N))

    def test_fourier_conjugation(self):
        x = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex128)
        x_torch = physical_to_torch_spectrum(x)
        torch.testing.assert_close(x_torch, torch.tensor([1.0 - 2.0j, 3.0 + 4.0j], dtype=torch.complex128))

if __name__ == "__main__":
    unittest.main()
