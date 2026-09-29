import unittest
import torch
from torchsimus.physics.propagation import propagation_2d_frequency

class TestM8MultiFrequency(unittest.TestCase):
    def test_multi_frequency_shape(self):
        r = torch.randn(2, 3, dtype=torch.float64).abs()
        freqs = torch.tensor([4e6, 5e6, 6e6], dtype=torch.float64)
        G = propagation_2d_frequency(r, freqs)
        self.assertEqual(G.shape, (3, 2, 3))

if __name__ == "__main__":
    unittest.main()
