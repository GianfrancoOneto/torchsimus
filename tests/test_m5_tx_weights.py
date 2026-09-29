import unittest
import torch
from torchsimus.physics.propagation import tx_weights

class TestM5TxWeights(unittest.TestCase):
    def test_tx_weights(self):
        f = torch.tensor(5e6, dtype=torch.float64)
        delays = torch.tensor([[1e-7]], dtype=torch.float64)
        apod = torch.tensor([[0.8]], dtype=torch.float64)
        A = tx_weights(f, delays, apod)
        self.assertEqual(A.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()
