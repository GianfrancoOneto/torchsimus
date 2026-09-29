import unittest
import torch
from physics.propagation import propagation_2d_frequency, propagation_with_attenuation

class TestM21Attenuation(unittest.TestCase):
    def test_attenuation_alpha_zero(self):
        r = torch.tensor([[0.01, 0.03]], dtype=torch.float64)
        frequencies = torch.tensor([1e6, 5e6], dtype=torch.float64)
        c = 1540.0

        G_normal = propagation_2d_frequency(r, frequencies, c=c)
        G_atten_zero = propagation_with_attenuation(r, frequencies, c=c, alpha_db=0.0)

        torch.testing.assert_close(G_atten_zero, G_normal)

    def test_attenuation_magnitude_reduction(self):
        r = torch.tensor([[0.03]], dtype=torch.float64)
        frequencies = torch.tensor([5e6], dtype=torch.float64)
        c = 1540.0

        G_normal = propagation_2d_frequency(r, frequencies, c=c)
        G_atten = propagation_with_attenuation(r, frequencies, c=c, alpha_db=1.0)

        self.assertTrue(torch.all(torch.abs(G_atten) < torch.abs(G_normal)))

    def test_attenuation_depth_decay(self):
        r1 = torch.tensor([[0.01]], dtype=torch.float64)
        r2 = torch.tensor([[0.05]], dtype=torch.float64)
        frequencies = torch.tensor([5e6], dtype=torch.float64)
        c = 1540.0
        alpha_db = 2.0

        G1 = propagation_with_attenuation(r1, frequencies, c=c, alpha_db=alpha_db)
        G2 = propagation_with_attenuation(r2, frequencies, c=c, alpha_db=alpha_db)

        self.assertTrue(torch.abs(G2).item() < torch.abs(G1).item())
        print("✅ Tests de atenuación (Milestone 21) superados con éxito.")

if __name__ == "__main__":
    unittest.main()
