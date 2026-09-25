import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM30Batch(unittest.TestCase):
    def test_batch_equivalence(self):
        fc = 5e6
        fs = 40e6
        n_fft = 512

        geometry = LinearArray(num_elements=8, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        class Medium:
            c = 1540.0
        medium = Medium()

        simulator = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft).double()

        scene1_scatterers = torch.tensor([[0.0, 0.0, 0.02], [0.001, 0.0, 0.025]], dtype=torch.float64)
        scene1_ref = torch.tensor([1.0, 0.5], dtype=torch.float64)

        scene2_scatterers = torch.tensor([[-0.001, 0.0, 0.03], [0.0, 0.0, 0.035]], dtype=torch.float64)
        scene2_ref = torch.tensor([0.8, -0.6], dtype=torch.float64)

        tx_delays = torch.zeros(8, 1, dtype=torch.float64)
        tx_apodization = torch.ones(8, 1, dtype=torch.float64)

        rf1 = simulator(scene1_scatterers, scene1_ref, tx_delays, tx_apodization)
        rf2 = simulator(scene2_scatterers, scene2_ref, tx_delays, tx_apodization)

        batch_scatterers = torch.stack([scene1_scatterers, scene2_scatterers])
        batch_ref = torch.stack([scene1_ref, scene2_ref])

        rf_batch = simulator(batch_scatterers, batch_ref, tx_delays, tx_apodization)

        torch.testing.assert_close(rf_batch[0], rf1, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(rf_batch[1], rf2, rtol=1e-5, atol=1e-6)
        print("✅ Test de batching para deep learning (Milestone 30) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
