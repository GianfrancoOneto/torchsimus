import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM15SimusModule(unittest.TestCase):
    def test_simus_equivalence(self):
        class Medium:
            c = 1540.0

        fc = 5e6
        fs = 40e6
        n_fft = 4096

        geometry = LinearArray(num_elements=16, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)
        medium = Medium()

        simulator = Simus(
            transducer=transducer,
            medium=medium,
            fs=fs,
            n_fft=n_fft
        )

        scatterers = torch.tensor([[0.0, 0.0, 0.03], [0.001, 0.0, 0.04]], dtype=torch.float64)
        reflectivity = torch.tensor([1.0, 0.5], dtype=torch.float64)
        tx_delays = torch.zeros(16, 1, dtype=torch.float64)
        tx_apodization = torch.ones(16, 1, dtype=torch.float64)

        rf = simulator(
            scatterers,
            reflectivity,
            tx_delays,
            tx_apodization
        )

        self.assertEqual(rf.shape, (n_fft, 16, 1))
        print("✅ Tests del módulo unificado Simus (Milestone 15) superados con éxito.")

if __name__ == "__main__":
    unittest.main()
