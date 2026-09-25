import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus
from chunking import forward_chunked

class TestM17Chunking(unittest.TestCase):
    def test_chunking_equivalence(self):
        class Medium:
            c = 1540.0

        fc = 5e6
        fs = 40e6
        n_fft = 1024

        geometry = LinearArray(num_elements=8, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)
        medium = Medium()

        simulator = Simus(
            transducer=transducer,
            medium=medium,
            fs=fs,
            n_fft=n_fft
        )

        scatterers = torch.tensor([
            [0.0, 0.0, 0.02],
            [0.0005, 0.0, 0.03],
            [-0.0005, 0.0, 0.035]
        ], dtype=torch.float64)
        reflectivity = torch.tensor([1.0, -0.5, 0.8], dtype=torch.float64)
        tx_delays = torch.zeros(8, 1, dtype=torch.float64)
        tx_apodization = torch.ones(8, 1, dtype=torch.float64)

        rf_full = simulator(scatterers, reflectivity, tx_delays, tx_apodization)

        # Probar con chunks absurdamente pequeños (1 y 1)
        rf_chunked = forward_chunked(
            transducer=transducer,
            scatterers=scatterers,
            reflectivity=reflectivity,
            tx_delays=tx_delays,
            tx_apodization=tx_apodization,
            fs=fs,
            n_fft=n_fft,
            c=medium.c,
            scatter_chunk=1,
            freq_chunk=1
        )

        torch.testing.assert_close(rf_chunked, rf_full, rtol=1e-5, atol=1e-6)
        print("✅ Tests de chunking (Milestone 17) superados con éxito.")

if __name__ == "__main__":
    unittest.main()
