import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM16GPU(unittest.TestCase):
    def _run_simulation(self, dtype, device):
        fc = 5e6
        fs = 40e6
        n_fft = 2048

        geometry = LinearArray(num_elements=8, pitch=0.3e-3).to(device)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7).to(device)

        class Medium:
            c = 1540.0
        medium = Medium()

        simulator = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft).to(device)

        if dtype == torch.float32:
            simulator = simulator.float()
        else:
            simulator = simulator.double()

        scatterers = torch.tensor([[0.0, 0.0, 0.03], [0.001, 0.0, 0.025]], dtype=dtype, device=device)
        reflectivity = torch.tensor([1.0, 0.5], dtype=dtype, device=device)
        tx_delays = torch.zeros(8, 1, dtype=dtype, device=device)
        tx_apodization = torch.ones(8, 1, dtype=dtype, device=device)

        rf = simulator(scatterers, reflectivity, tx_delays, tx_apodization)
        return rf

    def test_cpu_gpu_equivalence_double(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA no está disponible en este entorno.")

        rf_cpu = self._run_simulation(torch.float64, torch.device("cpu"))
        rf_gpu = self._run_simulation(torch.float64, torch.device("cuda"))

        torch.testing.assert_close(
            rf_gpu.cpu(),
            rf_cpu,
            rtol=1e-4,
            atol=1e-5
        )
        print("✅ Test CPU vs GPU (float64/complex128) superado con éxito.")

    def test_cpu_gpu_equivalence_single(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA no está disponible en este entorno.")

        rf_cpu = self._run_simulation(torch.float32, torch.device("cpu"))
        rf_gpu = self._run_simulation(torch.float32, torch.device("cuda"))

        torch.testing.assert_close(
            rf_gpu.cpu(),
            rf_cpu,
            rtol=1e-3,
            atol=1e-4
        )
        print("✅ Test CPU vs GPU (float32/complex64) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
