import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM18ReflectivityGrad(unittest.TestCase):
    def test_reflectivity_gradcheck(self):
        torch.manual_seed(42)
        fc = 5e6
        fs = 20e6
        n_fft = 64  # Tamaño reducido para eficiencia en gradcheck

        geometry = LinearArray(num_elements=2, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        class Medium:
            c = 1540.0
        medium = Medium()

        simulator = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft).double()

        scatterers = torch.tensor([[0.0, 0.0, 0.02], [0.0001, 0.0, 0.025]], dtype=torch.float64, requires_grad=False)
        tx_delays = torch.zeros(2, 1, dtype=torch.float64, requires_grad=False)
        tx_apodization = torch.ones(2, 1, dtype=torch.float64, requires_grad=False)

        def model_fn(R):
            rf = simulator(scatterers, R, tx_delays, tx_apodization)
            return rf.sum()

        R_init = torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True)

        # Verificación rigurosa de gradientes analíticos vs diferencias finitas
        test = torch.autograd.gradcheck(model_fn, R_init, eps=1e-6, atol=1e-4, rtol=1e-4, check_undefined_grad=False)
        self.assertTrue(test)
        print("✅ Test de autograd gradcheck para reflectividad (Milestone 18) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
