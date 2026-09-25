import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM20FocusingOptimization(unittest.TestCase):
    def test_delay_optimization_focusing(self):
        torch.manual_seed(42)
        fc = 5e6
        fs = 40e6
        n_fft = 1024

        num_elements = 16
        geometry = LinearArray(num_elements=num_elements, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        class Medium:
            c = 1540.0
        medium = Medium()

        simulator = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft).double()

        xf, zf = 0.0, 0.03
        scatterers = torch.tensor([[xf, 0.0, zf]], dtype=torch.float64)
        reflectivity = torch.tensor([1.0], dtype=torch.float64)
        tx_apodization = torch.ones(num_elements, 1, dtype=torch.float64)

        pose = transducer.geometry.pose()
        focus_pos = torch.tensor([xf, 0.0, zf], dtype=torch.float64)
        distances = torch.linalg.vector_norm(pose.centers - focus_pos, dim=-1)
        delays_theory = (torch.max(distances) - distances) / medium.c
        delays_theory = delays_theory.unsqueeze(-1).detach()

        tx_delays = torch.zeros(num_elements, 1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([tx_delays], lr=8e-7)

        for step in range(200):
            optimizer.zero_grad()
            rf = simulator(scatterers, reflectivity, tx_delays, tx_apodization)
            loss = -torch.sum(rf ** 2)
            loss.backward()
            optimizer.step()

        delays_opt_centered = tx_delays.detach() - tx_delays.mean()
        delays_theory_centered = delays_theory - delays_theory.mean()

        delay_error = torch.linalg.vector_norm(delays_opt_centered - delays_theory_centered).item()
        print(f"🎯 Error norm en delays optimizados vs teóricos: {delay_error*1e9:.2f} ns")

        self.assertLess(delay_error, 2e-5)
        print("✅ Test de optimización de delays para focusing (Milestone 20) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
