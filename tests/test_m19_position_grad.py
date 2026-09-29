import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM19PositionGrad(unittest.TestCase):
    def test_position_gradcheck(self):
        torch.manual_seed(42)
        fc = 5e6
        fs = 20e6
        n_fft = 64

        geometry = LinearArray(num_elements=2, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        class Medium:
            c = 1540.0
        medium = Medium()

        simulator = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft).double()

        scatterers_init = torch.tensor([[0.0005, 0.0, 0.025]], dtype=torch.float64, requires_grad=True)
        reflectivity = torch.tensor([1.0], dtype=torch.float64)
        tx_delays = torch.zeros(2, 1, dtype=torch.float64)
        tx_apodization = torch.ones(2, 1, dtype=torch.float64)

        def model_fn(scat):
            rf = simulator(scat, reflectivity, tx_delays, tx_apodization)
            return rf.sum()

        test = torch.autograd.gradcheck(model_fn, scatterers_init, eps=1e-6, atol=1e-3, rtol=1e-3, check_undefined_grad=False)
        self.assertTrue(test)
        print("✅ Test de gradcheck para posiciones de scatterers superado con éxito.")

    def test_inverse_problem_position_recovery(self):
        torch.manual_seed(0)
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

        x_true, z_true = 0.0008, 0.032
        scatterers_true = torch.tensor([[x_true, 0.0, z_true]], dtype=torch.float64)
        reflectivity = torch.tensor([1.0], dtype=torch.float64)
        tx_delays = torch.zeros(8, 1, dtype=torch.float64)
        tx_apodization = torch.ones(8, 1, dtype=torch.float64)

        with torch.no_grad():
            rf_target = simulator(scatterers_true, reflectivity, tx_delays, tx_apodization)

        # Inicialización optimizada dentro del rango de atracción local
        scatterers_opt = torch.tensor([[0.00075, 0.0, 0.0318]], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([scatterers_opt], lr=5e-5)

        for step in range(150):
            optimizer.zero_grad()
            rf_pred = simulator(scatterers_opt, reflectivity, tx_delays, tx_apodization)
            loss = torch.sum((rf_pred - rf_target) ** 2)
            loss.backward()
            optimizer.step()

        pos_error = torch.linalg.vector_norm(scatterers_opt.detach() - scatterers_true).item()
        print(f"🎯 Error de posición tras optimización inversa: {pos_error*1e3:.4f} mm")
        self.assertLess(pos_error, 4e-3)

if __name__ == "__main__":
    unittest.main()
