import unittest
import torch
import time
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM28Modes(unittest.TestCase):
    def test_direct_vs_scattering_equivalence(self):
        fc = 5e6
        fs = 40e6
        n_fft = 512

        geometry = LinearArray(num_elements=16, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        class Medium:
            c = 1540.0
        medium = Medium()

        sim_direct = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft, mode="direct").double()
        sim_scattering = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft, mode="scattering").double()

        scatterers = torch.tensor([[0.0, 0.0, 0.03], [0.001, 0.0, 0.025]], dtype=torch.float64)
        reflectivity = torch.tensor([1.0, 0.5], dtype=torch.float64)
        tx_delays = torch.randn(16, 5, dtype=torch.float64) * 1e-7
        tx_apodization = torch.ones(16, 5, dtype=torch.float64)

        rf_direct = sim_direct(scatterers, reflectivity, tx_delays, tx_apodization)
        rf_scattering = sim_scattering(scatterers, reflectivity, tx_delays, tx_apodization)

        torch.testing.assert_close(rf_scattering, rf_direct, rtol=1e-5, atol=1e-6)
        print("✅ Test de equivalencia Directo vs Scattering-Matrix (Milestone 28) superado con éxito.")

    def test_crossover_benchmark(self):
        fc = 5e6
        fs = 40e6
        n_fft = 256

        geometry = LinearArray(num_elements=16, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        class Medium:
            c = 1540.0
        medium = Medium()

        scatterers = torch.randn(30, 3, dtype=torch.float64) * 0.001
        scatterers[:, 2] += 0.03
        reflectivity = torch.randn(30, dtype=torch.float64)

        E_list = [1, 5, 15, 64, 128]
        print("\n--- Benchmark Crossover Directo vs Scattering ---")
        for E in E_list:
            tx_delays = torch.randn(16, E, dtype=torch.float64) * 1e-7
            tx_apodization = torch.ones(16, E, dtype=torch.float64)

            sim_d = Simus(transducer, medium, fs, n_fft, mode="direct").double()
            sim_s = Simus(transducer, medium, fs, n_fft, mode="scattering").double()

            # Warmup
            _ = sim_d(scatterers, reflectivity, tx_delays, tx_apodization)
            _ = sim_s(scatterers, reflectivity, tx_delays, tx_apodization)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            for _ in range(5):
                _ = sim_d(scatterers, reflectivity, tx_delays, tx_apodization)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_direct = (time.time() - t0) / 5.0

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            for _ in range(5):
                _ = sim_s(scatterers, reflectivity, tx_delays, tx_apodization)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_scattering = (time.time() - t0) / 5.0

            print(f"E = {E:3d} | Direct: {t_direct*1000:6.2f} ms | Scattering: {t_scattering*1000:6.2f} ms")

if __name__ == "__main__":
    unittest.main()
