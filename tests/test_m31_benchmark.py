import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus

class TestM31Benchmark(unittest.TestCase):
    def test_scientific_benchmark(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA requerido para el benchmark científico.")

        configs = [
            {"S": 1000, "N": 64, "E": 1, "n_fft": 512},
            {"S": 10000, "N": 64, "E": 1, "n_fft": 512},
            {"S": 50000, "N": 128, "E": 16, "n_fft": 1024}
        ]

        print("\n" + "="*85)
        print(f"{'S':>8} | {'N':>4} | {'E':>4} | {'Fwd (ms)':>10} | {'Bwd (ms)':>10} | {'VRAM (MB)':>10} | {'Scat/s':>12}")
        print("-" * 85)

        for cfg in configs:
            S, N, E, n_fft = cfg["S"], cfg["N"], cfg["E"], cfg["n_fft"]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            geometry = LinearArray(num_elements=N, pitch=0.3e-3).cuda()
            element_shape = PointElement()
            transducer = Transducer(geometry, element_shape, center_frequency=5e6, bandwidth=0.7).cuda()

            class Medium:
                c = 1540.0
            medium = Medium()

            simulator = Simus(transducer=transducer, medium=medium, fs=40e6, n_fft=n_fft).cuda().double()

            scatterers = torch.randn(S, 3, dtype=torch.float64, device="cuda") * 0.005
            scatterers[:, 2] += 0.03
            reflectivity = torch.ones(S, dtype=torch.float64, device="cuda", requires_grad=True)
            tx_delays = torch.zeros(N, E, dtype=torch.float64, device="cuda")
            tx_apodization = torch.ones(N, E, dtype=torch.float64, device="cuda")

            # Sincronización y medición Forward con eventos CUDA
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            rf = simulator(scatterers, reflectivity, tx_delays, tx_apodization)
            end_event.record()
            torch.cuda.synchronize()
            fwd_ms = start_event.elapsed_time(end_event)

            # Medición Backward
            loss = rf.square().sum()
            torch.cuda.synchronize()
            start_event.record()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            bwd_ms = start_event.elapsed_time(end_event)

            vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            scat_per_sec = S / ((fwd_ms + bwd_ms) / 1000.0)

            print(f"{S:8d} | {N:4d} | {E:4d} | {fwd_ms:10.2f} | {bwd_ms:10.2f} | {vram_mb:10.2f} | {scat_per_sec:12.2e}")

        print("="*85)
        print("✅ Benchmark científico final (Milestone 31) completado con éxito.")

if __name__ == "__main__":
    unittest.main()
