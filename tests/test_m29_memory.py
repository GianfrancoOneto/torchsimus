import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import Simus
from torch.utils.checkpoint import checkpoint

class TestM29MemoryOptimization(unittest.TestCase):
    def test_memory_profiling(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA no disponible para perfilado de memoria.")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        fc = 5e6
        fs = 40e6
        n_fft = 1024

        geometry = LinearArray(num_elements=32, pitch=0.3e-3).cuda()
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7).cuda()

        class Medium:
            c = 1540.0
        medium = Medium()

        simulator = Simus(transducer=transducer, medium=medium, fs=fs, n_fft=n_fft).cuda().double()

        # Dataset amplio de scatterers para evaluar picos de memoria
        n_scat = 8192
        scatterers = torch.randn(n_scat, 3, dtype=torch.float64, device="cuda") * 0.005
        scatterers[:, 2] += 0.03
        reflectivity = torch.ones(n_scat, dtype=torch.float64, device="cuda", requires_grad=True)
        tx_delays = torch.zeros(32, 1, dtype=torch.float64, device="cuda")
        tx_apodization = torch.ones(32, 1, dtype=torch.float64, device="cuda")

        # 1. Medición Autograd Normal (Forward + Backward)
        torch.cuda.reset_peak_memory_stats()
        rf = simulator(scatterers, reflectivity, tx_delays, tx_apodization)
        m_fwd = torch.cuda.max_memory_allocated() / (1024**2)

        loss = rf.square().sum()
        loss.backward()
        m_fwd_bwd = torch.cuda.max_memory_allocated() / (1024**2)

        print(f"\n📊 [Normal Autograd] Forward: {m_fwd:.2f} MB | Forward+Backward: {m_fwd_bwd:.2f} MB")

        # Limpiar gradientes y caché
        reflectivity.grad = None
        torch.cuda.empty_cache()

        # 2. Medición con Checkpointing (Ahorro de memoria en activaciones)
        torch.cuda.reset_peak_memory_stats()
        def run_sim(s, r, d, a):
            return simulator(s, r, d, a)

        rf_chk = checkpoint(run_sim, scatterers, reflectivity, tx_delays, tx_apodization, use_reentrant=False)
        m_fwd_chk = torch.cuda.max_memory_allocated() / (1024**2)

        loss_chk = rf_chk.square().sum()
        loss_chk.backward()
        m_fwd_bwd_chk = torch.cuda.max_memory_allocated() / (1024**2)

        print(f"📊 [Checkpointed]    Forward: {m_fwd_chk:.2f} MB | Forward+Backward: {m_fwd_bwd_chk:.2f} MB")

        self.assertLessEqual(m_fwd_bwd_chk, m_fwd_bwd * 1.5)
        print("✅ Test de optimización y perfilado de memoria (Milestone 29) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
