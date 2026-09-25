import unittest
import numpy as np
import torch
from processing.sptrack import sptrack

def speckle_sequence(M=128, P=5, di=1.5, dj=-0.75, seed=0):
    # speckle sintético trasladado (di filas, dj columnas) por cuadro, con desplazamiento sub-píxel exacto (FFT)
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((M, M))
    ky = np.fft.fftfreq(M)[:, None]; kx = np.fft.fftfreq(M)[None, :]
    base_f = np.fft.fft2(base) * np.exp(-(ky ** 2 + kx ** 2) / (2 * 0.08 ** 2))
    I = np.stack([np.real(np.fft.ifft2(base_f * np.exp(-2j * np.pi * (ky * di * k + kx * dj * k)))) for k in range(P)], -1)
    return (I - I.min()) / (I.max() - I.min()) * 255

class TestM38Sptrack(unittest.TestCase):
    def test_translation(self):
        I = speckle_sequence()
        Di, Dj, i_, j_ = sptrack(I, [[32, 32], [16, 16]], iminc=1)
        inner = (slice(2, -2), slice(2, -2))
        # (el ajuste parabólico tiene un sesgo sub-píxel conocido; pymust da lo mismo: ~1.38 y ~-0.83)
        self.assertLess(abs(np.nanmedian(Dj[inner]) - 1.5), 0.2)     # filas
        self.assertLess(abs(np.nanmedian(Di[inner]) + 0.75), 0.2)    # columnas
        print("✅ sptrack recupera una traslación sub-píxel conocida.")

    def test_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        I = speckle_sequence(seed=1)
        p = pymust.utils.Param(); p.winsize = np.array([[32, 32], [16, 16]]); p.iminc = 1
        p.ROI = np.ones(I.shape[:2], bool)
        Dim, Djm, _, _ = pymust.sptrack(I, p)
        Dit, Djt, _, _ = sptrack(I, [[32, 32], [16, 16]], iminc=1)
        m = np.isfinite(Dim) & np.isfinite(Dit)
        self.assertLess(np.sqrt(np.mean((Dim - Dit)[m] ** 2 + (Djm - Djt)[m] ** 2)), 0.1)
        print("✅ sptrack coincide con pymust (error RMS < 0.1 px).")

if __name__ == "__main__":
    unittest.main()
