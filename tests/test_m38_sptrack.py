import unittest
import numpy as np
import torch
from torchsimus.processing.sptrack import sptrack
from torchsimus.processing.smoothn import smoothn

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


def speckle_shift(di, dj, n=4, size=220, crop=(30, 170), seed=0):
    rng = np.random.default_rng(seed)
    fq = np.fft.fftfreq(size)
    base = np.abs(np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-((fq[:, None] ** 2 + fq[None] ** 2) / 0.02))))
    shift = lambda im, a, b: np.real(np.fft.ifft2(np.fft.fft2(im) * np.exp(-2j * np.pi * (fq[:, None] * a + fq[None] * b))))
    I = np.stack([shift(base, k * di, k * dj)[crop[0]:crop[1], crop[0]:crop[1]] for k in range(n)], -1)
    return 255 * (I - I.min()) / (I.max() - I.min())


class TestM38SmoothnYFlujoOptico(unittest.TestCase):
    def test_smoothn_2d_y_3d(self):
        rng = np.random.default_rng(0)
        g = np.linspace(0, 1, 30)
        Z = np.sin(4 * g)[:, None] * np.cos(3 * g)[None]
        z, s, ok = smoothn(Z + 0.1 * rng.standard_normal(Z.shape))
        self.assertLess(np.abs(z.numpy() - Z).mean(), 0.05)
        z3, _, _ = smoothn(rng.standard_normal((10, 11, 12)), S=10)
        self.assertEqual(tuple(z3.shape), (10, 11, 12))
        print("✅ smoothn completo en 2-D y 3-D.")

    def test_sptrack_flujo_optico(self):
        I = speckle_shift(1.35, -2.6)
        Di, Dj, _, _ = sptrack(I, [[32, 32], [16, 16]], subpix="OF")
        self.assertLess(np.nanmean(np.abs(Dj - 1.35)), 0.2)
        self.assertLess(np.nanmean(np.abs(Di + 2.6)), 0.2)
        print("✅ sptrack con subpix='OF' mide el desplazamiento sub-píxel conocido.")

    def test_smoothn_against_pymust(self):
        try:
            import pymust
        except ImportError:
            self.skipTest("pymust no instalado")
        rng = np.random.default_rng(0)
        y = np.sin(np.linspace(0, 6, 200)) + 0.2 * rng.standard_normal(200); y[[40, 120]] = 4
        for robust in (False, True):
            # con S fijo: idéntico a pymust
            za = pymust.smoothn(y.copy(), S=50.0, isrobust=robust)[0]
            zb = smoothn(y.copy(), S=50.0, robust=robust)[0].numpy()
            self.assertLess(np.linalg.norm(za - zb) / np.linalg.norm(za), 1e-8)
            # con S automático (GCV): pymust 0.1.9 calcula mal la cota inferior de S (escribe
            # hMax**(2/N)**2, que en Python es hMax**((2/N)**2)); aquí se usa la fórmula de MUST,
            # así que S cambia ~1 % y el resultado difiere ~3e-4.
            za = pymust.smoothn(y.copy(), isrobust=robust)[0]
            zb = smoothn(y.copy(), robust=robust)[0].numpy()
            self.assertLess(np.linalg.norm(za - zb) / np.linalg.norm(za), 1e-3)
        print("✅ smoothn coincide con pymust (S fijo: exacto; S automático: < 1e-3).")


if __name__ == "__main__":
    unittest.main()
