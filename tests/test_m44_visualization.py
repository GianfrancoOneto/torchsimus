import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from visualization import show_field, show_polar_field, slice_plot, single_slice_plot, doppler_cmap, movie_frames

class TestM44Visualization(unittest.TestCase):
    def test_smoke(self):
        x, z = np.meshgrid(np.linspace(-1e-2, 1e-2, 20), np.linspace(0, 2e-2, 30))
        P = torch.rand(30, 20) + 0.1
        show_field(P, x, z, "t"); plt.close("all")
        show_polar_field(P, x, z, "t"); plt.close("all")
        xi, yi, zi = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 6), np.linspace(0, 2, 7), indexing="ij")
        P3 = np.random.rand(5, 6, 7) + 0.1
        slice_plot(xi, yi, zi, P3, YZ=0, XZ=0, XY=1); plt.close("all")
        single_slice_plot(xi, yi, zi, P3, "XZ", 0.0); plt.close("all")
        movie_frames(np.random.randint(0, 255, (10, 8, 20), dtype=np.uint8), 3); plt.close("all")
        self.assertEqual(doppler_cmap().N, 256)
        print("✅ visualización: todas las funciones dibujan sin error.")

if __name__ == "__main__":
    unittest.main()
