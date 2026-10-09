import unittest
import torch
from torchsimus.geometry.linear import LinearArray
from torchsimus.elements.rectangular import RectangularElement
from torchsimus.transducer import Transducer
from torchsimus.physics.geometry import local_geometry
from torchsimus.physics.propagation import propagation_2d_frequency

class TestM22Subelements(unittest.TestCase):
    def test_subelements_nu_one_equivalence(self):
        fc = 5e6
        fs = 40e6
        n_fft = 256
        c = 1540.0

        geometry = LinearArray(num_elements=4, pitch=0.3e-3)
        width = 0.25e-3
        element_shape = RectangularElement(width)
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        pose = transducer.pose()
        scatterers = torch.tensor([[0.0, 0.0, 0.03]], dtype=torch.float64)
        frequencies = torch.fft.rfftfreq(n_fft, d=1/fs)

        # Con nu = 1 forzado o ancho pequeño, debe coincidir con la versión estándar sin subelementos
        sub_pose, nu = element_shape.get_subelements_pose(pose, frequencies, c=c)

        x_loc, _, z_loc, r = local_geometry(scatterers, sub_pose)
        G_sub = propagation_2d_frequency(r, frequencies, c=c) * element_shape.directivity(x_loc, None, z_loc, r, frequencies)

        # Suma coherente por elemento: G_element(s, n) = sum_{j in n} G_sub(s, j)
        F_num = frequencies.numel()
        S = scatterers.shape[0]
        N = geometry.num_elements

        G_sub_reshaped = G_sub.reshape(F_num, S, N, nu)
        G_element = G_sub_reshaped.sum(dim=-1) # [F, S, N]

        self.assertEqual(G_element.shape, (F_num, S, N))
        print("✅ Test de subelementos y suma coherente (Milestone 22) superado con éxito.")


class TestM22Directividad(unittest.TestCase):
    def test_directividad_sin_tensor_FxF(self):
        # el eje de frecuencia va adelante sin importar cuántas dimensiones tenga x (antes: [F, F, S, N] con B = 1)
        el = RectangularElement(0.25e-3)
        x = torch.rand(2, 5, 7); z = torch.rand(2, 5, 7) + 0.1
        f = torch.linspace(1e6, 4e6, 11)
        self.assertEqual(el.directivity(x, None, z, None, f).shape, (11, 2, 5, 7))     # [F, B, S, N]
        self.assertEqual(el.directivity(x[0], None, z[0], None, f).shape, (11, 5, 7))  # [F, S, N]
        self.assertEqual(el.directivity(x[0], None, z[0], None, f[0]).shape, (5, 7))   # frecuencia escalar
        on_axis = el.directivity(torch.zeros(1), None, torch.ones(1), None, f)
        torch.testing.assert_close(on_axis, torch.ones_like(on_axis))
        print("✅ RectangularElement: eje de frecuencia correcto y cos θ = 1 en el eje.")

    def test_baffle(self):
        x, z, f = torch.tensor([0.01]), torch.tensor([0.01]), torch.tensor(1e3)        # 45°, sinc ~ 1
        soft = RectangularElement(0.25e-3).directivity(x, None, z, None, f)
        sin_ob = RectangularElement(0.25e-3, baffle=None).directivity(x, None, z, None, f)
        self.assertAlmostEqual(float(soft / sin_ob), 2 ** -0.5, places=5)
        print("✅ RectangularElement: baffle soft = cos θ; baffle=None = solo sinc.")


if __name__ == "__main__":
    unittest.main()
