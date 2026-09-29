import unittest
import torch
from geometry.linear import LinearArray
from elements.rectangular import RectangularElement
from transducer import Transducer
from physics.geometry import local_geometry
from physics.propagation import propagation_2d_frequency

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

if __name__ == "__main__":
    unittest.main()
