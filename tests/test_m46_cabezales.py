import unittest
import torch
from torchsimus import cabezales
from torchsimus.geometry.base import validate_pose, ElementPose
from torchsimus.geometry.linear import LinearArray


class TestM46Cabezales(unittest.TestCase):
    def test_hay_5_por_tipo(self):
        for tipo in cabezales.TIPOS:
            self.assertEqual(len(cabezales.listar(tipo)), 5, tipo)

    def test_cargar_y_pose_valida(self):
        for nombre in cabezales.listar():
            c, n = cabezales.cargar(nombre)
            self.assertEqual(c.shape, n.shape)
            torch.testing.assert_close(torch.linalg.vector_norm(n, dim=-1), torch.ones(len(n), dtype=n.dtype))

    def test_circulares_son_un_solo_elemento(self):
        for nombre in cabezales.listar("circular"):
            c, n = cabezales.cargar(nombre)
            self.assertEqual(c.shape, (1, 3))
            self.assertGreater(cabezales.radio(nombre), 0)
        self.assertIsNone(cabezales.radio("lineal_1_128el_pitch0.3mm"))

    def test_lineal_coincide_con_LinearArray(self):
        c, _ = cabezales.cargar("lineal_1_128el_pitch0.3mm")
        torch.testing.assert_close(c, LinearArray(128, 0.3e-3).pose().centers, atol=1e-7, rtol=0)
        print(f"✅ cabezales: {len(cabezales.listar())} archivos en {cabezales.CARPETA}")


if __name__ == "__main__":
    unittest.main()
