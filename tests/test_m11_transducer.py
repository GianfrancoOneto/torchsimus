import unittest
from torchsimus.transducer import Transducer
from torchsimus.geometry.linear import LinearArray
from torchsimus.elements.point import PointElement
import torch

class TestM11Transducer(unittest.TestCase):
    def test_transducer(self):
        geom = LinearArray(10, 0.3e-3)
        shape = PointElement()
        t = Transducer(geom, shape, 5e6, 0.7)
        pose = t.pose()
        self.assertEqual(pose.centers.shape, (10, 3))

if __name__ == "__main__":
    unittest.main()
