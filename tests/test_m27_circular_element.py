import unittest
import torch
from elements.circular import CircularElement

class TestM27CircularElement(unittest.TestCase):
    def test_circular_element_status(self):
        elem = CircularElement(1e-3)
        self.assertEqual(elem.status, "experimental")

    def test_circular_element_on_axis(self):
        a = 1e-3
        elem = CircularElement(a)

        x = torch.tensor([[0.0]], dtype=torch.float64)
        y = torch.tensor([[0.0]], dtype=torch.float64)
        z = torch.tensor([[0.02]], dtype=torch.float64)
        r = torch.tensor([[0.02]], dtype=torch.float64)
        f = torch.tensor([5e6], dtype=torch.float64)

        D = elem.directivity(x, y, z, r, f)
        torch.testing.assert_close(D, torch.tensor([[[1.0]]], dtype=torch.float64))
        print("✅ Test circular element on-axis (D=1 at theta=0) superado con éxito.")

    def test_circular_element_first_zero(self):
        a = 1e-3
        elem = CircularElement(a)
        f = torch.tensor([5e6], dtype=torch.float64)
        c = 1540.0
        k = 2 * torch.pi * f / c

        target_arg = 3.8317
        sin_theta = target_arg / (k.item() * a)

        r = torch.tensor([[0.02]], dtype=torch.float64)
        rho = r * sin_theta
        x = rho
        y = torch.zeros_like(x)
        z = torch.sqrt(r**2 - rho**2)

        D = elem.directivity(x, y, z, r, f, c=c)
        self.assertLess(torch.abs(D).item(), 1e-4)
        print("✅ Test circular element first zero (ka*sin(theta) ≈ 3.8317) superado con éxito.")

if __name__ == "__main__":
    unittest.main()
