import torch
from torch import nn
from elements.base import ElementShape

class CircularElement(ElementShape):
    """
    Experimental Circular Piston Element.
    Marked as experimental until validated against independent acoustic reference.
    Directivity D(theta, k) = 2 * J1(k * a * sin(theta)) / (k * a * sin(theta))
    """
    def __init__(self, radius):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(float(radius)), requires_grad=False)
        self.status = "experimental"

    def directivity(self, x_local, y_local, z_local, r, frequencies, c=1540.0):
        rho = torch.sqrt(x_local * x_local + y_local * y_local)
        sin_theta = rho / r

        a = self.radius
        freqs = frequencies.unsqueeze(0) if frequencies.ndim == 0 else frequencies
        k = (2 * torch.pi * freqs / c)[:, None, None]

        arg = k * a * sin_theta[None, ...]

        mask = (arg == 0)
        safe_arg = torch.where(mask, torch.ones_like(arg), arg)

        val = 2.0 * torch.special.bessel_j1(safe_arg) / safe_arg
        val = torch.where(mask, torch.tensor(1.0, dtype=val.dtype, device=val.device), val)

        return val.squeeze(0) if frequencies.ndim == 0 else val
