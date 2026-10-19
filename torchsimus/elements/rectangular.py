import torch
from torch import nn
from torchsimus.elements.base import ElementShape

class RectangularElement(ElementShape):
    def __init__(self, width):
        super().__init__()
        self.width = nn.Parameter(torch.tensor(float(width)), requires_grad=False)

    def directivity_2d(self, x, z, frequencies, c=1540.0):
        r_xz = torch.sqrt(x*x + z*z)
        sin_theta = x / r_xz
        b = self.width / 2
        freqs = frequencies.unsqueeze(0) if frequencies.ndim == 0 else frequencies
        k = (2 * torch.pi * freqs / c)[:, None, None]
        q = k * b * sin_theta[None]
        res = torch.sinc(q / torch.pi)
        return res.squeeze(0) if frequencies.ndim == 0 else res

    def directivity(self, x_local, y_local, z_local, r, frequencies):
        return self.directivity_2d(x_local, z_local, frequencies)
