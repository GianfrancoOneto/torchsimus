import math
import torch
from torch import nn
try:
    from geometry.base import ElementPose
except ImportError:                      # tests antiguos que usan PYTHONPATH=/content
    from torchsimus.geometry.base import ElementPose

class ConvexArray(nn.Module):
    """
    Arreglo convexo en la convención de MUST/pymust (Milestone 33):
      * paso angular entre elementos = 2*asin(pitch / (2R))  (el pitch es la cuerda entre centros)
      * z = R*(cos(phi) - cos(phi_max)): los BORDES están en z = 0 y el CENTRO sobresale (z > 0)
      * n apunta hacia fuera (hacia el paciente), u es tangente, v = y
    El centro de curvatura queda en (0, 0, -R*cos(phi_max)).
    """
    def __init__(self, num_elements, pitch, radius):
        super().__init__()
        self.N = num_elements
        self.num_elements = num_elements
        self.register_buffer("index", torch.arange(num_elements, dtype=torch.float64))
        self.pitch = nn.Parameter(torch.tensor(float(pitch), dtype=torch.float64), requires_grad=False)
        self.radius = nn.Parameter(torch.tensor(float(radius), dtype=torch.float64), requires_grad=False)

    def angles(self):
        q = self.index - (self.N - 1) / 2
        dphi = 2 * torch.asin(self.pitch / (2 * self.radius))
        return q * dphi

    def curvature_center(self):
        phi_max = self.angles().abs().max()
        return torch.stack([torch.zeros_like(phi_max), torch.zeros_like(phi_max), -self.radius * torch.cos(phi_max)])

    def pose(self):
        phi = self.angles()
        phi_max = phi.abs().max()
        x = self.radius * torch.sin(phi)
        z = self.radius * (torch.cos(phi) - torch.cos(phi_max))
        y = torch.zeros_like(x)
        centers = torch.stack([x, y, z], dim=-1)
        u = torch.stack([torch.cos(phi), y, -torch.sin(phi)], dim=-1)
        v = torch.zeros_like(u); v[:, 1] = 1
        n = torch.stack([torch.sin(phi), y, torch.cos(phi)], dim=-1)
        return ElementPose(centers, u, v, n)
