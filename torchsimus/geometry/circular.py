import math
import torch
from torch import nn
from torchsimus.geometry.base import ElementPose


class CircularSingleElement(nn.Module):
    """
    Milestone 45: transductor de UN SOLO elemento circular (pistón plano de radio `radius`).

    pose()            -> 1 elemento: centro en `center`, mirando hacia +z (u = x̂, v = ŷ, n = ẑ).
    patches(paso)     -> el disco dividido en cuadraditos de lado `paso` (los que tienen el centro
                         dentro del círculo). Todos se excitan con la MISMA señal, así que juntos
                         se comportan como un solo elemento circular. Es lo que usa pfield3 para
                         calcular el campo, porque pfield3 trabaja con elementos rectangulares.

        radius : radio del elemento (m)
        center : posición del centro (m), por defecto el origen
    """
    def __init__(self, radius: float, center=(0.0, 0.0, 0.0)):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(float(radius), dtype=torch.float64), requires_grad=False)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float64).reshape(1, 3))
        self.num_elements = 1

    def pose(self) -> ElementPose:
        c = self.center.clone()
        u = torch.tensor([[1.0, 0.0, 0.0]], dtype=c.dtype, device=c.device)
        v = torch.tensor([[0.0, 1.0, 0.0]], dtype=c.dtype, device=c.device)
        n = torch.tensor([[0.0, 0.0, 1.0]], dtype=c.dtype, device=c.device)
        return ElementPose(c, u, v, n)

    def patches(self, paso: float) -> ElementPose:
        """Discretización del disco en cuadrados de lado `paso` (m), todos en el plano del elemento."""
        R = float(self.radius)
        m = int(math.ceil(R / paso))
        g = (torch.arange(-m, m, dtype=torch.float64, device=self.center.device) + 0.5) * paso   # centros de la grilla
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        dentro = xx ** 2 + yy ** 2 < R ** 2
        x, y = xx[dentro], yy[dentro]
        centers = torch.stack([x, y, torch.zeros_like(x)], -1) + self.center
        u = torch.zeros_like(centers); u[:, 0] = 1.0
        v = torch.zeros_like(centers); v[:, 1] = 1.0
        n = torch.zeros_like(centers); n[:, 2] = 1.0
        return ElementPose(centers, u, v, n)
