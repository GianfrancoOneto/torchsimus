import torch
from torch import nn
from geometry.base import ElementPose

class RingArray(nn.Module):
    def __init__(self, num_elements: int, radius: float):
        super().__init__()
        self.num_elements = num_elements
        self.register_buffer("index", torch.arange(num_elements, dtype=torch.float64))
        self.radius = nn.Parameter(torch.tensor(float(radius), dtype=torch.float64), requires_grad=False)

    def pose(self) -> ElementPose:
        phi = 2 * torch.pi * self.index / self.num_elements
        x = self.radius * torch.cos(phi)
        y = self.radius * torch.sin(phi)
        z = torch.zeros_like(x)
        centers = torch.stack([x, y, z], dim=-1)

        u = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], dim=-1)
        v = torch.stack([-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)], dim=-1)
        n = torch.zeros_like(centers)
        n[:, 2] = 1.0

        return ElementPose(centers, u, v, n)
