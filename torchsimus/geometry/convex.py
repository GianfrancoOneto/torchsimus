import torch
from torch import nn
from torchsimus.geometry.base import ElementPose

class ConvexArray(nn.Module):
    def __init__(self, num_elements, pitch, radius):
        super().__init__()
        self.N = num_elements
        self.register_buffer("index", torch.arange(num_elements, dtype=torch.float64))
        self.pitch = nn.Parameter(torch.tensor(float(pitch)), requires_grad=False)
        self.radius = nn.Parameter(torch.tensor(float(radius)), requires_grad=False)

    def pose(self):
        q = self.index - (self.N - 1) / 2
        phi = q * self.pitch / self.radius
        x = self.radius * torch.sin(phi)
        z = self.radius * (1 - torch.cos(phi))
        y = torch.zeros_like(x)

        centers = torch.stack([x, y, z], dim=-1)
        u = torch.stack([torch.cos(phi), y, torch.sin(phi)], dim=-1)
        v = torch.zeros_like(u)
        v[:, 1] = 1
        n = torch.stack([-torch.sin(phi), y, torch.cos(phi)], dim=-1)

        return ElementPose(centers, u, v, n)
