import torch
from torch import nn
from torchsimus.geometry.base import ElementPose

class LinearArray(nn.Module):
    def __init__(self, num_elements, pitch):
        super().__init__()
        self.num_elements = num_elements
        self.register_buffer(
            "index",
            torch.arange(num_elements, dtype=torch.float64)
        )
        self.pitch = nn.Parameter(
            torch.tensor(float(pitch), dtype=torch.float64),
            requires_grad=False
        )

    def pose(self):
        x = (self.index - (self.num_elements - 1) / 2) * self.pitch
        z = torch.zeros_like(x)
        centers = torch.stack([x, z, z], dim=-1)

        u = torch.zeros_like(centers)
        v = torch.zeros_like(centers)
        n = torch.zeros_like(centers)

        u[:, 0] = 1
        v[:, 1] = 1
        n[:, 2] = 1

        return ElementPose(centers, u, v, n)
