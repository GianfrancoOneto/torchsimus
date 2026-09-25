import torch
from torch import nn
from geometry.base import ElementPose

class CustomArray(nn.Module):
    def __init__(self, centers: torch.Tensor, u: torch.Tensor, v: torch.Tensor, n: torch.Tensor):
        super().__init__()
        self.register_buffer("centers", centers.clone())
        self.register_buffer("u", u.clone())
        self.register_buffer("v", v.clone())
        self.register_buffer("n", n.clone())

    def pose(self) -> ElementPose:
        return ElementPose(self.centers, self.u, self.v, self.n)
