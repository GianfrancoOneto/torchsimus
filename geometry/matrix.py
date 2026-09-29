import torch
from torch import nn
from geometry.base import ElementPose

class MatrixArray(nn.Module):
    def __init__(self, Nx: int, Ny: int, pitch_x: float, pitch_y: float):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.register_buffer("i_idx", torch.arange(Nx, dtype=torch.float64))
        self.register_buffer("j_idx", torch.arange(Ny, dtype=torch.float64))
        self.pitch_x = nn.Parameter(torch.tensor(float(pitch_x), dtype=torch.float64), requires_grad=False)
        self.pitch_y = nn.Parameter(torch.tensor(float(pitch_y), dtype=torch.float64), requires_grad=False)

    def pose(self) -> ElementPose:
        ii, jj = torch.meshgrid(self.i_idx, self.j_idx, indexing='ij')
        x = (ii - (self.Nx - 1) / 2.0) * self.pitch_x
        y = (jj - (self.Ny - 1) / 2.0) * self.pitch_y
        z = torch.zeros_like(x)

        num_elements = self.Nx * self.Ny
        centers = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)

        u = torch.zeros((num_elements, 3), dtype=torch.float64, device=centers.device)
        u[:, 0] = 1.0
        v = torch.zeros((num_elements, 3), dtype=torch.float64, device=centers.device)
        v[:, 1] = 1.0
        n = torch.zeros((num_elements, 3), dtype=torch.float64, device=centers.device)
        n[:, 2] = 1.0

        return ElementPose(centers, u, v, n)
