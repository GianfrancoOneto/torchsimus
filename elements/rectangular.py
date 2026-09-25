import torch
from torch import nn
from elements.base import ElementShape

class RectangularElement(ElementShape):
    def __init__(self, width):
        super().__init__()
        self.width = nn.Parameter(torch.tensor(float(width)), requires_grad=False)

    def get_subelements_pose(self, pose, frequencies, c=1540.0):
        # Determinar nu dinámicamente usando la frecuencia máxima (o última frecuencia del array)
        f_max = frequencies.max().item() if frequencies.numel() > 0 else 5e6
        lambda_min = c / f_max
        w = self.width.item()
        nu = max(1, int(torch.ceil(torch.tensor(2 * (w / 2) / lambda_min)).item()))

        # Para pruebas con nu controlado explícitamente si se desea, o automático:
        # Usaremos nu automático basado en w y lambda_min
        j = torch.arange(nu, dtype=pose.centers.dtype, device=pose.centers.device)
        delta_u = -w / 2.0 + (j + 0.5) * (w / nu) # [nu]

        # Expandir centros para cada elemento N y cada subelemento nu
        # pose.centers: [N, 3], pose.u: [N, 3]
        # c_sub = c_n + delta_u * u_n -> [N, nu, 3]
        centers_sub = pose.centers[:, None, :] + delta_u[None, :, None] * pose.u[:, None, :]
        centers_sub = centers_sub.reshape(-1, 3) # [N * nu, 3]

        # Las orientaciones u, v, n se heredan para cada subelemento
        u_sub = pose.u[:, None, :].expand(-1, nu, -1).reshape(-1, 3)
        v_sub = pose.v[:, None, :].expand(-1, nu, -1).reshape(-1, 3)
        n_sub = pose.n[:, None, :].expand(-1, nu, -1).reshape(-1, 3)

        from geometry.base import ElementPose
        return ElementPose(centers_sub, u_sub, v_sub, n_sub), nu

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
