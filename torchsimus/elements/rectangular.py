import torch
from torch import nn
from torchsimus.elements.base import ElementShape

_EPS32 = float(torch.finfo(torch.float32).eps)


class RectangularElement(ElementShape):
    """
    Elemento rectangular de ancho `width` (m).

    baffle (Milestone 22): factor de oblicuidad, como PARAM.baffle de MUST.
        "soft"  -> cos(theta) (por defecto, igual que MUST/pymust)
        "rigid" -> 1
        número  -> cos / (cos + baffle)  (impedancia relativa)
        None    -> sin factor de oblicuidad (comportamiento original: solo sinc)
    """
    def __init__(self, width, baffle="soft"):
        super().__init__()
        self.width = nn.Parameter(torch.tensor(float(width)), requires_grad=False)
        self.baffle = baffle

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

        from torchsimus.geometry.base import ElementPose
        return ElementPose(centers_sub, u_sub, v_sub, n_sub), nu

    def obliquity(self, cos_t):
        """Factor de oblicuidad del baffle (ObliFac de MUST)."""
        if self.baffle is None:
            return torch.ones_like(cos_t)
        if self.baffle == "rigid":
            ob = torch.ones_like(cos_t)
        elif self.baffle == "soft":
            ob = cos_t
        else:
            ob = cos_t / (cos_t + float(self.baffle))
        return torch.where(cos_t <= 0, torch.full_like(ob, _EPS32), ob)

    def directivity_2d(self, x, z, frequencies, c=1540.0):
        """
        sinc(k b sin(theta)) * oblicuidad, en el plano xz local.
        x, z: cualquier forma [...]. frequencies: escalar -> [...]; vector [F] -> [F, ...].
        Milestone 22: el eje de frecuencia se agrega adelante sin importar cuántas dimensiones tenga x
        (antes quedaba mal ubicado con x de 3 dimensiones [B, S, N] y se creaba un tensor [F, F, S, N]).
        """
        r_xz = torch.sqrt(x * x + z * z)
        sin_t, cos_t = x / r_xz, z / r_xz
        scalar = frequencies.ndim == 0
        freqs = frequencies.reshape(-1)
        k = (2 * torch.pi * freqs / c).reshape(-1, *([1] * x.ndim))
        res = torch.sinc(k * (self.width / 2) * sin_t[None] / torch.pi) * self.obliquity(cos_t)[None]
        return res[0] if scalar else res

    def directivity(self, x_local, y_local, z_local, r, frequencies):
        return self.directivity_2d(x_local, z_local, frequencies)
