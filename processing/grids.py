"""Milestone 41: grilla polar (sector) en PyTorch, equivalente a IMPOLGRID de MUST."""
import math
import torch


def impolgrid(transducer, siz, zmax, width=None):
    """
    Grilla polar [siz[0] (radial) x siz[1] (angular)] hasta la profundidad zmax.
    Arreglo lineal: sector de apertura `width` (rad). Arreglo convexo: el sector del propio arreglo.
    Devuelve (x, z) en m, tensores float64.
    """
    if isinstance(siz, int):
        siz = (siz, siz)
    geo = transducer.geometry
    N = int(geo.num_elements)
    p = float(geo.pitch)
    if not hasattr(geo, "curvature_center"):
        L = (N - 1) * p
        z0 = 0.0
        R = L / 2
        th0, th1 = width / 2, -width / 2
    else:
        R = float(geo.radius)
        L = 2 * R * math.sin(math.asin(p / 2 / R) * (N - 1))
        d = math.sqrt(R ** 2 - L ** 2 / 4)
        z0 = -d
        th0, th1 = math.atan2(L / 2, d), math.atan2(-L / 2, d)
    th = torch.linspace(th0, th1, siz[1], dtype=torch.float64) + math.pi / 2
    r = torch.linspace(R + p, -z0 + zmax, siz[0], dtype=torch.float64)
    r, th = torch.meshgrid(r, th, indexing="ij")
    return r * torch.cos(th), r * torch.sin(th) + z0
