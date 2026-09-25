"""
Milestone 39: leyes de retardo de transmisión en PyTorch (equivalentes a TXDELAY / TXDELAY3 de MUST).
Todas devuelven un tensor [E, N] (una fila por emisión) con min = 0 en cada fila, y son diferenciables.
"""
import math
import torch


def _pose_xyz(transducer):
    c = transducer.geometry.pose().centers
    return c[:, 0], c[:, 1], c[:, 2]


def _col(v, like):
    return torch.as_tensor(v, dtype=like.dtype, device=like.device).reshape(-1, 1)


def _finish(d):
    return d - d.min(dim=-1, keepdim=True).values


def _is_convex(transducer):
    return hasattr(transducer.geometry, "curvature_center")


def txdelay_focus(transducer, x0, z0, c=1540.0):
    """Foco(s) en (x0, z0). Con varios focos -> varias filas (MLT). z0 < 0 -> fuente virtual (onda divergente)."""
    xe, _, ze = _pose_xyz(transducer)
    x0, z0 = _col(x0, xe), _col(z0, xe)
    d = torch.sqrt((xe - x0) ** 2 + (ze - z0) ** 2) / c
    if _is_convex(transducer):
        R = float(transducer.geometry.radius)
        inside = torch.sqrt(x0 ** 2 + (R - z0) ** 2) < R
        d = torch.where(inside, -d, d)
    else:
        d = -d * torch.sign(z0)
    return _finish(d)


def txdelay_plane(transducer, tilt, c=1540.0):
    """Onda plana con inclinación tilt (rad). Varios ángulos -> varias filas."""
    xe, _, ze = _pose_xyz(transducer)
    tilt = _col(tilt, xe)
    if not _is_convex(transducer):
        d = xe * torch.sin(tilt) / c
    else:
        R = float(transducer.geometry.radius)
        h = -float(transducer.geometry.curvature_center()[2])
        xn, zn = R * torch.sin(tilt), R * torch.cos(tilt) - h
        d = -torch.abs(ze + xn / (zn + h) * xe - xn ** 2 / (zn + h) - zn) / torch.sqrt(1 + xn ** 2 / (zn + h) ** 2) / c
    return _finish(d)


def virtual_source(L, tilt, width):
    """Fuente virtual (x0, z0) de una onda divergente de apertura angular `width` e inclinación `tilt` (rad)."""
    tilt = (-tilt + math.pi / 2) % (2 * math.pi) - math.pi / 2
    sign = 1.0
    if abs(tilt) > math.pi / 2:
        tilt, sign = math.pi - tilt, -1.0
    z0 = sign * L / (math.tan(tilt - width / 2) - math.tan(tilt + width / 2))
    x0 = sign * z0 * math.tan(width / 2 - tilt) + L / 2
    return x0, z0


def txdelay_diverging(transducer, tilt, width, c=1540.0):
    """Onda divergente (circular) de apertura angular `width` e inclinación `tilt` (rad), arreglo lineal."""
    xe, _, _ = _pose_xyz(transducer)
    L = float(xe.max() - xe.min())
    rows = []
    for t in torch.as_tensor(tilt, dtype=torch.float64).reshape(-1).tolist():
        x0, z0 = virtual_source(L, t, float(width))
        rows.append(-torch.sqrt((xe - x0) ** 2 + z0 ** 2) / c * math.copysign(1.0, z0))
    return _finish(torch.stack(rows))


def txdelay3_focus(transducer, x0, y0, z0, c=1540.0):
    """Foco 3-D (x0, y0, z0) para un arreglo plano (matricial o CustomArray)."""
    xe, ye, _ = _pose_xyz(transducer)
    x0, y0, z0 = _col(x0, xe), _col(y0, xe), _col(z0, xe)
    d = torch.sqrt((xe - x0) ** 2 + (ye - y0) ** 2 + z0 ** 2)
    return _finish(-d / c * torch.sign(z0))


def embed_subaperture(delays, num_elements, start):
    """Pone los retardos de una subapertura en un arreglo de num_elements; el resto queda en NaN (apagado)."""
    delays = torch.as_tensor(delays).reshape(1, -1)
    out = torch.full((1, num_elements), float("nan"), dtype=delays.dtype, device=delays.device)
    out[0, start:start + delays.shape[1]] = delays[0]
    return out
