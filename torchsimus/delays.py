"""
Milestone 39: leyes de retardo de transmisión en PyTorch (equivalentes a TXDELAY / TXDELAY3 de MUST).
Milestones 39-40 (ampliados): txdelay3_plane, txdelay3_diverging, txdelay3_focus_line; txdelay_diverging con un ancho por ángulo.
Todas devuelven un tensor [E, N] (una fila por emisión) con min = 0 en cada fila, y son diferenciables.
"""
import math
import torch
from torchsimus.processing.optim import fminbound


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
    """
    Onda divergente (circular) de apertura angular `width` e inclinación `tilt` (rad), arreglo lineal.
    tilt y width pueden ser vectores: una fila por par (tilt, width), como TXDELAY(PARAM, TILT, WIDTH)
    de MUST. Un width escalar se usa para todos los tilts (Milestones 39-40).
    """
    xe, _, _ = _pose_xyz(transducer)
    L = float(xe.max() - xe.min())
    tilts = torch.as_tensor(tilt, dtype=torch.float64).reshape(-1)
    widths = torch.as_tensor(width, dtype=torch.float64).reshape(-1)
    if widths.numel() == 1:
        widths = widths.expand_as(tilts)
    if tilts.numel() == 1 and widths.numel() > 1:
        tilts = tilts.expand_as(widths)
    if tilts.numel() != widths.numel():
        raise ValueError("tilt y width deben tener el mismo largo (o uno de ellos ser escalar)")
    rows = []
    for t, w in zip(tilts.tolist(), widths.tolist()):
        x0, z0 = virtual_source(L, t, w)
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


# ---------------------------------------------------------------------------
# Milestones 39-40: las otras sintaxis de TXDELAY3 (onda plana, divergente y foco en una línea)
# ---------------------------------------------------------------------------
def txdelay3_plane(transducer, tiltx, tilty, c=1540.0):
    xe, ye, _ = _pose_xyz(transducer)
    tx = torch.as_tensor(tiltx, dtype=xe.dtype); ty = torch.as_tensor(tilty, dtype=xe.dtype)
    assert float(tx.abs()) < math.pi / 2 and float(ty.abs()) < math.pi / 2, "|tilt| debe ser < pi/2"
    return _finish(((xe * torch.sin(ty) - ye * torch.sin(tx)) / c)[None])


def _solid_angle(r, l, b, az, el):
    L1 = l / 2 + r * math.cos(el) * math.cos(az)
    L2 = -l / 2 + r * math.cos(el) * math.cos(az)
    B1 = b / 2 + r * math.cos(el) * math.sin(az)
    B2 = b / 2 - r * math.cos(el) * math.sin(az)
    H = r * math.sin(el)
    w = lambda l_, b_: math.asin(l_ * b_ / math.hypot(l_, H) / math.hypot(b_, H))
    return w(L1, B1) + w(L1, B2) - w(L2, B1) - w(L2, B2)


def txdelay3_diverging(transducer, tiltx, tilty, omega, c=1540.0, width=None, height=None):
    """Onda divergente que cubre un ángulo sólido omega (sr), inclinada (tiltx, tilty). Arreglo en grilla (plaid)."""
    xe, ye, _ = _pose_xyz(transducer)
    width = float(width if width is not None else transducer.element_shape.width)
    height = float(height if height is not None else getattr(transducer.element_shape, "height", getattr(transducer, "height")))
    assert 0 <= omega <= 2 * math.pi, "0 <= omega <= 2 pi"
    x = -math.sin(tilty) * math.cos(tiltx); y = math.sin(tiltx); z = -math.cos(tilty) * math.cos(tiltx)
    az, el = math.atan2(y, x), math.atan2(z, math.hypot(x, y))
    l = float(xe.max() - xe.min()) + width
    b = float(ye.max() - ye.min()) + height
    r = fminbound(lambda r_: abs(_solid_angle(r_, l, b, az, el) - omega), 0.0, 2 * math.pi, xtol=1e-6)
    x0 = r * math.cos(el) * math.cos(az); y0 = r * math.cos(el) * math.sin(az); z0 = r * math.sin(el)
    return txdelay3_focus(transducer, x0, y0, z0, c)


def txdelay3_focus_line(transducer, p1, p2, c=1540.0):
    """Enfoque en la línea que pasa por p1 = (x1, y1, z1) y p2 = (x2, y2, z2)."""
    xe, ye, _ = _pose_xyz(transducer)
    X = torch.stack([xe, ye, torch.zeros_like(xe)], 0)                      # [3, N]
    a = torch.as_tensor(p1, dtype=xe.dtype).reshape(3, 1); b = torch.as_tensor(p2, dtype=xe.dtype).reshape(3, 1)
    d = torch.linalg.vector_norm(torch.cross(X - a, X - b, dim=0), dim=0) / torch.linalg.vector_norm(b - a)
    z0 = torch.as_tensor([float(p1[2])], dtype=xe.dtype)
    return _finish((-d / c * torch.sign(z0))[None])
