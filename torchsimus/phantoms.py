"""
Milestone 47 — genscat: fantoma de scatterers (equivalente a pymust.genscat / GENSCAT de MUST).

    x, y, z, RC = genscat(roidim, meandist, I=None, g=40)

roidim  : [ancho, alto] (2-D) o [ancho, alto, profundidad] (3-D) en m; un NaN se deduce del tamaño de I.
meandist: distancia media entre scatterers (m).  I: imagen (2-D) o volumen (3-D) de ecogenicidad.
g       : compresión log (dB) usada para convertir I en reflectividad (g <= 1 -> potencia 1/g).

Los números aleatorios salen de numpy (np.random global, o un Generator en `rng`), en el mismo orden
que pymust: con la misma semilla se obtienen los mismos scatterers.
Diferencias deliberadas con pymust 0.1.9:
  * 3-D funciona (en pymust la rama 3-D usa una variable inexistente y falla).
  * grilla de interpolación de I con los centros de píxel en x (igual que en z). pymust usa
    linspace(xmin - dx/2, xmax - dx/2, nc), que no coincide con los centros de píxel (en z sí usa
    zmin + dz/2 ... zmax - dz/2); pymust_compat=True reproduce exactamente lo de pymust.
"""
import math
import numpy as np
import torch


def _interp_linear(grids, V, pts):
    """Interpolación multilineal en grilla regular; 0 fuera (como RegularGridInterpolator(fill_value=0))."""
    idx_list, w_list, inside = [], [], torch.ones(pts.shape[0], dtype=torch.bool)
    for d, g in enumerate(grids):
        g = torch.as_tensor(g, dtype=torch.float64)
        p = pts[:, d]
        inside &= (p >= g[0]) & (p <= g[-1])
        t = (p - g[0]) / (g[1] - g[0])
        i0 = t.floor().clamp(0, g.numel() - 2).long()
        idx_list.append(i0); w_list.append((t - i0).clamp(0, 1))
    out = torch.zeros(pts.shape[0], dtype=torch.float64)
    nd = len(grids)
    for corner in range(2 ** nd):
        w = torch.ones_like(out); ix = []
        for d in range(nd):
            bit = (corner >> d) & 1
            w = w * (w_list[d] if bit else 1 - w_list[d]); ix.append(idx_list[d] + bit)
        out = out + w * V[tuple(ix)]
    return torch.where(inside, out, torch.zeros_like(out))


def genscat(roidim, meandist, I=None, g=None, rng=None, pymust_compat=False):
    R = np.random if rng is None else rng
    roidim = np.asarray(roidim, dtype=float)
    width, height = roidim[0], roidim[1]
    three_d = roidim.size == 3
    if not three_d:
        if I is not None:
            m, n = np.shape(I)
            if not np.isfinite(width): width = n * height / m
            if not np.isfinite(height): height = m * width / n
    else:
        depth = roidim[2]
        if I is not None and not np.all(np.isfinite(roidim)):
            m, n, p = np.shape(I)
            if np.isfinite(height): width, depth = n * height / m, p * height / m
            elif np.isfinite(width): height, depth = m * width / n, p * width / n
            else: width, height = n * depth / p, m * depth / p
        ymin, ymax = -depth / 2, depth / 2
    xmin, xmax, zmin, zmax = -width / 2, width / 2, 0.0, height

    if not three_d:
        inc = meandist / math.sqrt(2 / 5)
        xs, zs = np.meshgrid(np.arange(xmin, xmax, inc), np.arange(zmin, zmax, inc))
        xs = xs + R.rand(*xs.shape) * inc - inc / 2
        zs = zs + R.rand(*zs.shape) * inc - inc / 2
        k = (xs > xmin) & (xs < xmax) & (zs > zmin) & (zs < zmax)
        xs, zs = xs[k], zs[k]; ys = np.zeros(xs.shape)
    else:
        inc = meandist / math.sqrt(16 / 39)
        xs, ys, zs = np.meshgrid(np.arange(xmin, xmax, inc), np.arange(ymin, ymax, inc), np.arange(zmin, zmax, inc))
        xs = xs + R.rand(*xs.shape) * inc - inc / 2
        zs = zs + R.rand(*zs.shape) * inc - inc / 2
        ys = ys + R.rand(*ys.shape) * inc - inc / 2
        k = (xs > xmin) & (xs < xmax) & (ys > ymin) & (ys < ymax) & (zs > zmin) & (zs < zmax)
        xs, ys, zs = xs[k], ys[k], zs[k]
    perm = R.permutation(len(xs))
    xs, ys, zs = xs[perm], ys[perm], zs[perm]

    if I is None:
        RC = R.rayleigh(1, xs.shape) / math.sqrt(math.pi / 2)
        RC = torch.as_tensor(RC)
    else:
        Iv = torch.as_tensor(np.asarray(I, dtype=float)); Iv = Iv / Iv.max()
        g = 40 if g is None else g
        if not three_d:
            nl, nc = Iv.shape
            dxi, dzi = (xmax - xmin) / nc, (zmax - zmin) / nl
            if pymust_compat:
                xi = np.linspace(xmin - dxi / 2, xmax - dxi / 2, nc)           # como pymust 0.1.9 (asimétrica)
            else:
                xi = np.linspace(xmin + dxi / 2, xmax - dxi / 2, nc)           # centros de píxel, igual que en z
            zi = np.linspace(zmin + dzi / 2, zmax - dzi / 2, nl)
            RC = _interp_linear([xi, zi], Iv.T, torch.as_tensor(np.stack([xs, zs], -1)))
        else:
            nl, nc, nr = Iv.shape
            dxi, dyi, dzi = (xmax - xmin) / nc, (ymax - ymin) / nr, (zmax - zmin) / nl
            xi = np.linspace(xmin + dxi / 2, xmax - dxi / 2, nc)
            yi = np.linspace(ymin + dyi / 2, ymax - dyi / 2, nr)
            zi = np.linspace(zmin + dzi / 2, zmax - dzi / 2, nl)
            RC = _interp_linear([zi, xi, yi], Iv, torch.as_tensor(np.stack([zs, xs, ys], -1)))
        RC = 10 ** (g / 20 * (RC - 1)) if g > 1 else RC ** (1 / g)
        RC = RC * torch.as_tensor(np.hypot(R.rand(*xs.shape), R.rand(*xs.shape)) / math.sqrt(math.pi / 2))
    return torch.as_tensor(xs), torch.as_tensor(ys), torch.as_tensor(zs), RC
