"""
Beamforming delay-and-sum en PyTorch (equivalente a DASMTX / DASMTX3 de MUST aplicados a los datos).

Milestones 41-43: das para arreglos lineales y convexos, RF o I/Q, uno o varios cuadros.
Incluye todas las opciones de DASMTX / DASMTX3 y das3 para arreglos 2-D (matriciales).

    img = das(SIG, x, z, tx_delays, transducer, fs, fc=None, c=1540, t0=0, fnumber=0, method="linear",
              passive=False, rx_angle=0, virtual_source=False, tx_interp="cubic")
    img = das3(SIG, x, y, z, tx_delays, transducer, fs, fc=None, c=1540, t0=0, fnumber=(0, 0), method="linear",
               passive=False, virtual_source=False)

method        : "nearest", "linear", "quadratic", "lanczos3", "5points", "lanczos5" (interpolación temporal).
fnumber       : 0 = apertura completa; None = automático (criterio de MUST, incluida su tolerancia xtol = π/100).
passive       : True = imagen pasiva (sin tiempo de ida; p. ej. fuentes que emiten solas).
rx_angle      : ángulo de recepción (rad) para Doppler vectorial: inclina la apertura de recepción (solo lineal).
virtual_source: True = tiempo de ida con el "transductor virtual" de MUST (vxdcr / vxdcr3) en vez del mínimo sobre elementos.
tx_interp     : "cubic" (como MUST: fuentes de transmisión sobre-muestreadas x4 con spline) o "linear" (das del M43).
SIG           : [nt, N] o [nt, N, F]; RF real o I/Q complejo.
"""
import math
import numpy as np
import torch

_BOUND = {"nearest": 1, "linear": 2, "quadratic": 3, "lanczos3": 3, "5points": 3, "lanczos5": 4}


def _delays(tx_delays, dev):
    """Retardos [N] en float64 (acepta tensor, numpy o lista; NaN = elemento que no transmitió)."""
    if torch.is_tensor(tx_delays):
        return tx_delays.to(dev, torch.float64).reshape(-1)
    return torch.as_tensor(np.asarray(tx_delays, dtype=float), dtype=torch.float64, device=dev).reshape(-1)


def _interp_lin(v, idx):
    """Interpolación lineal de v (sobre índices 0..n-1) en posiciones idx."""
    i0 = idx.floor().clamp(0, v.numel() - 2).long()
    a = idx - i0
    return v[i0] * (1 - a) + v[i0 + 1] * a


def _sincn(x):
    return torch.sinc(x)                                                   # sin(pi x) / (pi x), como np.sinc


def _kernel(method, a):
    """Desplazamientos de muestra y pesos del interpolador (a = parte fraccionaria), igual que MUST."""
    if method == "linear":
        return [0, 1], [1 - a, a]
    if method == "quadratic":
        return [0, 1, 2], [(a - 1) * (a - 2) / 2, -a * (a - 2), a * (a - 1) / 2]
    if method == "lanczos3":
        return [-1, 0, 1, 2], [_sincn(a + 1) * _sincn((a + 1) / 2), _sincn(a) * _sincn(a / 2),
                               _sincn(a - 1) * _sincn((a - 1) / 2), _sincn(a - 2) * _sincn((a - 2) / 2)]
    if method == "5points":
        a2 = a ** 2
        return [-2, -1, 0, 1, 2], [a2 / 7 - a / 5 - 3 / 35, -a2 / 14 - a / 10 + 12 / 35, -a2 / 7 + 17 / 35,
                                   -a2 / 14 + a / 10 + 12 / 35, a2 / 7 + a / 5 - 3 / 35]
    if method == "lanczos5":
        return [-2, -1, 0, 1, 2, 3], [_sincn(a - k) * _sincn((a - k) / 2) for k in (-2, -1, 0, 1, 2, 3)]
    raise ValueError(f"método desconocido: {method}")


def _spline(y, xi):
    """Spline cúbica 'not-a-knot' de y (muestreada en 0..n-1) evaluada en xi (= scipy interp1d kind='cubic')."""
    n = y.numel()
    if n < 4:
        return _interp_lin(y, xi)
    A = torch.zeros(n, n, dtype=torch.float64, device=y.device); b = torch.zeros(n, dtype=torch.float64, device=y.device)
    A[0, 0], A[0, 1], A[0, 2] = 1, -2, 1                                   # not-a-knot (3.a derivada continua en x1)
    A[-1, -3], A[-1, -2], A[-1, -1] = 1, -2, 1
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = 1, 4, 1
        b[i] = 6 * (y[i + 1] - 2 * y[i] + y[i - 1])
    M = torch.linalg.solve(A, b.to(torch.float64))
    i = xi.floor().clamp(0, n - 2).long(); t = xi - i
    return (1 - t) * y[i] + t * y[i + 1] + ((1 - t) ** 3 - (1 - t)) * M[i] / 6 + (t ** 3 - t) * M[i + 1] / 6


def auto_fnumber(width, fc, bandwidth, c=1540.0, rx_angle=0.0, xtol=math.pi / 100):
    """f-number óptimo de MUST: ángulo donde la directividad del elemento cae a 0.71."""
    import scipy.optimize
    lam = c / (fc * (1 + bandwidth / 2)); ra = abs(rx_angle)
    f = lambda th: abs(math.cos(th + ra) * np.sinc(width / lam * math.sin(th + ra)) - 0.71)
    return 1 / 2 / math.tan(scipy.optimize.fminbound(f, 0, math.pi / 2 - ra, xtol=xtol))


def _diff2(x, y):
    """Derivada de 2.o orden en una grilla no uniforme (diff2 de MUST)."""
    dx = torch.diff(x); dy = torch.zeros_like(y)
    dy[0] = (1 / dx[0] + 1 / dx[1]) * (y[1] - y[0]) + dx[0] / (dx[0] * dx[1] + dx[1] ** 2) * (y[0] - y[2])
    dy[-1] = (1 / dx[-2] + 1 / dx[-1]) * (y[-1] - y[-2]) + dx[-1] / (dx[-1] * dx[-2] + dx[-2] ** 2) * (y[-3] - y[-1])
    d1, d2 = dx[:-1], dx[1:]
    dy[1:-1] = 1 / (d1 * d2 * (d1 + d2)) * (-d2 ** 2 * y[:-2] + (d2 ** 2 - d1 ** 2) * y[1:-1] + d1 ** 2 * y[2:])
    return dy


def _vxdcr(xe, ze, dels, c):
    """Transductor virtual (retardos nulos) equivalente a la emisión (vxdcr de MUST)."""
    T2 = dels ** 2
    dT2 = _diff2(xe, T2)
    tmp = c * dels.abs()
    if bool((ze == 0).all()):
        xv = xe - 0.5 * c ** 2 * dT2
        zv = -torch.sqrt((c ** 2 * T2 - (xv - xe) ** 2).abs())
    else:
        xv = xe + tmp * _diff2(xe, ze) - 0.5 * c ** 2 * dT2
        zv = ze - tmp
    return xv, zv


def _gather(sig, col, nl, N, idxt, ok, method, Fr):
    """Suma sobre elementos de la señal interpolada en idxt (índices como en MUST: columna*nl + muestra)."""
    tot = nl * N
    if method == "nearest":
        j = (idxt.round().long() + col).clamp(0, tot - 1)
        val = sig[j.reshape(-1)].reshape(*j.shape, Fr)
    else:
        j0 = idxt.floor(); a = (idxt - j0)[..., None]
        offs, ws = _kernel(method, a)
        val = 0
        for o, w in zip(offs, ws):
            j = (j0.long() + o + col).clamp(0, tot - 1)
            val = val + sig[j.reshape(-1)].reshape(*j.shape, Fr) * w
    return val


def das(SIG, x, z, tx_delays, transducer, fs, fc=None, c=1540.0, t0=0.0, fnumber=0.0, method="linear",
        passive=False, rx_angle=0.0, virtual_source=False, tx_interp="cubic", width=None, bandwidth=None, budget=2 ** 25):
    SIG = torch.as_tensor(SIG)
    single = SIG.ndim == 2
    if single:
        SIG = SIG[..., None]
    nl, N, Fr = SIG.shape
    dev = SIG.device
    isIQ = torch.is_complex(SIG)
    X = torch.as_tensor(x, dtype=torch.float64, device=dev); shape = X.shape; X = X.reshape(-1)
    Z = torch.as_tensor(z, dtype=torch.float64, device=dev).reshape(-1)
    pose = transducer.geometry.pose()
    xe = pose.centers[:, 0].to(dev, torch.float64); ze = pose.centers[:, 2].to(dev, torch.float64)
    the = torch.atan2(pose.n[:, 0], pose.n[:, 2]).to(dev, torch.float64)
    convex = hasattr(transducer.geometry, "curvature_center")
    assert not (convex and rx_angle), "rx_angle debe ser 0 con un arreglo convexo (como en MUST)"
    fc = float(transducer.fc) if fc is None else float(fc)
    if fnumber is None:
        w = float(transducer.element_shape.width) if width is None else width
        fnumber = auto_fnumber(w, fc, float(transducer.bandwidth) if bandwidth is None else bandwidth, c, rx_angle)
    D = _delays(tx_delays, dev)
    act = ~torch.isnan(D); nTX = int(act.sum())
    if not passive and not virtual_source:
        if nTX > 1:
            idxi = torch.linspace(0, nTX - 1, 4 * nTX, dtype=torch.float64, device=dev)
            if tx_interp == "cubic":
                xT, dT = _spline(xe[act], idxi), _spline(D[act], idxi)
                zT = _spline(ze[act], idxi) if convex else torch.zeros_like(idxi)
            else:
                xT, dT = _interp_lin(xe[act], idxi), _interp_lin(D[act], idxi)
                zT = _interp_lin(ze[act], idxi) if convex else torch.zeros_like(idxi)
        else:
            xT, dT = xe[act], D[act]; zT = ze[act] if convex else torch.zeros(1, dtype=torch.float64, device=dev)
    if virtual_source and not passive:
        if nTX == 1:
            pass
        elif nTX < 3:
            raise ValueError("Con fuente virtual se necesitan 1 o al menos 3 elementos emisores")
        else:
            xv, zv = _vxdcr(xe[act], ze[act], D[act], c)
            dzv = _diff2(xv, zv)
    lim = nl - _BOUND[method]
    out = torch.zeros(X.numel(), Fr, dtype=torch.complex128 if isIQ else torch.float64, device=dev)
    sig = SIG.to(torch.complex128 if isIQ else torch.float64).permute(1, 0, 2).reshape(N * nl, Fr)
    col = torch.arange(N, device=dev)[None, :] * nl
    pc = max(1, int(budget // (N * max(Fr, 1) * 6)))
    tanr, cosr = math.tan(rx_angle), math.cos(rx_angle)
    for s in range(0, X.numel(), pc):
        xp, zp = X[s:s + pc, None], Z[s:s + pc, None]
        if passive:
            dTX = torch.zeros_like(xp)
        elif not virtual_source:
            dTX = (dT[None] * c + torch.sqrt((xT[None] - xp) ** 2 + (zT[None] - zp) ** 2)).min(1).values[:, None]
        elif nTX == 1:
            k = int(torch.nonzero(act)[0])
            dTX = torch.hypot(xe[k] - xp, ze[k] - zp) + D[k] * c
        else:
            Dn = (xp - xv[None] + dzv[None] * (zp - zv[None])).abs() / torch.hypot(torch.ones_like(dzv), dzv)[None]
            i = Dn.argmin(1)
            dTX = torch.hypot(xv[i][:, None] - xp, zv[i][:, None] - zp)
        dx = xp - xe[None]
        dRX = torch.sqrt(dx ** 2 + (zp - ze[None]) ** 2)
        tau = (dTX + dRX) / c
        idxt = (tau - t0) * fs
        ok = (idxt >= 0) & (idxt <= lim)
        if fnumber > 0:
            if convex:
                ok = ok & ((torch.asin(dx / dRX) - the[None]).abs() <= math.atan(1 / 2 / fnumber))
            elif rx_angle:
                ok = ok & ((dx - zp * tanr).abs() <= zp / cosr / 2 / fnumber)
            else:
                ok = ok & (dx.abs() <= zp / 2 / fnumber)
        val = _gather(sig, col, nl, N, idxt, ok, method, Fr)
        if isIQ:
            val = val * torch.exp(1j * 2 * math.pi * fc * tau)[..., None]
        out[s:s + pc] = (val * ok[..., None]).sum(1)
    out = out.reshape(*shape, Fr)
    return out[..., 0] if single else out


def _trigrad(x, y, z):
    """Gradiente de z sobre datos dispersos (x, y) con una triangulación de Delaunay (trigrad de MUST)."""
    import scipy.spatial, scipy.sparse, scipy.sparse.linalg
    x, y, z = [np.asarray(v, dtype=float).ravel() for v in (x, y, z)]
    dt = scipy.spatial.Delaunay(np.vstack((x, y)).T, qhull_options="Qt Qbb Qc").simplices
    nt = dt.shape[0]
    C = np.zeros((nt, 3))
    for t in range(nt):                                                    # plano z = a x + b y + d por triángulo
        C[t] = np.linalg.solve(np.stack([x[dt[t]], y[dt[t]], np.ones(3)], 1), z[dt[t]])
    s1 = np.hypot(x[dt[:, 1]] - x[dt[:, 0]], y[dt[:, 1]] - y[dt[:, 0]]); s2 = np.hypot(x[dt[:, 2]] - x[dt[:, 0]], y[dt[:, 2]] - y[dt[:, 0]])
    s3 = np.hypot(x[dt[:, 2]] - x[dt[:, 1]], y[dt[:, 2]] - y[dt[:, 1]]); s = (s1 + s2 + s3) / 2
    A = np.sqrt(s * (s - s1) * (s - s2) * (s - s3)); E = 8 * A ** 2 / (s * s1 * s2 * s3)
    M = scipy.sparse.coo_matrix((np.ones(3 * nt), (np.repeat(np.arange(nt), 3), dt.ravel())), shape=(nt, len(x)))
    return (M.T @ (C[:, 0] * E)) / (M.T @ E), (M.T @ (C[:, 1] * E)) / (M.T @ E)


def das3(SIG, x, y, z, tx_delays, transducer, fs, fc=None, c=1540.0, t0=0.0, fnumber=(0.0, 0.0), method="linear",
         passive=False, virtual_source=False, width=None, height=None, bandwidth=None, budget=2 ** 25):
    SIG = torch.as_tensor(SIG)
    single = SIG.ndim == 2
    if single:
        SIG = SIG[..., None]
    nl, N, Fr = SIG.shape
    dev = SIG.device
    isIQ = torch.is_complex(SIG)
    X = torch.as_tensor(x, dtype=torch.float64, device=dev); shape = X.shape; X = X.reshape(-1)
    Y = torch.as_tensor(y, dtype=torch.float64, device=dev).reshape(-1); Z = torch.as_tensor(z, dtype=torch.float64, device=dev).reshape(-1)
    ce = transducer.geometry.pose().centers.to(dev, torch.float64)
    xe, ye, ze = ce[:, 0], ce[:, 1], ce[:, 2]
    fc = float(transducer.fc) if fc is None else float(fc)
    if fnumber is None:
        bw = float(transducer.bandwidth) if bandwidth is None else bandwidth
        w = float(transducer.element_shape.width) if width is None else width
        h = float(getattr(transducer.element_shape, "height", getattr(transducer, "height"))) if height is None else height
        fnumber = (auto_fnumber(w, fc, bw, c), auto_fnumber(h, fc, bw, c))
    fx, fy = float(fnumber[0]), float(fnumber[1])
    D = _delays(tx_delays, dev)
    act = ~torch.isnan(D); nTX = int(act.sum())
    if virtual_source and not passive and nTX >= 3:
        xa, ya, da = [v[act].cpu().numpy() for v in (xe, ye, D)]
        T2 = da ** 2
        gx, gy = _trigrad(xa, ya, T2)
        xv = xa - c ** 2 * gx / 2; yv = ya - c ** 2 * gy / 2
        zv = -np.sqrt(np.abs(c ** 2 * T2 - (xa - xv) ** 2 - (ya - yv) ** 2))
        dzx, dzy = _trigrad(xv, yv, zv)
        xv, yv, zv, dzx, dzy = [torch.as_tensor(v, dtype=torch.float64, device=dev) for v in (xv, yv, zv, dzx, dzy)]
    Dinf = torch.where(torch.isnan(D), torch.full_like(D, float("inf")), D)
    lim = nl - _BOUND[method]
    out = torch.zeros(X.numel(), Fr, dtype=torch.complex128 if isIQ else torch.float64, device=dev)
    sig = SIG.to(torch.complex128 if isIQ else torch.float64).permute(1, 0, 2).reshape(N * nl, Fr)
    col = torch.arange(N, device=dev)[None, :] * nl
    pc = max(1, int(budget // (N * max(Fr, 1) * 6)))
    for s in range(0, X.numel(), pc):
        xp, yp, zp = X[s:s + pc, None], Y[s:s + pc, None], Z[s:s + pc, None]
        dx, dy, dz = xp - xe[None], yp - ye[None], zp - ze[None]
        dRX = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if passive:
            dTX = torch.zeros_like(xp)
        elif not virtual_source:
            dTX = (Dinf[None] * c + dRX).min(1, keepdim=True).values
        elif nTX == 1:
            k = int(torch.nonzero(act)[0])
            dTX = torch.hypot(xe[k] - xp, ze[k] - zp) + D[k] * c
        else:
            Dn = (dzx[None] * (yp - yv[None]) + dzy[None] * (zp - zv[None]) - (xp - xv[None])).abs() / (1 + dzx ** 2 + dzy ** 2).abs()[None]
            i = Dn.argmin(1)
            dTX = torch.sqrt((xv[i][:, None] - xp) ** 2 + (yv[i][:, None] - yp) ** 2 + (zv[i][:, None] - zp) ** 2)
        tau = (dTX + dRX) / c
        idxt = (tau - t0) * fs
        ok = (idxt >= 0) & (idxt <= lim)
        if fx != 0 or fy != 0:
            ok = ok & (dx.abs() <= dz.abs() / 2 / fx if fx else True) & (dy.abs() <= dz.abs() / 2 / fy if fy else True)
        val = _gather(sig, col, nl, N, idxt, ok, method, Fr)
        if isIQ:
            val = val * torch.exp(1j * 2 * math.pi * fc * tau)[..., None]
        out[s:s + pc] = (val * ok[..., None]).sum(1)
    out = out.reshape(*shape, Fr)
    return out[..., 0] if single else out
