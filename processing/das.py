"""
Milestone 43: beamforming delay-and-sum en PyTorch (equivalente a DASMTX de MUST aplicado a los datos).
Funciona con arreglos lineales y convexos, cualquier ley de retardos de transmisión (onda plana,
divergente, enfocada) y datos RF o I/Q, con uno o varios cuadros.
"""
import math
import torch
from processing.optim import fminbound


def auto_fnumber(width, fc, bandwidth, c=1540.0):
    """f-number óptimo según la directividad del elemento (criterio de MUST: directividad = 0.71)."""
    lam = c / (fc * (1 + bandwidth / 2))
    def f(th):
        s = width / lam * math.sin(th)
        sinc = 1.0 if s == 0 else math.sin(math.pi * s) / (math.pi * s)
        return abs(math.cos(th) * sinc - 0.71)
    alpha = fminbound(f, 0.0, math.pi / 2, xtol=1e-8)
    return 1 / 2 / math.tan(alpha)


def _interp_rows(v, idx):
    """Interpolación lineal de v (sobre índices 0..n-1) en posiciones idx."""
    i0 = idx.floor().clamp(0, v.numel() - 2).long()
    a = idx - i0
    return v[i0] * (1 - a) + v[i0 + 1] * a


def das(SIG, x, z, tx_delays, transducer, fs, fc=None, c=1540.0, t0=0.0, fnumber=0.0,
        method="linear", bandwidth=None, width=None, budget=2 ** 25):
    """
    SIG : [nt, N] o [nt, N, F] (RF real o I/Q complejo; F cuadros con la misma ley de retardos).
    x, z: grilla de pixeles (m), cualquier forma.   tx_delays: [N] (NaN = elemento que no transmitió).
    fnumber: 0 = apertura completa; None = automático (requiere width y bandwidth).
    Devuelve la imagen beamformada con forma x.shape (+ [F] si hay cuadros).
    """
    SIG = torch.as_tensor(SIG)
    single = SIG.ndim == 2
    if single:
        SIG = SIG[..., None]
    nl, N, Fr = SIG.shape
    dev = SIG.device
    isIQ = torch.is_complex(SIG)
    X = torch.as_tensor(x, dtype=torch.float64, device=dev)
    shape = X.shape
    X = X.reshape(-1)
    Z = torch.as_tensor(z, dtype=torch.float64, device=dev).reshape(-1)

    pose = transducer.geometry.pose()
    xe = pose.centers[:, 0].to(dev, torch.float64)
    ze = pose.centers[:, 2].to(dev, torch.float64)
    the = torch.atan2(pose.n[:, 0], pose.n[:, 2]).to(dev, torch.float64)
    convex = hasattr(transducer.geometry, "curvature_center")

    if fnumber is None:
        width = width if width is not None else float(getattr(transducer.element_shape, "width"))
        bandwidth = bandwidth if bandwidth is not None else float(transducer.bandwidth)
        fnumber = auto_fnumber(width, fc if fc is not None else transducer.fc, bandwidth, c)
    if fc is None:
        fc = float(transducer.fc)

    # fuentes de transmisión: elementos activos sobre-muestreados x4 (como MUST)
    D = torch.as_tensor(tx_delays, dtype=torch.float64, device=dev).reshape(-1)
    act = ~torch.isnan(D)
    nTX = int(act.sum())
    if nTX > 1:
        idxi = torch.linspace(0, nTX - 1, 4 * nTX, dtype=torch.float64, device=dev)
        xT = _interp_rows(xe[act], idxi)
        zT = _interp_rows(ze[act], idxi) if convex else torch.zeros_like(idxi)
        dT = _interp_rows(D[act], idxi)
    else:
        xT, zT, dT = xe[act], ze[act] if convex else torch.zeros(1, dtype=torch.float64, device=dev), D[act]

    lim = {"nearest": nl - 1, "linear": nl - 2}[method]
    out = torch.zeros(X.numel(), Fr, dtype=torch.complex128 if isIQ else torch.float64, device=dev)
    sig = SIG.to(torch.complex128 if isIQ else torch.float64).permute(1, 0, 2).reshape(N * nl, Fr)   # índice = n*nl + t
    col = torch.arange(N, device=dev)[None, :] * nl
    pc = max(1, int(budget // (N * max(Fr, 1) * 2)))
    for s in range(0, X.numel(), pc):
        xp, zp = X[s:s + pc, None], Z[s:s + pc, None]
        dTX = (dT[None] * c + torch.sqrt((xT[None] - xp) ** 2 + (zT[None] - zp) ** 2)).min(1).values[:, None]
        dx = xp - xe[None]
        dRX = torch.sqrt(dx ** 2 + (zp - ze[None]) ** 2)
        tau = (dTX + dRX) / c
        idxt = (tau - t0) * fs
        ok = (idxt >= 0) & (idxt <= lim)
        if fnumber > 0:
            if convex:
                ok = ok & ((torch.asin(dx / dRX) - the[None]).abs() <= math.atan(1 / 2 / fnumber))
            else:
                ok = ok & (dx.abs() <= zp / 2 / fnumber)
        if method == "nearest":
            j = idxt.round().clamp(0, nl - 1).long()
            val = sig[(j + col).reshape(-1)].reshape(*j.shape, Fr)
        else:
            j0 = idxt.floor().clamp(0, nl - 2).long()
            a = (idxt - j0)[..., None]
            val = sig[(j0 + col).reshape(-1)].reshape(*j0.shape, Fr) * (1 - a) + \
                  sig[(j0 + 1 + col).reshape(-1)].reshape(*j0.shape, Fr) * a
        if isIQ:
            val = val * torch.exp(1j * 2 * math.pi * fc * tau)[..., None]
        out[s:s + pc] = (val * ok[..., None]).sum(1)
    out = out.reshape(*shape, Fr)
    return out[..., 0] if single else out
