"""
Milestone 38 — smoothn completo (equivalente a pymust.smoothn / SMOOTHN de Damien Garcia), en PyTorch.

    z, s, exitflag = smoothn(y, W=None, S=None, axis=None, tol_z=1e-3, max_iter=100, initial=None,
                             spacing=None, order=2, weight="bisquare", robust=False)

y       : arreglo N-D (con NaN para datos faltantes). Con axis=[0] (o y como lista) la 1.a dimensión son
          componentes de un campo vectorial que se suavizan juntas.
W       : pesos >= 0 (misma forma que un componente).  S: parámetro de suavizado (None = automático por GCV).
order   : 0, 1 o 2 (orden de la penalización).  spacing: separación de la grilla en cada dimensión.
weight  : "bisquare", "talworth" o "cauchy" (pesos robustos).  robust: suavizado robusto (3 pasos, como MUST).
initial : estimación inicial de z (acelera la convergencia con datos faltantes o pesos).

A diferencia de processing.sptrack.smoothn (2-D, orden 2, bisquare, la versión rápida que usa sptrack), esta versión
tiene todas las opciones de MUST. La DCT es la de tipo II sin normalizar (la de scipy.fft.dctn), para que
el criterio GCV dé exactamente lo mismo que pymust.
"""
import math
import numpy as np
import torch

_DCT = {}


def _dct_mats(n):
    if n not in _DCT:
        k = torch.arange(n, dtype=torch.float64)[:, None]; i = torch.arange(n, dtype=torch.float64)[None]
        D = 2 * torch.cos(math.pi * k * (2 * i + 1) / (2 * n))            # DCT-II sin normalizar (scipy.fft.dct)
        _DCT[n] = (D, torch.linalg.inv(D))
    return _DCT[n]


def _dctn(x, inverse=False, axes=None):
    axes = range(x.ndim) if axes is None else axes
    for ax in axes:
        n = x.shape[ax]
        if n == 1:
            continue
        D, Di = _dct_mats(n)
        x = torch.movedim(torch.tensordot(Di if inverse else D, torch.movedim(x, ax, 0), dims=([1], [0])), 0, ax)
    return x


def _median(v):
    s = torch.sort(v.reshape(-1)).values; n = s.numel()
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _robust_weights(y, z, I, h, wstr):
    r = (y - z).reshape(y.shape[0], -1)
    rI = r[:, I.reshape(-1)]
    MMED = _median(rI)
    AD = torch.linalg.vector_norm(rI - MMED, dim=0)
    MAD = _median(AD)
    u = (torch.linalg.vector_norm(r, dim=0) / (1.4826 * MAD) / math.sqrt(1 - h)).reshape(I.shape)
    if wstr == "cauchy":
        W = 1 / (1 + (u / 2.385) ** 2)
    elif wstr == "talworth":
        W = (u < 2.795).double()
    else:
        W = (1 - (u / 4.685) ** 2) ** 2 * ((u / 4.685) < 1)
    return torch.nan_to_num(W, nan=0.0)


def _initial_guess(y, I):
    z = y.clone()
    if bool((~I).any()):
        from scipy.ndimage import distance_transform_edt
        _, L = distance_transform_edt(~I.cpu().numpy(), return_indices=True)
        mask = ~I.cpu().numpy()
        idx = tuple(torch.as_tensor(L[j][mask]) for j in range(L.shape[0]))
        for i in range(y.shape[0]):
            z[i][torch.as_tensor(mask)] = y[i][idx]
    z = _dctn(z, axes=range(1, z.ndim))
    ind = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float64) / s for s in z.shape], indexing="ij")).amax(0)
    z[ind > 0.1] = 0
    return _dctn(z, inverse=True, axes=range(1, z.ndim))


def _fminbound(f, a, b, xtol):
    import scipy.optimize
    return scipy.optimize.fminbound(f, a, b, xtol=xtol)


def smoothn(y, W=None, S=None, axis=None, tol_z=1e-3, max_iter=100, initial=None, spacing=None, order=2,
            weight="bisquare", robust=False):
    if isinstance(y, (list, tuple)):
        y = torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in y]); axis = [0]
    y = torch.as_tensor(y, dtype=torch.float64).clone()
    orig = y.shape
    if axis is None:
        y = y[None]
    sizy = y.shape[1:]; ny = y.shape[0]; noe = int(np.prod(sizy)); d = len(sizy)
    if noe == 1:
        return y.reshape(orig), None, True
    W = torch.ones(sizy, dtype=torch.float64) if W is None else torch.as_tensor(W, dtype=torch.float64).clone()
    isauto = S is None
    m = order; assert m in (0, 1, 2)
    weight = weight.lower(); assert weight in ("bisquare", "talworth", "cauchy")
    dI = torch.ones(d, dtype=torch.float64) if spacing is None else torch.as_tensor(spacing, dtype=torch.float64)
    dI = dI / dI.max()
    IsFinite = torch.isfinite(y).all(0)
    nof = int(IsFinite.sum())
    W = W * IsFinite
    isweighted = bool((W != 1).any())
    Lam = torch.zeros(sizy, dtype=torch.float64)
    for i in range(d):
        sh = [1] * d; sh[i] = sizy[i]
        Lam = Lam + (2 - 2 * torch.cos(math.pi * torch.arange(sizy[i], dtype=torch.float64).reshape(sh) / sizy[i])) / dI[i] ** 2
    if not isauto:
        Gamma = 1 / (1 + S * Lam ** m)
    N = sum(s != 1 for s in sizy)
    hMin, hMax = 1e-6, 0.99
    if m == 0:
        sMin, sMax = 1 / hMax ** (1 / N) - 1, 1 / hMin ** (1 / N) - 1
    elif m == 1:
        sMin, sMax = (1 / hMax ** (2 / N) - 1) / 4, (1 / hMin ** (2 / N) - 1) / 4
    else:
        sMin = (((1 + math.sqrt(1 + 8 * hMax ** (2 / N))) / 4 / hMax ** (2 / N)) ** 2 - 1) / 16
        sMax = (((1 + math.sqrt(1 + 8 * hMin ** (2 / N))) / 4 / hMin ** (2 / N)) ** 2 - 1) / 16
    Wtot = W
    if isweighted:
        z = torch.as_tensor(initial, dtype=torch.float64).reshape(y.shape).clone() if initial is not None else _initial_guess(y, IsFinite)
    else:
        z = torch.zeros_like(y)
    z0 = z.clone()
    y = torch.where(IsFinite[None], y, torch.zeros_like(y))
    tol, nit, step, errp = 1.0, 0, 1, 0.1
    RF = 1 + 0.75 * isweighted                                            # fijo desde el inicio, como MUST
    DCTy = torch.zeros_like(y)

    def gcv(p, aow):
        G = 1 / (1 + 10 ** p * Lam ** m)
        if aow > 0.95:
            RSS = sum(float(torch.linalg.vector_norm(DCTy[k] * (G - 1)) ** 2) for k in range(ny))
        else:
            RSS = 0.0
            for k in range(ny):
                yh = _dctn(G * DCTy[k], inverse=True)
                RSS += float(torch.linalg.vector_norm(torch.sqrt(Wtot[IsFinite]) * (y[k][IsFinite] - yh[IsFinite])) ** 2)
        return RSS / nof / (1 - float(G.sum()) / noe) ** 2

    while True:                                                           # hasta 3 pasadas robustas, como MUST
        aow = float(Wtot.sum() / W.max() / noe)
        while tol > tol_z and nit < max_iter:
            nit += 1
            for i in range(ny):
                DCTy[i] = _dctn(Wtot * (y[i] - z[i]) + z[i])
            if isauto and float(np.log2(nit)).is_integer():
                S = 10 ** _fminbound(lambda p: gcv(p, aow), math.log10(sMin), math.log10(sMax), errp)
                Gamma = 1 / (1 + S * Lam ** m)
            for i in range(ny):
                z[i] = RF * _dctn(Gamma * DCTy[i], inverse=True) + (1 - RF) * z[i]
            nz0 = float(torch.linalg.vector_norm(z0))
            tol = float(torch.linalg.vector_norm(z0 - z)) / nz0 if (isweighted and nz0 > 0) else 0.0
            z0 = z.clone()
        exitflag = nit < max_iter
        if not robust:
            break
        h = 1.0
        for k in range(N):
            if m == 0:
                h0 = 1 / (1 + S / float(dI[k]) ** (2 ** m))
            elif m == 1:
                h0 = 1 / math.sqrt(1 + 4 * S / float(dI[k]) * (2 ** m))
            else:
                h0 = math.sqrt(1 + 16 * S / float(dI[k]) ** (2 ** m)); h0 = math.sqrt(1 + h0) / math.sqrt(2) / h0
            h *= h0
        Wtot = W * _robust_weights(y, z, IsFinite, h, weight)
        isweighted = True; tol = 1.0; nit = 0; step += 1
        if step >= 4:
            break
    return z.reshape(orig), S, exitflag
