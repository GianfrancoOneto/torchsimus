"""
Milestone 38: speckle tracking (equivalente a pymust.sptrack) en PyTorch.

Algoritmo (PIV multipasada, Garcia et al.):
  * ventanas de interrogación con traslape, ventana de Hann;
  * correlación cruzada por FFT de TODAS las ventanas a la vez (batch en torch) y promedio de
    ensamble sobre los pares de imágenes (k, k + iminc);
  * pico entero + ajuste parabólico sub-píxel;
  * en cada pasada el desplazamiento previo se interpola a la grilla nueva y se usa como offset;
  * suavizado robusto penalizado (DCT + GCV + pesos bisquare, Garcia 2010) con pesos sqrt(C).
El suavizado es una implementación propia del método publicado: los vectores coinciden de cerca
con pymust, no bit a bit.
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fnn


# ------------------------------------------------------------------------------ utilidades
def _dct_matrix(n, dtype, device):
    k = torch.arange(n, dtype=dtype, device=device)[:, None]
    i = torch.arange(n, dtype=dtype, device=device)[None, :]
    C = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    C[0] = C[0] / math.sqrt(2.0)
    return C                                                          # ortonormal: C @ C.T = I


def _inpaint_nan(A, n_iter=500):
    """Rellena NaN resolviendo Laplace (difusión) con los valores finitos como condición de borde."""
    A = A.clone()
    miss = ~torch.isfinite(A)
    if not miss.any():
        return A
    if miss.all():
        return torch.zeros_like(A)
    A[miss] = A[~miss].mean()
    for _ in range(n_iter):
        P = Fnn.pad(A[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        avg = (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]) / 4
        A = torch.where(miss, avg, A)
    return A


def smoothn(y, W=None, robust=True, tol_z=1e-3, max_iter=100):
    """
    Suavizado robusto 2-D multicomponente (y: [ny, H, W], puede tener NaN). Devuelve z [ny, H, W].
    Penalización de orden 2 en el dominio DCT; s elegido por GCV; 3 pasos robustos (bisquare).
    """
    from processing.optim import fminbound
    y = y.clone().double()
    ny, H, Wd = y.shape
    dev = y.device
    if W is None:
        W = torch.ones(H, Wd, dtype=torch.float64, device=dev)
    W = W.double().clone()
    finite = torch.isfinite(y).all(0)
    W = torch.where(finite, W, torch.zeros_like(W))
    W = torch.nan_to_num(W, nan=0.0)
    nof, noe = int(finite.sum()), H * Wd
    if nof == 0:
        return torch.zeros_like(y)
    Ch, Cw = _dct_matrix(H, torch.float64, dev), _dct_matrix(Wd, torch.float64, dev)
    dct = lambda X: Ch @ X @ Cw.T
    idct = lambda X: Ch.T @ X @ Cw
    Lam = (2 - 2 * torch.cos(math.pi * torch.arange(H, dtype=torch.float64, device=dev) / H))[:, None] + \
          (2 - 2 * torch.cos(math.pi * torch.arange(Wd, dtype=torch.float64, device=dev) / Wd))[None, :]
    Nr = int(H > 1) + int(Wd > 1)
    hmin, hmax = 1e-6, 0.99
    smin = ((1 + math.sqrt(1 + 8 * hmax ** (2 / Nr))) / 4 / hmax ** (2 / Nr)) ** 2 - 1
    smax = ((1 + math.sqrt(1 + 8 * hmin ** (2 / Nr))) / 4 / hmin ** (2 / Nr)) ** 2 - 1
    smin, smax = smin / 16, smax / 16

    y0 = torch.stack([_inpaint_nan(torch.where(finite, y[i], torch.full_like(y[i], float("nan")))) for i in range(ny)])
    y = torch.where(finite[None], y, torch.zeros_like(y))
    weighted = bool((W != 1).any())
    z = y0.clone() if weighted else torch.zeros_like(y)
    Wtot = W.clone()
    s = None
    for robust_step in range(3 if robust else 1):
        aow = float(Wtot.sum() / W.max() / noe)
        RF = 1 + 0.75 * weighted
        tol, nit, zprev = 1.0, 0, z.clone()
        while tol > tol_z and nit < max_iter:
            nit += 1
            DCTy = torch.stack([dct(Wtot * (y[i] - z[i]) + z[i]) for i in range(ny)])
            if s is None or float(np.log2(nit)).is_integer():
                def gcv(p):
                    G = 1 / (1 + 10 ** p * Lam ** 2)
                    if aow > 0.95:
                        rss = float(((DCTy * (G - 1)) ** 2).sum())
                    else:
                        yh = torch.stack([idct(G * DCTy[i]) for i in range(ny)])
                        rss = float((Wtot[finite] * (y[:, finite] - yh[:, finite]) ** 2).sum())
                    return rss / nof / (1 - float(G.sum()) / noe) ** 2
                s = 10 ** fminbound(gcv, math.log10(smin), math.log10(smax), xtol=0.1)
            G = 1 / (1 + s * Lam ** 2)
            z = RF * torch.stack([idct(G * DCTy[i]) for i in range(ny)]) + (1 - RF) * z
            tol = float(weighted) * float(torch.linalg.norm(zprev - z) / torch.linalg.norm(zprev).clamp_min(1e-30))
            zprev = z.clone()
        if robust:
            h0 = math.sqrt(1 + 16 * s)
            h = (math.sqrt(1 + h0) / math.sqrt(2) / h0) ** Nr
            r = (y - z).reshape(ny, -1)
            rI = r[:, finite.reshape(-1)]
            mmed = rI.median()
            mad = torch.linalg.norm(rI - mmed, dim=0).median()
            u = torch.linalg.norm(r, dim=0) / (1.4826 * mad) / math.sqrt(1 - h)
            u = u.reshape(H, Wd)
            wr = (1 - (u / 4.685) ** 2) ** 2 * (u / 4.685 < 1)
            Wtot = W * torch.nan_to_num(wr, nan=0.0)
            weighted = True
    return z


def _interp_grid(A, ic0, jc0, ic, jc):
    """Interpolación bicúbica de A (definida en centros ic0 x jc0) a los centros ic x jc; fuera -> NaN."""
    A = _inpaint_nan(A)
    gi = (torch.as_tensor(ic, dtype=torch.float64) - ic0[0]) / (ic0[-1] - ic0[0]) * 2 - 1
    gj = (torch.as_tensor(jc, dtype=torch.float64) - jc0[0]) / (jc0[-1] - jc0[0]) * 2 - 1
    GI, GJ = torch.meshgrid(gi, gj, indexing="ij")
    grid = torch.stack([GJ, GI], -1)[None].to(A.device)
    out = Fnn.grid_sample(A[None, None], grid, mode="bicubic", align_corners=True)[0, 0]
    outside = (GI.abs() > 1) | (GJ.abs() > 1)
    return torch.where(outside.to(A.device), torch.full_like(out, float("nan")), out)


# ------------------------------------------------------------------------------ sptrack
def sptrack(I, winsize, iminc=1, overlap=50, roi=None, device=None):
    """
    I       : [M, N, P] serie de imágenes (modo B, p. ej. uint8).
    winsize : [[m1, n1], [m2, n2], ...] tamaños de ventana (decrecientes, cuadradas).
    iminc   : se correlaciona la imagen k con la k + iminc.
    roi     : máscara booleana [M, N] (por defecto, todo lo finito).
    Devuelve (Di, Dj, id, jd) con la misma convención de pymust.sptrack:
        Di = desplazamiento en columnas (x), Dj = en filas (z), id/jd = centros de las ventanas.
    """
    I = torch.as_tensor(np.asarray(I, dtype=np.float64), device=device)
    M, N, P = I.shape
    dev = I.device
    winsize = np.atleast_2d(np.asarray(winsize, dtype=int))
    ov = overlap / 100
    if roi is None:
        roi = torch.isfinite(I).all(2)
    roi = torch.as_tensor(np.asarray(roi, dtype=bool), device=dev)
    I = torch.where(roi[..., None], I, torch.full_like(I, float("nan")))

    di = dj = None
    for kk, (m, n) in enumerate(winsize):
        inci, incj = int(math.ceil(m * (1 - ov))), int(math.ceil(n * (1 - ov)))
        i_arr = np.arange(0, M - m + 1, inci)
        j_arr = np.arange(0, N - n + 1, incj)
        ic, jc = (2 * i_arr + m) / 2, (2 * j_arr + n) / 2
        if kk == 0:
            di = torch.zeros(len(i_arr), len(j_arr), dtype=torch.float64, device=dev)
            dj = torch.zeros_like(di)
        else:
            di = torch.round(_inpaint_nan(_interp_grid(di, ic0, jc0, ic, jc)))
            dj = torch.round(_inpaint_nan(_interp_grid(dj, ic0, jc0, ic, jc)))
        ic0, jc0 = ic, jc

        # ventanas de todas las posiciones a la vez: [nW, m, n, P]
        pi = torch.as_tensor(i_arr, device=dev)[:, None].expand(len(i_arr), len(j_arr)).reshape(-1)
        pj = torch.as_tensor(j_arr, device=dev)[None, :].expand(len(i_arr), len(j_arr)).reshape(-1)
        oi, oj = di.reshape(-1).long(), dj.reshape(-1).long()
        valid = (pi + oi >= 0) & (pj + oj >= 0) & (pi + oi + m < M) & (pj + oj + n < N)
        ri = torch.arange(m, device=dev)
        rj = torch.arange(n, device=dev)
        r1 = (pi[:, None] + ri[None]).clamp(0, M - 1)
        c1 = (pj[:, None] + rj[None]).clamp(0, N - 1)
        r2 = (pi[:, None] + oi[:, None] + ri[None]).clamp(0, M - 1)
        c2 = (pj[:, None] + oj[:, None] + rj[None]).clamp(0, N - 1)
        W1 = I[r1[:, :, None], c1[:, None, :], :][..., :-iminc]
        W2 = I[r2[:, :, None], c2[:, None, :], :][..., iminc:]
        valid = valid & torch.isfinite(W1).flatten(1).all(1) & torch.isfinite(W2).flatten(1).all(1)
        W1 = torch.nan_to_num(W1); W2 = torch.nan_to_num(W2)
        hm = torch.hann_window(m + 2, periodic=False, dtype=torch.float64, device=dev)[1:-1]
        hn = torch.hann_window(n + 2, periodic=False, dtype=torch.float64, device=dev)[1:-1]
        Hw = (hm[:, None] * hn[None, :])[None, :, :, None]
        A1 = (W1 - W1.mean(dim=(1, 2, 3), keepdim=True)) * Hw
        A2 = (W2 - W2.mean(dim=(1, 2, 3), keepdim=True)) * Hw
        R = torch.fft.fft2(A2, dim=(1, 2)) * torch.fft.fft2(A1, dim=(1, 2)).conj()        # [nW, m, n, P-iminc]
        R2 = torch.fft.ifft2((R / (R.abs() + 1e-6)).mean(-1), dim=(1, 2)).real
        C = R2.flatten(1).max(1).values
        Rs = torch.fft.fftshift(torch.fft.ifft2(R.sum(-1), dim=(1, 2)).real, dim=(1, 2))   # [nW, m, n]
        idx = Rs.flatten(1).argmax(1)
        a0, b0 = idx // n, idx % n
        w = torch.arange(Rs.shape[0], device=dev)
        Rc = Rs[w, a0, b0]
        am, ap = Rs[w, (a0 - 1).clamp(0, m - 1), b0], Rs[w, (a0 + 1).clamp(0, m - 1), b0]
        bm, bp = Rs[w, a0, (b0 - 1).clamp(0, n - 1)], Rs[w, a0, (b0 + 1).clamp(0, n - 1)]
        sub_i = torch.where((a0 > 0) & (a0 < m - 1), (am - ap) / (2 * am - 4 * Rc + 2 * ap), torch.zeros_like(Rc))
        sub_j = torch.where((b0 > 0) & (b0 < n - 1), (bm - bp) / (2 * bm - 4 * Rc + 2 * bp), torch.zeros_like(Rc))
        new_i = di.reshape(-1) + (a0 - m // 2) + sub_i
        new_j = dj.reshape(-1) + (b0 - n // 2) + sub_j
        nan = torch.full_like(new_i, float("nan"))
        di = torch.where(valid, new_i, nan).reshape(len(i_arr), len(j_arr))
        dj = torch.where(valid, new_j, nan).reshape(len(i_arr), len(j_arr))
        C = torch.where(valid, C, torch.zeros_like(C)).reshape(len(i_arr), len(j_arr))

        tol = 1e-3 if kk == len(winsize) - 1 else 0.1
        z = smoothn(torch.stack([di, dj]), torch.sqrt(C.clamp_min(0)), robust=True, tol_z=tol)
        di, dj = z[0], z[1]

    ii = np.clip(np.round(ic).astype(int), 0, M - 1)
    jj = np.clip(np.round(jc).astype(int), 0, N - 1)
    inroi = roi[torch.as_tensor(ii, device=dev)[:, None], torch.as_tensor(jj, device=dev)[None, :]]
    di = torch.where(inroi, di, torch.full_like(di, float("nan")))
    dj = torch.where(inroi, dj, torch.full_like(dj, float("nan")))
    j_grid, i_grid = np.meshgrid(jc, ic)
    return dj.cpu().numpy(), di.cpu().numpy(), j_grid, i_grid
