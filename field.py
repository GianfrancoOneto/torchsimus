"""
Campo de presión acústica (Milestones 34-35): equivalentes en PyTorch de PFIELD y PFIELD3 de MUST.

    pfield(...)   -> arreglos 1-D (lineales o convexos). Modelo 2-D (y = 0) o 3-D con
                     enfoque en elevación (MGBM) si algún punto tiene y != 0.
    pfield3(...)  -> arreglos planos 2-D (matriciales o cualquier CustomArray), elementos
                     rectangulares width x height, propagación 3-D.

Ambos devuelven el campo de presión RMS en cada punto (misma forma que x), y opcionalmente
el espectro complejo [..., F] y las frecuencias usadas. Todo es diferenciable (retardos,
apodización, posiciones de los puntos).
"""
import math
import numpy as np
import torch
from spectra import pulse_spectrum, probe_spectrum

# Coeficientes MGBM (multi-Gaussian beam model) de 4 términos, los mismos que usa MUST
_MGBM_A = [0.187 + 0.275j, 0.288 - 1.954j]
_MGBM_B = [4.558 - 25.59j, 8.598 - 7.924j]
MGBM_A = _MGBM_A + [complex(a).conjugate() for a in _MGBM_A]
MGBM_B = _MGBM_B + [complex(b).conjugate() for b in _MGBM_B]

_EPS32 = float(np.finfo(np.float32).eps)


def _sinc(x):
    # sin(|x|)/|x| (convención de MUST, sin pi)
    ax = x.abs() + 1e-16
    return torch.sin(ax) / ax


def _as_tensor(a, dtype, device):
    if torch.is_tensor(a):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)


def _prepare_tx(tx_delays, tx_apodization, N, dtype, device):
    """Retardos [E, N] (NaN = elemento apagado) y apodización [N]. Igual que MUST."""
    D = _as_tensor(tx_delays, dtype, device)
    if D.ndim == 1:
        D = D[None, :]
    if D.shape[-1] != N:
        raise ValueError(f"tx_delays debe tener {N} columnas (una por elemento); tiene {D.shape[-1]}")
    if tx_apodization is None:
        apod = torch.ones(N, dtype=dtype, device=device)
    else:
        apod = _as_tensor(tx_apodization, dtype, device).reshape(-1)
    off = torch.isnan(D).any(dim=0)
    apod = torch.where(off, torch.zeros_like(apod), apod)
    D = torch.nan_to_num(D, nan=0.0)
    if (D < 0).any():
        raise ValueError("Los retardos deben ser >= 0")
    return D, apod


def _frequency_grid(fc, bandwidth, n_cycles, df, db_thresh, dtype, device):
    """f = linspace(0, 2fc, Nf) y se conserva el rango contiguo donde |pulso x sonda| > db_thresh."""
    Nf = int(2 * math.ceil(fc / df) + 1)
    f_all = torch.linspace(0, 2 * fc, Nf, dtype=torch.float64, device=device)
    S = (pulse_spectrum(f_all, fc, n_cycles) * probe_spectrum(f_all, fc, bandwidth)).abs()
    GdB = 20 * torch.log10(1e-200 + S / S.max())
    ids = torch.nonzero(GdB > db_thresh).flatten()
    keep = torch.zeros(Nf, dtype=torch.bool, device=device)
    keep[ids[0]:ids[-1] + 1] = True
    df_eff = float(f_all[1])
    return f_all, keep, df_eff


def _point_chunk(n_elem_sub, n_freq_chunk, budget):
    return max(1, int(budget // max(1, n_elem_sub * n_freq_chunk)))


# ---------------------------------------------------------------------------------------------
# PFIELD (arreglos 1-D: lineales o convexos)
# ---------------------------------------------------------------------------------------------
def pfield(transducer, x, y, z, tx_delays, tx_apodization=None, *,
           c=1540.0, attenuation=0.0, width=None, height=None, elevation_focus=None,
           baffle="soft", n_cycles=1, element_splitting=None, db_thresh=-60.0,
           frequency_step=1.0, full_frequency_directivity=False, df=None,
           return_spectrum=False, dtype=torch.float32, device=None, budget=2 ** 24):
    """
    Campo de presión RMS de un arreglo 1-D (equivalente a pymust.pfield).

    transducer : Transducer de torchsimus (geometry con pose(), element_shape con .width,
                 fc en Hz y bandwidth FRACCIONAL, p. ej. 0.74).
    x, y, z    : coordenadas (m) de los puntos, cualquier forma (y puede ser None -> 0).
    tx_delays  : [E, N] o [N] en s. Varias filas = emisiones SIMULTÁNEAS (MLT). NaN = elemento apagado.
    tx_apodization : [N] (por defecto 1).
    attenuation: dB/cm/MHz.   height / elevation_focus: altura y foco en elevación (m) de los
                 elementos, solo se usan si algún y != 0 (modelo 3-D).
    df         : paso de frecuencia fijo (lo usa mkmovie). Si es None se elige como MUST.
    Devuelve RP (misma forma que x) y, si return_spectrum=True, (RP, SPECT[..., F], f).
    """
    if device is None:
        device = transducer.geometry.pose().centers.device
    fc = float(transducer.fc)
    bw = float(transducer.bandwidth)
    if width is None:
        width = getattr(transducer.element_shape, "width", None)
        if width is None:
            raise ValueError("pfield necesita el ancho del elemento (element_shape.width o width=...)")
    width = float(width)
    if height is None:
        height = getattr(transducer, "height", math.inf)
    if elevation_focus is None:
        elevation_focus = getattr(transducer, "elevation_focus", math.inf)
    height, Rf = float(height), float(elevation_focus)

    X = _as_tensor(x, dtype, device)
    shape = X.shape
    X = X.reshape(-1)
    Z = _as_tensor(z, dtype, device).reshape(-1)
    Y = torch.zeros_like(X) if y is None else _as_tensor(y, dtype, device).reshape(-1)
    elevation = bool((Y.abs() >= 1e-9).any())
    if not elevation:
        Y = torch.zeros_like(X)
    elif not math.isfinite(height):
        raise ValueError("Hay puntos con y != 0: define height (y elevation_focus) del elemento")
    P = X.numel()
    pts = torch.stack([X, Y, Z], dim=-1)                                  # [P, 3]

    pose = transducer.geometry.pose()
    centers = pose.centers.to(device=device, dtype=dtype)
    u = pose.u.to(device=device, dtype=dtype)
    nrm = pose.n.to(device=device, dtype=dtype)
    N = centers.shape[0]
    D, apod = _prepare_tx(tx_delays, tx_apodization, N, dtype, device)

    # sub-elementos a lo largo de u (element splitting)
    if element_splitting is None:
        lambda_min = c / (fc * (1 + bw / 2))
        M = int(math.ceil(width / lambda_min))
    else:
        M = int(element_splitting)
    seg = width / M
    s = -width / 2 + seg / 2 + torch.arange(M, dtype=dtype, device=device) * seg          # [M]
    sub = centers[:, None, :] + s[None, :, None] * u[:, None, :]                            # [N, M, 3]

    # puntos "fuera": detrás del arreglo (z < 0) o dentro del círculo de un arreglo convexo
    is_out = Z < 0
    if hasattr(transducer.geometry, "curvature_center"):
        c0 = transducer.geometry.curvature_center().to(device=device, dtype=dtype)
        R = float(transducer.geometry.radius)
        is_out = is_out | (((pts - c0) ** 2).sum(-1) <= R ** 2)

    small_d = c / fc / 2

    the = torch.atan2(nrm[:, 0], nrm[:, 2])                                                 # ángulo de cada elemento (convexo)

    def geom(p):
        d = p[:, None, None, :] - sub[None]                                                  # [p, N, M, 3]
        dxz = torch.sqrt(d[..., 0] ** 2 + d[..., 2] ** 2)
        r = torch.linalg.vector_norm(d, dim=-1).clamp_min(small_d)
        # ángulo respecto a la normal del elemento, en el plano xz (misma fórmula que MUST)
        th = torch.asin(((d[..., 0] + _EPS32) / (dxz + _EPS32)).clamp(-1.0, 1.0)) - the[None, :, None]
        sin_t = torch.sin(th)
        cos_t = torch.where(th.abs() >= math.pi / 2, torch.zeros_like(th), torch.cos(th))
        return r, sin_t, cos_t

    # paso de frecuencia (como MUST: 1 / (máx. tiempo de vuelo + máx. retardo))
    if df is None:
        rmax = 0.0
        pc = _point_chunk(N * M, 1, budget)
        with torch.no_grad():
            for i in range(0, P, pc):
                rmax = max(rmax, float(geom(pts[i:i + pc])[0].max()))
        df = frequency_step / (rmax / c + float(D.detach().max()))
    f_all, keep, df = _frequency_grid(fc, bw, n_cycles, df, db_thresh, torch.float64, device)
    f = f_all[keep]
    F = f.numel()

    spect = (pulse_spectrum(f, fc, n_cycles) * probe_spectrum(f, fc, bw)).to(torch.complex128)
    kw = (2 * math.pi * f / c)                                                              # [F]
    kwa = attenuation / 8.69 * f / 1e6 * 1e2                                                # [F] Np/m
    ctype = torch.complex64 if dtype == torch.float32 else torch.complex128
    # DELAPOD[f, n] = sum_e exp(i w tau_en) * apod_n
    delapod = (torch.exp(1j * (2 * math.pi * f.to(dtype))[:, None, None] * D[None]).sum(1) * apod[None]).to(ctype)

    # MGBM se actualiza solo en Nmgbm frecuencias (igual que MUST) y se mantiene entre ellas
    if elevation:
        Nm = max(3, int(np.round(F / 20)))
        k4set = set((np.round(np.linspace(1, F, Nm)).astype(int) - 1).tolist())
    delapod_rep = delapod.repeat_interleave(M, dim=1) / M                                     # [F, N*M]
    kc = 2 * math.pi * fc / c
    dkw = 2 * math.pi * df / c
    dkwa = attenuation / 8.69 * df / 1e6 * 1e2
    reanchor = 32

    fchunk = 1
    RP2 = torch.zeros(P, dtype=torch.float64, device=device)
    SPECT = torch.zeros(P, F, dtype=ctype, device=device) if return_spectrum else None
    pc = _point_chunk(N * M, fchunk, budget)
    for i in range(0, P, pc):
        p = pts[i:i + pc]
        r, sin_t, cos_t = geom(p)
        obli = torch.ones_like(cos_t) if baffle == "rigid" else (
            cos_t if baffle == "soft" else cos_t / (cos_t + float(baffle)))
        obli = torch.where(cos_t <= 0, torch.full_like(obli, _EPS32), obli)
        amp = obli / (r if elevation else torch.sqrt(r))
        if not full_frequency_directivity:
            amp = amp * _sinc(kc * seg / 2 * sin_t)
        if elevation:
            rm = r.mean(-1)                                                                  # [p, N]
            yy = p[:, 1:2]
            alpha = 0.5j * (1 / Rf - 1 / rm)
            gamma = 0.5j * yy ** 2 / rm
            beta2 = -(yy / rm) ** 2
        out_i = is_out[i:i + pc]
        z_df = (-dkwa + 1j * dkw) * r                                                       # paso de frecuencia
        mg = None
        for kk in range(F):
            if kk % reanchor == 0:                                                           # exp directa (evita deriva numérica)
                E = torch.exp((-float(kwa[kk]) + 1j * float(kw[kk])) * r) * amp
                Edf = torch.exp(z_df)
            else:
                E = E * Edf
            if not elevation and not full_frequency_directivity:
                # vía rápida: el promedio sobre sub-elementos se funde con la suma sobre elementos
                RPk = (E.reshape(E.shape[0], -1) @ delapod_rep[kk]) * spect[kk].to(ctype)
                RPk = torch.where(out_i, torch.zeros_like(RPk), RPk)
                RP2[i:i + pc] = RP2[i:i + pc] + (RPk.abs() ** 2).to(torch.float64)
                if return_spectrum:
                    SPECT[i:i + pc, kk] = RPk
                continue
            if full_frequency_directivity:
                RPmono = (E * _sinc(float(kw[kk]) * seg / 2 * sin_t)).mean(-1)
            else:
                RPmono = E.mean(-1)                                                          # [p, N]
            if elevation:
                if mg is None or (kk in k4set):
                    kmg = float(kw[kk])
                    mg = 0
                    for A_, B_ in zip(MGBM_A, MGBM_B):
                        tmp = 1 / (kmg * alpha + B_ / height ** 2)
                        mg = mg + A_ * torch.sqrt(math.pi * tmp) * torch.exp(kmg ** 2 * beta2 / 4 * tmp + kmg * gamma)
                    mg = mg.to(ctype)
                RPmono = RPmono * mg
            RPk = (RPmono @ delapod[kk]) * spect[kk].to(ctype)                               # [p]
            RPk = torch.where(out_i, torch.zeros_like(RPk), RPk)
            RP2[i:i + pc] = RP2[i:i + pc] + (RPk.abs() ** 2).to(torch.float64)
            if return_spectrum:
                SPECT[i:i + pc, kk] = RPk

    cor = df * (1.0 if elevation else width)
    RP = torch.sqrt(RP2 * cor).to(dtype).reshape(shape)
    if return_spectrum:
        return RP, (SPECT * cor).reshape(*shape, F), f, keep
    return RP


# ---------------------------------------------------------------------------------------------
# PFIELD3 (arreglos planos 2-D: matriciales, o cualquier arreglo descrito con CustomArray)
# ---------------------------------------------------------------------------------------------
def pfield3(transducer, x, y, z, tx_delays, tx_apodization=None, *,
            c=1540.0, attenuation=0.0, width=None, height=None, baffle="soft", n_cycles=1,
            element_splitting=None, db_thresh=-60.0, frequency_step=1.0,
            full_frequency_directivity=False, return_spectrum=False,
            dtype=torch.float32, device=None, budget=2 ** 24):
    """
    Campo de presión RMS 3-D de un arreglo plano de elementos rectangulares width x height
    (equivalente a pymust.pfield3). La orientación de cada elemento viene de pose():
    width va a lo largo de u, height a lo largo de v, y n es la normal.
    element_splitting: None (automático, como MUST) o (Mu, Mv).
    """
    if device is None:
        device = transducer.geometry.pose().centers.device
    fc = float(transducer.fc)
    bw = float(transducer.bandwidth)
    if width is None:
        width = getattr(transducer.element_shape, "width", None)
    if height is None:
        height = getattr(transducer.element_shape, "height", getattr(transducer, "height", None))
    if width is None or height is None:
        raise ValueError("pfield3 necesita width y height de los elementos")
    width, height = float(width), float(height)

    X = _as_tensor(x, dtype, device)
    shape = X.shape
    X = X.reshape(-1)
    Y = _as_tensor(y, dtype, device).reshape(-1)
    Z = _as_tensor(z, dtype, device).reshape(-1)
    P = X.numel()
    pts = torch.stack([X, Y, Z], dim=-1)

    pose = transducer.geometry.pose()
    centers = pose.centers.to(device=device, dtype=dtype)
    u = pose.u.to(device=device, dtype=dtype)
    v = pose.v.to(device=device, dtype=dtype)
    nrm = pose.n.to(device=device, dtype=dtype)
    N = centers.shape[0]
    D, apod = _prepare_tx(tx_delays, tx_apodization, N, dtype, device)

    if element_splitting is None:
        lambda_min = c / (fc * (1 + bw / 2))
        Mu, Mv = int(math.ceil(width / lambda_min)), int(math.ceil(height / lambda_min))
    else:
        Mu, Mv = int(element_splitting[0]), int(element_splitting[1])
    segw, segh = width / Mu, height / Mv
    su = -width / 2 + segw / 2 + torch.arange(Mu, dtype=dtype, device=device) * segw
    sv = -height / 2 + segh / 2 + torch.arange(Mv, dtype=dtype, device=device) * segh
    su, sv = torch.meshgrid(su, sv, indexing="ij")
    su, sv = su.reshape(-1), sv.reshape(-1)
    M = su.numel()
    sub = centers[:, None, :] + su[None, :, None] * u[:, None, :] + sv[None, :, None] * v[:, None, :]  # [N, M, 3]

    is_out = Z < 0
    small_d = c / fc / 2

    def geom(p):
        d = p[:, None, None, :] - sub[None]
        xl = (d * u[None, :, None, :]).sum(-1)
        yl = (d * v[None, :, None, :]).sum(-1)
        zl = (d * nrm[None, :, None, :]).sum(-1)
        r0 = torch.linalg.vector_norm(d, dim=-1)
        rho = torch.sqrt(xl ** 2 + yl ** 2)
        cos_t = (zl + _EPS32) / (r0 + _EPS32)
        sin_t = (rho + _EPS32) / (r0 + _EPS32)
        cos_p = (xl + _EPS32) / (rho + _EPS32)
        sin_p = (yl + _EPS32) / (rho + _EPS32)
        return r0.clamp_min(small_d), cos_t, sin_t * cos_p, sin_t * sin_p

    rmax = 0.0
    pc = _point_chunk(N * M, 1, budget)
    with torch.no_grad():
        for i in range(0, P, pc):
            rmax = max(rmax, float(geom(pts[i:i + pc])[0].max()))
    df = frequency_step / (rmax / c + float(D.detach().max()))
    f_all, keep, df = _frequency_grid(fc, bw, n_cycles, df, db_thresh, torch.float64, device)
    f = f_all[keep]
    F = f.numel()

    ctype = torch.complex64 if dtype == torch.float32 else torch.complex128
    spect = (pulse_spectrum(f, fc, n_cycles) * probe_spectrum(f, fc, bw)).to(ctype)
    kw = 2 * math.pi * f / c
    kwa = attenuation / 8.69 * f / 1e6 * 1e2
    delapod = (torch.exp(1j * (2 * math.pi * f.to(dtype))[:, None, None] * D[None]).sum(1) * apod[None]).to(ctype)
    delapod_rep = delapod.repeat_interleave(M, dim=1) / M                                     # [F, N*M]
    kc = 2 * math.pi * fc / c
    dkw = 2 * math.pi * df / c
    dkwa = attenuation / 8.69 * df / 1e6 * 1e2
    reanchor = 32

    fchunk = 1
    RP2 = torch.zeros(P, dtype=torch.float64, device=device)
    SPECT = torch.zeros(P, F, dtype=ctype, device=device) if return_spectrum else None
    pc = _point_chunk(N * M, fchunk, budget)
    for i in range(0, P, pc):
        p = pts[i:i + pc]
        r, cos_t, sx, sy = geom(p)
        obli = torch.ones_like(cos_t) if baffle == "rigid" else (
            cos_t if baffle == "soft" else cos_t / (cos_t + float(baffle)))
        amp = obli / r
        if not full_frequency_directivity:
            amp = amp * _sinc(kc * segw / 2 * sx) * _sinc(kc * segh / 2 * sy)
        out_i = is_out[i:i + pc]
        z_df = (-dkwa + 1j * dkw) * r
        for kk in range(F):
            if kk % reanchor == 0:
                E = torch.exp((-float(kwa[kk]) + 1j * float(kw[kk])) * r) * amp
                Edf = torch.exp(z_df)
            else:
                E = E * Edf
            if full_frequency_directivity:
                k_ = float(kw[kk])
                RPk = ((E * _sinc(k_ * segw / 2 * sx) * _sinc(k_ * segh / 2 * sy)).mean(-1) @ delapod[kk]) * spect[kk]
            else:
                RPk = (E.reshape(E.shape[0], -1) @ delapod_rep[kk]) * spect[kk]
            RPk = torch.where(out_i, torch.zeros_like(RPk), RPk)
            RP2[i:i + pc] = RP2[i:i + pc] + (RPk.abs() ** 2).to(torch.float64)
            if return_spectrum:
                SPECT[i:i + pc, kk] = RPk

    RP = torch.sqrt(RP2 * df).to(dtype).reshape(shape)
    if return_spectrum:
        return RP, (SPECT * df).reshape(*shape, F), f, keep
    return RP
