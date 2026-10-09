"""
Milestones 34-35: simus / simus3 con la física completa de MUST (equivalentes a pymust.simus y pymust.simus3).

A diferencia de simulator.Simus (modelo 2-D simplificado: sinc x exp(ikr)/sqrt(r)), estas funciones
reutilizan el mismo motor que field.pfield / field.pfield3 (que ya coincide con pymust) y le agregan
la recepción, exactamente como hace MUST:

    * factor de oblicuidad del baffle (soft = cos θ por defecto, rígido, o impedancia)
    * sub-elementos (element splitting) y directividad a fc o a todas las frecuencias
    * atenuación (dB/cm/MHz)
    * elevación 3-D (MGBM) si algún scatterer tiene y != 0     [simus]
    * propagación 3-D y elementos rectangulares width x height  [simus3]
    * radio mínimo c/fc/2 cerca de los elementos, scatterers "fuera" (z < 0 o dentro del convexo)
    * retardos de recepción, rejilla de frecuencias y umbral de -100 dB como SIMUS
    * pulso de n_cycles ciclos (PARAM.TXnow) o chirp lineal (freq_sweep = PARAM.TXfreqsweep)

    RF = simus(transducer, x, y, z, RC, tx_delays, tx_apodization, fs=None, ...)   -> [nt, N]
    RF = simus3(transducer, x, y, z, RC, tx_delays, tx_apodization, fs=None, ...)  -> [nt, N]

Todo es diferenciable respecto de RC, posiciones de los scatterers, retardos y apodización.
"""
import math
import numpy as np
import torch
from torchsimus.field import _as_tensor, _sinc, _EPS32, MGBM_A, MGBM_B, _frequency_grid
from torchsimus.pulse import getpulse
from torchsimus.spectra import pulse_spectrum_full, probe_spectrum


def _prep_delays(tx_delays, tx_apodization, N, dtype, device):
    D = _as_tensor(tx_delays, dtype, device)
    if D.ndim == 1:
        D = D[None, :]
    apod = torch.ones(N, dtype=dtype, device=device) if tx_apodization is None else _as_tensor(tx_apodization, dtype, device).reshape(-1)
    off = torch.isnan(D).any(0)
    apod = torch.where(off, torch.zeros_like(apod), apod)
    valid = D[~torch.isnan(D)]
    return torch.nan_to_num(D, nan=0.0), apod, valid


def _finish_rf(RFspec, keep, Nf, fs, fc):
    """Espectro [F, N] en las frecuencias conservadas -> RF en el tiempo, igual que SIMUS."""
    full = torch.zeros(Nf, RFspec.shape[1], dtype=RFspec.dtype, device=RFspec.device)
    full[keep] = RFspec
    nf = int(math.ceil(fs / 2 / fc * (Nf - 1)))
    RF = torch.fft.irfft(full.conj(), n=nf, dim=0)[: (nf + 1) // 2]
    rel = 1e-5                                                           # -100 dB: se anulan los valores muy pequeños
    w = 0.5 * (1 + torch.tanh((RF.abs() / RF.abs().max() - rel) / (rel / 10)))
    w = torch.round(w / (rel / 10)) * (rel / 10)
    return RF * w


def simus(transducer, x, y, z, RC, tx_delays, tx_apodization=None, *, fs=None, c=1540.0, attenuation=0.0,
          baffle="soft", n_cycles=1, element_splitting=None, full_frequency_directivity=False,
          db_thresh=-100.0, frequency_step=1.0, rx_delays=None, width=None, height=None,
          elevation_focus=None, freq_sweep=None, return_spectrum=False, dtype=torch.float64, device=None, budget=2 ** 24):
    """RF de un arreglo 1-D (lineal o convexo) con la física de pymust.simus. Devuelve RF [nt, N]."""
    if device is None:
        device = transducer.geometry.pose().centers.device
    fc, bw = float(transducer.fc), float(transducer.bandwidth)
    fs = 4 * fc if fs is None else float(fs)
    assert fs >= 4 * fc, "fs debe ser >= 4 fc"
    width = float(width if width is not None else transducer.element_shape.width)
    height = float(height if height is not None else getattr(transducer, "height", math.inf))
    Rf = float(elevation_focus if elevation_focus is not None else getattr(transducer, "elevation_focus", math.inf))

    X = _as_tensor(x, dtype, device).reshape(-1)
    Z = _as_tensor(z, dtype, device).reshape(-1)
    Y = torch.zeros_like(X) if y is None else _as_tensor(y, dtype, device).reshape(-1)
    RCt = _as_tensor(RC, dtype, device).reshape(-1)
    elevation = bool((Y.abs() >= 1e-9).any())
    if elevation and not math.isfinite(height):
        raise ValueError("Hay scatterers con y != 0: define height y elevation_focus del elemento")
    pts = torch.stack([X, Y, Z], -1)
    P = X.numel()

    pose = transducer.geometry.pose()
    centers, u, nrm = [t.to(device=device, dtype=dtype) for t in (pose.centers, pose.u, pose.n)]
    N = centers.shape[0]
    D, apod, valid = _prep_delays(tx_delays, tx_apodization, N, dtype, device)
    RX = torch.zeros(N, dtype=dtype, device=device) if rx_delays is None else _as_tensor(rx_delays, dtype, device).reshape(-1)

    # --- rejilla de frecuencias de SIMUS (evita aliasing temporal) ---
    dxz = torch.sqrt((X.detach()[:, None] - centers[None, :, 0]) ** 2 + (Z[:, None] - centers[None, :, 2]) ** 2)
    maxD = float(dxz.detach().max()) + float(getpulse(transducer, 2, n_cycles, freq_sweep)[1][-1]) * c
    df = 1 / 2 / (2 * maxD / c + float(torch.cat([valid, RX]).detach().max())) * frequency_step
    Nf = int(2 * math.ceil(fc / df) + 1)
    f_all, keep, df = _frequency_grid(fc, bw, n_cycles, df, db_thresh, torch.float64, device, freq_sweep)
    f = f_all[keep]
    F = f.numel()

    # --- sub-elementos ---
    if element_splitting is None:
        M = int(math.ceil(width / (c / (fc * (1 + bw / 2)))))
    else:
        M = int(element_splitting)
    seg = width / M
    s = -width / 2 + seg / 2 + torch.arange(M, dtype=dtype, device=device) * seg
    sub = centers[:, None, :] + s[None, :, None] * u[:, None, :]                            # [N, M, 3]
    the = torch.atan2(nrm[:, 0], nrm[:, 2])

    is_out = Z < 0
    if hasattr(transducer.geometry, "curvature_center"):
        c0 = transducer.geometry.curvature_center().to(device=device, dtype=dtype)
        is_out = is_out | (((pts - c0) ** 2).sum(-1) <= float(transducer.geometry.radius) ** 2)
    small_d = c / fc / 2

    ctype = torch.complex128 if dtype == torch.float64 else torch.complex64
    pulseS = pulse_spectrum_full(f, fc, n_cycles, freq_sweep).to(ctype)
    probeS = probe_spectrum(f, fc, bw).to(ctype)
    kw = 2 * math.pi * f / c
    kwa = attenuation / 8.69 * f / 1e6 * 1e2
    dkw, dkwa = 2 * math.pi * df / c, attenuation / 8.69 * df / 1e6 * 1e2
    kc = 2 * math.pi * fc / c
    delapod = (torch.exp(1j * (2 * math.pi * f.to(dtype))[:, None, None] * D[None]).sum(1) * apod[None]).to(ctype)   # [F, N]
    if elevation:
        Nm = max(3, int(np.round(F / 20)))
        k4set = set((np.round(np.linspace(1, F, Nm)).astype(int) - 1).tolist())

    SPECT = torch.zeros(F, N, dtype=ctype, device=device)
    pc = max(1, int(budget // (N * M)))
    for i in range(0, P, pc):
        p = pts[i:i + pc]
        d = p[:, None, None, :] - sub[None]                                                  # [p, N, M, 3]
        d2xz = torch.sqrt(d[..., 0] ** 2 + d[..., 2] ** 2)
        r = torch.linalg.vector_norm(d, dim=-1).clamp_min(small_d)
        th = torch.asin(((d[..., 0] + _EPS32) / (d2xz + _EPS32)).clamp(-1, 1)) - the[None, :, None]
        sin_t = torch.sin(th)
        cos_t = torch.where(th.abs() >= math.pi / 2, torch.zeros_like(th), torch.cos(th))
        obli = torch.ones_like(cos_t) if baffle == "rigid" else (cos_t if baffle == "soft" else cos_t / (cos_t + float(baffle)))
        obli = torch.where(cos_t <= 0, torch.full_like(obli, _EPS32), obli)
        amp = obli / (r if elevation else torch.sqrt(r))
        if not full_frequency_directivity:
            amp = amp * _sinc(kc * seg / 2 * sin_t)
        if elevation:
            rm = r.mean(-1)
            yy = p[:, 1:2]
            alpha = 0.5j * (1 / Rf - 1 / rm)
            gamma = 0.5j * yy ** 2 / rm
            beta2 = -(yy / rm) ** 2
        rc_i = RCt[i:i + pc].to(ctype)
        out_i = is_out[i:i + pc]
        mg = None
        for kk in range(F):
            if kk % 32 == 0:                                                                 # re-anclaje: evita deriva numérica
                E = torch.exp((-float(kwa[kk]) + 1j * float(kw[kk])) * r) * amp
                Edf = torch.exp((-dkwa + 1j * dkw) * r)
            else:
                E = E * Edf
            if full_frequency_directivity:
                RPmono = (E * _sinc(float(kw[kk]) * seg / 2 * sin_t)).mean(-1)
            else:
                RPmono = E.mean(-1)                                                          # [p, N]
            if elevation:
                if mg is None or kk in k4set:
                    k_ = float(kw[kk]); mg = 0
                    for A_, B_ in zip(MGBM_A, MGBM_B):
                        tmp = 1 / (k_ * alpha + B_ / height ** 2)
                        mg = mg + A_ * torch.sqrt(math.pi * tmp) * torch.exp(k_ ** 2 * beta2 / 4 * tmp + k_ * gamma)
                    mg = mg.to(ctype)
                RPmono = RPmono * mg
            RPk = (RPmono @ delapod[kk]) * pulseS[kk] * probeS[kk]                           # presión en cada scatterer
            RPk = torch.where(out_i, torch.zeros_like(RPk), RPk)
            SPECT[kk] = SPECT[kk] + probeS[kk] * ((RPk * rc_i) @ RPmono)                    # recepción (reciprocidad)
    if bool((RX != 0).any()):
        SPECT = SPECT * torch.exp(1j * kw[:, None] * c * RX[None])
    SPECT = SPECT * (df * (1.0 if elevation else width))
    RF = _finish_rf(SPECT, keep, Nf, fs, fc)
    return (RF, SPECT, f) if return_spectrum else RF


def simus3(transducer, x, y, z, RC, tx_delays, tx_apodization=None, *, fs=None, c=1540.0, attenuation=0.0,
           baffle="soft", n_cycles=1, element_splitting=None, full_frequency_directivity=False,
           db_thresh=-100.0, frequency_step=1.0, rx_delays=None, width=None, height=None,
           freq_sweep=None, return_spectrum=False, dtype=torch.float64, device=None, budget=2 ** 24):
    """RF de un arreglo plano 2-D (p. ej. matricial) con la física de pymust.simus3. Devuelve RF [nt, N]."""
    if device is None:
        device = transducer.geometry.pose().centers.device
    fc, bw = float(transducer.fc), float(transducer.bandwidth)
    fs = 4 * fc if fs is None else float(fs)
    width = float(width if width is not None else transducer.element_shape.width)
    height = float(height if height is not None else getattr(transducer.element_shape, "height", getattr(transducer, "height", None)))

    X, Y, Z = [_as_tensor(a, dtype, device).reshape(-1) for a in (x, y, z)]
    RCt = _as_tensor(RC, dtype, device).reshape(-1)
    pts = torch.stack([X, Y, Z], -1)
    P = X.numel()
    pose = transducer.geometry.pose()
    centers, u, v, nrm = [t.to(device=device, dtype=dtype) for t in (pose.centers, pose.u, pose.v, pose.n)]
    N = centers.shape[0]
    D, apod, _ = _prep_delays(tx_delays, tx_apodization, N, dtype, device)
    RX = torch.zeros(N, dtype=dtype, device=device) if rx_delays is None else _as_tensor(rx_delays, dtype, device).reshape(-1)

    d3 = torch.cdist(pts.detach(), centers.detach())
    maxD = float(d3.detach().max()) + float(getpulse(transducer, 2, n_cycles, freq_sweep)[1][-1]) * c
    df = 1 / 2 / (2 * maxD / c + float((D + RX[None]).detach().max())) * frequency_step            # como simus3 de pymust
    Nf = int(2 * math.ceil(fc / df) + 1)
    f_all, keep, df = _frequency_grid(fc, bw, n_cycles, df, db_thresh, torch.float64, device, freq_sweep)
    f = f_all[keep]
    F = f.numel()

    if element_splitting is None:
        lam = c / (fc * (1 + bw / 2))
        Mu, Mv = int(math.ceil(width / lam)), int(math.ceil(height / lam))
    else:
        Mu, Mv = int(element_splitting[0]), int(element_splitting[1])
    segw, segh = width / Mu, height / Mv
    su = -width / 2 + segw / 2 + torch.arange(Mu, dtype=dtype, device=device) * segw
    sv = -height / 2 + segh / 2 + torch.arange(Mv, dtype=dtype, device=device) * segh
    su, sv = [t.reshape(-1) for t in torch.meshgrid(su, sv, indexing="ij")]
    M = su.numel()
    sub = centers[:, None, :] + su[None, :, None] * u[:, None, :] + sv[None, :, None] * v[:, None, :]
    is_out = Z < 0
    small_d = c / fc / 2

    ctype = torch.complex128 if dtype == torch.float64 else torch.complex64
    pulseS = pulse_spectrum_full(f, fc, n_cycles, freq_sweep).to(ctype)
    probeS = probe_spectrum(f, fc, bw).to(ctype)
    kw = 2 * math.pi * f / c
    kwa = attenuation / 8.69 * f / 1e6 * 1e2
    dkw, dkwa = 2 * math.pi * df / c, attenuation / 8.69 * df / 1e6 * 1e2
    kc = 2 * math.pi * fc / c
    delapod = (torch.exp(1j * (2 * math.pi * f.to(dtype))[:, None, None] * D[None]).sum(1) * apod[None]).to(ctype)

    SPECT = torch.zeros(F, N, dtype=ctype, device=device)
    pc = max(1, int(budget // (N * M)))
    for i in range(0, P, pc):
        p = pts[i:i + pc]
        d = p[:, None, None, :] - sub[None]
        xl = (d * u[None, :, None, :]).sum(-1); yl = (d * v[None, :, None, :]).sum(-1); zl = (d * nrm[None, :, None, :]).sum(-1)
        r0 = torch.linalg.vector_norm(d, dim=-1)
        rho = torch.sqrt(xl ** 2 + yl ** 2)
        cos_t = (zl + _EPS32) / (r0 + _EPS32)
        sin_t = (rho + _EPS32) / (r0 + _EPS32)
        sx = sin_t * (xl + _EPS32) / (rho + _EPS32)
        sy = sin_t * (yl + _EPS32) / (rho + _EPS32)
        r = r0.clamp_min(small_d)
        obli = torch.ones_like(cos_t) if baffle == "rigid" else (cos_t if baffle == "soft" else cos_t / (cos_t + float(baffle)))
        amp = obli / r
        if not full_frequency_directivity:
            amp = amp * _sinc(kc * segw / 2 * sx) * _sinc(kc * segh / 2 * sy)
        rc_i = RCt[i:i + pc].to(ctype)
        out_i = is_out[i:i + pc]
        for kk in range(F):
            if kk % 32 == 0:
                E = torch.exp((-float(kwa[kk]) + 1j * float(kw[kk])) * r) * amp
                Edf = torch.exp((-dkwa + 1j * dkw) * r)
            else:
                E = E * Edf
            if full_frequency_directivity:
                k_ = float(kw[kk])
                RPmono = (E * _sinc(k_ * segw / 2 * sx) * _sinc(k_ * segh / 2 * sy)).mean(-1)
            else:
                RPmono = E.mean(-1)
            RPk = (RPmono @ delapod[kk]) * pulseS[kk] * probeS[kk]
            RPk = torch.where(out_i, torch.zeros_like(RPk), RPk)
            SPECT[kk] = SPECT[kk] + probeS[kk] * ((RPk * rc_i) @ RPmono)
    if bool((RX != 0).any()):
        SPECT = SPECT * torch.exp(1j * kw[:, None] * c * RX[None])
    # Nota: a diferencia de simus, MUST/pymust NO multiplican el espectro por df en PFIELD3/SIMUS3
    # (la línea "SPECT = SPECT*CorFac" está comentada en MUST). Se replica para coincidir con MUST;
    # la consecuencia es que la amplitud absoluta de simus3 depende del paso de frecuencia (tamaño de la escena).
    RF = _finish_rf(SPECT, keep, Nf, fs, fc)
    return (RF, SPECT, f) if return_spectrum else RF
