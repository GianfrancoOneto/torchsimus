"""
Milestone 36: animación de la propagación de onda (equivalente a pymust.mkmovie).

El campo de presión se calcula en el dominio de la frecuencia con field.pfield (con un paso
de frecuencia fijo que cubre todo el tiempo de vuelo de la ROI) y se pasa al tiempo con una
IFFT. Cada cuadro es la presión instantánea en la grilla (x, z).

También acepta scatterers (película con los ecos), n_cycles y freq_sweep (chirp).
"""
import math
import numpy as np
import torch
from torchsimus.field import pfield


def mkmovie(transducer, tx_delays, tx_apodization=None, *, movie=None, scatterers=None, c=1540.0,
            attenuation=0.0, frequency_step=1.0, db_thresh=-60.0, gamma=1.0, n_cycles=1,
            freq_sweep=None, gif_path=None, duration=15.0, fps=10.0, dtype=torch.float32,
            device=None, budget=2 ** 24):
    """
    movie : [ancho_cm, alto_cm, pix_por_cm] (como PARAM.movie de MUST; por defecto 200L x 200L, 50 pix/cm).
            También acepta [ancho_cm, alto_cm, pix_por_cm, duración_s, fps] (reemplaza duration y fps).
    scatterers : None o (x, z, RC). Cada scatterer re-emite el campo que recibe y su eco se propaga
                 a toda la grilla (sintaxis MKMOVIE(x, z, RC, delaysTX, param) de MUST).
    n_cycles, freq_sweep : pulso de n ciclos o chirp lineal, como en field.pfield.
    Devuelve F (uint8, [nz, nx, n_cuadros]) e info (Xgrid, Zgrid en m, TimeStep en s, duration, fps).
    Si gif_path no es None, guarda la animación como GIF.
    """
    pose = transducer.geometry.pose()
    L = float(pose.centers[:, 0].max() - pose.centers[:, 0].min())
    if movie is None:
        movie = [200 * L, 200 * L, 50]
    movie = list(movie)
    if len(movie) >= 4:
        duration = float(movie[3])
    if len(movie) >= 5:
        fps = float(movie[4])
    roi_w, roi_h, pix = movie[0] * 1e-2, movie[1] * 1e-2, 1 / movie[2] * 1e-2
    xi = np.arange(pix / 2, roi_w + pix / 2, pix)
    zi = np.arange(pix / 2, roi_h + pix / 2, pix)
    xi, zi = np.meshgrid(xi - np.mean(xi), zi)

    fc = float(transducer.fc)
    maxD = math.hypot((roi_w + L) / 2, roi_h)                       # distancia máxima recorrida
    df = 1 / 2 / (maxD / c) * frequency_step
    Nf = int(2 * math.ceil(fc / df) + 1)

    kw = dict(c=c, attenuation=attenuation, element_splitting=1, db_thresh=db_thresh, df=df,
              n_cycles=n_cycles, freq_sweep=freq_sweep, return_spectrum=True, dtype=dtype,
              device=device, budget=budget)
    _, S, f, keep = pfield(transducer, xi, None, zi, tx_delays, tx_apodization, **kw)   # campo incidente [nz, nx, F]
    if scatterers is not None:
        xs, zs, rc = [np.ravel(np.asarray(v, dtype=float)) for v in scatterers]
        _, Ss, _, _ = pfield(transducer, xs, None, zs, tx_delays, tx_apodization, **kw)  # campo en los scatterers [S, F]
        dev, rdt = S.device, (torch.float64 if S.dtype == torch.complex128 else torch.float32)
        X = torch.as_tensor(xi.reshape(-1), dtype=rdt, device=dev)
        Z = torch.as_tensor(zi.reshape(-1), dtype=rdt, device=dev)
        xs_, zs_ = torch.as_tensor(xs, dtype=rdt, device=dev), torch.as_tensor(zs, dtype=rdt, device=dev)
        r = torch.sqrt((X[:, None] - xs_[None]) ** 2 + (Z[:, None] - zs_[None]) ** 2)   # [P, S]
        kw_ = 2 * math.pi * f / c
        kwa = attenuation / 8.69 * f / 1e6 * 1e2
        src = Ss * torch.as_tensor(rc, device=dev).to(S.dtype)[:, None]                  # [S, F]
        eco = torch.stack([torch.exp((-float(kwa[k]) + 1j * float(kw_[k])) * r).to(S.dtype) @ src[:, k]
                           for k in range(f.numel())], -1)
        S = S + eco.reshape(*xi.shape, -1)
    full = torch.zeros(*xi.shape, Nf, dtype=S.dtype, device=S.device)
    full[..., keep] = S                                             # espectro en las Nf frecuencias
    F = torch.fft.irfft(full, dim=-1)                               # [nz, nx, 2(Nf-1)]
    F = torch.flip(F, dims=[-1])
    F = F[..., : int(round(F.shape[-1] // 2))]
    F = F / F.abs().max()
    if gamma != 1:
        F = torch.sign(F) * F.abs() ** gamma
    F = ((F + 1) / 2 * 255).to(torch.uint8).cpu().numpy()

    info = {"Xgrid": xi[0, :], "Zgrid": zi[:, 0], "TimeStep": maxD / c / F.shape[2],
            "duration": duration, "fps": fps}
    if gif_path is not None:
        save_gif(F, info, gif_path, duration=duration, fps=fps)
    return F, info


def save_gif(F, info, gif_path, duration=15.0, fps=10.0):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    vals = np.array([matplotlib.cm.hot(2 * i) for i in range(127)] + [matplotlib.cm.hot(2 * i) for i in range(128)])
    vals[:127, :3] = 1 - vals[:127, :3]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("hot2", vals)
    nk = max(1, int(round(F.shape[2] / (duration * fps))))
    ks = np.arange(0, F.shape[2], nk)
    ext = [info["Xgrid"][0] * 1e2, info["Xgrid"][-1] * 1e2, info["Zgrid"][-1] * 1e2, info["Zgrid"][0] * 1e2]
    fig, ax = plt.subplots()
    def animate(i):
        ax.clear()
        im = ax.imshow(F[:, :, ks[i]], cmap=cmap, extent=ext, vmin=0, vmax=255)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("z (cm)")
        return im,
    FuncAnimation(fig, animate, frames=len(ks), blit=True, repeat=False).save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
