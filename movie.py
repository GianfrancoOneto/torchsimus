"""
Milestone 36: animación de la propagación de onda (equivalente a pymust.mkmovie).

El campo de presión se calcula en el dominio de la frecuencia con field.pfield (con un paso
de frecuencia fijo que cubre todo el tiempo de vuelo de la ROI) y se pasa al tiempo con una
IFFT. Cada cuadro es la presión instantánea en la grilla (x, z).
"""
import math
import numpy as np
import torch
from field import pfield


def mkmovie(transducer, tx_delays, tx_apodization=None, *, movie=None, c=1540.0,
            attenuation=0.0, frequency_step=1.0, db_thresh=-60.0, gamma=1.0,
            gif_path=None, duration=15.0, fps=10.0, dtype=torch.float32, device=None,
            budget=2 ** 24):
    """
    movie : [ancho_cm, alto_cm, pix_por_cm] (como PARAM.movie de MUST; por defecto 200L x 200L, 50 pix/cm).
    Devuelve F (uint8, [nz, nx, n_cuadros]) e info (Xgrid, Zgrid en m, TimeStep en s).
    Si gif_path no es None, guarda la animación como GIF (duration s, fps cuadros/s).
    """
    pose = transducer.geometry.pose()
    N = pose.centers.shape[0]
    L = float(pose.centers[:, 0].max() - pose.centers[:, 0].min())
    if movie is None:
        movie = [200 * L, 200 * L, 50]
    roi_w, roi_h, pix = movie[0] * 1e-2, movie[1] * 1e-2, 1 / movie[2] * 1e-2
    xi = np.arange(pix / 2, roi_w + pix / 2, pix)
    zi = np.arange(pix / 2, roi_h + pix / 2, pix)
    xi, zi = np.meshgrid(xi - np.mean(xi), zi)

    fc = float(transducer.fc)
    maxD = math.hypot((roi_w + L) / 2, roi_h)                       # distancia máxima recorrida
    df = 1 / 2 / (maxD / c) * frequency_step
    Nf = int(2 * math.ceil(fc / df) + 1)

    _, S, f, keep = pfield(transducer, xi, None, zi, tx_delays, tx_apodization, c=c,
                           attenuation=attenuation, element_splitting=1, db_thresh=db_thresh,
                           df=df, return_spectrum=True, dtype=dtype, device=device, budget=budget)
    full = torch.zeros(*xi.shape, Nf, dtype=S.dtype, device=S.device)
    full[..., keep] = S                                             # espectro en las Nf frecuencias
    F = torch.fft.irfft(full, dim=-1)                               # [nz, nx, 2(Nf-1)]
    F = torch.flip(F, dims=[-1])
    F = F[..., : int(round(F.shape[-1] // 2))]
    F = F / F.abs().max()
    if gamma != 1:
        F = torch.sign(F) * F.abs() ** gamma
    F = ((F + 1) / 2 * 255).to(torch.uint8).cpu().numpy()

    info = {"Xgrid": xi[0, :], "Zgrid": zi[:, 0], "TimeStep": maxD / c / F.shape[2]}
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
