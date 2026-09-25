"""
Milestone 44: utilidades de visualización (matplotlib) para los resultados de torchsimus.
"""
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


def to_np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def db(P):
    P = to_np(P)
    return 20 * np.log10(P / np.max(P) + 1e-300)


def show_field(P, x, z, title="", clim=(-20, 0), points=None, points_label=None, elements=None, ax=None):
    """Campo de presión RMS en dB sobre una grilla cartesiana (x, z) en m."""
    x, z = to_np(x), to_np(z)
    ax = ax or plt.gca()
    im = ax.imshow(db(P), extent=np.array([x[0, 0], x[-1, -1], z[-1, -1], z[0, 0]]) * 1e2, cmap="hot",
                   vmin=clim[0], vmax=clim[1])
    ax.set_aspect("equal"); plt.colorbar(im, ax=ax)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("z (cm)"); ax.set_title(title, fontsize=10)
    if points is not None:
        ax.scatter(to_np(points[0]).ravel() * 1e2, to_np(points[1]).ravel() * 1e2, c="b", label=points_label)
        if points_label: ax.legend()
    if elements is not None:
        ax.plot(to_np(elements[0]).ravel() * 1e2, to_np(elements[1]).ravel() * 1e2, c="g", linewidth=5)
    return ax


def show_polar_field(P, x, z, title="", clim=(-20, 0), elements=None, ax=None):
    """Campo en dB sobre una grilla polar (impolgrid)."""
    ax = ax or plt.gca()
    pc = ax.pcolormesh(to_np(x) * 1e2, to_np(z) * 1e2, db(P), cmap="hot", vmin=clim[0], vmax=clim[1], shading="auto")
    ax.set_facecolor("white"); ax.invert_yaxis(); ax.set_aspect("equal")
    cb = plt.colorbar(pc, ax=ax); cb.set_label("Presión (dB)")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("z (cm)"); ax.set_title(title, fontsize=10)
    if elements is not None:
        ax.plot(to_np(elements[0]).ravel() * 1e2, to_np(elements[1]).ravel() * 1e2, c="g", linewidth=4)
    return ax


def show_field_3d(slices, title="", clim=(-20, 0), ax=None, alpha=0.15):
    """slices: lista de (P, X, Y, Z) (cortes 2-D) dibujados como superficies en 3-D."""
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")
    cm = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(*clim), "hot")
    for P, X, Y, Z in slices:
        color = cm.to_rgba(db(P)); color[:, :, 3] = alpha
        ax.plot_surface(to_np(X), to_np(Y), to_np(Z), rstride=1, cstride=1, antialiased=True, facecolors=color, alpha=0.05)
    ax.set_title(title)
    return ax


def slice_plot(xi, yi, zi, P, YZ=None, XZ=None, XY=None, clim=(-20, 0)):
    """Cortes YZ (x fijo), XZ (y fijo) y XY (z fijo) de un campo 3-D [nx, ny, nz] (meshgrid indexing='ij'), en mm."""
    xi, yi, zi, P = map(to_np, (xi, yi, zi, P))
    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    cm = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(*clim), "hot")
    top = P.max()
    for val, axis in [(YZ, 0), (XZ, 1), (XY, 2)]:
        if val is None:
            continue
        coord = [xi[:, 0, 0], yi[0, :, 0], zi[0, 0, :]][axis]
        k = int(np.argmin(np.abs(coord - val)))
        sl = [slice(None)] * 3; sl[axis] = k; sl = tuple(sl)
        color = cm.to_rgba(20 * np.log10(P[sl] / top + 1e-300))
        ax.plot_surface(xi[sl] * 1e3, yi[sl] * 1e3, zi[sl] * 1e3, rstride=1, cstride=1, facecolors=color,
                        antialiased=True, alpha=0.75, linewidth=0)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
    return fig, ax


def single_slice_plot(xi, yi, zi, P, plane, value, focus=None, elements=None, ax=None, clim=(-20, 0)):
    """Un corte 2-D ('YZ', 'XZ' o 'XY') de un campo 3-D [nx, ny, nz] (indexing='ij'), normalizado a su máximo."""
    xi, yi, zi, P = map(to_np, (xi, yi, zi, P))
    ax = ax or plt.gca()
    axes = {"YZ": (0, yi[0, :, 0], zi[0, 0, :], "y", "z"), "XZ": (1, xi[:, 0, 0], zi[0, 0, :], "x", "z"),
            "XY": (2, xi[:, 0, 0], yi[0, :, 0], "x", "y")}
    axis, h, v, lh, lv = axes[plane]
    coord = [xi[:, 0, 0], yi[0, :, 0], zi[0, 0, :]][axis]
    k = int(np.argmin(np.abs(coord - value)))
    S = np.take(P, k, axis=axis).T
    im = ax.imshow(20 * np.log10(S / S.max() + 1e-300), origin="lower", cmap="hot", aspect="auto",
                   extent=[h[0] * 1e2, h[-1] * 1e2, v[0] * 1e2, v[-1] * 1e2], vmin=clim[0], vmax=clim[1])
    plt.colorbar(im, ax=ax)
    fixed = {"YZ": "x", "XZ": "y", "XY": "z"}[plane]
    ax.set_title(f"Campo de presión RMS\nplano {plane}, {fixed} = {coord[k] * 1e2:.2f} cm", fontsize=10)
    ax.set_xlabel(f"{lh} [cm]"); ax.set_ylabel(f"{lv} [cm]")
    idx = {"x": 0, "y": 1, "z": 2}
    if focus is not None:
        ax.scatter(focus[idx[lh]] * 1e2, focus[idx[lv]] * 1e2, c="b", label="punto focal")
    if elements is not None and plane != "XY":
        e = to_np(elements)
        ax.scatter(e[:, idx[lh]] * 1e2, e[:, idx[lv]] * 1e2, c="m", s=4, label="elementos")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    return ax


def doppler_cmap():
    """Mapa de colores Doppler: azul (se aleja) - negro (0) - rojo/amarillo (se acerca)."""
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "doppler", [(0.0, (0.6, 1.0, 1.0)), (0.25, (0.0, 0.3, 1.0)), (0.5, (0.0, 0.0, 0.0)),
                    (0.75, (1.0, 0.1, 0.0)), (1.0, (1.0, 1.0, 0.4))])


def movie_frames(F, n=5, title="", axs=None):
    """Muestra n cuadros equiespaciados de una película de mkmovie."""
    idx = np.linspace(0, F.shape[2] - 1, n).astype(int)
    if axs is None:
        _, axs = plt.subplots(1, n, figsize=(3.2 * n, 5.6))
    for a, k in zip(axs, idx):
        a.imshow(F[:, :, k], cmap="hot", vmin=0, vmax=255); a.set_title(f"cuadro {k}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    axs[0].set_ylabel(title)
    return axs
