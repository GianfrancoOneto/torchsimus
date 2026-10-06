"""
Milestone 46: carpeta de cabezales de ejemplo (archivos CSV) que viaja con el paquete.

Cada archivo `<nombre>.csv` tiene una fila por elemento con columnas x, y, z (mm) y nx, ny, nz (normal).

    from torchsimus import cabezales
    cabezales.listar()                       # nombres disponibles (opcional: tipo="convexo")
    cabezales.ruta("convexo_1_128el_R49.57mm")
    centros, normales = cabezales.cargar("anillo_1_64el_R6mm")      # tensores [N, 3] en metros
    cabezales.generar_ejemplos()             # vuelve a crear los 35 CSV en esta carpeta
"""
import csv
import math
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TIPOS = ["lineal", "convexo", "matricial", "circular", "anillo", "espiral", "otro"]


def listar(tipo=None):
    """Nombres (sin .csv) de los cabezales guardados, ordenados por tipo. tipo: 'lineal', 'convexo', ..."""
    nombres = [p.stem for p in CARPETA.glob("*.csv")]
    orden = lambda n: (TIPOS.index(n.split("_")[0]) if n.split("_")[0] in TIPOS else len(TIPOS), n)
    nombres = sorted(nombres, key=orden)
    return [n for n in nombres if tipo is None or n.startswith(tipo + "_")]


def ruta(nombre):
    """Ruta completa del archivo de un cabezal guardado (acepta el nombre con o sin .csv)."""
    p = CARPETA / (nombre if nombre.endswith(".csv") else nombre + ".csv")
    if not p.exists():
        raise FileNotFoundError(f"No existe el cabezal '{nombre}'. Disponibles: {listar()}")
    return p


def cargar(nombre, dtype=torch.float64):
    """Devuelve (centros [N, 3] en m, normales [N, 3] unitarias) de un cabezal guardado."""
    with open(ruta(nombre), newline="") as f:
        filas = list(csv.DictReader(f))
    c = torch.tensor([[float(r["x"]), float(r["y"]), float(r["z"])] for r in filas], dtype=dtype) * 1e-3
    if filas and "nx" in filas[0]:
        n = torch.tensor([[float(r["nx"]), float(r["ny"]), float(r["nz"])] for r in filas], dtype=dtype)
    else:
        n = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype).repeat(len(c), 1)
    return c, n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)


def _guardar(carpeta, nombre, centros_m, normales=None):
    c = np.asarray(centros_m, dtype=float) * 1e3
    n = np.tile([0.0, 0.0, 1.0], (len(c), 1)) if normales is None else np.asarray(normales, dtype=float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    with open(Path(carpeta) / f"{nombre}.csv", "w", newline="") as f:
        f.write("x,y,z,nx,ny,nz\n")
        for p, q in zip(c, n):
            f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f}\n")


def generar_ejemplos(carpeta=CARPETA):
    """Crea los 35 cabezales de ejemplo (5 por tipo) como CSV en `carpeta`. Devuelve la lista de nombres."""
    from torchsimus.geometry.linear import LinearArray
    from torchsimus.geometry.convex import ConvexArray
    from torchsimus.geometry.matrix import MatrixArray
    from torchsimus.geometry.ring import RingArray
    from torchsimus.geometry.circular import CircularSingleElement

    hechos = []
    def guardar(nombre, c, n=None):
        _guardar(carpeta, nombre, c, n); hechos.append(nombre)
    def pose(p):
        return p.centers.numpy(), p.n.numpy()

    for i, (N, pi) in enumerate([(128, 0.300), (192, 0.200), (64, 0.300), (32, 0.500), (256, 0.150)], 1):
        guardar(f"lineal_{i}_{N}el_pitch{pi:g}mm", *pose(LinearArray(N, pi * 1e-3).pose()))
    for i, (N, pi, R) in enumerate([(128, 0.508, 49.57), (128, 0.300, 40.0), (64, 0.400, 20.0), (96, 0.250, 10.0), (192, 0.300, 60.0)], 1):
        guardar(f"convexo_{i}_{N}el_R{R:g}mm", *pose(ConvexArray(N, pi * 1e-3, R * 1e-3).pose()))
    for i, (nx, ny, pi) in enumerate([(32, 32, 0.300), (16, 16, 0.300), (8, 8, 0.500), (64, 8, 0.250), (24, 24, 0.200)], 1):
        guardar(f"matricial_{i}_{nx}x{ny}_pitch{pi:g}mm", *pose(MatrixArray(nx, ny, pi * 1e-3, pi * 1e-3).pose()))
    for i, (R, paso) in enumerate([(3.0, 0.19), (5.0, 0.25), (1.5, 0.10), (6.35, 0.30), (2.0, 0.13)], 1):
        guardar(f"circular_{i}_R{R:g}mm_parches{paso:g}mm", *pose(CircularSingleElement(R * 1e-3).patches(paso * 1e-3)))
    for i, (N, R) in enumerate([(64, 6.0), (128, 6.0), (32, 3.0), (256, 15.0), (48, 4.0)], 1):
        guardar(f"anillo_{i}_{N}el_R{R:g}mm", *pose(RingArray(N, R * 1e-3).pose()))
    for i, (N, R) in enumerate([(256, 6.0), (128, 5.0), (64, 3.0), (512, 10.0), (192, 8.0)], 1):
        k = np.arange(N); r = R * np.sqrt((k + 0.5) / N); th = k * math.pi * (3 - math.sqrt(5))
        guardar(f"espiral_{i}_{N}el_R{R:g}mm", np.stack([r * np.cos(th), r * np.sin(th), 0 * r], -1) * 1e-3)

    # formas libres
    N, R, ph = 64, 30.0, 0.4                                         # cóncavo: normales hacia el centro de curvatura
    p = np.linspace(-ph, ph, N)
    guardar("otro_1_concavo_64el_R30mm", np.stack([R * np.sin(p), 0 * p, R * (1 - np.cos(p))], -1) * 1e-3,
            np.stack([-np.sin(p), 0 * p, np.cos(p)], -1))
    x = np.linspace(-4.5, 4.5, 32); z = 0.15 * np.abs(x); s = np.where(x >= 0, 1.0, -1.0)   # V con elementos inclinados
    guardar("otro_2_V_32el", np.stack([x, 0 * x, z.max() - z], -1) * 1e-3, np.stack([0.15 * s, 0 * x, np.ones_like(x)], -1))
    a = (np.arange(33) - 16) * 0.3                                    # cruz (Mills cross)
    cruz = np.vstack([np.stack([a, 0 * a, 0 * a], -1), np.stack([0 * a, a, 0 * a], -1)[np.abs(a) > 1e-9]])
    guardar("otro_3_cruz_65el", cruz * 1e-3)
    t1, t2 = np.arange(48) * 2 * np.pi / 48, (np.arange(64) + 0.5) * 2 * np.pi / 64   # doble anillo
    guardar("otro_4_doble_anillo_112el", np.vstack([np.stack([3 * np.cos(t1), 3 * np.sin(t1), 0 * t1], -1),
                                                    np.stack([5 * np.cos(t2), 5 * np.sin(t2), 0 * t2], -1)]) * 1e-3)
    rng = np.random.default_rng(0)                                    # matriz dispersa aleatoria (semilla fija)
    g = (np.arange(32) - 15.5) * 0.3; xx, yy = np.meshgrid(g, g); m = rng.random(xx.shape) < 0.25
    guardar("otro_5_matriz_dispersa_aleatoria", np.stack([xx[m], yy[m], 0 * xx[m]], -1) * 1e-3)
    return hechos
