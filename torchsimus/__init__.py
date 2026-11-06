"""torchsimus: simulador de ultrasonido (SIMUS/MUST) diferenciable en PyTorch.

Uso:
    from torchsimus.probes import get_probe
    from torchsimus.simulation import simus          # RF con la física completa de MUST (M34-35)
    from torchsimus.field import pfield
    from torchsimus.simulator import Simus           # modelo 2-D rápido, con batch (M15-M30)

Organización del paquete:
    geometry/      arreglos: lineal, convexo, matricial, anillo, personalizado, circular
    elements/      forma de los elementos: puntual, rectangular (sinc x cos θ), circular
    physics/       piezas del modelo: geometría local, propagación, atenuación, elevación
    cabezales/     35 cabezales de ejemplo (CSV)
    spectra.py     espectros: pulso (seno o chirp), sonda, pulso-eco
    pulse.py       getpulse: el pulso en el tiempo
    delays.py      leyes de retardo 2-D y 3-D (txdelay, txdelay3)
    probes.py      las 10 sondas de MUST
    phantoms.py    genscat: fantomas de scatterers
    simulator.py   Simus (nn.Module) y chunking.py
    simulation.py  simus / simus3 (equivalentes a pymust)
    field.py       pfield / pfield3        movie.py   mkmovie
    processing/    rf2iq, das/das3, tgc, doppler/bmode, sptrack, smoothn, grids, optim
    visualization.py
"""
__version__ = "0.2.0"

# --- Hilos de CPU -------------------------------------------------------------------------
# PyTorch usa por defecto un hilo por núcleo FÍSICO; en Colab gratis eso es 1, aunque hay 2 CPU
# virtuales. Al importar torchsimus se usan todos los CPU disponibles (en Colab: 2 hilos), salvo
# que la cantidad se fije con la variable de entorno TORCHSIMUS_NUM_THREADS (u OMP_NUM_THREADS).
import os as _os


def _cpus_disponibles():
    try:
        return len(_os.sched_getaffinity(0))
    except AttributeError:
        return _os.cpu_count() or 1


def set_threads(n=None):
    """Fija los hilos de PyTorch en CPU (por defecto: todos los CPU disponibles). Devuelve los hilos en uso."""
    import torch
    n = _cpus_disponibles() if n is None else int(n)
    torch.set_num_threads(max(1, n))
    return torch.get_num_threads()


def _hilos_al_importar():
    n = _os.environ.get("TORCHSIMUS_NUM_THREADS")
    if n is not None:
        set_threads(n)
    elif "OMP_NUM_THREADS" not in _os.environ:
        import torch
        if torch.get_num_threads() < _cpus_disponibles():
            set_threads()


_hilos_al_importar()
