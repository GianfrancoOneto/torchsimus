"""torchsimus: simulador de ultrasonido (SIMUS/MUST) diferenciable en PyTorch.

Uso:
    from torchsimus.probes import get_probe
    from torchsimus.field import pfield
    from torchsimus.simulator import Simus
"""
__version__ = "0.1.0"
<<<<<<< Updated upstream
=======

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
>>>>>>> Stashed changes
