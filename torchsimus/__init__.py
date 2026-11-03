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
