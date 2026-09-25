"""
Milestone 40: sondas comerciales (mismos valores que la base de datos de MUST) como Transducer de torchsimus.
"""
import math
from geometry.linear import LinearArray
from geometry.convex import ConvexArray
from geometry.matrix import MatrixArray
from elements.rectangular import RectangularElement
from transducer import Transducer

PROBES = {
    "P4-2v":  dict(fc=2.72e6, pitch=0.300e-3, width=0.250e-3, num_elements=64,  bandwidth=0.74, radius=math.inf, height=14e-3,  focus=60e-3),
    "L11-5v": dict(fc=7.60e6, pitch=0.300e-3, width=0.270e-3, num_elements=128, bandwidth=0.77, radius=math.inf, height=5e-3,   focus=18e-3),
    "L12-3v": dict(fc=7.54e6, pitch=0.200e-3, width=0.170e-3, num_elements=192, bandwidth=0.93, radius=math.inf, height=5e-3,   focus=20e-3),
    "C5-2v":  dict(fc=3.57e6, pitch=0.508e-3, width=0.460e-3, num_elements=128, bandwidth=0.79, radius=49.57e-3, height=13.5e-3, focus=60e-3),
}


def make_transducer(fc, pitch, width, num_elements, bandwidth, radius=math.inf, height=math.inf, focus=math.inf):
    geo = LinearArray(num_elements, pitch) if math.isinf(radius) else ConvexArray(num_elements, pitch, radius)
    el = RectangularElement(width)
    el.height = height
    tr = Transducer(geo, el, center_frequency=fc, bandwidth=bandwidth)
    tr.height, tr.elevation_focus, tr.pitch, tr.width = height, focus, pitch, width
    return tr


def get_probe(name, num_elements=None):
    """Transducer de la sonda `name`; num_elements permite crear una subapertura del mismo arreglo."""
    p = dict(PROBES[name])
    if num_elements is not None:
        p["num_elements"] = num_elements
    tr = make_transducer(**p)
    tr.name = name
    return tr


def matrix_probe(nx, ny, pitch, width, height, fc, bandwidth):
    el = RectangularElement(width)
    el.height = height
    tr = Transducer(MatrixArray(nx, ny, pitch, pitch), el, center_frequency=fc, bandwidth=bandwidth)
    tr.height, tr.width, tr.pitch = height, width, pitch
    return tr
