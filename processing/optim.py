import math


def fminbound(fun, a, b, xtol=1e-5, maxiter=500):
    """Minimiza una función escalar en [a, b] por sección áurea (sin scipy)."""
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fun(c), fun(d)
    for _ in range(maxiter):
        if abs(b - a) < xtol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = fun(d)
    return (a + b) / 2
