"""
Milestone 37: post-procesamiento Doppler y modo B en PyTorch
(equivalentes a pymust.iq2doppler, pymust.getNyquistVelocity y pymust.bmode).
Aceptan arrays de numpy o tensores (reales o complejos) y devuelven tensores.
"""
import math
import torch
import torch.nn.functional as Fnn


def _t(a, device=None):
    return a.to(device) if torch.is_tensor(a) else torch.as_tensor(a, device=device)


def nyquist_velocity(fc, prf, c=1540.0, lag=1):
    """Velocidad de Nyquist VN = c * PRF / (4 fc lag)."""
    return c * prf / 4 / fc / lag


def _hamming(n, dtype, device):
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    k = torch.arange(n, dtype=dtype, device=device)
    return 0.54 - 0.46 * torch.cos(2 * math.pi * k / (n - 1))


def _smooth2d_symmetric(A, M):
    """Convolución 2-D 'same' con ventana de Hamming M[0] x M[1] y bordes simétricos (como scipy 'symmetric')."""
    h = _hamming(M[0], torch.float64, A.device)[:, None] * _hamming(M[1], torch.float64, A.device)[None, :]
    h = torch.flip(h, dims=[0, 1])                                   # convolución (no correlación)
    pt, pb = M[0] // 2, (M[0] - 1) // 2
    pl, pr = M[1] // 2, (M[1] - 1) // 2

    def pad_sym(X):
        top = torch.flip(X[:pt], [0]) if pt else X[:0]
        bot = torch.flip(X[X.shape[0] - pb:], [0]) if pb else X[:0]
        X = torch.cat([top, X, bot], 0)
        lef = torch.flip(X[:, :pl], [1]) if pl else X[:, :0]
        rig = torch.flip(X[:, X.shape[1] - pr:], [1]) if pr else X[:, :0]
        return torch.cat([lef, X, rig], 1)

    def conv(X):
        Xp = pad_sym(X)[None, None]
        return Fnn.conv2d(Xp, h[None, None].to(Xp.dtype))[0, 0]

    if torch.is_complex(A):
        return torch.complex(conv(A.real.double()), conv(A.imag.double()))
    return conv(A.double())


def iq2doppler(IQ, fc, prf, c=1540.0, M=1, lag=1):
    """
    Velocidad Doppler color (autocorrelación de lag `lag`, estimador de Kasai) y su varianza.
    IQ : [nz, nx, n_emisiones] complejo (señales I/Q ya beamformadas).
    Devuelve (vel [m/s], varianza), igual que pymust.iq2doppler(IQ, param, M, lag).
    """
    IQ = _t(IQ).to(torch.complex128)
    if isinstance(M, int):
        M = (M, M)
    IQ1, IQ2 = IQ[:, :, :-lag], IQ[:, :, lag:]
    AC = (IQ1 * IQ2.conj()).sum(2)
    P = (IQ.real ** 2 + IQ.imag ** 2).sum(2)
    if tuple(M) != (1, 1):
        AC = _smooth2d_symmetric(AC, M)
        P = _smooth2d_symmetric(P, M)
    VN = nyquist_velocity(fc, prf, c, lag)
    vel = -VN * torch.angle(AC) / math.pi
    variance = 2 * (VN / math.pi) ** 2 * (1 - AC.abs() / P)
    return vel, variance


def bmode(IQ, DR=40.0):
    """Imagen modo B de 8 bits (compresión logarítmica con rango dinámico DR en dB)."""
    I = _t(IQ).abs().double()
    if DR >= 1:
        I = 20 * torch.log10(I / I.max()) + DR
        I = 255 * I / DR
    else:
        I = (I / I.max()) ** DR * 255
    return I.clamp(0, 255).to(torch.uint8)
