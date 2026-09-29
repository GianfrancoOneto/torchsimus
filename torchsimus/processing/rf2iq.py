"""Milestone 42: demodulación RF -> I/Q en PyTorch (equivalente a RF2IQ de MUST)."""
import math
import torch


def rf2iq(RF, fs, fc, bandwidth=None, t0=0.0, order=5):
    """
    RF [nt, ...] real -> IQ [nt, ...] complejo (envolvente compleja en banda base).
    bandwidth: fraccional (p. ej. 0.74). Si es None, corte del filtro en min(2 fc/fs, 0.5) (como MUST).
    Filtro: Butterworth pasa-bajos de orden `order` aplicado ida y vuelta (fase cero, como filtfilt):
    |H(w)|^2 = 1 / (1 + (tan(w/2) / tan(pi Wn / 2))^(2 order)), con extensión impar en los bordes.
    """
    RF = torch.as_tensor(RF)
    if not torch.is_floating_point(RF):
        RF = RF.double()
    nt = RF.shape[0]
    t = torch.arange(nt, dtype=torch.float64, device=RF.device) / fs + t0
    Wn = min(2 * fc / fs, 0.5) if bandwidth is None else fc * bandwidth / fs
    shape = [-1] + [1] * (RF.ndim - 1)
    IQ = torch.exp(-2j * math.pi * fc * t).reshape(shape) * RF.to(torch.float64)

    pad = min(3 * (order + 1), nt - 1)                                  # extensión impar como filtfilt
    head = 2 * IQ[:1] - torch.flip(IQ[1:pad + 1], [0])
    tail = 2 * IQ[-1:] - torch.flip(IQ[-pad - 1:-1], [0])
    X = torch.cat([head, IQ, tail], 0)
    n = X.shape[0]
    nfft = 1 << (2 * n - 1).bit_length()                               # cero-relleno: sin aliasing circular
    w = 2 * math.pi * torch.fft.fftfreq(nfft, dtype=torch.float64, device=RF.device)
    H2 = 1 / (1 + (torch.tan(w.abs() / 2) / math.tan(math.pi * Wn / 2)) ** (2 * order))
    Y = torch.fft.ifft(torch.fft.fft(X, n=nfft, dim=0) * H2.reshape(shape), dim=0)[:n]
    return 2 * Y[pad:pad + nt]                                          # factor 2: conserva la amplitud de la envolvente
