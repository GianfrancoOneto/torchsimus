"""
Milestones 41-43 — tgc: compensación de ganancia en profundidad (equivalente a pymust.tgc / TGC de MUST).

    S_tgc, C = tgc(S)

S: RF real [nt, N] (se usa la envolvente de Hilbert) o I/Q complejo [nt, N] / imagen [nz, nx].
Se ajusta una recta robusta (Theil-Sen: mediana de pendientes entre pares) al log de la amplitud media
entre el 10 % y el 90 % de la profundidad, y se compensa ese decaimiento.
Como MUST, usa como mucho 200 puntos elegidos al azar (np.random global o `rng`): con la misma
semilla da exactamente lo mismo que pymust.
"""
import math
import numpy as np
import torch


def _hilbert_abs(S):
    n = S.shape[0]
    X = torch.fft.fft(S, dim=0)
    h = torch.zeros(n, dtype=torch.float64, device=S.device)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1; h[1:n // 2] = 2
    else:
        h[0] = 1; h[1:(n + 1) // 2] = 2
    return torch.fft.ifft(X * h.reshape(-1, *([1] * (S.ndim - 1))), dim=0).abs()


def _median(v):
    s = torch.sort(v).values; n = s.numel()
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def tgc(S, n_points=200, rng=None):
    R = np.random if rng is None else rng
    S = torch.as_tensor(S)
    A = S.abs().double() if torch.is_complex(S) else _hilbert_abs(S.double())
    C = A.mean(1)
    n = C.numel()
    n1, n2 = int(math.ceil(n / 10)), int(math.floor(n * 9 / 10))
    p = min(n_points / (n2 - n1) * 100, 100)
    x = torch.arange(n1, n2, dtype=torch.float64); y = torch.log(C[n1:n2])
    I = torch.as_tensor(R.permutation(len(x))[: int(round(len(x) * p / 100))])
    x, y = x[I], y[I]
    i, j = torch.triu_indices(len(x), len(x), 1)
    slope = _median((y[j] - y[i]) / (x[j] - x[i]))
    intercept = _median(y - slope * x)
    Cfit = torch.exp(intercept + slope * torch.arange(n, dtype=torch.float64)).reshape(-1, *([1] * (S.ndim - 1)))
    Cfit = Cfit[0] / Cfit
    return S * Cfit, Cfit
