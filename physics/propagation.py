import torch

def propagation_2d(r: torch.Tensor, frequency: torch.Tensor, c: float = 1540.0) -> torch.Tensor:
    k = 2 * torch.pi * frequency / c
    return torch.exp(1j * k * r) / torch.sqrt(r)

def propagation_2d_frequency(r: torch.Tensor, frequencies: torch.Tensor, c: float = 1540.0) -> torch.Tensor:
    k = (2 * torch.pi * frequencies / c)[:, None, None]
    return torch.exp(1j * k * r[None, :, :]) / torch.sqrt(r)[None, :, :]

def propagation_with_attenuation(r: torch.Tensor, frequencies: torch.Tensor, c: float = 1540.0, alpha_db: float = 0.0) -> torch.Tensor:
    k = (2 * torch.pi * frequencies / c)[:, None, None]
    ka = (alpha_db * frequencies / (8.69 * 1e4))[:, None, None]
    exponent = (-ka + 1j * k) * r[None, :, :]
    return torch.exp(exponent) / torch.sqrt(r)[None, :, :]

def elevation_factor(y, r, k, h, Rf, A, B):
    alpha = (
        B[None, None, None, :] / (h ** 2)
        + 0.5j * k[:, None, None, None] * (1.0 / Rf - 1.0 / r[None, :, :, None])
    )
    beta = -1j * k[:, None, None] * y[None, :, :] / r[None, :, :]
    gamma = 0.5j * k[:, None, None] * (y[None, :, :] ** 2) / r[None, :, :]

    term = (
        A[None, None, None, :]
        * torch.sqrt(torch.tensor(torch.pi, dtype=alpha.dtype, device=alpha.device) / alpha)
        * torch.exp((beta[..., None] ** 2) / (4.0 * alpha) + gamma[..., None])
    )
    return term.sum(dim=-1)

def tx_weights(frequency: torch.Tensor, delays: torch.Tensor, apodization: torch.Tensor) -> torch.Tensor:
    omega = 2 * torch.pi * frequency
    if frequency.ndim == 0:
        return apodization * torch.exp(1j * omega * delays)
    else:
        phase = omega[:, None, None] * delays[None, :, :]
        return apodization[None, :, :] * torch.exp(1j * phase)
