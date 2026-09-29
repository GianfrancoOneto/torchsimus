import torch

def propagation_with_attenuation(r: torch.Tensor, frequencies: torch.Tensor, c: float = 1540.0, alpha_db: float = 0.0) -> torch.Tensor:
    k = (2 * torch.pi * frequencies / c)[:, None, None]
    ka = (alpha_db * frequencies / (8.69 * 1e4))[:, None, None]
    exponent = (-ka + 1j * k) * r[None, :, :]
    return torch.exp(exponent) / torch.sqrt(r)[None, :, :]
