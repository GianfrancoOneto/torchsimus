import torch

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
