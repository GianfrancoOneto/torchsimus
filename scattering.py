import torch

def single_scatter(G: torch.Tensor, reflectivity: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    tx = G @ A
    scattered = reflectivity[:, None] * tx
    return G.transpose(-1, -2) @ scattered

def scatter_operator(G: torch.Tensor, reflectivity: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    if G.ndim == 2:
        tx = G @ A
        scattered = reflectivity[:, None] * tx
        return G.transpose(-1, -2) @ scattered
    else:
        tx = torch.matmul(G, A)
        scattered = reflectivity[None, :, None] * tx
        return torch.matmul(G.transpose(-1, -2), scattered)
