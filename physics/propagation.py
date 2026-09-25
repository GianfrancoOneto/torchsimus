import torch

def propagation_2d(r, frequency, c=1540.0):
    k = 2 * torch.pi * frequency / c
    return torch.exp(1j * k * r) / torch.sqrt(r)
