import torch

def propagation_2d(r, frequency, c=1540.0):
    k = 2 * torch.pi * frequency / c
    return torch.exp(1j * k * r) / torch.sqrt(r)

def tx_weights(frequency, delays, apodization):
    omega = 2 * torch.pi * frequency
    A = apodization * torch.exp(1j * omega * delays)
    return A.T
