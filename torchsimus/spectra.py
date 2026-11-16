import torch

def simus_spectrum(frequencies, fc, bandwidth):
    sigma = (bandwidth * fc) / (2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0, dtype=frequencies.dtype, device=frequencies.device))))
    return torch.exp(-0.5 * ((frequencies - fc) / sigma) ** 2)

def physical_to_torch_spectrum(x):
    return x.conj()
