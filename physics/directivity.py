import torch

def rectangular_directivity(x_local, z_local, frequencies, width, c=1540.0):
    r_xz = torch.sqrt(x_local*x_local + z_local*z_local)
    sin_theta = x_local / r_xz
    b = width / 2
    freqs = frequencies.unsqueeze(0) if frequencies.ndim == 0 else frequencies
    k = (2 * torch.pi * freqs / c)[:, None, None]
    q = k * b * sin_theta[None]
    res = torch.sinc(q / torch.pi)
    return res.squeeze(0) if frequencies.ndim == 0 else res
