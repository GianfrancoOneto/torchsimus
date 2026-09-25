import torch

def physical_to_torch_spectrum(x: torch.Tensor) -> torch.Tensor:
    """
    Aplica la conjugación compleja explícita para pasar de la convención física e^{-i w t}
    a la convención FFT estándar de PyTorch e^{+i w t} antes de irfft().
    """
    return x.conj()
