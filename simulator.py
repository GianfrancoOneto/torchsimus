import torch
from torch import nn
from physics.geometry import local_geometry
from physics.propagation import propagation_2d_frequency, tx_weights
from scattering import scatter_operator
from spectra import simus_spectrum

class Simus(nn.Module):
    def __init__(self, transducer, medium, fs, n_fft, mode="direct"):
        super().__init__()
        self.transducer = transducer
        self.medium = medium
        self.fs = fs
        self.n_fft = n_fft
        self.mode = mode

    def forward(self, scatterers, reflectivity, tx_delays, tx_apodization):
        pose = self.transducer.geometry.pose()
        x_loc, y_loc, z_loc, r = local_geometry(scatterers, pose)
        c = getattr(self.medium, 'c', 1540.0)
        frequencies = torch.fft.rfftfreq(self.n_fft, d=1/self.fs).to(scatterers.device)
        G = propagation_2d_frequency(r, frequencies, c=c)
        directivity = self.transducer.element_shape.directivity(x_loc, y_loc, z_loc, r, frequencies)
        G_eff = G * directivity
        A = tx_weights(frequencies, tx_delays, tx_apodization)

        E = A.shape[-1] if A.ndim == 3 else 1

        mode = self.mode
        if mode == "auto":
            # Heurística empírica de crossover: si hay más de 8 emisiones, usar scattering-matrix mode
            mode = "scattering" if E > 8 else "direct"

        if mode == "scattering":
            R_G = reflectivity[None, :, None] * G_eff
            K = torch.matmul(G_eff.transpose(-1, -2), R_G)
            Y = torch.matmul(K, A)
        else:
            tx = torch.matmul(G_eff, A)
            scattered = reflectivity[None, :, None] * tx
            Y = torch.matmul(G_eff.transpose(-1, -2), scattered)

        Q = simus_spectrum(frequencies, self.transducer.fc, self.transducer.bandwidth)
        spectrum = Q[:, None, None] * Y
        rf = torch.fft.irfft(spectrum.conj(), n=self.n_fft, dim=0)
        return rf

def compute_rf_signal(transducer, scatterers, reflectivity, delays, apodization, fs, n_fft, c=1540.0, mode="direct"):
    class Medium:
        def __init__(self, c):
            self.c = c
    simulator = Simus(transducer, Medium(c), fs, n_fft, mode=mode)
    return simulator(scatterers, reflectivity, delays, apodization), torch.fft.rfftfreq(n_fft, d=1/fs).to(scatterers.device)
