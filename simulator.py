import torch
from torch import nn
from physics.geometry import local_geometry
from physics.propagation import propagation_2d_frequency, tx_weights
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
        is_batched = (scatterers.ndim == 3)
        if not is_batched:
            scatterers = scatterers.unsqueeze(0)
            reflectivity = reflectivity.unsqueeze(0)

        if tx_delays.ndim == 2:
            tx_delays = tx_delays.unsqueeze(0)
        if tx_apodization.ndim == 2:
            tx_apodization = tx_apodization.unsqueeze(0)

        pose = self.transducer.geometry.pose()
        N = pose.centers.shape[0]
        if tx_delays.shape[1] != N and tx_delays.shape[2] == N:
            tx_delays = tx_delays.transpose(1, 2)
            tx_apodization = tx_apodization.transpose(1, 2)

        # Geometría local batteada: [B, S, N, 3]
        d = scatterers[:, :, None, :] - pose.centers[None, None, :, :]
        r = torch.linalg.vector_norm(d, dim=-1)
        x_loc = (d * pose.u[None, None, :, :]).sum(-1)
        z_loc = (d * pose.n[None, None, :, :]).sum(-1)
        y_loc = (d * pose.v[None, None, :, :]).sum(-1)

        c = getattr(self.medium, 'c', 1540.0)
        frequencies = torch.fft.rfftfreq(self.n_fft, d=1/self.fs).to(scatterers.device)

        k = (2 * torch.pi * frequencies / c)
        G = torch.exp(1j * k[:, None, None, None] * r[None, :, :, :]) / torch.sqrt(r)[None, :, :, :]

        directivity = self.transducer.element_shape.directivity(x_loc, y_loc, z_loc, r, frequencies)
        if directivity.ndim == 3:
            directivity = directivity.unsqueeze(0)
        G_eff = G * directivity

        omega = 2 * torch.pi * frequencies
        phase = omega[:, None, None, None] * tx_delays[None, :, :, :]
        A = tx_apodization[None, :, :, :] * torch.exp(1j * phase)

        mode = self.mode
        if mode == "auto":
            E = A.shape[-1]
            mode = "scattering" if E > 8 else "direct"

        if mode == "scattering":
            R_G = reflectivity[None, :, :, None] * G_eff
            K = torch.matmul(G_eff.transpose(-1, -2), R_G)
            Y = torch.matmul(K, A)
        else:
            T = torch.matmul(G_eff, A)
            scattered = reflectivity[None, :, :, None] * T
            Y = torch.matmul(G_eff.transpose(-1, -2), scattered)

        Q = simus_spectrum(frequencies, self.transducer.fc, self.transducer.bandwidth)
        spectrum = Q[:, None, None, None] * Y
        rf = torch.fft.irfft(spectrum.conj(), n=self.n_fft, dim=0) # [Nt, B, N, E]
        rf = rf.permute(1, 0, 3, 2) # [B, Nt, E, N]

        if not is_batched:
            rf = rf.squeeze(0)
        return rf

def compute_rf_signal(transducer, scatterers, reflectivity, delays, apodization, fs, n_fft, c=1540.0, mode="direct"):
    class Medium:
        def __init__(self, c):
            self.c = c
    simulator = Simus(transducer, Medium(c), fs, n_fft, mode=mode)
    return simulator(scatterers, reflectivity, delays, apodization), torch.fft.rfftfreq(n_fft, d=1/fs).to(scatterers.device)
