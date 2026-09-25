import torch
from physics.propagation import tx_weights
from spectra import simus_spectrum

def forward_chunked(
    transducer,
    scatterers,
    reflectivity,
    tx_delays,
    tx_apodization,
    fs,
    n_fft,
    c=1540.0,
    scatter_chunk=8192,
    freq_chunk=16
):
    frequencies = torch.fft.rfftfreq(n_fft, d=1/fs).to(scatterers.device)
    F = frequencies.numel()
    pose = transducer.geometry.pose()

    A = tx_weights(frequencies, tx_delays, tx_apodization)

    blocks = []
    for f0 in range(0, F, freq_chunk):
        f1 = min(f0 + freq_chunk, F)
        freqs_b = frequencies[f0:f1]
        A_b = A[f0:f1] if A.ndim == 3 else A

        Yb = None
        for s0 in range(0, scatterers.shape[0], scatter_chunk):
            s1 = min(s0 + scatter_chunk, scatterers.shape[0])

            scatterers_b = scatterers[s0:s1]
            R_b = reflectivity[s0:s1]

            d = scatterers_b[:, None, :] - pose.centers[None, :, :]
            r = torch.linalg.vector_norm(d, dim=-1)

            k = (2 * torch.pi * freqs_b / c)[:, None, None]
            G = torch.exp(1j * k * r[None, :, :]) / torch.sqrt(r)[None, :, :]

            x_loc = (d * pose.u[None, :, :]).sum(-1)
            z_loc = (d * pose.n[None, :, :]).sum(-1)
            directivity = transducer.element_shape.directivity(x_loc, None, z_loc, r, freqs_b)
            G_eff = G * directivity

            T = torch.matmul(G_eff, A_b)
            T = R_b[None, :, None] * T

            contribution = torch.matmul(G_eff.transpose(-1, -2), T)

            Yb = contribution if Yb is None else Yb + contribution

        blocks.append(Yb)

    Y = torch.cat(blocks, dim=0)
    Q = simus_spectrum(frequencies, transducer.fc, transducer.bandwidth)
    spectrum = Q[:, None, None] * Y
    rf = torch.fft.irfft(spectrum.conj(), n=n_fft, dim=0)
    return rf
