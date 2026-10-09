"""
Milestone 32: pulso de transmisión (equivalente a pymust.getpulse).

    pulse, t = getpulse(transducer, way=2, n_cycles=1, freq_sweep=None, kind="pressure")

way       : 1 = una vía (emisión); 2 = pulso-eco (emisión + recepción).
n_cycles  : duración del pulso en ciclos (PARAM.TXnow de MUST).
freq_sweep: None -> seno enventanado; un número (Hz) -> chirp lineal de ese barrido (PARAM.TXfreqsweep).
kind      : "pressure", "vel2d" o "vel3d" (velocidad de la partícula en 2-D / 3-D).
"""
import math
import torch
from torchsimus.spectra import pulse_spectrum_full, probe_spectrum


def getpulse(transducer, way=2, n_cycles=1, freq_sweep=None, dt=1e-9, kind="pressure"):
    assert way in (1, 2), "way debe ser 1 (una vía) o 2 (ida y vuelta)"
    fc = float(transducer.fc)
    bw = float(transducer.bandwidth)
    df = fc / n_cycles / 32
    Nf = 2 ** int(math.ceil(math.log2(1 / dt / 2 / df)))
    f = torch.linspace(0, 1 / dt / 2, Nf, dtype=torch.float64)
    F = pulse_spectrum_full(f, fc, n_cycles, freq_sweep) * probe_spectrum(f, fc, bw) ** way
    if kind in ("vel2d", "velocity2d"):
        F = F / (torch.sqrt(f) + 1e-9)
    elif kind in ("vel3d", "velocity3d"):
        F = F / (f + 1e-9)
    pulse = torch.fft.fftshift(torch.fft.irfft(F))
    pulse = pulse / pulse.abs().max()
    idx = torch.nonzero(pulse > 1 / 1023).flatten()
    i1, i2 = int(idx[0]), int(idx[-1])
    k = min(i1 + 1, 2 * Nf - 1 - i2 - 1)
    L = pulse.numel()
    pulse = pulse[k - 1: L - k + 1].flip(0)          # = pulse[-k : k-2 : -1] de pymust
    t = torch.arange(pulse.numel(), dtype=torch.float64) * dt
    return pulse, t
