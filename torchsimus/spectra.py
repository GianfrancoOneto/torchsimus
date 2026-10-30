import math
import torch

def physical_to_torch_spectrum(x: torch.Tensor) -> torch.Tensor:
    """
    Aplica la conjugación compleja explícita para pasar de la convención física e^{-i w t}
    a la convención FFT estándar de PyTorch e^{+i w t} antes de irfft().
    """
    return x.conj()

def sinc_unormalized(x: torch.Tensor) -> torch.Tensor:
    return torch.sinc(x / torch.pi)

def simus_spectrum(
    frequencies,
    fc,
    fractional_bandwidth,
    n_cycles=1
):
    w = 2 * torch.pi * frequencies
    wc = 2 * torch.pi * fc
    wb = fractional_bandwidth * wc

    T = n_cycles / fc

    Sp = 1j * (
        sinc_unormalized(
            T * (w - wc) / 2
        )
        -
        sinc_unormalized(
            T * (w + wc) / 2
        )
    )

    p = (
        torch.log(
            torch.tensor(
                126.,
                device=w.device,
                dtype=w.dtype
            )
        )
        /
        torch.log(
            torch.tensor(
                2 * wc / wb,
                device=w.device,
                dtype=w.dtype
            )
        )
    )

    St = torch.exp(
        -torch.log(
            torch.tensor(
                2.,
                device=w.device,
                dtype=w.dtype
            )
        )
        *
        (
            2 * torch.abs(w - wc)
            / wb
        ) ** p
    )

    return Sp * St

# ---------------------------------------------------------------------------
# Milestone 32: espectros de UNA vía (transmisión sola), necesarios para pfield
# ---------------------------------------------------------------------------
def pulse_spectrum(frequencies, fc, n_cycles=1):
    """Espectro del pulso de excitación (ventana de n_cycles ciclos a fc). Igual a MUST: getPulseSpectrumFunction."""
    w = 2 * torch.pi * frequencies
    wc = 2 * torch.pi * fc
    T = n_cycles / fc
    return 1j * (sinc_unormalized(T * (w - wc) / 2) - sinc_unormalized(T * (w + wc) / 2))

def probe_spectrum(frequencies, fc, fractional_bandwidth):
    """Respuesta de UNA vía del transductor (raíz de la respuesta pulso-eco). Igual a MUST: getProbeFunction."""
    w = 2 * torch.pi * frequencies
    wc = 2 * torch.pi * fc
    wb = fractional_bandwidth * wc
    p = torch.log(torch.tensor(126., device=w.device, dtype=w.dtype)) / torch.log(torch.tensor(2 * wc / wb, device=w.device, dtype=w.dtype))
    ln2 = torch.log(torch.tensor(2., device=w.device, dtype=w.dtype))
    return torch.exp(-0.5 * ln2 * (2 * torch.abs(w - wc) / wb) ** p)

def oneway_spectrum(frequencies, fc, fractional_bandwidth, n_cycles=1):
    """pulso x transductor (una vía). simus_spectrum = pulso x transductor^2 (pulso-eco)."""
    return pulse_spectrum(frequencies, fc, n_cycles) * probe_spectrum(frequencies, fc, fractional_bandwidth)

# ---------------------------------------------------------------------------
# Milestone 32: pulso chirp (barrido lineal de frecuencia), PARAM.TXfreqsweep de MUST
# ---------------------------------------------------------------------------
def fresnelint(x):
    """Integral de Fresnel C(x) + i S(x), con la aproximación de Mielenz (la misma que usa MUST)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    small = x.abs() <= 1.6
    c = torch.zeros_like(x); s = torch.zeros_like(x)
    n = torch.arange(0, 11, dtype=torch.float64)
    cn = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(-math.pi ** 2 * (4 * n + 1) / (4 * (2 * n + 1) * (2 * n + 2) * (4 * n + 5)), 0)])
    sn = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(-math.pi ** 2 * (4 * n + 3) / (4 * (2 * n + 2) * (2 * n + 3) * (4 * n + 7)), 0)]) * math.pi / 6
    n12 = torch.arange(0, 12, dtype=torch.float64)
    xs = torch.where(small, x, torch.zeros_like(x))[..., None]
    c = torch.where(small, (cn * xs ** (4 * n12 + 1)).sum(-1), c)
    s = torch.where(small, (sn * xs ** (4 * n12 + 3)).sum(-1), s)
    fn = torch.tensor([0.318309844, 9.34626e-08, -0.09676631, 0.000606222, 0.325539361, 0.325206461, -7.450551455,
                       32.20380908, -78.8035274, 118.5343352, -102.4339798, 39.06207702], dtype=torch.float64)
    gn = torch.tensor([0, 0.101321519, -4.07292e-05, -0.152068115, -0.046292605, 1.622793598, -5.199186089,
                       7.477942354, -0.695291507, -15.10996796, 22.28401942, -10.89968491], dtype=torch.float64)
    xl = torch.where(small, torch.ones_like(x), x)
    fx = (fn * xl[..., None] ** (-2 * n12 - 1)).sum(-1); gx = (gn * xl[..., None] ** (-2 * n12 - 1)).sum(-1)
    arg = math.pi / 2 * xl ** 2
    c = torch.where(small, c, 0.5 * torch.sign(xl) + fx * torch.sin(arg) - gx * torch.cos(arg))
    s = torch.where(small, s, 0.5 * torch.sign(xl) - fx * torch.cos(arg) - gx * torch.sin(arg))
    return torch.complex(c, s)


def pulse_spectrum_full(frequencies, fc, n_cycles=1, freq_sweep=None):
    """Espectro del pulso de excitación: seno enventanado (freq_sweep=None) o chirp lineal (freq_sweep en Hz)."""
    if freq_sweep is None or (isinstance(freq_sweep, float) and math.isinf(freq_sweep)):
        return pulse_spectrum(frequencies, fc, n_cycles)
    f = torch.as_tensor(frequencies, dtype=torch.float64)
    w = 2 * math.pi * f; wc = 2 * math.pi * fc
    T = n_cycles / fc; dw = 2 * math.pi * freq_sweep
    def s2(w_):
        a = math.sqrt(math.pi * dw / T)
        return math.sqrt(math.pi * T / dw) * torch.exp(-1j * (w_ - wc) ** 2 * T / 2 / dw) * \
            (fresnelint((dw / 2 + w_ - wc) / a) + fresnelint((dw / 2 - w_ + wc) / a))
    return (1j * s2(w) - 1j * s2(-w)) / T
