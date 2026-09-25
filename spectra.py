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
