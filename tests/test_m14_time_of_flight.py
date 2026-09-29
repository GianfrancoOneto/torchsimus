import unittest
import torch
from geometry.linear import LinearArray
from elements.point import PointElement
from transducer import Transducer
from simulator import compute_rf_signal


def hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    """
    Envolvente de la señal vía transformada de Hilbert (dominio de frecuencia).

    Es necesaria porque el pulso generado por simus_spectrum es de fase
    puramente imaginaria (tipo "seno", impar) alrededor del tiempo de vuelo
    real: la señal RF cruda tiene un cruce por cero justo en t_theory, no un
    pico, flanqueada por dos lóbulos de amplitud casi idéntica. Por eso
    argmax(abs(rf)) puede elegir el lóbulo equivocado. La envolvente (como en
    procesamiento de ultrasonido real) sí tiene su máximo en t_theory.
    """
    n = x.shape[0]
    Xf = torch.fft.fft(x, dim=0)
    h = torch.zeros(n, dtype=x.dtype, device=x.device)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = torch.fft.ifft(Xf * h, dim=0)
    return torch.abs(analytic)


class TestM14TimeOfFlight(unittest.TestCase):
    def test_time_of_flight_single_scatterer(self):
        c = 1540.0
        fc = 5e6
        fs = 50e6
        n_fft = 2048

        geometry = LinearArray(num_elements=1, pitch=0.3e-3)
        element_shape = PointElement()
        transducer = Transducer(geometry, element_shape, center_frequency=fc, bandwidth=0.7)

        zs = 0.03
        scatterers = torch.tensor([[0.0, 0.0, zs]], dtype=torch.float64)
        reflectivity = torch.tensor([1.0], dtype=torch.float64)

        delays = torch.tensor([[0.0]], dtype=torch.float64)
        apodization = torch.tensor([[1.0]], dtype=torch.float64)

        rf, frequencies = compute_rf_signal(
            transducer, scatterers, reflectivity, delays, apodization, fs, n_fft, c=c
        )

        rf_signal = rf[:, 0, 0]
        envelope = hilbert_envelope(rf_signal)
        peak_idx = torch.argmax(envelope)

        t_measured = peak_idx.item() / fs
        t_theory = 2 * zs / c

        tolerance = 1.0 / fs
        self.assertLessEqual(abs(t_measured - t_theory), tolerance)
        print(f"✅ Tests de Time-of-Flight (Milestone 14) superados con éxito. t_theory={t_theory*1e6:.2f} us, t_measured={t_measured*1e6:.2f} us")

if __name__ == "__main__":
    unittest.main()
