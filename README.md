# PyTorch Ultrasound Simulator (`torchsimus`)

`torchsimus` es un simulador de ultrasonido de alta precisión orientado a PyTorch.

## Unidades (Sistema Internacional)
Todas las magnitudes internas del simulador se expresan estrictamente en unidades del SI:
* **Posiciones espaciales ($x, y, z$):** Metros ($\mathrm{m}$)
* **Tiempo ($t$):** Segundos ($\mathrm{s}$)
* **Frecuencia ($f$):** Hertz ($\mathrm{Hz}$)
* **Velocidad del sonido ($c$):** Metros por segundo ($\mathrm{m/s}$)

## Dimensiones y Convenciones de Tensores
El framework opera estructurando los tensores principales de la siguiente manera:
* `scatterer_positions`: $[S, 3]$ (o $[B, S, 3]$ en modo batch)
* `reflectivity`: $[S]$ (o $[B, S]$)
* `element_centers`: $[N, 3]$
* `element_frames`: $[N, 3, 3]$
* `tx_delays`: $[E, N]$ (o $[B, E, N]$)
* `tx_apodization`: $[E, N]$ (o $[B, E, N]$)
* `frequencies`: $[F]$
* `spectrum`: $[F, N_{rx}, E]$
* `RF` de `Simus`: $[N_t, E, N]$ (o $[B, N_t, E, N]$)
* `RF` de `simulation.simus`: $[N_t, N]$ (una emisión; varias filas de retardos = emisiones simultáneas, MLT)

## Convención de Fourier
Las ecuaciones analíticas internas emplean la dependencia temporal $e^{-i\omega t}$ y la propagación $e^{+ikr}$. Dado que la FFT estándar de PyTorch difiere en signo, se debe aplicar explícitamente la conversión de conjugado antes de ejecutar `irfft()`:

```python
def physical_to_torch_spectrum(x):
    return x.conj()
```

## Instalación

```bash
pip install git+https://github.com/GianfrancoOneto/torchsimus.git
pip install torchsimus-0.2.0-py3-none-any.whl     # desde el wheel generado
pip install -e /ruta/a/torchsimus                 # desde el código fuente (modo desarrollo)
```

```python
from torchsimus.probes import get_probe
from torchsimus.field import pfield
```

## Organización del paquete

| carpeta / archivo | contenido | equivalente en MUST / pymust |
|---|---|---|
| `geometry/` | `LinearArray`, `ConvexArray`, `MatrixArray`, `RingArray`, `CustomArray`, `CircularSingleElement` | `getElementPositions` |
| `elements/` | `PointElement`, `RectangularElement` (sinc × factor de baffle), `CircularElement` | directividad + `baffle` |
| `physics/` | geometría local, propagación 2-D, atenuación, elevación (MGBM) | — |
| `cabezales/` | 35 cabezales de ejemplo en CSV | — |
| `spectra.py` | espectro del pulso (seno o chirp), de la sonda y pulso-eco | `getPulseSpectrumFunction`, `getProbeFunction` |
| `pulse.py` | `getpulse` | `getpulse` |
| `delays.py` | `txdelay_focus`, `txdelay_plane`, `txdelay_diverging`, `txdelay3_focus`, `txdelay3_plane`, `txdelay3_diverging`, `txdelay3_focus_line` | `txdelay`, `txdelay3` |
| `probes.py` | las 10 sondas de MUST, `get_probe`, `matrix_probe` | `getparam` |
| `phantoms.py` | `genscat` (2-D y 3-D) | `genscat` |
| `simulator.py`, `chunking.py` | `Simus`: modelo 2-D rápido, con batch y modos directo/scattering | — |
| `simulation.py` | `simus`, `simus3`: RF con la física completa de MUST | `simus`, `simus3` |
| `field.py` | `pfield`, `pfield3` (con chirp) | `pfield`, `pfield3` |
| `movie.py` | `mkmovie` (con scatterers y chirp), `save_gif` | `mkmovie` |
| `processing/rf2iq.py` | `rf2iq` | `rf2iq` |
| `processing/das.py` | `das`, `das3` (6 interpolaciones, f-number automático, pasivo, `rx_angle`, fuente virtual) | `dasmtx`, `dasmtx3` |
| `processing/tgc.py` | `tgc` | `tgc` |
| `processing/doppler.py` | `iq2doppler`, `bmode`, `nyquist_velocity` | `iq2doppler`, `bmode`, `getNyquistVelocity` |
| `processing/sptrack.py` | `sptrack` (`subpix="PF"` u `"OF"`) | `sptrack` |
| `processing/smoothn.py` | `smoothn` completo (N-D, órdenes 0-2, pesos robustos) | `smoothn` |
| `processing/grids.py` | `impolgrid` | `impolgrid` |
| `visualization.py` | figuras de campos, cortes 3-D, Doppler, películas | — |

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests          # un archivo por milestone: tests/test_mXX_*.py
```
