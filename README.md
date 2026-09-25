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
* `RF`: $[N_t, N_{rx}, E]$ (o $[B, N_t, E, N]$)

## Convención de Fourier
Las ecuaciones analíticas internas emplean la dependencia temporal $e^{-i\omega t}$ y la propagación $e^{+ikr}$. Dado que la FFT estándar de PyTorch difiere en signo, se debe aplicar explícitamente la conversión de conjugado antes de ejecutar `irfft()`:

```python
def physical_to_torch_spectrum(x):
    return x.conj()
