import torch
from elements.base import ElementShape

class PointElement(ElementShape):
    def directivity(self, x_local, y_local, z_local, r, frequencies):
        ones = torch.ones_like(r)
        if frequencies.ndim > 0:
            ones = ones[None, ...].expand(frequencies.shape[0], *r.shape)
        return ones
