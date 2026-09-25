from torch import nn

class ElementShape(nn.Module):
    def directivity(self, x_local, y_local, z_local, r, frequencies):
        raise NotImplementedError
