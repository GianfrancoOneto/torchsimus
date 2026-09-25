from torch import nn

class Transducer(nn.Module):
    def __init__(self, geometry, element_shape, center_frequency, bandwidth):
        super().__init__()
        self.geometry = geometry
        self.element_shape = element_shape
        self.fc = center_frequency
        self.bandwidth = bandwidth

    def pose(self):
        return self.geometry.pose()
