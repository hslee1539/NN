from .computing import batchnormalization_module as computing
from . import interface_module
import tensor

class Batchnormalization(interface_module.Forwardable, interface_module.Backwardable):
    """정규화 층입니다."""
    def __init__(self):
        self.out = tensor.Tensor([0], [0,1])
        self.dispersion = None

    def forward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
            self.dispersion = tensor.create_zeros([x.shape[0]])
        return self.forward_line(x)

    def forward_line(self, x):
        computing.forward(x.array, x.shape, self.dispersion.array, self.out.array)
        return self.out

    def backward(self, dx):
        return self.backward_line(dx)

    def backward_line(self, dx):
        computing.backward(dx.array, self.dispersion.array, self.out.array)
        return self.out
    
