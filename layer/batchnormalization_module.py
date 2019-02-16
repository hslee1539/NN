from .computing import batchnormalization_module as computing
from . import interface_module
import tensor

class Batchnormalization(interface_module.Forwardable, interface_module.Backwardable):
    def __init__(self, dispersion):
        self.out = tensor.Tensor([0], [0,1])
        self.dispersion = dispersion

    def forward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
        
        return self.forward_line(self, x)

    def forward_line(self, x):
        computing.forward(x.array, x.shape, self.dispersion.array, self.out.array)
        return self.out

    def backward(self, dx):
        return self.backward_line(self, dx)

    def backward_line(self, dx):
        computing.forward(dx.array, self.dispersion.array, self.out.array)
        return self.out
    
