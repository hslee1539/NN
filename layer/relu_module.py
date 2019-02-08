from .computing import relu_module as computing
from . import interface_module
import tensor

class Relu(interface_module.Forwardable, interface_module.Backwardable):
    def __init__(self):
        self.out = tensor.Tensor([0], [1,1])

    def forward(self, x):
        if(self.out.shape[-2] != x.shape[-2]):
            self.out = x.copy()

        return self.forward_line(self, x)

    def forward_line(self, x):
        computing.forward(x, self.out)
        return self.out

    def backward(self, dx):
        return self.backward_line(dx)

    def backward_line(self, dx):
        computing.backward(dx, self.out)
        return self.out
    
