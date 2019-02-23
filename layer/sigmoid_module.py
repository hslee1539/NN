from .computing import sigmoid_module as computing
from . import interface_module
import tensor

class Sigmoid(interface_module.Forwardable, interface_module.Backwardable):
    def __init__(self):
        self.out = tensor.Tensor([0], [1,1])

    def forward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
        return self.forward_line(x)
    
    def forward_line(self, x):
        computing.forward(x.array, self.out.array)
        return self.out

    def backward(self, dout):
        return self.backward_line(dout)

    def backward_line(self, dout):
        computing.backward(dout.array, self.out.array)
        return self.out
