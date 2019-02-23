from .computing import softmax_module as computing
from . import interface_module
import tensor

class Softmax(interface_module.Forwardable, interface_module.Backwardable, interface_module.BackwardStartable):
    def __init__(self):
        self.out = tensor.Tensor([0], [1,1])

    def forward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()

        return self.forward_line(x)
    
    def forward_line(self, x):
        computing.forward(x.array, x.shape, self.out.array)
        return self.out

    def backward(self, dout):
        return self.backward_line(dout)

    def backward_line(self, dout):
        computing.backward(dout.array, dout.shape, self.out.array)
        return self.out

    def startBackward(self, t):
        return self.startBackward_line(t)

    def startBackward_line(self, t):
        computing.startBackward(t.array, self.out.array)
        return self.out
