from .computing import softmax_module as computing
from . import interface_module
import tensor

class Softmax(interface_module.Forwardable, interface_module.Backwardable, interface_module.Learnable):
    def __init__(self):
        self.out = tensor.Tensor([0], [1,1])

    def forward(self, x):
        if(self.out.shape[-2] != x.shape[-2]):
            self.out = x.copy()

        return self.forward_line(x)
    
    def forward_line(self, x):
        computing.forward(x.array, x.shape, self.out.array)
        return self.out

    def backward(self, dx):
        return self.backward_line(dx)

    def backward_line(self, dx):
        computing.forward(dx.array, dx.shape, self.out.array)
        return self.out

    def learn(self, t):
        return self.learn_line(t)

    def learn_line(self, t):
        computing.learn(t.array, self.out.array)
        return self.out
