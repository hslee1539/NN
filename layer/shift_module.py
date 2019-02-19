from .computing import shift_module as computing
from . import interface_module
from ..optimizer.interface_module import Base as Optimizer
import tensor

class Shift(interface_module.Forwardable, interface_module.Backwardable, interface_module.Updatable):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.out = tensor.Tensor([0], [0,1])

        self.dw = w.copy()
        self.db = b.copy()

    def forward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
        return self.forward_line(x)

    def forward_line(self, x):
        computing.forward(x.array, self.w.array, self.b.array, self.out.array)
        return self.out

    def backward(self, dx):
        return self.backward_line(dx)

    def backward_line(self, dx):
        computing.backward(dx.array, self.w.array, self.dw.array, self.db.array, self.out.array)
        return self.out

    def update(self, optimizer = Optimizer()):
        optimizer.update(self.w, self.dw)
        optimizer.update(self.b, self.db)

        return None
    
