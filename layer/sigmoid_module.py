from .computing import sigmoid_module as computing
from . import interface_module
import tensor

class Sigmoid(interface_module.PartialBackwardable):
    def __init__(self):
        self.out = tensor.Tensor([0], [1,1])
        self.x = None
        self.dout = None
        self.max_index = 0

    def setX(self, x):
        self.x = x
        return self

    def initForward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
        self.x = x
        return self.out

    def forward(self):
        computing.forward(self.x.array, self.out.array)
        return self.out
    
    def initBackward(self, dout, t = None):
        self.dout = dout
        return self.out

    def backward(self):
        computing.backward(self.dout.array, self.out.array)
        return self.out
    
    def initPartial(self, max_index):
        self.max_index = max_index

    def partialForward(self, index):
        computing.partialForward(self.x.array, self.out.array, index, self.max_index)
        yield 0
    
    def partialBackward(self, index):
        computing.partialBackward(self.dout.array, self.out.array, index, self.max_index)
        yield 0

    