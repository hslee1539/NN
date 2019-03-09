from .computing import shift_module as computing
from . import interface_module
from ..optimizer.interface_module import Base as Optimizer
import tensor

class Shift(interface_module.PartialUpdatable, interface_module.Updatable):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.out = tensor.Tensor([0], [0,1])

        self.dw = w.copy()
        self.db = b.copy()
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
        computing.forward(self.x.array, self.w.array, self.b.array, self.out.array)
        return self.out
    
    def initBackward(self, dout, t = None):
        self.dout = dout
        return self.out
    
    def backward(self):
        computing.backward(self.dout.array, self.w.array, self.out.array)
        computing.backward_variables(self.dout.array, self.x.array, self.dw.array, self.db.array)
        return self.out
    
    def initPartial(self, max_index):
        self.max_index = max_index

    def partialForward(self, index):
        computing.partialForward(self.x.array, self.w.array, self.b.array, self.out.array, index, self.max_index)
        yield 0

    
    def partialBackward(self, index):
        computing.partialBackward(self.dout.array, self.w.array, self.out.array, index, self.max_index)
        computing.partialBackward_variables(self.dout.array, self.x.array, self.dw.array, self.db.array, index, self.max_index)
        yield 0


    def update(self, optimizer = Optimizer()):
        optimizer.update(self.w, self.dw)
        optimizer.update(self.b, self.db)
        return None
    
    def partialUpdate(self, optimizer, index):
        optimizer.partialUpdate(self.w, self.dw, index, self.max_index)
        optimizer.partialUpdate(self.b, self.db, index, self.max_index)
        yield 0
    
