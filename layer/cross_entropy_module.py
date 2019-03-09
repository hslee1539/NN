from .computing import cross_entropy_module as computing
from . import interface_module
import tensor

class CrossEntropy (interface_module.PartialBackwardable):
    """크로스 엔트로피 레이어입니다. forward의 결과를 그대로 보냅니다."""
    def __init__(self):
        self.dx = tensor.Tensor([0], [1,1])
        self.x = None
        self.out = 0
        self.dout = None
        self.t = None
        self.max_index = 0

    def setX(self, x):
        self.x = x
        return self
    
    def setT(self, t):
        self.t = t
        return self
    
    def initForward(self, x):
        self.x = x
        return self.x
    
    def forward(self):
        return self.x
    
    def initBackward(self, dout, t):
        if(self.dx.shape[0] != t.shape[0]):
            self.dx = tensor.create_zeros(t.shape)
        
        if(type(dout) == tensor.Tensor):
            if(len(dout.array) == len(t.array)):
                self.backward = self._backward2
                self.partialBackward = self._partialBackward2
        else:
            self.backward = self._backward1
            self.partialBackward = self._partialBackward1
            
        self.dout = dout
        self.t = t
        return self.dx
    
    def _backward1(self):
        computing.backward(self.x.array, self.t.array, self.dx.array)
        return self.dx
    
    def _backward2(self):
        computing.backward_with_dout(self.dout.array, self.x.array, self.t.array, self.dx.array)
        return self.dx
    
    def initPartial(self, max_index):
        self.max_index = max_index
    
    def partialForward(self, index):
        yield 0

    def _partialBackward1(self, index):
        computing.partialBackward(self.x.array, self.t.array, self.dx.array, index, self.max_index)
        yield 0
    
    def _partialBackward2(self, index):
        computing.partialBackward_with_dout(self.dout.array, self.x.array, self.t.array, self.dx.array, index, self.max_index)
        yield 0
    
