from .computing import batchnormalization_module as computing
from . import interface_module
import tensor

class Batchnormalization(interface_module.PartialBackwardable):
    """정규화 층입니다. 0차원(데이터 셋)을 제외한 나머지 차원들의 값을 0차원 """
    def __init__(self):
        self.out = tensor.Tensor([0], [0,1])
        self.dispersion = None
        self.x = None
        self.dout = None
        self.max_index = 0 

    def initForward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
            self.dispersion = tensor.create_zeros([len(x.array) // x.shape[0]])
        self.x = x
        return self.out
    
    def setX(self, x):
        self.x = x
        return self

    def forward(self):
        computing.forward(self.x.array, self.x.shape, self.dispersion.array, self.out.array)
        return self.out

    def initBackward(self, dout, t = None):
        self.dout = dout
        return self.out

    def backward(self):
        computing.backward(self.dout.array, self.dispersion.array, self.out.array)
        return self.out

    def initPartial(self, max_index):
        self.max_index = max_index
        return None
    
    def partialForward(self, index):
        computing.partialForward(self.x.array, self.x.shape, self.dispersion.array, self.out.array, index, self.max_index)
        yield 0

    def partialBackward(self, index):
        computing.partialBackward(self.dout.array, self.dispersion.array, self.out.array, index, self.max_index)
        yield 0

    
