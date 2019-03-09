from .computing import softmax_module as computing
from . import interface_module
import tensor

class Softmax(interface_module.PartialBackwardable):
    def __init__(self):
        self.out = tensor.Tensor([0], [1,1])
        self.x = None
        self.dout = None
        self.t = None
        self.max_index = 0

    def setX(self, x):
        self.x = x
        return self

    def setT(self, t):
        self.t = t
        return self
    
    def _backward(self):
        computing.backward(self.dout.array, self.dout.shape, self.out.array)
        return self.out
    
    def _partialBackward(self, index):
        computing.partialBackward(self.dout.array, self.dout.shape, self.out.array, index, self.max_index)
        yield 0


    def _backward_with_cross_entropy(self):
        """자체적으로 cross entropy를 합니다."""
        computing.backward_with_table(self.t.array, self.out.array)
        return self.out
    
    def _partialBackward_with_cross_entropy(self, index):
        computing.partialBackward_with_table(self.t.array, self.out.array, index, self.max_index)
        yield 0


    def _backward_with_rnn(self):
        computing.backward_with_table_and_dout(self.dout.array, self.t.array, self.out.array)
        return self.out
    
    def _partialBackward_with_rnn(self, index):
        computing.partialBackward_with_table_and_dout(self.dout.array, self.t.array, self.out.array, index, self.max_index)
        yield 0


    def initForward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = x.copy()
        self.x = x
        return self.out

    def forward(self):
        computing.forward(self.x.array, self.x.shape, self.out.array)
        return self.out
    
    def initBackward(self, dout, t):
        if(type(t) == tensor.Tensor):
            # true이면 table이 있다고 판단
            if(type(dout) == tensor.Tensor):
                if(len(dout.array) == len(t.array)):
                    # table과 앞 레이어의 역전파 값과 shape이 같다고 판단
                    self.backward = self._backward_with_rnn
                    self.partialBackward = self._partialBackward_with_rnn
            else:
                self.backward = self._backward_with_cross_entropy
                self.partialBackward = self._partialBackward_with_cross_entropy
        else:
            self.backward = self._backward
            self.partialBackward = self._partialBackward

        self.dout = dout
        self.t = t
        return self.out
    
    def initPartial(self, max_index):
        self.max_index = max_index

    def partialForward(self, index):
        computing.partialForward(self.x.array, self.x.shape, self.out.array, index, self.max_index)
        yield 0

    
