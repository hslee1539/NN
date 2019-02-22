from .computing import cross_entropy_module as computing
from . import interface_module
import tensor

class CrossEntropy (interface_module.Forwardable, interface_module.BackwardStartable):
    """크로스 엔트로피 레이어입니다. loss계산은 하지 않습니다."""
    def __init__(self):
        self.dx = tensor.Tensor([0], [1,1])
        self.x = None
    
    def forward(self, x):
        self.x = x
        return x
    
    def forward_line(self, x):
        self.x = x
        return x
    
    def startBackward(self, t):
        if(dx.shape[0] != t.shape[0]):
            dx = tensor.create_zeros(t.shape)
        return self.startBackward_line(t)
    
    def startBackward_line(self, t):
        computing.backward(self.x.array, t.array, self.dx.array)
        return self.dx
