from .computing import affine_module as computing
from . import interface_module
import tensor


class Affine(interface_module.Forwardable, interface_module.Backwardable, interface_module.ForwardStartable, interface_module.Updatable):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.out = tensor.Tensor([0],[1,1])
        self.dx = tensor.Tensor([0],[1,1])

        self.dw = w.copy()
        self.db = b.copy()
        

    def forward(self, x):
        if(self.out.shape[-2] != x.shape[-2]):
            self.out = tensor.create_matrix_product(x, self.w)
        
        return self.forward_line(x)
        
    def forward_line(self, x):
        self.x = x # 읽기 전용
        computing.forward(self.x.array, self.w.array, self.b.array, self.out.array)
        return self.out

    def backward(self, dout):
        if(self.dx.shape[-2] != dout.shape[-2]):
            self.dx = self.x.copy()
        
        return self.backward_line(dout)

    def backward_line(self, dout):
        computing.backward(dout.array, self.w.array, self.w.shape, self.dx.array)
        computing.backward_variables(self.x.array, dout.array, self.dw.array, self.db.array)

        return self.dx

    def startForward(self, x):
        if(self.out.shape[-2] != x.shape[-2]):
            self.out = tensor.create_matrix_product(x, self.w)
        
        return self.startForward_line(x)

    def startForward_line(self, x):
        self.x = x # 읽기 전용
        computing.forward(self.x.array, self.w.array, self.b.array, self.out.array)
        return self.out

    def update(self, optimizer):
        optimizer.update(self.w, self.dw)
        optimizer.update(self.b, self.db)

        return None

        
