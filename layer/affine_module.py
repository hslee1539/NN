from .computing import affine_module as computing
from . import interface_module
import tensor


class Affine(interface_module.Updatable, interface_module.PartialUpdatable):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dout = None
        self.max_index = 0
        self.out = tensor.Tensor([0],[1,1])
        self.dx = tensor.Tensor([0],[1,1])
        self.dw = w.copy()
        self.db = b.copy()
    
    def setX(self, x):
        self.x = x
        return self

    def initForward(self, x):
        if(self.out.shape[0] != x.shape[0]):
            self.out = tensor.create_matrix_product(x, self.w)
        self.x = x # 읽기 전용
        return self.out
        
    def forward(self):
        computing.forward(self.x.array, self.w.array, self.b.array, self.out.array)
        return self.out

    def initBackward(self, dout, t = None):
        if(self.dx.shape[0] != dout.shape[0]):
            self.dx = self.x.copy()
        self.dout = dout
        return self.dx

    def backward(self):
        computing.backward(self.dout.array, self.w.array, self.w.shape, self.dx.array)
        computing.backward_dw(self.x.array, self.x.shape, self.dout.array, self.dw.array, self.dw.shape)
        computing.backward_db(self.dout.array, self.db.array)
        return self.dx

    def initPartial(self, max_index):
        self.max_index = max_index
        return None

    def partialForward(self, index):
        computing.partialForward(self.x.array, self.w.array, self.b.array, self.out.array, index, self.max_index)
        yield 0
    
    def partialBackward(self, index):
        computing.partialBackward(self.dout.array, self.w.array, self.w.shape, self.dx.array, index, self.max_index)
        computing.partialBackward_dw(self.x.array, self.x.shape, self.dout.array, self.dw.array, self.dw.shape, index, self.max_index)
        computing.partialBackward_db(self.dout.array, self.db.array, index, self.max_index)
        yield 0
        

    def update(self, optimizer):
        #computing.backward_variables(self.x.array, self.dout.array, self.dw.array, self.db.array)
        optimizer.update(self.w, self.dw)
        optimizer.update(self.b, self.db)
        return None
    
    def partialUpdate(self, optimizer, index):
        optimizer.partialUpdate(self.w, self.dw, index, self.max_index)
        optimizer.partialUpdate(self.b, self.db, index, self.max_index)
        yield 0


        
