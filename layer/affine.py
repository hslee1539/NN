from .computing import affine as computing
from . import interface

class Affine(interface.Forwardable, interface.Backwardable, interface.Updatable):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.out = None
        self.dout = None

    def forward(self, x):
        self.out = tensor.create_matrix_product(x, self.w)

        self.x = x # 읽기 전용
        
        
    def forward_line(self, x):
        pass
