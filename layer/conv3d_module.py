from .computing import conv3d_module as computing
from . import interface_module
import tensor

class Conv3d(interface_module.Forwardable, interface_module.Backwardable, interface_module.Updatable):
    def __init__(self, filter, bias, stride, pad, padding):
        self.filter = filter
        self.bias = bias
        self.dfilter = filter.copy()
        self.dbias = bias.copy()
        self.out = tensor.Tensor([1], [1,1,1,1])
        self.x = None
        self.dx = tensor.Tensor([1], [1,1,1,1])
        self.stride = stride
        self.pad = pad
        self.padding = padding

    def forward(self, x):
        if(x.shape[0] != self.out.shape[0]):
            self.out = tensor.create_zeros(computing.create_shape(x.shape, self.filter.shape, self.stride, self.pad))
        return self.forward_line(x)
    
    def forward_line(self, x):
        self.x = x
        computing.forward(self.x, self.x.shape, self.filter.array, self.filter.shape, self.bias.array, self.stride, self.pad, self.padding, self.out.array, self.out.shape)
        return self.out
    
    def backward(self, dout):
        if(dout.shape[0] != self.dx.shape[0]):
            if(dout.shape[0] != self.x.shape[0]):
                raise 'forward를 먼저 하시오....'
            self.dx = self.x.copy()
        return self.backward_line(dout)
    
    def backward_line(self, dout):
        computing.backward(self.x.array, dout.array, dout.shape, self.filter.array, self.filter.shape, self.stride, self.pad, self.padding, self.dfilter.array, self.dbias.array, self.dx.array, self.dx.shape)
        return self.dx
    
    def update(self, optimizer):
        optimizer.update(self.filter, self.dfilter)
        optimizer.update(self.bias, self.dbias)
        return None

            


        
