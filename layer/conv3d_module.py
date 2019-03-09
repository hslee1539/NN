from .computing import conv3d_module as computing
from . import interface_module
import tensor

class Conv3d(interface_module.Updatable,interface_module.PartialUpdatable):
    def __init__(self, filter, bias, stride, pad, padding):
        self.filter = filter
        self.bias = bias
        self.dfilter = filter.copy()
        self.dbias = bias.copy()
        self.out = tensor.Tensor([1], [1,1,1,1])
        self.x = None
        self.dout = None
        self.dx = tensor.Tensor([1], [1,1,1,1])
        self.stride = stride
        self.pad = pad
        self.padding = padding
        self.max_index = 0
    
    def setX(self, x):
        self.x = x
        return self

    def initForward(self, x):
        if(x.shape[0] != self.out.shape[0]):
            self.out = tensor.create_zeros(computing.create_shape(x.shape, self.filter.shape, self.stride, self.pad))
        self.x = x
        return self.out

    def forward(self):
        computing.forward(self.x.array, self.x.shape, self.filter.array, self.filter.shape, self.bias.array, self.stride, self.pad, self.padding, self.out.array, self.out.shape)
        return self.out
    
    def initBackward(self, dout, t = None):
        if(dout.shape[0] != self.dx.shape[0]):
            if(dout.shape[0] != self.x.shape[0]):
                raise 'forward를 먼저 하시오....'
            self.dx = self.x.copy()
        self.dout = dout
        return self.dx
    
    def backward(self):
        computing.backward(self.dout.array, self.dout.shape, self.filter.array, self.filter.shape, self.stride, self.pad, self.dx.array, self.dx.shape)
        computing.backward_filter(self.dout.array, self.dout.shape, self.x.array, self.x.shape, self.stride, self.pad, self.padding, self.dfilter.array, self.dfilter.shape)
        computing.backward_bias(self.dout.array, self.dout.shape, self.dbias.array)
        return self.dx
    
    def update(self, optimizer):
        optimizer.update(self.filter, self.dfilter)
        optimizer.update(self.bias, self.dbias)
        return None
    
    def initPartial(self, max_index):
        self.max_index = max_index
    
    def partialForward(self, index):
        computing.partialForward(self.x.array, self.x.shape, self.filter.array, self.filter.shape, self.bias.array, self.stride, self.pad, self.padding, self.out.array, self.out.shape, index, self.max_index)
        yield 0
    
    def partialBackward(self, index):
        computing.partialBackward(self.dout.array, self.dout.shape, self.filter.array, self.filter.shape, self.stride, self.pad, self.dx.array, self.dx.shape, index, self.max_index)
        computing.partialBackward_filter(self.dout.array, self.dout.shape, self.x.array, self.x.shape, self.stride, self.pad, self.padding, self.dfilter.array, self.dfilter.shape, index, self.max_index)
        computing.partialBackward_bias(self.dout.array, self.dout.shape, self.dbias.array, index, self.max_index)
        yield 0
    
    def partialUpdate(self, optimizer, index):
        optimizer.partialUpdate(self.filter, self.dfilter, index, self.max_index)
        optimizer.partialUpdate(self.bias, self.dbias, index, self.max_index)
        yield 0
    