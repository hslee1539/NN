import sys
sys.path.append(__file__.replace("NN\\layer\\computing\\test\\conv3d_module.py", ""))

import tensor
import NN.layer.computing.conv3d_module as conv3d_module

def test_conv3d_forward(data_shape, filter_shape, stride, padding, pad):
    x = tensor.create_gauss(data_shape)
    b = tensor.create_ones([filter_shape[0]])
    filter = tensor.create_gauss(filter_shape)
    out = tensor.create_zeros(conv3d_module.create_shape(x.shape, filter.shape, stride, padding) )
    conv3d_module.forward(x.array, x.shape, filter.array, filter_shape, b.array, stride, padding, pad, out.array, out.shape)
    print(x)
    print(filter)
    print(b)
    print(out)

test_conv3d_forward([2,1,2,2], [2,1,2,2],1,1,0)

