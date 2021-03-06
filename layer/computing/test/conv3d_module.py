import sys
sys.path.append(__file__.replace("NN\\layer\\computing\\test\\conv3d_module.py", ""))

import tensor
import NN.layer.computing.conv3d_module as conv3d_module

import time

def test_compare_alg1(data_shape, filter_shape, stride, pad, padding):
    x = tensor.create_gauss(data_shape)
    bias = tensor.create_ones([filter_shape[0]])
    filter = tensor.create_gauss(filter_shape)
    out1 = tensor.create_zeros(conv3d_module.create_shape(x.shape, filter.shape, stride, pad))
    out2 = tensor.create_zeros(conv3d_module.create_shape(x.shape, filter.shape, stride, pad))
    time1 = time.time_ns()
    #conv3d_module.forward_test(x.array, x.shape, filter.array, filter.shape, bias.array,stride, pad, padding, out1.array, out1.shape)
    time2 = time.time_ns()
    conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, bias.array,stride, pad, padding, out2.array, out2.shape)
    time3 = time.time_ns()
    print('{0}, {1}'.format(time2 - time1, time3 - time2))
    print('{0}'.format(tensor.isSame(out1, out2)))



def test_conv3d_forward(data_shape, filter_shape, stride, pad, padding):
    x = tensor.create_gauss(data_shape)
    b = tensor.create_ones([filter_shape[0]])
    filter = tensor.create_gauss(filter_shape)
    out = tensor.create_zeros(conv3d_module.create_shape(x.shape, filter.shape, stride, pad) )
    conv3d_module.forward(x.array, x.shape, filter.array, filter_shape, b.array, stride, pad, padding, out.array, out.shape)
    print(x)
    print(filter)
    print(b)
    print(out)

def test_conv3d_backward(data_shape, filter_shape, stride, pad, padding):
    x = tensor.create_gauss(data_shape)
    dx = tensor.create_gauss(data_shape)
    dx_diff = x.copy()
    b = tensor.create_ones([filter_shape[0]])
    db = b.copy()
    db_diff = b.copy()
    filter = tensor.create_gauss(filter_shape)
    dfilter = tensor.create_gauss(filter_shape)
    dfilter_diff = dfilter.copy()
    out = tensor.create_gauss(conv3d_module.create_shape(data_shape, filter_shape, stride, pad))
    dout = tensor.create_ones(out.shape)



    conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, b.array, stride, pad, padding, out.array, out.shape)
    #conv3d_module.backward(x.array, dout.array, dout.shape, filter.array, filter.shape, stride, pad, padding, dfilter.array, db.array, dx.array, dx.shape)
    conv3d_module.backward(dout.array,dout.shape, filter.array, filter.shape, stride, pad, dx.array,dx.shape)
    conv3d_module.backward_filter(dout.array, dout.shape, x.array, x.shape, stride, pad, padding, dfilter.array, dfilter.shape)
    conv3d_module.backward_bias(dout.array, dout.shape, db.array)

    for i in range(len(x.array)):
        origen = x.array[i]
        x.array[i] = origen + 0.0001
        conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, b.array, stride, pad, padding, out.array, out.shape)
        out1 = 0
        for j in range(len(out.array)):
            out1 += out.array[j]
        x.array[i] = origen - 0.0001
        conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, b.array, stride, pad, padding, out.array, out.shape)
        out2 = 0
        for j in range(len(out.array)):
            out2 += out.array[j]
        dx_diff.array[i] = (out1 - out2) / (2 * 0.0001)
        x.array[i] = origen
    
    for i in range(len(filter.array)):
        origen = filter.array[i]
        filter.array[i] = origen + 0.0001
        conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, b.array, stride, pad, padding, out.array, out.shape)
        out1 = 0
        for j in range(len(out.array)):
            out1 += out.array[j]
        filter.array[i] = origen - 0.0001
        conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, b.array, stride, pad, padding, out.array, out.shape)
        out2 = 0
        for j in range(len(out.array)):
            out2 += out.array[j]
        dfilter_diff.array[i] = (out1 - out2) / (2 * 0.0001)
        filter.array[i] = origen
    
    for i in range(len(b.array)):
        origen = b.array[i]
        b.array[i] = origen + 0.0001
        conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, b.array, stride, pad, padding, out.array, out.shape)
        out1 = 0
        for j in range(len(out.array)):
            out1 += out.array[j]
        b.array[i] = origen - 0.0001
        conv3d_module.forward(x.array, x.shape, filter.array, filter.shape, b.array, stride, pad, padding, out.array, out.shape)
        out2 = 0
        for j in range(len(out.array)):
            out2 += out.array[j]
        db_diff.array[i] = (out1 - out2) / (2 * 0.0001)
        b.array[i] = origen
    
    
    #print(dx)
    #print(dx_diff)
    #print(dfilter)
    #print(dfilter_diff)
    print(tensor.isSame(dx,dx_diff))
    print(tensor.isSame(dfilter, dfilter_diff))
    print(tensor.isSame(db, db_diff))


#test_conv3d_forward([2,2,5,5], [2,2,3,3],2,1,0)

#test_conv3d_backward([2,2,2,2], [2,2,2,2],1,3,1)

#test_compare_alg1([50,3,28,28], [2, 3,5,5],1,2,0)

test_conv3d_backward([2,2,9,9],[2,2,3,3], 2,1,1)
