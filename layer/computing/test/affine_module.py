import sys
sys.path.append(__file__.replace("NN\\layer\\computing\\test\\affine_module.py", ""))
import tensor
import NN
import random
import time
rd = random.randint

def test_alg(data_shape, w_shape):
    x = tensor.create_gauss(data_shape)
    w = tensor.create_gauss(w_shape)
    b = tensor.create_gauss([w_shape[-1]])

    out = tensor.create_matrix_product(x,w)
    out_test = out.copy()
    time1 = time.time_ns()
    NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out.array)
    time2 = time.time_ns()
    NN.layer.computing.affine_module.forward_old1(x.array, w.array, b.array, out_test.array)
    time3 = time.time_ns()
    print('{0}, {1}'.format(time2 - time1, time3 - time2))
    print('{0}'.format(tensor.isSame(out, out_test)))


    
#test_alg([100,300], [300,25])


def test(data_shape, w_shape):
    x = tensor.create_gauss(data_shape)
    w = tensor.create_gauss(w_shape)
    b = tensor.create_gauss([w_shape[-1]])

    out_affine = tensor.create_matrix_product(x,w)
    t = tensor.create_zeros( out_affine.shape.copy(), int)
    for i in range(t.shape[0]):
        t.array[i * t.shape[1] + random.randint(0, t.shape[1] - 1)] = 1 
    
    out_sigmoid = out_affine.copy()

    dw1 = w.copy()
    dw2 = w.copy()
    db1 = b.copy()

    dout_sigmoid = out_affine.copy()

    dout1 = x.copy()
    dout2 = x.copy()

    NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
    NN.layer.computing.sigmoid_module.forward(out_affine.array, out_sigmoid.array)
    NN.layer.computing.cross_entropy_module.backward(out_sigmoid.array, t.array, dout_sigmoid.array)
    NN.layer.computing.sigmoid_module.backward(dout_sigmoid.array, out_sigmoid.array)
    NN.layer.computing.affine_module.backward(out_sigmoid.array, w.array, w.shape, dout1.array)
    #NN.layer.computing.affine_module.backward_variables(x.array, out_sigmoid.array, dw1.array, db1.array)
    NN.layer.computing.affine_module.backward_dw(x.array, x.shape, out_sigmoid.array, dw1.array, dw1.shape)

    for i in range(len(x.array)):
        origen = x.array[i]
        x.array[i] = origen + 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        NN.layer.computing.sigmoid_module.forward(out_affine.array, out_sigmoid.array)
        out1 = NN.layer.computing.cross_entropy_module.forward(out_sigmoid.array, t.array)

        x.array[i] = origen - 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        NN.layer.computing.sigmoid_module.forward(out_affine.array, out_sigmoid.array)
        out2 = NN.layer.computing.cross_entropy_module.forward(out_sigmoid.array, t.array)
        dout2.array[i] = (out1 - out2) / (2 * 0.0001)
        x.array[i] = origen

    for i in range(len(dw2.array)):
        origen = w.array[i]
        w.array[i] = origen + 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        NN.layer.computing.sigmoid_module.forward(out_affine.array, out_sigmoid.array)
        out1 = NN.layer.computing.cross_entropy_module.forward(out_sigmoid.array, t.array)

        w.array[i] = origen - 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        NN.layer.computing.sigmoid_module.forward(out_affine.array, out_sigmoid.array)
        out2 = NN.layer.computing.cross_entropy_module.forward(out_sigmoid.array, t.array)
        dw2.array[i] = (out1 - out2) / (2 * 0.0001)
        w.array[i] = origen
    """
    print("최종 역전파 결과")
    print(dout1)
    print("편미분 최정 결과")
    print(dout2)
    print('최종 w 역전파 결과')
    print(dw1)
    print('최종 w 편미분 결과')
    print(dw2)
    """
    print(tensor.isSame(dout1, dout2))
    print(tensor.isSame(dw1, dw2))

test([5,23],[23,3])

def test2(data_shape, w_shape):
    x = tensor.create_gauss(data_shape)
    w = tensor.create_gauss(w_shape)
    b = tensor.create_gauss([w_shape[-1]])

    out_affine = tensor.create_matrix_product(x,w)
    dout_affine = tensor.create_ones(out_affine.shape)

    dw1 = w.copy()
    dw2 = w.copy()
    db1 = b.copy()

    dx1 = x.copy()
    dx2 = x.copy()

    #NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
    NN.layer.computing.affine_module.backward(dout_affine.array, w.array, w.shape, dx1.array)
    NN.layer.computing.affine_module.backward_variables(x.array, dout_affine.array, dw1.array, db1.array)

    for i in range(len(x.array)):
        origen = x.array[i]
        x.array[i] = origen + 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        out1 = 0
        for j in range(len(out_affine.array)):
            out1 += out_affine.array[j]

        x.array[i] = origen - 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        out2 = 0
        for j in range(len(out_affine.array)):
            out2 += out_affine.array[j]

        dx2.array[i] = (out1 - out2) / (2 * 0.0001)
        x.array[i] = origen

    for i in range(len(dw2.array)):
        origen = w.array[i]
        w.array[i] = origen + 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        out1 = 0
        for j in range(len(out_affine.array)):
            out1 += out_affine.array[j]

        w.array[i] = origen - 0.0001
        NN.layer.computing.affine_module.forward(x.array, w.array, b.array, out_affine.array)
        out2 = 0
        for j in range(len(out_affine.array)):
            out2 += out_affine.array[j]
            
        dw2.array[i] = (out1 - out2) / (2 * 0.0001)
        w.array[i] = origen
    
    print("최종 역전파 결과")
    print(dx1)
    print("편미분 최정 결과")
    print(dx2)
    print('최종 w 역전파 결과')
    print(dw1)
    print('최종 w 편미분 결과')
    print(dw2)

#test2([4,3],[3,2])





    



