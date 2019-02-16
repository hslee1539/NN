import sys
sys.path.append(__file__.replace("NN\\layer\\computing\\test_module.py", ""))

import tensor
import affine_module as affine
import batchnormalization_module as norm
import math

def isSame(x,y):
    return x - y

def affine_forward(x_shape, w_shape):
    x = tensor.create_gauss(x_shape)
    w = tensor.create_gauss(w_shape)
    b = tensor.create_gauss([w_shape[-1]])
    standard_out = tensor.create_matrix_product(x,w)
    new_out = tensor.create_zeros(standard_out.shape.copy(), float)
    compare_out = tensor.create_zeros(standard_out.shape.copy(), bool)

    tensor.matmul(x,w,standard_out)
    tensor.add(standard_out, b, standard_out)
    affine.forward(x.array, w.array, b.array, new_out.array)

    tensor.function_element_wise(standard_out, new_out, isSame, compare_out)

    print(compare_out)

def affine_backward(x_shape, w_shape):
    """backward에서 뒷쪽 노드에 보내줄 dout을 테스트합니다."""
    x = tensor.create_gauss(x_shape)
    w = tensor.create_gauss(w_shape)
    w_t = tensor.create_transpose(w)
    forward_out = tensor.create_matrix_product(x,w)
    forward_out = tensor.create_gauss(forward_out.shape)
    standard_out = tensor.create_matrix_product(forward_out, w_t)
    new_out = tensor.create_zeros(standard_out.shape.copy(), float)
    compare_out = tensor.create_zeros(standard_out.shape.copy(), bool)

    #잘 알려진 방법
    tensor.transpose(w, w_t)
    tensor.matmul(forward_out, w_t, standard_out)

    #새로운 방법
    affine.backward(forward_out.array, w.array, w.shape, new_out.array)

    tensor.function_element_wise(standard_out, new_out, isSame, compare_out)

    print(compare_out)

def affine_backward_variables(x_shape, w_shape):
    """backward에서 업데이트를 위한 변수들의 미분값 테스트입니다."""
    x = tensor.create_gauss(x_shape)
    w = tensor.create_gauss(w_shape)
    b = tensor.create_gauss([w_shape[-1]])
    x_t = tensor.create_transpose(x)
    forward_out = tensor.create_matrix_product(x,w)
    forward_out = tensor.create_gauss(forward_out.shape)
    standard_dw = tensor.create_zeros(w.shape, float)
    new_dw = tensor.create_zeros(w.shape, float)
    compare_dw = tensor.create_zeros(w.shape, bool)
    standard_db = tensor.create_zeros(b.shape, float)
    new_db = tensor.create_zeros(b.shape, float)
    compare_db = tensor.create_zeros(b.shape, float)

    #잘 알려진 방법
    tensor.transpose(x, x_t)
    tensor.matmul(x_t, forward_out, standard_dw)
    tensor.sum_axis(forward_out, 0, standard_db)

    #새로운 방법
    affine.backward_variables(x.array, forward_out.array, new_dw.array, new_db.array)

    tensor.function_element_wise(standard_dw, new_dw, isSame, compare_dw)
    tensor.function_element_wise(standard_db, new_db, isSame, compare_db)

    print(compare_dw)
    print(compare_db)

def batch_nrom_forward(x_shape):
    def process_no_zero(x):
        return x + 10e-7
    x = tensor.create_randomly(x_shape, -3,4)
    #x = tensor.Tensor([1.,2.,3.,5.],[1,4])
    mean = tensor.create_sum(x, 0)
    deviation = tensor.create_element_wise_product(x,mean)
    jegop = tensor.create_element_wise_product(deviation, deviation)
    dispersion = tensor.create_sum(jegop, 0)
    dispersion2 = dispersion.copy()
    std = dispersion.copy()
    
    forward_out = x.copy()
    forward_new_out = x.copy()
    compare_out = x.copy()

    print(x)
    #잘 알려진 방법
    tensor.mean_axis(x, 0, mean)
    print(mean)
    tensor.sub(x, mean, deviation)
    tensor.mul(deviation, deviation, jegop)
    tensor.mean_axis(jegop, 0, dispersion)
    tensor.function(dispersion, process_no_zero, dispersion)
    tensor.function(dispersion, math.sqrt, dispersion)
    tensor.div(deviation, dispersion, forward_out)

    #새로운 방법
    norm.forward(x.array, x.shape, dispersion2.array, forward_new_out.array)

    tensor.function_element_wise(forward_out, forward_new_out, isSame, compare_out)

    print(compare_out)
    
    pass
    
def test_norm_backward(x_shape, h = 0.001):
    def process_no_zero(x):
        return x + 10e-7
    x = tensor.create_randomly(x_shape, -3,4)
    #x = tensor.Tensor([1.,2.,3.,5.],[1,4])
    mean = tensor.create_sum(x, 0)
    d_mean = mean.copy()
    d_mean2 = mean.copy()
    deviation = tensor.create_element_wise_product(x,mean)
    jegop = tensor.create_element_wise_product(deviation, deviation)
    print(jegop.shape)
    dispersion = tensor.create_sum(jegop, 0)
    print(dispersion.shape)
    dispersion2 = dispersion.copy()
    d_dispersion = dispersion.copy()
    d_deviation = deviation.copy()

    batch_size = tensor.Tensor([x_shape[0]], [1])

    tmp_x = x.copy()
    
    forward_out = x.copy()
    forward_new_out = x.copy()
    backward_out = x.copy()
    backward_new_out = x.copy()
    compare_out = x.copy()
    #dx = tensor.create_ones(x.shape)
    dx = tensor.create_randomly(x.shape)

    print(x)
    #잘 알려진 방법
    #forward
    tensor.mean_axis(x, 0, mean)
    print(mean)
    tensor.sub(x, mean, deviation)
    tensor.mul(deviation, deviation, jegop)
    tensor.mean_axis(jegop, 0, dispersion)
    tensor.function(dispersion, process_no_zero, dispersion)
    tensor.function(dispersion, math.sqrt, dispersion)
    tensor.div(deviation, dispersion, forward_out)
    #backward
    tensor.div(dx, dispersion, d_deviation)
    tensor.mul(dx, deviation, tmp_x)
    tensor.div(tmp_x, dispersion, tmp_x)
    tensor.sum_axis(tmp_x, 0, d_dispersion)
    tensor.div(tmp_x, dispersion, tmp_x)
    tensor.sum_axis(tmp_x, 0, d_dispersion) #
    def mul_minus(x):
        return -x
    tensor.function(d_dispersion, mul_minus, d_dispersion)
    
    tensor.div(deviation, dispersion, tmp_x)
    tensor.mul(tmp_x, d_dispersion, tmp_x)
    tensor.div(tmp_x, batch_size, tmp_x)
    tensor.add(tmp_x, d_deviation, d_deviation)
    tensor.sum_axis(d_deviation, 0, d_mean)
    tensor.div(d_mean, batch_size, d_mean)
    tensor.sub(d_deviation, d_mean, backward_out)

    
    #새로운 방법
    norm.forward(x.array, x.shape, dispersion2.array, forward_new_out.array)
    backward_new_out = forward_new_out.copy()
    norm.backward(dx.array, dispersion2.array, backward_new_out.array)
    
    tensor.function_element_wise(backward_out, backward_new_out, isSame, compare_out)
    print('back = ')
    print(compare_out)
    tensor.function_element_wise(forward_out, forward_new_out, isSame, compare_out)
    print('forward = ')
    print(compare_out)
    dispersion_compare = dispersion.copy()
    tensor.function_element_wise(dispersion, dispersion2, isSame, dispersion_compare)
    print('dispersion = ')
    print(dispersion_compare)
