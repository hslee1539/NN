import sys
sys.path.append(__file__.replace("NN/layer/computing/test.py", ""))

import tensor
import affine

def isSame(x,y):
    #print(str(x) + ', ' + str(y))
    return x == y

def forward(x_shape, w_shape):
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

def backward(x_shape, w_shape):
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

def backward_variables(x_shape, w_shape):
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
    
