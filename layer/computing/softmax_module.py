import math

def forward(x_array, x_shape, out_array):
    """softmax의 순전파를 구합니다.
        backward의 out_array와 공유합니다."""
    col = x_shape[-1]
    row = len(x_array) // col

    sigma = 0
    pass_row = 0
    index = 0
    
    for r in range(row):
        pass_row = r * col
        sigma = 0
        
        for c in range(col):
            index = pass_row + c
            out_array[index] = math.exp(x_array[index]) # backward의 out이 손실됨.
            sigma += out_array[index]
        for c in range(col):
            index = pass_row + c
            out_array[index] /= sigma

    return None

def partialForward(x_array, x_shape, out_array, index, max_index):
    zero = type(out_array[0])(0)
    x1 = x_shape[-1]
    col_range = range(x1)

    for r in range(index * len(x_array) // x1 // max_index, (index + 1) * len(x_array) // x1 // max_index):
        pass_row = r * x1
        sigma = zero

        for c in col_range:
            out_array[pass_row + c] = math.exp(x_array[pass_row + c]) # backward의 out이 손실됨.
            sigma += out_array[pass_row + c]
        for c in col_range:
            out_array[pass_row + c] /= sigma

    return None

def backward(dout_array, dout_shape, out_array):
    """softmax의 순전파 결과를 역전파 결과로 연산합니다.
        forward의 out_array와 공유합니다."""
    col = dout_shape[-1]
    row = len(dout_array) // col
    
    pass_row = 0
    index = 0
    sigma = 0

    for r in range(row):
        pass_row = r * col
        sigma = 0
        
        for c in range(col):
            index = pass_row + c
            sigma += dout_array[index] * out_array[index]
        for c in range(col):
            index = pass_row + c
            out_array[index] *= dout_array[index] - sigma
            
    return None

def partialBackward(dout_array, dout_shape, out_array, index, max_index):
    zero = type(out_array[0])(0)
    colMax = dout_shape[-1]
    col_range = range(colMax)
    rowMax = len(dout_array) // colMax

    for r in range(index * rowMax // max_index, (index + 1) * rowMax // max_index):
        pass_row = r * colMax
        sigma = zero

        for c in col_range:
            sigma += dout_array[pass_row + c] * out_array[pass_row + c]
        for c in col_range:
            out_array[pass_row + c] *= dout_array[pass_row + c] - sigma
        
    return None

def backward_with_table(t_array, out_array):
    """softmax의 순전파 결과를 역전파 결과로 연산합니다.
        table을 가지고 cross entropy로 역전파를 시작합니다.
        params t_array
        one-hot인 table입니다.
        params out_array
        softmax의 순전결과 배열입니다. 또, 역전파 결과를 이곳에 저장합니다."""
    length = len(t_array)

    for i in range(length):
        out_array[i] -= t_array[i]
    return None

def partialBackward_with_table(t_array, out_array, index, max_index):
    for i in range(index * len(t_array) // max_index, (index + 1) * len(t_array) // max_index):
        out_array[i] -= t_array[i]
    return None

def backward_with_table_and_dout(dout_array, t_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = (out_array[i] - t_array[i]) * dout_array[i]
    return None

def partialBackward_with_table_and_dout(dout_array, t_array, out_array, index, max_index):
    for i in range(index * len(out_array) // max_index, (index + 1) * len(out_array) // max_index):
        out_array[i] = (out_array[i] - t_array[i]) * dout_array[i]
    return None