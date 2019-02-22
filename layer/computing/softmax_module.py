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

def startBackward(t_array, out_array):
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
