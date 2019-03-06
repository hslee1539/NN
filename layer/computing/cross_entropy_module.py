import math


def forward(x_array, t_array):
    """loss를 반환함."""
    loss = 0
    for i in range(len(x_array)):
        loss += t_array[i] * math.log(x_array[i])
    return -loss

def backward(x_array, t_array, dx_array):
    """dx에 결과를 저장"""
    for i in range(len(x_array)):
        dx_array[i] = - t_array[i] / x_array[i]
    return None

def backward_with_dout(dout_array, x_array, t_array, dx_array):
    """나만의 RNN cross entropy를 계산합니다. (미완성)"""
    for i in range(len(x_array)):
        dx_array[i] = (- t_array[i] / x_array[i]) * dout_array[i]
    return None

def partialForward(x_array, t_array, loss_map, index, max_index):
    """forward(x_array, t_array)처럼 loss를 반환하지 않고, 분할된 영역의 계산을 loss_map에 저장합니다."""
    zero = type(loss_map[0])(0)
    tmp = zero
    for i in range(index * len(x_array) // max_index, (index + 1) * len(x_array) // max_index):
        tmp = t_array[i] * math.log(x_array[i])
    loss_map[index] = tmp
    return None

def partialBackward(x_array, t_array, dx_array, index, max_index):
    for i in range(index * len(x_array) // max_index, (index + 1) * len(x_array) // max_index):
        dx_array[i] = - t_array[i] / x_array[i]
    return None

def partialBackward_with_dout(dout_array, x_array, t_array, dx_array, index, max_index):
    for i in range(index * len(x_array) // max_index, (index + 1) * len(x_array) // max_index):
        dx_array[i] = (- t_array[i] / x_array[i]) * dout_array[i]
    return None
