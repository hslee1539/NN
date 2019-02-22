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