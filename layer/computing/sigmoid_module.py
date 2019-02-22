import math

def forward(x_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = 1 / (1 + math.exp(-x_array[i]))
    return None

def backward(dout_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = dout_array[i] * (1 - out_array[i]) * out_array[i]
    return None
