import math

def forward(x_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = 1 / (1 + math.exp(-x_array[i]))
    return None

def backward(dout_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = dout_array[i] * (1 - out_array[i]) * out_array[i]
    return None

def partialForward(x_array, out_array, index, max_index):
    for i in range(index * len(x_array) // max_index, (index + 1) * len(x_array) // max_index):
        out_array[i] = 1 / (1 + math.exp(-x_array[i]))
    return None

def partialBackward(dout_array, out_array, index, max_index):
    for i in range(index * len(dout_array) // max_index, (index + 1) * len(dout_array) // max_index):
        out_array[i] = dout_array[i] * (1 - out_array[i]) * out_array[i]
    return None