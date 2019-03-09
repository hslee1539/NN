def forward(x_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = x_array[i] * (x_array[i] > 0)
    return None

def backward(dout_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = dout_array[i] * (out_array[i] > 0)
    return None

def partialForward(x_array, out_array, index, max_index):
    for i in range(index * len(out_array) // max_index, (index + 1) * len(out_array) // max_index):
        out_array[i] = x_array[i] * (x_array[i] > 0)
    return None

def partialBackward(dout_array, out_array, index, max_index):
    for i in range(index * len(out_array) // max_index, (index + 1) * len(out_array) // max_index):
        out_array[i] = dout_array[i] * (out_array[i] > 0)
    return None
