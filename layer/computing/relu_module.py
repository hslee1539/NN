def forward(x_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = x_array[i] * (x_array[i] > 0)
    return None

def backward(dout_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = dout_array[i] * (out_array[i] > 0)
    return None
