def forward(x_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = x_array[i] * (x_array[i] > 0)
    return None

def backward(dx_array, out_array):
    for i in range(len(out_array)):
        out_array[i] = dx_array[i] * (out_array[i] > 0)
    return None
