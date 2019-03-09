def forward(x_array, w_array, b_array, out_array):
    column = len(w_array)
    for i in range(len(x_array)):
        out_array[i] = x_array[i] * w_array[i % column] + b_array[i % column]
    return None

def backward(dout_array, w_array, out_array):
    column = len(w_array)
    for i in range(len(out_array)):
        out_array[i] = dout_array[i] * w_array[i % column]
    return None

def backward_variables(dout_array, x_array, dw_array, db_array):
    zero = type(dw_array[0])(0)

    column = len(dw_array)
    row_range = range(len(dout_array) // column)
    for c in range(column):
        tmp_dw = zero
        tmp_db = zero
        for r in row_range:
            tmp_dw += dout_array[c + r * column] * x_array[c + r * column]
            tmp_db += dout_array[c + r * column]
        dw_array[c] = tmp_dw
        db_array[c] = tmp_db
    return None

def partialForward(x_array, w_array, b_array, out_array, index, max_index):
    column = len(w_array)
    for i in range(index * len(x_array) // max_index, (index + 1) * len(x_array) // max_index):
        out_array[i] = x_array[i] * w_array[i % column] + b_array[i % column]
    return None

def partialBackward(dout_array, w_array, out_array, index, max_index):
    """0축 기준으로 나눔"""
    column = len(w_array)
    for i in range(index * len(out_array) // max_index, (index + 1) * len(out_array) // max_index):
        out_array[i] = dout_array[i] * w_array[i % column]
    return None

def partialBackward_variables(dout_array, x_array, dw_array, db_array, index, max_index):
    """0축을 기준으로 나눔"""
    zero = type(dw_array[0])(0)

    column = len(dw_array)
    row_range = range(len(dout_array) // column)
    for c in range(index * column // max_index, (index + 1) * column // max_index):
        tmp_dw = zero
        tmp_db = zero
        for r in row_range:
            tmp_dw += dout_array[c + r * column] * x_array[c + r * column]
            tmp_db += dout_array[c + r * column]
        dw_array[c] = tmp_dw
        db_array[c] = tmp_db
    return None
