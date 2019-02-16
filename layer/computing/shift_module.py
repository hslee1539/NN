def forward(x_array, w_array, b_array, out_array):
    column = len(w_array)
    for i in range(len(x_array)):
        out_array[i] = x_array[i] * w_array[i % column] + b_array[i % column]
    return None

def backward(dx_array, w_array, dw_array, db_array, out_array):
    column = len(w_array)
    row = len(dx_array) // column
    index = 0
    
    for c in range(column):
        db_array[c] = 0
        dw_array[c] = 0
        for r in range(row):
            index = r * column + c
            db_array[c] += dx_array[index]
            dw_array[c] += dx_array[index] * out_array[index]
            out_array[index] = dx_array[index] * w_array[c]