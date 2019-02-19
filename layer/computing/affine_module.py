

def forward(x_array, w_array, b_array, out):
    """summary
        2차원 텐서에 최적회 된 matmul로 계산과 동시에 편향 연산을 수행합니다.
        params x_array
        앞 레이어의 결과값을 1차원 배열로 받습니다.
        params w_array
        이 신경의 가중치값을 1차원 배열로 받습니다.
        params b_array
        이 신경의 편향을 1차원 배열로 받습니다.
        params out
        계산 결과 저장할 것을 1차원 배열로 받습니다."""
    x_size = len(x_array)
    w_size = len(w_array)
    
    w_col = len(b_array)
    w_row = w_size // w_col
    #x_row = x_size // w_row
    
    for index in range(len(out)):
        pass_x_index = index // w_col * w_row
        pass_w_index = index % w_col
        
        out[index] = 0
        for product_index in range(w_row):
            out[index] += x_array[(pass_x_index + product_index) % x_size]* w_array[(pass_w_index + product_index * w_col) % w_size]
        out[index] += b_array[pass_w_index]
    return None

def backward(dx_array, w_array, w_shape, out):
    dx_size = len(dx_array)
    w_size = len(w_array)

    for index in range(len(out)):
        pass_dx_index = index // w_shape[0] * w_shape[1]
        pass_w_index = index * w_shape[1]

        out[index] = 0
        for product_index in range(w_shape[1]):
            out[index] += dx_array[(pass_dx_index + product_index) % dx_size] * w_array[(pass_w_index + product_index) % w_size]
    return None

def backward_variables(x_array, dx_array, dw_array, db_array):
    #x_size = len(x_array)
    dx_size = len(dx_array)
    dw_size = len(dw_array)
    db_size = len(db_array)

    x_col = dw_size // db_size
    #dx_col = db_size
    
    product_max = dx_size // db_size

    # matmul과 axis 계산을 합치기 위해 3중 루프 구조로 만듬
    for col in range(db_size):
        db_array[col] = 0
        for row in range(x_col):
            dw_array[row * db_size + col] = 0
            
        for product_index in range(product_max):
            db_array[col] += dx_array[product_index * db_size + col]
            for row in range(x_col):
                dw_array[row * db_size + col] += x_array[row + product_index * x_col] * dx_array[col + product_index * db_size]
        
