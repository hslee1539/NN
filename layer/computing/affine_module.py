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
            out[index] += x_array[(pass_x_index + product_index) % x_size] * w_array[(pass_w_index + product_index * w_col) % w_size]
        out[index] += b_array[pass_w_index]
    return None

def partialForward(x_array, w_array, b_array, out_array, index, max_index):
    """반드시 out_array는 max_index에 나누어 떨어질수록 잘 분배됨"""
    zero = type(out_array[0])(0)
    x_size = len(x_array)
    w_size = len(w_array)
    
    w_col = len(b_array)
    w_row = w_size // w_col
    #x_row = x_size // w_row
    
    start = index * len(out_array) // max_index
    end = (index + 1) * len(out_array) // max_index

    for index in range(start, end):
        pass_x_index = index // w_col * w_row
        pass_w_index = index % w_col
        
        tmp = zero
        for product_index in range(w_row):
            tmp += x_array[(pass_x_index + product_index) % x_size] * w_array[(pass_w_index + product_index * w_col) % w_size]
        out_array[index] = tmp + b_array[pass_w_index]
        #out_array[index] += b_array[pass_w_index]
    return None

def backward(dout_array, w_array, w_shape, dx_array):
    dout_size = len(dout_array)
    w_size = len(w_array)

    for index in range(len(dx_array)):
        pass_dx_index = index // w_shape[0] * w_shape[1]
        pass_w_index = index * w_shape[1]

        dx_array[index] = 0
        for product_index in range(w_shape[1]):
            dx_array[index] += dout_array[(pass_dx_index + product_index) % dout_size] * w_array[(pass_w_index + product_index) % w_size]
    return None

def partialBackward(dout_array, w_array, w_shape, dx_array, index, max_index):
    """dx_array는 max_index에 나누어 떨어질수록 잘 분배됨"""
    zero = type(dx_array[0])(0)
    dout_size = len(dout_array)
    w_size = len(w_array)

    start = index * len(dx_array) // max_index
    end = (index + 1) * len(dx_array) // max_index

    for index in range(start, end):
        pass_dx_index = index // w_shape[0] * w_shape[1]
        pass_w_index = index * w_shape[1]

        dx_array[index] = zero
        for product_index in range(w_shape[1]):
            dx_array[index] += dout_array[(pass_dx_index + product_index) % dout_size] * w_array[(pass_w_index + product_index) % w_size]
    return None

def backward_variables(x_array, dout_array, dw_array, db_array):
    #x_size = len(x_array)
    dout_size = len(dout_array)
    dw_size = len(dw_array)
    db_size = len(db_array)

    x_col = dw_size // db_size
    #dx_col = db_size
    
    product_max = dout_size // db_size

    # matmul과 axis 계산을 합치기 위해 3중 루프 구조로 만듬
    for col in range(db_size):
        db_array[col] = 0
        for row in range(x_col):
            dw_array[row * db_size + col] = 0
            
        for product_index in range(product_max):
            db_array[col] += dout_array[product_index * db_size + col]
            for row in range(x_col):
                dw_array[row * db_size + col] += x_array[row + product_index * x_col] * dout_array[col + product_index * db_size]

def partialBackward_dw(x_array, x_shape, dout_array, dw_array, dw_shape, index, max_index):
    """len(dw_array)는 max_index에 나누어 떨어 질 수록 잘 분배됨.
    총 len(dw_array) * x_shape[0] 만큼 반복문을 돌아야 함.
    x, dout, dw는 모두 2차원 tensor여야 함."""
    zero = type(dw_array[0])(0)
    dw_len = len(dw_array)
    start = index * dw_len // max_index
    end = (index + 1) * dw_len // max_index

    for dw_index in range(start, end):
        product_index = dw_index // dw_shape[-1]
        dw_m1 = dw_index % dw_shape[-1]
        dw_array[dw_index] = zero
        for x0 in range(x_shape[0]):
            x_index = x0 * x_shape[-1] + product_index
            dout_index = x0 * dw_shape[-1] + dw_m1

            dw_array[dw_index] = x_array[x_index] * dout_array[dout_index]
    
def partialBackward_db(dout_array, dbias_array, index, max_index):
    """len(dbias_array)는 max_index와 나누어 떨어질 수록잘 분배 됨.
    총 len(dout_array)만큼 반복문이 돌아야 함.
    dout은 반드시 2차원 tensor이고, dbias는 1차원 tensor여야 함."""
    zero = type(dbias_array[0])(0)

    dbias_len = len(dbias_array)
    start = index * dbias_len // max_index
    end = (index + 1) * dbias_len // max_index

    dout0 = len(dout_array) // dbias_len

    for dbias_index in range(start, end):
        dbias_array[dbias_index] = zero
        for d0 in range(dout0):
            dbias_array[dbias_index] += dout_array[d0 * dbias_len + dbias_index]