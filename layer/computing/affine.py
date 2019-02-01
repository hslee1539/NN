

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
    w_col = len(b_array)
    w_row = len(w_array) // w_col
    x_row = len(x_array) // w_row

    w_size = w_col * w_row
    
    for index in range(len(out)):
        pass_x_index = index // w_col * w_col
        pass_w_index = index // w_col
        
        out[index] = 0
        for product_index in range(w_col):
            out[index] += x_array[pass_x_index + product_index]* w_array[(pass_w_index + product_index * w_col) % w_size]
        out[index] += b_array[pass_w_index]
    return None
