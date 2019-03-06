import math

def forward(x_array, x_shape, dispersion_array, out_array):
    # 데이터 수
    # 이 0차원을 제외한 모든 차원들은 특별한 의미가 있어야 함. ( column, 이미지, 영상, 다채널 이미지 등등)
    N = x_shape[0]
    # 데이터 하나의 크기
    # x_array에는 의미 있는 텐서들이 N개가 있고, N개를 나누면 의미 있는 텐서들의 크기가 됨.
    D = len(x_array) // N
    mean = 0
    deviation = 0

    for data_index in range(D):
        #평균 계산
        mean = 0
        for i in range(N):
            mean += x_array[i * D + data_index]
        mean /= N

        #분산 계산
        dispersion_array[data_index] = 0
        for i in range(N):
            deviation = x_array[i * D + data_index] - mean
            dispersion_array[data_index] += deviation * deviation
        dispersion_array[data_index] /= N

        #x를 평균0 분산1로 변환 계산(out_array에 저장)
        dispersion_array[data_index] = math.sqrt(dispersion_array[data_index] + 10E-7)
        for i in range(N):
            out_array[i * D + data_index] = x_array[i * D + data_index] - mean
            out_array[i * D + data_index] /= dispersion_array[data_index]
            
    return None

def partialForward(x_array, x_shape, dispersion_array, out_array, index, max_index):
    """0차원을 제외한 부분의 수가 max_index와 %연산이 결과가 0에 가까울 수록 계산 분산이 잘 됩니다."""
    # 데이터 수
    # 이 0차원을 제외한 모든 차원들은 특별한 의미가 있어야 함. ( column, 이미지, 영상, 다채널 이미지 등등)
    zero = type(out_array[0])(0)
    N = x_shape[0]
    # 데이터 하나의 크기
    # x_array에는 의미 있는 텐서들이 N개가 있고, N개를 나누면 의미 있는 텐서들의 크기가 됨.
    D = len(x_array) // N

    for data_index in range(index * D // max_index, (index + 1) * D // max_index):
        #평균 계산
        mean = zero
        for i in range(N):
            mean += x_array[i * D + data_index]
        mean /= N

        #분산 계산
        dispersion = zero
        for i in range(N):
            dispersion += (x_array[i * D + data_index] - mean) * (x_array[i * D + data_index] - mean)
        #dispersion = dispersion / N

        #x를 평균0 분산1로 변환 계산(out_array에 저장)
        dispersion = math.sqrt(dispersion / N + 10E-7)
        dispersion_array[data_index] = dispersion
        for i in range(N):
            out_array[i * D + data_index] = (x_array[i * D + data_index] - mean) / dispersion
    return None

def backward(dout_array, dispersion_array, out_array):
    # dispersion_array는 dx의 shape에서 0차원을 제외한 shape임
    # 즉, D는 dim(dx) - 1d의 의미를 가지는 텐서임(1tensor의 column이거나, 2tensor의 단색 이미지이거나, 3tensor의 다 채널의 이미지거나, 기타 등등)
    D = len(dispersion_array)
    # 따라서 dx.length에 D를 나누면, 의미 있는 텐서들의 데이터 수가 됨.
    N = len(dout_array) // D

    # dx에서 분산 노드로 가는 미분 값
    tmp1 = 0
    # 평균 노드에서 out으로 가는 미분 값
    tmp2 = 0

    for data_index in range(D):
        #tmp1 계산
        tmp1 = 0
        for i in range(N):
            tmp1 += out_array[i * D + data_index] * dout_array[i * D + data_index]
        
        #tmp2 계산
        tmp2 = 0
        for i in range(N):
            tmp2 += tmp1 * out_array[i * D + data_index] / N - dout_array[i * D + data_index]
        tmp2 /= N
        
        #최종 역전파 계산 (forward값 손실)
        for i in range(N):
            out_array[i * D + data_index] = tmp2 - tmp1 * out_array[i * D + data_index] / N + dout_array[i * D + data_index]
            out_array[i * D + data_index] /= dispersion_array[data_index]
        
    return None

def partialBackward(dout_array, dispersion_array, dx_array, index, max_index):
    # dispersion_array는 dx의 shape에서 0차원을 제외한 shape임
    # 즉, D는 dim(dx) - 1d의 의미를 가지는 텐서임(1tensor의 column이거나, 2tensor의 단색 이미지이거나, 3tensor의 다 채널의 이미지거나, 기타 등등)
    D = len(dispersion_array)
    # 따라서 dx.length에 D를 나누면, 의미 있는 텐서들의 데이터 수가 됨.
    N = len(dout_array) // D

    zero = type(dx_array[0])(0)

    for data_index in range(index  * D // max_index, (index + 1) * D // max_index):
        # dx에서 분산 노드로 가는 미분 값
        tmp1 = zero
        for i in range(N):
            tmp1 += dx_array[i * D + data_index] * dout_array[i * D + data_index]
        
        # 평균 노드에서 out으로 가는 미분 값
        tmp2 = zero
        for i in range(N):
            tmp2 += tmp1 * dx_array[i * D + data_index] / N - dout_array[i * D + data_index]
        tmp2 /= N
        
        #최종 역전파 계산 (forward값 손실)
        for i in range(N):
            dx_array[i * D + data_index] = (tmp2 - tmp1 * dx_array[i * D + data_index] / N + dout_array[i * D + data_index]) / dispersion_array[data_index]
        
    return None
    

# 일반적은 정규화층은 0차원 기준으로 정규화를 해야 하는데
# 아래의 코드는 -1차원을 기준으로 함
"""
def forward(x_array, x_shape, dispersion_array, out_array):
    batch_size = x_shape[0]
    norn_size = len(x_array) // batch_size
    mean = 0
    pass_index = 0
    deviation = 0

    for r in range(batch_size):
        pass_index = r * norn_size
        #평균 계산
        mean = 0
        for c in range(norn_size):
            mean += x_array[pass_index + c]
        mean /= norn_size

        #분산 계산
        dispersion_array[r] = 0
        for c in range(norn_size):
            deviation = x_array[pass_index + c] - mean
            dispersion_array[r] += deviation * deviation
        dispersion_array[r] /= norn_size

        #x를 평균0 분산1로 변환 계산(out_array에 저장)
        dispersion_array[r] = math.sqrt(dispersion_array[r] + 10E-7)
        for c in range(norn_size):
            out_array[pass_index + c] = x_array[pass_index + c] - mean
            out_array[pass_index + c] /= dispersion_array[r]
            
    return None

def backward(dout_array, dispersion_array, out_array):
    batch_size = dx_shape[0]
    norn_size = len(dout_array) // batch_size
    pass_index = 0

    # dx에서 분산 노드로 가는 미분 값
    tmp1 = 0
    # 평균 노드에서 out으로 가는 미분 값
    tmp2 = 0

    for r in range(batch_size):
        pass_index = r * norn_size
        #tmp1 계산
        tmp1 = 0
        for c in range(norn_size):
            tmp1 += out_array[pass_index + c] * dout_array[pass_index + c]

        #tmp2 계산
        tmp2 = 0
        for c in range(norn_size):
            tmp2 += tmp1 * out_array[pass_index + c] / norn_size - dout_array[pass_index + c]
        tmp2 /= norn_size

        #최종 역전파 계산 (forward값 손실)
        for c in range(norn_size):
            out_array[pass_index + c] = tmp2 - tmp1 * out_array[pass_index + c] + dout_array[pass_index]
            out_array[pass_index + c] /= dispersion_array[r]
        
    return None
"""
