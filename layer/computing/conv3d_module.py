def create_shape(x_shape, filter_shape, stride, pad):
    shape = [x_shape[0], filter_shape[0], (x_shape[2] + 2 * pad - filter_shape[2]) // stride + 1, (x_shape[3] + 2 * pad - filter_shape[3]) // stride + 1]
    return shape

def forward(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    """pad : 빈 공간, padding : 빈 공간에 대체되는 수"""
    multipler1 = 0
    multipler2 = 0
    multiplerX = 0

    #반복문을 4중첩 쓰는 것 보다 풀어서 해결하는게 cpu의 제어 해저드가 들 걸림.
    for o in range(len(out_array)):
        #이곳을 반복문으로 해결하면 conv3d가 아닌 conv_N_dim으로 응용 가능
        multipler1 = out_shape[3]
        o3 = o % multipler1
        o2 = o // multipler1 % out_shape[2]
        multipler1 *= out_shape[2]
        o1 = o // multipler1 % out_shape[1]
        multipler1 *= out_shape[1]
        o0 = o // multipler1 % out_shape[0]

        out_array[o] = 0

        #반복문을 3중첩 쓰는 것 보다 풀어서 해결하는게 cpu의 제어 해저드가 들 걸림.
        for f in range(len(filter_array) // filter_shape[0]):
            #똑같이 반복문으로 해결하면 conv3d가 아닌 conv_N_dim으로 응용 가능
            multipler2 = filter_shape[3]
            f3 = f % multipler2
            f2 = f // multipler2 % filter_shape[2]
            multipler2 *= filter_shape[2]
            f1 = f // multipler2 % filter_shape[1]# 필요 없는 변수
            multipler2 *= filter_shape[1]
            #마지막은 out의 진행 상황에 따라 변함
            f0 = o1 % filter_shape[0]

            #최종 filter index
            filter_index = f + f0 * multipler2

            x3 = f3 + o3 * stride - pad
            x2 = f2 + o2 * stride - pad
            x1 = f1
            x0 = o0

            multiplerX = x_shape[3]
            x_index = x3
            x_index += x2 * multiplerX
            multiplerX *= x_shape[2]
            x_index += x1 * multiplerX
            multiplerX *= x_shape[1]
            #최종 x index
            x_index += x0 * multiplerX

            # 여기에 반대쪽의 경우를 코딩 해라.
            # x의 -1차원 부분의 index가 음수라면, pad 부분임.
            # 이 index를 아주 큰 수로 right 연산하면 양수면 0 음수면 -1이 나옴.
            # 이를 +1을 하여 음수면 0 양수면 1 즉, pad 부분이면 0이 되어 이 값을 곱하면 pad 처리가 됨.
            isPass = (x3 >> 10000) + 1
            # 이 경우, 같은 원리로, -2차원의 index가 pad 영역인지 판단함.
            isPass *= (x2 >> 10000) + 1
            # 이 경우, x_shape[-1]의 값을 넘으면 반대쪽 pad 부분임.
            isPass *= 1  + (-(x3 // x_shape[3]) >> 10000)
            # 이 경우 같은 원리로, -2차원의 경우임.
            isPass *= 1  + (-(x2 // x_shape[2]) >> 10000)

            # 인덱스 오버플로 방지
            x_index *= isPass

            # x와 filter와 곱함.
            out_array[o] += x_array[x_index] * filter_array[filter_index] * isPass
            # pad 부분은 padding 값으로 채움.
            out_array[o] += padding * (1 - isPass)
        # 바이어스와 더함.
        out_array[o] += bias_array[o1]
    return None

def backward(x_array, dout_array, dout_shape, filter_array, filter_shape, stride, pad, padding, dfilter_array, dbias_array, dx_array, dx_shape):
    multiplerO = 0
    multiplerF = 0
    multiplerDout = 0

    
    for dout_index in range(len(dx_array)):
        dx_array[dout_index] = 0
    
    for dfilter_index in range(len(dfilter_array)):
        dfilter_array[dfilter_index] = 0
    
    for dbias_index in range(len(dbias_array)):
        dbias_array[dbias_index] = 0
    
    
    for dout_index in range(len(dout_array)):
        dout3 = dout_index % dout_shape[3]
        multiplerDout = dout_shape[3]
        dout2 = dout_index // multiplerDout % dout_shape[2]
        multiplerDout *= dout_shape[2]
        dout1 = dout_index // multiplerDout % dout_shape[1]
        multiplerDout *= dout_shape[1]
        dout0 = dout_index // multiplerDout

        for f in range(len(filter_array) // filter_shape[0]):
            filter3 = f % filter_shape[3]
            multiplerF = filter_shape[3]
            filter2 = f // multiplerF % filter_shape[2]
            multiplerF *= filter_shape[2]
            filter1 = f // multiplerF % filter_shape[1]
            multiplerF *= filter_shape[1]
            filter0 = dout1

            #최종 filter index
            filter_index = f + filter0 * multiplerF

            x3 = filter3 + dout3 * stride - pad
            x2 = filter2 + dout2 * stride - pad
            x1 = filter1
            x0 = dout0

            multiplerO = dx_shape[3]
            x_index = x3
            x_index += x2 * multiplerO
            multiplerO *= dx_shape[2]
            x_index += x1 * multiplerO
            multiplerO *= dx_shape[1]
            x_index += x0 * multiplerO

            isPass = (x3 >> 10000) + 1
            isPass *= (x2 >> 10000) + 1
            isPass *= 1  + (-(x3 // dx_shape[3]) >> 10000)
            isPass *= 1  + (-(x2 // dx_shape[2]) >> 10000)

            # 인덱스 오버플로 방지
            x_index *= isPass

            dx_array[x_index] += isPass * dout_array[dout_index] * filter_array[filter_index]
            dfilter_array[filter_index] += isPass * dout_array[dout_index] * x_array[x_index]
            # ??? 아래를 막아야 됨... X가 pad영역일때, filter * padding가 되기 때문에 이 경우의 미분은 아래가 맞지만...
            # 막아야 편미분한 것과 값이 같아짐....
            # 왜일까...?
            #dfilter_array[filter_index] += (1 - isPass) * padding * dout_array[dout_index]
        dbias_array[dout1] += dout_array[dout_index]
    return None